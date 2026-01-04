"""
Learning 05: Build Basic RAG Chain
===================================
This file teaches you how to build a complete RAG (Retrieval Augmented Generation)
pipeline - the core of RuleMirror!

Run this file: python simple_rag.py

Prerequisites:
    pip install qdrant-client sentence-transformers mistralai python-dotenv

What you'll learn:
    1. What is RAG and why we need it
    2. The RAG pipeline step by step
    3. How to connect retrieval to LLM
    4. Building a compliance checker
    5. Error handling and edge cases
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check dependencies
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False
    print("⚠️  Run: pip install qdrant-client")

try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False
    print("⚠️  Run: pip install sentence-transformers")

try:
    from mistralai import Mistral
    HAS_MISTRAL = True
except ImportError:
    HAS_MISTRAL = False
    print("⚠️  Run: pip install mistralai")


def explain_rag():
    """
    LESSON 1: What is RAG?
    ======================
    """
    print("=" * 60)
    print("LESSON 1: What is RAG?")
    print("=" * 60)
    print("""
    RAG = Retrieval Augmented Generation
    
    The Problem:
    ┌─────────────────────────────────────────────────────────┐
    │  LLMs don't know YOUR company's policies!               │
    │  They were trained on general internet data.            │
    │  We need to GIVE them the relevant context.             │
    └─────────────────────────────────────────────────────────┘
    
    The Solution: RAG
    ┌─────────────────────────────────────────────────────────┐
    │  1. RETRIEVE: Find relevant policies (vector search)    │
    │  2. AUGMENT: Add policies to the LLM prompt             │
    │  3. GENERATE: LLM analyzes with full context            │
    └─────────────────────────────────────────────────────────┘
    
    For RuleMirror:
    ┌─────────────────────────────────────────────────────────┐
    │  Contract Clause → Find Related Policies → Ask LLM:     │
    │  "Does this clause comply with these policies?"         │
    └─────────────────────────────────────────────────────────┘
    """)


def setup_vector_store():
    """
    LESSON 2: Setting Up the Knowledge Base
    ========================================
    """
    if not HAS_QDRANT or not HAS_EMBEDDINGS:
        print("\n❌ Missing dependencies - skipping setup")
        return None, None, None
    
    print("\n" + "=" * 60)
    print("LESSON 2: Setting Up the Knowledge Base")
    print("=" * 60)
    
    # Initialize components
    print("\n📦 Initializing components...")
    client = QdrantClient(":memory:")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Our policy knowledge base
    policies = [
        {
            "id": 1,
            "text": "All vendor payments must be processed within thirty (30) days of invoice receipt. No exceptions.",
            "section": "Payment Terms",
            "rule_id": "PAY-001"
        },
        {
            "id": 2,
            "text": "Early payment discounts of up to 2% are authorized for payments within 10 days.",
            "section": "Payment Terms",
            "rule_id": "PAY-002"
        },
        {
            "id": 3,
            "text": "Minimum notice period for contract termination is thirty (30) days written notice.",
            "section": "Termination",
            "rule_id": "TERM-001"
        },
        {
            "id": 4,
            "text": "Immediate termination is permitted only in cases of material breach or fraud.",
            "section": "Termination",
            "rule_id": "TERM-002"
        },
        {
            "id": 5,
            "text": "All intellectual property created during the engagement shall remain the exclusive property of the Company.",
            "section": "IP Rights",
            "rule_id": "IP-001"
        },
        {
            "id": 6,
            "text": "Vendors must assign all IP rights to the Company within 30 days of project completion.",
            "section": "IP Rights",
            "rule_id": "IP-002"
        },
        {
            "id": 7,
            "text": "Total vendor liability shall not exceed the total value of the contract.",
            "section": "Liability",
            "rule_id": "LIA-001"
        },
        {
            "id": 8,
            "text": "All vendors must maintain professional liability insurance of at least $1 million.",
            "section": "Liability",
            "rule_id": "LIA-002"
        },
    ]
    
    print(f"   📚 Loaded {len(policies)} policy rules")
    
    # Create embeddings and store
    texts = [p["text"] for p in policies]
    embeddings = model.encode(texts)
    
    client.create_collection(
        collection_name="policies",
        vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE)
    )
    
    points = [
        PointStruct(
            id=p["id"],
            vector=embeddings[i].tolist(),
            payload={"text": p["text"], "section": p["section"], "rule_id": p["rule_id"]}
        )
        for i, p in enumerate(policies)
    ]
    
    client.upsert(collection_name="policies", points=points)
    print("   ✅ Policies indexed in vector store")
    
    return client, model, policies


def demo_retrieval(client, model):
    """
    LESSON 3: Retrieval - Finding Relevant Policies
    ================================================
    """
    if client is None:
        return []
    
    print("\n" + "=" * 60)
    print("LESSON 3: Retrieval - Finding Relevant Policies")
    print("=" * 60)
    
    # Sample contract clause to check
    clause = "Payment shall be made within 45 days of receiving the invoice."
    
    print(f"\n📄 Contract clause to check:")
    print(f"   \"{clause}\"")
    
    # Embed and search
    clause_embedding = model.encode([clause])[0]
    
    results = client.query_points(
        collection_name="policies",
        query=clause_embedding.tolist(),
        limit=3,
        with_payload=True
    )
    
    print("\n🔍 Retrieved relevant policies:")
    retrieved_policies = []
    for r in results.points:
        print(f"   [{r.payload['rule_id']}] (score: {r.score:.3f})")
        print(f"   {r.payload['text'][:70]}...")
        print()
        retrieved_policies.append(r.payload)
    
    return retrieved_policies


def build_prompt(clause: str, policies: list) -> str:
    """
    LESSON 4: Building the Augmented Prompt
    =======================================
    This is the "A" in RAG - we AUGMENT the prompt with retrieved context.
    """
    
    policy_context = "\n".join([
        f"- [{p['rule_id']}] {p['text']}" 
        for p in policies
    ])
    
    prompt = f"""You are a legal compliance expert. Your job is to review contract clauses against company policies.

## Company Policies (MUST be followed):
{policy_context}

## Contract Clause to Review:
"{clause}"

## Your Task:
1. Analyze if the clause COMPLIES with or VIOLATES any of the policies above.
2. If there's a violation, explain WHAT is wrong and cite the specific policy rule ID.
3. Suggest how to FIX the clause to make it compliant.

## Response Format:
COMPLIANCE STATUS: [COMPLIANT / VIOLATION / NEEDS REVIEW]

ANALYSIS:
[Your detailed analysis here]

RECOMMENDATION:
[Suggested fix if needed]
"""
    return prompt


def demo_generation(clause: str, policies: list):
    """
    LESSON 5: Generation - LLM Analysis
    ====================================
    """
    print("\n" + "=" * 60)
    print("LESSON 5: Generation - LLM Analysis")
    print("=" * 60)
    
    # Build the augmented prompt
    prompt = build_prompt(clause, policies)
    
    print("\n📝 Augmented prompt (showing structure):")
    print("-" * 40)
    print(prompt[:500] + "...")
    print("-" * 40)
    
    # Check for API key
    api_key = os.getenv("MISTRAL_API_KEY")
    
    if not api_key or api_key == "your_mistral_key_here":
        print("""
    ⚠️  MISTRAL_API_KEY not set in .env file!
    
    To run the full demo:
    1. Get a free API key from https://console.mistral.ai/
    2. Add it to your .env file: MISTRAL_API_KEY=your_key_here
    3. Run this script again
    
    For now, showing a SIMULATED response:
        """)
        
        # Simulated response for demo
        simulated_response = """
COMPLIANCE STATUS: VIOLATION

ANALYSIS:
The contract clause specifies a 45-day payment term, which directly 
violates policy PAY-001 that mandates all payments must be processed 
within 30 days of invoice receipt with "no exceptions."

The clause extends the payment window by 15 days beyond what is 
permitted by company policy.

RECOMMENDATION:
Revise the clause to: "Payment shall be made within thirty (30) days 
of receiving the invoice, in accordance with standard payment terms."
        """
        print("\n🤖 Simulated LLM Response:")
        print("-" * 40)
        print(simulated_response)
        print("-" * 40)
        return simulated_response
    
    if not HAS_MISTRAL:
        print("\n❌ mistralai package not installed")
        return None
    
    # Real API call
    print("\n🤖 Calling Mistral AI...")
    
    try:
        client = Mistral(api_key=api_key)
        
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        result = response.choices[0].message.content
        
        print("\n🤖 LLM Response:")
        print("-" * 40)
        print(result)
        print("-" * 40)
        
        return result
        
    except Exception as e:
        print(f"\n❌ API Error: {e}")
        return None


def demo_full_rag_pipeline():
    """
    LESSON 6: Complete RAG Pipeline
    ================================
    """
    print("\n" + "=" * 60)
    print("LESSON 6: Complete RAG Pipeline")
    print("=" * 60)
    
    print("""
    Let's put it all together!
    
    ┌─────────────────────────────────────────────────────────────┐
    │                   RAG PIPELINE                              │
    │                                                             │
    │  Contract Clause                                            │
    │       ↓                                                     │
    │  [Embed with SentenceTransformer]                           │
    │       ↓                                                     │
    │  [Search Qdrant for similar policies]     ← RETRIEVAL       │
    │       ↓                                                     │
    │  [Build prompt with clause + policies]    ← AUGMENTATION    │
    │       ↓                                                     │
    │  [Send to Mistral AI]                     ← GENERATION      │
    │       ↓                                                     │
    │  Compliance Analysis Report                                 │
    └─────────────────────────────────────────────────────────────┘
    """)


def run_compliance_check():
    """
    Interactive compliance check demo.
    """
    if not HAS_QDRANT or not HAS_EMBEDDINGS:
        print("\n❌ Missing dependencies")
        return
    
    print("\n" + "=" * 60)
    print("🔄 INTERACTIVE COMPLIANCE CHECK")
    print("=" * 60)
    
    # Setup
    client, model, _ = setup_vector_store()
    
    # Test clauses with different compliance levels
    test_clauses = [
        {
            "clause": "Payment shall be made within 45 days of receiving the invoice.",
            "expected": "VIOLATION - exceeds 30 day limit"
        },
        {
            "clause": "Either party may terminate this agreement with 7 days written notice.",
            "expected": "VIOLATION - less than 30 day notice"
        },
        {
            "clause": "All intellectual property developed under this contract belongs to the Contractor.",
            "expected": "VIOLATION - IP should belong to Company"
        }
    ]
    
    print("\n📋 Testing 3 contract clauses for compliance...")
    print("-" * 60)
    
    for i, test in enumerate(test_clauses, 1):
        clause = test["clause"]
        expected = test["expected"]
        
        print(f"\n{'='*60}")
        print(f"TEST {i}: {clause[:50]}...")
        print(f"Expected: {expected}")
        print("=" * 60)
        
        # Retrieve relevant policies
        clause_embedding = model.encode([clause])[0]
        results = client.query_points(
            collection_name="policies",
            query=clause_embedding.tolist(),
            limit=2,
            with_payload=True
        )
        
        policies = [r.payload for r in results.points]
        
        # Run generation (will be simulated if no API key)
        demo_generation(clause, policies)


def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     RULEMIRROR LEARNING: Build Basic RAG Chain           ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Lesson 1: What is RAG
    explain_rag()
    
    # Lesson 2: Setup knowledge base
    client, model, policies = setup_vector_store()
    
    # Lesson 3: Retrieval demo
    retrieved = demo_retrieval(client, model)
    
    # Lesson 4 & 5: Prompt building and generation
    if retrieved:
        clause = "Payment shall be made within 45 days of receiving the invoice."
        demo_generation(clause, retrieved)
    
    # Lesson 6: Full pipeline overview
    demo_full_rag_pipeline()
    
    print("\n" + "=" * 60)
    print("✅ SUMMARY: What You Learned")
    print("=" * 60)
    print("""
    1. RAG = Retrieval + Augmentation + Generation
    2. We RETRIEVE relevant policies using vector similarity
    3. We AUGMENT the prompt with retrieved context
    4. We GENERATE analysis using the LLM (Mistral)
    5. The LLM can now reason about YOUR company's policies!
    
    This is the CORE of RuleMirror! 🎯
    
    Next: Learning 06 - Multi-Agent Systems (for complex analysis)
    """)


if __name__ == "__main__":
    main()
