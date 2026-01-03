"""
Learning 02: Understanding Embeddings
======================================
This file teaches you what embeddings are and why they matter for RuleMirror.

Run this file: python learn_embeddings.py

Prerequisites:
    pip install sentence-transformers numpy

What you'll learn:
    1. What are embeddings?
    2. How to create them
    3. Why they're essential for RAG
    4. Semantic similarity (the magic behind search)
"""

import numpy as np

# We'll use sentence-transformers for embeddings
# This is what powers our semantic search in RuleMirror
try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False
    print("⚠️  Run: pip install sentence-transformers")
    print("   Then run this file again!\n")


def explain_embeddings():
    """
    LESSON 1: What are Embeddings?
    ==============================
    
    Embeddings convert text into numbers (vectors) that capture MEANING.
    
    Example:
        "The contract expires in 30 days" → [0.23, -0.45, 0.12, ...]
        
    Why numbers? Because computers can:
        - Compare numbers (find similarity)
        - Store numbers in databases
        - Search through millions of texts quickly
    """
    print("=" * 60)
    print("LESSON 1: What are Embeddings?")
    print("=" * 60)
    print("""
    Embeddings convert text → numbers (vectors) that capture MEANING.
    
    Think of it like GPS coordinates for text:
        - "cat" and "kitten" → close coordinates (similar meaning)
        - "cat" and "contract" → far coordinates (different meaning)
    
    For RuleMirror:
        - Policy: "Payment must be within 30 days" → [0.23, -0.45, ...]
        - Contract: "Invoice due in 30 days" → [0.21, -0.44, ...]
        
    These vectors are CLOSE because they mean similar things!
    The AI can find matching policies for contract clauses.
    """)


def demo_embeddings():
    """
    LESSON 2: Creating Embeddings (Hands-on)
    ========================================
    """
    if not HAS_EMBEDDINGS:
        print("\n❌ Skipping demo - install sentence-transformers first")
        return None
    
    print("\n" + "=" * 60)
    print("LESSON 2: Creating Embeddings (Hands-on)")
    print("=" * 60)
    
    # Load a small, fast embedding model
    print("\n📦 Loading embedding model (first time takes a minute)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')  # 80MB, very fast
    
    # Example texts - imagine these are from contracts and policies
    texts = [
        "Payment must be made within 30 days",           # Policy
        "Invoice is due within 30 days of receipt",      # Contract (similar!)
        "The vendor shall maintain insurance coverage",  # Different topic
        "All payments should be completed in one month", # Similar to payment
    ]
    
    print("\n📝 Sample texts:")
    for i, text in enumerate(texts):
        print(f"   [{i}] {text}")
    
    # Create embeddings
    print("\n🔄 Creating embeddings...")
    embeddings = model.encode(texts)
    
    print(f"\n📊 Each text becomes a vector of {embeddings.shape[1]} numbers!")
    print(f"   Shape: {embeddings.shape} (4 texts × 384 dimensions)")
    
    # Show a snippet of the first embedding
    print(f"\n   First embedding (snippet): {embeddings[0][:5]}...")
    
    return model, texts, embeddings


def demo_similarity(model, texts, embeddings):
    """
    LESSON 3: Semantic Similarity (The Magic!)
    ==========================================
    This is HOW RuleMirror finds matching policies for contract clauses.
    """
    if model is None:
        return
    
    print("\n" + "=" * 60)
    print("LESSON 3: Semantic Similarity (The Magic!)")
    print("=" * 60)
    
    # Calculate similarity between all pairs
    # Using cosine similarity (like angle between vectors)
    from numpy import dot
    from numpy.linalg import norm
    
    def cosine_similarity(a, b):
        return dot(a, b) / (norm(a) * norm(b))
    
    print("\n🔍 Comparing text [0] with all others:")
    print(f"   [0] = '{texts[0]}'")
    print()
    
    for i in range(1, len(texts)):
        similarity = cosine_similarity(embeddings[0], embeddings[i])
        bar = "█" * int(similarity * 20)  # Visual bar
        print(f"   vs [{i}]: {similarity:.3f} {bar}")
        print(f"        '{texts[i]}'")
        print()
    
    print("""
    💡 KEY INSIGHT:
    ───────────────
    Text [0] and [1] are MOST similar (both about 30-day payment)
    even though they use different words!
    
    This is semantic search - finding meaning, not just keywords.
    
    For RuleMirror:
    - We embed all policy documents
    - When a contract clause comes in, we embed it too
    - We find the most similar policies (highest similarity score)
    - Then the AI compares them for compliance!
    """)


def demo_rulemirror_usecase(model):
    """
    LESSON 4: How RuleMirror Uses This
    ==================================
    """
    if model is None:
        return
        
    print("\n" + "=" * 60)
    print("LESSON 4: How RuleMirror Uses This")
    print("=" * 60)
    
    # Simulated policy repository
    policies = [
        "All contracts must include a termination clause with 30 days notice",
        "Intellectual property rights remain with the company",
        "Liability shall not exceed the contract value",
        "Payment terms must not exceed net-60 days",
        "Data protection clauses are mandatory for all vendors",
    ]
    
    # A clause from a draft contract
    contract_clause = "Either party may terminate with 14 days written notice"
    
    print("\n📚 Policy Repository:")
    for i, p in enumerate(policies):
        print(f"   [{i}] {p}")
    
    print(f"\n📄 Contract Clause to Check:")
    print(f"   '{contract_clause}'")
    
    # Embed everything
    policy_embeddings = model.encode(policies)
    clause_embedding = model.encode([contract_clause])[0]
    
    # Find most similar policy
    from numpy import dot
    from numpy.linalg import norm
    
    similarities = []
    for i, pe in enumerate(policy_embeddings):
        sim = dot(clause_embedding, pe) / (norm(clause_embedding) * norm(pe))
        similarities.append((sim, i, policies[i]))
    
    # Sort by similarity
    similarities.sort(reverse=True)
    
    print("\n🎯 Matching Results (sorted by relevance):")
    for sim, idx, policy in similarities[:3]:  # Top 3
        print(f"   {sim:.3f} - {policy}")
    
    print("""
    
    ⚠️  COMPLIANCE ISSUE DETECTED!
    ──────────────────────────────
    The contract says 14 days notice, but policy requires 30 days!
    
    This is exactly what RuleMirror does:
    1. Embed policies (done once, stored in Qdrant)
    2. Embed contract clause (when reviewing)
    3. Find most relevant policy (similarity search)
    4. AI compares them (Mistral LLM)
    5. Report violations (like this 14 vs 30 day issue!)
    """)


def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     RULEMIRROR LEARNING: Understanding Embeddings        ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Lesson 1: Explanation
    explain_embeddings()
    
    # Lesson 2: Create embeddings
    result = demo_embeddings()
    
    if result:
        model, texts, embeddings = result
        
        # Lesson 3: Similarity
        demo_similarity(model, texts, embeddings)
        
        # Lesson 4: RuleMirror use case
        demo_rulemirror_usecase(model)
    
    print("\n" + "=" * 60)
    print("✅ SUMMARY: What You Learned")
    print("=" * 60)
    print("""
    1. Embeddings = Text converted to numbers (vectors)
    2. Similar meanings = Similar vectors = High similarity score
    3. This enables SEMANTIC search (by meaning, not keywords)
    4. RuleMirror uses this to find relevant policies for each clause
    
    Next: We'll set up Qdrant to STORE these embeddings!
    """)


if __name__ == "__main__":
    main()
