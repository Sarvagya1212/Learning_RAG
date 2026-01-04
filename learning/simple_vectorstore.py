"""
Learning 04: Simple Vector Store
=================================
This file teaches you how vector databases work and how to use Qdrant.

Run this file: python simple_vectorstore.py

Prerequisites:
    pip install qdrant-client sentence-transformers

What you'll learn:
    1. What is a vector database?
    2. Why we need one for RAG
    3. How to store embeddings
    4. How to search by similarity
    5. How RuleMirror uses this
"""

import os
from pathlib import Path

# Check dependencies
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
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


def explain_vectorstore():
    """
    LESSON 1: What is a Vector Database?
    =====================================
    """
    print("=" * 60)
    print("LESSON 1: What is a Vector Database?")
    print("=" * 60)
    print("""
    A vector database stores EMBEDDINGS (number arrays) and lets you
    search by SIMILARITY instead of exact matches.
    
    Traditional Database:
    ┌─────────────────────────────────────────────────────┐
    │  SELECT * FROM policies WHERE text LIKE '%30 days%' │
    │  → Only finds exact keyword matches                 │
    └─────────────────────────────────────────────────────┘
    
    Vector Database:
    ┌─────────────────────────────────────────────────────┐
    │  Find policies SIMILAR to "payment deadline"        │
    │  → Finds "30 days", "one month", "net-30" etc!      │
    └─────────────────────────────────────────────────────┘
    
    For RuleMirror:
    • Store all policy chunks as embeddings
    • When reviewing a contract clause, find SIMILAR policies
    • Compare them for compliance violations
    """)


def explain_qdrant():
    """
    LESSON 2: Why Qdrant?
    =====================
    """
    print("\n" + "=" * 60)
    print("LESSON 2: Why Qdrant?")
    print("=" * 60)
    print("""
    We chose Qdrant because:
    
    ┌────────────────────────────────────────────────────────┐
    │ ✅ In-memory mode for development (no server needed!)  │
    │ ✅ Hybrid search (dense + sparse vectors)              │
    │ ✅ Production-ready with persistence                   │
    │ ✅ Metadata filtering (by policy section, date, etc.)  │
    │ ✅ Open source with Python client                      │
    └────────────────────────────────────────────────────────┘
    
    Modes:
    • In-memory: QdrantClient(":memory:")     ← For learning
    • Local file: QdrantClient(path="./db")   ← For persistence
    • Server: QdrantClient("localhost:6333")  ← For production
    """)


def demo_basic_operations():
    """
    LESSON 3: Basic Operations (Hands-on)
    =====================================
    """
    if not HAS_QDRANT or not HAS_EMBEDDINGS:
        print("\n❌ Skipping demo - install dependencies first")
        return None, None
    
    print("\n" + "=" * 60)
    print("LESSON 3: Basic Operations (Hands-on)")
    print("=" * 60)
    
    # Initialize Qdrant in-memory (no server needed!)
    print("\n📦 Creating in-memory Qdrant database...")
    client = QdrantClient(":memory:")
    
    # Load embedding model
    print("📦 Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Sample policy chunks (from our sample policy)
    policy_chunks = [
        {
            "id": 1,
            "text": "All vendor payments must be processed within thirty (30) days of invoice receipt.",
            "section": "Payment Terms",
            "policy_id": "POL-2024-001"
        },
        {
            "id": 2,
            "text": "Minimum notice period for termination is thirty (30) days.",
            "section": "Contract Termination",
            "policy_id": "POL-2024-001"
        },
        {
            "id": 3,
            "text": "All intellectual property created during engagement remains company property.",
            "section": "Intellectual Property",
            "policy_id": "POL-2024-001"
        },
        {
            "id": 4,
            "text": "All vendors handling personal data must comply with GDPR/CCPA.",
            "section": "Data Protection",
            "policy_id": "POL-2024-001"
        },
        {
            "id": 5,
            "text": "Vendor liability shall not exceed the total contract value.",
            "section": "Liability",
            "policy_id": "POL-2024-001"
        },
    ]
    
    print(f"\n📝 Policy chunks to store: {len(policy_chunks)}")
    for chunk in policy_chunks:
        print(f"   [{chunk['id']}] {chunk['section']}: {chunk['text'][:50]}...")
    
    # Create embeddings
    print("\n🔄 Creating embeddings...")
    texts = [chunk["text"] for chunk in policy_chunks]
    embeddings = model.encode(texts)
    
    vector_size = embeddings.shape[1]
    print(f"   Vector size: {vector_size} dimensions")
    
    # Create collection (like a table)
    print("\n📊 Creating 'policies' collection...")
    client.create_collection(
        collection_name="policies",
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE  # Cosine similarity
        )
    )
    
    # Insert points (vectors + metadata)
    print("💾 Storing vectors with metadata...")
    points = [
        PointStruct(
            id=chunk["id"],
            vector=embeddings[i].tolist(),
            payload={
                "text": chunk["text"],
                "section": chunk["section"],
                "policy_id": chunk["policy_id"]
            }
        )
        for i, chunk in enumerate(policy_chunks)
    ]
    
    client.upsert(
        collection_name="policies",
        points=points
    )
    
    print(f"   ✅ Stored {len(points)} vectors!")
    
    return client, model


def demo_similarity_search(client, model):
    """
    LESSON 4: Similarity Search (The Core of RAG!)
    ===============================================
    """
    if client is None:
        return
    
    print("\n" + "=" * 60)
    print("LESSON 4: Similarity Search (The Core of RAG!)")
    print("=" * 60)
    
    # Contract clauses to check
    contract_clauses = [
        "Payment will be made within 14 days of invoice",
        "Either party may terminate with 7 days notice",
        "All code developed belongs to the contractor",
    ]
    
    print("\n📄 Contract clauses to review:")
    for i, clause in enumerate(contract_clauses):
        print(f"   [{i+1}] {clause}")
    
    print("\n" + "-" * 60)
    
    for clause in contract_clauses:
        print(f"\n🔍 Checking: \"{clause}\"")
        
        # Embed the contract clause
        clause_embedding = model.encode([clause])[0]
        
        # Search for similar policies (using query_points for newer qdrant-client)
        results = client.query_points(
            collection_name="policies",
            query=clause_embedding.tolist(),
            limit=2,  # Top 2 matches
            with_payload=True
        )
        
        print("   📚 Most relevant policies:")
        for result in results.points:
            score = result.score
            policy_text = result.payload["text"]
            section = result.payload["section"]
            
            # Visual score bar
            bar = "█" * int(score * 20)
            
            print(f"      {score:.3f} {bar}")
            print(f"      [{section}] {policy_text[:60]}...")
            print()


def demo_metadata_filtering(client, model):
    """
    LESSON 5: Metadata Filtering (Advanced)
    =======================================
    """
    if client is None:
        return
    
    print("\n" + "=" * 60)
    print("LESSON 5: Metadata Filtering (Advanced)")
    print("=" * 60)
    
    print("""
    Sometimes you want to search ONLY within specific sections.
    
    Example: "Find payment-related policies" but ONLY in
             the "Payment Terms" section.
    
    Qdrant supports filtering on metadata (payload):
    """)
    
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    
    query = "invoice payment deadline"
    print(f"\n🔍 Query: \"{query}\"")
    
    # Without filter
    print("\n   Without filter (all sections):")
    query_embedding = model.encode([query])[0]
    
    results = client.query_points(
        collection_name="policies",
        query=query_embedding.tolist(),
        limit=3,
        with_payload=True
    )
    
    for r in results.points:
        print(f"      {r.score:.3f} [{r.payload['section']}] {r.payload['text'][:40]}...")
    
    # With filter
    print("\n   With filter (Payment Terms only):")
    
    results_filtered = client.query_points(
        collection_name="policies",
        query=query_embedding.tolist(),
        limit=3,
        with_payload=True,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="section",
                    match=MatchValue(value="Payment Terms")
                )
            ]
        )
    )
    
    for r in results_filtered.points:
        print(f"      {r.score:.3f} [{r.payload['section']}] {r.payload['text'][:40]}...")
    
    print("""
    
    💡 Why filtering matters for RuleMirror:
    ────────────────────────────────────────
    • Filter by policy type (HR, Legal, Finance)
    • Filter by effective date (only current policies)
    • Filter by department (IT, Legal, Procurement)
    """)


def demo_rulemirror_workflow():
    """
    LESSON 6: Complete RuleMirror Workflow
    ======================================
    """
    print("\n" + "=" * 60)
    print("LESSON 6: Complete RuleMirror Workflow")
    print("=" * 60)
    
    print("""
    Here's how all the pieces fit together:
    
    ┌─────────────────────────────────────────────────────────────┐
    │ STEP 1: INGESTION (Done Once)                               │
    │                                                             │
    │   PDF Policies → Extract Text → Chunk → Embed → Store      │
    │       ↓              ↓           ↓        ↓        ↓       │
    │   [PyPDF]      [Unstructured] [Smart]  [Model]  [Qdrant]   │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ STEP 2: REVIEW (Per Contract)                               │
    │                                                             │
    │   Contract → Extract Clauses → Embed → Search Qdrant       │
    │                                           ↓                 │
    │                                    Top-K Similar Policies   │
    │                                           ↓                 │
    │                                    Compare via LLM          │
    │                                           ↓                 │
    │                                    Compliance Report        │
    └─────────────────────────────────────────────────────────────┘
    
    Next: We'll build the LLM comparison step!
    """)


def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     RULEMIRROR LEARNING: Simple Vector Store             ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Lesson 1: What is a vector DB
    explain_vectorstore()
    
    # Lesson 2: Why Qdrant
    explain_qdrant()
    
    # Lesson 3: Basic operations
    client, model = demo_basic_operations()
    
    # Lesson 4: Similarity search
    if client:
        demo_similarity_search(client, model)
    
    # Lesson 5: Metadata filtering
    if client:
        demo_metadata_filtering(client, model)
    
    # Lesson 6: Full workflow
    demo_rulemirror_workflow()
    
    print("\n" + "=" * 60)
    print("✅ SUMMARY: What You Learned")
    print("=" * 60)
    print("""
    1. Vector databases store embeddings for similarity search
    2. Qdrant is our choice (in-memory for dev, server for prod)
    3. Basic flow: Create collection → Store vectors → Search
    4. Metadata filtering narrows search to specific sections
    5. This is the "Retrieval" part of RAG!
    
    Next: Learning 05 - LLM Integration (the "Generation" part!)
    """)


if __name__ == "__main__":
    main()
