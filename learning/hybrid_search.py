"""
Learning 07: Hybrid Search (BM25 + Vector)
==========================================
This file teaches you how to combine keyword search (BM25) with 
semantic search (vectors) for better retrieval.

Run this file: python hybrid_search.py

Prerequisites:
    pip install rank-bm25 qdrant-client sentence-transformers

What you'll learn:
    1. Why hybrid search beats single-method search
    2. How BM25 keyword search works
    3. Combining BM25 + vector scores
    4. Reciprocal Rank Fusion (RRF)
    5. Building a hybrid retriever
"""

import os
from pathlib import Path

# Check dependencies
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    print("⚠️  Run: pip install rank-bm25")

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


def explain_hybrid_search():
    """
    LESSON 1: Why Hybrid Search?
    ============================
    """
    print("=" * 60)
    print("LESSON 1: Why Hybrid Search?")
    print("=" * 60)
    print("""
    Two Types of Search:
    
    ┌────────────────────────────────────────────────────────────┐
    │ VECTOR SEARCH (Semantic)                                   │
    │ ✅ Understands meaning: "30 days" ≈ "one month"            │
    │ ❌ Misses exact terms: "GDPR" might not match "GDPR"       │
    │ ❌ Weak on rare words, acronyms, IDs                       │
    └────────────────────────────────────────────────────────────┘
    
    ┌────────────────────────────────────────────────────────────┐
    │ KEYWORD SEARCH (BM25)                                      │
    │ ✅ Exact match: "GDPR" always finds "GDPR"                 │
    │ ✅ Great for specific terms, policy IDs                    │
    │ ❌ Misses synonyms: "payment" won't find "invoice"         │
    └────────────────────────────────────────────────────────────┘
    
    ┌────────────────────────────────────────────────────────────┐
    │ HYBRID SEARCH = BEST OF BOTH! 🎯                           │
    │ ✅ Semantic understanding                                  │
    │ ✅ Exact keyword matching                                  │
    │ ✅ Better recall for legal/policy documents                │
    └────────────────────────────────────────────────────────────┘
    
    For RuleMirror:
    - Vector search finds "payment terms" when clause says "invoice due"
    - BM25 ensures "Policy PAY-001" is found when searching "PAY-001"
    """)


def explain_bm25():
    """
    LESSON 2: How BM25 Works
    ========================
    """
    print("\n" + "=" * 60)
    print("LESSON 2: How BM25 Works")
    print("=" * 60)
    print("""
    BM25 (Best Match 25) is a ranking function for keyword search.
    
    Key Ideas:
    ┌────────────────────────────────────────────────────────────┐
    │ 1. Term Frequency (TF)                                     │
    │    More occurrences = more relevant (with diminishing      │
    │    returns)                                                │
    │                                                            │
    │ 2. Inverse Document Frequency (IDF)                        │
    │    Rare words are more important than common words         │
    │    "GDPR" is more informative than "the"                   │
    │                                                            │
    │ 3. Document Length                                         │
    │    Shorter docs with the term are more relevant            │
    └────────────────────────────────────────────────────────────┘
    
    Formula (simplified):
        score = IDF * (TF * (k1 + 1)) / (TF + k1 * (1 - b + b * |D|/avgDL))
    
    Where: k1=1.5, b=0.75 (typical values)
    """)


def demo_bm25():
    """
    LESSON 3: BM25 in Action
    ========================
    """
    if not HAS_BM25:
        print("\n❌ rank-bm25 not installed")
        return None, None
    
    print("\n" + "=" * 60)
    print("LESSON 3: BM25 in Action")
    print("=" * 60)
    
    # Sample policy documents
    documents = [
        "All payments must be processed within 30 days of invoice receipt.",
        "Payment terms are NET-30 with 2% discount for early payment.",
        "Termination requires 30 days written notice from either party.",
        "GDPR compliance is mandatory for all data processing activities.",
        "Policy PAY-001 governs all vendor payment procedures.",
    ]
    
    print("\n📚 Documents:")
    for i, doc in enumerate(documents):
        print(f"   [{i}] {doc}")
    
    # Tokenize documents (simple word split)
    tokenized_docs = [doc.lower().split() for doc in documents]
    
    # Create BM25 index
    bm25 = BM25Okapi(tokenized_docs)
    
    # Test queries
    queries = [
        "payment 30 days",
        "PAY-001",
        "GDPR data",
    ]
    
    print("\n🔍 BM25 Search Results:")
    print("-" * 50)
    
    for query in queries:
        print(f"\n   Query: \"{query}\"")
        query_tokens = query.lower().split()
        scores = bm25.get_scores(query_tokens)
        
        # Get top results
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        
        for idx, score in ranked[:2]:
            if score > 0:
                print(f"      {score:.3f} - {documents[idx][:50]}...")
    
    return bm25, documents


def demo_vector_search():
    """
    LESSON 4: Vector Search Comparison
    ==================================
    """
    if not HAS_QDRANT or not HAS_EMBEDDINGS:
        print("\n❌ Missing dependencies")
        return None, None
    
    print("\n" + "=" * 60)
    print("LESSON 4: Vector Search Comparison")
    print("=" * 60)
    
    # Same documents
    documents = [
        "All payments must be processed within 30 days of invoice receipt.",
        "Payment terms are NET-30 with 2% discount for early payment.",
        "Termination requires 30 days written notice from either party.",
        "GDPR compliance is mandatory for all data processing activities.",
        "Policy PAY-001 governs all vendor payment procedures.",
    ]
    
    # Initialize
    client = QdrantClient(":memory:")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create embeddings and store
    embeddings = model.encode(documents)
    
    client.create_collection(
        collection_name="docs",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    
    points = [
        PointStruct(id=i, vector=embeddings[i].tolist(), payload={"text": doc})
        for i, doc in enumerate(documents)
    ]
    client.upsert(collection_name="docs", points=points)
    
    # Test same queries
    queries = [
        "payment 30 days",
        "PAY-001",
        "GDPR data",
    ]
    
    print("\n🔍 Vector Search Results:")
    print("-" * 50)
    
    for query in queries:
        print(f"\n   Query: \"{query}\"")
        query_embedding = model.encode([query])[0]
        
        results = client.query_points(
            collection_name="docs",
            query=query_embedding.tolist(),
            limit=2,
            with_payload=True
        )
        
        for r in results.points:
            print(f"      {r.score:.3f} - {r.payload['text'][:50]}...")
    
    return client, model


class HybridSearcher:
    """
    LESSON 5: Building a Hybrid Searcher
    =====================================
    Combines BM25 and vector search using Reciprocal Rank Fusion (RRF).
    """
    
    def __init__(self):
        self.documents = []
        self.bm25 = None
        self.qdrant_client = None
        self.embedding_model = None
        
    def initialize(self, documents: list):
        """Initialize both search indices."""
        self.documents = documents
        
        # Initialize BM25
        tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
        
        # Initialize Vector Store
        self.qdrant_client = QdrantClient(":memory:")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        embeddings = self.embedding_model.encode(documents)
        
        self.qdrant_client.create_collection(
            collection_name="hybrid",
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        
        points = [
            PointStruct(id=i, vector=embeddings[i].tolist(), payload={"text": doc})
            for i, doc in enumerate(documents)
        ]
        self.qdrant_client.upsert(collection_name="hybrid", points=points)
        
        print(f"   ✅ Indexed {len(documents)} documents (BM25 + Vector)")
        
    def search_bm25(self, query: str, top_k: int = 5) -> list:
        """Search using BM25."""
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(idx, score) for idx, score in ranked[:top_k] if score > 0]
    
    def search_vector(self, query: str, top_k: int = 5) -> list:
        """Search using vectors."""
        query_embedding = self.embedding_model.encode([query])[0]
        
        results = self.qdrant_client.query_points(
            collection_name="hybrid",
            query=query_embedding.tolist(),
            limit=top_k,
            with_payload=True
        )
        
        return [(r.id, r.score) for r in results.points]
    
    def reciprocal_rank_fusion(self, bm25_results: list, vector_results: list, k: int = 60) -> list:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        
        RRF score = Σ 1 / (k + rank)
        
        This gives higher weight to documents ranked highly in BOTH methods.
        """
        scores = {}
        
        # Add BM25 scores
        for rank, (doc_id, _) in enumerate(bm25_results):
            if doc_id not in scores:
                scores[doc_id] = 0
            scores[doc_id] += 1 / (k + rank + 1)
        
        # Add vector scores
        for rank, (doc_id, _) in enumerate(vector_results):
            if doc_id not in scores:
                scores[doc_id] = 0
            scores[doc_id] += 1 / (k + rank + 1)
        
        # Sort by combined score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked
    
    def hybrid_search(self, query: str, top_k: int = 3) -> list:
        """
        Perform hybrid search combining BM25 and vector search.
        """
        # Get results from both methods
        bm25_results = self.search_bm25(query, top_k=5)
        vector_results = self.search_vector(query, top_k=5)
        
        # Combine with RRF
        combined = self.reciprocal_rank_fusion(bm25_results, vector_results)
        
        # Return top-k with documents
        results = []
        for doc_id, score in combined[:top_k]:
            results.append({
                'doc_id': doc_id,
                'score': score,
                'text': self.documents[doc_id]
            })
        
        return results


def demo_hybrid_search():
    """
    LESSON 6: Hybrid Search in Action
    ==================================
    """
    if not HAS_BM25 or not HAS_QDRANT or not HAS_EMBEDDINGS:
        print("\n❌ Missing dependencies")
        return
    
    print("\n" + "=" * 60)
    print("LESSON 6: Hybrid Search in Action")
    print("=" * 60)
    
    # Sample policy documents
    documents = [
        "All payments must be processed within 30 days of invoice receipt.",
        "Payment terms are NET-30 with 2% discount for early payment.",
        "Termination requires 30 days written notice from either party.",
        "GDPR compliance is mandatory for all data processing activities.",
        "Policy PAY-001 governs all vendor payment procedures.",
        "Data protection requires compliance with CCPA and GDPR regulations.",
        "Invoice processing follows policy PAY-001 guidelines.",
        "Vendors must provide 30 day notice for contract changes.",
    ]
    
    print("\n📦 Initializing Hybrid Searcher...")
    searcher = HybridSearcher()
    searcher.initialize(documents)
    
    # Test queries
    test_queries = [
        ("PAY-001", "Exact term match - BM25 excels"),
        ("monthly payment deadline", "Semantic match - Vector excels"),
        ("GDPR data compliance", "Hybrid - both contribute"),
    ]
    
    print("\n" + "-" * 60)
    
    for query, description in test_queries:
        print(f"\n🔍 Query: \"{query}\"")
        print(f"   ({description})")
        
        # Show individual results
        bm25_results = searcher.search_bm25(query, top_k=2)
        vector_results = searcher.search_vector(query, top_k=2)
        
        print(f"\n   BM25 Top-2:")
        for doc_id, score in bm25_results:
            print(f"      {score:.3f} - {documents[doc_id][:45]}...")
        
        print(f"\n   Vector Top-2:")
        for doc_id, score in vector_results:
            print(f"      {score:.3f} - {documents[doc_id][:45]}...")
        
        # Show hybrid results
        hybrid_results = searcher.hybrid_search(query, top_k=3)
        
        print(f"\n   🎯 HYBRID Top-3 (RRF combined):")
        for r in hybrid_results:
            print(f"      {r['score']:.4f} - {r['text'][:45]}...")


def explain_rrf():
    """
    LESSON 7: Understanding RRF
    ===========================
    """
    print("\n" + "=" * 60)
    print("LESSON 7: Understanding Reciprocal Rank Fusion (RRF)")
    print("=" * 60)
    print("""
    RRF combines rankings from multiple search methods.
    
    ┌────────────────────────────────────────────────────────────┐
    │ Formula: RRF_score = Σ  1 / (k + rank)                     │
    │                                                            │
    │ Where:                                                     │
    │   k = constant (typically 60)                              │
    │   rank = position in result list (0-indexed)               │
    └────────────────────────────────────────────────────────────┘
    
    Example:
    ┌────────────────────────────────────────────────────────────┐
    │ Document A:  BM25 rank=0, Vector rank=2                    │
    │   RRF = 1/(60+0) + 1/(60+2) = 0.0167 + 0.0161 = 0.0328    │
    │                                                            │
    │ Document B:  BM25 rank=1, Vector rank=0                    │
    │   RRF = 1/(60+1) + 1/(60+0) = 0.0164 + 0.0167 = 0.0331    │
    │                                                            │
    │ Document C:  BM25 only, rank=2                             │
    │   RRF = 1/(60+2) = 0.0161                                  │
    └────────────────────────────────────────────────────────────┘
    
    Result: B > A > C (documents appearing in BOTH methods rank higher)
    
    Why RRF works:
    ✅ No need to normalize different score scales
    ✅ Documents in multiple lists get boosted
    ✅ Simple and effective
    """)


def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     RULEMIRROR LEARNING: Hybrid Search (BM25 + Vector)   ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Lesson 1: Why hybrid
    explain_hybrid_search()
    
    # Lesson 2: BM25 theory
    explain_bm25()
    
    # Lesson 3: BM25 demo
    demo_bm25()
    
    # Lesson 4: Vector comparison
    demo_vector_search()
    
    # Lesson 5 & 6: Hybrid searcher
    demo_hybrid_search()
    
    # Lesson 7: RRF explanation
    explain_rrf()
    
    print("\n" + "=" * 60)
    print("✅ SUMMARY: What You Learned")
    print("=" * 60)
    print("""
    1. Vector search = semantic understanding
    2. BM25 search = exact keyword matching
    3. Hybrid = combine both for better results
    4. RRF = simple way to merge rankings
    5. Legal docs need BOTH (exact terms + meaning)
    
    For RuleMirror:
    - Vector finds related policies by meaning
    - BM25 ensures policy IDs are always matched
    - Hybrid gives the best of both worlds! 🎯
    
    Next: We'll integrate this into the full pipeline!
    """)


if __name__ == "__main__":
    main()
