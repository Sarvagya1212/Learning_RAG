# Learning 04: Vector Stores (Qdrant)

## 📅 Date: January 4, 2026

## 🎯 What We Learned
How to store and search embeddings using Qdrant vector database.

---

## 🧠 Vector Database vs Traditional Database

| Traditional DB | Vector DB |
|----------------|-----------|
| `WHERE text LIKE '%30 days%'` | Find similar to "payment deadline" |
| Exact keyword matches | Semantic similarity |
| Misses synonyms | Catches "30 days", "one month", "net-30" |

---

## 🔧 Qdrant Modes

```python
# Learning/Dev (no server needed!)
client = QdrantClient(":memory:")

# Persistent local
client = QdrantClient(path="./qdrant_db")

# Production server
client = QdrantClient("localhost:6333")
```

---

## 📦 Basic Operations

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# 1. Create client
client = QdrantClient(":memory:")

# 2. Create collection
client.create_collection(
    collection_name="policies",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# 3. Store vectors
client.upsert(collection_name="policies", points=[
    PointStruct(id=1, vector=[...], payload={"text": "...", "section": "..."})
])

# 4. Search
results = client.search(
    collection_name="policies",
    query_vector=[...],
    limit=5
)
```

---

## 🔍 Metadata Filtering

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Search only in "Payment Terms" section
results = client.search(
    collection_name="policies",
    query_vector=embedding,
    query_filter=Filter(must=[
        FieldCondition(key="section", match=MatchValue(value="Payment Terms"))
    ])
)
```

---

## 🏃 Try It

```bash
pip install qdrant-client sentence-transformers
python learning/simple_vectorstore.py
```

---


    ╔══════════════════════════════════════════════════════════╗
    ║     RULEMIRROR LEARNING: Simple Vector Store             ║
    ╚══════════════════════════════════════════════════════════╝

============================================================
LESSON 1: What is a Vector Database?
============================================================

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


============================================================
LESSON 2: Why Qdrant?
============================================================

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


============================================================
LESSON 3: Basic Operations (Hands-on)
============================================================

📦 Creating in-memory Qdrant database...
📦 Loading embedding model...

📝 Policy chunks to store: 5
   [1] Payment Terms: All vendor payments must be processed within thirt...
   [2] Contract Termination: Minimum notice period for termination is thirty (3...
   [3] Intellectual Property: All intellectual property created during engagemen...
   [4] Data Protection: All vendors handling personal data must comply wit...
   [5] Liability: Vendor liability shall not exceed the total contra...

🔄 Creating embeddings...
   Vector size: 384 dimensions

📊 Creating 'policies' collection...
💾 Storing vectors with metadata...
   ✅ Stored 5 vectors!

============================================================
LESSON 4: Similarity Search (The Core of RAG!)
============================================================

📄 Contract clauses to review:
   [1] Payment will be made within 14 days of invoice
   [2] Either party may terminate with 7 days notice
   [3] All code developed belongs to the contractor

------------------------------------------------------------

🔍 Checking: "Payment will be made within 14 days of invoice"
   📚 Most relevant policies:
      0.745 ██████████████
      [Payment Terms] All vendor payments must be processed within thirty (30) day...

      0.377 ███████
      [Contract Termination] Minimum notice period for termination is thirty (30) days....


🔍 Checking: "Either party may terminate with 7 days notice"
   📚 Most relevant policies:
      0.611 ████████████
      [Contract Termination] Minimum notice period for termination is thirty (30) days....

      0.309 ██████
      [Payment Terms] All vendor payments must be processed within thirty (30) day...


🔍 Checking: "All code developed belongs to the contractor"
   📚 Most relevant policies:
      0.362 ███████
      [Liability] Vendor liability shall not exceed the total contract value....

      0.342 ██████
      [Intellectual Property] All intellectual property created during engagement remains ...


============================================================
LESSON 5: Metadata Filtering (Advanced)
============================================================

    Sometimes you want to search ONLY within specific sections.

    Example: "Find payment-related policies" but ONLY in
             the "Payment Terms" section.

    Qdrant supports filtering on metadata (payload):


🔍 Query: "invoice payment deadline"

   Without filter (all sections):
      0.676 [Payment Terms] All vendor payments must be processed wi...
      0.337 [Contract Termination] Minimum notice period for termination is...
      0.199 [Liability] Vendor liability shall not exceed the to...

   With filter (Payment Terms only):
      0.676 [Payment Terms] All vendor payments must be processed wi...


    💡 Why filtering matters for RuleMirror:
    ────────────────────────────────────────
    • Filter by policy type (HR, Legal, Finance)
    • Filter by effective date (only current policies)
    • Filter by department (IT, Legal, Procurement)


============================================================
LESSON 6: Complete RuleMirror Workflow
============================================================

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


============================================================
✅ SUMMARY: What You Learned
============================================================

    1. Vector databases store embeddings for similarity search
    2. Qdrant is our choice (in-memory for dev, server for prod)
    3. Basic flow: Create collection → Store vectors → Search
    4. Metadata filtering narrows search to specific sections
    5. This is the "Retrieval" part of RAG!
