"""
Learning 08: Hierarchical Chunking
===================================
This file teaches you about Hierarchical Chunking - a powerful technique
where you "Index Small but Retrieve Big".

Run this file: python hierarchical_chunker.py

Prerequisites:
    pip install qdrant-client sentence-transformers

What you'll learn:
    1. Why standard chunking has limitations
    2. What is hierarchical chunking?
    3. How to index sentences but retrieve paragraphs
    4. Building parent-child relationships
    5. Practical implementation for RuleMirror
"""

import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

# Check dependencies
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
    )
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


def explain_problem():
    """
    LESSON 1: The Problem with Standard Chunking
    =============================================
    """
    print("=" * 70)
    print("LESSON 1: The Problem with Standard Chunking")
    print("=" * 70)
    print("""
    Standard chunking splits documents into fixed-size pieces:
    
    ┌──────────────────────────────────────────────────────────────────┐
    │ PROBLEM: Big chunks vs Small chunks                              │
    ├──────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │  BIG CHUNKS (512 tokens):                                        │
    │  ✅ Good context for LLM                                         │
    │  ❌ Poor precision - query matches only part of chunk            │
    │  ❌ Lower similarity scores - irrelevant text dilutes match      │
    │                                                                  │
    │  SMALL CHUNKS (sentences):                                       │
    │  ✅ High precision - exact semantic match                        │
    │  ❌ No context for LLM - "it", "this", "the policy" = meaningless│
    │  ❌ Fragmented information                                       │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘
    
    SOLUTION: Hierarchical Chunking
    ───────────────────────────────
    Index on SMALL chunks (sentences) for precision
    BUT retrieve the PARENT chunk (paragraph) for context!
    """)


def explain_hierarchical_chunking():
    """
    LESSON 2: Hierarchical Chunking Architecture
    =============================================
    """
    print("\n" + "=" * 70)
    print("LESSON 2: Hierarchical Chunking Architecture")
    print("=" * 70)
    print("""
    The key insight: Decouple INDEXING from RETRIEVAL
    
    ┌────────────────────────────────────────────────────────────────────┐
    │  DOCUMENT HIERARCHY                                                │
    │                                                                    │
    │  Document (Full Policy)                                            │
    │      │                                                             │
    │      ├── Section: "Payment Terms"                                  │
    │      │       │                                                     │
    │      │       ├── Paragraph 1 (PARENT - what we RETRIEVE)           │
    │      │       │       ├── Sentence 1 (CHILD - what we INDEX)        │
    │      │       │       ├── Sentence 2                                │
    │      │       │       └── Sentence 3                                │
    │      │       │                                                     │
    │      │       └── Paragraph 2 (PARENT)                              │
    │      │               ├── Sentence 1 (CHILD)                        │
    │      │               └── Sentence 2                                │
    │      │                                                             │
    │      └── Section: "Termination"                                    │
    │              └── ...                                               │
    │                                                                    │
    └────────────────────────────────────────────────────────────────────┘
    
    WORKFLOW:
    ─────────
    1. Split document into PARENT chunks (paragraphs/sections)
    2. Split each parent into CHILD chunks (sentences)
    3. Store BOTH, with child → parent references
    4. INDEX: Embed and index the CHILD chunks (small, precise)
    5. SEARCH: Query returns matching CHILD chunks
    6. RETRIEVE: Look up and return PARENT chunks (big, contextual)
    """)


@dataclass
class Chunk:
    """Represents a chunk of text with hierarchical relationships."""
    id: str
    text: str
    chunk_type: str  # 'parent' or 'child'
    parent_id: Optional[str] = None  # For child chunks
    metadata: dict = field(default_factory=dict)
    
    @classmethod
    def create_parent(cls, text: str, metadata: dict = None) -> "Chunk":
        """Create a parent chunk (paragraph-level)."""
        return cls(
            id=str(uuid.uuid4()),
            text=text,
            chunk_type="parent",
            parent_id=None,
            metadata=metadata or {}
        )
    
    @classmethod
    def create_child(cls, text: str, parent_id: str, metadata: dict = None) -> "Chunk":
        """Create a child chunk (sentence-level) linked to a parent."""
        return cls(
            id=str(uuid.uuid4()),
            text=text,
            chunk_type="child",
            parent_id=parent_id,
            metadata=metadata or {}
        )


class HierarchicalChunker:
    """
    A chunker that creates hierarchical parent-child relationships.
    Index small (sentences), retrieve big (paragraphs).
    """
    
    def __init__(
        self,
        parent_chunk_size: int = 512,
        parent_chunk_overlap: int = 50,
        use_sentence_splitting: bool = True
    ):
        """
        Initialize the hierarchical chunker.
        
        Args:
            parent_chunk_size: Max characters for parent chunks
            parent_chunk_overlap: Overlap between parent chunks
            use_sentence_splitting: If True, split into sentences. If False, smaller fixed chunks.
        """
        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.use_sentence_splitting = use_sentence_splitting
    
    def split_into_parent_chunks(self, text: str) -> list[str]:
        """
        Split text into parent-level chunks (paragraphs or sections).
        
        This uses paragraph breaks first, then falls back to size-based splitting.
        """
        # First, try splitting by paragraphs (double newline)
        paragraphs = re.split(r'\n\s*\n', text.strip())
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # If paragraphs are too big, split them further
        parent_chunks = []
        for para in paragraphs:
            if len(para) <= self.parent_chunk_size:
                parent_chunks.append(para)
            else:
                # Split large paragraphs by size with overlap
                start = 0
                while start < len(para):
                    end = start + self.parent_chunk_size
                    chunk = para[start:end]
                    parent_chunks.append(chunk)
                    start = end - self.parent_chunk_overlap
        
        return parent_chunks
    
    def _split_text_into_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences using regex-based sentence boundary detection.
        
        Handles common abbreviations and edge cases.
        """
        # Pattern for sentence endings, avoiding common abbreviations
        abbreviations = r'(?<!\bMr)(?<!\bMrs)(?<!\bDr)(?<!\bMs)(?<!\bvs)(?<!\bSec)(?<!\bNo)'
        sentence_pattern = abbreviations + r'[.!?]+\s+'
        
        # Split by sentence boundaries
        sentences = re.split(sentence_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Ensure minimum sentence length (merge very short fragments)
        merged = []
        buffer = ""
        for sent in sentences:
            if len(sent) < 20 and buffer:
                buffer += " " + sent
            elif len(sent) < 20:
                buffer = sent
            else:
                if buffer:
                    merged.append(buffer + " " + sent)
                    buffer = ""
                else:
                    merged.append(sent)
        if buffer:
            merged.append(buffer)
        
        return merged
    
    def create_hierarchy(self, text: str, source: str = "unknown") -> tuple[list[Chunk], list[Chunk]]:
        """
        Create hierarchical chunks from text.
        
        Returns:
            Tuple of (parent_chunks, child_chunks)
        """
        parent_texts = self.split_into_parent_chunks(text)
        
        parent_chunks = []
        child_chunks = []
        
        for i, parent_text in enumerate(parent_texts):
            # Create parent chunk
            parent = Chunk.create_parent(
                text=parent_text,
                metadata={
                    "source": source,
                    "parent_index": i,
                    "level": "parent"
                }
            )
            parent_chunks.append(parent)
            
            # Create child chunks (sentences)
            if self.use_sentence_splitting:
                sentences = self._split_text_into_sentences(parent_text)
            else:
                # Fixed-size small chunks as alternative
                sentences = [parent_text[i:i+100] for i in range(0, len(parent_text), 80)]
            
            for j, sentence in enumerate(sentences):
                child = Chunk.create_child(
                    text=sentence,
                    parent_id=parent.id,
                    metadata={
                        "source": source,
                        "parent_index": i,
                        "child_index": j,
                        "level": "child"
                    }
                )
                child_chunks.append(child)
        
        return parent_chunks, child_chunks


class HierarchicalVectorStore:
    """
    Vector store that supports hierarchical chunking:
    - Indexes CHILD chunks for precise matching
    - Retrieves PARENT chunks for full context
    """
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the hierarchical vector store."""
        if not HAS_QDRANT or not HAS_EMBEDDINGS:
            raise RuntimeError("Install: pip install qdrant-client sentence-transformers")
        
        self.client = QdrantClient(":memory:")
        self.model = SentenceTransformer(embedding_model_name)
        self.vector_size = self.model.get_sentence_embedding_dimension()
        
        # Store for parent chunks (not indexed, just stored for retrieval)
        self.parent_store: dict[str, Chunk] = {}
        
        # Create collection for child chunks (what we index)
        self.client.create_collection(
            collection_name="child_chunks",
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE
            )
        )
        
        # Internal counter for point IDs
        self._point_counter = 0
    
    def add_chunks(self, parent_chunks: list[Chunk], child_chunks: list[Chunk]):
        """
        Add hierarchical chunks to the store.
        
        Parent chunks are stored in memory.
        Child chunks are indexed in the vector database.
        """
        # Store parent chunks for later retrieval
        for parent in parent_chunks:
            self.parent_store[parent.id] = parent
        
        # Index child chunks
        child_texts = [c.text for c in child_chunks]
        embeddings = self.model.encode(child_texts)
        
        points = []
        for i, child in enumerate(child_chunks):
            self._point_counter += 1
            point = PointStruct(
                id=self._point_counter,
                vector=embeddings[i].tolist(),
                payload={
                    "chunk_id": child.id,
                    "text": child.text,
                    "parent_id": child.parent_id,
                    "chunk_type": child.chunk_type,
                    **child.metadata
                }
            )
            points.append(point)
        
        self.client.upsert(collection_name="child_chunks", points=points)
        
        print(f"   ✅ Indexed {len(child_chunks)} child chunks")
        print(f"   ✅ Stored {len(parent_chunks)} parent chunks")
    
    def search(
        self,
        query: str,
        top_k: int = 3,
        return_parents: bool = True
    ) -> list[dict]:
        """
        Search for relevant content.
        
        Args:
            query: Search query
            top_k: Number of results
            return_parents: If True, return parent chunks. If False, return child chunks.
        
        Returns:
            List of results with text and metadata
        """
        # Embed query
        query_embedding = self.model.encode([query])[0]
        
        # Search child chunks
        results = self.client.query_points(
            collection_name="child_chunks",
            query=query_embedding.tolist(),
            limit=top_k,
            with_payload=True
        )
        
        output = []
        seen_parents: set[str] = set()
        
        for result in results.points:
            parent_id = result.payload["parent_id"]
            
            if return_parents:
                # Deduplicate parents (multiple child matches from same parent)
                if parent_id in seen_parents:
                    continue
                seen_parents.add(parent_id)
                
                # Return parent chunk for context
                parent = self.parent_store.get(parent_id)
                if parent:
                    output.append({
                        "text": parent.text,
                        "score": result.score,
                        "matched_sentence": result.payload["text"],
                        "metadata": parent.metadata
                    })
            else:
                # Return child chunk directly
                output.append({
                    "text": result.payload["text"],
                    "score": result.score,
                    "parent_id": parent_id,
                    "metadata": {
                        k: v for k, v in result.payload.items()
                        if k not in ["text", "chunk_id", "parent_id", "chunk_type"]
                    }
                })
        
        return output


def demo_hierarchical_chunking():
    """
    LESSON 3: Hands-On Demo
    =======================
    """
    if not HAS_QDRANT or not HAS_EMBEDDINGS:
        print("\n❌ Skipping demo - install dependencies first")
        return
    
    print("\n" + "=" * 70)
    print("LESSON 3: Hands-On Demo - Index Small, Retrieve Big")
    print("=" * 70)
    
    # Sample policy document
    policy_document = """
    Payment Terms and Conditions
    
    All vendor invoices must be submitted within 14 days of service completion. 
    Late submissions may result in delayed payment processing. The standard 
    payment terms are Net-30, meaning payment will be issued within thirty 
    business days of approved invoice receipt.
    
    For urgent payments, vendors may request expedited processing. Expedited 
    requests must be approved by the Finance Director. An expedited payment 
    fee of 2% may apply to cover additional processing costs.
    
    Termination and Exit Procedures
    
    Either party may terminate this agreement with 60 days written notice. 
    Upon termination, all outstanding invoices must be settled within 15 days. 
    Any prepaid amounts will be refunded on a pro-rata basis.
    
    In case of breach, immediate termination without notice is permitted. The 
    non-breaching party reserves the right to pursue legal remedies for damages. 
    All confidential materials must be returned within 7 days of termination.
    """
    
    print("\n📄 Sample Policy Document:")
    print("-" * 40)
    print(policy_document[:300] + "...")
    
    # Create hierarchical chunks
    print("\n🔄 Creating hierarchical chunks...")
    chunker = HierarchicalChunker(
        parent_chunk_size=300,  # Paragraph-ish size
        parent_chunk_overlap=30
    )
    
    parent_chunks, child_chunks = chunker.create_hierarchy(
        policy_document,
        source="vendor_policy.pdf"
    )
    
    print(f"\n📊 Chunking Results:")
    print(f"   Parent chunks (paragraphs): {len(parent_chunks)}")
    print(f"   Child chunks (sentences):   {len(child_chunks)}")
    
    # Show hierarchy
    print("\n📁 Hierarchy Structure:")
    for parent in parent_chunks[:2]:  # Show first 2 parents
        print(f"\n   PARENT [{parent.id[:8]}...]: {parent.text[:60]}...")
        children = [c for c in child_chunks if c.parent_id == parent.id]
        for child in children[:3]:  # Show first 3 children per parent
            print(f"      └── CHILD: {child.text[:50]}...")
    
    # Create vector store
    print("\n📦 Creating hierarchical vector store...")
    store = HierarchicalVectorStore()
    store.add_chunks(parent_chunks, child_chunks)
    
    # Search demo
    print("\n" + "-" * 70)
    print("🔍 SEARCH DEMO: Compare child vs parent retrieval")
    print("-" * 70)
    
    query = "how long to pay invoices"
    print(f"\n   Query: \"{query}\"")
    
    # Search returning child chunks (what we index)
    print("\n   A) Retrieving CHILD chunks (sentences):")
    child_results = store.search(query, top_k=2, return_parents=False)
    for i, r in enumerate(child_results):
        print(f"      [{i+1}] Score: {r['score']:.3f}")
        print(f"          Text: {r['text']}")
    
    # Search returning parent chunks (what we want)
    print("\n   B) Retrieving PARENT chunks (paragraphs) - BETTER!")
    parent_results = store.search(query, top_k=2, return_parents=True)
    for i, r in enumerate(parent_results):
        print(f"      [{i+1}] Score: {r['score']:.3f}")
        print(f"          Matched sentence: {r['matched_sentence'][:50]}...")
        print(f"          Full context: {r['text'][:100]}...")
    
    print("""
    
    💡 KEY INSIGHT:
    ────────────────
    Notice how returning the PARENT chunk gives you the full context!
    
    • Child match: "Net-30, meaning payment will be issued within thirty..."
    • Parent gives: The WHOLE paragraph about payment terms!
    
    This is essential for LLM-based RAG because the LLM needs context
    to understand what "it", "this policy", or "the deadline" refers to.
    """)


def demo_rulemirror_application():
    """
    LESSON 4: Application to RuleMirror
    ====================================
    """
    print("\n" + "=" * 70)
    print("LESSON 4: Application to RuleMirror")
    print("=" * 70)
    print("""
    How hierarchical chunking helps RuleMirror:
    
    ┌────────────────────────────────────────────────────────────────────┐
    │ CONTRACT REVIEW SCENARIO                                           │
    │                                                                    │
    │  Contract Clause: "Payment within 14 days"                         │
    │                          │                                         │
    │                          ▼                                         │
    │  ┌─────────────────────────────────────────────┐                   │
    │  │ Vector Search on SENTENCE-level index       │                   │
    │  │ → High precision match: "Net-30 payment"    │                   │
    │  └─────────────────────────────────────────────┘                   │
    │                          │                                         │
    │                          ▼                                         │
    │  ┌─────────────────────────────────────────────┐                   │
    │  │ Retrieve PARAGRAPH-level parent             │                   │
    │  │ → Full policy section with context          │                   │
    │  └─────────────────────────────────────────────┘                   │
    │                          │                                         │
    │                          ▼                                         │
    │  ┌─────────────────────────────────────────────┐                   │
    │  │ LLM Comparison with FULL CONTEXT            │                   │
    │  │ → "Contract says 14 days, policy says 30"   │                   │
    │  │ → "This is a VIOLATION of payment terms"    │                   │
    │  └─────────────────────────────────────────────┘                   │
    │                                                                    │
    └────────────────────────────────────────────────────────────────────┘
    
    BENEFITS FOR RULEMIRROR:
    ────────────────────────
    ✅ Precise matching on specific policy requirements
    ✅ Full context for LLM to understand policy nuances
    ✅ Can cite exact sentence that triggered the match
    ✅ Parent chunk provides surrounding context for explanation
    """)


def explain_alternatives():
    """
    LESSON 5: Alternative Approaches
    =================================
    """
    print("\n" + "=" * 70)
    print("LESSON 5: Alternative Hierarchical Approaches")
    print("=" * 70)
    print("""
    There are several ways to implement hierarchical retrieval:
    
    ┌────────────────────────────────────────────────────────────────────┐
    │ 1. PARENT-CHILD (What we built)                                    │
    │    • Index: Sentences                                              │
    │    • Retrieve: Paragraphs                                          │
    │    • Best for: Document QA, policy matching                        │
    ├────────────────────────────────────────────────────────────────────┤
    │ 2. SENTENCE WINDOW                                                 │
    │    • Index: Sentences                                              │
    │    • Retrieve: Sentence + N surrounding sentences                  │
    │    • Best for: When paragraphs aren't well-defined                 │
    ├────────────────────────────────────────────────────────────────────┤
    │ 3. DOCUMENT SUMMARY                                                │
    │    • Index: Document/section summaries                             │
    │    • Retrieve: Full document/section                               │
    │    • Best for: Topic-level search                                  │
    ├────────────────────────────────────────────────────────────────────┤
    │ 4. RECURSIVE (Multi-level)                                         │
    │    • Index: Multiple levels (sentence, paragraph, section)         │
    │    • Retrieve: Merge results across levels                         │
    │    • Best for: Complex documents with clear hierarchy              │
    └────────────────────────────────────────────────────────────────────┘
    
    For RuleMirror, PARENT-CHILD is ideal because:
    • Policies have clear paragraph structure
    • We need precise clause matching (sentence-level)
    • LLM needs full context for accurate comparison (paragraph-level)
    """)


def dsa_practice_questions():
    """
    DSA PRACTICE: Data Structure Questions from Hierarchical Chunking
    ==================================================================
    These questions help you understand the core data structures and
    algorithms used in hierarchical chunking implementations.
    """
    print("\n" + "=" * 70)
    print("DSA PRACTICE: Questions from Hierarchical Chunking Code")
    print("=" * 70)
    print("""
    ┌────────────────────────────────────────────────────────────────────┐
    │ Q1: TREE TRAVERSAL (Parent-Child Relationships)                    │
    ├────────────────────────────────────────────────────────────────────┤
    │ The hierarchical chunker creates a tree structure:                 │
    │                                                                    │
    │     Document                                                       │
    │     ├── Paragraph1 (Parent)                                        │
    │     │   ├── Sentence1 (Child)                                      │
    │     │   ├── Sentence2 (Child)                                      │
    │     │   └── Sentence3 (Child)                                      │
    │     └── Paragraph2 (Parent)                                        │
    │         ├── Sentence1 (Child)                                      │
    │         └── Sentence2 (Child)                                      │
    │                                                                    │
    │ QUESTION: How would you implement a function to:                   │
    │   a) Find all children of a given parent (O(n) vs O(1))?           │
    │   b) Find the parent of a given child (O(1) with parent_id)?       │
    │   c) Get all chunks at depth k (BFS/level-order traversal)?        │
    │                                                                    │
    │ HINT: Look at how `parent_store` uses a dict[str, Chunk]          │
    └────────────────────────────────────────────────────────────────────┘
    
    ┌────────────────────────────────────────────────────────────────────┐
    │ Q2: HASH MAP FOR O(1) LOOKUP (Parent Store)                        │
    ├────────────────────────────────────────────────────────────────────┤
    │ In the code:                                                       │
    │     self.parent_store: dict[str, Chunk] = {}                       │
    │                                                                    │
    │ QUESTION: Why use a hash map instead of a list?                    │
    │                                                                    │
    │   Option A: parent_store = []  # List of Chunk                     │
    │   Option B: parent_store = {}  # Dict mapping id -> Chunk          │
    │                                                                    │
    │ Compare time complexity:                                           │
    │   • List: Find parent by ID = O(n) scan                            │
    │   • Dict: Find parent by ID = O(1) lookup                          │
    │                                                                    │
    │ PRACTICE: Implement both and measure performance with 10,000       │
    │           parent chunks and 100,000 child lookups.                 │
    └────────────────────────────────────────────────────────────────────┘
    
    ┌────────────────────────────────────────────────────────────────────┐
    │ Q3: DEDUPLICATION WITH SET (Seen Parents)                          │
    ├────────────────────────────────────────────────────────────────────┤
    │ In search():                                                       │
    │     seen_parents: set[str] = set()                                 │
    │     if parent_id in seen_parents:                                  │
    │         continue                                                   │
    │     seen_parents.add(parent_id)                                    │
    │                                                                    │
    │ QUESTION: Why use a SET for deduplication?                         │
    │                                                                    │
    │   • set.add(x) = O(1) average                                      │
    │   • x in set = O(1) average                                        │
    │   • Vs list.append + `in list` = O(n)                              │
    │                                                                    │
    │ CODING CHALLENGE:                                                  │
    │   Given a list of (child_id, parent_id, score) tuples,             │
    │   return unique parents sorted by their BEST child score.          │
    │                                                                    │
    │   Input: [("c1", "p1", 0.9), ("c2", "p1", 0.8), ("c3", "p2", 0.7)]  │
    │   Output: [("p1", 0.9), ("p2", 0.7)]                                │
    └────────────────────────────────────────────────────────────────────┘
    
    ┌────────────────────────────────────────────────────────────────────┐
    │ Q4: SLIDING WINDOW (Chunk Overlap)                                 │
    ├────────────────────────────────────────────────────────────────────┤
    │ In split_into_parent_chunks():                                     │
    │     while start < len(para):                                       │
    │         end = start + self.parent_chunk_size                       │
    │         chunk = para[start:end]                                    │
    │         parent_chunks.append(chunk)                                │
    │         start = end - self.parent_chunk_overlap  # OVERLAP!        │
    │                                                                    │
    │ QUESTION: This is a SLIDING WINDOW with overlap!                   │
    │                                                                    │
    │ CLASSIC PROBLEM: Maximum sum subarray of size K                    │
    │   → Use sliding window to avoid O(n*k) recomputation               │
    │                                                                    │
    │ PRACTICE:                                                          │
    │   Given text of length N and chunk_size K with overlap O,          │
    │   how many chunks are created? Derive the formula:                 │
    │   chunks = ceil((N - O) / (K - O))                                 │
    └────────────────────────────────────────────────────────────────────┘
    
    ┌────────────────────────────────────────────────────────────────────┐
    │ Q5: STRING MATCHING / REGEX (Sentence Splitting)                   │
    ├────────────────────────────────────────────────────────────────────┤
    │ In _split_text_into_sentences():                                   │
    │     sentence_pattern = abbreviations + r'[.!?]+\\s+'               │
    │     sentences = re.split(sentence_pattern, text)                   │
    │                                                                    │
    │ QUESTION: What is the time complexity of regex split?              │
    │   → O(n * m) where n=text length, m=pattern complexity             │
    │                                                                    │
    │ ALTERNATIVE APPROACHES:                                            │
    │   1. KMP algorithm for fixed patterns                              │
    │   2. Finite automaton for regex                                    │
    │   3. Suffix trees for multiple pattern matching                    │
    │                                                                    │
    │ CODING CHALLENGE:                                                  │
    │   Implement sentence splitting WITHOUT regex using a state machine │
    │   that handles abbreviations like "Mr.", "Dr.", etc.               │
    └────────────────────────────────────────────────────────────────────┘
    
    ┌────────────────────────────────────────────────────────────────────┐
    │ Q6: MERGE SMALL FRAGMENTS (Buffer Pattern)                         │
    ├────────────────────────────────────────────────────────────────────┤
    │ In _split_text_into_sentences():                                   │
    │     merged = []                                                    │
    │     buffer = ""                                                    │
    │     for sent in sentences:                                         │
    │         if len(sent) < 20 and buffer:                              │
    │             buffer += " " + sent                                   │
    │         elif len(sent) < 20:                                       │
    │             buffer = sent                                          │
    │         else:                                                      │
    │             if buffer:                                             │
    │                 merged.append(buffer + " " + sent)                 │
    │                 buffer = ""                                        │
    │             else:                                                  │
    │                 merged.append(sent)                                │
    │                                                                    │
    │ QUESTION: This is stream processing with buffering!                │
    │                                                                    │
    │ SIMILAR PROBLEMS:                                                  │
    │   • Merge intervals (LeetCode #56)                                 │
    │   • Partition labels (LeetCode #763)                               │
    │   • Text justification (LeetCode #68)                              │
    └────────────────────────────────────────────────────────────────────┘
    
    ┌────────────────────────────────────────────────────────────────────┐
    │ Q7: UUID GENERATION (Unique IDs)                                   │
    ├────────────────────────────────────────────────────────────────────┤
    │ In Chunk.create_parent():                                          │
    │     id=str(uuid.uuid4())                                           │
    │                                                                    │
    │ QUESTION: Why UUID4 (random) vs UUID1 (timestamp-based)?           │
    │                                                                    │
    │   UUID4: 122 bits of randomness → collision probability:           │
    │          P(collision) = 1 - e^(-n²/2d) where d = 2^122             │
    │          For n=1 billion: ~0.0000000001% chance                    │
    │                                                                    │
    │ ALTERNATIVE: Use incremental IDs (0, 1, 2, 3...)                   │
    │   Pros: Smaller, sortable, no collision possible                   │
    │   Cons: Not suitable for distributed systems                       │
    │                                                                    │
    │ CODING CHALLENGE:                                                  │
    │   Implement your own ID generator that produces unique,            │
    │   sortable IDs with timestamp + random suffix (like ULID).         │
    └────────────────────────────────────────────────────────────────────┘
    """)


def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║     RULEMIRROR LEARNING: Hierarchical Chunking                   ║
    ║     "Index Small, Retrieve Big"                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Lesson 1: The problem
    explain_problem()
    
    # Lesson 2: The solution
    explain_hierarchical_chunking()
    
    # Lesson 3: Hands-on demo
    demo_hierarchical_chunking()
    
    # Lesson 4: RuleMirror application
    demo_rulemirror_application()
    
    # Lesson 5: Alternatives
    explain_alternatives()
    
    # DSA Practice Questions
    dsa_practice_questions()
    
    print("\n" + "=" * 70)
    print("✅ SUMMARY: What You Learned")
    print("=" * 70)
    print("""
    1. Standard chunking forces a tradeoff: precision vs context
    2. Hierarchical chunking: Index SMALL (sentences), Retrieve BIG (paragraphs)
    3. Parent-child relationships link sentences to their paragraphs
    4. Vector search on children, but return parents for LLM context
    5. Essential for RuleMirror's accurate policy-contract comparison
    
    KEY CLASSES:
    • HierarchicalChunker: Splits text into parent/child hierarchy
    • HierarchicalVectorStore: Indexes children, retrieves parents
    
    Next: Apply this to your actual PDF policies!
    """)


if __name__ == "__main__":
    main()
