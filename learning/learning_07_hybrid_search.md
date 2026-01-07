# Learning 07: Hybrid Search (BM25 + Vector)

## 📅 Date: January 4, 2026

## 🎯 What We Learned
Combining keyword search (BM25) with semantic search (vectors) for better retrieval.

---

## ⚖️ Why Hybrid?

| Method | Strengths | Weaknesses |
|--------|-----------|------------|
| **Vector** | "30 days" ≈ "one month" | Misses exact terms like "PAY-001" |
| **BM25** | "GDPR" always finds "GDPR" | Misses synonyms |
| **Hybrid** | Best of both! | Slightly more complex |

---

## 🔧 Reciprocal Rank Fusion (RRF)

```
RRF_score = Σ 1/(k + rank)
```

**Example:**
- Doc A: BM25 rank=0, Vector rank=2 → RRF = 0.0328
- Doc B: BM25 rank=1, Vector rank=0 → RRF = 0.0331 ✓ Winner

Documents appearing in BOTH methods get boosted!

---

## 📦 HybridSearcher Class

```python
from hybrid_search import HybridSearcher

searcher = HybridSearcher()
searcher.initialize(documents)

results = searcher.hybrid_search("PAY-001 payment terms")
# → Best of keyword + semantic search
```

---

## 🏃 Try It

```bash
pip install rank-bm25
python learning/hybrid_search.py
```

---



    ╔══════════════════════════════════════════════════════════╗
    ║     RULEMIRROR LEARNING: Hybrid Search (BM25 + Vector)   ║
    ╚══════════════════════════════════════════════════════════╝

============================================================
LESSON 1: Why Hybrid Search?
============================================================

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


============================================================
LESSON 2: How BM25 Works
============================================================

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


============================================================
LESSON 3: BM25 in Action
============================================================

📚 Documents:
   [0] All payments must be processed within 30 days of invoice receipt.
   [1] Payment terms are NET-30 with 2% discount for early payment.
   [2] Termination requires 30 days written notice from either party.
   [3] GDPR compliance is mandatory for all data processing activities.
   [4] Policy PAY-001 governs all vendor payment procedures.

🔍 BM25 Search Results:
--------------------------------------------------

   Query: "payment 30 days"
      0.680 - Termination requires 30 days written notice from e...
      0.618 - All payments must be processed within 30 days of i...

   Query: "PAY-001"
      1.231 - Policy PAY-001 governs all vendor payment procedur...

   Query: "GDPR data"
      2.219 - GDPR compliance is mandatory for all data processi...

============================================================
LESSON 4: Vector Search Comparison
============================================================

🔍 Vector Search Results:
--------------------------------------------------

   Query: "payment 30 days"
      0.604 - All payments must be processed within 30 days of i...
      0.601 - Payment terms are NET-30 with 2% discount for earl...

   Query: "PAY-001"
      0.591 - Policy PAY-001 governs all vendor payment procedur...
      0.377 - Payment terms are NET-30 with 2% discount for earl...

   Query: "GDPR data"
      0.525 - GDPR compliance is mandatory for all data processi...
      0.071 - Policy PAY-001 governs all vendor payment procedur...

============================================================
LESSON 6: Hybrid Search in Action
============================================================

📦 Initializing Hybrid Searcher...
   ✅ Indexed 8 documents (BM25 + Vector)

------------------------------------------------------------

🔍 Query: "PAY-001"
   (Exact term match - BM25 excels)

   BM25 Top-2:
      1.113 - Invoice processing follows policy PAY-001 gui...
      1.050 - Policy PAY-001 governs all vendor payment pro...

   Vector Top-2:
      0.591 - Policy PAY-001 governs all vendor payment pro...
      0.529 - Invoice processing follows policy PAY-001 gui...

   🎯 HYBRID Top-3 (RRF combined):
      0.0325 - Invoice processing follows policy PAY-001 gui...
      0.0325 - Policy PAY-001 governs all vendor payment pro...
      0.0159 - Payment terms are NET-30 with 2% discount for...

🔍 Query: "monthly payment deadline"
   (Semantic match - Vector excels)

   BM25 Top-2:
      1.050 - Policy PAY-001 governs all vendor payment pro...
      0.898 - Payment terms are NET-30 with 2% discount for...

   Vector Top-2:
      0.490 - All payments must be processed within 30 days...
      0.441 - Payment terms are NET-30 with 2% discount for...

   🎯 HYBRID Top-3 (RRF combined):
      0.0323 - Payment terms are NET-30 with 2% discount for...
      0.0318 - Policy PAY-001 governs all vendor payment pro...
      0.0164 - All payments must be processed within 30 days...

🔍 Query: "GDPR data compliance"
   (Hybrid - both contribute)

   BM25 Top-2:
      2.830 - GDPR compliance is mandatory for all data pro...
      2.830 - Data protection requires compliance with CCPA...

   Vector Top-2:
      0.758 - GDPR compliance is mandatory for all data pro...
      0.730 - Data protection requires compliance with CCPA...

   🎯 HYBRID Top-3 (RRF combined):
      0.0328 - GDPR compliance is mandatory for all data pro...
      0.0323 - Data protection requires compliance with CCPA...
      0.0159 - Invoice processing follows policy PAY-001 gui...

============================================================
LESSON 7: Understanding Reciprocal Rank Fusion (RRF)
============================================================

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


============================================================
✅ SUMMARY: What You Learned
============================================================

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

---

## 🧩 Practice Problems (Data Structures)

These problems will help you understand the core algorithms used in `hybrid_search.py`.

---

### Problem 1: Term Frequency Counter (BM25 Foundation)
**Concept:** This is how BM25 calculates term frequency

```
Given a document and a query, count how many times each query term appears.

Input:
  document = "payment must be made within 30 days of payment receipt"
  query = ["payment", "days", "invoice"]

Output:
  {"payment": 2, "days": 1, "invoice": 0}
```

**Your Task:** Implement `count_term_frequency(document, query_terms)`

```python
def count_term_frequency(document: str, query_terms: list) -> dict:
    """
    Count occurrences of each query term in the document.
    
    Args:
        document: The document text
        query_terms: List of terms to count
        
    Returns:
        Dictionary mapping each term to its count
    """
    # YOUR CODE HERE
    pass

# Test
doc = "the quick brown fox jumps over the lazy dog the fox"
query = ["the", "fox", "cat"]
assert count_term_frequency(doc, query) == {"the": 3, "fox": 2, "cat": 0}
```

<details>
<summary>💡 Hint</summary>
Use a Counter or dict, split document into words, count matches
</details>

<details>
<summary>✅ Solution</summary>

```python
def count_term_frequency(document: str, query_terms: list) -> dict:
    words = document.lower().split()
    result = {}
    
    for term in query_terms:
        term_lower = term.lower()
        result[term] = words.count(term_lower)
    
    return result
```
</details>

---

### Problem 2: Reciprocal Rank Fusion (RRF)
**Concept:** This is how `reciprocal_rank_fusion()` combines search results

```
Given rankings from two search methods, compute RRF scores.
Formula: RRF_score(doc) = Σ 1/(k + rank), where k=60

Input:
  bm25_ranking = ["doc_a", "doc_b", "doc_c"]  # ranks: 0, 1, 2
  vector_ranking = ["doc_b", "doc_a", "doc_d"]  # ranks: 0, 1, 2
  k = 60

Output (sorted by RRF score):
  [("doc_b", 0.0331), ("doc_a", 0.0328), ("doc_c", 0.0161), ("doc_d", 0.0161)]
```

**Your Task:** Implement `compute_rrf(rankings, k)`

```python
def compute_rrf(rankings: list, k: int = 60) -> list:
    """
    Compute RRF scores for documents across multiple rankings.
    
    Args:
        rankings: List of ranking lists (each is a list of doc_ids)
        k: RRF constant (default 60)
        
    Returns:
        List of (doc_id, score) tuples sorted by score descending
    """
    # YOUR CODE HERE
    pass

# Test
bm25 = ["A", "B", "C"]
vector = ["B", "A", "D"]
result = compute_rrf([bm25, vector], k=60)
assert result[0][0] == "B"  # B should rank first
assert result[1][0] == "A"  # A should rank second
```

<details>
<summary>💡 Hint</summary>
1. Create a dict to accumulate scores per doc
2. For each ranking list, add 1/(k + rank) to each doc's score
3. Sort by score descending
</details>

<details>
<summary>✅ Solution</summary>

```python
def compute_rrf(rankings: list, k: int = 60) -> list:
    scores = {}
    
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            if doc_id not in scores:
                scores[doc_id] = 0
            scores[doc_id] += 1 / (k + rank)
    
    # Sort by score descending
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results
```
</details>

---

### Problem 3: Merge K Sorted Lists (Top-K Selection)
**Concept:** This is useful for efficiently combining results from multiple search sources

```
Given K sorted lists of (score, doc_id), merge into one sorted list.
This is a classic heap problem!

Input:
  lists = [
      [(0.9, "A"), (0.7, "B"), (0.5, "C")],  # BM25 results
      [(0.8, "D"), (0.6, "A"), (0.4, "E")],  # Vector results
  ]
  
Output (merged, sorted by score descending):
  [(0.9, "A"), (0.8, "D"), (0.7, "B"), (0.6, "A"), (0.5, "C"), (0.4, "E")]
```

**Your Task:** Implement `merge_k_sorted(lists)`

```python
import heapq

def merge_k_sorted(lists: list) -> list:
    """
    Merge K sorted lists into one sorted list.
    
    Args:
        lists: List of sorted lists, each containing (score, doc_id) tuples
               (sorted descending by score)
        
    Returns:
        Merged list sorted by score descending
    """
    # YOUR CODE HERE
    pass

# Test
l1 = [(0.9, "A"), (0.5, "B")]
l2 = [(0.8, "C"), (0.3, "D")]
result = merge_k_sorted([l1, l2])
assert result == [(0.9, "A"), (0.8, "C"), (0.5, "B"), (0.3, "D")]
```

<details>
<summary>💡 Hint</summary>
Use a max-heap (negate scores for min-heap). Push first element from each list, pop smallest, push next from that list.
</details>

<details>
<summary>✅ Solution</summary>

```python
import heapq

def merge_k_sorted(lists: list) -> list:
    # Use min-heap with negated scores for max behavior
    heap = []
    
    # Initialize heap with first element from each list
    for i, lst in enumerate(lists):
        if lst:
            score, doc_id = lst[0]
            # (negated_score, list_index, item_index, doc_id)
            heapq.heappush(heap, (-score, i, 0, doc_id))
    
    result = []
    
    while heap:
        neg_score, list_idx, item_idx, doc_id = heapq.heappop(heap)
        result.append((-neg_score, doc_id))
        
        # Push next item from same list
        next_idx = item_idx + 1
        if next_idx < len(lists[list_idx]):
            next_score, next_doc = lists[list_idx][next_idx]
            heapq.heappush(heap, (-next_score, list_idx, next_idx, next_doc))
    
    return result
```
</details>

---

## 🎯 Why These Matter

| Problem | Hybrid Search Function | Real-World Use |
|---------|------------------------|----------------|
| Term Frequency | `BM25Okapi` scoring | Keyword relevance |
| RRF | `reciprocal_rank_fusion()` | Combining search rankings |
| Merge K Sorted | Efficient result merging | Multi-source aggregation |

