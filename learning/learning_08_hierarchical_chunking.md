# Learning 08: Hierarchical Chunking

> **Goal**: Index small (sentences) but retrieve big (paragraphs)

## Overview

Hierarchical chunking solves the precision vs context tradeoff in RAG systems:

| Approach | Precision | Context | Problem |
|----------|-----------|---------|---------|
| Big chunks (512 tokens) | ❌ Poor | ✅ Good | Irrelevant text dilutes similarity |
| Small chunks (sentences) | ✅ High | ❌ None | "it", "this policy" meaningless |
| **Hierarchical** | ✅ High | ✅ Good | Best of both! |

## Key Concept

```
Index SMALL chunks (sentences) for precise semantic matching
Retrieve PARENT chunks (paragraphs) for full LLM context
```

## Architecture

```
Document
├── Paragraph 1 (PARENT - what we RETRIEVE)
│   ├── Sentence 1 (CHILD - what we INDEX)
│   ├── Sentence 2 (CHILD)
│   └── Sentence 3 (CHILD)
└── Paragraph 2 (PARENT)
    ├── Sentence 1 (CHILD)
    └── Sentence 2 (CHILD)
```

## Key Classes

### `HierarchicalChunker`
Creates parent-child relationships from text.

```python
chunker = HierarchicalChunker(
    parent_chunk_size=300,
    parent_chunk_overlap=30,
    use_sentence_splitting=True
)

parents, children = chunker.create_hierarchy(document_text)
```

### `HierarchicalVectorStore`
Indexes children, retrieves parents.

```python
store = HierarchicalVectorStore()
store.add_chunks(parents, children)

# Search returns full paragraphs for LLM context!
results = store.search("payment deadline", return_parents=True)
```

---

## DSA Practice Questions

### Q1: Tree Traversal (Parent-Child Relationships)

The chunker creates a tree structure. Implement:
- Find all children of a parent → O(n) scan or O(1) with index
- Find parent of a child → O(1) via `parent_id`
- Get all chunks at depth k → BFS level-order traversal

### Q2: Hash Map for O(1) Lookup

```python
# Why dict instead of list?
parent_store: dict[str, Chunk] = {}  # O(1) lookup
# vs
parent_list: list[Chunk] = []  # O(n) to find by ID
```

**Practice**: Implement both and benchmark with 10,000 parents.

### Q3: Deduplication with Set

```python
seen_parents: set[str] = set()
if parent_id in seen_parents:  # O(1)
    continue
seen_parents.add(parent_id)    # O(1)
```

**Challenge**: Given `[(child_id, parent_id, score), ...]`, return unique parents sorted by best child score.

### Q4: Sliding Window (Chunk Overlap)

```python
while start < len(para):
    end = start + chunk_size
    chunk = para[start:end]
    start = end - overlap  # SLIDING WINDOW!
```

**Formula**: For text length N, chunk size K, overlap O:
```
num_chunks = ceil((N - O) / (K - O))
```

**Related LeetCode**: Maximum sum subarray of size K

### Q5: String Matching / Regex

```python
sentence_pattern = r'[.!?]+\s+'
sentences = re.split(sentence_pattern, text)
```

**Time Complexity**: O(n × m) where n=text length, m=pattern complexity

**Challenge**: Implement sentence splitting with a state machine (no regex).

### Q6: Stream Processing with Buffer

```python
merged = []
buffer = ""
for sent in sentences:
    if len(sent) < 20:
        buffer += " " + sent
    else:
        merged.append(buffer + sent)
        buffer = ""
```

**Related LeetCode**:
- #56 Merge Intervals
- #763 Partition Labels
- #68 Text Justification

### Q7: UUID Generation

```python
id = str(uuid.uuid4())  # 122 bits randomness
```

**Collision probability**: For n=1 billion IDs → ~0.0000000001%

**Challenge**: Implement a ULID-like generator (timestamp + random suffix).

---

## Running the Demo

```bash
python learning/hierarchical_chunker.py
```

## Next Steps

Apply hierarchical chunking to your actual PDF policies with `pdf_rag.py`!
