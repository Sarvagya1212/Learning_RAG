# Learning 06: PDF RAG Pipeline

## 📅 Date: January 4, 2026

## 🎯 What We Learned
How to process real PDF policy documents and use them in a RAG compliance checker.

---

## 🔄 Complete Pipeline

```
PDF Document
     ↓
[Extract Text (pypdf)]
     ↓
[Parse Sections]
     ↓
[Smart Chunking]
     ↓
[Embed & Store (Qdrant)]
     ↓
Contract Clause → Search → Retrieve → LLM Analysis
     ↓
Compliance Report
```

---

## 📦 Key Components

### 1. PDF Text Extraction
```python
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    return "\n".join([page.extract_text() for page in reader.pages])
```

### 2. Smart Chunking
- Respects sentence boundaries
- Adds overlap between chunks
- Section-aware parsing

### 3. PolicyRAG Class
```python
rag = PolicyRAG()
rag.initialize()
rag.ingest_pdf("policy.pdf")
result = rag.analyze_clause("Payment in 45 days")
```

---

## 🏃 Try It

```bash
python learning/pdf_rag.py
```

Add your own PDFs to `data/policies/` and test!

---

## ✅ Next Steps
- **Learning 07:** Multi-Agent Systems

---

## 🧩 Practice Problems (Data Structures)

These problems will help you understand the core logic used in `pdf_rag.py`.

---

### Problem 1: Sliding Window Chunking
**Concept:** This is how we create overlapping chunks in `smart_chunk_text()`

```
Given a string and chunk_size=5, overlap=2, create overlapping chunks.

Input:  "ABCDEFGHIJKLMNO"
Output: ["ABCDE", "DEFGH", "GHIJK", "JKLMN", "MNO"]
                  ↑↑      ↑↑      ↑↑      ↑↑
              (overlap of 2 characters)
```

**Your Task:** Implement `chunk_with_overlap(text, chunk_size, overlap)`

```python
def chunk_with_overlap(text: str, chunk_size: int, overlap: int) -> list:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input string
        chunk_size: Size of each chunk
        overlap: Number of overlapping characters
        
    Returns:
        List of chunks
    """
    # YOUR CODE HERE
    pass

# Test
assert chunk_with_overlap("ABCDEFGHIJ", 4, 2) == ["ABCD", "CDEF", "EFGH", "GHIJ"]
```

<details>
<summary>💡 Hint</summary>
Use a while loop with step size = chunk_size - overlap
</details>

<details>
<summary>✅ Solution</summary>

```python
def chunk_with_overlap(text: str, chunk_size: int, overlap: int) -> list:
    chunks = []
    start = 0
    step = chunk_size - overlap
    
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += step
        
        # Stop if we've covered everything
        if end >= len(text):
            break
    
    return chunks
```
</details>

---

### Problem 2: Section Parser (Tree/Hierarchy)
**Concept:** This is how we parse document sections in `extract_sections_from_text()`

```
Given a list of lines with headers (UPPERCASE), group content under each header.

Input:
[
    "SECTION A",
    "line 1",
    "line 2",
    "SECTION B",
    "line 3"
]

Output:
{
    "SECTION A": ["line 1", "line 2"],
    "SECTION B": ["line 3"]
}
```

**Your Task:** Implement `parse_sections(lines)`

```python
def parse_sections(lines: list) -> dict:
    """
    Parse lines into sections based on uppercase headers.
    
    Args:
        lines: List of text lines
        
    Returns:
        Dictionary mapping section names to their content
    """
    # YOUR CODE HERE
    pass

# Test
lines = ["INTRO", "hello", "world", "BODY", "content here", "more content"]
expected = {"INTRO": ["hello", "world"], "BODY": ["content here", "more content"]}
assert parse_sections(lines) == expected
```

<details>
<summary>💡 Hint</summary>
Track current_section, append lines to it until you hit another header
</details>

<details>
<summary>✅ Solution</summary>

```python
def parse_sections(lines: list) -> dict:
    sections = {}
    current_section = None
    
    for line in lines:
        if line.isupper() and len(line) > 2:
            # New section header
            current_section = line
            sections[current_section] = []
        elif current_section:
            sections[current_section].append(line)
    
    return sections
```
</details>

---

### Problem 3: Find Top-K Similar (Vector Search)
**Concept:** This is how Qdrant finds similar policies in `search()`

```
Given a query vector and a list of document vectors, find the top-K most similar.
Use cosine similarity: cos(A,B) = (A·B) / (|A| * |B|)

Input:
  query = [1, 0, 1]
  documents = [
      ([1, 0, 1], "doc1"),   # Same as query
      ([0, 1, 0], "doc2"),   # Orthogonal
      ([1, 1, 1], "doc3"),   # Partial match
  ]
  k = 2

Output: ["doc1", "doc3"]  # Top 2 most similar
```

**Your Task:** Implement `find_top_k_similar(query, documents, k)`

```python
import math

def cosine_similarity(a: list, b: list) -> float:
    """Calculate cosine similarity between two vectors."""
    # YOUR CODE HERE
    pass

def find_top_k_similar(query: list, documents: list, k: int) -> list:
    """
    Find top-K most similar documents to the query.
    
    Args:
        query: Query vector
        documents: List of (vector, doc_id) tuples
        k: Number of results to return
        
    Returns:
        List of doc_ids sorted by similarity (highest first)
    """
    # YOUR CODE HERE
    pass

# Test
query = [1, 0, 1]
docs = [([1, 0, 1], "doc1"), ([0, 1, 0], "doc2"), ([1, 1, 0], "doc3")]
assert find_top_k_similar(query, docs, 2) == ["doc1", "doc3"]
```

<details>
<summary>💡 Hint</summary>
1. Calculate similarity for each doc
2. Sort by similarity descending
3. Take first K
</details>

<details>
<summary>✅ Solution</summary>

```python
import math

def cosine_similarity(a: list, b: list) -> float:
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(x * x for x in b))
    
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    
    return dot_product / (magnitude_a * magnitude_b)

def find_top_k_similar(query: list, documents: list, k: int) -> list:
    # Calculate similarity for each document
    scored = []
    for vector, doc_id in documents:
        score = cosine_similarity(query, vector)
        scored.append((score, doc_id))
    
    # Sort by score descending
    scored.sort(reverse=True, key=lambda x: x[0])
    
    # Return top K doc_ids
    return [doc_id for score, doc_id in scored[:k]]
```
</details>

---

## 🎯 Why These Matter

| Problem | PDF RAG Function | Real-World Use |
|---------|------------------|----------------|
| Sliding Window | `smart_chunk_text()` | Document chunking with context |
| Section Parser | `extract_sections_from_text()` | Structured document parsing |
| Top-K Similar | `search()` | Vector similarity search |
