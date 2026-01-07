# Learning 09: Evaluation Test Cases

## 📅 Date: January 7, 2026

## 🎯 What We Learned
How to build a comprehensive evaluation set to measure RAG pipeline quality.

---

## 📊 Evaluation Set Contents

| Component | Count |
|-----------|-------|
| Policy Rules | 16 |
| Violation Tests | 10 |
| Compliant Tests | 5 |
| Needs Review Tests | 3 |
| **Total Test Cases** | **18** |

---

## 📦 Test Case Structure

```python
@dataclass
class TestCase:
    id: str                       # Unique identifier
    clause: ContractClause        # The clause to test
    expected_status: ComplianceStatus  # VIOLATION/COMPLIANT/NEEDS_REVIEW
    expected_policy_ids: list     # Ground truth: which policies match
    expected_reasoning: str       # What the LLM should identify
    suggested_fix: str           # How to fix violations
    difficulty: str              # easy/medium/hard
```

---

## 📏 Evaluation Metrics

### Retrieval Quality
| Metric | Formula | Meaning |
|--------|---------|---------|
| **Precision** | correct / retrieved | How accurate are results? |
| **Recall** | correct / expected | Did we find all relevant policies? |
| **MRR** | 1 / rank_of_first_correct | How high is the first correct result? |

### Generation Quality
| Metric | Meaning |
|--------|---------|
| **Status Accuracy** | Did it predict VIOLATION/COMPLIANT correctly? |
| **Reasoning Overlap** | Does reasoning match expected keywords? |

---

## 🏃 Try It

```bash
python learning/test_cases.py
```

This exports `tests/data/evaluation_set.json` for use with your RAG pipeline.

---

## 🧩 Practice Problems (Data Structures)

---

### Problem 1: Precision & Recall Calculator
**Concept:** Set intersection for IR metrics

```python
def calculate_metrics(expected: list, retrieved: list) -> tuple:
    """
    Input:
        expected = ["PAY-001", "PAY-002"]
        retrieved = ["PAY-001", "LIA-001", "PAY-002"]
    
    Output:
        precision = 2/3 = 0.67 (2 correct out of 3 retrieved)
        recall = 2/2 = 1.0 (found all 2 expected)
    """
    # YOUR CODE HERE
    pass
```

<details>
<summary>✅ Solution</summary>

```python
def calculate_metrics(expected: list, retrieved: list) -> tuple:
    expected_set = set(expected)
    retrieved_set = set(retrieved)
    correct = expected_set & retrieved_set
    
    precision = len(correct) / len(retrieved) if retrieved else 0
    recall = len(correct) / len(expected) if expected else 0
    
    return precision, recall
```
</details>

---

### Problem 2: Mean Reciprocal Rank (MRR)
**Concept:** Finding position in ranked list

```python
def calculate_mrr(expected: list, retrieved: list) -> float:
    """
    MRR = 1 / position of first correct result
    
    Input:
        expected = ["PAY-001"]
        retrieved = ["LIA-001", "PAY-001", "TERM-001"]
    
    Output: 0.5 (PAY-001 at position 2 → 1/2)
    """
    # YOUR CODE HERE
    pass
```

<details>
<summary>✅ Solution</summary>

```python
def calculate_mrr(expected: list, retrieved: list) -> float:
    expected_set = set(expected)
    for i, item in enumerate(retrieved):
        if item in expected_set:
            return 1.0 / (i + 1)
    return 0.0
```
</details>

---

### Problem 3: Jaccard Similarity for Reasoning
**Concept:** Set similarity for text comparison

```python
def jaccard_similarity(text1: str, text2: str) -> float:
    """
    Jaccard = |intersection| / |union|
    
    Input:
        text1 = "payment 30 days invoice"
        text2 = "invoice payment terms"
    
    Output: 2/5 = 0.4 (overlap: payment, invoice)
    """
    # YOUR CODE HERE
    pass
```

<details>
<summary>✅ Solution</summary>

```python
def jaccard_similarity(text1: str, text2: str) -> float:
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    intersection = words1 & words2
    union = words1 | words2
    
    return len(intersection) / len(union) if union else 0.0
```
</details>

---

## 🎯 Why These Matter

| Problem | Evaluation Function | Real-World Use |
|---------|---------------------|----------------|
| Precision/Recall | `calculate_retrieval_metrics()` | RAG retrieval quality |
| MRR | `calculate_retrieval_metrics()` | Ranking effectiveness |
| Jaccard | `calculate_reasoning_overlap()` | NLG evaluation |
