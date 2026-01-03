# Learning 02: Embeddings & Semantic Search

## 📅 Date: January 3, 2026

## 🎯 What We Learned
How embeddings power the semantic search in RuleMirror.

---

## 🧠 Key Concept: Embeddings

**Embeddings** convert text into numbers (vectors) that capture meaning.

```
"Payment due in 30 days" → [0.23, -0.45, 0.12, ..., 0.67]
                           ↑ 384+ dimensional vector
```

### Why Vectors?
| Text Approach | Vector Approach |
|---------------|-----------------|
| Keyword matching only | Understands meaning |
| "30 days" ≠ "one month" | "30 days" ≈ "one month" |
| Brittle, misses synonyms | Robust, catches intent |

---

## 🔍 Semantic Similarity

Similar meanings → Similar vectors → High similarity score

```
"Payment within 30 days"  ─┬─ Similarity: 0.92 (HIGH!)
"Invoice due in 30 days"  ─┘

"Payment within 30 days"  ─┬─ Similarity: 0.34 (LOW)
"Maintain insurance"      ─┘
```

This is how RuleMirror finds **relevant policies** for each contract clause!

---

## 🔧 How RuleMirror Uses It

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Index Policies (done once)                         │
│                                                             │
│   Policy Documents → Chunk → Embed → Store in Qdrant       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Review Contract (per review)                        │
│                                                             │
│   Contract Clause → Embed → Find similar policies → Compare │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Tools We Use

| Tool | Purpose |
|------|---------|
| `sentence-transformers` | Create embeddings locally |
| `mistral-embed` | Mistral's embedding API |
| `Qdrant` | Store & search vectors fast |

---

## 🏃 Try It Yourself

```bash
# Install the dependency
pip install sentence-transformers

# Run the interactive lesson
python learning/learn_embeddings.py
```

---

## 💡 Key Takeaways

1. **Embeddings = Text → Numbers** that capture meaning
2. **Cosine Similarity** measures how alike two texts are
3. **Semantic Search** finds by meaning, not just keywords
4. **Vector Databases** (Qdrant) store millions of embeddings

---

## ✅ Next Steps
- **Learning 03:** PDF parsing (extract text from policy documents)
