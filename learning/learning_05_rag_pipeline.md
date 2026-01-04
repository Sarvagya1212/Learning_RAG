# Learning 05: RAG Pipeline

## 📅 Date: January 4, 2026

## 🎯 What We Learned
How to build a complete RAG (Retrieval Augmented Generation) pipeline for contract compliance checking.

---

## 🧠 What is RAG?

**RAG = Retrieval + Augmentation + Generation**

```
┌─────────────────────────────────────────────────────────────┐
│  1. RETRIEVE: Find relevant policies (vector search)        │
│  2. AUGMENT: Add policies to the LLM prompt                 │
│  3. GENERATE: LLM analyzes with full context                │
└─────────────────────────────────────────────────────────────┘
```

**Why?** LLMs don't know YOUR company's policies. We need to GIVE them the relevant context.

---

## 🔄 RAG Pipeline Flow

```
Contract Clause
     ↓
[Embed with SentenceTransformer]
     ↓
[Search Qdrant for similar policies]     ← RETRIEVAL
     ↓
[Build prompt with clause + policies]    ← AUGMENTATION
     ↓
[Send to Mistral AI]                     ← GENERATION
     ↓
Compliance Analysis Report
```

---

## 📦 Code Example

```python
# 1. RETRIEVE
clause_embedding = model.encode([clause])[0]
results = client.query_points(
    collection_name="policies",
    query=clause_embedding.tolist(),
    limit=3
)
policies = [r.payload for r in results.points]

# 2. AUGMENT
prompt = f"""
Company Policies:
{policies}

Contract Clause:
{clause}

Does this clause comply with the policies?
"""

# 3. GENERATE
response = mistral_client.chat.complete(
    model="mistral-large-latest",
    messages=[{"role": "user", "content": prompt}]
)
```

---

## 🏃 Try It

```bash
pip install qdrant-client sentence-transformers mistralai python-dotenv
python learning/simple_rag.py
```

💡 Add your `MISTRAL_API_KEY` to `.env` for full functionality!

---



    ╔══════════════════════════════════════════════════════════╗
    ║     RULEMIRROR LEARNING: Build Basic RAG Chain           ║
    ╚══════════════════════════════════════════════════════════╝

============================================================
LESSON 1: What is RAG?
============================================================

    RAG = Retrieval Augmented Generation

    The Problem:
    ┌─────────────────────────────────────────────────────────┐
    │  LLMs don't know YOUR company's policies!               │
    │  They were trained on general internet data.            │
    │  We need to GIVE them the relevant context.             │
    └─────────────────────────────────────────────────────────┘

    The Solution: RAG
    ┌─────────────────────────────────────────────────────────┐
    │  1. RETRIEVE: Find relevant policies (vector search)    │
    │  2. AUGMENT: Add policies to the LLM prompt             │
    │  3. GENERATE: LLM analyzes with full context            │
    └─────────────────────────────────────────────────────────┘

    For RuleMirror:
    ┌─────────────────────────────────────────────────────────┐
    │  Contract Clause → Find Related Policies → Ask LLM:     │
    │  "Does this clause comply with these policies?"         │
    └─────────────────────────────────────────────────────────┘


============================================================
LESSON 2: Setting Up the Knowledge Base
============================================================

📦 Initializing components...
   📚 Loaded 8 policy rules
   ✅ Policies indexed in vector store

============================================================
LESSON 3: Retrieval - Finding Relevant Policies
============================================================

📄 Contract clause to check:
   "Payment shall be made within 45 days of receiving the invoice."

🔍 Retrieved relevant policies:
   [PAY-001] (score: 0.743)
   All vendor payments must be processed within thirty (30) days of invoi...

   [PAY-002] (score: 0.511)
   Early payment discounts of up to 2% are authorized for payments within...

   [TERM-001] (score: 0.425)
   Minimum notice period for contract termination is thirty (30) days wri...


============================================================
LESSON 5: Generation - LLM Analysis
============================================================

📝 Augmented prompt (showing structure):
----------------------------------------
You are a legal compliance expert. Your job is to review contract clauses against company policies.

## Company Policies (MUST be followed):
- [PAY-001] All vendor payments must be processed within thirty (30) days of invoice receipt. No exceptions.
- [PAY-002] Early payment discounts of up to 2% are authorized for payments within 10 days.
- [TERM-001] Minimum notice period for contract termination is thirty (30) days written notice.

## Contract Clause to Review:
"Payment shall be made within 4...
----------------------------------------

🤖 Calling Mistral AI...

🤖 LLM Response:
----------------------------------------
**COMPLIANCE STATUS:** VIOLATION

**ANALYSIS:**
The contract clause states that **"Payment shall be made within 45 days of receiving the invoice."** This directly violates the company policy **[PAY-001]**, which mandates that **all vendor payments must be processed within thirty (30) days of invoice receipt**. The proposed 45-day payment term exceeds the maximum allowed 30-day window, making it non-compliant.

Additionally, while not a violation, the clause does not take advantage of the optional early payment discount authorized under **[PAY-002]**, which allows for a 2% discount if payment is made within 10 days. This is not a compliance issue but could be an opportunity for cost savings.

**RECOMMENDATION:**
To comply with **[PAY-001]**, the clause should be revised to align with the 30-day payment requirement. If the vendor is open to early payment discounts, the clause could also incorporate the 10-day option. Suggested fixes:

1. **Basic Compliance Fix:**
   *"Payment shall be made within thirty (30) days of receiving the invoice."*

2. **Enhanced Fix (if early payment discounts are desired):**
   *"Payment shall be made within thirty (30) days of receiving the invoice. A 2% discount will be applied if payment is made within ten (10) days of invoice receipt."*

This ensures adherence to **[PAY-001]** and **[PAY-002]** while maintaining flexibility for cost optimization. No changes are needed for **[TERM-001]** as the clause does not address termination.
----------------------------------------

============================================================
LESSON 6: Complete RAG Pipeline
============================================================

    Let's put it all together!

    ┌─────────────────────────────────────────────────────────────┐
    │                   RAG PIPELINE                              │
    │                                                             │
    │  Contract Clause                                            │
    │       ↓                                                     │
    │  [Embed with SentenceTransformer]                           │
    │       ↓                                                     │
    │  [Search Qdrant for similar policies]     ← RETRIEVAL       │
    │       ↓                                                     │
    │  [Build prompt with clause + policies]    ← AUGMENTATION    │
    │       ↓                                                     │
    │  [Send to Mistral AI]                     ← GENERATION      │
    │       ↓                                                     │
    │  Compliance Analysis Report                                 │
    └─────────────────────────────────────────────────────────────┘


============================================================
✅ SUMMARY: What You Learned
============================================================

    1. RAG = Retrieval + Augmentation + Generation
    2. We RETRIEVE relevant policies using vector similarity
    3. We AUGMENT the prompt with retrieved context
    4. We GENERATE analysis using the LLM (Mistral)
    5. The LLM can now reason about YOUR company's policies!

    This is the CORE of RuleMirror! 🎯