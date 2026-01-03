# RuleMirror Tech Stack

## Overview

| Component | Technology | Why |
|-----------|------------|-----|
| **Orchestration** | LangChain / LlamaIndex | Agent coordination, chain management |
| **Agent Framework** | CrewAI / AutoGen | Multi-agent workflows, role-based delegation |
| **Vector Database** | Qdrant / Deep Lake | Persistent embeddings, hybrid search support |
| **LLM** | Mistral AI | Contract analysis, chain-of-thought reasoning |
| **Document Processing** | PyPDFLoader / Unstructured | PDF text extraction |
| **Reranking** | Cross-Encoder (sentence-transformers) | Precision retrieval filtering |
| **Evaluation** | Ragas / ARES | Faithfulness & context recall metrics |
| **Backend** | FastAPI | API for file upload, agent orchestration |
| **Frontend** | Streamlit | Rapid prototyping, file upload UI |
| **Compute** | AWS SageMaker (optional) | Self-hosted models for data privacy |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     FRONTEND (Streamlit)                        │
│              File Upload UI + Results Display                   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     BACKEND (FastAPI)                           │
│              API Endpoints + Agent Orchestration                │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
┌─────────────────┐ ┌───────────┐ ┌─────────────────┐
│ Document Parser │ │   Agents  │ │  Vector Store   │
│  (PyPDFLoader)  │ │ (CrewAI)  │ │    (Qdrant)     │
└────────┬────────┘ └─────┬─────┘ └────────┬────────┘
         │                │                │
         ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LLM (Mistral AI)                           │
│            Contract Analysis + Reasoning                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why These Choices?

### Mistral AI
- Open-weight models with commercial license
- Excellent reasoning capabilities
- Can be self-hosted for data privacy

### Qdrant
- Purpose-built for production RAG
- Hybrid search (dense + sparse vectors)
- Scales well for enterprise

### CrewAI
- Clean multi-agent abstraction
- Role-based delegation fits our compliance use case
- LangChain-compatible

### FastAPI + Streamlit
- FastAPI: High-performance async API
- Streamlit: Fastest path to working demo
