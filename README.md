# RuleMirror

AI-powered contract compliance reviewer.

## What It Does

Compares draft contracts against organizational policies to identify:
- ⚠️ Compliance violations
- 📋 Missing clauses
- ❓ Ambiguous language

## Project Structure

```
rule_mirror/
├── src/rule_mirror/     # Main source code
│   ├── agents/          # AI agents for analysis
│   ├── rag/             # RAG pipeline components
│   ├── parsers/         # Document parsers (PDF, text)
│   └── utils/           # Helper utilities
├── data/
│   ├── contracts/       # Draft contracts to review
│   └── policies/        # Company policy documents
├── tests/               # Unit and integration tests
├── docs/                # Documentation
└── learning/            # Step-by-step learning notes
```

## Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Coming soon...
