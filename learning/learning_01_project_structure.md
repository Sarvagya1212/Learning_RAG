# Learning 01: Python Project Structure

## 📅 Date: January 3, 2026

## 🎯 What We Did
Set up a clean Python project structure for RuleMirror.

---

## 📁 Project Structure Explained

```
rule_mirror/
├── src/rule_mirror/     ← Main source code (package)
│   ├── __init__.py      ← Makes it a Python package
│   ├── agents/          ← AI agents (will analyze contracts)
│   ├── rag/             ← RAG components (retrieval + generation)
│   ├── parsers/         ← Document parsers (PDF, text)
│   └── utils/           ← Helper functions
├── data/
│   ├── contracts/       ← Input: draft contracts
│   └── policies/        ← Input: policy documents
├── tests/               ← Unit tests
├── learning/            ← These learning files!
├── requirements.txt     ← Python dependencies
├── .gitignore           ← Files Git should ignore
├── .env.example         ← Template for secrets
└── README.md            ← Project documentation
```

---

## 🧠 Key Concepts Learned

### 1. Virtual Environment (venv)
**What:** An isolated Python environment for your project.  
**Why:** Keeps dependencies separate from other projects.

```bash
# Create
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate

# You'll see (venv) in your terminal
```

### 2. `__init__.py` Files
**What:** Makes a folder a Python "package".  
**Why:** Lets you import code like `from rule_mirror.agents import SomeAgent`

### 3. `src/` Layout
**What:** Putting code inside `src/package_name/`.  
**Why:** Best practice - separates source code from config files.

### 4. `.gitignore`
**What:** Tells Git which files to NOT track.  
**Why:** Keeps secrets (`.env`), cache (`__pycache__`), and large data files out of version control.

### 5. `.env` for Secrets
**What:** Environment variables file for API keys.  
**Why:** Never commit secrets to Git! Use `.env.example` as a template.

---

## 📝 Files Created

| File | Purpose |
|------|---------|
| `src/rule_mirror/__init__.py` | Main package |
| `src/rule_mirror/agents/__init__.py` | AI agents module |
| `src/rule_mirror/rag/__init__.py` | RAG pipeline module |
| `src/rule_mirror/parsers/__init__.py` | Document parsers |
| `src/rule_mirror/utils/__init__.py` | Utilities |
| `requirements.txt` | Dependencies list |
| `.gitignore` | Git ignore rules |
| `.env.example` | Environment template |
| `README.md` | Project docs |

---

## ✅ Next Steps
- **Learning 02:** Build the PDF parser (extract text from policy documents)

---

## 💡 Tip
Always activate your virtual environment before running Python commands!
