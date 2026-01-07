# Learning 10: Simple Agent with Mistral

## 📅 Date: January 7, 2026

## 🎯 What We Learned
Building AI agents with CrewAI and Mistral for contract compliance.

---

## 🤖 Agent vs Simple LLM

| Aspect | Simple LLM | AI Agent |
|--------|------------|----------|
| Execution | One-shot | Think → Act → Observe loop |
| Tools | None | Can search, calculate, retrieve |
| Memory | Forgets | Maintains context |
| Control | Human-driven | Autonomous decisions |

---

## 🔄 Think-Act-Observe Pattern

```
         ┌───────────────────┐
         │    User Input     │
         └─────────┬─────────┘
                   ▼
         ┌───────────────────┐
    ┌───►│      THINK        │ ← Analyze, plan
    │    └─────────┬─────────┘
    │              ▼
    │    ┌───────────────────┐
    │    │       ACT         │ ← Execute (call LLM, tools)
    │    └─────────┬─────────┘
    │              ▼
    │    ┌───────────────────┐
    └────│     OBSERVE       │ ← Check result, iterate?
         └─────────┬─────────┘
                   ▼
         ┌───────────────────┐
         │     Response      │
         └───────────────────┘
```

---

## 📦 CrewAI Components

```python
from crewai import Agent, Task, Crew

# 1. AGENT - Who
agent = Agent(
    role="Legal Compliance Expert",
    goal="Review contracts against policies",
    backstory="Expert legal analyst..."
)

# 2. TASK - What
task = Task(
    description="Review this clause...",
    expected_output="Compliance status",
    agent=agent
)

# 3. CREW - Team
crew = Crew(agents=[agent], tasks=[task])
result = crew.kickoff()
```

---

## 🏃 Try It

```bash
pip install crewai crewai-tools
python learning/first_agent.py
```

---

## ✅ Next Steps
- **Learning 11:** Multi-Agent Systems
