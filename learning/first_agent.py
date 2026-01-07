"""
Learning 10: Simple Agent with Mistral
========================================
This file teaches you how to build your first AI agent using CrewAI
with Mistral as the LLM backbone.

Run this file: python first_agent.py

Prerequisites:
    pip install crewai crewai-tools mistralai python-dotenv

What you'll learn:
    1. What are AI Agents?
    2. Agent vs RAG: Key differences
    3. CrewAI fundamentals (Agent, Task, Crew)
    4. Building a simple compliance checking agent
    5. Connecting to Mistral AI
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check dependencies
try:
    from crewai import Agent, Task, Crew, Process
    HAS_CREWAI = True
except ImportError:
    HAS_CREWAI = False
    print("⚠️  Run: pip install crewai crewai-tools")

try:
    from mistralai import Mistral
    HAS_MISTRAL = True
except ImportError:
    HAS_MISTRAL = False
    print("⚠️  Run: pip install mistralai")


# =============================================================================
# LESSON 1: What are AI Agents?
# =============================================================================

def explain_agents():
    """Explain what AI agents are and why we need them."""
    print("=" * 70)
    print("LESSON 1: What are AI Agents?")
    print("=" * 70)
    print("""
    ┌────────────────────────────────────────────────────────────────────┐
    │ SIMPLE LLM CALL                                                    │
    │                                                                    │
    │   User Query → LLM → Response                                      │
    │                                                                    │
    │   ❌ One-shot: No iteration or refinement                          │
    │   ❌ No tools: Can't search, calculate, or take actions            │
    │   ❌ No memory: Forgets between calls                              │
    └────────────────────────────────────────────────────────────────────┘
    
    ┌────────────────────────────────────────────────────────────────────┐
    │ AI AGENT                                                           │
    │                                                                    │
    │   User Query → Agent → [Think → Act → Observe] → Response          │
    │                           ↑         ↓                              │
    │                           └─────────┘  (loop until done)           │
    │                                                                    │
    │   ✅ Iterative: Refines until goal is achieved                     │
    │   ✅ Tool use: Can search, retrieve, calculate                     │
    │   ✅ Memory: Maintains context across steps                        │
    │   ✅ Autonomous: Makes decisions on its own                        │
    └────────────────────────────────────────────────────────────────────┘
    
    For RuleMirror:
    ───────────────
    We need agents because contract review is COMPLEX:
    • Multiple policies to check
    • Different sections need different analysis
    • Must decide which policies are relevant
    • Must reason about compliance/violation
    • Must generate actionable recommendations
    """)


# =============================================================================
# LESSON 2: Agent vs RAG
# =============================================================================

def explain_agent_vs_rag():
    """Explain the difference between RAG and Agents."""
    print("\n" + "=" * 70)
    print("LESSON 2: Agent vs RAG - What's the Difference?")
    print("=" * 70)
    print("""
    ┌────────────────────────────────────────────────────────────────────┐
    │ RAG (Retrieval Augmented Generation)                               │
    │                                                                    │
    │   Query → Embed → Search VectorDB → Retrieve → LLM → Response      │
    │                                                                    │
    │   ✅ Great for: QA, finding information                            │
    │   ❌ Limited: Fixed pipeline, no decision making                   │
    └────────────────────────────────────────────────────────────────────┘
    
    ┌────────────────────────────────────────────────────────────────────┐
    │ AGENTIC RAG                                                        │
    │                                                                    │
    │   Agent decides:                                                   │
    │     1. "Should I search the policy database?"                      │
    │     2. "What query should I use?"                                  │
    │     3. "Is this enough info or should I search again?"             │
    │     4. "Now I can analyze and respond"                             │
    │                                                                    │
    │   ✅ Great for: Complex analysis, multi-step reasoning             │
    │   ✅ Flexible: Agent controls the workflow                         │
    └────────────────────────────────────────────────────────────────────┘
    
    RuleMirror uses AGENTIC RAG:
    ────────────────────────────
    • Agent receives contract clause
    • Agent decides which policies to search for
    • Agent retrieves relevant policies (RAG)
    • Agent analyzes compliance (reasoning)
    • Agent generates report with recommendations
    """)


# =============================================================================
# LESSON 3: CrewAI Fundamentals
# =============================================================================

def explain_crewai():
    """Explain CrewAI concepts."""
    print("\n" + "=" * 70)
    print("LESSON 3: CrewAI Fundamentals")
    print("=" * 70)
    print("""
    CrewAI has three core components:
    
    ┌────────────────────────────────────────────────────────────────────┐
    │ 1. AGENT                                                           │
    │    A role-playing AI with:                                         │
    │    • Role: "Legal Compliance Expert"                               │
    │    • Goal: "Review contracts against policies"                     │
    │    • Backstory: Context about their expertise                      │
    │    • Tools: Functions they can use                                 │
    └────────────────────────────────────────────────────────────────────┘
    
    ┌────────────────────────────────────────────────────────────────────┐
    │ 2. TASK                                                            │
    │    A specific job for an agent:                                    │
    │    • Description: What to do                                       │
    │    • Expected Output: Format of result                             │
    │    • Agent: Who does this task                                     │
    └────────────────────────────────────────────────────────────────────┘
    
    ┌────────────────────────────────────────────────────────────────────┐
    │ 3. CREW                                                            │
    │    A team of agents working together:                              │
    │    • Agents: List of agents                                        │
    │    • Tasks: List of tasks                                          │
    │    • Process: Sequential or Hierarchical                           │
    └────────────────────────────────────────────────────────────────────┘
    
    Think of it like a company:
    • Agents = Employees with specific roles
    • Tasks = Work assignments
    • Crew = The team that executes a project
    """)


# =============================================================================
# LESSON 4: Simple Agent Example (No API needed)
# =============================================================================

def demo_agent_structure():
    """Show agent structure without running (for learning)."""
    print("\n" + "=" * 70)
    print("LESSON 4: Agent Structure (Code Example)")
    print("=" * 70)
    
    print("""
    Here's how you define an agent in CrewAI:
    
    ```python
    from crewai import Agent, Task, Crew
    
    # Define the agent
    compliance_agent = Agent(
        role="Legal Compliance Expert",
        goal="Review contract clauses against company policies",
        backstory='''
            You are an expert legal analyst specializing in 
            contract compliance. You have deep knowledge of 
            corporate policies and can identify violations,
            missing clauses, and areas of concern.
        ''',
        verbose=True,  # See agent's thinking
        allow_delegation=False  # Single agent for now
    )
    
    # Define a task
    review_task = Task(
        description='''
            Review this contract clause: {clause}
            
            Check against these policies:
            - Payments must be within 30 days
            - Termination requires 30 days notice
            - IP belongs to the Company
            
            Determine if it's COMPLIANT or VIOLATION.
            Explain your reasoning.
        ''',
        expected_output="Compliance status with detailed analysis",
        agent=compliance_agent
    )
    
    # Create and run the crew
    crew = Crew(
        agents=[compliance_agent],
        tasks=[review_task],
        verbose=True
    )
    
    result = crew.kickoff(inputs={
        "clause": "Payment will be made within 45 days"
    })
    ```
    """)


# =============================================================================
# LESSON 5: Building a Real Agent
# =============================================================================

class SimpleComplianceAgent:
    """
    A simple compliance checking agent using Mistral directly.
    
    This shows the agent pattern without CrewAI overhead for learning.
    """
    
    def __init__(self):
        self.api_key = os.getenv("MISTRAL_API_KEY")
        self.client = None
        self.policies = self._load_policies()
        
        if self.api_key and self.api_key != "your_mistral_key_here" and HAS_MISTRAL:
            self.client = Mistral(api_key=self.api_key)
    
    def _load_policies(self) -> dict:
        """Load our policy knowledge base."""
        return {
            "PAY-001": "All payments must be within 30 days of invoice receipt. No exceptions.",
            "TERM-001": "Termination requires 30 days written notice from either party.",
            "IP-001": "All intellectual property created belongs to the Company.",
            "LIA-001": "Vendor liability shall not exceed total contract value.",
            "DATA-001": "Data breaches must be reported within 24 hours."
        }
    
    def think(self, clause: str) -> dict:
        """
        The agent's thinking step.
        
        Analyzes the clause and decides which policies might be relevant.
        """
        print("\n🧠 THINKING...")
        
        # Simple keyword matching for demo (real agent would use embeddings)
        relevant = []
        clause_lower = clause.lower()
        
        if any(word in clause_lower for word in ["payment", "pay", "days", "invoice"]):
            relevant.append("PAY-001")
        if any(word in clause_lower for word in ["terminate", "termination", "notice"]):
            relevant.append("TERM-001")
        if any(word in clause_lower for word in ["ip", "intellectual", "property", "code", "work"]):
            relevant.append("IP-001")
        if any(word in clause_lower for word in ["liability", "liable", "damages"]):
            relevant.append("LIA-001")
        if any(word in clause_lower for word in ["data", "breach", "security"]):
            relevant.append("DATA-001")
        
        if not relevant:
            relevant = ["PAY-001", "TERM-001", "IP-001"]  # Default
        
        print(f"   Identified relevant policies: {relevant}")
        
        return {
            "clause": clause,
            "relevant_policies": relevant,
            "policy_texts": {k: self.policies[k] for k in relevant}
        }
    
    def act(self, context: dict) -> str:
        """
        The agent's action step.
        
        Uses LLM to analyze compliance.
        """
        print("\n⚡ ACTING (calling LLM)...")
        
        clause = context["clause"]
        policies = context["policy_texts"]
        
        policy_text = "\n".join([f"- [{k}]: {v}" for k, v in policies.items()])
        
        prompt = f"""You are a legal compliance expert. Analyze this contract clause against company policies.

## Company Policies:
{policy_text}

## Contract Clause:
"{clause}"

## Your Analysis:
1. Is this COMPLIANT or VIOLATION?
2. Which specific policy is relevant?
3. What is wrong (if violation)?
4. How should it be fixed?

Respond in this format:
STATUS: [COMPLIANT/VIOLATION]
POLICY: [policy ID]
ANALYSIS: [your analysis]
RECOMMENDATION: [suggested fix if needed]
"""
        
        if self.client:
            try:
                response = self.client.chat.complete(
                    model="mistral-small-latest",
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Error calling LLM: {e}"
        else:
            # Simulated response for demo
            return self._simulate_response(clause, policies)
    
    def _simulate_response(self, clause: str, policies: dict) -> str:
        """Simulate LLM response when no API key available."""
        clause_lower = clause.lower()
        
        if "45 days" in clause_lower or "60 days" in clause_lower:
            return """STATUS: VIOLATION
POLICY: PAY-001
ANALYSIS: The clause specifies a payment term that exceeds the 30-day maximum required by policy PAY-001.
RECOMMENDATION: Revise to "Payment shall be made within thirty (30) days of invoice receipt." """
        
        elif "7 days" in clause_lower and "notice" in clause_lower:
            return """STATUS: VIOLATION
POLICY: TERM-001
ANALYSIS: The 7-day notice period violates TERM-001 which requires minimum 30 days notice.
RECOMMENDATION: Revise to "Either party may terminate with thirty (30) days written notice." """
        
        elif "contractor" in clause_lower and "ip" in clause_lower:
            return """STATUS: VIOLATION
POLICY: IP-001
ANALYSIS: IP ownership by contractor violates policy IP-001.
RECOMMENDATION: Revise to "All intellectual property shall belong to the Company." """
        
        else:
            return """STATUS: NEEDS REVIEW
POLICY: Multiple
ANALYSIS: This clause should be reviewed by legal counsel.
RECOMMENDATION: Submit for manual review."""
    
    def observe(self, result: str) -> dict:
        """
        The agent's observation step.
        
        Parses the result and determines if more action is needed.
        """
        print("\n👁️ OBSERVING...")
        
        # Parse status from result
        status = "UNKNOWN"
        if "VIOLATION" in result.upper():
            status = "VIOLATION"
        elif "COMPLIANT" in result.upper():
            status = "COMPLIANT"
        elif "NEEDS REVIEW" in result.upper():
            status = "NEEDS_REVIEW"
        
        print(f"   Extracted status: {status}")
        
        return {
            "status": status,
            "full_analysis": result,
            "needs_more_action": False  # Single-pass for now
        }
    
    def run(self, clause: str) -> dict:
        """
        Execute the full Think → Act → Observe loop.
        """
        print("\n" + "=" * 60)
        print("🤖 AGENT EXECUTION")
        print("=" * 60)
        print(f"\n📄 Input: \"{clause}\"")
        
        # Step 1: Think
        context = self.think(clause)
        
        # Step 2: Act
        result = self.act(context)
        
        # Step 3: Observe
        observation = self.observe(result)
        
        print("\n" + "-" * 60)
        print("📋 FINAL RESULT:")
        print("-" * 60)
        print(observation["full_analysis"])
        
        return observation


def demo_simple_agent():
    """Demo the simple agent."""
    print("\n" + "=" * 70)
    print("LESSON 5: Running a Simple Agent")
    print("=" * 70)
    
    agent = SimpleComplianceAgent()
    
    # Test clauses
    test_clauses = [
        "Payment shall be made within 45 business days of invoice receipt.",
        "Either party may terminate with 7 days written notice.",
        "All IP and code developed belongs to the Contractor.",
    ]
    
    for clause in test_clauses:
        agent.run(clause)
        print("\n" + "=" * 60 + "\n")


# =============================================================================
# LESSON 6: CrewAI Full Example
# =============================================================================

def demo_crewai_agent():
    """Demo with actual CrewAI (requires API key)."""
    if not HAS_CREWAI:
        print("\n❌ CrewAI not installed. Run: pip install crewai")
        return
    
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key or api_key == "your_mistral_key_here":
        print("\n⚠️  Set MISTRAL_API_KEY in .env to run CrewAI demo")
        print("   Showing code structure instead...")
        demo_agent_structure()
        return
    
    print("\n" + "=" * 70)
    print("LESSON 6: CrewAI Agent in Action")
    print("=" * 70)
    
    # Set up LLM for CrewAI - uses environment variable
    os.environ["OPENAI_API_KEY"] = "dummy"  # CrewAI needs this
    os.environ["OPENAI_MODEL_NAME"] = "mistral/mistral-small-latest"
    
    try:
        # Create compliance agent
        compliance_agent = Agent(
            role="Legal Compliance Expert",
            goal="Review contract clauses against company policies and identify violations",
            backstory="""You are an expert legal analyst specializing in contract compliance.
            You have deep knowledge of corporate procurement policies including:
            - PAY-001: All payments must be within 30 days
            - TERM-001: Termination requires 30 days notice
            - IP-001: All IP belongs to the Company
            You are thorough, precise, and always cite specific policy violations.""",
            verbose=True,
            allow_delegation=False
        )
        
        # Create review task
        clause = "Payment shall be made within 45 days of invoice receipt."
        
        review_task = Task(
            description=f"""
            Review this contract clause for compliance:
            
            CLAUSE: "{clause}"
            
            Check against these policies:
            - PAY-001: All payments must be within 30 days
            - TERM-001: Termination requires 30 days notice
            - IP-001: All IP belongs to the Company
            
            Determine if COMPLIANT or VIOLATION.
            Cite the specific policy violated.
            Suggest how to fix the clause.
            """,
            expected_output="Compliance status with policy citation and fix",
            agent=compliance_agent
        )
        
        # Create and run crew
        crew = Crew(
            agents=[compliance_agent],
            tasks=[review_task],
            process=Process.sequential,
            verbose=True
        )
        
        print("\n🚀 Running CrewAI agent...")
        result = crew.kickoff()
        
        print("\n" + "=" * 60)
        print("📋 CREWAI RESULT:")
        print("=" * 60)
        print(result)
        
    except Exception as e:
        print(f"\n❌ CrewAI error: {e}")
        print("   This often means LLM configuration issue.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║     RULEMIRROR LEARNING: Simple Agent with Mistral               ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Lesson 1: What are agents
    explain_agents()
    
    # Lesson 2: Agent vs RAG
    explain_agent_vs_rag()
    
    # Lesson 3: CrewAI concepts
    explain_crewai()
    
    # Lesson 4: Code structure
    demo_agent_structure()
    
    # Lesson 5: Simple agent demo
    demo_simple_agent()
    
    # Lesson 6: CrewAI demo (if API key available)
    # demo_crewai_agent()  # Uncomment to run
    
    print("\n" + "=" * 70)
    print("✅ SUMMARY: What You Learned")
    print("=" * 70)
    print("""
    1. Agents = LLMs that Think → Act → Observe in a loop
    2. Agentic RAG > Simple RAG for complex tasks
    3. CrewAI: Agent (who) + Task (what) + Crew (team)
    4. Think-Act-Observe pattern for agent reasoning
    5. Mistral as the LLM backbone
    
    For RuleMirror:
    • Agents will orchestrate the compliance review
    • RAG provides the policy knowledge
    • Multiple agents can collaborate (next lesson!)
    
    Next: Multi-Agent Systems with CrewAI! 🚀
    """)


if __name__ == "__main__":
    main()
