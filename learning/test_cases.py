"""
Learning 09: Build Evaluation Set
==================================
This file creates a comprehensive test suite for evaluating RuleMirror's
contract compliance checking capabilities.

Run this file: python test_cases.py

What you'll learn:
    1. How to structure evaluation datasets
    2. Ground truth labeling for RAG systems
    3. Metrics for retrieval quality (Precision, Recall, MRR)
    4. Metrics for generation quality (RAGAS-style)
    5. Building a reproducible test harness
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path
from enum import Enum


# =============================================================================
# COMPLIANCE STATUS
# =============================================================================

class ComplianceStatus(Enum):
    """Possible compliance verdicts."""
    COMPLIANT = "COMPLIANT"
    VIOLATION = "VIOLATION"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    

# =============================================================================
# TEST CASE DATA STRUCTURES
# =============================================================================

@dataclass
class PolicyRule:
    """Represents a single policy rule."""
    rule_id: str
    section: str
    text: str
    keywords: list = field(default_factory=list)  # For BM25 testing
    
    def to_dict(self):
        return asdict(self)


@dataclass
class ContractClause:
    """Represents a contract clause to be tested."""
    id: str
    text: str
    category: str  # payment, termination, ip, liability, data_protection
    
    def to_dict(self):
        return asdict(self)


@dataclass
class TestCase:
    """
    A complete test case for RuleMirror evaluation.
    
    Contains:
    - The contract clause being tested
    - Expected compliance status
    - Which policy rules SHOULD be retrieved
    - Expected analysis reasoning
    - Suggested fix (if violation)
    """
    id: str
    clause: ContractClause
    expected_status: ComplianceStatus
    expected_policy_ids: list  # Ground truth: which policies should match
    expected_reasoning: str    # What the LLM should identify
    suggested_fix: Optional[str] = None
    difficulty: str = "medium"  # easy, medium, hard
    
    def to_dict(self):
        return {
            "id": self.id,
            "clause": self.clause.to_dict(),
            "expected_status": self.expected_status.value,
            "expected_policy_ids": self.expected_policy_ids,
            "expected_reasoning": self.expected_reasoning,
            "suggested_fix": self.suggested_fix,
            "difficulty": self.difficulty
        }


# =============================================================================
# POLICY KNOWLEDGE BASE (Ground Truth)
# =============================================================================

POLICY_RULES = [
    # --- PAYMENT TERMS ---
    PolicyRule(
        rule_id="PAY-001",
        section="Payment Terms",
        text="All vendor payments must be processed within thirty (30) days of invoice receipt. No exceptions are permitted.",
        keywords=["payment", "30 days", "invoice", "vendor"]
    ),
    PolicyRule(
        rule_id="PAY-002",
        section="Payment Terms",
        text="Early payment discounts of up to 2% are authorized for payments made within ten (10) days of invoice receipt.",
        keywords=["early payment", "discount", "10 days"]
    ),
    PolicyRule(
        rule_id="PAY-003",
        section="Payment Terms",
        text="Late payment penalties in contracts shall not exceed 1.5% per month of the outstanding amount.",
        keywords=["late payment", "penalty", "1.5%"]
    ),
    
    # --- TERMINATION ---
    PolicyRule(
        rule_id="TERM-001",
        section="Termination",
        text="All contracts must include a termination clause with a minimum of thirty (30) days written notice from either party.",
        keywords=["termination", "30 days", "notice", "written"]
    ),
    PolicyRule(
        rule_id="TERM-002",
        section="Termination",
        text="Immediate termination without notice is permitted only in cases of material breach, fraud, or violation of applicable laws.",
        keywords=["immediate termination", "breach", "fraud"]
    ),
    PolicyRule(
        rule_id="TERM-003",
        section="Termination",
        text="Termination for convenience requires sixty (60) days advance notice and payment for all work completed to date.",
        keywords=["termination for convenience", "60 days"]
    ),
    
    # --- INTELLECTUAL PROPERTY ---
    PolicyRule(
        rule_id="IP-001",
        section="Intellectual Property",
        text="All intellectual property created during the engagement shall remain the exclusive property of the Company.",
        keywords=["intellectual property", "IP", "ownership", "company"]
    ),
    PolicyRule(
        rule_id="IP-002",
        section="Intellectual Property",
        text="Vendors must execute an IP assignment agreement prior to project commencement.",
        keywords=["IP assignment", "agreement"]
    ),
    PolicyRule(
        rule_id="IP-003",
        section="Intellectual Property",
        text="Pre-existing intellectual property must be disclosed and properly licensed to the Company before use.",
        keywords=["pre-existing IP", "license", "disclosure"]
    ),
    
    # --- LIABILITY ---
    PolicyRule(
        rule_id="LIA-001",
        section="Liability",
        text="Total vendor liability shall not exceed the total value of the contract unless gross negligence or willful misconduct is involved.",
        keywords=["liability", "cap", "contract value"]
    ),
    PolicyRule(
        rule_id="LIA-002",
        section="Liability",
        text="All vendors must maintain professional liability insurance of at least $1,000,000.",
        keywords=["insurance", "professional liability", "1 million"]
    ),
    PolicyRule(
        rule_id="LIA-003",
        section="Liability",
        text="Indemnification clauses must be mutual and reasonable, covering negligence and IP infringement.",
        keywords=["indemnification", "mutual", "negligence"]
    ),
    
    # --- DATA PROTECTION ---
    PolicyRule(
        rule_id="DATA-001",
        section="Data Protection",
        text="All vendors handling personal data must comply with GDPR, CCPA, and all applicable data protection regulations.",
        keywords=["GDPR", "CCPA", "data protection", "personal data"]
    ),
    PolicyRule(
        rule_id="DATA-002",
        section="Data Protection",
        text="Data Processing Agreements (DPAs) are mandatory for all vendors processing personal data.",
        keywords=["DPA", "data processing agreement"]
    ),
    PolicyRule(
        rule_id="DATA-003",
        section="Data Protection",
        text="Data breach notification must occur within 24 hours of discovery.",
        keywords=["data breach", "notification", "24 hours"]
    ),
    PolicyRule(
        rule_id="DATA-004",
        section="Data Protection",
        text="All company data must be deleted within 30 days of contract termination.",
        keywords=["data deletion", "termination", "30 days"]
    ),
]


# =============================================================================
# TEST CASES - VIOLATIONS
# =============================================================================

VIOLATION_TEST_CASES = [
    # --- PAYMENT VIOLATIONS ---
    TestCase(
        id="V-PAY-001",
        clause=ContractClause(
            id="C001",
            text="Payment for services shall be made within forty-five (45) business days of invoice receipt.",
            category="payment"
        ),
        expected_status=ComplianceStatus.VIOLATION,
        expected_policy_ids=["PAY-001"],
        expected_reasoning="Contract specifies 45 days payment term which exceeds the maximum 30 days allowed by policy PAY-001.",
        suggested_fix="Payment for services shall be made within thirty (30) days of invoice receipt.",
        difficulty="easy"
    ),
    TestCase(
        id="V-PAY-002",
        clause=ContractClause(
            id="C002",
            text="Late payments shall incur a penalty of 3% per month on outstanding balances.",
            category="payment"
        ),
        expected_status=ComplianceStatus.VIOLATION,
        expected_policy_ids=["PAY-003"],
        expected_reasoning="Late payment penalty of 3% exceeds the maximum 1.5% per month allowed by policy PAY-003.",
        suggested_fix="Late payments shall incur a penalty of 1.5% per month on outstanding balances.",
        difficulty="easy"
    ),
    
    # --- TERMINATION VIOLATIONS ---
    TestCase(
        id="V-TERM-001",
        clause=ContractClause(
            id="C003",
            text="Either party may terminate this agreement with seven (7) days written notice.",
            category="termination"
        ),
        expected_status=ComplianceStatus.VIOLATION,
        expected_policy_ids=["TERM-001"],
        expected_reasoning="7-day termination notice is insufficient. Policy TERM-001 requires minimum 30 days notice.",
        suggested_fix="Either party may terminate this agreement with thirty (30) days written notice.",
        difficulty="easy"
    ),
    TestCase(
        id="V-TERM-002",
        clause=ContractClause(
            id="C004",
            text="The Client may terminate immediately for any reason without cause.",
            category="termination"
        ),
        expected_status=ComplianceStatus.VIOLATION,
        expected_policy_ids=["TERM-001", "TERM-002", "TERM-003"],
        expected_reasoning="Immediate termination 'without cause' violates policies. TERM-002 only allows immediate termination for breach/fraud. TERM-003 requires 60 days for convenience termination.",
        suggested_fix="The Client may terminate for convenience with sixty (60) days advance written notice.",
        difficulty="medium"
    ),
    
    # --- IP VIOLATIONS ---
    TestCase(
        id="V-IP-001",
        clause=ContractClause(
            id="C005",
            text="All intellectual property and work product developed under this agreement shall remain the property of the Contractor.",
            category="ip"
        ),
        expected_status=ComplianceStatus.VIOLATION,
        expected_policy_ids=["IP-001"],
        expected_reasoning="IP ownership by Contractor violates policy IP-001 which requires all IP to remain Company property.",
        suggested_fix="All intellectual property and work product developed under this agreement shall be the exclusive property of the Company.",
        difficulty="easy"
    ),
    TestCase(
        id="V-IP-002",
        clause=ContractClause(
            id="C006",
            text="The Contractor grants the Company a non-exclusive license to use the deliverables.",
            category="ip"
        ),
        expected_status=ComplianceStatus.VIOLATION,
        expected_policy_ids=["IP-001", "IP-002"],
        expected_reasoning="A 'non-exclusive license' does not satisfy IP-001 which requires exclusive ownership. IP-002 requires full assignment, not licensing.",
        suggested_fix="The Contractor assigns all rights, title, and interest in the deliverables to the Company.",
        difficulty="medium"
    ),
    
    # --- LIABILITY VIOLATIONS ---
    TestCase(
        id="V-LIA-001",
        clause=ContractClause(
            id="C007",
            text="Vendor liability under this agreement shall be limited to $10,000.",
            category="liability"
        ),
        expected_status=ComplianceStatus.VIOLATION,
        expected_policy_ids=["LIA-001"],
        expected_reasoning="Fixed $10,000 cap may be below total contract value, violating LIA-001 which sets liability cap at contract value.",
        suggested_fix="Vendor liability shall not exceed the total value of this agreement.",
        difficulty="medium"
    ),
    TestCase(
        id="V-LIA-002",
        clause=ContractClause(
            id="C008",
            text="The Vendor shall maintain professional liability insurance of $500,000.",
            category="liability"
        ),
        expected_status=ComplianceStatus.VIOLATION,
        expected_policy_ids=["LIA-002"],
        expected_reasoning="$500,000 insurance is below the minimum $1,000,000 required by policy LIA-002.",
        suggested_fix="The Vendor shall maintain professional liability insurance of at least $1,000,000.",
        difficulty="easy"
    ),
    
    # --- DATA PROTECTION VIOLATIONS ---
    TestCase(
        id="V-DATA-001",
        clause=ContractClause(
            id="C009",
            text="The Vendor shall notify the Company of any data breach within 72 hours.",
            category="data_protection"
        ),
        expected_status=ComplianceStatus.VIOLATION,
        expected_policy_ids=["DATA-003"],
        expected_reasoning="72-hour notification exceeds the 24-hour requirement in policy DATA-003.",
        suggested_fix="The Vendor shall notify the Company of any data breach within 24 hours of discovery.",
        difficulty="easy"
    ),
    TestCase(
        id="V-DATA-002",
        clause=ContractClause(
            id="C010",
            text="Upon termination, the Vendor may retain company data for up to 90 days for archival purposes.",
            category="data_protection"
        ),
        expected_status=ComplianceStatus.VIOLATION,
        expected_policy_ids=["DATA-004"],
        expected_reasoning="90-day retention exceeds the 30-day deletion requirement in policy DATA-004.",
        suggested_fix="Upon termination, all company data shall be deleted within 30 days.",
        difficulty="easy"
    ),
]


# =============================================================================
# TEST CASES - COMPLIANT
# =============================================================================

COMPLIANT_TEST_CASES = [
    TestCase(
        id="C-PAY-001",
        clause=ContractClause(
            id="C011",
            text="Payment shall be made within thirty (30) days of receipt of a valid invoice.",
            category="payment"
        ),
        expected_status=ComplianceStatus.COMPLIANT,
        expected_policy_ids=["PAY-001"],
        expected_reasoning="Payment term of 30 days matches policy PAY-001 requirement.",
        difficulty="easy"
    ),
    TestCase(
        id="C-TERM-001",
        clause=ContractClause(
            id="C012",
            text="Either party may terminate with 30 days written notice. Immediate termination is permitted for material breach.",
            category="termination"
        ),
        expected_status=ComplianceStatus.COMPLIANT,
        expected_policy_ids=["TERM-001", "TERM-002"],
        expected_reasoning="Clause satisfies both TERM-001 (30 days notice) and TERM-002 (immediate only for breach).",
        difficulty="medium"
    ),
    TestCase(
        id="C-IP-001",
        clause=ContractClause(
            id="C013",
            text="All intellectual property created during the engagement shall be the exclusive property of the Company. Contractor shall execute IP assignment agreements.",
            category="ip"
        ),
        expected_status=ComplianceStatus.COMPLIANT,
        expected_policy_ids=["IP-001", "IP-002"],
        expected_reasoning="Clause properly assigns IP to Company (IP-001) and includes assignment agreement (IP-002).",
        difficulty="easy"
    ),
    TestCase(
        id="C-LIA-001",
        clause=ContractClause(
            id="C014",
            text="Total vendor liability shall not exceed the aggregate fees paid under this agreement. Vendor must maintain $2,000,000 professional liability insurance.",
            category="liability"
        ),
        expected_status=ComplianceStatus.COMPLIANT,
        expected_policy_ids=["LIA-001", "LIA-002"],
        expected_reasoning="Liability capped at contract value (LIA-001) and insurance exceeds $1M minimum (LIA-002).",
        difficulty="medium"
    ),
    TestCase(
        id="C-DATA-001",
        clause=ContractClause(
            id="C015",
            text="Vendor shall comply with GDPR and CCPA. Data breaches must be reported within 24 hours. All data deleted within 30 days of termination.",
            category="data_protection"
        ),
        expected_status=ComplianceStatus.COMPLIANT,
        expected_policy_ids=["DATA-001", "DATA-003", "DATA-004"],
        expected_reasoning="Clause addresses all major data protection requirements: regulations, breach notification, and deletion timelines.",
        difficulty="medium"
    ),
]


# =============================================================================
# TEST CASES - NEEDS REVIEW (Ambiguous/Edge Cases)
# =============================================================================

NEEDS_REVIEW_TEST_CASES = [
    TestCase(
        id="R-001",
        clause=ContractClause(
            id="C016",
            text="Payment shall be made in accordance with standard terms.",
            category="payment"
        ),
        expected_status=ComplianceStatus.NEEDS_REVIEW,
        expected_policy_ids=["PAY-001"],
        expected_reasoning="'Standard terms' is ambiguous. Need clarification on what payment timeline is defined.",
        difficulty="hard"
    ),
    TestCase(
        id="R-002",
        clause=ContractClause(
            id="C017",
            text="The agreement may be terminated by either party with reasonable notice.",
            category="termination"
        ),
        expected_status=ComplianceStatus.NEEDS_REVIEW,
        expected_policy_ids=["TERM-001"],
        expected_reasoning="'Reasonable notice' is subjective. Policy requires specific 30-day minimum.",
        difficulty="hard"
    ),
    TestCase(
        id="R-003",
        clause=ContractClause(
            id="C018",
            text="IP ownership shall be determined based on the nature of the deliverables.",
            category="ip"
        ),
        expected_status=ComplianceStatus.NEEDS_REVIEW,
        expected_policy_ids=["IP-001"],
        expected_reasoning="Conditional IP ownership is unclear. Policy IP-001 requires all IP to be Company property.",
        difficulty="hard"
    ),
]


# =============================================================================
# EVALUATION METRICS
# =============================================================================

@dataclass
class EvaluationResult:
    """Results from evaluating a single test case."""
    test_case_id: str
    
    # Retrieval metrics
    retrieved_policy_ids: list
    retrieval_precision: float  # How many retrieved were correct
    retrieval_recall: float     # How many correct were retrieved
    retrieval_mrr: float        # Mean Reciprocal Rank of first correct
    
    # Generation metrics  
    predicted_status: Optional[str]
    status_correct: bool
    reasoning_overlap: float    # Jaccard similarity of key terms
    
    def to_dict(self):
        return asdict(self)


def calculate_retrieval_metrics(
    expected: list, 
    retrieved: list
) -> tuple[float, float, float]:
    """
    Calculate retrieval quality metrics.
    
    Returns:
        (precision, recall, mrr)
    """
    if not retrieved:
        return 0.0, 0.0, 0.0
    
    expected_set = set(expected)
    retrieved_set = set(retrieved)
    
    # Precision: correct / retrieved
    correct = expected_set & retrieved_set
    precision = len(correct) / len(retrieved) if retrieved else 0.0
    
    # Recall: correct / expected
    recall = len(correct) / len(expected) if expected else 0.0
    
    # MRR: 1/rank of first correct result
    mrr = 0.0
    for i, policy_id in enumerate(retrieved):
        if policy_id in expected_set:
            mrr = 1.0 / (i + 1)
            break
    
    return precision, recall, mrr


def calculate_reasoning_overlap(expected: str, actual: str) -> float:
    """
    Calculate Jaccard similarity between expected and actual reasoning.
    
    This is a simple proxy for semantic similarity.
    """
    if not expected or not actual:
        return 0.0
    
    expected_words = set(expected.lower().split())
    actual_words = set(actual.lower().split())
    
    intersection = expected_words & actual_words
    union = expected_words | actual_words
    
    return len(intersection) / len(union) if union else 0.0


# =============================================================================
# TEST HARNESS
# =============================================================================

class TestHarness:
    """
    Runs evaluation tests against RuleMirror components.
    """
    
    def __init__(self):
        self.test_cases = (
            VIOLATION_TEST_CASES + 
            COMPLIANT_TEST_CASES + 
            NEEDS_REVIEW_TEST_CASES
        )
        self.policy_rules = POLICY_RULES
        self.results = []
        
    def get_test_cases_by_category(self, category: str) -> list:
        """Filter test cases by category."""
        return [tc for tc in self.test_cases if tc.clause.category == category]
    
    def get_test_cases_by_difficulty(self, difficulty: str) -> list:
        """Filter test cases by difficulty."""
        return [tc for tc in self.test_cases if tc.difficulty == difficulty]
    
    def get_test_cases_by_status(self, status: ComplianceStatus) -> list:
        """Filter test cases by expected status."""
        return [tc for tc in self.test_cases if tc.expected_status == status]
    
    def export_to_json(self, filepath: str):
        """Export test cases to JSON for external use."""
        data = {
            "policies": [p.to_dict() for p in self.policy_rules],
            "test_cases": [tc.to_dict() for tc in self.test_cases],
            "metadata": {
                "total_policies": len(self.policy_rules),
                "total_test_cases": len(self.test_cases),
                "violations": len(VIOLATION_TEST_CASES),
                "compliant": len(COMPLIANT_TEST_CASES),
                "needs_review": len(NEEDS_REVIEW_TEST_CASES)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Exported to {filepath}")
    
    def print_summary(self):
        """Print test case summary."""
        print("\n" + "=" * 60)
        print("📊 TEST CASE SUMMARY")
        print("=" * 60)
        
        print(f"\n📚 Policy Rules: {len(self.policy_rules)}")
        sections = {}
        for p in self.policy_rules:
            sections[p.section] = sections.get(p.section, 0) + 1
        for section, count in sections.items():
            print(f"   • {section}: {count} rules")
        
        print(f"\n🧪 Test Cases: {len(self.test_cases)}")
        print(f"   • Violations: {len(VIOLATION_TEST_CASES)}")
        print(f"   • Compliant: {len(COMPLIANT_TEST_CASES)}")
        print(f"   • Needs Review: {len(NEEDS_REVIEW_TEST_CASES)}")
        
        print("\n📁 By Category:")
        categories = {}
        for tc in self.test_cases:
            cat = tc.clause.category
            categories[cat] = categories.get(cat, 0) + 1
        for cat, count in categories.items():
            print(f"   • {cat}: {count}")
        
        print("\n⚡ By Difficulty:")
        difficulties = {"easy": 0, "medium": 0, "hard": 0}
        for tc in self.test_cases:
            difficulties[tc.difficulty] += 1
        for diff, count in difficulties.items():
            print(f"   • {diff}: {count}")


# =============================================================================
# DEMO
# =============================================================================

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     RULEMIRROR LEARNING: Build Evaluation Set                ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Create test harness
    harness = TestHarness()
    
    # Print summary
    harness.print_summary()
    
    # Show sample test cases
    print("\n" + "=" * 60)
    print("📝 SAMPLE TEST CASES")
    print("=" * 60)
    
    print("\n--- VIOLATION Example ---")
    tc = VIOLATION_TEST_CASES[0]
    print(f"ID: {tc.id}")
    print(f"Clause: \"{tc.clause.text}\"")
    print(f"Expected: {tc.expected_status.value}")
    print(f"Should match policies: {tc.expected_policy_ids}")
    print(f"Reasoning: {tc.expected_reasoning}")
    print(f"Fix: {tc.suggested_fix}")
    
    print("\n--- COMPLIANT Example ---")
    tc = COMPLIANT_TEST_CASES[0]
    print(f"ID: {tc.id}")
    print(f"Clause: \"{tc.clause.text}\"")
    print(f"Expected: {tc.expected_status.value}")
    print(f"Should match policies: {tc.expected_policy_ids}")
    
    # Export to JSON
    print("\n" + "=" * 60)
    print("💾 EXPORTING DATA")
    print("=" * 60)
    
    output_dir = Path(__file__).parent.parent / "tests" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    harness.export_to_json(str(output_dir / "evaluation_set.json"))
    
    print("\n" + "=" * 60)
    print("✅ SUMMARY")
    print("=" * 60)
    print("""
    Created evaluation set with:
    
    1. 16 Policy Rules covering:
       • Payment Terms (3)
       • Termination (3)
       • Intellectual Property (3)
       • Liability (3)
       • Data Protection (4)
    
    2. 18 Test Cases:
       • 10 Violations (known policy breaches)
       • 5 Compliant (passing clauses)
       • 3 Needs Review (ambiguous edge cases)
    
    3. Metrics Framework:
       • Retrieval: Precision, Recall, MRR
       • Generation: Status accuracy, Reasoning overlap
    
    Use this to evaluate your RAG pipeline! 🎯
    """)


if __name__ == "__main__":
    main()
