"""
Learning 06: Process Real PDFs with RAG
========================================
This file teaches you how to process real PDF policy documents and
use them in a RAG pipeline for contract compliance checking.

Run this file: python pdf_rag.py

Prerequisites:
    pip install pypdf qdrant-client sentence-transformers mistralai python-dotenv

What you'll learn:
    1. Loading and parsing PDF documents
    2. Smart chunking strategies for policies
    3. Building a searchable knowledge base from PDFs
    4. End-to-end RAG with real documents
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check dependencies
try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False
    print("⚠️  Run: pip install pypdf")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False
    print("⚠️  Run: pip install qdrant-client")

try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False
    print("⚠️  Run: pip install sentence-transformers")

try:
    from mistralai import Mistral
    HAS_MISTRAL = True
except ImportError:
    HAS_MISTRAL = False
    print("⚠️  Run: pip install mistralai")


# =============================================================================
# PDF PROCESSING UTILITIES
# =============================================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    if not HAS_PYPDF:
        raise ImportError("pypdf is required. Run: pip install pypdf")
    
    reader = PdfReader(pdf_path)
    text_parts = []
    
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            text_parts.append(f"[Page {page_num + 1}]\n{text}")
    
    return "\n\n".join(text_parts)


def smart_chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Split text into overlapping chunks, trying to respect sentence boundaries.
    
    Args:
        text: The text to chunk
        chunk_size: Target size for each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    # Split into sentences (simple approach)
    sentences = []
    current = ""
    
    for char in text:
        current += char
        if char in '.!?' and len(current) > 20:
            sentences.append(current.strip())
            current = ""
    
    if current.strip():
        sentences.append(current.strip())
    
    # Build chunks from sentences
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Add overlap between chunks
    overlapped_chunks = []
    for i, chunk in enumerate(chunks):
        if i > 0 and len(chunks[i-1]) > overlap:
            # Add end of previous chunk to start
            prefix = chunks[i-1][-overlap:]
            chunk = prefix + " " + chunk
        overlapped_chunks.append(chunk)
    
    return overlapped_chunks


def extract_sections_from_text(text: str) -> list:
    """
    Extract sections from policy document based on headers.
    
    Returns:
        List of dicts with 'section' and 'content' keys
    """
    sections = []
    current_section = "General"
    current_content = []
    
    for line in text.split('\n'):
        line = line.strip()
        
        # Detect section headers (various patterns)
        if (line.isupper() and len(line) > 5 and len(line) < 100) or \
           (line.startswith('SECTION') or line.startswith('Article') or 
            line.startswith('Chapter') or line.endswith(':')):
            
            # Save previous section
            if current_content:
                sections.append({
                    'section': current_section,
                    'content': '\n'.join(current_content)
                })
            
            current_section = line.strip(':').strip()
            current_content = []
        else:
            if line:
                current_content.append(line)
    
    # Don't forget the last section
    if current_content:
        sections.append({
            'section': current_section,
            'content': '\n'.join(current_content)
        })
    
    return sections


# =============================================================================
# SAMPLE POLICY DOCUMENT CREATION
# =============================================================================

def create_sample_policy_pdf():
    """
    Create a sample policy document (as text file for demo).
    In real use, you'd have actual PDF files.
    """
    policy_dir = Path(__file__).parent.parent / "data" / "policies"
    policy_dir.mkdir(parents=True, exist_ok=True)
    
    sample_content = """
CORPORATE PROCUREMENT POLICY
============================
Document ID: PROC-POL-2024
Effective Date: January 1, 2024
Last Updated: December 15, 2023
Department: Legal & Procurement

SECTION 1: PAYMENT TERMS
------------------------
1.1 Standard Payment Terms
All vendor payments must be processed within thirty (30) days of invoice 
receipt. This is a mandatory requirement with no exceptions permitted.

1.2 Early Payment Discounts
Early payment discounts of up to 2% are authorized for payments made 
within ten (10) days of invoice receipt. Department heads may approve 
early payments when discount benefits exceed administrative costs.

1.3 Late Payment Penalties
The company shall not agree to late payment penalties exceeding 1.5% 
per month. Any contract proposing higher penalties must be escalated 
to the Legal department for review.

SECTION 2: CONTRACT TERMINATION
-------------------------------
2.1 Notice Requirements
All contracts must include a termination clause requiring a minimum of 
thirty (30) days written notice from either party.

2.2 Material Breach
Immediate termination without notice is permitted only in cases of:
- Material breach of contract terms
- Fraud or misrepresentation
- Violation of applicable laws
- Failure to maintain required insurance

2.3 Termination for Convenience
The company reserves the right to terminate for convenience with sixty 
(60) days notice and payment for work completed to date.

SECTION 3: INTELLECTUAL PROPERTY
--------------------------------
3.1 Work Product Ownership
All intellectual property created during the engagement shall remain 
the exclusive property of the Company. This includes but is not limited 
to: source code, designs, documentation, and methodologies.

3.2 IP Assignment
Vendors must execute an IP assignment agreement prior to project 
commencement. The assignment must cover all work product created 
during the engagement.

3.3 Pre-existing IP
Vendors retain ownership of pre-existing intellectual property. Any 
use of pre-existing IP must be disclosed and properly licensed to 
the Company.

SECTION 4: LIABILITY AND INSURANCE
----------------------------------
4.1 Liability Cap
Total vendor liability shall not exceed the total value of the 
contract unless gross negligence or willful misconduct is involved.

4.2 Insurance Requirements
All vendors must maintain:
- Professional liability insurance: minimum $1,000,000
- General liability insurance: minimum $2,000,000
- Cyber liability insurance: minimum $1,000,000 (for IT vendors)

4.3 Indemnification
Vendors must indemnify the Company against claims arising from:
- Vendor negligence or misconduct
- IP infringement claims
- Data breaches caused by vendor

SECTION 5: DATA PROTECTION
--------------------------
5.1 Data Handling Requirements
All vendors handling personal data must:
- Comply with GDPR, CCPA, and applicable data protection laws
- Execute a Data Processing Agreement (DPA)
- Implement appropriate security measures
- Report data breaches within 24 hours

5.2 Data Retention
Vendors must delete all company data within 30 days of contract 
termination unless legal retention requirements apply.

5.3 Subprocessors
Use of subprocessors for data processing requires prior written 
approval from the Company's Data Protection Officer.

END OF POLICY
"""
    
    sample_path = policy_dir / "corporate_procurement_policy.txt"
    sample_path.write_text(sample_content)
    print(f"📄 Created sample policy: {sample_path}")
    
    return sample_path


# =============================================================================
# RAG PIPELINE
# =============================================================================

class PolicyRAG:
    """
    A complete RAG pipeline for policy document analysis.
    """
    
    def __init__(self):
        self.client = None
        self.model = None
        self.llm_client = None
        self.collection_name = "policies"
        
    def initialize(self):
        """Initialize all components."""
        if not HAS_QDRANT or not HAS_EMBEDDINGS:
            raise ImportError("Missing required dependencies")
        
        print("📦 Initializing RAG pipeline...")
        self.client = QdrantClient(":memory:")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize LLM if API key present
        api_key = os.getenv("MISTRAL_API_KEY")
        if api_key and api_key != "your_mistral_key_here" and HAS_MISTRAL:
            self.llm_client = Mistral(api_key=api_key)
            print("   ✅ LLM client initialized")
        else:
            print("   ⚠️  LLM client not available (no API key)")
        
        # Create collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print("   ✅ Vector store ready")
        
    def ingest_document(self, text: str, source: str = "unknown"):
        """
        Process and ingest a document into the vector store.
        
        Args:
            text: Document text
            source: Source filename
        """
        print(f"\n📥 Ingesting document: {source}")
        
        # Extract sections
        sections = extract_sections_from_text(text)
        print(f"   Found {len(sections)} sections")
        
        # Chunk each section
        all_chunks = []
        for section in sections:
            chunks = smart_chunk_text(section['content'], chunk_size=400)
            for chunk in chunks:
                all_chunks.append({
                    'text': chunk,
                    'section': section['section'],
                    'source': source
                })
        
        print(f"   Created {len(all_chunks)} chunks")
        
        # Create embeddings
        texts = [c['text'] for c in all_chunks]
        embeddings = self.model.encode(texts)
        
        # Store in vector DB
        points = [
            PointStruct(
                id=i,
                vector=embeddings[i].tolist(),
                payload={
                    'text': c['text'],
                    'section': c['section'],
                    'source': c['source']
                }
            )
            for i, c in enumerate(all_chunks)
        ]
        
        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"   ✅ Indexed {len(points)} chunks")
        
        return len(all_chunks)
    
    def ingest_pdf(self, pdf_path: str):
        """Ingest a PDF file."""
        text = extract_text_from_pdf(pdf_path)
        source = Path(pdf_path).name
        return self.ingest_document(text, source)
    
    def search(self, query: str, limit: int = 3) -> list:
        """
        Search for relevant policy chunks.
        
        Args:
            query: Search query
            limit: Number of results
            
        Returns:
            List of relevant chunks with metadata
        """
        query_embedding = self.model.encode([query])[0]
        
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=limit,
            with_payload=True
        )
        
        return [
            {
                'text': r.payload['text'],
                'section': r.payload['section'],
                'source': r.payload['source'],
                'score': r.score
            }
            for r in results.points
        ]
    
    def analyze_clause(self, clause: str, top_k: int = 3) -> dict:
        """
        Analyze a contract clause for compliance.
        
        Args:
            clause: The contract clause to analyze
            top_k: Number of policies to retrieve
            
        Returns:
            Analysis result dict
        """
        print(f"\n🔍 Analyzing clause...")
        print(f"   \"{clause[:60]}...\"")
        
        # Retrieve relevant policies
        policies = self.search(clause, limit=top_k)
        
        print(f"\n📚 Retrieved {len(policies)} relevant policies:")
        for p in policies:
            print(f"   [{p['section']}] (score: {p['score']:.3f})")
            print(f"   {p['text'][:60]}...")
        
        # Build prompt
        policy_context = "\n".join([
            f"[{p['section']}] {p['text']}" 
            for p in policies
        ])
        
        prompt = f"""You are a legal compliance expert reviewing contracts against company policies.

## Relevant Company Policies:
{policy_context}

## Contract Clause to Review:
"{clause}"

## Your Analysis:
1. Does this clause COMPLY with or VIOLATE the policies?
2. If violation, cite the specific policy section.
3. Provide a RECOMMENDATION for fixing non-compliant clauses.

Format your response as:
STATUS: [COMPLIANT / VIOLATION / NEEDS REVIEW]
ANALYSIS: [Your analysis]
RECOMMENDATION: [Suggested fix if needed]
"""
        
        # Generate response
        if self.llm_client:
            print("\n🤖 Calling LLM for analysis...")
            try:
                response = self.llm_client.chat.complete(
                    model="mistral-small-latest",
                    messages=[{"role": "user", "content": prompt}]
                )
                analysis = response.choices[0].message.content
            except Exception as e:
                analysis = f"Error calling LLM: {e}"
        else:
            # Simulated analysis
            analysis = self._simulate_analysis(clause, policies)
        
        print("\n📋 Analysis Result:")
        print("-" * 50)
        print(analysis)
        print("-" * 50)
        
        return {
            'clause': clause,
            'policies': policies,
            'analysis': analysis
        }
    
    def _simulate_analysis(self, clause: str, policies: list) -> str:
        """Simulate LLM analysis when no API key is available."""
        # Simple keyword-based simulation for demo
        clause_lower = clause.lower()
        
        if "45 days" in clause_lower or "60 days" in clause_lower:
            return """STATUS: VIOLATION

ANALYSIS: The clause specifies a payment term that exceeds the 30-day 
maximum allowed by company policy. Section 1.1 clearly states that 
all vendor payments must be processed within thirty (30) days.

RECOMMENDATION: Revise to "Payment shall be made within thirty (30) 
days of invoice receipt." """
        
        elif "7 days" in clause_lower and "terminat" in clause_lower:
            return """STATUS: VIOLATION

ANALYSIS: The termination notice period is insufficient. Section 2.1 
requires a minimum of thirty (30) days written notice for termination.

RECOMMENDATION: Revise to "Either party may terminate with thirty (30) 
days written notice." """
        
        elif "contractor" in clause_lower and ("ip" in clause_lower or "intellectual" in clause_lower):
            return """STATUS: VIOLATION

ANALYSIS: The IP ownership clause conflicts with Section 3.1 which 
states all intellectual property created during engagement belongs 
exclusively to the Company.

RECOMMENDATION: Revise to "All intellectual property created during 
this engagement shall be the exclusive property of the Company." """
        
        else:
            return """STATUS: NEEDS REVIEW

ANALYSIS: The clause should be reviewed by legal counsel to ensure 
full compliance with company policies. No obvious violations detected 
but manual review recommended.

RECOMMENDATION: Submit to Legal department for formal review."""


# =============================================================================
# MAIN DEMO
# =============================================================================

def demo_pdf_rag():
    """Run the complete PDF RAG demo."""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     RULEMIRROR LEARNING: Process Real PDFs with RAG      ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Create sample policy
    sample_path = create_sample_policy_pdf()
    
    # Initialize RAG pipeline
    rag = PolicyRAG()
    rag.initialize()
    
    # Ingest the document
    sample_text = sample_path.read_text()
    rag.ingest_document(sample_text, source=sample_path.name)
    
    # Test contract clauses
    test_clauses = [
        "Payment for services shall be made within 45 business days of invoice receipt.",
        "Either party may terminate this agreement with 7 days written notice.",
        "All software code developed under this contract shall remain the intellectual property of the Contractor.",
        "The vendor's maximum liability shall be limited to $10,000 regardless of contract value.",
    ]
    
    print("\n" + "=" * 60)
    print("CONTRACT COMPLIANCE ANALYSIS")
    print("=" * 60)
    
    for i, clause in enumerate(test_clauses, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}")
        print("=" * 60)
        rag.analyze_clause(clause)
    
    print("\n" + "=" * 60)
    print("✅ SUMMARY")
    print("=" * 60)
    print("""
    You've learned how to:
    
    1. Extract text from PDF documents
    2. Smart chunk with section awareness
    3. Build a searchable policy knowledge base
    4. Run compliance analysis with RAG
    
    This is the foundation of RuleMirror! 🎯
    
    Next: Add your own policy PDFs to data/policies/ and test!
    """)


if __name__ == "__main__":
    demo_pdf_rag()
