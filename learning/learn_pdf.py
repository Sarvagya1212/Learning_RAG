"""
Learning 03: PDF Processing Basics
===================================
This file teaches you how to extract text from PDF documents.

Run this file: python learn_pdf.py

Prerequisites:
    pip install pypdf

What you'll learn:
    1. Why PDF processing matters for RuleMirror
    2. How to extract text from PDFs
    3. Chunking strategies for long documents
    4. Handling real-world PDF challenges
"""

import os
from pathlib import Path

# We use pypdf for PDF parsing (lightweight and reliable)
try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False
    print("⚠️  Run: pip install pypdf")
    print("   Then run this file again!\n")


def explain_pdf_processing():
    """
    LESSON 1: Why PDF Processing Matters
    =====================================
    """
    print("=" * 60)
    print("LESSON 1: Why PDF Processing Matters")
    print("=" * 60)
    print("""
    RuleMirror needs to read your company policies (PDFs).
    
    The problem?
    ┌─────────────────────────────────────────────┐
    │  PDFs are NOT plain text!                   │
    │                                             │
    │  They're complex documents with:            │
    │  • Text positioned at specific coordinates  │
    │  • Images, tables, headers, footers         │
    │  • Multiple columns, different fonts        │
    │  • Sometimes scanned images (need OCR!)     │
    └─────────────────────────────────────────────┘
    
    Our job: Extract clean text while preserving meaning.
    
    Pipeline:
        PDF → Extract Text → Clean → Chunk → Embed → Search
              ^^^^^^^^^^^
              We're here!
    """)


def create_sample_policy():
    """
    Create a sample PDF for testing.
    Returns the path to the sample file.
    """
    # We'll create a text file to simulate a policy
    # (Creating actual PDFs requires additional dependencies)
    
    sample_content = """
COMPANY POLICY DOCUMENT
=======================
Policy ID: POL-2024-001
Effective Date: January 1, 2024
Department: Legal & Compliance

SECTION 1: PAYMENT TERMS
------------------------
1.1 All vendor payments must be processed within thirty (30) days of invoice receipt.
1.2 Early payment discounts of 2% are authorized for payments within 10 days.
1.3 Late payment penalties shall not exceed 1.5% per month.

SECTION 2: CONTRACT TERMINATION
-------------------------------
2.1 All contracts must include a termination clause.
2.2 Minimum notice period for termination is thirty (30) days.
2.3 Immediate termination is permitted in cases of material breach.

SECTION 3: INTELLECTUAL PROPERTY
--------------------------------
3.1 All intellectual property created during engagement remains company property.
3.2 Contractors must sign IP assignment agreements before project start.
3.3 Third-party IP must be properly licensed.

SECTION 4: DATA PROTECTION
--------------------------
4.1 All vendors handling personal data must comply with GDPR/CCPA.
4.2 Data processing agreements (DPAs) are mandatory.
4.3 Data breach notification must occur within 72 hours.

SECTION 5: LIABILITY
--------------------
5.1 Vendor liability shall not exceed the total contract value.
5.2 Vendors must maintain professional liability insurance.
5.3 Indemnification clauses must be mutual and reasonable.
"""
    
    # Save to data/policies folder
    policy_dir = Path(__file__).parent.parent / "data" / "policies"
    policy_dir.mkdir(parents=True, exist_ok=True)
    
    sample_path = policy_dir / "sample_policy.txt"
    sample_path.write_text(sample_content)
    
    return sample_path, sample_content


def demo_text_extraction():
    """
    LESSON 2: Extracting Text
    =========================
    """
    print("\n" + "=" * 60)
    print("LESSON 2: Extracting Text")
    print("=" * 60)
    
    # Create sample policy
    sample_path, content = create_sample_policy()
    
    print(f"\n📄 Created sample policy at:")
    print(f"   {sample_path}")
    
    print("\n📖 Extracted text (first 500 chars):")
    print("-" * 40)
    print(content[:500] + "...")
    print("-" * 40)
    
    print("""
    💡 For real PDFs, we use:
    
    ```python
    from pypdf import PdfReader
    
    reader = PdfReader("policy.pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    ```
    """)
    
    return content


def demo_chunking(text):
    """
    LESSON 3: Chunking Strategies
    =============================
    Why: Long documents won't fit in LLM context windows.
         Also, embeddings work better on smaller chunks.
    """
    print("\n" + "=" * 60)
    print("LESSON 3: Chunking Strategies")
    print("=" * 60)
    
    print("""
    Problem: Policy documents can be hundreds of pages!
    
    Solutions:
    ┌────────────────────────────────────────────────────────┐
    │ Strategy 1: Fixed-size chunks                          │
    │   Split every N characters (simple but may break mid-  │
    │   sentence)                                            │
    ├────────────────────────────────────────────────────────┤
    │ Strategy 2: Sentence-based chunks                      │
    │   Keep sentences together (better meaning)             │
    ├────────────────────────────────────────────────────────┤
    │ Strategy 3: Section-based chunks ✓ (Best for policies) │
    │   Use headers/sections as natural boundaries           │
    └────────────────────────────────────────────────────────┘
    """)
    
    # Demo: Section-based chunking
    print("🔪 Demo: Section-based Chunking")
    print("-" * 40)
    
    sections = []
    current_section = []
    
    for line in text.split('\n'):
        # Detect section headers (lines with all caps or ending with ---)
        if line.strip().startswith('SECTION') or line.strip().endswith('---'):
            if current_section:
                sections.append('\n'.join(current_section))
            current_section = [line]
        else:
            current_section.append(line)
    
    if current_section:
        sections.append('\n'.join(current_section))
    
    print(f"\n📊 Split into {len(sections)} chunks:")
    for i, section in enumerate(sections):
        # Get first line as preview
        preview = section.strip().split('\n')[0][:50]
        char_count = len(section)
        print(f"   Chunk {i}: {char_count:4d} chars - {preview}...")
    
    return sections


def demo_overlap_strategy():
    """
    LESSON 4: Chunk Overlap
    =======================
    Why: Context at chunk boundaries can be lost.
    """
    print("\n" + "=" * 60)
    print("LESSON 4: Chunk Overlap (Advanced)")
    print("=" * 60)
    
    print("""
    Problem: What if important context spans two chunks?
    
    "...payment within 30 days | of invoice receipt..."
                          ↑ chunk boundary = lost context!
    
    Solution: Overlapping chunks
    
    ┌─────────────────────────────────────────┐
    │ Chunk 1: "...payment within 30 days of" │
    └────────────────────────────────┬────────┘
                              overlap│
    ┌────────────────────────────────┴────────┐
    │ Chunk 2: "30 days of invoice receipt..."│
    └─────────────────────────────────────────┘
    
    Common settings:
    • Chunk size: 500-1000 characters
    • Overlap: 50-100 characters (10-20%)
    """)
    
    # Demo implementation
    def chunk_with_overlap(text, chunk_size=200, overlap=50):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap  # Go back by overlap amount
        return chunks
    
    sample = "The payment must be made within thirty days of invoice receipt. Late payments incur a 1.5% monthly fee."
    
    print("\n📝 Example text:")
    print(f"   '{sample}'")
    print(f"\n🔪 Chunks (size=50, overlap=15):")
    
    chunks = chunk_with_overlap(sample, chunk_size=50, overlap=15)
    for i, chunk in enumerate(chunks):
        print(f"   [{i}] '{chunk}'")


def demo_real_pdf():
    """
    LESSON 5: Real PDF Processing
    =============================
    """
    print("\n" + "=" * 60)
    print("LESSON 5: Real PDF Processing")
    print("=" * 60)
    
    if not HAS_PYPDF:
        print("\n⚠️  pypdf not installed - showing code example only")
    
    print("""
    Here's how to process a real PDF:
    
    ```python
    from pypdf import PdfReader
    
    def extract_pdf_text(pdf_path):
        '''Extract text from a PDF file.'''
        reader = PdfReader(pdf_path)
        
        text_parts = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                text_parts.append(f"--- Page {page_num + 1} ---")
                text_parts.append(text)
        
        return "\\n".join(text_parts)
    
    # Usage
    policy_text = extract_pdf_text("data/policies/company_policy.pdf")
    ```
    
    For RuleMirror, we'll store this in:
        src/rule_mirror/parsers/pdf_parser.py
    """)
    
    # Check if any PDFs exist in policies folder
    policy_dir = Path(__file__).parent.parent / "data" / "policies"
    pdf_files = list(policy_dir.glob("*.pdf"))
    
    if pdf_files:
        print(f"\n📂 Found {len(pdf_files)} PDF(s) in data/policies/:")
        for pdf in pdf_files:
            print(f"   • {pdf.name}")
        
        if HAS_PYPDF:
            print("\n🔄 Processing first PDF...")
            reader = PdfReader(pdf_files[0])
            print(f"   Pages: {len(reader.pages)}")
            first_page = reader.pages[0].extract_text()[:300]
            print(f"   Preview: {first_page}...")
    else:
        print("\n📂 No PDFs found in data/policies/")
        print("   Add your policy PDFs there to test!")


def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     RULEMIRROR LEARNING: PDF Processing Basics           ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Lesson 1: Why it matters
    explain_pdf_processing()
    
    # Lesson 2: Extract text
    text = demo_text_extraction()
    
    # Lesson 3: Chunking
    demo_chunking(text)
    
    # Lesson 4: Overlap
    demo_overlap_strategy()
    
    # Lesson 5: Real PDFs
    demo_real_pdf()
    
    print("\n" + "=" * 60)
    print("✅ SUMMARY: What You Learned")
    print("=" * 60)
    print("""
    1. PDFs need special parsing (not plain text)
    2. Use pypdf or unstructured for extraction
    3. Chunk documents for embedding (section-based is best)
    4. Overlap chunks to preserve context at boundaries
    
    Files created:
    • data/policies/sample_policy.txt (sample for testing)
    
    Next: We'll set up Qdrant to store document chunks!
    """)


if __name__ == "__main__":
    main()
