  ╔══════════════════════════════════════════════════════════╗
    ║     RULEMIRROR LEARNING: PDF Processing Basics           ║
    ╚══════════════════════════════════════════════════════════╝

============================================================
LESSON 1: Why PDF Processing Matters
============================================================

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


============================================================
LESSON 2: Extracting Text
============================================================

📄 Created sample policy at:
   F:\projects\rule_mirror\data\policies\sample_policy.txt

📖 Extracted text (first 500 chars):
----------------------------------------

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
2.1 All contracts must i...
----------------------------------------

    💡 For real PDFs, we use:

    ```python
    from pypdf import PdfReader

    reader = PdfReader("policy.pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    ```


============================================================
LESSON 3: Chunking Strategies
============================================================

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

🔪 Demo: Section-based Chunking
----------------------------------------

📊 Split into 11 chunks:
   Chunk 0:  136 chars - COMPANY POLICY DOCUMENT...
   Chunk 1:   24 chars - SECTION 1: PAYMENT TERMS...
   Chunk 2:  249 chars - ------------------------...
   Chunk 3:   31 chars - SECTION 2: CONTRACT TERMINATION...
   Chunk 4:  216 chars - -------------------------------...
   Chunk 5:   32 chars - SECTION 3: INTELLECTUAL PROPERTY...
   Chunk 6:  234 chars - --------------------------------...
   Chunk 7:   26 chars - SECTION 4: DATA PROTECTION...
   Chunk 8:  204 chars - --------------------------...
   Chunk 9:   20 chars - SECTION 5: LIABILITY...
   Chunk 10:  204 chars - --------------------...

============================================================
LESSON 4: Chunk Overlap (Advanced)
============================================================

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


📝 Example text:
   'The payment must be made within thirty days of invoice receipt. Late payments incur a 1.5% monthly fee.'

🔪 Chunks (size=50, overlap=15):
   [0] 'The payment must be made within thirty days of inv'
   [1] 'rty days of invoice receipt. Late payments incur a'
   [2] 'ayments incur a 1.5% monthly fee.'

============================================================
LESSON 5: Real PDF Processing
============================================================

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

        return "\n".join(text_parts)

    # Usage
    policy_text = extract_pdf_text("data/policies/company_policy.pdf")
    ```

    For RuleMirror, we'll store this in:
        src/rule_mirror/parsers/pdf_parser.py


📂 No PDFs found in data/policies/
   Add your policy PDFs there to test!

============================================================
✅ SUMMARY: What You Learned
============================================================

    1. PDFs need special parsing (not plain text)
    2. Use pypdf or unstructured for extraction
    3. Chunk documents for embedding (section-based is best)
    4. Overlap chunks to preserve context at boundaries

    Files created:
    • data/policies/sample_policy.txt (sample for testing)

    Next: We'll set up Qdrant to store document chunks!