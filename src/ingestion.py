"""
ingestion.py — PDF loading and text chunking.

WHY THIS MODULE EXISTS:
RAG systems don't feed entire documents to an LLM (they're too long and
the LLM's context window is limited). Instead, we:
  1. PARSE PDFs into raw text (PyMuPDF handles complex academic layouts
     with columns, equations, and tables better than most PDF libraries)
  2. CHUNK the text into small, overlapping pieces (~512 chars each).
     Each chunk becomes a separate entry in our vector database, so when
     a user asks a question, we retrieve only the most relevant chunks —
     not the entire 30-page paper.

WHY OVERLAPPING CHUNKS?
Without overlap, a sentence at a chunk boundary gets split in half, and
neither chunk contains the full meaning. A 50-char overlap ensures
boundary sentences appear in at least one complete chunk.

WHY WE KEEP METADATA:
Each chunk carries metadata (source filename, page number). When the LLM
answers a question, we can show "Source: paper_xyz.pdf, page 7" — this
is critical for research credibility.
"""

import os
from typing import List, Optional

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import Config


def load_pdf(file_path: str) -> List[Document]:
    """
    Extract text from a single PDF, page by page.

    Returns a list of LangChain Document objects (one per page) so that
    page-level metadata is preserved for citation purposes.
    """
    documents = []
    pdf_name = os.path.basename(file_path)

    doc = fitz.open(file_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        if text.strip():  # Skip blank pages
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_name,
                        "page": page_num + 1,  # 1-indexed for humans
                    },
                )
            )
    doc.close()
    return documents


def load_all_pdfs(directory: Optional[str] = None) -> List[Document]:
    """
    Load every PDF from the papers directory.

    WHY A BATCH FUNCTION?
    In a research assistant, you typically ingest an entire folder of
    papers at once (e.g., 20 ArXiv PDFs on transformers). This function
    handles the loop and error handling so the rest of the code doesn't
    need to worry about filesystem details.
    """
    if directory is None:
        directory = Config.PAPERS_DIR

    documents = []
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created papers directory: {directory}")
        return documents

    pdf_files = [f for f in os.listdir(directory) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in {directory}")
        return documents

    for pdf_file in pdf_files:
        file_path = os.path.join(directory, pdf_file)
        try:
            docs = load_pdf(file_path)
            documents.extend(docs)
            print(f"✓ Loaded {pdf_file} ({len(docs)} pages)")
        except Exception as e:
            print(f"✗ Error loading {pdf_file}: {e}")

    print(f"\nTotal: {len(documents)} pages from {len(pdf_files)} PDFs")
    return documents


def chunk_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """
    Split documents into smaller chunks for embedding.

    WHY RecursiveCharacterTextSplitter?
    It tries to split on natural boundaries in this order:
      paragraph breaks → sentence breaks → word breaks → characters
    This produces chunks that are more semantically coherent than
    blindly cutting every N characters. Better chunks = better retrieval.
    """
    if chunk_size is None:
        chunk_size = Config.CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = Config.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],  # Try paragraph first, then sentence, etc.
    )

    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} pages into {len(chunks)} chunks "
          f"(size={chunk_size}, overlap={chunk_overlap})")
    return chunks
