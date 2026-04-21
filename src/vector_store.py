"""
vector_store.py — ChromaDB and FAISS vector store operations.

WHY THIS MODULE EXISTS:
After we convert text chunks into vectors (embeddings), we need a
"vector database" to store them and efficiently search for the most
similar vectors when a user asks a question. This is the core of
retrieval in RAG.

WHY TWO VECTOR STORES?
  1. ChromaDB (primary): Persistent on disk, survives restarts, has
     built-in metadata filtering. Perfect for production.
  2. FAISS (alternative): Facebook's ultra-fast similarity search.
     In-memory by default, great for benchmarking speed differences.

Having both lets you compare retrieval performance — an important
skill to demonstrate in AI research interviews.
"""

from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma, FAISS

from config import Config


def create_chroma_store(
    documents: List[Document],
    embedding_model,
    persist_directory: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> Chroma:
    """
    Create a ChromaDB vector store from documents.

    WHY PERSIST TO DISK?
    Embedding 500 papers can take minutes. By persisting to
    ./data/chroma_db/, we only need to do it once. Future app
    restarts load the existing store instantly.
    """
    if persist_directory is None:
        persist_directory = Config.CHROMA_PERSIST_DIR
    if collection_name is None:
        collection_name = Config.COLLECTION_NAME

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    print(f"✓ Created ChromaDB store with {len(documents)} chunks "
          f"in '{persist_directory}'")
    return vector_store


def load_chroma_store(
    embedding_model,
    persist_directory: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> Chroma:
    """
    Load an existing ChromaDB store from disk.

    WHY A SEPARATE LOAD FUNCTION?
    The first time you run the app, you create the store (slow).
    Every time after, you just load it (instant). The UI checks
    if the store exists and calls the right function.
    """
    if persist_directory is None:
        persist_directory = Config.CHROMA_PERSIST_DIR
    if collection_name is None:
        collection_name = Config.COLLECTION_NAME

    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_name=collection_name,
    )

    count = vector_store._collection.count()
    print(f"✓ Loaded ChromaDB store with {count} chunks")
    return vector_store


def get_existing_sources(vector_store: Chroma) -> set:
    """
    Get the set of paper filenames already in the vector store.

    WHY?
    Before ingesting, we check if a paper is already stored.
    This prevents duplicate chunks that waste storage and
    pollute search results with repeated content.
    """
    try:
        all_meta = vector_store._collection.get()["metadatas"]
        return set(m.get("source", "") for m in all_meta)
    except Exception:
        return set()


def add_to_chroma_store(
    documents: List[Document],
    vector_store: Chroma,
) -> Chroma:
    """
    Add new documents to an existing ChromaDB store.

    WHY APPEND INSTEAD OF RECREATE?
    Recreating the store from scratch every time is wasteful —
    you'd re-embed papers you already have. By appending, we
    only embed the new chunks, saving time and API calls.
    """
    vector_store.add_documents(documents)
    count = vector_store._collection.count()
    print(f"✓ Added {len(documents)} chunks (total: {count})")
    return vector_store


def create_faiss_store(
    documents: List[Document],
    embedding_model,
) -> FAISS:
    """
    Create an in-memory FAISS vector store.

    WHY FAISS?
    It's the industry standard for fast approximate nearest-neighbor
    search. Including it shows you understand the trade-offs:
      - FAISS: Faster search, but no persistence by default
      - ChromaDB: Slightly slower, but persistent and has metadata filtering

    For a research interview, being able to compare these is valuable.
    """
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embedding_model,
    )

    print(f"✓ Created FAISS store with {len(documents)} chunks (in-memory)")
    return vector_store
