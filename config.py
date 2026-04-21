"""
config.py — Central configuration for the RAG Research Assistant.

WHY THIS FILE EXISTS:
Instead of scattering hardcoded values across every module, we load all
settings from a single .env file into one Config class. This means:
  1. Changing a model or chunk size is a one-line edit in .env
  2. No secrets (API keys) ever get committed to Git
  3. Every module imports from one place, so nothing falls out of sync
"""

import os
from dotenv import load_dotenv

# Load .env file from project root (if it exists)
load_dotenv()


class Config:
    """
    All configuration values, with sensible defaults.
    Values come from environment variables (set in .env file).
    """

    # --- API Keys ---
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

    # --- Embedding Settings ---
    # "gemini" uses Google's embedding-001 model (requires API key)
    # "huggingface" uses local all-MiniLM-L6-v2 (free, no key needed)
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "gemini")

    # --- LLM Settings ---
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-2.0-flash")

    # --- Chunking Settings ---
    # CHUNK_SIZE: how many characters per text chunk.
    #   512 is a sweet spot — large enough to preserve context around
    #   a sentence, but small enough for precise retrieval.
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))

    # CHUNK_OVERLAP: characters shared between adjacent chunks.
    #   Overlap prevents information at chunk boundaries from being lost.
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # --- Vector Store Settings ---
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "research_papers")
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")

    # --- Retrieval Settings ---
    TOP_K: int = int(os.getenv("TOP_K", "5"))
    USE_RERANKING: bool = os.getenv("USE_RERANKING", "true").lower() == "true"

    # --- Paths ---
    PAPERS_DIR: str = os.getenv("PAPERS_DIR", "./data/papers")
