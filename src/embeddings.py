"""
embeddings.py — Embedding model factory.

WHY THIS MODULE EXISTS:
An "embedding" converts text into a vector (a list of numbers) that
captures its meaning. Similar texts get similar vectors. This is the
mathematical foundation of retrieval — we find relevant chunks by
computing which vectors are closest to the question's vector.

WHY TWO PROVIDERS?
  1. Google Gemini (models/embedding-001): High quality, fast, requires
     API key. Best for production use.
  2. HuggingFace (all-MiniLM-L6-v2): Runs 100% locally, no API key
     needed, completely free. Great for development/testing or if you
     want zero data sent to the cloud.

The factory pattern (get_embedding_model) lets the rest of the code
not care which provider is being used — it just calls the function
and gets back an embedding model.
"""

from typing import Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import Config


def get_gemini_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Google Gemini embedding model.

    WHY gemini-embedding-001?
    It's Google's current stable embedding model for retrieval tasks.
    Produces 768-dimensional vectors and handles academic text well.
    (Older models like embedding-001 and text-embedding-004 are deprecated.)
    """
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=Config.GOOGLE_API_KEY,
    )


def get_hf_embeddings() -> HuggingFaceEmbeddings:
    """
    HuggingFace local embedding model.

    WHY all-MiniLM-L6-v2?
    It's the most popular sentence-transformer: fast, lightweight
    (80MB), and produces surprisingly good 384-dimensional embeddings.
    Runs entirely on CPU with no GPU required.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


def get_embedding_model(provider: Optional[str] = None):
    """
    Factory function — returns the configured embedding model.

    WHY A FACTORY?
    So you can switch between Gemini and HuggingFace by changing one
    line in your .env file, without touching any Python code.
    """
    if provider is None:
        provider = Config.EMBEDDING_PROVIDER

    if provider.lower() == "gemini":
        return get_gemini_embeddings()
    elif provider.lower() == "huggingface":
        return get_hf_embeddings()
    else:
        raise ValueError(f"Unknown embedding provider: {provider}. Use 'gemini' or 'huggingface'.")
