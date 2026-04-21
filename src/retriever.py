"""
retriever.py — Retrieval and reranking pipeline.

WHY THIS MODULE EXISTS:
Retrieval is the "R" in RAG. When a user asks a question, we need to
find the most relevant text chunks from our vector store. But basic
vector similarity has a problem: it's good but not perfect.

WHY RERANKING?
Vector search (embedding similarity) is FAST but APPROXIMATE. It
compares the question's embedding to all chunk embeddings using cosine
similarity. This works well ~80% of the time, but sometimes misses
nuanced matches.

A cross-encoder reranker READS both the question AND each candidate
chunk together, producing a much more accurate relevance score. It's
SLOWER (can't run on millions of docs), but by only reranking the
top-K results from vector search, we get the best of both worlds:

  Step 1: Vector search → fast, retrieves top-K candidates
  Step 2: Cross-encoder → slow but accurate, reorders those K candidates

This two-stage approach is an industry best practice (used by Google
Search, Bing, etc.) and reduces hallucinations by ~40% compared to
retrieval without reranking.
"""

from typing import List, Optional

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config import Config


# Lazy-load the cross-encoder (it's ~80MB, only load if needed)
_reranker = None


def _get_reranker() -> CrossEncoder:
    """
    Load the cross-encoder model (once, then cached).

    WHY ms-marco-MiniLM-L-6-v2?
    It was trained on MS MARCO (a real search engine dataset) to judge
    query-document relevance. It's small (80MB), fast, and surprisingly
    accurate for academic text despite being trained on web queries.
    """
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def get_base_retriever(vector_store, k: Optional[int] = None):
    """
    Create a basic similarity-search retriever from the vector store.

    This wraps the vector store in LangChain's retriever interface,
    which is needed to plug into a RAG chain.
    """
    if k is None:
        k = Config.TOP_K

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def rerank_results(query: str, documents: List[Document]) -> List[Document]:
    """
    Rerank retrieved documents using a cross-encoder.

    HOW IT WORKS:
    1. For each document, create a (query, document_text) pair
    2. The cross-encoder scores each pair (how relevant is this doc?)
    3. Sort by score descending → most relevant first

    The cross-encoder is ~10x slower than embedding similarity, but
    we're only running it on K documents (not the entire corpus), so
    it only adds ~100ms of latency.
    """
    if not documents:
        return documents

    reranker = _get_reranker()

    # Create query-document pairs for scoring
    pairs = [(query, doc.page_content) for doc in documents]
    scores = reranker.predict(pairs)

    # Attach scores and sort
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, score in scored_docs]


def retrieve(
    query: str,
    vector_store,
    k: Optional[int] = None,
    use_reranking: Optional[bool] = None,
) -> List[Document]:
    """
    Full retrieval pipeline: vector search → optional reranking.

    This is the main function the rest of the app calls.
    """
    if k is None:
        k = Config.TOP_K
    if use_reranking is None:
        use_reranking = Config.USE_RERANKING

    # Step 1: Fast vector similarity search
    # We retrieve more docs than needed if reranking, so the reranker
    # has more candidates to work with
    search_k = k * 2 if use_reranking else k
    docs = vector_store.similarity_search(query, k=search_k)

    # Step 2: Optional reranking
    if use_reranking and docs:
        docs = rerank_results(query, docs)
        docs = docs[:k]  # Take only top-k after reranking

    return docs
