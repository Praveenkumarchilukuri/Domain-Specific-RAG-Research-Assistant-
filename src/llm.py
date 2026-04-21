"""
llm.py — LLM query interface with automatic model fallback.

WHY MODEL FALLBACK?
Google's free tier has per-model daily quotas. Once you exhaust
gemini-2.5-pro's quota, the app would crash. Instead, we automatically
try the next model in the list. Since each model has SEPARATE quota,
this gives you 4x the free-tier capacity:
  gemini-2.5-pro → gemini-2.5-flash → gemini-2.0-flash → gemini-2.0-flash-lite

This is a common production pattern called "model cascading".
"""

import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from typing import List, Dict, Optional

from config import Config


# Fallback chain — each model has separate free-tier quota
FALLBACK_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]


# The RAG prompt
RAG_PROMPT_TEMPLATE = """You are a helpful research assistant that answers questions 
based on academic papers. Use ONLY the context provided below to answer the question. 
If the context doesn't contain enough information to answer, say "I don't have enough 
information in the provided papers to answer this question."

Always cite your sources by mentioning the paper name and page number when available.

Context from research papers:
{context}

Question: {question}

Answer (with citations):"""


RAG_PROMPT = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)


def get_llm(model_name: Optional[str] = None) -> ChatGoogleGenerativeAI:
    """Create a Google Gemini LLM instance."""
    if model_name is None:
        model_name = Config.LLM_MODEL

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=Config.GOOGLE_API_KEY,
        temperature=0.3,
    )


def _try_invoke_with_fallback(chain_builder, inputs, models=None):
    """
    Try invoking a chain with automatic model fallback.

    WHY THIS PATTERN?
    Each Gemini model has its own rate limit quota. If gemini-2.5-pro
    returns 429 (rate limited), we immediately try gemini-2.5-flash,
    then gemini-2.0-flash, then gemini-2.0-flash-lite. This gives
    you 4x the effective free-tier capacity.

    Also retries within each model with a short wait for per-minute
    limits (which reset quickly).
    """
    if models is None:
        models = FALLBACK_MODELS

    # Make sure the configured model is tried first
    if Config.LLM_MODEL not in models:
        models = [Config.LLM_MODEL] + models

    last_error: BaseException = RuntimeError("All fallback models failed")
    used_model = None

    for model_name in models:
        llm = get_llm(model_name)
        chain = chain_builder(llm)

        # Try up to 2 times per model (handles per-minute limits)
        for attempt in range(2):
            try:
                response = chain.invoke(inputs)
                return response, model_name
            except Exception as e:
                error_str = str(e)
                last_error = e
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    if attempt == 0:
                        # Quick retry for per-minute limits
                        time.sleep(5)
                    else:
                        # This model's quota is exhausted, try next model
                        print(f"⚠️ {model_name} rate limited, trying next model...")
                        break
                else:
                    raise  # Non-rate-limit error, don't fallback

    # All models exhausted
    raise last_error


def query_with_context(
    question: str,
    context_docs: List[Document],
    llm=None,
) -> Dict:
    """
    Query the LLM with pre-retrieved context documents.
    Automatically falls back through multiple models if rate-limited.
    """
    # Format context from documents
    context = "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'Unknown')}, "
        f"Page: {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
        for doc in context_docs
    )

    inputs = {"context": context, "question": question}

    if llm is not None:
        # Direct call without fallback if LLM is provided explicitly
        chain = RAG_PROMPT | llm
        response = chain.invoke(inputs)
        used_model = Config.LLM_MODEL
    else:
        # Use fallback chain
        response, used_model = _try_invoke_with_fallback(
            lambda llm_instance: RAG_PROMPT | llm_instance,
            inputs,
        )

    return {
        "answer": response.content,
        "source_documents": context_docs,
        "model": used_model,
    }
