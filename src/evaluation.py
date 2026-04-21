"""
evaluation.py — Answer quality evaluation metrics.

WHY THIS MODULE EXISTS:
Building a RAG system is one thing; proving it works well is another.
In AI research interviews, you'll be asked "how do you know your RAG
system gives good answers?" This module provides concrete metrics.

WHAT DO WE MEASURE?

1. FAITHFULNESS — Does the answer only use information from the retrieved
   context? An unfaithful answer contains hallucinated facts.
   Score: 0.0 (all hallucinated) to 1.0 (fully grounded)

2. ANSWER RELEVANCE — Does the answer actually address the question?
   A faithful answer that talks about the wrong topic is irrelevant.
   Score: 0.0 (off-topic) to 1.0 (directly answers the question)

3. CONTEXT RELEVANCE — Were the retrieved chunks actually relevant to
   the question? Bad retrieval leads to bad answers.
   Score: 0.0 (irrelevant chunks) to 1.0 (all chunks are relevant)

WHY NOT JUST USE RAGAS DIRECTLY?
RAGAS is great but has complex dependencies and sometimes breaks.
We implement lightweight versions of the same metrics using the LLM
itself as a judge. This is more reliable and easier to demonstrate.
"""

import time
from typing import List, Dict
import pandas as pd

from langchain_core.documents import Document

from src.llm import get_llm, FALLBACK_MODELS
from config import Config


def _invoke_with_fallback(prompt: str, llm=None) -> str:
    """
    Invoke the LLM with automatic model fallback on rate limits.
    Returns the response content string.
    """
    if llm is not None:
        response = llm.invoke(prompt)
        return response.content

    models = list(FALLBACK_MODELS)
    if Config.LLM_MODEL not in models:
        models = [Config.LLM_MODEL] + models

    last_error: BaseException = RuntimeError("All fallback models failed")
    for model_name in models:
        current_llm = get_llm(model_name)
        for attempt in range(2):
            try:
                response = current_llm.invoke(prompt)
                return response.content
            except Exception as e:
                error_str = str(e)
                last_error = e
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    if attempt == 0:
                        time.sleep(5)
                    else:
                        break
                else:
                    raise
    raise last_error


def evaluate_faithfulness(
    question: str,
    answer: str,
    context_docs: List[Document],
    llm=None,
) -> float:
    """
    Check if the answer is grounded in the provided context.

    HOW IT WORKS:
    We ask the LLM itself to judge: "Does this answer contain any
    claims NOT supported by the context?" This is called "LLM-as-judge"
    and is a widely accepted evaluation technique in the field.
    """
    if llm is None:
        llm = get_llm()

    context = "\n\n".join(doc.page_content for doc in context_docs)

    eval_prompt = f"""Evaluate the faithfulness of the following answer.
A faithful answer only contains information that can be found in or directly 
inferred from the provided context. 

Context:
{context}

Question: {question}
Answer: {answer}

On a scale of 0.0 to 1.0, how faithful is this answer to the context?
- 1.0 = Every claim in the answer is supported by the context
- 0.5 = Some claims are supported, some are not
- 0.0 = The answer contains mostly unsupported claims

Respond with ONLY a number between 0.0 and 1.0, nothing else."""

    content = _invoke_with_fallback(eval_prompt, llm)
    try:
        score = float(content.strip())
        return min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
    except ValueError:
        return 0.5  # Default if LLM doesn't return a clean number


def evaluate_relevance(
    question: str,
    answer: str,
    llm=None,
) -> float:
    """
    Check if the answer is relevant to the question asked.

    A separate metric from faithfulness: an answer can be faithful
    (all facts from context) but irrelevant (talks about wrong topic).
    """
    if llm is None:
        llm = get_llm()

    eval_prompt = f"""Evaluate how relevant the following answer is to the question.

Question: {question}
Answer: {answer}

On a scale of 0.0 to 1.0, how relevant is this answer?
- 1.0 = Directly and completely answers the question
- 0.5 = Partially answers the question
- 0.0 = Does not address the question at all

Respond with ONLY a number between 0.0 and 1.0, nothing else."""

    content = _invoke_with_fallback(eval_prompt, llm)
    try:
        score = float(content.strip())
        return min(max(score, 0.0), 1.0)
    except ValueError:
        return 0.5


def evaluate_context_relevance(
    question: str,
    context_docs: List[Document],
    llm=None,
) -> float:
    """
    Check if the retrieved context chunks are relevant to the question.

    WHY THIS MATTERS:
    If your retrieval is bad (pulling irrelevant chunks), even the best
    LLM can't give a good answer. This metric tells you if the problem
    is in retrieval or in generation.
    """
    if llm is None:
        llm = get_llm()

    context = "\n\n---\n\n".join(
        f"Chunk {i+1}: {doc.page_content[:300]}"
        for i, doc in enumerate(context_docs)
    )

    eval_prompt = f"""Evaluate how relevant the following context chunks are to the question.

Question: {question}

Retrieved Context:
{context}

On a scale of 0.0 to 1.0, how relevant are these chunks to answering the question?
- 1.0 = All chunks are highly relevant
- 0.5 = Some chunks are relevant, some are not
- 0.0 = None of the chunks are relevant

Respond with ONLY a number between 0.0 and 1.0, nothing else."""

    content = _invoke_with_fallback(eval_prompt, llm)
    try:
        score = float(content.strip())
        return min(max(score, 0.0), 1.0)
    except ValueError:
        return 0.5


def full_evaluation(
    question: str,
    answer: str,
    context_docs: List[Document],
    llm=None,
) -> Dict[str, float]:
    """
    Run all three evaluation metrics and return a summary.

    Returns a dict like: {
        "faithfulness": 0.9,
        "answer_relevance": 0.85,
        "context_relevance": 0.8
    }
    """
    if llm is None:
        llm = get_llm()

    return {
        "faithfulness": evaluate_faithfulness(question, answer, context_docs, llm),
        "answer_relevance": evaluate_relevance(question, answer, llm),
        "context_relevance": evaluate_context_relevance(question, context_docs, llm),
    }


def compare_models_evaluation(
    question: str,
    results: List[Dict],
) -> pd.DataFrame:
    """
    Compare evaluation scores across multiple model runs.

    WHY THIS?
    In your resume, you mention "comparing models (GPT-4o vs Mistral
    vs Llama)". This function produces a side-by-side comparison table
    that you can screenshot for your evaluation report.

    results: list of dicts, each with keys:
      {model, answer, source_documents, scores}
    """
    rows = []
    for result in results:
        row = {
            "Model": result.get("model", "Unknown"),
            "Faithfulness": result.get("scores", {}).get("faithfulness", "N/A"),
            "Answer Relevance": result.get("scores", {}).get("answer_relevance", "N/A"),
            "Context Relevance": result.get("scores", {}).get("context_relevance", "N/A"),
        }
        rows.append(row)

    return pd.DataFrame(rows)
