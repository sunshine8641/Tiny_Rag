"""
Query pipeline: query → retrieve → build prompt → generate → answer.

Single public function:
  answer_query(query, top_k=5, where=None, backend="mock", model=None) -> dict

This is the user-facing entry point. All other modules are implementation detail.

Usage:
    from tiny_rag.query.pipeline import answer_query

    # Mock (no API key needed)
    result = answer_query("What is Tiny_RAG?", backend="mock")
    print(result["answer"])
    print(result["sources"])

    # Real LLM
    result = answer_query("What is Tiny_RAG?", backend="openai", model="gpt-4o-mini")

Returns:
    {
        "query": str,
        "answer": str,
        "sources": list[dict],  # each: index, filename, page, similarity, snippet
        "retrieval_time": float,  # seconds for retrieve + (optional) rerank
        "generation_time": float,  # seconds for LLM call
        "total_time": float,       # end-to-end
    }
"""
import time
from typing import Optional, Dict, Any

from tiny_rag.query.retriever import retrieve
from tiny_rag.query.reranker import rerank
from tiny_rag.query.context_builder import build_context
from tiny_rag.query.generator import generate


def answer_query(
    query: str,
    top_k: int = 5,
    where: Optional[Dict[str, Any]] = None,
    backend: Optional[str] = None,
    model: Optional[str] = None,
    max_context_chars: int = 8000,
    use_rerank: bool = False,
    rerank_fetch_k: int = 10,
) -> Dict[str, Any]:
    """End-to-end query pipeline: query → retrieve → [rerank] → LLM answer.

    Args:
        query: user question (any language)
        top_k: final number of chunks to return (default 5)
        where: optional Chroma metadata filter, e.g. {"filename": "book.pdf"}
        backend: LLM backend — None (infer from model), "mock", "openai",
                 "anthropic", or "ollama"
        model: model name override (default: settings.llm_model)
        max_context_chars: max characters for the context section
        use_rerank: if True, retrieve rerank_fetch_k candidates and rerank to top_k
        rerank_fetch_k: how many candidates to fetch before reranking
                         (should be > top_k for rerank to have effect)

    Returns:
        dict with keys:
            - "query": the original query
            - "answer": LLM-generated answer (or mock text)
            - "sources": list of citation dicts (filename, page, similarity, snippet)
            - "retrieval_time": seconds (retrieve + optional rerank)
            - "generation_time": seconds (LLM call only)
            - "total_time": seconds (end-to-end)
    """
    t_total_start = time.time()

    # 1. Retrieve candidates (more than top_k if reranking)
    fetch_k = rerank_fetch_k if use_rerank else top_k
    t_retrieve_start = time.time()
    hits = retrieve(query, top_k=fetch_k, where=where)

    # 2. Optionally rerank
    if use_rerank and len(hits) > top_k:
        hits = rerank(query, hits, top_k=top_k)
    retrieval_time = time.time() - t_retrieve_start

    # 3. Build LLM prompt with sources (negligible time)
    prompt, sources = build_context(hits, query, max_chars=max_context_chars)

    # 4. Call LLM
    t_gen_start = time.time()
    answer = generate(prompt, backend=backend, model=model)
    generation_time = time.time() - t_gen_start

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
        "total_time": time.time() - t_total_start,
    }
