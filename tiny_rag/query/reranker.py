"""
Cross-encoder reranker: rerank retrieved hits using bge-reranker-v2-m3.

Why rerank after bi-encoder retrieval:
  - Bi-encoder encodes q and d independently → fast but imprecise
  - Cross-encoder reads (q, d) pair jointly → slow but precise
  - Two-stage: bi-encoder top-100 → CE rerank → top-3

Model: BAAI/bge-reranker-v2-m3 (568M, multilingual, pairs with bge-m3)

Cost:
  - M3 CPU: ~50-100 ms per (q, d) pair
  - For 100 candidates: ~5-10 seconds
  - Worth it: typically +5-10 pt on retrieval quality
"""
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

from tiny_rag.config import settings


_MODEL = None


def _get_model():
    """Lazy-init bge-reranker. Downloads once (~1.5 GB), cached on disk."""
    global _MODEL
    if _MODEL is None:
        print(f"Loading rerank model: {settings.rerank_model}")
        print(f"  (first run downloads ~1.5GB, subsequent runs use HF cache)")
        _MODEL = CrossEncoder(settings.rerank_model, max_length=512)
    return _MODEL


def rerank(
    query: str,
    hits: List[Dict[str, Any]],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Rerank hits using a cross-encoder.

    Args:
        query: user query
        hits: list of hits from retriever (each must have "text" field)
        top_k: return top-k after rerank

    Returns:
        subset of hits, sorted by cross-encoder score desc,
        with "ce_score" added to each
    """
    if not hits:
        return []

    model = _get_model()
    pairs = [(query, hit["text"]) for hit in hits]
    scores = model.predict(pairs, show_progress_bar=False)

    # Pair each hit with its CE score
    scored = list(zip(hits, scores))
    scored.sort(key=lambda x: -float(x[1]))  # descending

    # Take top-k, attach score
    reranked = []
    for hit, score in scored[:top_k]:
        hit = dict(hit)  # copy to avoid mutating caller's list
        hit["ce_score"] = float(score)
        reranked.append(hit)
    return reranked
