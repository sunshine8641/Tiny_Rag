"""
Retriever: query → top-k relevant chunks from vector store.

Pipeline:
  query (str)
    → embed_query (bge-m3 → 1024-dim vector)
    → store.query (Chroma cosine search)
    → list of hits (sorted by distance asc)

Each hit:
  {
    "id": str,           # chunk_id (e.g. "book.pdf#p3#c0")
    "text": str,         # chunk content (Markdown)
    "metadata": dict,    # source, filename, page, title, etc.
    "distance": float,   # cosine distance: 0 = identical, 2 = opposite
    "similarity": float, # 1 - distance, ∈ [0, 1], higher = more relevant
  }

For bge-m3 (which produces positive-only embeddings in practice), distance is
typically in [0, 1], so similarity = 1 - distance is a clean 0-1 score.
"""
from typing import List, Dict, Any, Optional

from tiny_rag.ingest.embedder import embed_query
from tiny_rag.ingest.store import query as store_query


def retrieve(
    query: str,
    top_k: int = 5,
    where: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Search for the top-k chunks most relevant to the query.

    Args:
        query: query string
        top_k: number of hits to return (default 5)
        where: optional Chroma metadata filter, e.g. {"filename": "book.pdf"}

    Returns:
        list of hits, each with id/text/metadata/distance/similarity,
        sorted by similarity desc (most relevant first)
    """
    # 1. Embed query
    q_vec = embed_query(query)

    # 2. Vector search via Chroma (cosine distance, sorted asc)
    hits = store_query(q_vec, top_k=top_k, where=where)

    # 3. Convert distance → similarity for human-readable scores
    for h in hits:
        h["similarity"] = 1.0 - h["distance"]

    return hits
