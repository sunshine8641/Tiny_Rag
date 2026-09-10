"""
Vector store: Chroma persistent client for embeddings + metadata.

Chroma 0.6.x API surface we use:
  - PersistentClient(path=...)           # file-based persistence
  - get_or_create_collection(name, ...)  # idempotent
  - collection.upsert(ids, embeddings, metadatas, documents)
  - collection.query(query_embeddings, n_results)
  - collection.delete(where={...})
  - collection.count() / collection.get(include=[...])

Why Chroma (not raw FAISS / Qdrant):
  - File-based, zero server, zero config
  - Built-in metadata filtering (FAISS needs manual mapping)
  - Persistent out of the box
  - Pythonic API

Why cosine distance (not L2 / IP):
  - bge-m3 vectors are L2-normalized at embedder time
  - Under L2 norm = 1, cosine distance = 1 - dot product
  - Chroma returns "distance" where lower = better match
"""
from typing import List, Dict, Any, Optional
import os
import sys

# Suppress chromadb 0.6.3 telemetry warnings caused by opentelemetry version mismatch.
# chromadb calls `capture(self, name, attributes)` (3 args), but newer opentelemetry
# changed the signature. The fix is non-trivial (full opentelemetry version pinning
# breaks other things in the env), so we filter the harmless stderr spam instead.
# Functionality is unaffected — telemetry is opt-out by design.
class _TelemetryFilter:
    # Suppress known chromadb 0.6.3 harmless warnings
    _PATTERNS = [
        "Failed to send telemetry event",         # opentelemetry API mismatch
        "Delete of nonexisting embedding ID",     # delete_by_source on missing ids
    ]
    def __init__(self, original):
        self._original = original
    def write(self, msg):
        for p in self._PATTERNS:
            if p in msg:
                return
        self._original.write(msg)
    def flush(self):
        self._original.flush()
    def __getattr__(self, name):
        return getattr(self._original, name)

sys.stderr = _TelemetryFilter(sys.stderr)

# Also disable Chroma telemetry at the env level
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
from chromadb.config import Settings as ChromaSettings

from tiny_rag.config import settings


# Module-level singletons
_CLIENT = None
_COLLECTION = None


# ============================================================
# Internal helpers
# ============================================================
def _get_client():
    """Lazy-init Chroma PersistentClient. Settings dir is shared."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(
                anonymized_telemetry=False,  # no phone-home
                allow_reset=False,
            ),
        )
    return _CLIENT


def _get_collection():
    """Lazy-init the default collection. Cosine distance for normalized vectors."""
    global _COLLECTION
    if _COLLECTION is None:
        client = _get_client()
        _COLLECTION = client.get_or_create_collection(
            name="tiny_rag",
            metadata={"hnsw:space": "cosine"},
        )
    return _COLLECTION


def _chunk_to_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Pick the fields we want to store as Chroma metadata.

    Chroma metadata values must be str/int/float/bool/None — no nested dicts.
    """
    return {
        "source": str(chunk["source"]),
        "filename": str(chunk["filename"]),
        "title": str(chunk["title"]),
        "page": int(chunk["page"]),
        "chunk_index": int(chunk["chunk_index"]),
        "token_count": int(chunk["token_count"]),
    }


# ============================================================
# Public API
# ============================================================
def add_chunks(chunks: List[Dict[str, Any]], vectors) -> int:
    """Upsert chunks + their vectors into the collection.

    Args:
        chunks: list of chunk dicts (each must have chunk_id, text, source, ...)
        vectors: np.ndarray of shape (len(chunks), 1024), float32, L2-normalized

    Returns:
        number of chunks upserted
    """
    if not chunks:
        return 0

    coll = _get_collection()
    coll.upsert(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=vectors.tolist(),  # Chroma wants list of list
        metadatas=[_chunk_to_metadata(c) for c in chunks],
        documents=[c["text"] for c in chunks],
    )
    return len(chunks)


def query(
    query_vector,
    top_k: int = 10,
    where: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Search by vector, return ranked hits.

    Args:
        query_vector: 1D np.ndarray of shape (1024,), L2-normalized
        top_k: number of hits to return
        where: optional metadata filter, e.g. {"filename": "book.pdf"}

    Returns:
        list of {"id", "text", "metadata", "distance"}, sorted by distance asc

    Workaround for Chroma 0.6.3: query's `where` filter can return 0 hits even
    when matching chunks exist (get with same filter works). We work around by
    fetching top-K*5 and post-filtering in Python.
    """
    coll = _get_collection()
    # Fetch more than needed, then post-filter
    fetch_k = top_k * 5 if where else top_k
    results = coll.query(
        query_embeddings=[query_vector.tolist()],
        n_results=fetch_k,
    )

    hits = []
    for i, doc_id in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i] or {}
        if where:
            # Post-filter: check all where conditions
            if not all(meta.get(k) == v for k, v in where.items()):
                continue
        hits.append({
            "id": doc_id,
            "text": results["documents"][0][i],
            "metadata": meta,
            "distance": results["distances"][0][i],
        })
        if len(hits) >= top_k:
            break
    return hits


def delete_by_source(source: str) -> int:
    """Delete all chunks belonging to a given source file.

    Args:
        source: absolute path string (matches the "source" metadata field)

    Returns:
        number of chunks deleted (approximate — Chroma may not report exactly)
    """
    coll = _get_collection()
    # Get count before delete (for return value)
    before = coll.get(where={"source": source}, include=[])
    n = len(before["ids"])
    coll.delete(where={"source": source})
    return n


def count() -> int:
    """Total number of chunks in the collection."""
    return _get_collection().count()


def list_sources() -> List[str]:
    """List unique source file paths in the collection, sorted."""
    coll = _get_collection()
    # Get all metadata, extract source field
    results = coll.get(include=["metadatas"])
    sources = {
        m["source"]
        for m in results["metadatas"]
        if m and "source" in m
    }
    return sorted(sources)
