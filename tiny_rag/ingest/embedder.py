"""
Embedder: turn text into dense vectors using bge-m3.

Key design:
  - Use sentence-transformers (handles batching, device, pooling correctly)
  - Lazy-init model (load once, cache on disk in ~/.cache/huggingface/)
  - L2-normalize vectors so cosine sim = dot product (faster)
  - batch_size configurable for CPU vs GPU
  - Truncation safety (chunk shouldn't exceed 512 bge-m3 tokens, but defense in depth)

bge-m3 specifics:
  - 1024-dim dense vectors
  - Supports dense / sparse / multi-vector (we use dense only)
  - M3 MPS has known padding_idx issues, so default to CPU
"""
from typing import List, Dict, Any
import numpy as np

from tiny_rag.config import settings

# Lazy globals
_MODEL = None
_DEVICE = None


def _resolve_device() -> str:
    """Resolve settings.device='auto' to actual device string.

    Note: bge-m3 on MPS is known to NaN/OOM (padding_idx bug in older transformers
    + bge-m3 interaction). Default to CPU for safety on Mac M-series.
    Set TINY_RAG_DEVICE=cuda or mps to override.
    """
    if settings.device != "auto":
        return settings.device
    import torch
    if torch.cuda.is_available():
        return "cuda"
    # Skip MPS by default for bge-m3 — known compatibility issues
    return "cpu"


def _get_model():
    """Lazy-init bge-m3 model. Downloaded once, cached on disk."""
    global _MODEL, _DEVICE
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _DEVICE = _resolve_device()
        print(f"Loading embedding model: {settings.embedding_model}")
        print(f"  (first run downloads ~2GB, subsequent runs use HF cache)")
        print(f"  Device: {_DEVICE}")
        _MODEL = SentenceTransformer(settings.embedding_model, device=_DEVICE)
    return _MODEL


def embed_chunks(
    chunks: List[Dict[str, Any]],
    batch_size: int = 16,
    show_progress: bool = True,
) -> np.ndarray:
    """Embed a list of chunks into a (N, D) float32 array.

    Args:
        chunks: list of chunk dicts, each must have "text" field
        batch_size: encoding batch size (16-32 for CPU, 64+ for GPU)
        show_progress: whether to show tqdm bar (if available)

    Returns:
        np.ndarray of shape (len(chunks), 1024), dtype float32, L2-normalized
    """
    if not chunks:
        return np.zeros((0, 1024), dtype="float32")

    model = _get_model()
    texts = [c["text"] for c in chunks]

    # sentence-transformers handles batching, device, pooling internally
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalize → cosine sim = dot product
    )
    return vectors.astype("float32")


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string.

    Args:
        query: query text

    Returns:
        np.ndarray of shape (1024,), dtype float32, L2-normalized
    """
    model = _get_model()
    vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vector[0].astype("float32")
