"""
Tiny_RAG global configuration.

All tunable knobs in one place. Reads from .env if present, falls back
to defaults. Use `from tiny_rag.config import settings` to access.

Design choices:
  - Pydantic v2 for type safety and validation
  - Single global `settings` instance (singleton pattern)
  - Environment variables override defaults
"""
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root (assumes config.py is at tiny_rag/config.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Global settings for Tiny_RAG."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="TINY_RAG_",
        extra="ignore",
    )

    # ─── Storage paths ───────────────────────────────────
    data_dir: Path = PROJECT_ROOT / "data"
    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma"
    state_file: Path = PROJECT_ROOT / "data" / "state.json"
    cache_dir: Path = PROJECT_ROOT / "data" / "cache"

    # ─── Models ──────────────────────────────────────────
    embedding_model: str = "BAAI/bge-m3"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    llm_model: str = "gpt-4o-mini"

    # ─── Chunking ────────────────────────────────────────
    chunk_size: int = 512            # tokens per chunk
    chunk_overlap: int = 100         # overlap tokens

    # ─── Retrieval ───────────────────────────────────────
    top_k_retrieve: int = 50         # bi-encoder top-k
    top_k_rerank: int = 5            # final top-k after rerank

    # ─── Device ──────────────────────────────────────────
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"

    # ─── LLM ─────────────────────────────────────────────
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    # ─── Misc ────────────────────────────────────────────
    log_level: str = "INFO"

    def ensure_dirs(self) -> None:
        """Create data dirs if they don't exist."""
        for d in [self.data_dir, self.chroma_dir, self.cache_dir]:
            d.mkdir(parents=True, exist_ok=True)


# Singleton — import this everywhere
settings = Settings()
settings.ensure_dirs()
