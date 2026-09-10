"""
Chunker: split page-level text into smaller, embeddable chunks.

Key design:
  - Use langchain RecursiveCharacterTextSplitter (battle-tested)
  - Length measured in TOKENS via bge-m3's AutoTokenizer
    (so chunks are guaranteed to fit bge-m3's 512 max_length)
  - Per-page chunking (don't cross page boundaries) — keeps page metadata clean
  - Inherit page metadata, add chunk_index, chunk_id, token_count

Why bge-m3 tokenizer (not tiktoken):
  - tiktoken (cl100k_base) and bge-m3 (BERT WordPiece) tokenize differently
  - For Chinese-heavy text, counts can differ 5-15%
  - 5-10% of chunks would be silently truncated at embedding time
  - Using the matching tokenizer makes chunk sizing precise

Why per-page (not whole-doc):
  - Simple, predictable metadata
  - No risk of losing page attribution when chunk spans 2 pages

Cost:
  - One-time download of bge-m3 tokenizer (~10 MB, seconds)
  - Slightly slower split (each length check is an encode call)
"""
from typing import List, Dict, Any, Optional
import re

from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from tiny_rag.config import settings


# Module-level singletons
_TOKENIZER = None
_SPLITTER_CACHE: dict = {}


def _get_tokenizer():
    """Lazy-init bge-m3 tokenizer. Downloaded once, cached on disk."""
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(settings.embedding_model)
    return _TOKENIZER


def _make_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """Build a splitter that measures length in bge-m3 tokens.

    Caches by (chunk_size, chunk_overlap) to avoid rebuilding per call.
    """
    key = (chunk_size, chunk_overlap)
    if key not in _SPLITTER_CACHE:
        tok = _get_tokenizer()
        _SPLITTER_CACHE[key] = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=lambda t: len(tok.encode(t, add_special_tokens=False)),
            separators=["\n\n", "\n", "。", ".", "！", "!", "？", "?", " ", ""],
        )
    return _SPLITTER_CACHE[key]


def _make_chunk_id(source: str, page: int, idx: int) -> str:
    """Stable unique ID: source#page#index.

    Same source/page/index → same ID, so re-ingesting the same file
    updates existing chunks instead of duplicating.
    """
    return f"{source}#p{page}#c{idx}"


def _count_tokens(text: str) -> int:
    """Count bge-m3 tokens for a string (excludes special tokens)."""
    return len(_get_tokenizer().encode(text, add_special_tokens=False))


def _normalize_whitespace(text: str) -> str:
    """Collapse excessive whitespace, preserve paragraph structure.

    Why:
      - PDF extraction often leaves 3-10 consecutive newlines (page-break residues)
      - OCR can leave tabs and double spaces
      - PDFs sometimes interleave "\\n" and " " in weird ways (e.g. "\\n \\n \\n")
      - These waste tokens and confuse the splitter

    Rules (conservative, only collapse excess, never destroy structure):
      - 3+ "newline + optional whitespace" repetitions → 2 newlines
      - tabs → single space
      - 2+ spaces → 1
      - final strip (kill leading/trailing whitespace)

    Key regex: r"(\\n\\s*){3,}" matches 3+ repetitions of (newline + any whitespace).
    This catches "\\n\\n\\n", "\\n \\n \\n", "\\n\\t\\n" — all the common variants.
    Plain r"\\n{3,}" would miss any pattern with a space/tab between newlines.
    """
    text = text.strip()
    # tabs → space
    text = text.replace("\t", " ")
    # 3+ "newline + any whitespace" repeats → 2 newlines
    text = re.sub(r"(\n\s*){3,}", "\n\n", text)
    # 2+ spaces → 1
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def chunk_pages(
    pages: List[Dict[str, Any]],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Split a list of page dicts into chunks.

    Args:
        pages: output of loaders.load_pdf — list of
               {"text", "page", "source", "filename", "title"}
        chunk_size: max tokens per chunk (default: settings.chunk_size, in bge-m3 tokens)
        chunk_overlap: overlap tokens between consecutive chunks (default: settings.chunk_overlap)

    Returns:
        list of chunk dicts, each:
            {
                "text": str,
                "page": int,
                "source": str,
                "filename": str,
                "title": str,
                "chunk_index": int,        # within this page
                "chunk_id": str,           # unique: source#p{page}#c{index}
                "token_count": int,        # bge-m3 tokens
            }
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    splitter = _make_splitter(chunk_size, chunk_overlap)

    chunks = []
    for page in pages:
        # Normalize whitespace BEFORE splitting (kills 3+ \n, tabs, double spaces)
        text = _normalize_whitespace(page["text"])
        if not text:
            # Skip pages with no real content after normalization
            continue

        texts = splitter.split_text(text)
        for idx, chunk_text in enumerate(texts):
            chunk_text = chunk_text.strip()
            # Skip chunks that are empty or whitespace-only after strip
            # (happens when splitter cuts at consecutive separators)
            if not chunk_text:
                continue
            chunk = {
                # Inherit page metadata
                "source": page["source"],
                "filename": page["filename"],
                "title": page["title"],
                "page": page["page"],
                # Chunk-specific
                "text": chunk_text,
                "chunk_index": idx,
                "chunk_id": _make_chunk_id(page["source"], page["page"], idx),
                "token_count": _count_tokens(chunk_text),
            }
            chunks.append(chunk)
    return chunks
