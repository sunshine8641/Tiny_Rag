"""
File loaders: turn a file into structured pages with metadata.

For now: PDF only (pymupdf).
Future: Markdown, TXT, EPUB, URL.

Output format (a "page" is a unit that may go into a chunk):
    {
        "text": str,            # raw text of this page
        "page": int,            # 1-indexed page number (PDF only)
        "source": str,          # absolute path
        "filename": str,        # basename
        "title": str,           # title (for now: filename stem)
    }

Loader returns: list[dict] (one dict per page).
Chunker takes that list and splits it into smaller chunks later.
"""
from pathlib import Path
from typing import List, Dict, Any

import pymupdf  # PyMuPDF


def load_pdf(path: str | Path) -> List[Dict[str, Any]]:
    """Load a PDF file, return one dict per non-blank page.

    Args:
        path: path to PDF file

    Returns:
        list of {"text", "page", "source", "filename", "title"}

    Raises:
        FileNotFoundError: if path doesn't exist
        RuntimeError: if pymupdf fails to open
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a .pdf file: {path}")

    pages = []
    try:
        doc = pymupdf.open(path)
        for i, page in enumerate(doc):
            text = page.get_text("text")
            text = text.strip()
            if not text:
                # Skip blank pages (e.g., pure-image pages, chapter dividers)
                continue
            pages.append({
                "text": text,
                "page": i + 1,           # 1-indexed for human-friendly
                "source": str(path),
                "filename": path.name,
                "title": path.stem,      # placeholder; will improve later
            })
    except Exception as e:
        raise RuntimeError(f"Failed to parse PDF {path}: {e}") from e
    finally:
        # pymupdf docs recommend close() but it's a no-op in CPython
        pass

    return pages


def load(path: str | Path) -> List[Dict[str, Any]]:
    """Auto-detect format and dispatch to the right loader.

    Args:
        path: path to file (PDF for now)

    Returns:
        list of page dicts

    Raises:
        ValueError: if format unsupported
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    else:
        raise ValueError(
            f"Unsupported file type: {suffix}. "
            f"Supported: .pdf (more coming soon)"
        )
