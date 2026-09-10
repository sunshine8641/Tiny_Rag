"""
File loaders: turn a file into structured pages with metadata.

Supported formats:
  - PDF  (pymupdf4llm — preserves tables/headings/lists)
  - Markdown (.md, .markdown — direct read, treat each file as one "page")
  - TXT  (.txt — direct read, plain text)
  - DOCX (.docx — python-docx, extracts paragraphs + tables)

Output format (a "page" is a unit that may go into a chunk):
    {
        "text": str,            # Markdown / plain text
        "page": int,            # 1-indexed page number (PDF: per page; MD/TXT/DOCX: always 1)
        "source": str,          # absolute path
        "filename": str,        # basename
        "title": str,           # title (PDF: from metadata; others: filename stem)
    }
"""
from pathlib import Path
from typing import List, Dict, Any

import pymupdf4llm


# ============================================================
# PDF
# ============================================================
def load_pdf(path: str | Path) -> List[Dict[str, Any]]:
    """Load PDF as Markdown, one dict per non-empty page.

    See docstring at top of file for full schema.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a .pdf file: {path}")

    try:
        chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    except Exception as e:
        raise RuntimeError(f"Failed to parse PDF {path}: {e}") from e

    pages = []
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue
        meta = chunk.get("metadata", {}) or {}
        page_num = meta.get("page_number") or (len(pages) + 1)
        title = (meta.get("title") or "").strip() or path.stem
        pages.append({
            "text": text,
            "page": page_num,
            "source": str(path),
            "filename": path.name,
            "title": title,
        })
    return pages


# ============================================================
# Markdown
# ============================================================
def load_markdown(path: str | Path) -> List[Dict[str, Any]]:
    """Load a Markdown file as a single "page".

    Markdown files usually don't have pages — treat the whole file as one unit.
    The chunker will split it into smaller pieces later.

    If frontmatter (YAML between ---) exists, title from frontmatter overrides filename.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Markdown not found: {path}")
    if path.suffix.lower() not in (".md", ".markdown"):
        raise ValueError(f"Not a markdown file: {path}")

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="gbk", errors="ignore")  # fallback

    # Extract frontmatter for title (--- ... ---)
    title = path.stem
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            frontmatter = text[4:end]
            for line in frontmatter.split("\n"):
                if line.startswith("title:"):
                    title = line.split(":", 1)[1].strip().strip("'\"")
                    break
            text = text[end + 5:].strip()

    if not text.strip():
        return []

    return [{
        "text": text,
        "page": 1,
        "source": str(path),
        "filename": path.name,
        "title": title,
    }]


# ============================================================
# Plain text
# ============================================================
def load_txt(path: str | Path) -> List[Dict[str, Any]]:
    """Load a plain text file as a single "page"."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"TXT not found: {path}")
    if path.suffix.lower() != ".txt":
        raise ValueError(f"Not a .txt file: {path}")

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="gbk", errors="ignore")

    text = text.strip()
    if not text:
        return []

    return [{
        "text": text,
        "page": 1,
        "source": str(path),
        "filename": path.name,
        "title": path.stem,
    }]


# ============================================================
# Word (DOCX)
# ============================================================
def load_docx(path: str | Path) -> List[Dict[str, Any]]:
    """Load a Word document. Extracts paragraphs and tables as Markdown.

    Headings are prefixed with # (Heading 1 → #, Heading 2 → ##, etc.)
    Tables are converted to Markdown table syntax.
    """
    from docx import Document

    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"DOCX not found: {path}")
    if path.suffix.lower() != ".docx":
        raise ValueError(f"Not a .docx file: {path}")

    doc = Document(str(path))
    parts = []
    title = path.stem  # default

    # Paragraphs (with heading detection)
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style_name = para.style.name if para.style else ""
        if style_name.startswith("Heading"):
            level = style_name.replace("Heading", "").strip()
            try:
                level_num = int(level) if level else 1
            except ValueError:
                level_num = 1
            text = f"{'#' * level_num} {text}"
            # First heading becomes title
            if title == path.stem:
                title = text.lstrip("#").strip()
        parts.append(text)

    # Tables → Markdown
    for table in doc.tables:
        rows = []
        for row in table.rows:
            cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            rows.append("| " + " | ".join(cells) + " |")
        if rows:
            # Markdown table: header + separator + body
            n_cols = len(table.columns)
            separator = "| " + " | ".join(["---"] * n_cols) + " |"
            parts.append("\n".join([rows[0], separator] + rows[1:]))

    full_text = "\n\n".join(parts).strip()
    if not full_text:
        return []

    return [{
        "text": full_text,
        "page": 1,
        "source": str(path),
        "filename": path.name,
        "title": title,
    }]


# ============================================================
# Dispatcher
# ============================================================
def load(path: str | Path) -> List[Dict[str, Any]]:
    """Auto-detect format and dispatch to the right loader.

    Args:
        path: path to file (.pdf / .md / .markdown / .txt / .docx)

    Returns:
        list of page dicts

    Raises:
        ValueError: if format unsupported
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    elif suffix in (".md", ".markdown"):
        return load_markdown(path)
    elif suffix == ".txt":
        return load_txt(path)
    elif suffix == ".docx":
        return load_docx(path)
    else:
        raise ValueError(
            f"Unsupported file type: {suffix}. "
            f"Supported: .pdf / .md / .markdown / .txt / .docx"
        )
