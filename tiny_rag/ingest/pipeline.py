"""
Ingestion pipeline: load → chunk → embed → store, end to end.

Public API:
  - ingest_file(path) -> int              # single file → chunk count
  - ingest_directory(directory) -> dict    # batch → {filename: chunks}

Both handle errors gracefully so one bad file doesn't kill the batch.

Usage:
    from tiny_rag.ingest.pipeline import ingest_file, ingest_directory
    n = ingest_file("book.pdf")
    results = ingest_directory("~/books")  # ~ is expanded automatically
"""
import os
from pathlib import Path
from typing import Dict, Union

from tqdm import tqdm

from tiny_rag.ingest.loaders import load
from tiny_rag.ingest.chunker import chunk_pages
from tiny_rag.ingest.embedder import embed_chunks
from tiny_rag.ingest.store import add_chunks


def ingest_file(path: Union[str, Path]) -> int:
    """Ingest a single file end-to-end.

    Steps: load → chunk → embed → upsert to Chroma.

    Args:
        path: path to a file (.pdf / .md / .markdown / .txt / .docx).
              "~" is expanded automatically.

    Returns:
        number of chunks added to the store

    Raises:
        FileNotFoundError: if path doesn't exist
        ValueError: if file type unsupported
    """
    path = Path(os.path.expanduser(str(path)))
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    print(f"\n→ Ingesting: {path.name}")

    # 1. Load (auto-detect format via suffix)
    pages = load(path)
    if not pages:
        print(f"  ⚠ No content extracted, skipping")
        return 0
    print(f"  Loaded:  {len(pages)} pages")

    # 2. Chunk
    chunks = chunk_pages(pages)
    if not chunks:
        print(f"  ⚠ No chunks generated (text might be too short), skipping")
        return 0
    print(f"  Chunked: {len(chunks)} chunks")

    # 3. Embed
    vectors = embed_chunks(chunks, batch_size=16, show_progress=False)
    print(f"  Embedded: {vectors.shape[0]} vectors of dim {vectors.shape[1]}")

    # 4. Store
    n = add_chunks(chunks, vectors)
    print(f"  ✓ Stored: {n} chunks")

    return n


def ingest_directory(
    directory: Union[str, Path],
    pattern: str = "*.pdf",
    recursive: bool = False,
) -> Dict[str, int]:
    """Ingest all matching files in a directory.

    Args:
        directory: path to a directory. "~" is expanded automatically.
        pattern: glob pattern (default "*.pdf")
        recursive: if True, scan subdirectories

    Returns:
        dict of {filename: chunks_added}; 0 means failed or empty
    """
    directory = Path(os.path.expanduser(str(directory)))
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    glob_func = directory.rglob if recursive else directory.glob
    files = sorted(glob_func(pattern))
    if not files:
        print(f"No files matching {pattern!r} in {directory}")
        return {}

    print(f"Found {len(files)} files in {directory}")

    results: Dict[str, int] = {}
    for f in tqdm(files, desc="Ingesting"):
        try:
            n = ingest_file(f)
            results[f.name] = n
        except Exception as e:
            print(f"  ✗ Failed: {f.name} — {e}")
            results[f.name] = 0

    # Summary
    total_chunks = sum(results.values())
    total_files = len(results)
    success_files = sum(1 for v in results.values() if v > 0)
    print(
        f"\n✓ Done: {success_files}/{total_files} files ingested, "
        f"{total_chunks} total chunks"
    )

    return results
