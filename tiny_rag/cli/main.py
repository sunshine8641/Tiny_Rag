"""
Tiny_RAG command-line interface.

Usage:
    python -m tiny_rag ingest <file_or_dir> [--pattern *.pdf] [--recursive]
    python -m tiny_rag sync   <file_or_dir> [--pattern *.pdf] [--recursive] [--force]
    python -m tiny_rag query  "<question>" [--top-k 5] [--backend mock]
    python -m tiny_rag status
    python -m tiny_rag remove <file_path>

Examples:
    python -m tiny_rag ingest ~/books/my_book.pdf              # full ingest
    python -m tiny_rag sync   ~/books --recursive              # incremental, skips unchanged
    python -m tiny_rag sync   ~/books --force                  # re-ingest everything
    python -m tiny_rag query  "What is Tiny_RAG?" --backend mock
    python -m tiny_rag query  "什么是 X" --backend openai --model gpt-4o-mini
    python -m tiny_rag status
    python -m tiny_rag remove ~/books/my_book.pdf
"""
import argparse
import sys
from pathlib import Path

from tiny_rag.config import settings
from tiny_rag.ingest.pipeline import ingest_file, ingest_directory
from tiny_rag.ingest.sync import sync_file, sync_directory
from tiny_rag.ingest.store import count, list_sources, delete_by_source
from tiny_rag.query.pipeline import answer_query


# ============================================================
# Subcommands
# ============================================================
def cmd_ingest(args):
    path = Path(args.path)
    if not path.exists():
        print(f"Error: {path} does not exist", file=sys.stderr)
        sys.exit(1)

    if path.is_file():
        n = ingest_file(path)
        sys.exit(0 if n >= 0 else 1)
    elif path.is_dir():
        # ingest_directory also takes pattern; accept multi
        # (one pattern keeps backward compat)
        results = ingest_directory(
            path,
            pattern=args.pattern[0] if args.pattern else "*.pdf",
            recursive=args.recursive,
        )
        # Exit non-zero if nothing ingested
        sys.exit(0 if any(v > 0 for v in results.values()) else 1)


def cmd_sync(args):
    path = Path(args.path)
    if not path.exists():
        print(f"Error: {path} does not exist", file=sys.stderr)
        sys.exit(1)

    if path.is_file():
        result = sync_file(path, force=args.force)
        print(f"  {result['action']:8s} {path.name} (chunks={result.get('chunks', 0)})")
    elif path.is_dir():
        # Default to all supported formats if user didn't specify
        patterns = args.pattern or ["*.pdf", "*.md", "*.markdown", "*.txt", "*.docx"]
        sync_directory(
            path,
            pattern=patterns,
            recursive=args.recursive,
            force=args.force,
        )


def cmd_query(args):
    result = answer_query(
        args.query,
        top_k=args.top_k,
        backend=args.backend,
        model=args.model,
    )
    print(f"\n📝 Q: {result['query']}\n")
    print(f"💡 A:\n{result['answer']}\n")
    if result["sources"]:
        print("📖 Sources:")
        for s in result["sources"]:
            print(f"  [{s['index']}] {s['filename']} (p.{s['page']}, sim={s['similarity']:.3f})")
            print(f"      {s['snippet']}")


def cmd_status(args):
    n = count()
    print(f"Total chunks: {n}")
    print(f"Data dir:     {settings.chroma_dir}")
    print(f"Embedding:    {settings.embedding_model}")
    print(f"LLM:          {settings.llm_model}")
    print()
    sources = list_sources()
    if sources:
        print(f"Sources ({len(sources)} files):")
        for s in sources:
            print(f"  {s}")
    else:
        print("Sources: (empty — run 'tiny_rag ingest' first)")


def cmd_remove(args):
    path = args.path
    n = delete_by_source(path)
    if n == 0:
        print(f"No chunks found for {path}")
    else:
        print(f"✓ Deleted {n} chunks from {path}")


def cmd_web(args):
    """Launch the Gradio web UI."""
    import gradio as gr
    from tiny_rag.gui.app import build_ui
    demo = build_ui()
    print(f"\n🚀 Tiny_RAG web UI")
    print(f"   Open: http://{args.host}:{args.port}\n")
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=False,
        theme=gr.themes.Soft(),
    )


# ============================================================
# Parser
# ============================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tiny_rag",
        description="Tiny_RAG: personal knowledge base RAG tool",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest files into the vector store")
    p_ingest.add_argument("path", help="File or directory to ingest")
    p_ingest.add_argument(
        "--pattern", default="*.pdf",
        help="Glob pattern for directory mode (default: *.pdf)",
    )
    p_ingest.add_argument(
        "--recursive", action="store_true",
        help="Scan subdirectories",
    )
    p_ingest.set_defaults(func=cmd_ingest)

    # sync (incremental)
    p_sync = sub.add_parser(
        "sync", help="Incremental sync: ingest new/changed, skip unchanged, cleanup deleted"
    )
    p_sync.add_argument("path", help="File or directory to sync")
    p_sync.add_argument(
        "--pattern", action="append",
        default=None,
        help="Glob pattern to include (repeatable). "
             "Default: *.pdf *.md *.markdown *.txt *.docx",
    )
    p_sync.add_argument("--recursive", action="store_true", help="Scan subdirectories")
    p_sync.add_argument(
        "--force", action="store_true",
        help="Re-ingest all files regardless of state",
    )
    p_sync.set_defaults(func=cmd_sync)

    # query
    p_query = sub.add_parser("query", help="Query the knowledge base")
    p_query.add_argument("query", help="Question to ask")
    p_query.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve (default 5)")
    p_query.add_argument(
        "--backend", default=None,
        help="LLM backend: mock | openai | anthropic | ollama (default: infer from model)",
    )
    p_query.add_argument(
        "--model", default=None,
        help="Model name override (default: settings.llm_model)",
    )
    p_query.set_defaults(func=cmd_query)

    # status
    p_status = sub.add_parser("status", help="Show vector store status")
    p_status.set_defaults(func=cmd_status)

    # web (Gradio UI)
    p_web = sub.add_parser("web", help="Launch Gradio web UI")
    p_web.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1)")
    p_web.add_argument("--port", type=int, default=7860, help="Bind port (default 7860)")
    p_web.add_argument("--share", action="store_true", help="Create public gradio.live URL")
    p_web.set_defaults(func=cmd_web)

    # remove
    p_remove = sub.add_parser("remove", help="Remove all chunks from a source file")
    p_remove.add_argument("path", help="Absolute path of the source file to remove")
    p_remove.set_defaults(func=cmd_remove)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
