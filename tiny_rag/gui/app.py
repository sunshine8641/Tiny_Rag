"""
Gradio web UI for Tiny_RAG.

Launch: python -m tiny_rag web
Then open http://127.0.0.1:7860 in browser.

Features:
  - Chat tab: ask questions, see sources
  - Ingest tab: drop PDF/MD/TXT/DOCX files, ingest
  - Status tab: see chunk count and source list
"""
from pathlib import Path
import time

import gradio as gr

from tiny_rag.config import settings
from tiny_rag.ingest.pipeline import ingest_file, ingest_directory
from tiny_rag.ingest.store import count, list_sources
from tiny_rag.query.pipeline import answer_query
from tiny_rag.query.generator import generate_stream


# ============================================================
# Helpers
# ============================================================
def _format_sources(sources):
    """Build Markdown block for source citations."""
    if not sources:
        return ""
    md = "\n\n---\n**📖 来源：**\n\n"
    for s in sources:
        md += f"- **[{s['index']}]** {s['filename']} p.{s['page']} (sim={s['similarity']:.3f})\n"
        md += f"  > {s['snippet'][:100]}...\n"
    return md


def _status_text():
    """Render current store status as text."""
    n = count()
    sources = list_sources()
    src_md = "\n".join(f"- {s}" for s in sources) if sources else "_(empty)_"
    return (
        f"**Total chunks:** {n}\n"
        f"**Data dir:**    {settings.chroma_dir}\n"
        f"**Embedding:**   {settings.embedding_model}\n"
        f"**LLM:**         {settings.llm_model}\n"
        f"**Sources ({len(sources)}):**\n{src_md}"
    )


# ============================================================
# Event handlers
# ============================================================
def chat_respond(message, history, backend, model, top_k, use_rerank):
    """Chat handler (streaming). Yields (cleared_input, updated_history) repeatedly.

    Gradio 6.x Chatbot expects each message as dict: {"role": ..., "content": ...}
    """
    from tiny_rag.query.retriever import retrieve
    from tiny_rag.query.reranker import rerank
    from tiny_rag.query.context_builder import build_context

    if not message.strip():
        yield "", history
        return

    # Normalize history → list of dicts
    normalized = []
    for msg in history:
        if isinstance(msg, dict):
            normalized.append(msg)
        elif isinstance(msg, (list, tuple)) and len(msg) == 2:
            normalized.append({"role": "user", "content": msg[0]})
            normalized.append({"role": "assistant", "content": msg[1]})

    # Add user message + empty assistant placeholder
    normalized.append({"role": "user", "content": message})
    normalized[-1]["content"] = ""  # ensure trailing element is empty
    # Actually we need TWO new entries: user + empty assistant
    # Fix: if last dict is user (from append above), add assistant
    if normalized[-1]["role"] == "user":
        normalized.append({"role": "assistant", "content": "▌ thinking..."})
    yield "", normalized

    try:
        # 1. Retrieve (non-streaming, fast)
        t0 = time.time()
        fetch_k = (10 if use_rerank else int(top_k))
        hits = retrieve(message, top_k=fetch_k)
        if use_rerank and len(hits) > int(top_k):
            hits = rerank(message, hits, top_k=int(top_k))
        prompt, sources = build_context(hits, message)
        t_retrieve = time.time() - t0

        sources_md = _format_sources(sources)

        # Show "retrieving done, generating..." state
        normalized[-1]["content"] = (
            f"⏱️ retrieve: `{t_retrieve:.2f}s`\n\n▌ generating..."
        )
        yield "", normalized

        # 2. Stream LLM response
        t_gen_start = time.time()
        full_answer = ""
        for chunk in generate_stream(
            prompt,
            backend=backend,
            model=model.strip() or None,
        ):
            full_answer += chunk
            normalized[-1]["content"] = (
                f"{full_answer}▌"
            )
            yield "", normalized

        # 3. Finalize: replace with full answer + sources + final timing
        t_gen = time.time() - t_gen_start
        normalized[-1]["content"] = (
            full_answer
            + sources_md
            + f"\n\n---\n"
            + f"⏱️ retrieve: `{t_retrieve:.2f}s`  "
            + f"generate: `{t_gen:.2f}s`  "
            + f"total: `{t_retrieve + t_gen:.2f}s`"
        )
        yield "", normalized

    except Exception as e:
        normalized[-1]["content"] = f"❌ Error: {e}"
        yield "", normalized


def do_ingest(files, recursive):
    """Ingest uploaded files. Returns (result_text, status_text)."""
    if not files:
        return "No files selected.", _status_text()

    lines = []
    for f in files:
        # Gradio gives a tempfile path
        path = Path(f.name if hasattr(f, "name") else f)
        if not path.exists():
            lines.append(f"✗ {path}: not found")
            continue
        try:
            if path.is_file():
                n = ingest_file(path)
                lines.append(f"✓ {path.name}: {n} chunks")
            elif path.is_dir():
                res = ingest_directory(path, recursive=recursive)
                total = sum(res.values())
                lines.append(f"✓ {path.name}/: {total} chunks in {len(res)} file(s)")
            else:
                lines.append(f"✗ {path}: not a file or dir")
        except Exception as e:
            lines.append(f"✗ {path.name}: {e}")

    return "\n".join(lines), _status_text()


# ============================================================
# UI
# ============================================================
def build_ui():
    with gr.Blocks(title="Tiny_RAG") as demo:
        gr.Markdown("# 📚 Tiny_RAG — 个人知识库 RAG")
        top_bar = gr.Markdown(
            f"📂 {settings.data_dir}  ·  "
            f"🧠 {settings.embedding_model}  ·  "
            f"💬 {settings.llm_model}"
        )

        with gr.Tabs():
            # ---- Chat tab ----
            with gr.Tab("💬 Chat"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(label="对话", height=450)
                        msg = gr.Textbox(
                            label="你的问题",
                            placeholder="问点什么…（按 Enter 发送）",
                        )
                    with gr.Column(scale=1):
                        backend = gr.Dropdown(
                            choices=["mock", "ollama", "openai", "anthropic", "minimax"],
                            value="ollama",
                            label="LLM Backend",
                        )
                        model = gr.Textbox(
                            value="qwen3:8b",
                            label="Model (留空用默认)",
                        )
                        top_k = gr.Slider(
                            minimum=1, maximum=20, value=5, step=1,
                            label="Top-K（检索多少 chunk）",
                        )
                        use_rerank = gr.Checkbox(
                            value=False,
                            label="Use Rerank（慢但更准）",
                        )
                        gr.Markdown(
                            "💡 **Mock** 模式不要 API key\n"
                            "**Ollama** 需 `ollama serve` 跑着\n"
                            "**OpenAI / Anthropic / MiniMax** 需设 API key"
                        )

                def update_top_bar(backend_val, model_val):
                    """Update top bar to reflect current selection."""
                    model_display = model_val.strip() or settings.llm_model
                    return (
                        f"📂 {settings.data_dir}  ·  "
                        f"🧠 {settings.embedding_model}  ·  "
                        f"💬 {backend_val} / {model_display}"
                    )

                # Update top bar when backend or model changes
                backend.change(update_top_bar, [backend, model], top_bar)
                model.change(update_top_bar, [backend, model], top_bar)
                # Also update after each chat
                msg.submit(
                    chat_respond,
                    inputs=[msg, chatbot, backend, model, top_k, use_rerank],
                    outputs=[msg, chatbot],
                ).then(
                    update_top_bar, [backend, model], top_bar
                )

            # ---- Ingest tab ----
            with gr.Tab("📥 入库"):
                gr.Markdown(
                    "**支持格式：** `.pdf` `.md` `.markdown` `.txt` `.docx`\n\n"
                    "拖文件 / 目录到这里，或点上传。\n"
                    "目录模式会**全量入库**（重算），用 `sync` 命令做增量。"
                )
                files = gr.File(
                    label="选择文件或目录",
                    file_count="multiple",
                )
                recursive = gr.Checkbox(
                    value=False, label="递归子目录（仅目录）",
                )
                ingest_btn = gr.Button("📥 入库", variant="primary")
                ingest_out = gr.Textbox(label="入库结果", lines=10)
                ingest_status = gr.Textbox(label="库状态", lines=8)
                ingest_btn.click(
                    do_ingest,
                    inputs=[files, recursive],
                    outputs=[ingest_out, ingest_status],
                )

            # ---- Status tab ----
            with gr.Tab("📊 状态"):
                refresh_btn = gr.Button("🔄 刷新", variant="primary")
                status_text = gr.Textbox(
                    label="库状态", lines=20, value=_status_text(),
                )
                refresh_btn.click(_status_text, outputs=status_text)

    return demo


def main():
    demo = build_ui()
    print(f"\n🚀 Tiny_RAG web UI")
    print(f"   Open: http://127.0.0.1:7860\n")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=False,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
