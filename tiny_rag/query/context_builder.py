"""
Context builder: turn retrieved hits into an LLM-ready prompt + source citations.

Output format:
  - prompt: "你是一个知识助手。基于以下参考资料回答问题...\n\n[1] book.pdf (p.42): ...\n[2] ..."
  - sources: list of {"index", "filename", "page", "title", "snippet", "similarity"}
            for showing "this answer came from book.pdf p.42" to the user

Why split prompt and sources:
  - prompt goes to LLM (needs structured context)
  - sources go to user (needs human-readable citation)
  - they have different formats
"""
from typing import List, Dict, Any, Tuple
from tiny_rag.config import settings


# Prompt template — instructs LLM to use only the provided context
DEFAULT_PROMPT_TEMPLATE = """你是一个知识助手。请基于规则回答用户的问题。

规则：
1. 如果【参考资料】里没有答案，在答案前指明，"当前回答没有采用参考资料。"
2. 回答时，如果有参考资料，在末尾用 [1] [2] 等标注引用了哪条资料
3. 用中文回答（除非用户用其他语言提问）

【参考资料】
{context}

【用户问题】
{query}

【回答】"""


def build_context(
    hits: List[Dict[str, Any]],
    query: str,
    max_chars: int = 8000,
    template: str = DEFAULT_PROMPT_TEMPLATE,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Build LLM prompt from retrieved hits.

    Args:
        hits: list of hits from retriever.retrieve()
              each: {"id", "text", "metadata", "distance", "similarity"}
        query: user query
        max_chars: max characters for the context section (prompt size guard)
        template: prompt template with {context} and {query} placeholders

    Returns:
        (prompt, sources) where:
            prompt: str, full LLM input
            sources: list of dicts with citation info for the user

    Note: max_chars is a soft cap on the context. We may include fewer hits
    if the cumulative text would exceed it.
    """
    if not hits:
        context = "（无相关资料）"
        sources = []
    else:
        context_parts = []
        sources = []
        used_chars = 0

        for i, hit in enumerate(hits, start=1):
            meta = hit.get("metadata", {}) or {}
            text = hit.get("text", "")
            similarity = hit.get("similarity", 0.0)

            # Source citation
            sources.append({
                "index": i,
                "filename": meta.get("filename", "?"),
                "title": meta.get("title", ""),
                "page": meta.get("page"),
                "source": meta.get("source", ""),
                "similarity": similarity,
                "snippet": text[:120].replace("\n", " "),
            })

            # Build a labeled chunk: "[i] filename (p.N): text"
            page = meta.get("page", "?")
            filename = meta.get("filename", "?")
            chunk_str = f"[{i}] {filename} (p.{page}):\n{text}"

            # Check if adding this chunk exceeds max_chars
            if used_chars + len(chunk_str) > max_chars and context_parts:
                # Skip the rest — context is full
                break
            context_parts.append(chunk_str)
            used_chars += len(chunk_str)

        context = "\n\n---\n\n".join(context_parts)

    prompt = template.format(context=context, query=query)
    return prompt, sources
