"""
LLM generator: take a prompt, return an answer.

Backends:
  - "openai"    (default, requires OPENAI_API_KEY)
  - "anthropic" (requires ANTHROPIC_API_KEY)
  - "ollama"    (local, no key needed; ollama must be running)
  - "minimax"   (OpenAI-compatible API, requires MINIMAX_API_KEY)
  - "mock"      (no API call, returns echo with citation listing — for testing)

Configuration via env:
  TINY_RAG_LLM_MODEL   (model name, e.g. "gpt-4o-mini" or "claude-3-5-sonnet")
  TINY_RAG_LLM_BACKEND (default: "openai")

Two modes:
  - generate(prompt) → str              # blocking, returns full answer
  - generate_stream(prompt) → Iterator[str]  # streaming, yields chunks
"""
from typing import Optional, Iterator
import os

from tiny_rag.config import settings


def generate(
    prompt: str,
    backend: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> str:
    """Generate an answer from a prompt (blocking, full text).

    Returns:
        Generated answer text
    """
    if backend is None:
        backend = _infer_backend(model or settings.llm_model)

    if backend == "mock":
        return _generate_mock(prompt)
    elif backend == "openai":
        return _generate_openai(prompt, model or "gpt-4o-mini", max_tokens, temperature)
    elif backend == "anthropic":
        return _generate_anthropic(prompt, model or "claude-3-5-sonnet-20241022", max_tokens, temperature)
    elif backend == "ollama":
        return _generate_ollama(prompt, model or "qwen3:8b", max_tokens, temperature)
    elif backend == "minimax":
        return _generate_minimax(prompt, model or settings.minimax_model, max_tokens, temperature)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def generate_stream(
    prompt: str,
    backend: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> Iterator[str]:
    """Stream an answer as a sequence of text chunks.

    Yields incremental text from the LLM. First chunk arrives in 0.5-1s
    (vs 3-5s for full response with blocking mode).
    """
    if backend is None:
        backend = _infer_backend(model or settings.llm_model)

    if backend == "mock":
        # Mock can't really stream — return as single chunk
        yield _generate_mock(prompt)
        return
    elif backend == "ollama":
        yield from _stream_ollama(prompt, model or "qwen3:8b", max_tokens, temperature)
    elif backend == "openai":
        yield from _stream_openai(prompt, model or "gpt-4o-mini", max_tokens, temperature)
    elif backend == "anthropic":
        yield from _stream_anthropic(prompt, model or "claude-3-5-sonnet-20241022", max_tokens, temperature)
    elif backend == "minimax":
        yield from _stream_minimax(prompt, model or settings.minimax_model, max_tokens, temperature)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _infer_backend(model: str) -> str:
    """Guess backend from model name."""
    m = model.lower()
    if "gpt" in m or "openai" in m:
        return "openai"
    if "claude" in m or "anthropic" in m:
        return "anthropic"
    if "qwen" in m or "llama" in m or "mistral" in m:
        return "ollama"
    if "minimax" in m or "abab" in m:
        return "minimax"
    return "openai"  # default


def _generate_mock(prompt: str) -> str:
    """Mock: return a canned response that shows the prompt was received.

    Useful for testing the full pipeline without API calls.
    """
    return (
        "[MOCK RESPONSE — no LLM was actually called]\n\n"
        "（这里是 LLM 应该回答的内容。当前是 mock 模式。\n"
        "要接真实 LLM，设环境变量 OPENAI_API_KEY 并用 backend='openai'，"
        "或装 Ollama 跑本地模型用 backend='ollama'。）\n\n"
        "Prompt 收到了：\n" + "─" * 40 + "\n"
        f"{prompt[:200]}..." if len(prompt) > 200 else prompt
    )


def _generate_openai(prompt: str, model: str, max_tokens: int, temperature: float) -> str:
    """Call OpenAI Chat Completions API."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    api_key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not set. Set it in .env or environment."
        )

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def _generate_anthropic(prompt: str, model: str, max_tokens: int, temperature: float) -> str:
    """Call Anthropic Claude API."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package not installed. Run: pip install anthropic")

    api_key = settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not set. Set it in .env or environment."
        )

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    # Extract text from content blocks
    parts = [block.text for block in resp.content if hasattr(block, "text")]
    return "\n".join(parts)


def _generate_ollama(prompt: str, model: str, max_tokens: int, temperature: float) -> str:
    """Call local Ollama (default http://localhost:11434)."""
    try:
        import requests
    except ImportError:
        raise ImportError("requests package not installed. Run: pip install requests")

    url = settings.ollama_base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
        },
    }
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json().get("response", "")


def _generate_minimax(prompt: str, model: str, max_tokens: int, temperature: float) -> str:
    """Call MiniMax via OpenAI-compatible API.

    MiniMax's API mimics the OpenAI Chat Completions interface,
    so we can use the openai SDK with a custom base_url.

    Args:
        prompt: user prompt
        model: model name (e.g. "MiniMax-Text-01", "abab6.5s-chat")
        max_tokens: max response length
        temperature: sampling temperature
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    api_key = settings.minimax_api_key or os.environ.get("MINIMAX_API_KEY", "")
    if not api_key:
        raise ValueError(
            "MINIMAX_API_KEY not set. Set it in .env or environment."
        )

    base_url = settings.minimax_base_url or os.environ.get(
        "MINIMAX_BASE_URL", "https://api.minimaxi.com/v1"
    )

    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


# ============================================================
# Streaming variants — yield chunks as LLM produces them
# ============================================================
def _stream_ollama(prompt: str, model: str, max_tokens: int, temperature: float) -> Iterator[str]:
    """Ollama streaming via JSON-line response."""
    import json
    import requests
    url = settings.ollama_base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"num_predict": max_tokens, "temperature": temperature},
    }
    with requests.post(url, json=payload, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("done"):
                return
            text = chunk.get("response", "")
            if text:
                yield text


def _stream_openai(prompt: str, model: str, max_tokens: int, temperature: float) -> Iterator[str]:
    """OpenAI Chat Completions streaming."""
    from openai import OpenAI
    api_key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def _stream_anthropic(prompt: str, model: str, max_tokens: int, temperature: float) -> Iterator[str]:
    """Anthropic Messages streaming."""
    import anthropic
    api_key = settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic(api_key=api_key)
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            if text:
                yield text


def _stream_minimax(prompt: str, model: str, max_tokens: int, temperature: float) -> Iterator[str]:
    """MiniMax streaming via OpenAI-compatible API."""
    from openai import OpenAI
    api_key = settings.minimax_api_key or os.environ.get("MINIMAX_API_KEY", "")
    if not api_key:
        raise ValueError("MINIMAX_API_KEY not set")
    base_url = settings.minimax_base_url or os.environ.get(
        "MINIMAX_BASE_URL", "https://api.minimaxi.com/v1"
    )
    client = OpenAI(api_key=api_key, base_url=base_url)
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
