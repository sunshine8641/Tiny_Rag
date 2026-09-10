# Tiny_RAG

> 个人知识库 RAG 工具 — 把书 / 笔记 / 文章向量化本地存储，用自然语言检索，跟 LLM 协作回答。

## 特性

- **本地优先**：所有数据 + embedding 都在本地
- **隐私友好**：只有 query 和 top-5 chunk 发给 LLM
- **多 LLM 后端**：OpenAI / Anthropic / Ollama / Mock
- **结构保留**：PDF 表格、标题、列表都识别（pymupdf4llm）
- **多语言**：bge-m3 中文 SOTA embedding
- **可选 rerank**：bge-reranker-v2-m3 提精度（默认关闭）

## 快速开始

### 1. 安装

```bash
git clone <repo>
cd Tiny_RAG
pip install -r requirements.txt
pip install -e .   # 让 tiny_rag 可以 import
```

### 2. 配置 LLM（任选一个）

```bash
# 复制模板
cp .env.example .env

# 编辑 .env，设其中一个：
OPENAI_API_KEY=sk-...
# 或
ANTHROPIC_API_KEY=sk-ant-...
# 或装 Ollama + qwen2.5:7b（本地免费）
brew install ollama
ollama serve &
ollama pull qwen2.5:7b
```

### 3. 入库

```bash
# 单文件
python -m tiny_rag ingest ~/books/my_book.pdf

# 整个目录
python -m tiny_rag ingest ~/books --recursive

# 整个目录的 Markdown
python -m tiny_rag ingest ~/notes --pattern "*.md" --recursive
```

### 4. 查询

```bash
# Mock 模式（无 API key 也能演示）
python -m tiny_rag query "什么是 X" --backend mock

# Ollama 本地
python -m tiny_rag query "什么是 X" --backend ollama --model qwen2.5:7b

# OpenAI
python -m tiny_rag query "什么是 X" --backend openai --model gpt-4o-mini
```

### 5. 管理

```bash
python -m tiny_rag status      # 看库状态
python -m tiny_rag remove <path>  # 删某个文件的所有 chunks
```

## 命令

### `python -m tiny_rag ingest <path>`

```
位置参数:
  path              文件或目录

选项:
  --pattern PATTERN  glob 模式（默认 *.pdf）
  --recursive        递归子目录
```

### `python -m tiny_rag query "<question>"`

```
位置参数:
  query             问题（任意语言）

选项:
  --top-k N         检索 N 个 chunk（默认 5）
  --backend BACKEND mock | openai | anthropic | ollama
  --model MODEL     模型名（默认从 settings.llm_model 推断）
```

### `python -m tiny_rag status`

显示总 chunk 数、数据目录、embedding 模型、LLM、所有 source 文件。

### `python -m tiny_rag remove <file_path>`

删除指定文件的所有 chunks（按 source path 匹配）。

## 项目结构

```
Tiny_RAG/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env.example
│
├── tiny_rag/
│   ├── __init__.py
│   ├── __main__.py              # python -m tiny_rag 入口
│   ├── config.py                # 全局配置（pydantic settings）
│   │
│   ├── ingest/                  # 数据摄入
│   │   ├── loaders.py           # pymupdf4llm PDF 解析
│   │   ├── chunker.py           # bge-m3 tokenizer + 切分
│   │   ├── embedder.py          # bge-m3 + L2 normalize
│   │   ├── store.py             # Chroma 0.6.3 持久化
│   │   └── pipeline.py          # ingest_file / ingest_directory
│   │
│   ├── query/                   # 检索 + 生成
│   │   ├── retriever.py         # bi-encoder + Chroma
│   │   ├── reranker.py          # bge-reranker-v2-m3（可选）
│   │   ├── context_builder.py   # 拼 prompt + 来源标注
│   │   ├── generator.py         # 4 个 LLM backend
│   │   └── pipeline.py          # 端到端 answer_query
│   │
│   └── cli/
│       └── main.py              # argparse 4 个子命令
│
├── data/                        # 数据存储（gitignore）
│   ├── chroma/                  # Chroma 持久化
│   └── cache/                   # HF 模型缓存
│
└── tests/
    └── fixtures/                # 测试 PDF
```

## 数据流

### 摄入

```
PDF file
  ↓ loaders.load_pdf()          [pymupdf4llm → Markdown]
  ↓ chunker.chunk_pages()       [bge-m3 tokenizer, 512 tokens, overlap 100, normalize whitespace]
  ↓ embedder.embed_chunks()     [bge-m3 → 1024-dim L2-normalized]
  ↓ store.add_chunks()          [Chroma upsert]
  ↓ data/chroma/                [持久化]
```

### 查询

```
user query
  ↓ embedder.embed_query()      [bge-m3 → 1024-dim]
  ↓ retriever.retrieve()        [Chroma cosine, top-K]
  ↓ [reranker.rerank()]         [可选, 默认关闭]
  ↓ context_builder.build()     [拼 prompt, 标 [1][2]...]
  ↓ generator.generate()         [OpenAI/Ollama/etc]
  ↓ answer + sources
```

## 配置（`tiny_rag/config.py`）

通过环境变量覆盖（`TINY_RAG_*` 前缀）或 `.env` 文件：

| 字段 | 默认 | 说明 |
|---|---|---|
| `data_dir` | `./data` | 数据目录 |
| `chroma_dir` | `./data/chroma` | Chroma 持久化目录 |
| `embedding_model` | `BAAI/bge-m3` | Embedding 模型 |
| `rerank_model` | `BAAI/bge-reranker-v2-m3` | Rerank 模型 |
| `llm_model` | `gpt-4o-mini` | LLM 模型 |
| `chunk_size` | 512 | Token 数（bge-m3 tokenizer）|
| `chunk_overlap` | 100 | 重叠 tokens |
| `top_k_retrieve` | 50 | 检索候选数 |
| `top_k_rerank` | 5 | Rerank 后保留数 |
| `device` | `auto` | `auto` / `cpu` / `mps` / `cuda` |

## 模块细节

### `loaders.py` — PDF 解析

用 **pymupdf4llm** 把 PDF 转 Markdown，保留：
- 标题（# ## ###）
- 列表（- 1.）
- 表格（Markdown 表格语法）
- 多栏阅读顺序

不识别公式语义、扫描版 PDF（需要 OCR）。

### `chunker.py` — 切分

- **Tokenizer**：bge-m3 的 AutoTokenizer（不是 tiktoken），保证 chunk 切出来精确 ≤ 512 bge-m3 tokens
- **Splitter**：`RecursiveCharacterTextSplitter`，按 `\n\n` `。` `.` 优先级切分
- **Normalize**：3+ 连续 `\n + 任意空白` → `\n\n`（清理 PDF 排版残留）
- **过滤**：strip 后空白的 page / chunk 跳过
- **不跨页**：每页独立切分，保留 page metadata

### `embedder.py` — Embedding

- **Model**：bge-m3（1024 维，多语言）
- **L2 normalize**：cosine sim = dot product（检索更快）
- **Device 默认 CPU**（M3 + MPS + bge-m3 有 NaN/OOM 问题）

### `store.py` — 持久化

- **Chroma 0.6.3** + `PersistentClient`
- **Cosine distance**（bge-m3 向量已 L2 norm）
- **where filter workaround**：Chroma 0.6.3 的 `query` API 跟 `get` API 对 where 处理不一致，先取 top-K*5 再 Python 端 filter

### `generator.py` — LLM 调用

| Backend | 需要什么 |
|---|---|
| `mock` | 无（返回占位符）|
| `ollama` | Ollama 服务跑着 + 装了对应模型 |
| `openai` | `OPENAI_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |

Backend 不传时，根据 model 名自动推断：
- `gpt*` → openai
- `claude*` → anthropic
- `qwen*` / `llama*` / `mistral*` → ollama

## 已知问题

- **Chroma 0.6.3 vs 0.6.2 / 1.x**：升级可能让 HNSW 索引失真。**升级必须清库重建**。
- **bge-m3 + MPS**：M3 Mac 上跑 bge-m3 有 NaN/OOM 问题，默认强制 CPU。
- **增量同步未实现**：文件改了/删了，需要手动 `ingest`（会重算 embedding）或 `remove`。
- **仅 PDF loader**：Markdown / EPUB / URL loader 计划中。
- **`use_rerank=False` 默认**：rerank 模型 568M，每次 ~10-20s CPU 推理。默认关闭，按需开。

## Roadmap

- [ ] 增量同步（state.json + file hash）
- [ ] Markdown / TXT / EPUB / URL loader
- [ ] bge-reranker 默认开 + batch 推理加速
- [ ] console_script 入口（`tiny_rag` 直接当命令）
- [ ] Web UI（Tauri 桌面 / 浏览器插件）
- [ ] 多模态（图片 OCR / 公式识别）

## 调试技巧

```bash
# 看库里有什么
python -m tiny_rag status

# 用 mock 测流程（不调 API）
python -m tiny_rag query "X" --backend mock

# 看 prompt 长啥样（mock 模式会回显 prompt 头）
python -m tiny_rag query "X" --backend mock 2>&1 | head -50

# Python 里直接用
from tiny_rag.query.pipeline import answer_query
result = answer_query("X", backend="ollama", model="qwen3:8b")
print(result["answer"])
print(result["sources"])
```

## 性能参考（M3 Mac mini）

| 操作 | 耗时 |
|---|---|
| 首次加载 bge-m3（CPU）| ~15s |
| 后续加载（cache hit）| ~2s |
| Embed 100 chunks | ~3s |
| Chroma 检索 100 candidates | ~50ms |
| bge-reranker 20 pairs | ~10-20s |
| Ollama qwen3:8b 单次回答 | ~3-5s |
| 端到端 query（无 rerank）| ~5-10s |
| 端到端 query（带 rerank）| ~30-50s |

## License

Personal use.
