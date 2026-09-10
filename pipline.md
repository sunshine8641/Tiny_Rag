# Tiny_RAG 总体规划

## 1. 项目目标（一句话）

**个人资料本地 RAG 工具**：把 PDF / 书 / 笔记 / 网页向量化本地存储，用自然语言检索，跟商用 LLM 协作回答。

---

## 2. 核心架构

```
┌──────────────────────────────────────────────────────────────┐
│                                                               │
│  ┌────────────┐      ┌────────────┐      ┌────────────┐      │
│  │  Ingestion │ ───► │  Storage   │ ◄─── │   Query    │      │
│  │            │      │            │      │            │      │
│  │  loaders   │      │  Chroma    │      │ retriever  │      │
│  │  chunker   │      │  + state   │      │ reranker   │      │
│  │  embedder  │      │            │      │ generator  │      │
│  └────────────┘      └────────────┘      └────────────┘      │
│        ▲                                         │             │
│        │                                         ▼             │
│      资料                                  ┌────────────┐      │
│   (PDF/MD/EPUB/URL)                        │ 商用 LLM   │      │
│                                            │  (云端)    │      │
│                                            └────────────┘      │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

**三块独立**：Ingestion（写入） / Storage（持久化） / Query（读取）。可以分别开发、测试、替换。

---

## 3. 完整文件树

```
Tiny_RAG/
├── README.md                      # 项目说明 + 快速开始
├── requirements.txt               # 依赖
├── .env.example                   # API key 模板（OPENAI_API_KEY 等）
│
├── tiny_rag/                      # 核心包
│   ├── __init__.py
│   ├── config.py                  # 全局配置（路径、模型、参数）
│   │
│   ├── ingest/                    # ─── 数据摄入 ───
│   │   ├── __init__.py
│   │   ├── loaders.py             # PDF / MD / TXT / EPUB / URL
│   │   ├── chunker.py             # 切分（按 token, 512/100）
│   │   ├── embedder.py            # bge-m3 embedding + 缓存
│   │   ├── store.py               # Chroma 封装（add / query / delete）
│   │   ├── metadata.py            # 元数据 schema + 提取
│   │   ├── sync.py                # 增量更新（mtime / hash）
│   │   └── pipeline.py            # 串联：load → chunk → embed → store
│   │
│   ├── query/                     # ─── 检索 + 生成 ───
│   │   ├── __init__.py
│   │   ├── retriever.py           # bi-encoder + Chroma 检索
│   │   ├── reranker.py            # bge-reranker-v2-m3
│   │   ├── context_builder.py     # 拼 prompt + 来源标注
│   │   ├── generator.py           # 调 OpenAI / Claude / Ollama
│   │   └── pipeline.py            # 串联：query → retrieve → rerank → generate
│   │
│   ├── utils/                     # ─── 工具 ───
│   │   ├── __init__.py
│   │   ├── text.py                # 文本处理（normalize / tokenize）
│   │   ├── files.py               # 文件 hash / 扫描
│   │   └── display.py             # 终端美化（rich）
│   │
│   └── cli/                       # ─── 命令行入口 ───
│       ├── __init__.py
│       └── main.py                # argparse: ingest / query / status / remove
│
├── data/                          # ─── 数据存储（gitignore）───
│   ├── chroma/                    # Chroma 持久化
│   ├── state.json                 # 增量同步状态
│   └── cache/                     # embedding cache
│
├── tests/                         # ─── 测试 ───
│   ├── test_loaders.py
│   ├── test_chunker.py
│   ├── test_ingest.py
│   ├── test_query.py
│   └── fixtures/                  # 测试用 PDF / MD
│
└── docs/                          # ─── 文档 ───
    ├── architecture.md
    └── usage.md
```

---

## 4. 数据流（一次完整 Ingestion）

```
PDF file
   ↓
loaders.load_pdf(path)
   ↓ 返回 list of pages: [{text, page_num, title}, ...]
   ↓
chunker.chunk_pages(pages, chunk_size=512, overlap=100)
   ↓ 返回 list of chunks: [{text, page, title, ...}, ...]
   ↓
embedder.embed(chunks)
   ↓ 返回 list of vectors: [[0.1, 0.2, ...], ...]
   ↓
store.upsert(collection, chunks, vectors)
   ↓ 写入 Chroma（id 唯一，重复 upsert 更新）
   ↓
sync.update_state(file_path, hash)
   ↓ 保存到 state.json（用于下次跳过）
```

**一次完整 Query**：

```
user query
   ↓
embedder.embed_query(query)
   ↓
retriever.search(query_vector, top_k=50)
   ↓
reranker.rerank(query, top_50, top_k=5)
   ↓
context_builder.build(query, top_5_chunks)
   ↓
  {
    "prompt": "基于以下资料回答：\n[1] ...\n[2] ...\n\n问题：...",
    "sources": [{file, page, title}, ...]
  }
   ↓
generator.generate(prompt, llm="gpt-4o")
   ↓
display: 答案 + "📖 来源：book.pdf p.42"
```

---

## 5. 关键技术选型

| 模块 | 选型 | 备选 | 理由 |
|---|---|---|---|
| **Embedding** | `BAAI/bge-m3` | bge-large-zh-v1.5 | 中文 SOTA，跨语言 |
| **Rerank** | `BAAI/bge-reranker-v2-m3` | ms-marco CE | 跟 bge-m3 配对 |
| **Vector DB** | Chroma (持久化) | Qdrant | 文件存储，开箱即用 |
| **LLM** | OpenAI / Claude API | Ollama 本地 | 看隐私 / 质量 |
| **PDF** | pymupdf | pdfplumber | 快，结构保留好 |
| **EPUB** | ebooklib | — | 标准库 |
| **URL** | requests + readability-lxml | — | 干净正文 |
| **切分** | langchain RecursiveCharTextSplitter | 手写 tiktoken | 简单 |
| **CLI** | argparse (内置) | click | 0 依赖 |
| **进度条** | tqdm | rich | 轻量 |
| **配置** | pydantic + .env | yaml | 类型安全 |

---

## 6. 模块依赖图

```
cli/main.py
    │
    ├──► ingest/pipeline.py ──► loaders.py
    │                       ├──► chunker.py
    │                       ├──► embedder.py
    │                       ├──► store.py
    │                       └──► sync.py
    │
    ├──► query/pipeline.py  ──► retriever.py ──► store.py
    │                       ├──► reranker.py
    │                       ├──► context_builder.py
    │                       └──► generator.py
    │
    └──► config.py (所有模块共享)
```

**单向依赖**：`cli → pipeline → 基础模块 → config`。无循环。

---

## 7. 开发路线（分阶段）

### Phase 1：核心 MVP（今天 - 明天）
**目标**：能 ingest 1 本 PDF + 能 query 拿到答案

```
□ 项目脚手架（目录结构 + requirements + config）
□ config.py（路径、模型、参数）
□ ingest/loaders.py（PDF only，先 pymupdf）
□ ingest/chunker.py（按段落切分）
□ ingest/embedder.py（bge-m3 + 简单 cache）
□ ingest/store.py（Chroma 持久化）
□ ingest/pipeline.py（串联）
□ query/retriever.py（Chroma search）
□ query/context_builder.py（拼 prompt）
□ query/generator.py（OpenAI / Claude）
□ query/pipeline.py（串联）
□ cli/main.py（ingest / query 两个命令）
```

**跑通 demo**：
```bash
python -m tiny_rag ingest ~/books/test.pdf
python -m tiny_rag query "这本书讲了什么" --llm gpt-4o
```

### Phase 2：多格式 + 增量（第 3-5 天）
```
□ Markdown / TXT loader
□ EPUB loader
□ URL loader
□ sync.py（mtime + hash 增量）
□ reranker.py（CE rerank）
□ cli/remove / status / list
□ 进度条 + 错误处理
```

### Phase 3：工程化（第 6-10 天）
```
□ tests/（核心模块单测）
□ README.md（完整文档）
□ .env 管理 API key
□ 配置文件支持（YAML）
□ 日志系统
□ 性能 benchmark
```

### Phase 4：GUI / 集成（可选，1-2 周+）
```
□ Tauri 桌面 app
□ 浏览器插件
□ Obsidian 集成
□ 全本地 LLM (Ollama)
```

---

## 8. 配置草案（`config.py`）

```python
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # 存储路径
    data_dir: Path = Path("./data")
    chroma_dir: Path = Path("./data/chroma")
    state_file: Path = Path("./data/state.json")
    
    # 模型
    embedding_model: str = "BAAI/bge-m3"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    llm_model: str = "gpt-4o-mini"
    
    # 切分
    chunk_size: int = 512
    chunk_overlap: int = 100
    
    # 检索
    top_k_retrieve: int = 50
    top_k_rerank: int = 5
    
    # 设备
    device: str = "auto"  # auto / cpu / mps / cuda
    
    class Config:
        env_file = ".env"

settings = Settings()
```

---

## 9. 关键设计决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 引 LangChain？ | **不引** | 手写 200 行够用，调试容易 |
| 引 LlamaIndex？ | **不引** | 同上 |
| Chroma 还是 Qdrant？ | **Chroma** | 0 配置，文件存储 |
| LangChain splitters？ | **用** | RecursiveCharTextSplitter 写得比我自己好 |
| bge-m3 必要吗？ | **是** | 中文 SOTA，多语言 |
| 一开始就 rerank？ | **不用** | Phase 1 不上，Phase 2 加 |
| GUI 选型 | Tauri > Electron | Tauri 轻量，Rust 后端 |

---

## 10. 第一个 PR 范围（今天能搞定）

**MVP 1.0**：PDF → Chroma → Query → OpenAI 答案

```python
# 用到的文件（最少）
config.py            # 50 行
ingest/loaders.py    # 30 行（PDF only）
ingest/chunker.py    # 40 行
ingest/embedder.py   # 30 行
ingest/store.py      # 40 行
ingest/pipeline.py   # 30 行
query/retriever.py   # 30 行（复用 store）
query/generator.py   # 50 行（OpenAI）
query/pipeline.py    # 30 行
cli/main.py          # 60 行
requirements.txt     # 10 行
```
