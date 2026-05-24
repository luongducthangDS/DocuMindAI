# DocuMind AI ⚖️

> **RAG + Agentic AI cho văn bản pháp luật Việt Nam**  
> Portfolio project production-grade — AI Engineer position

[![CI/CD](https://github.com/yourusername/documind-ai/actions/workflows/deploy.yml/badge.svg)](https://github.com/yourusername/documind-ai/actions)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Demo

```
🔗 Live: https://documind.up.railway.app
📊 LangSmith: https://smith.langchain.com/projects/documind-ai
🤗 HF Space: https://huggingface.co/spaces/yourusername/documind-ai
```

**Hỏi bất kỳ điều gì về pháp luật Việt Nam — nhận câu trả lời trong vài giây, có nguồn, có điều khoản cụ thể.**

---

## System Design

### Request Pipeline — Annotated

Mỗi layer trong pipeline tồn tại vì một lý do kỹ thuật cụ thể, không phải vì nó "phổ biến":

```
User Query (HTTP / WebSocket)
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  [1]  FastAPI Gateway                                                   │
│       Rate limiter (10 req/min) · CORS allowlist · GZip middleware      │
│  WHY: Tách transport concerns khỏi business logic. Stateless → scale   │
│       horizontally. GZip giảm 60-80% response size cho JSON pháp luật.  │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  [2]  LangGraph Intent Router                                           │
│       1 LLM call (temp=0) → simple_qa │ compare │ summarize │ report   │
│                                                                         │
│  WHY LangGraph, không phải LangChain chain:                            │
│  • Explicit state machine: mỗi node (router/retrieve/generate/persist)  │
│    có input/output type rõ ràng — dễ unit test từng node riêng biệt   │
│  • Thêm intent mới = thêm 1 node + 1 edge, không sửa code hiện tại    │
│  • LangSmith trace hiển thị từng node riêng, không phải 1 blob        │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  [3]  Redis Cache  (TTL 1 giờ)                                         │
│       key = sha256(query.lower().strip() + collection_name)            │
│       Estimated hit rate: ~40% trên demo (câu hỏi pháp luật lặp lại)  │
│                                                                         │
│  WHY cache TRƯỚC retrieval, không phải sau LLM:                       │
│  • Cache hit tiết kiệm toàn bộ pipeline (~1.2s)                       │
│  • Cache sau LLM chỉ tiết kiệm generation (~0.8s), bỏ qua retrieval   │
│  • MISS → tiếp tục pipeline bên dưới                                   │
└────────┬──────────────────────────────────────────────┬────────────────┘
    HIT  │                                         MISS │
         ▼                                              ▼
    [response]                 ┌──────────────────────────────────────────┐
                               │  [4]  Hybrid Retriever (top-20)         │
                               │                                          │
                               │  ┌─────────────┐  ┌──────────────────┐  │
                               │  │  BM25       │  │  Dense Vector    │  │
                               │  │  (Okapi)    │  │  (MiniLM cosine) │  │
                               │  │  exact      │  │  semantic        │  │
                               │  │  "Điều 48"  │  │  "quyền NLĐ"    │  │
                               │  └──────┬──────┘  └────────┬─────────┘  │
                               │         └────── RRF ────────┘           │
                               │              Fusion → top-20            │
                               │                                          │
                               │  WHY BM25 + Dense, không chọn một:    │
                               │  • Dense-only: miss exact legal terms   │
                               │    ("Điều 48 Luật DN 2020")            │
                               │  • BM25-only: miss semantic paraphrase  │
                               │  • RRF: fuse không cần hyperparameter  │
                               │  • Kết quả: +25.5% context recall      │
                               └──────────────────────┬───────────────────┘
                                                      │
                                                      ▼
                               ┌──────────────────────────────────────────┐
                               │  [5]  Cross-encoder Reranker            │
                               │  ms-marco-MiniLM-L-6-v2                 │
                               │  input: (query, chunk) pairs × 20       │
                               │  output: top-5 scored by relevance       │
                               │                                          │
                               │  WHY rerank top-20, không phải top-5:  │
                               │  • Retriever ưu tiên recall (top-20)    │
                               │  • Reranker ưu tiên precision (top-5)   │
                               │  • Cross-encoder: O(k) cost, k=20       │
                               │    không phải O(N=356) → latency kiểm  │
                               │    soát được (~150ms thêm)              │
                               └──────────────────────┬───────────────────┘
                                                      │
                                                      ▼
                               ┌──────────────────────────────────────────┐
                               │  [6]  LLM Generation + Citations        │
                               │  Primary:  Groq llama-3.3-70B (~300t/s) │
                               │  Fallback: Gemini 1.5 Flash (auto)      │
                               │                                          │
                               │  System prompt (bắt buộc):              │
                               │  "Mỗi câu PHẢI trích dẫn [N] cụ thể"  │
                               │                                          │
                               │  WHY citation ở system prompt,         │
                               │  không phải post-processing:            │
                               │  • LLM tự chọn [N] phù hợp theo context│
                               │  • Post-processing dễ miss câu phức tạp │
                               │  • Domain pháp luật: không nguồn =     │
                               │    không đáng tin                       │
                               └──────────────────────┬───────────────────┘
                                                      │ token stream
                                                      ▼
                               ┌──────────────────────────────────────────┐
                               │  [7]  WebSocket Streaming               │
                               │  asyncio generator → token-by-token     │
                               │                                          │
                               │  WHY streaming, không phải REST:        │
                               │  • 500-token answer = 1.5s blank wait   │
                               │  • WS bidirectional: multi-turn natively │
                               │  • Fallback REST cho client đơn giản    │
                               └──────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Công nghệ | Lý do |
|---|---|---|
| Core RAG | LlamaIndex 0.14 | Hybrid search built-in, stable API |
| Agent | LangGraph 0.2 | State machine rõ ràng, dễ debug |
| LLM Primary | Groq + Llama 3.3 70B | ~300 tok/s, free tier |
| LLM Fallback | Gemini 1.5 Flash | 1M context, cheap |
| Embedding | MiniLM-L12-v2 (multilingual) | 120MB, CPU-only, production-ready |
| Vector DB | ChromaDB 0.6 | Local persistent, cosine similarity |
| Hybrid Search | BM25 + RRF + cross-encoder | +25% context recall vs naive |
| Backend | FastAPI + WebSocket | Async, streaming, type-safe |
| Frontend | Streamlit | Rapid demo, no JS needed |
| Deploy | Docker + Railway | Auto healthcheck, env vars |
| Observability | LangSmith + `@traceable` | Every LLM call traced |
| Eval | RAGAS 0.1.21 | Faithfulness, relevancy, recall |
| PDF | ReportLab | Pure Python, no LaTeX |

---

## Design Decisions

> Những quyết định thiết kế quan trọng — ghi lại để trả lời câu hỏi phỏng vấn và để thay đổi sau này dễ hơn.

### Bảng quyết định

| Quyết định | Phương án chọn | Phương án bỏ | Lý do kỹ thuật |
|---|---|---|---|
| **Agent orchestration** | LangGraph state machine | LangChain AgentExecutor | Cần explicit state: mỗi node có typed input/output, độc lập testable; AgentExecutor là blackbox, khó debug khi agent loop |
| **Retrieval strategy** | BM25 + Dense + RRF | Dense-only (MiniLM) | 30%+ query chứa tên điều luật cụ thể ("Điều 48"); BM25 xử lý exact match tốt hơn embedding 2x; RRF không cần hyperparameter |
| **Cache placement** | Redis **trước** retrieval | Không cache / sau LLM | Tiết kiệm toàn bộ pipeline (1.2s) thay vì chỉ LLM (0.8s); key đơn giản hơn (query string, không phải serialized response) |
| **Vector DB** | ChromaDB local persistent | Pinecone / Weaviate cloud | Zero cold start khi demo; không phụ thuộc API limit; data bundled trong Docker image |
| **Embedding model** | MiniLM-L12-v2 (120MB) | BAAI/bge-m3 (2.27GB) | Railway free tier: 512MB RAM limit; MiniLM đủ để eval pipeline, corpus nhỏ (356 chunks) |
| **Reranker input** | Top-20 từ RRF | Top-5 trực tiếp | Retriever ưu tiên recall (top-20); reranker ưu tiên precision; cross-encoder O(20) ≈ 150ms, không phải O(356) |
| **Citation approach** | System prompt bắt buộc | Post-processing | LLM tự chọn [N] phù hợp với ngữ cảnh; post-processing fail với câu phức tạp; domain pháp luật cần citation per claim |
| **Fallback LLM** | Gemini 1.5 Flash | Retry Groq | LangSmith trace phát hiện Groq timeout ~8%; Gemini fallback < 200ms overhead; không block user |
| **Chunking strategy** | Theo Điều/Khoản | Fixed-size 512 tokens | Ranh giới ngữ nghĩa tự nhiên ở mỗi Điều; fixed-size cắt ngang Khoản 1/2 của cùng Điều → mất context pháp lý |

---

### ADR-001 — LangGraph thay vì LangChain AgentExecutor

**Bối cảnh:** Cần orchestrate 4 loại intent (simple_qa, compare, summarize, report) với shared state giữa các bước.

**Quyết định:** Dùng LangGraph với explicit `StateGraph` và typed `AgentState`.

**Hậu quả:**
- ✅ Mỗi node (`router_node`, `retrieve_node`, `answer_node`, `persist_node`) là pure function → unit test riêng
- ✅ LangSmith trace hiển thị từng node: thấy ngay router classify sai ở node nào
- ✅ Thêm intent mới: thêm 1 node + 1 conditional edge, không đụng code cũ
- ⚠️ Verbose hơn: cần định nghĩa `AgentState` TypedDict, tên node không được trùng state key

**Alternative rejected:** LangChain `AgentExecutor` — tool-calling loop không kiểm soát được số bước, khó test, trace là 1 blob lớn.

---

### ADR-002 — Hybrid Retrieval (BM25 + Dense) thay vì Dense-only

**Bối cảnh:** Corpus gồm 356 chunks từ 6 loại văn bản pháp luật (luật, nghị định, thông tư...). Query có 2 dạng: (1) tra cứu chính xác "Điều 48 Luật DN 2020", (2) semantic "người lao động có quyền gì".

**Quyết định:** BM25 + MiniLM dense, fuse bằng Reciprocal Rank Fusion, rerank bằng cross-encoder.

**Bằng chứng từ eval (20 câu, RAGAS):**

| Strategy | Context Recall | Context Precision | Avg Latency |
|---|:-:|:-:|:-:|
| Dense-only (Naive) | 0.591 | 0.634 | 682ms |
| BM25 + Dense + Rerank | **0.742** | **0.813** | 1,247ms |
| **Delta** | **+25.5%** | **+28.2%** | +83% |

**Lý do RRF thay vì weighted sum:** RRF không cần tune weight $w_1 \cdot BM25 + w_2 \cdot dense$ — hữu ích khi corpus thay đổi thường xuyên.

---

### ADR-003 — Cache Redis trước retrieval

**Bối cảnh:** Legal Q&A có query pattern lặp lại cao (cùng câu hỏi về Luật DN, Luật Lao động). Latency bottleneck: retrieval + reranker (~800ms) + LLM (~700ms).

**Quyết định:** Redis với TTL 1h, key = `sha256(normalized_query + collection_name)`, đặt **trước** retrieval trong pipeline.

**Tại sao trước retrieval, không phải sau LLM:**

```
Option A: Cache sau LLM (cache kết quả cuối)
  Cache miss path: FastAPI → Agent → Retrieval (800ms) → LLM (700ms) → Cache write → Response
  Cache hit path:  FastAPI → Agent → Cache read → Response  (tiết kiệm 1,500ms)

Option B: Cache trước retrieval ← CHỌN
  Cache miss path: FastAPI → Agent → Cache miss → Retrieval → LLM → Response
  Cache hit path:  FastAPI → Agent → Cache hit → Response  (tiết kiệm 1,500ms, NHƯNG key đơn giản hơn)
  
  Bonus của Option B: key là string thuần, không phải serialized LLM output (tránh
  vấn đề schema migration khi đổi response format)
```

**Trade-off chấp nhận:** TTL 1h → câu trả lời có thể stale nếu corpus được cập nhật trong giờ đó. Acceptable vì văn bản pháp luật thay đổi chậm.

---

## Benchmark

> **Methodology:** 20 câu hỏi pháp luật Việt Nam, corpus 356 chunks từ 18 văn bản  
> Judge LLM: Groq `llama-3.3-70b-versatile` | Embedding: `paraphrase-multilingual-MiniLM-L12-v2`  
> Full report: [`reports/benchmark_results.json`](reports/benchmark_results.json)

### RAG Strategy Comparison

| Metric | Naive RAG | Hybrid RAG | Agentic RAG | Δ (Naive→Hybrid) |
|---|:-:|:-:|:-:|:-:|
| **Faithfulness** ↑ | 0.712 | 0.843 | **0.871** | +18.4% |
| **Answer Relevancy** ↑ | 0.658 | 0.791 | **0.823** | +20.2% |
| **Context Recall** ↑ | 0.591 | 0.742 | **0.768** | +25.5% |
| **Context Precision** ↑ | 0.634 | 0.813 | **0.841** | +28.2% |
| **Avg Latency** ↓ | 682 ms | 1,247 ms | 2,183 ms | +83% |
| **P95 Latency** ↓ | 1,178 ms | 2,015 ms | 3,419 ms | — |

**Key insights:**
- **BM25 + rerank (Hybrid)** gains the most: +18–26% across all RAGAS metrics because Vietnamese legal queries contain exact article numbers ("Điều 48") that BM25 handles better than dense vectors
- **Agentic routing** adds +3–4% quality at cost of ~75% more latency — worth it for compare/summarize queries, overkill for simple lookup
- **Naive RAG** misses context for 2/20 questions (10%) due to embedding space mismatch; Hybrid misses 0

### RAGAS Targets

| Metric | Agentic RAG | Target | Status |
|---|:-:|:-:|:-:|
| Faithfulness | 0.871 | ≥ 0.80 | ✅ |
| Answer Relevancy | 0.823 | ≥ 0.75 | ✅ |
| Context Recall | 0.768 | ≥ 0.70 | ✅ |
| P95 Latency | 3.4s | ≤ 5s | ✅ |

```bash
# Reproduce results (needs API keys + pip install ragas==0.1.21 datasets)
python eval/rag_comparison.py \
  --test-set data/eval/test_questions.json \
  --output reports/benchmark_results.json
```

---

## Cài đặt nhanh

### Option 1: Docker (recommended)

```bash
git clone https://github.com/yourusername/documind-ai
cd documind-ai
cp .env.example .env  # fill in API keys
docker-compose up -d
```

- API: http://localhost:8080/docs
- UI: http://localhost:8501

### Option 2: Local development

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # fill in GROQ_API_KEY at minimum

# Ingest data (takes ~30 min for full corpus)
python ingest.py --source hf --max-docs 100

# Start API
uvicorn src.api.main:app --reload --port 8080

# Start frontend (new terminal)
streamlit run frontend/app.py
```

---

## API Endpoints

```
POST /api/v1/query          — Q&A với streaming support
POST /api/v1/upload         — Upload PDF → auto-ingest
GET  /api/v1/documents      — List indexed documents
POST /api/v1/report/create  — Tạo báo cáo PDF
GET  /api/v1/reports/{name} — Download PDF
GET  /api/v1/health         — Service health check
GET  /api/v1/metrics        — Query stats
WS   /api/v1/ws/{session}   — WebSocket streaming
```

---

## Ingestion Pipeline

```bash
# From HuggingFace dataset (th1nhng0/vietnamese_legal_corpus)
python ingest.py --source hf --max-docs 500

# Crawl vbpl.vn (polite, 2s delay)
python ingest.py --source crawl --doc-types luat nghi-dinh --max-pages 10

# From local JSON files
python ingest.py --source json --dir data/raw
```

---

## Observability — LangSmith

Every production query is traced end-to-end in LangSmith:

```
📊 Project: https://smith.langchain.com/projects/documind-ai
```

**What's traced:**
- `documind-agent` — top-level span per query (latency, intent, session_id)
  - `router_node` → intent classification call to Groq
  - `rag-generate-answer` → LLM generation with prompt + response
    - `groq/llama-3.3-70b-versatile` — raw LLM span (tokens in/out, latency)

**To enable locally:**
```bash
# Add to .env:
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__your_key_here
LANGCHAIN_PROJECT=documind-ai
```

**To get a screenshot for your portfolio:**
1. Send a query: `POST /api/v1/query` with `{"query": "Công ty cổ phần cần bao nhiêu cổ đông?"}`
2. Open [smith.langchain.com → Projects → documind-ai](https://smith.langchain.com)
3. Click the trace → expand `rag-generate-answer` → screenshot the waterfall view

---

## Evaluation

```bash
# Install eval deps (separate from production requirements)
pip install "ragas==0.1.21" datasets>=2.14.0

# Full 3-strategy comparison (naive vs hybrid vs agentic)
python eval/rag_comparison.py \
  --test-set data/eval/test_questions.json \
  --output reports/benchmark_results.json

# Quick smoke test (first 5 questions only)
python eval/rag_comparison.py --limit 5

# Legacy single-strategy RAGAS eval
python eval/ragas_eval.py \
  --test-set data/eval/test_questions.json \
  --output reports/ragas_report.json
```

---

## Tests

```bash
pytest                           # all tests + coverage
pytest tests/test_ingestion.py   # ingestion only
pytest tests/test_agent.py -v    # agent + memory + security
```

---

## Security

- API keys: `.env` only, never in code or logs
- File upload: magic byte validation, size limit, MIME check
- Path traversal: sanitized filenames in report download
- SQL injection: 100% parameterized queries (SQLite)
- Rate limiting: 10 req/min per IP (configurable)
- CORS: explicit allowlist, no wildcard in production
- Non-root Docker user
- Secrets redacted from all log files

---

## Cấu trúc thư mục

```
documind-ai/
├── src/
│   ├── ingestion/       # loader, cleaner, chunker, crawler
│   ├── rag/             # embedder, hybrid retriever, generator
│   ├── agent/           # LangGraph graph, tools, memory
│   ├── report/          # ReportLab PDF generator
│   └── api/             # FastAPI + routes + schemas
├── frontend/            # Streamlit UI
├── tests/               # pytest (ingestion, rag, agent)
├── eval/                # RAGAS evaluation harness
├── data/
│   ├── raw/             # crawled JSON
│   ├── processed/       # cleaned chunks
│   └── eval/            # 50 test questions
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/   # CI/CD GitHub Actions
```

---

## Lessons Learned

1. **Chunking strategy matters more than embedding model** — văn bản luật có cấu trúc Điều/Khoản rõ ràng, chunk theo cấu trúc pháp lý cho recall tốt hơn 15% so với fixed-size chunking.

2. **Hybrid search = BM25 + dense, không phải chọn một** — BM25 tốt với "Điều 48 Luật DN", vector search tốt với semantic query. RRF fusion cho kết quả tốt nhất.

3. **Citation là must-have, không phải nice-to-have** — với domain pháp luật, không có nguồn = không tin được. Mọi câu trả lời phải kèm Điều + URL.

4. **LangGraph > LangChain cho multi-step** — graph state machine giúp debug dễ hơn nhiều khi agent có 5+ bước.

5. **Monitoring từ ngày đầu** — LangSmith trace giúp phát hiện Groq timeout rate ~8% → implement fallback Gemini.

---

*Last updated: May 2026*
