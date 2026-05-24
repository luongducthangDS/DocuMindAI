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

## Kiến trúc

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend                       │
│              (chat UI + upload + report export)             │
└───────────────────────┬─────────────────────────────────────┘
                        │ REST + WebSocket
┌───────────────────────▼─────────────────────────────────────┐
│                    FastAPI Backend                           │
│         Rate Limiting (slowapi) | CORS | GZip              │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              LangGraph Agent                         │   │
│  │  Router → [simple_qa | compare | summarize | report] │   │
│  │  Tools: search, summarize, compare, extract, pdf,    │   │
│  │         metadata                                     │   │
│  └──────────────────┬──────────────────────────────────┘   │
│                     │                                        │
│  ┌──────────────────▼──────────────────────────────────┐   │
│  │              RAG Pipeline                            │   │
│  │  Embedder (bge-m3) → Hybrid Search (BM25 + Dense)  │   │
│  │  → RRF Fusion → Cross-encoder Reranker             │   │
│  │  → Groq Llama-3.1-70B + Citation Formatter         │   │
│  └──────────────────┬──────────────────────────────────┘   │
│                     │                                        │
│  ┌──────────────────▼──────────────────────────────────┐   │
│  │  ChromaDB (Vector) + SQLite (Memory) + Redis (Cache) │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
         │ LangSmith traces │ Loguru structured logs
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
