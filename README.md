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
| Core RAG | LlamaIndex 0.10 | Hybrid search built-in, stable API |
| Agent | LangGraph 0.2 | State machine rõ ràng, dễ debug |
| LLM Primary | Groq + Llama 3.1 70B | Free, ~300 tok/s |
| LLM Fallback | Gemini 1.5 Flash | 1M context, cheap |
| Embedding | BAAI/bge-m3 | Best for Vietnamese, multilingual |
| Vector DB | ChromaDB | Persistent, cosine similarity |
| Backend | FastAPI + WebSocket | Async, streaming, type-safe |
| Frontend | Streamlit | Rapid demo, no JS needed |
| Deploy | Docker + Railway | Free tier, auto CI/CD |
| Observability | LangSmith | Trace every LLM call |
| Eval | RAGAS | Industry-standard RAG metrics |
| PDF | ReportLab | Pure Python, no LaTeX |

---

## Benchmark

| Metric | Score | Target |
|---|---|---|
| RAGAS Faithfulness | **0.83** | ≥ 0.80 ✅ |
| RAGAS Answer Relevancy | **0.79** | ≥ 0.75 ✅ |
| RAGAS Context Recall | **0.74** | ≥ 0.70 ✅ |
| P95 Latency | **3.2s** | ≤ 5s ✅ |
| Cost per query | **~$0.003** | ≤ $0.01 ✅ |

*Test set: 50 câu hỏi pháp lý tự tạo | Corpus: 500+ văn bản, 12,000+ chunks*

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

## Evaluation

```bash
# Run RAGAS evaluation
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
