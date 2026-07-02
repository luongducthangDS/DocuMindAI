# DocuMind AI — CLAUDE.md

Project hướng dẫn cho Claude Code. Đọc file này trước khi làm bất cứ gì.

## Project overview

Chatbot RAG nội bộ hỗ trợ sinh viên UNETI (Trường Đại học Kinh tế - Kỹ thuật Công nghiệp) tra cứu nội quy, học bổng, điểm rèn luyện, chuẩn đầu ra.

**Stack:**
- Backend: FastAPI + LangGraph agent + vector store qua `VECTOR_STORE_PROVIDER`
  (`chroma` mặc định local, hoặc `qdrant` cho Qdrant Cloud — xem `src/rag/vector_backend.py`)
- Embedding: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (local, `data/hf_cache/`)
- Retriever: Hybrid BM25 + dense vector + RRF fusion + cross-encoder reranker
- LLM: Groq Llama-3.3-70B (primary) → Gemini fallback
- Frontend: React + Vite (port 5174), proxies `/api` → backend port 8081

**Deploy (2026-07, theo default-tech-stack skill):**
- Local dev: như cũ, không đổi gì (`.\start.ps1`)
- Production: frontend → Vercel (`frontend/vercel.json`, env `VITE_API_URL`),
  backend → Render (`render.yaml`, dùng chung Dockerfile hiện có)
- CI/CD cũ (`.github/workflows/deploy.yml`) build Docker + deploy Railway (monolith,
  serve luôn React build) — vẫn còn, dùng nếu không muốn tách kiến trúc
- Không dùng Supabase/Postgres — dự án không có bảng quan hệ nào (chỉ log JSONL)

## Corpus hiện tại

7 văn bản UNETI, tổng 91 chunks trong ChromaDB (`data/chroma_db/`), chunk theo Điều/Khoản:
- QĐ-740: Quy định học, kiểm tra & chuẩn đầu ra ngoại ngữ (15 chunks)
- QĐ-747: Quy chế đánh giá điểm rèn luyện sinh viên (19 chunks)
- QĐ-748: Quy định học bổng khuyến khích học tập (11 chunks)
- QĐ-670: Quy định chuẩn đầu ra tin học (8 chunks)
- QĐ-828: Quy định hướng dẫn & đánh giá khóa luận tốt nghiệp (18 chunks)
- QĐ-1228: Phương thức đánh giá chuẩn đầu ra chương trình đào tạo (7 chunks)
- QĐ-853: Quy định ngoại ngữ tiếng Anh cho sinh viên chính quy (13 chunks)

Nguồn MD: `D:\Projects for CV\chatbot_uneti_final\source\{pdf,scan}\md\*.md`

Để ingest lại hoặc thêm tài liệu mới:
```powershell
python scripts/ingest_uneti_md.py --reset
```

## Chạy local

**Một lệnh duy nhất (mở 2 cửa sổ):**
```powershell
.\start.ps1
```
Mở 2 terminal riêng — backend + frontend. Chờ backend in `Application startup complete` (~25s) rồi mở UI.

**Thủ công (2 terminal):**
```powershell
# Terminal 1 — Backend (chạy trước)
uvicorn src.api.main:app --host 0.0.0.0 --port 8081 --reload

# Terminal 2 — Frontend (chờ backend sẵn sàng rồi mới chạy)
cd frontend
npm run dev
```

Mở `http://localhost:5174`. API docs: `http://localhost:8081/docs`.

## Cấu trúc thư mục quan trọng

```
src/
  api/          FastAPI app, routes (query, documents, health, reports)
  rag/
    embedder.py       SentenceTransformer embedder + Chroma/Qdrant client factories
    vector_backend.py Provider-agnostic vector store access (count/fetch/query) —
                       đổi VECTOR_STORE_PROVIDER không cần sửa call site nào khác
    retriever.py      Hybrid retriever + reranker, RetrievedChunk dataclass
    generator.py      LLM generation (Groq primary / Gemini fallback), score filtering
  agent/
    graph.py      LangGraph agent graph
    memory.py     ShortTermMemory (in-process, max 10 turns)
  config.py       Pydantic settings — đọc từ .env
  logger.py       Loguru setup

frontend/src/
  main.tsx        React SPA — chat, docs list, PDF upload tabs
  styles.css

scripts/
  ingest_uneti.py    Ingest UNETI pre-chunked JSON vào ChromaDB
  rebuild_chroma_direct.py   Ingest từ data/raw/ (dùng cho JSON pháp luật)
  expand_corpus.py   Crawl thêm văn bản
  migrate_chroma_to_qdrant.py   Migrate corpus ChromaDB local → Qdrant Cloud
                                 (đọc embeddings có sẵn, không re-embed)

eval/
  rag_comparison.py  Benchmark pipeline (RAGAS + local metrics)
  run_evals.py       Entry point eval
  metrics.py         Local metrics: hit_rate, MRR, citation_rate, ooc_refusal_rate

logs/
  chat_history.jsonl   Log mỗi query/response (append, structured JSON)
```

## Config quan trọng (.env)

- `GOOGLE_API_KEY`, `GOOGLE_API_KEY_2`, `GOOGLE_API_KEY_3` — 3 Gemini keys
- `GEMINI_JUDGE_MODELS` — danh sách model cho RAGAS eval (phân cách bằng dấu phẩy)
- `PRIMARY_LLM=groq/llama-3.3-70b-versatile`
- `FALLBACK_LLM=gemini/gemini-2.5-flash-lite`
- `EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- `API_PORT=8081`
- **Không được** thêm `HF_HOME` vào `.env` — pydantic `extra_forbidden` sẽ reject

## Các lưu ý kỹ thuật

**HuggingFace cache:** Model nằm ở `data/hf_cache/`. Hệ thống có `G:\My Drive\HF_Cache_Models` (Google Drive, thường offline) được set làm `HF_HOME` trong system env. Mọi script cần force-set tất cả HF env vars sang local trước khi import `sentence_transformers`:
```python
for k in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "SENTENCE_TRANSFORMERS_HOME"):
    os.environ[k] = str(local_hf_path)
```

**Score filtering:** `generator.py` lọc chunk có `score < 0.05` trước khi gọi LLM. Nếu tất cả chunk dưới ngưỡng → trả về message "không tìm thấy" + gợi ý upload, không gọi LLM, không show nguồn.

**LLM temperature:** `0.0` (không phải 0.1) để citation ổn định giữa các lần chạy.

**ChromaDB:** Dùng local `PersistentClient` (không cần server). HTTP server ở `localhost:8000` thường không chạy — code tự fallback sang local.

**Vite proxy:** `frontend/vite.config.ts` proxy `/api` → `http://localhost:8081`. Nếu đổi port backend phải cập nhật cả đây. Khi deploy tách domain (Vercel), frontend dùng `VITE_API_URL` thay vì proxy — xem `frontend/.env.example`.

**Vector store provider:** mặc định `chroma`. Đổi sang Qdrant Cloud: chạy
`python scripts/migrate_chroma_to_qdrant.py --verify`, rồi set `VECTOR_STORE_PROVIDER=qdrant`
+ `QDRANT_URL` + `QDRANT_API_KEY` trong `.env`. Toàn bộ code retrieval (main.py init, BM25
corpus load, health check, direct-query fallback) đi qua `src/rag/vector_backend.py` nên
không cần sửa gì thêm. Đã test roundtrip với dữ liệu thật (91 chunks) — top-1 khớp tuyệt đối
giữa Chroma và Qdrant.

## Eval (RAGAS)

Chỉ chạy khi cần benchmark, không phải production:
```powershell
python eval/run_evals.py --strategies dense hybrid --output reports/ragas_50q.json
```

RAGAS dùng Gemini làm judge — 3 keys × 5 models = 15 pairs, rate limit 3 req/61s/pair.
Checkpoint tự động lưu tại `reports/ragas_checkpoints/` sau mỗi 10 câu.
