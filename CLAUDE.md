# DocuMind AI — CLAUDE.md

Project hướng dẫn cho Claude Code. Đọc file này trước khi làm bất cứ gì.

## Project overview

AI agent RAG nội bộ tra cứu tài liệu ngân hàng — quy định, biểu phí, sản phẩm (vay, thẻ,
tiền gửi), quy trình nghiệp vụ. Trả lời kèm trích dẫn nguồn, từ chối khi câu hỏi ngoài
phạm vi tài liệu đã nạp.

**Trạng thái hiện tại:** Đã hoàn thiện toàn diện bộ dữ liệu ngân hàng mẫu, compliance check, và test suite 100% pass:
- Corpus ngân hàng chuẩn gồm 6 văn bản quy định & nghiệp vụ tại `data/raw/banking_docs/` với `manifest.json`.
- Đã ingest hoàn chỉnh 36 chunks vào ChromaDB `data/chroma_db/`.
- Tiêu chí kiểm định tuân thủ ngân hàng `data/compliance/criteria.json` với 5 bộ quy tắc định lượng (thu nhập vay tín chấp, trần tỷ lệ DTI, trần hạn mức thẻ tín chấp, trần lãi suất không kỳ hạn và dưới 6 tháng theo NHNN).
- Bộ câu hỏi benchmark 25 câu tại `data/eval/test_questions.json`.
- Test suite: 84/84 tests passed (100%).

**Stack:**
- Backend: FastAPI + LangGraph agent + vector store qua `VECTOR_STORE_PROVIDER`
  (`chroma` mặc định local, hoặc `qdrant` cho Qdrant Cloud — xem `src/rag/vector_backend.py`)
- Embedding: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (local, `data/hf_cache/`)
- Retriever: Hybrid BM25 + dense vector + RRF fusion + cross-encoder reranker
- LLM: Groq Llama-3.3-70B (primary) → Gemini fallback
- Frontend: React + Vite (port 5174), proxies `/api` → backend port 8081

**Deploy (2026-07, theo default-tech-stack skill):**
- Local dev: như cũ, không đổi gì (`.\start.ps1`)
- Production: **chỉ Render** — frontend → Vercel (`frontend/vercel.json`, env `VITE_API_URL`),
  backend → Render (`render.yaml`, dùng chung Dockerfile hiện có)
- Đã xoá CI/CD Railway (`.github/workflows/deploy.yml`, `railway.toml`, `deploy_railway.ps1`,
  `scripts/start_railway.sh`) — không còn dùng Railway ở bất kỳ đâu trong repo
- Không dùng Supabase/Postgres — dự án không có bảng quan hệ nào (chỉ log JSONL)

## Corpus & Compliance

Corpus ngân hàng hiện có 6 văn bản chuẩn hoá cấu trúc `Điều ...`:
1. `01_thong_tu_39_2016_cho_vay.md`: Quy định cho vay & điều kiện cấp tín dụng.
2. `02_thong_tu_18_2024_the_ngan_hang.md`: Nghiệp vụ thẻ & hạn mức tín chấp tối đa 100tr.
3. `03_thong_tu_48_2018_tien_gui_tiet_kiem.md`: Tiền gửi tiết kiệm & quy tắc rút trước hạn.
4. `04_quyet_dinh_lai_suat_nhnn.md`: Trần lãi suất huy động và cho vay ưu tiên (QĐ 1124/1125).
5. `05_quy_che_cho_vay_tieu_dung_tin_chap.md`: Vay tiêu dùng tín chấp, thu nhập tối thiểu 5tr, DTI <= 60%.
6. `06_bieu_phi_dich_vu_tai_khoan_va_the.md`: Biểu phí duy trì tài khoản, thẻ, chính sách miễn lãi 45 ngày.

Lệnh ingest tài liệu:
```powershell
# Ingest tài liệu ngân hàng cùng manifest vào ChromaDB
python scripts/ingest_documents.py --source-dir data/raw/banking_docs --manifest data/raw/manifest.json --reset

# Xem trước số chunk mà không ghi vào DB
python scripts/ingest_documents.py --source-dir data/raw/banking_docs --dry-run
```

`data/compliance/criteria.json` định nghĩa các tiêu chí kiểm định tuân thủ định lượng (pass/fail) cho agent node `compliance_check`.
Đánh giá retrieval benchmark:
```powershell
python eval/run_evals.py --strategies dense rerank --retrieval-only --limit 5
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
    compliance.py     Compliance-check engine (pass/fail theo tiêu chí JSON hand-curated)
  agent/
    graph.py      LangGraph agent graph
    memory.py     ShortTermMemory (in-process, max 10 turns)
  config.py       Pydantic settings — đọc từ .env
  logger.py       Loguru setup

frontend/src/
  main.tsx        React SPA — chat, docs list, PDF upload tabs
  styles.css

scripts/
  ingest_documents.py        Ingest thư mục .md bất kỳ (chunk theo Điều/Khoản) vào ChromaDB
  rebuild_chroma_direct.py   Ingest từ data/raw/ (dùng cho JSON pháp luật)
  expand_corpus.py           Crawl thêm văn bản
  migrate_chroma_to_qdrant.py   Migrate corpus ChromaDB local → Qdrant Cloud
                                 (đọc embeddings có sẵn, không re-embed)

eval/
  rag_comparison.py  Benchmark pipeline (RAGAS + local metrics)
  run_evals.py       Entry point eval
  metrics.py         Local metrics: hit_rate, MRR, citation_rate, ooc_refusal_rate

logs/
  chat_history.jsonl   Log mỗi query/response (append, structured JSON; tự tạo lại khi chạy)
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

**ChromaDB:** Dùng local `PersistentClient` (không cần server). HTTP server ở `localhost:8000` thường không chạy — code tự fallback sang local. Corpus hiện đang rỗng — `data/chroma_db/` sẽ được tạo lại khi chạy ingest.

**Vite proxy:** `frontend/vite.config.ts` proxy `/api` → `http://localhost:8081`. Nếu đổi port backend phải cập nhật cả đây. Khi deploy tách domain (Vercel), frontend dùng `VITE_API_URL` thay vì proxy — xem `frontend/.env.example`.

**Vector store provider:** mặc định `chroma`. Đổi sang Qdrant Cloud: chạy
`python scripts/migrate_chroma_to_qdrant.py --verify`, rồi set `VECTOR_STORE_PROVIDER=qdrant`
+ `QDRANT_URL` + `QDRANT_API_KEY` trong `.env`. Toàn bộ code retrieval (main.py init, BM25
corpus load, health check, direct-query fallback) đi qua `src/rag/vector_backend.py` nên
không cần sửa gì thêm — nhưng cần chạy migrate lại sau khi có corpus ngân hàng thật (script
đọc embeddings có sẵn trong Chroma, không re-embed).

## Eval (RAGAS)

Chỉ chạy khi cần benchmark, không phải production. `data/eval/test_questions.json` đang rỗng
(bộ câu hỏi UNETI cũ đã xoá) — cần soạn lại bộ câu hỏi ngân hàng trước khi eval có ý nghĩa:
```powershell
python eval/run_evals.py --strategies dense hybrid --output reports/ragas_50q.json
```

RAGAS dùng Gemini làm judge — 3 keys × 5 models = 15 pairs, rate limit 3 req/61s/pair.
Checkpoint tự động lưu tại `reports/ragas_checkpoints/` sau mỗi 10 câu.
