# DocuMind AI — CLAUDE.md

Project hướng dẫn cho Claude Code. Đọc file này trước khi làm bất cứ gì.

## Project overview

AI agent RAG tra cứu **pháp luật lao động và bảo hiểm xã hội Việt Nam** — Bộ luật Lao động,
Luật BHXH, Luật Việc làm cùng nghị định/thông tư hướng dẫn. Trả lời kèm trích dẫn điều khoản,
từ chối khi câu hỏi ngoài phạm vi tài liệu đã nạp, và tra cứu được **theo thời điểm**
(`as_of_date`) vì nhiều văn bản trong corpus đã bị thay thế.

> **Phạm vi sản phẩm là nguồn sự thật, không phải văn bản tự do.** Khai báo ở
> `src/config.py` (`DOMAIN_NAME` / `DOMAIN_SCOPE` / `DOMAIN_TOPICS`) và
> `frontend/src/main.tsx` (`PRODUCT`). Mọi prompt, message từ chối, nhãn UI phải lấy từ đó —
> đừng viết lại chuỗi mô tả phạm vi ở chỗ khác.

**Trạng thái hiện tại:**
- Corpus lao động/BHXH: 20 văn bản tại `data/raw/lao_dong/` (+ `_versions/` cho bản sửa đổi
  cấp khoản). Corpus ngân hàng cũ đã bỏ — `data/raw/banking_docs/` không còn tồn tại.
- ChromaDB `data/chroma_db/` (collection `documind_legal`): **1151 chunks**, 3072-dim.
- `data/compliance/criteria.json`: 6 tiêu chí định lượng lao động (trần làm thêm giờ
  năm/tháng, thời gian thử việc, lương thử việc 85%, nghỉ hằng năm, lương tối thiểu vùng I).
  Mọi tiêu chí phải trích dẫn văn bản CÓ trong `data/raw/lao_dong/` — có test chặn
  (`tests/test_compliance.py::TestCriteriaDataIntegrity`).
- Gold set: `data/eval/temporal_questions.json` (30 câu, viết tay trước khi chạy hệ thống).
  Bộ 25 câu ngân hàng tiền-pivot đã archive sang `data/eval/_archive/`.
- Bộ câu khó: `data/eval/hard_questions.json` (19 câu, 6 loại: ghép nhiều điều khoản, ngày
  giao thời, ngoài phạm vi, tiền đề sai, tình huống tuân thủ, tra bảng). Claude soạn từ
  nguyên văn corpus, `reviewed: false` — **cần người duyệt**. Bộ 30 câu temporal đã bão hoà
  (1.000), dùng bộ này để đo cải tiến:
  `python eval/temporal_eval.py --gold data/eval/hard_questions.json --output reports/hard_eval.json`
- **Gold set chính: `data/eval/legal_qa_200.json` (199 câu)** = 30 temporal + 19 hard + 150 câu
  Claude soạn (`g2_*`, `reviewed: false`). Kiểm tra với index thật:
  `python eval/validate_gold.py data/eval/legal_qa_200.json`. Metric: recall@1/3/5/8, MRR,
  faithfulness (`--judge <arms>`), p50/p95; `--mlflow <run>` log vào MLflow (`mlflow.db`, gitignored);
  `--embed-cache` chỉ dùng cho CI (latency khi đó không phải số thật). Kết quả: README mục Đánh giá.
- CI: `.github/workflows/ci.yml` — pytest + retrieval eval trên Qdrant (reranker tắt như Render),
  gate bằng `eval/ci_gate.py` so với `reports/eval_baseline.json`. Cần secrets `GOOGLE_API_KEY*`,
  `QDRANT_URL`, `QDRANT_API_KEY`.
- **Chroma Cloud** (database `DocuMind`, biến `CHROMA_CLOUD_*`): chỉ là bản sao để xem dashboard
  và so provider — app KHÔNG đọc từ đó. Đồng bộ: `python scripts/copy_chroma_to_cloud.py`; so
  sánh: `python eval/provider_bench.py` (kết quả README mục Đánh giá). Gọi qua REST
  (`src/rag/chroma_cloud.py`) vì `chromadb` 0.6.3 không tương thích server Cloud 1.x.
- **Reranker local đang KHÔNG chạy** (2026-09-23): Windows Smart App Control chặn DLL `pyarrow`
  → `sentence_transformers` không import được → retriever log "Reranker unavailable" và bỏ qua.
  Report nào ghi rerank phải xem `meta.reranker_active`, không tin `ENABLE_RERANKER`.
- Test suite: 284/284 tests passed (đo 2026-09-23).

**Stack:**
- Backend: FastAPI + LangGraph agent + vector store qua `VECTOR_STORE_PROVIDER`
  (`chroma` mặc định local, hoặc `qdrant` cho Qdrant Cloud — xem `src/rag/vector_backend.py`)
- Embedding: `gemini-embedding-001` (3072-dim) qua Gemini Embedding API — **không nạp
  model nào vào RAM**. Model đọc từ `EMBEDDING_MODEL`, không hard-code. Lịch sử: chạy
  model local `AITeamVN/Vietnamese_Embedding` (1024-dim) tới 2026-09-19.
- Retriever: Hybrid BM25 + dense vector + RRF fusion + cross-encoder reranker
  (reranker `BAAI/bge-reranker-v2-m3` vẫn chạy LOCAL — ngoại lệ duy nhất)
- LLM: **chỉ Gemini** (2026-09-19). Mọi lời gọi đi qua
  `src.rag.generator.gemini_generate`, xoay vòng (3 key × các model trong
  `GEMINI_GENERATION_MODELS`); hết mọi cặp thì trích nguyên văn nguồn, không LLM.
  Đã gỡ tuyệt đối Groq, OpenAI-compatible, embedding local và HF Inference API.
- Frontend: React + Vite (port 5174), proxies `/api` → backend port 8081

**Deploy (2026-07, theo default-tech-stack skill):**
- Local dev: như cũ, không đổi gì (`.\start.ps1`)
- Production: **chỉ Render** — frontend → Vercel (`frontend/vercel.json`, env `VITE_API_URL`),
  backend → Render (`render.yaml`, dùng chung Dockerfile hiện có)
- Đã xoá CI/CD Railway (`.github/workflows/deploy.yml`, `railway.toml`, `deploy_railway.ps1`,
  `scripts/start_railway.sh`) — không còn dùng Railway ở bất kỳ đâu trong repo
- Không dùng Supabase/Postgres — dự án không có bảng quan hệ nào (chỉ log JSONL)

## Corpus & Compliance

Corpus tại `data/raw/lao_dong/` — 20 văn bản, frontmatter YAML (`doc_id`, `so_hieu`, `ten`,
`ngay_hieu_luc`, `trang_thai`), thân bài chuẩn hoá theo `Điều ...`; `_versions/` chứa bản
sửa đổi ở cấp khoản. Bốn nhóm:
1. **Lao động**: `45-2019-QH14` (Bộ luật Lao động), `18-VBHN-VPQH`, `145-2020-ND-CP`, `10-2020-TT-BLDTBXH`.
2. **Lương tối thiểu**: `293-2025-ND-CP` (hiệu lực 01/01/2026, vùng I 5.310.000đ) thay `74-2024-ND-CP` (4.960.000đ).
3. **BHXH**: `41-2024-QH15` thay `58-2014-QH13`; nghị định/thông tư `115-2015`, `134-2015`, `158-2025`, `159-2025`, `59-2015-TT-BLDTBXH`, `11`/`12-2025-TT-BNV`.
4. **Việc làm & BHTN**: `74-2025-QH15` thay `38-2013-QH13`; `374-2025-ND-CP`, `28-2015-ND-CP`. Thêm `135-2020-ND-CP` (tuổi nghỉ hưu).

Cả bản cũ lẫn bản mới đều nằm trong index — đó là điều làm `as_of_date` có ý nghĩa.

Lệnh ingest tài liệu:
```powershell
# Ingest corpus lao động vào ChromaDB.
# BẮT BUỘC có --manifest: thiếu cờ này script rơi về chế độ quét thư mục, nuốt luôn
# _versions/ thành tài liệu độc lập (chunk rác thay vì 1151).
python scripts/ingest_documents.py --source-dir data/raw/lao_dong `
  --manifest docs/corpus/corpus_manifest.yaml --reset

# Xem trước số chunk mà không ghi vào DB
python scripts/ingest_documents.py --source-dir data/raw/lao_dong `
  --manifest docs/corpus/corpus_manifest.yaml --dry-run
```

**Access control / multi-tenant / point-in-time (DEC-0003):** mọi truy hồi đi qua
`RetrievalContext` (`src/rag/context.py`) — tenant + nhãn ACL + `as_of_date` compile thành
filter đẩy **xuống** vector store trước khi search. Bốn field metadata bắt buộc:
`tenant_id`, `acl_label`, `effective_from_i`, `effective_to_i` (ngày dạng int YYYYMMDD —
Chroma 0.6.3 không so sánh `$lte` trên chuỗi). Chunk thiếu 4 field này **biến mất khỏi mọi
kết quả**, nên sau khi ingest lại corpus cũ phải chạy:

```powershell
python scripts/backfill_access_meta.py          # xem trước
python scripts/backfill_access_meta.py --yes    # ghi
```

Tenant demo khai ở `data/tenants/tenants.json`; client gửi `x-api-key`, **không bao giờ**
tự khai `acl_labels` (server tra ra). Không header = ẩn danh, chỉ thấy `tenant_id=public`.
UI web **không** có ô nhập key (đã gỡ 2026-09-23) — luôn chạy ẩn danh/`public`; tenant
chỉ dùng qua API. WebSocket: trình duyệt không gắn được header lên handshake, nên client
gửi key trong message (`{"query": ..., "api_key": ...}`), server resolve theo từng lượt. Header vẫn
dùng được cho client không phải trình duyệt. `GET /api/v1/whoami` cho biết key ứng với
tenant nào. CORS phải giữ `X-API-Key` trong `allow_headers` (`src/api/main.py`).
`src/rag/temporal.py` vẫn chạy làm lớp lọc thứ hai — đừng gỡ.

`data/compliance/criteria.json` định nghĩa 6 tiêu chí kiểm định tuân thủ định lượng
(pass/fail) cho agent node `compliance_check`. Engine này phát ✅/❌ kèm trích dẫn nên là
đường đi thẳng tới "câu trả lời sai có thẩm quyền" — khi thêm tiêu chí:
- Số liệu phải tra từ chính văn bản trong `data/raw/lao_dong/`, không lấy từ trí nhớ.
- `so_hieu` phải trỏ tới văn bản CÓ trong corpus (test `TestCriteriaDataIntegrity` chặn).
- `keywords` chấm theo tổng ĐỘ DÀI cụm khớp, ngưỡng `_MIN_KEYWORD_SCORE = 6`. Tránh cụm
  quá rộng (`"trả lương"`, `"/tháng"`) — chúng kéo nhầm tiêu chí khác.
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
    generator.py      LLM generation (chỉ Gemini, xoay key×model), score filtering
    compliance.py     Compliance-check engine (pass/fail theo tiêu chí JSON hand-curated)
  agent/
    graph.py      LangGraph agent graph
    memory.py     ShortTermMemory (cache in-process, max 10 turns, persist SQLite
                  qua session_id — sống qua được restart/redeploy)
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
- `GEMINI_GENERATION_MODELS` — danh sách model Gemini cho generation (xoay vòng với 3 key)
- `EMBEDDING_MODEL=gemini-embedding-001` — nguồn sự thật duy nhất cho model
  embedding. Đổi giá trị này **bắt buộc** chạy `python scripts/reembed_corpus.py --yes`;
  collection mang nhãn model đã index, lệch nhãn là `EmbeddingModelMismatch` lúc mở store.
- `API_PORT=8081`
- **Không được** thêm `HF_HOME` vào `.env` — pydantic `extra_forbidden` sẽ reject
- `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` (hoặc `LANGFUSE_HOST`,
  cả hai tên đều được đọc) — tracing tuỳ chọn, rỗng thì tắt hẳn. Xem `src/langfuse_otel.py`:
  gửi OTLP/HTTP thuần qua `requests`, KHÔNG cài SDK `langfuse` chính thức (xung đột version
  `opentelemetry-api/sdk` với `chromadb` trong venv này — đã thử thật, phá 17 test). Một trace
  = một lượt `run_agent()`/1 câu hỏi WS, span con cho retrieve (`retriever`), mọi lời gọi Gemini
  (`generation`, kèm model/token/cost), compliance-check (`tool`).

## Các lưu ý kỹ thuật

**HuggingFace cache:** Model nằm ở `data/hf_cache/`. Hệ thống có `G:\My Drive\HF_Cache_Models` (Google Drive, thường offline) được set làm `HF_HOME` trong system env. Mọi script cần force-set tất cả HF env vars sang local trước khi import `sentence_transformers`:
```python
for k in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "SENTENCE_TRANSFORMERS_HOME"):
    os.environ[k] = str(local_hf_path)
```

**Score filtering:** `generator.py` lọc chunk có `score < 0.05` trước khi gọi LLM. Nếu tất cả chunk dưới ngưỡng → trả về message "không tìm thấy" + gợi ý upload, không gọi LLM, không show nguồn.

**LLM temperature:** `0.0` (không phải 0.1) để citation ổn định giữa các lần chạy.

**ChromaDB:** Dùng local `PersistentClient` (không cần server). HTTP server ở `localhost:8000` thường không chạy — code tự fallback sang local. Collection `documind_legal` đang có 1151 chunks (3072-dim, `gemini-embedding-001`), kèm metadata `embedding_model`/`embedding_dim` để phát hiện lệch model.

**Đổi model embedding:** sửa `EMBEDDING_MODEL` trong `.env` → `python scripts/reembed_corpus.py --yes` (re-embed tại chỗ, giữ nguyên chunk + metadata temporal, tự sao lưu collection cũ sang `documind_legal__backup_<model cũ>`) → `pytest -q` + `python eval/temporal_eval.py`. So sánh ứng viên trước khi đổi: `python eval/embedding_ab.py --models current <model-moi>`.

**Cache HuggingFace:** mọi entrypoint gọi `use_local_hf_cache()` từ `src/hf_env.py` — không tự set biến env HF nữa (trước đây 8 bản sao, 4 biến thể, đã gây lỗi thật). Riêng embedder truyền `cache_folder` thẳng làm tham số vì `huggingface_hub` đóng băng đường dẫn cache ngay lúc import, mọi thao tác env sau đó là quá muộn.

**Vite proxy:** `frontend/vite.config.ts` proxy `/api` → `http://localhost:8081`. Nếu đổi port backend phải cập nhật cả đây. Khi deploy tách domain (Vercel), frontend dùng `VITE_API_URL` thay vì proxy — xem `frontend/.env.example`.

**Vector store provider:** mặc định `chroma`. Đổi sang Qdrant Cloud: chạy
`python scripts/migrate_chroma_to_qdrant.py --verify`, rồi set `VECTOR_STORE_PROVIDER=qdrant`
+ `QDRANT_URL` + `QDRANT_API_KEY` trong `.env`. Toàn bộ code retrieval (main.py init, BM25
corpus load, health check, direct-query fallback) đi qua `src/rag/vector_backend.py` nên
không cần sửa gì thêm — nhưng cần chạy migrate lại mỗi khi corpus đổi (script
đọc embeddings có sẵn trong Chroma, không re-embed).

## Eval (RAGAS)

Chỉ chạy khi cần benchmark, không phải production. Gold set đang dùng:
`data/eval/temporal_questions.json` (30 câu lao động, có `source_clause`). Bộ 25 câu ngân
hàng tiền-pivot nằm ở `data/eval/_archive/` — không đo được gì trên corpus này:
```powershell
python eval/run_evals.py --strategies dense hybrid --output reports/ragas_50q.json
```

RAGAS dùng Gemini làm judge — 3 keys × 5 models = 15 pairs, rate limit 3 req/61s/pair.
Checkpoint tự động lưu tại `reports/ragas_checkpoints/` sau mỗi 10 câu.
