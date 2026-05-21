# DocuMind AI — Development Log

> Ghi lại toàn bộ quá trình triển khai, các vấn đề gặp phải và cách giải quyết.  
> Tác giả: Lương Đức Thắng | Bắt đầu: 21/05/2026

---

## Mục lục

1. [Tổng quan dự án](#1-tổng-quan-dự-án)
2. [Quá trình triển khai theo giai đoạn](#2-quá-trình-triển-khai-theo-giai-đoạn)
3. [Vấn đề gặp phải và cách giải quyết](#3-vấn-đề-gặp-phải-và-cách-giải-quyết)
4. [Quyết định kỹ thuật quan trọng](#4-quyết-định-kỹ-thuật-quan-trọng)
5. [Bài học rút ra](#5-bài-học-rút-ra)
6. [Trạng thái hiện tại](#6-trạng-thái-hiện-tại)
7. [Việc cần làm tiếp theo](#7-việc-cần-làm-tiếp-theo)

---

## 1. Tổng quan dự án

**DocuMind AI** là hệ thống hỏi đáp pháp luật Việt Nam kết hợp:
- **RAG** (Retrieval-Augmented Generation): tìm kiếm hybrid (vector + BM25) + reranker
- **LangGraph Agent**: multi-step reasoning, 6 tools, conversation memory
- **FastAPI** backend với WebSocket streaming
- **Streamlit** frontend

**Mục tiêu:** Portfolio project production-grade để apply vị trí AI Engineer.

---

## 2. Quá trình triển khai theo giai đoạn

### Giai đoạn 1 — Đọc plan và thiết kế cấu trúc

- Đọc `DocuMind_AI_Project_Plan.md`, xác định 7 tuần công việc
- Tạo toàn bộ cấu trúc thư mục theo plan:
  ```
  src/ingestion/, src/rag/, src/agent/, src/report/, src/api/routes/
  frontend/, tests/, eval/, data/, .github/workflows/
  ```
- Viết `src/config.py` với `pydantic-settings` để load `.env` an toàn

### Giai đoạn 2 — Ingestion Pipeline

Viết theo thứ tự: `cleaner.py` → `chunker.py` → `loader.py` → `crawler.py`

**Chiến lược chunking:**
- Chunk theo Điều/Khoản (cấu trúc pháp lý) thay vì fixed-size
- Regex: `r'(Điều\s+\d+[a-z]?[\.\:]\s+.*?)(?=Điều\s+\d+[a-z]?[\.\:]|$)'`
- Fallback sliding-window cho văn bản không có cấu trúc Điều

**Crawler vbpl.vn:**
- Delay 2 giây/request (lịch sự với server)
- Validate URL phải thuộc `vbpl.vn` trước khi fetch (ngăn SSRF)
- Sanitize filename để tránh path traversal

### Giai đoạn 3 — RAG Pipeline

- **Embedder**: `BAAI/bge-m3` cho production, fallback `paraphrase-multilingual-MiniLM-L12-v2`
- **Retriever**: QueryFusionRetriever (dense + BM25) với RRF, cross-encoder reranker
- **Generator**: Groq primary → Gemini fallback, streaming via `AsyncIterator`

### Giai đoạn 4 — LangGraph Agent

Graph topology:
```
START → router → [do_retrieve → do_answer | do_compare | do_summarize | do_report] → do_persist → END
```

6 tools: `search_legal_docs`, `summarize_document`, `compare_documents`,
`extract_articles`, `generate_pdf_report`, `get_document_metadata`

### Giai đoạn 5 — API & Frontend

- FastAPI với rate limiting (slowapi), CORS, GZip middleware
- WebSocket streaming endpoint `/api/v1/ws/{session_id}`
- Pydantic v2 schemas với input validation và security guards
- Streamlit UI với REST + WebSocket dual-mode

### Giai đoạn 6 — Tests, Docker, CI/CD

- pytest với fixtures, mocks, security tests (SQL injection, path traversal, XSS)
- Multi-stage Dockerfile với non-root user
- GitHub Actions: lint → test → build image → deploy Railway

---

## 3. Vấn đề gặp phải và cách giải quyết

### Vấn đề 1: API Key để nhầm chỗ — BẢO MẬT NGHIÊM TRỌNG

**Ngày:** 21/05/2026  
**Mô tả:** User điền Groq API key và Google API key thật vào file `.env.example` thay vì `.env`.  
File `.env.example` thường được commit lên git → lộ key công khai.

**Giải quyết:**
1. Tạo file `.env` đúng với keys thật (file này có trong `.gitignore`)
2. Khôi phục `.env.example` về placeholder `gsk_xxxx...`
3. Đảm bảo `.gitignore` đã có entry cho `.env`

**Lesson:** Luôn kiểm tra `.gitignore` trước khi `git add`. Không bao giờ để secret trong file example.

---

### Vấn đề 2: Groq model bị khai tử

**Ngày:** 21/05/2026  
**Mô tả:** Model `llama-3.1-70b-versatile` đã bị Groq decommission.

```
groq.BadRequestError: The model `llama-3.1-70b-versatile` has been decommissioned
```

**Giải quyết:** Thay bằng `llama-3.3-70b-versatile` (model kế thừa tương đương).  
Cập nhật tất cả 5 file có reference:
- `src/rag/generator.py`
- `src/agent/graph.py`
- `src/agent/tools.py`
- `src/config.py`
- `.env`

**Lesson:** Pin model version nhưng cần monitoring deprecation notices từ provider.

---

### Vấn đề 3: groq 0.9.0 không tương thích với httpx mới

**Ngày:** 21/05/2026  
**Mô tả:** `groq==0.9.0` dùng argument `proxies` đã bị xóa trong `httpx>=0.28`:

```
TypeError: Client.__init__() got an unexpected keyword argument 'proxies'
```

**Giải quyết:** Nâng `groq>=0.11.0` trong requirements.txt.

---

### Vấn đề 4: `.env` có ký tự Unicode gây lỗi encoding Windows

**Ngày:** 21/05/2026  
**Mô tả:** File `.env` dùng box-drawing characters (`═══`, `──`) trong comments.  
`slowapi` dùng `starlette.Config` để đọc `.env` với encoding mặc định Windows (cp1252).  
cp1252 không encode được `═` (U+2550).

```
UnicodeDecodeError: 'charmap' codec can't decode byte 0x90 in position 4
```

**Giải quyết:** Viết lại `.env` và `.env.example` chỉ dùng ký tự ASCII thuần (`#`, `-`, `=`).

**Lesson:** File config (`.env`, `.ini`) phải ASCII-only để tránh encoding issues cross-platform.

---

### Vấn đề 5: LangGraph node name trùng với state key

**Ngày:** 21/05/2026  
**Mô tả:** LangGraph 0.2 không cho phép node name trùng với `AgentState` key.

```
ValueError: 'answer' is already being used as a state key
```

`AgentState` có key `"answer"` và code đặt tên node cũng là `"answer"`.

**Giải quyết:** Đổi tất cả node names thêm prefix `do_`:
- `"answer"` → `"do_answer"`
- `"retrieve"` → `"do_retrieve"`
- `"compare"` → `"do_compare"`
- `"summarize"` → `"do_summarize"`
- `"report"` → `"do_report"`
- `"persist"` → `"do_persist"`

Cập nhật `route_by_intent()` và `build_graph()` tương ứng.

**Lesson:** Khi dùng LangGraph, đọc kỹ constraint về naming — node names ≠ state keys.

---

### Vấn đề 6: ChromaDB version conflict

**Ngày:** 21/05/2026  
**Mô tả:** Requirements pin `chromadb==0.5.7` nhưng pip cài `chromadb==1.5.9` (latest).  
`llama-index-vector-stores-chroma` yêu cầu `chromadb>=0.5.17`.  
ChromaDB 1.5.9 thay đổi `PersistentClient` — yêu cầu HTTP server ngay cả cho local mode.

```
ValueError: Could not connect to a Chroma server. Are you sure it is running?
```

**Giải quyết:**
1. Tìm version tương thích: `chromadb>=0.5.17,<1.0.0`
2. pip cài `chromadb==0.6.3` — vẫn dùng local `PersistentClient` không cần server
3. Cập nhật requirements.txt: `chromadb>=0.5.17,<1.0.0`

**Lesson:** Luôn pin upper bound cho major versions khi dependency có breaking changes.

---

### Vấn đề 7: `PersistentClient` fail trong async context

**Ngày:** 21/05/2026  
**Mô tả:** `chromadb.PersistentClient` hoạt động khi test trực tiếp nhưng fail khi gọi từ FastAPI lifespan (async context).  
Nguyên nhân: ChromaDB dùng socket operations trong `__init__` → conflict với asyncio event loop.

**Giải quyết hai bước:**
1. Bỏ `settings=ChromaSettings(...)` khi gọi `PersistentClient` — để chromadb tự chọn local mode
2. Wrap toàn bộ `_init_rag()` trong `asyncio.to_thread()`:
   ```python
   await asyncio.to_thread(_init_rag_sync)
   ```

**Lesson:** Mọi blocking I/O trong FastAPI async lifespan phải chạy trong thread pool.

---

### Vấn đề 8: LlamaIndex version mismatch

**Ngày:** 21/05/2026  
**Mô tả:** Requirements pin `llama-index==0.10.65` nhưng pip cài `llama-index-core==0.14.22`.  
Sub-packages (`llama-index-embeddings-huggingface==0.2.3`) không tương thích với core 0.14.x.

```
ImportError: No module named 'llama_index.embeddings.huggingface'
```

**Giải quyết:** Bỏ pin version cứng, dùng:
```
llama-index-core>=0.12.0
llama-index-embeddings-huggingface
llama-index-vector-stores-chroma
llama-index-retrievers-bm25
```
Để pip tự resolve compatible versions.

---

### Vấn đề 9: `persist_node` trả về dict rỗng

**Ngày:** 21/05/2026  
**Mô tả:** LangGraph yêu cầu mỗi node phải return ít nhất 1 state key.

```
ValueError: Must write to at least one of ['messages', 'query', 'intent', ...]
```

`persist_node` ban đầu `return {}`.

**Giải quyết:**
```python
return {"error": state.get("error")}  # pass-through, không thay đổi giá trị
```

---

### Vấn đề 10: `retrieve_node` import biến không tồn tại

**Ngày:** 21/05/2026  
**Mô tả:** Code cũ import `_retriever` (tên ban đầu) nhưng module đã đổi thành `_active_retriever`:

```python
from src.rag.retriever import _retriever as global_retriever  # ImportError
```

**Giải quyết:** Xóa dòng import sai, dùng `getattr(r_module, "_active_retriever", None)` đã có sẵn ở dưới.

---

### Vấn đề 11: HuggingFace dataset không tồn tại

**Ngày:** 21/05/2026  
**Mô tả:** Dataset `th1nhng0/vietnamese_legal_corpus` trong plan không còn trên HuggingFace Hub.

```
DatasetNotFoundError: Dataset 'th1nhng0/vietnamese_legal_corpus' doesn't exist
```

**Giải quyết:** Tạo 3 file JSON mẫu từ văn bản pháp luật thực tế:
- `luat_doanh_nghiep_2020.json` — Luật DN 2020 (7 articles, 7 chunks)
- `bo_luat_lao_dong_2019.json` — Bộ luật Lao động 2019 (5 articles, 5 chunks)
- `luat_thue_tndn.json` — Luật Thuế TNDN (3 articles, 3 chunks)

Ingest bằng: `python ingest.py --source json --dir data/raw` → **15 chunks**

**Lesson:** Khi build project phụ thuộc external dataset, có backup plan với sample data.

---

### Vấn đề 12: Port 8080 bị chiếm sẵn

**Ngày:** 21/05/2026  
**Mô tả:** Port 8080 đã bị EnterpriseDB PostgreSQL chiếm trên máy local.

**Giải quyết:** Đổi `API_PORT=8081` trong `.env`.

---

### Vấn đề 13: Windows console không encode tiếng Việt

**Ngày:** 21/05/2026  
**Mô tả:** Khi print kết quả tiếng Việt ra Windows terminal (cp1252):

```
UnicodeEncodeError: 'charmap' codec can't encode character 'ấ'
```

**Giải quyết trong test script:**
```python
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
```

**Giải quyết dài hạn:** Server trả JSON UTF-8 đúng, lỗi chỉ ở terminal display — không ảnh hưởng API.

---

### Vấn đề 14: BM25 retriever nhận `nodes=[]` — không index được corpus

**Ngày:** 21/05/2026  
**Mô tả:** Server khởi động thành công nhưng mọi query đều nhận 0 chunks từ BM25.  
Log: `"BM25 init failed, using vector-only: Please pass exactly one of index, nodes, or docstore"`

**Nguyên nhân:** `_init_rag_sync()` gọi `build_hybrid_retriever(index, nodes=[], rerank=True)` — pass list rỗng thay vì nodes thực từ ChromaDB.

**Giải quyết:**
```python
# api/main.py — trước khi build retriever, load nodes từ collection
existing_nodes = _load_nodes_from_collection(collection)
r_module._active_retriever = build_hybrid_retriever(index, nodes=existing_nodes, rerank=True)
```

Hàm `_load_nodes_from_collection()` dùng `collection.get(include=["documents", "metadatas"])` để lấy toàn bộ documents, convert sang `TextNode` list.

---

### Vấn đề 15: ChromaDB schema KeyError: `_type`

**Ngày:** 21/05/2026  
**Mô tả:** Sau khi upgrade chromadb, `list_collections()` raise:
```
KeyError: '_type'
```

**Nguyên nhân:** Collection `test` được tạo từ session khác với `config_json_str='{}'` — JSON rỗng, không có field `_type` mà chromadb 0.6.3 yêu cầu khi đọc lại.

**Giải quyết:** Xóa collection lỗi trực tiếp từ SQLite:
```python
conn.execute('DELETE FROM segments WHERE collection=?', (broken_id,))
conn.execute('DELETE FROM collections WHERE id=?', (broken_id,))
```

**Lesson:** Khi gặp schema mismatch trong embedded DB, kiểm tra SQLite trực tiếp trước khi assume library bug.

---

## 4. Quyết định kỹ thuật quan trọng

### 4.1 Chunking theo Điều thay vì fixed-size
**Lý do:** Văn bản luật có ranh giới ngữ nghĩa tự nhiên ở mỗi Điều. Fixed-size có thể cắt ngang giữa Khoản 1 và Khoản 2 của cùng một Điều → mất context pháp lý.

### 4.2 `asyncio.to_thread()` cho RAG init
**Lý do:** ChromaDB, sentence-transformers đều có blocking socket/disk I/O trong `__init__`. Chạy trong thread pool tránh block event loop → server không bị treo trong startup.

### 4.3 ChromaDB local persistent thay vì Docker-only
**Lý do:** Giúp developer chạy local không cần Docker. Production vẫn dùng HTTP server qua docker-compose. Code tự detect và fallback.

### 4.4 Embedding model nhỏ cho dev, lớn cho prod
**Lý do:** `bge-m3` (2.3GB) quá nặng để download mỗi lần setup. Dùng `paraphrase-multilingual-MiniLM-L12-v2` (~500MB) cho local dev đủ nhanh để test pipeline.

### 4.5 LangGraph node naming với prefix `do_`
**Lý do:** Tránh conflict với `AgentState` keys. Convention `do_<action>` rõ ràng hơn về intent của node.

### 4.6 API keys chỉ trong `.env`, không trong code
**Lý do:** Security best practice. `pydantic-settings` tự load từ `.env`, không cần `os.getenv()` rải rác trong code.

---

## 5. Bài học rút ra

### Security
- **Không bao giờ để secret trong file example** — `.env.example` chỉ chứa placeholder
- **File config phải ASCII-only** — tránh encoding issues cross-platform
- **Validate URL trước khi fetch** — ngăn SSRF attacks trong crawler
- **Parameterized SQL queries** — không bao giờ string-format SQL

### Dependency Management
- **Pin major version upper bound** — `chromadb>=0.5.17,<1.0.0`
- **Test compatibility trước khi pin** — chạy nhanh `pip install` xem conflict
- **Kiểm tra deprecation** — model names, API signatures thay đổi nhanh trong AI ecosystem

### LangGraph
- **Node names ≠ State keys** — đặt convention ngay từ đầu (prefix `do_`)
- **Mọi node phải return ít nhất 1 key** — không được return `{}`
- **Sync nodes chạy trong thread** — LangGraph dùng `run_in_executor` cho sync functions

### FastAPI + Async
- **Blocking I/O → thread pool** — `asyncio.to_thread()` hoặc `BackgroundTasks`
- **Lifespan cho heavy init** — không init model trong request handler
- **GZipMiddleware** — giảm 60-80% response size cho JSON lớn

### Development Process
- **Test từng layer trước khi tích hợp** — debug `get_chroma_collection()` trực tiếp trước khi test qua API
- **Luôn có sample data** — không phụ thuộc external dataset 100%
- **Log đủ thông tin** — `loguru` với structured format giúp debug rất nhanh

---

## 6. Trạng thái hiện tại

### Đang hoạt động ✅

| Component | Chi tiết |
|---|---|
| FastAPI server | `http://localhost:8081` — port 8081 |
| Swagger UI | `http://localhost:8081/docs` |
| Groq LLM | `llama-3.3-70b-versatile` — API key configured |
| ChromaDB | Local PersistentClient — `data/chroma_db/` |
| Embedding | `paraphrase-multilingual-MiniLM-L12-v2` |
| LangGraph Agent | Router + 6 nodes + 6 tools |
| SQLite Memory | `data/documind.db` |
| Ingestion CLI | `python ingest.py --source json --dir data/raw` |
| Sample Data | 3 văn bản, 15 chunks |

### Chưa hoạt động / Chưa test ⚠️

| Component | Lý do | Priority |
|---|---|---|
| Redis cache | Chưa có Redis server local | Medium |
| Streamlit frontend | Chưa test end-to-end với server | Medium |
| PDF report generation | Chưa test ReportLab flow | Medium |
| WebSocket streaming | Chưa test từ frontend | Medium |
| RAGAS evaluation | Cần thêm data để eval | Low |
| bge-m3 embedding | Cần download 2.3GB | Low (prod only) |
| Email delivery | Cần SMTP credentials | Low |

---

## 7. Việc cần làm tiếp theo

### Ưu tiên cao (để demo được)

- [x] **Fix BM25**: Load nodes từ ChromaDB vào `BM25Retriever` khi server init — `_load_nodes_from_collection()` trong `api/main.py`
- [x] **Ingest server-aware**: Thêm `POST /api/v1/reload` endpoint; sau khi upload file cũng rebuild retriever tự động

- [ ] **Test Streamlit UI**: Chạy `streamlit run frontend/app.py` và test toàn bộ flow

- [ ] **Thêm 50+ văn bản**: Crawl vbpl.vn hoặc download PDF từ nguồn công khai

### Ưu tiên trung bình (để production-ready)

- [ ] **Redis setup**: Cài Redis local để cache queries, giảm latency
- [ ] **bge-m3**: Thay embedding model khi cần accuracy cao hơn
- [ ] **RAGAS eval**: Chạy `python eval/ragas_eval.py` để có benchmark số

### Ưu tiên thấp (cho portfolio)

- [ ] **Deploy Railway**: Cần `RAILWAY_TOKEN` và public URL
- [ ] **LangSmith tracing**: Cần `LANGCHAIN_API_KEY` hợp lệ
- [ ] **Demo video**: Record 2-3 phút sau khi mọi thứ ổn định
- [ ] **README benchmark**: Cập nhật số thật sau khi chạy RAGAS

---

## Changelog

| Ngày | Thay đổi |
|---|---|
| 21/05/2026 | Khởi tạo toàn bộ project từ plan.md |
| 21/05/2026 | Fix security: API keys ra khỏi .env.example |
| 21/05/2026 | Fix: Groq model → llama-3.3-70b-versatile |
| 21/05/2026 | Fix: groq>=0.11.0, chromadb>=0.5.17,<1.0.0 |
| 21/05/2026 | Fix: .env ASCII-only (no box-drawing chars) |
| 21/05/2026 | Fix: LangGraph node names với prefix do_ |
| 21/05/2026 | Fix: _init_rag_sync() trong asyncio.to_thread() |
| 21/05/2026 | Fix: persist_node return {"error": ...} |
| 21/05/2026 | Fix: retrieve_node bỏ import sai |
| 21/05/2026 | Add: 3 sample JSON files, 15 chunks ingested |
| 21/05/2026 | Confirmed: API server running end-to-end với Groq |
| 21/05/2026 | Fix: BM25 hybrid search — load nodes từ ChromaDB vào retriever khi init |
| 21/05/2026 | Fix: corpus_chunks metric — dùng PersistentClient thay vì HttpClient |
| 21/05/2026 | Fix: chunk_count response — dùng len(retrieved_chunks) thay vì missing key |
| 21/05/2026 | Add: POST /api/v1/reload endpoint — rebuild retriever sau CLI ingest |
| 21/05/2026 | Fix: ChromaDB schema error — xóa broken 'test' collection có config_json_str='{}' |
| 21/05/2026 | Confirmed: Hybrid retriever trả 5 chunks có citation từ 3 văn bản pháp luật |

---

*File này được cập nhật liên tục trong quá trình phát triển.*
