# Spec: DocuMind AI — Pivot sang vertical Lao động – Tiền lương – BHXH

> **Trạng thái:** DRAFT chờ Ted duyệt. Chưa implement.
> **Điểm bắt lại:** memory `documind-labor-pivot` (session 4, 2026-09-09) + `docs/corpus/CORPUS_SPEC.md` + `docs/corpus/corpus_manifest.yaml`.
> Đây là spec gốc của initiative. Mỗi module có file `SPEC-<module-id>.md` riêng cạnh file này.

---

## Objective

Đổi DocuMind AI từ vertical **ngân hàng** sang **lao động – tiền lương – bảo hiểm xã hội khu vực tư (Việt Nam)**, corpus dựng lại hoàn toàn từ nguồn chính thức.

**Vấn đề gốc đang giải:** corpus không có biên giới — người dùng không biết hỏi gì được, không biết gì ngoài phạm vi, hệ thống trả lời cả câu nó không nên trả lời. Đây **không** phải vấn đề chất lượng retrieval.

**Người dùng:** người lao động / HR / kế toán tiền lương VN cần tra cứu quy định; và interviewer AI Engineer / hội đồng tuyển sinh master's đọc kỹ repo (thị trường VN, đọc được tiếng Việt).

**Định vị đã chốt (phương án (a), Ted xác nhận 2026-09-10):**
- **Điểm bán chính = corpus có biên giới:** manifest hiển thị trang chủ + scope classifier cưỡng chế + thư viện câu hỏi theo tình huống. Đây là thứ P-153 (trợ lý đa phòng ban) KHÔNG có.
- Tính năng thời gian **vẫn làm**, gọi đúng tên **"temporal-aware retrieval + hiệu lực metadata cấp điều/khoản"** — KHÔNG gọi là "point-in-time reconstruction".
- N1 (số khoản sửa in-place trong VB còn hiệu lực) đã bound ≈ 3–12, chỉ **~1 câu** đổi đáp án qua cơ chế in-place (Điều 139 thai sản, từ 2026-07-01) → giữ làm demo phụ, không phải headline.
- Các mốc lớn khác (lương tối thiểu vùng theo ngày, BHXH 2014↔2024, BHTN 2013↔2026) = doc-level metadata filter.

**Success criteria (toàn initiative):**
1. Trang chủ hiển thị corpus manifest: đúng N văn bản VERIFIED + ngày cập nhật + danh sách "ngoài phạm vi" viết thẳng.
2. `scope_gate` từ chối cả 12 loại câu ngoài phạm vi kèm (a) nêu phạm vi corpus, (b) loại việc không xử lý, (c) ≤3 câu gần nhất trả lời được. **Fail-closed** khi classifier lỗi (thông báo "hệ thống tạm thời không phân loại được", không giả vờ ngoài phạm vi).
3. Đổi `as_of_date` làm đổi câu trả lời cho các mốc doc-level + case Điều 139.
4. Eval gold set ≥ 50 câu (tình huống + ngoài phạm vi + temporal), **gold viết trước khi chạy**, báo accuracy thật + failure analysis (xem memory `eval-methodology`).
5. README + EVALUATION.md mô tả vertical lao động + định vị (a), **0 tham chiếu ngân hàng, 0 con số eval vô căn cứ** (xem memory `documind-benchmark-integrity`).

---

## Tech Stack

Giữ nguyên stack hiện có (không đổi trừ khi module spec nói rõ):

| Lớp | Thành phần |
|---|---|
| Ngôn ngữ | Python 3.11 (target), 3.10+ |
| API | FastAPI 0.115, uvicorn, WebSocket streaming |
| Agent | LangGraph 0.2.28, langchain-core 0.2.43 |
| Retrieval | LlamaIndex 0.14 (QueryFusionRetriever: dense + BM25 + RRF) |
| Embedding | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dim) |
| Reranker | `BAAI/bge-reranker-v2-m3` (cross-encoder, local) |
| Vector store | ChromaDB 0.6 (mặc định); Qdrant Cloud optional qua `VECTOR_STORE_PROVIDER` |
| LLM chain | Groq llama-3.3-70b → Gemini *-flash-lite → OpenAI-compat → extractive (không LLM) |
| Frontend | React 19 + Vite 6 (`frontend/`) |
| Eval | Harness tự viết (`eval/`) + RAGAS |

**Không thêm dependency mới** cho toàn initiative này. `scope_gate` + temporal filter + question library đều tái dùng LLM client + embedder + config sẵn có.

---

## Commands

```bash
# Cài đặt
pip install -e .              # + requirements.txt
cd frontend && npm install

# Test / lint
pytest                        # cấu hình ở pytest.ini (cov=src, asyncio auto)
pytest tests/test_scope_gate.py -v      # 1 file
ruff check src tests          # lint (cấu hình pyproject.toml, line-length 100)
ruff check --fix src tests
mypy src                      # optional, strict=false

# Dev
uvicorn src.api.main:app --reload --port 8081
cd frontend && npm run dev

# Corpus / ingest
python scripts/ingest_documents.py --source-dir data/raw/lao_dong --manifest docs/corpus/corpus_manifest.yaml --reset

# Eval
python eval/run_evals.py --skip-ragas      # bước 1: sinh answer + metric local
python eval/run_evals.py                    # bước 2: RAGAS judge từ cache
```

---

## Project Structure

```
src/
  agent/          → LangGraph: graph.py (topology), memory.py, tools.py
  rag/            → retriever.py, embedder.py, generator.py, grader.py,
                    compliance.py, vector_backend.py
  ingestion/      → chunker.py (chunk_by_dieu), cleaner.py, loader.py, crawler.py
  api/            → main.py, routes/{query,documents,reports,health}.py, schemas.py
  report/         → generator.py (PDF)
  config.py       → Settings (pydantic-settings, đọc .env)
  guardrails.py   → citation validation
scripts/          → ingest_documents.py, cleanup_chroma.py, ...
eval/             → run_evals.py, rag_comparison.py, metrics.py, ragas_eval.py
tests/            → test_*.py mirror src/ (pytest, offline)
data/
  raw/            → toàn văn văn bản nguồn (.md/.txt sạch)
  chroma_db/      → vector index
  compliance/     → criteria.json
  eval/           → test_questions.json (gold set)
  scope/          → topics.yaml (MỚI — allowlist chủ đề cho scope_gate prefilter)
docs/
  corpus/         → CORPUS_SPEC.md, corpus_manifest.yaml (đã có, DRAFT)
  spec/           → SPEC.md (file này) + SPEC-<module>.md
frontend/src/     → main.tsx, styles.css
```

**Thư mục/tên MỚI initiative này tạo ra:** `data/scope/topics.yaml`, `data/eval/` gold set mới, `docs/spec/`, `data/raw/lao_dong/`, cột metadata clause-level trong ChromaDB.

---

## Code Style

Theo code hiện có. Một ví dụ đại diện (từ `src/agent/graph.py`):

```python
def _contextualize_query(query: str, history: list[dict]) -> str:
    """Rewrite a context-dependent follow-up into a standalone question.
    Tries Groq first, then Gemini ... Falls back to the original query
    unchanged only if both providers fail."""
    settings = get_settings()
    ...
    if settings.groq_api_key:
        try:
            ...
        except Exception as exc:
            logger.warning("contextualize_node Groq failed, trying Gemini: {}", exc)
    ...
    logger.warning("contextualize_node: both Groq and Gemini failed, using original query")
    return query
```

Quy ước:
- `from __future__ import annotations` đầu file; type hint đầy đủ; `list[dict]` không `List`.
- Docstring + comment giải thích **"tại sao"**, không chỉ "làm gì". Comment dài giải thích quyết định thiết kế / failure mode là bình thường trong repo này.
- `loguru.logger` với `{}` placeholder, không f-string trong log.
- Graceful degradation: mọi lời gọi LLM/model đều có fallback chain + `except Exception as exc` + log, không để crash lan.
- `@dataclass` cho value object; `dict` cho state đi qua LangGraph.
- **User-facing text = tiếng Việt.** Comment/docstring có thể Anh hoặc Việt (repo trộn cả hai).
- `snake_case` Python; tên node LangGraph prefix `do_` để không đụng key của `AgentState`.
- Số/ngày pháp lý: luôn kèm nguồn; chỗ chưa xác minh đánh dấu `PARTIAL`/`UNVERIFIED`, không đoán.

---

## Testing Strategy

- **Framework:** pytest (`pytest.ini`: `pythonpath=.`, `asyncio_mode=auto`, `--cov=src`).
- **Vị trí:** `tests/test_<module>.py` mirror `src/`.
- **Offline bắt buộc:** test không gọi mạng. LLM/embedder/retriever mock qua fixture (xem `tests/conftest.py`).
- **Coverage:** giữ mức repo hiện tại (không có ngưỡng cứng ở CI hiện tại, không hạ). Mọi node/hàm mới có test.
- **Test level theo mối lo:**
  - Temporal filter: unit test thuần (bảng version → tập version áp dụng tại T) — không cần index.
  - `scope_gate`: bộ fixture ~30–40 câu gán nhãn (IN_SCOPE / 12 loại OOC / NEEDS_PERSONAL_DATA / INDIVIDUAL_ADVICE), assert `decision` + prefilter không nuốt câu mơ hồ. **Fail-closed path** có test riêng (mock LLM raise → assert từ chối kèm thông báo lỗi hệ thống).
  - Corpus manifest: test schema validation (mọi dòng có field bắt buộc; không dòng `UNVERIFIED` nào lọt vào `locked_list_v1` khi ingest).
  - Eval gold set: test cấu trúc file (mỗi case có `id`, `question`, `ground_truth`, `category`, `as_of_date` nếu là câu temporal).
- **Eval ≠ test:** gold set chạy qua `eval/run_evals.py`, không phải pytest. Gold viết + commit **trước** khi chạy hệ thống.

---

## Boundaries

**Always:**
- Chạy `pytest` + `ruff check` trước mỗi commit.
- User-facing text tiếng Việt; trích dẫn `[N]` gắn điều/khoản + số hiệu + nguồn.
- Mọi phát biểu về một văn bản truy được về 1 dòng VERIFIED trong `corpus_manifest.yaml`.
- Commit theo `<type>(<scope>): <subject>` (xem `~/.claude/gitflow.md`); rebase + squash 1 commit trước PR.
- Cập nhật spec file trước khi đổi hướng implement.

**Ask first:**
- Thêm bất kỳ dependency nào (mặc định của initiative này là KHÔNG thêm).
- Đổi schema metadata ChromaDB / contract API `/query` / `AgentState` keys.
- Bulk-download corpus (đã hoãn tới khi corpus-acquisition bước verify xong).
- Đổi model LLM/embedder/reranker.
- Đổi chiến lược nhánh git / tạo `develop`.

**Never:**
- Commit `.env` / secret; key trong `.env` hiện đang trống — giữ trống trong repo.
- Ingest dòng `UNVERIFIED` hoặc quan hệ sửa đổi chưa đối chiếu lược đồ vbpl.vn.
- Bịa nội dung điều luật, ngày ban hành, hoặc con số benchmark.
- Xoá test đang fail để cho "xanh".
- Bán "point-in-time reconstruction" / "full reconstruction" làm điểm khác biệt (mâu thuẫn quyết định (a)).
- Mở rộng corpus quá `locked_list_v1` khi verification nợ chưa đóng (feedback Ted session 4).

---

## Capability Map

| Module id | Trách nhiệm | Phụ thuộc |
|---|---|---|
| `corpus-acquisition` | Đóng verification nợ (6 dòng UNVERIFIED, Công báo, lược đồ vbpl.vn); acquire toàn văn sạch 18 VB + VBHN vào `data/raw/lao_dong/`; xoá tài liệu ngân hàng bịa; xuất `corpus_manifest.yaml` VERIFIED + bảng quan hệ sửa đổi. | — |
| `clause-schema-ingestion` | Parse toàn văn → chunk điều/khoản; gắn metadata version cấp khoản (`clause_uid`, `version_id`, `effective_from/to`, `status`, `superseded_by`, `amended_by_doc`); build index. | `corpus-acquisition` |
| `temporal-retrieval` | Lớp lọc `as_of_date` phía trên retrieval node: chỉ giữ version có hiệu lực tại T; luồng `as_of_date` qua `AgentState` + API + generator. | `clause-schema-ingestion` |
| `scope-gate` | Node `scope_gate` trước router: classifier 4 quyết định, 2 tầng (rule prefilter + LLM), **fail-closed**, template từ chối. | `clause-schema-ingestion` |
| `question-library` | Thư viện câu hỏi tĩnh 40–60 câu sinh từ tên Điều của corpus, gom theo tình huống; cấp `suggested_questions` cho scope-gate + feed frontend. | `clause-schema-ingestion` |
| `compliance-criteria` | Thay `criteria.json` ngân hàng bằng tiêu chí lao động/BHXH (lương tối thiểu vùng, làm thêm 200/300h, tỷ lệ đóng BHXH, trợ cấp thôi việc, tuổi hưu 2026); tiêu chí gắn mốc áp dụng. | `corpus-acquisition`, `temporal-retrieval` |
| `eval-goldset` | Gold set mới ≥50 câu (tình huống + OOC + temporal); wire vào `eval/`; báo số thật + failure analysis. | `question-library`, `temporal-retrieval`, `scope-gate` |
| `frontend-boundary-ui` | Panel corpus manifest trang chủ; UI thư viện câu hỏi theo tình huống; hiển thị hiệu lực + bộ chọn `as_of_date`; render template từ chối. | `temporal-retrieval`, `scope-gate`, `question-library` |
| `docs-rewrite` | Viết lại README + EVALUATION.md cho vertical lao động + định vị (a); gỡ sạch tham chiếu ngân hàng + số vô căn cứ. | `eval-goldset` |

**Build order:**

```
corpus-acquisition
  └─► clause-schema-ingestion
        ├─► temporal-retrieval ─┐
        ├─► scope-gate ─────────┤
        └─► question-library ───┤
                                ├─► compliance-criteria
                                ├─► eval-goldset
                                ├─► frontend-boundary-ui
                                └─► docs-rewrite   (sau eval-goldset)
```

Không có chu trình. Interface giữa các module ghi trong spec của module **cung cấp** (vd contract `as_of_date` nằm trong `SPEC-temporal-retrieval.md`; `ScopeGateResult` nằm trong `SPEC-scope-gate.md`).

**Ghi chú nhánh git:** repo chỉ có `main` (không có `develop`). 2 commit chưa push trên `feature/llm-openai-backup` (llm backup chain — memory `llm-backup-chain`). Initiative này nên: (1) Ted quyết push/merge nhánh llm-backup trước, (2) tạo nhánh feature mới từ `main` cho pivot. → Open question, không tự quyết.

---

## Open Questions (toàn initiative)

1. **Nhánh git:** xử lý `feature/llm-openai-backup` chưa push thế nào trước khi bắt đầu pivot?
2. **corpus-acquisition:** nguồn lấy toàn văn — vbpl.vn bản mới là SPA (WebFetch chỉ lấy vỏ). Ted tự tải về `data/raw/lao_dong/` hay thử luatvietnam/chinhphu.vn? (chi tiết trong `SPEC-corpus-acquisition.md`).
3. **scope-gate:** ngưỡng `confidence` prefilter ↔ LLM (đề xuất 0.85) — chốt sau khi có eval hay chốt trước?
4. **question-library:** sinh câu hỏi bằng LLM (1 lần, offline, người review) hay viết tay hoàn toàn? 40–60 câu.
5. **compliance-criteria:** giữ `compliance_check` là rule engine hardcode (như bản ngân hàng) hay mở rộng cho tra công thức + liệt kê dữ liệu thiếu (khớp `NEEDS_PERSONAL_DATA`)?
6. **Deploy:** bản demo sống (Render/Vercel) có nằm trong initiative này không, hay để sau (memory `portfolio-cv-selection` nói demo 1 sống + 2 video)?

---

## Phase tiếp theo

Sau khi Ted duyệt map + spec này → Phase 2 (Plan): `tasks/plan.md` + `tasks/todo.md` theo `planning-and-task-breakdown`, đi theo build order, mỗi module 1 lát cắt dọc có checkpoint.
