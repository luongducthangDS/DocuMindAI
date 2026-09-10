# DocuMind Labor Pivot — Task List

> Kèm `tasks/plan.md`. Nguồn: `docs/spec/`. Đánh dấu `[x]` khi acceptance + verification đều pass.
> Test command mặc định: `pytest <file> -v` · Lint: `ruff check src tests` · Frontend build: `cd frontend && npm run build`.

---

## Phase 0 — Decision gate (Ted)

### D1: Chốt nhánh git
**Description:** Repo chỉ có `main`; 2 commit chưa push trên `feature/llm-openai-backup` (llm backup chain). Chốt: push nhánh đó lên `origin` rồi tạo `feature/labor-pivot` từ nó, hay merge `main` trước.
**Acceptance:**
- [ ] Ted chốt phương án; nhánh làm việc cho pivot đã tạo từ base đúng
**Verification:** `git branch --show-current` = nhánh pivot; `git log --oneline -3` đúng base
**Dependencies:** None · **Files:** — · **Scope:** XS

### D2: Chốt nguồn lấy toàn văn corpus
**Description:** vbpl.vn bản mới là SPA (WebFetch chỉ lấy vỏ). Chốt nguồn cho 18 VB + VBHN + VB lịch sử.
**Acceptance:**
- [ ] Ted chốt: tự tải vào `data/raw/lao_dong/` HAY em thử luatvietnam/chinhphu.vn; nguồn cho VB lịch sử + VBHN rõ
**Verification:** ghi quyết định vào `SPEC-corpus-acquisition.md` Open Questions
**Dependencies:** None · **Files:** `docs/spec/SPEC-corpus-acquisition.md` · **Scope:** XS

---

## Phase 1 — corpus-acquisition

### T1: Xoá tài liệu ngân hàng bịa + audit `data/raw/`
**Description:** Kiểm `data/raw/` thực tế (memory 2 ngày tuổi), xoá file corpus ngân hàng bịa: `05_quy_che_cho_vay_...md`, `06_bieu_phi_...md`, file 01–04 (tóm lược viết tay + URL `sbv.gov.vn` bịa). Ghi lại file nào đã xoá.
**Acceptance:**
- [ ] `data/raw/` không còn file corpus ngân hàng bịa
- [ ] Liệt kê file đã xoá trong commit message
**Verification:** `ls data/raw/` + `git status`
**Dependencies:** D1 · **Files:** `data/raw/*` · **Scope:** XS

### T2: Tra 1 vòng 6 dòng UNVERIFIED
**Description:** Tra `11/2025/TT-BNV`, `12/2025/TT-BNV`, `25/2025/TT-BYT`, `134/2015/NĐ-CP`, `28/2015/NĐ-CP`, `18+19/2021/TT-BLĐTBXH`, `24/2022/TT-BLĐTBXH`. Không ra số hiệu + ngày ban hành + lược đồ → LOẠI khỏi `locked_list_v1`, ghi lý do ở comment.
**Acceptance:**
- [ ] Mỗi dòng: hoặc lên `VERIFIED` (có nguồn ≥2), hoặc bị xoá khỏi `locked_list_v1` với lý do
- [ ] `corpus_manifest.yaml` `locked_list_v1` không còn dòng `UNVERIFIED`
**Verification:** đọc manifest; grep `UNVERIFIED` trong phần `locked_list_v1` = 0
**Dependencies:** D2 · **Files:** `docs/corpus/corpus_manifest.yaml` · **Scope:** S

### T3: Đối chiếu Công báo các mốc ngày PARTIAL
**Description:** Đối chiếu `congbao.chinhphu.vn`: `ngay_ban_hanh`/`ngay_hieu_luc` mọi dòng PARTIAL (đặc biệt NĐ 293/2025, 158/2025, 159/2025, 374/2025, 115/2015, TT 59/2015).
**Acceptance:**
- [ ] Mọi dòng `locked_list_v1` có `ngay_ban_hanh` + `ngay_hieu_luc` xác nhận từ Công báo hoặc toàn văn chính thức
- [ ] `verify_status` cập nhật `PARTIAL` → `VERIFIED` khi đủ nguồn
**Verification:** đọc manifest; mỗi dòng có `nguon` trỏ Công báo/chinhphu.vn
**Dependencies:** T2 · **Files:** `docs/corpus/corpus_manifest.yaml` · **Scope:** S–M

### T4: Đối chiếu lược đồ sửa đổi + xác nhận điều BLLĐ bị sửa in-place
**Description:** Đối chiếu quan hệ `sua_doi_boi`/`thay_the`/`bi_thay_the_boi`. Trọng tâm: 4 luật nghi sửa BLLĐ (41/2024, 113/2025, 71/2025, 124/2025) — xác nhận CHÍNH XÁC điều/khoản nào bị sửa in-place (đọc điều "sửa đổi, bổ sung" trong toàn văn từng luật sửa).
**Acceptance:**
- [ ] Bảng `sua_doi_boi` BLLĐ 2019: mỗi quan hệ có `pham_vi` tới điều/khoản cụ thể + `verify: VERIFIED`
- [ ] Danh sách cuối cùng "khoản BLLĐ bị sửa in-place" (kỳ vọng ≈ chỉ Điều 139 k1)
- [ ] Ghi rõ 71/2025 và 124/2025 có/không sửa BLLĐ (hiện UNVERIFIED)
**Verification:** đọc manifest; đối chiếu 1 quan hệ với toàn văn luật sửa
**Dependencies:** T3 · **Files:** `docs/corpus/corpus_manifest.yaml` · **Scope:** M

### T5: Acquire toàn văn sạch 18 VB + VBHN BLLĐ
**Description:** Lấy toàn văn sạch vào `data/raw/lao_dong/<doc_id>.md`, mỗi file có frontmatter YAML (`doc_id`, `so_hieu`, `ten`, `ngay_ban_hanh`, `ngay_hieu_luc`, `nguon`). VBHN BLLĐ = `18/VBHN-VPQH` bản 2026. Ưu tiên text/HTML sạch hơn PDF. (Nguồn theo D2.)
**Acceptance:**
- [ ] `data/raw/lao_dong/` có 1 file/VB trong `locked_list_v1` sau T2 + VBHN BLLĐ
- [ ] Mỗi file là toàn văn chính thức (không tóm lược), có frontmatter đủ trường
- [ ] Điều đầu + điều cuối của 3 file ngẫu nhiên khớp nguồn chinhphu.vn
**Verification:** `ls data/raw/lao_dong/`; đọc 3 file spot-check
**Dependencies:** T4 · **Files:** `data/raw/lao_dong/*.md` · **Scope:** M

### T6: Lấy 2 bản text Điều 139 khoản 1 BLLĐ
**Description:** Lưu riêng bản khoản 1 Điều 139 hiện hành (2021) và bản mới (Luật Dân số 2025, HL 2026-07-01) để T9 dựng version.
**Acceptance:**
- [ ] `data/raw/lao_dong/_versions/45-2019-QH14__d139_k1__2021.md` + `__2026-07-01.md` (hoặc format tương đương), mỗi file có `effective_from`, `amended_by_doc`
**Verification:** đọc 2 file, khác nhau đúng nội dung thai sản (nữ 6th vs con thứ 2 = 7th…)
**Dependencies:** T5 · **Files:** `data/raw/lao_dong/_versions/*` · **Scope:** XS

### ☑ Checkpoint C1
- [ ] `grep -c UNVERIFIED` trong `locked_list_v1` = 0
- [ ] `data/raw/lao_dong/` đủ file toàn văn + frontmatter hợp lệ
- [ ] Danh sách "khoản sửa in-place" đã chốt
- [ ] Review với Ted trước Phase 2

---

## Phase 2 — clause-schema-ingestion

### T7: Manifest loader YAML + bản đồ hiệu lực
**Description:** `src/ingestion/manifest.py`: đọc `corpus_manifest.yaml`, trả `{doc_id → {so_hieu, ten, ngay_hieu_luc, het_hieu_luc_tu, in_place_amended_clauses[]}}`.
**Acceptance:**
- [ ] `load_manifest(path)` trả dict đúng cấu trúc cho `locked_list_v1`
- [ ] Bỏ qua dòng ngoài `locked_list_v1`
**Verification:** `pytest tests/test_ingestion.py::test_load_manifest -v`
**Dependencies:** C1 · **Files:** `src/ingestion/manifest.py`, `tests/test_ingestion.py` · **Scope:** S

### T8: Mở rộng `chunker.py` — `clause_uid` + metadata version
**Description:** Với mỗi Điều sinh `clause_uid` cấp điều; với Điều nằm trong `in_place_amended_clauses` tách tới `khoan`. Gắn `dieu`, `dieu_tieu_de`, `khoan`, `effective_from/to`, `status`, `verify_status`, `version_id`. Giữ nguyên metadata cũ (`source_url, title, doc_type, so_hieu, dieu_header, khoan_count, ...`). Chốt quy ước `effective_to` rỗng.
**Acceptance:**
- [ ] `chunk_by_dieu(text, doc_meta)` trả chunk có `clause_uid` + trường version
- [ ] Metadata cũ còn nguyên (test regression)
- [ ] Điều bị sửa in-place tách tới khoản; điều khác giữ cấp điều
**Verification:** `pytest tests/test_ingestion.py -v`
**Dependencies:** T7 · **Files:** `src/ingestion/chunker.py`, `tests/test_ingestion.py` · **Scope:** M

### T9: Dựng 2-version cho Điều 139 k1
**Description:** Đọc 2 file `_versions/` (T6). Sinh 2 record khoản 1 Điều 139: bản cũ `effective_to=2026-07-01, status=superseded, superseded_by=<v mới>`; bản mới `effective_from=2026-07-01, amended_by_doc="Luật Dân số 2025 (113/2025/QH15)"`, `status` tính theo ngày build.
**Acceptance:**
- [ ] Sau ingest, `clause_uid = 45-2019-QH14__d139_k1` có đúng 2 `version_id`
- [ ] `superseded_by` bản cũ trỏ `version_id` bản mới (tồn tại)
**Verification:** `pytest tests/test_ingestion.py::test_dieu139_two_versions -v` + query index
**Dependencies:** T8 · **Files:** `src/ingestion/chunker.py` hoặc `src/ingestion/versions.py`, `tests/test_ingestion.py` · **Scope:** S

### T10: Mở rộng `scripts/ingest_documents.py`
**Description:** Đọc `data/raw/lao_dong/` + `corpus_manifest.yaml` (YAML, hiện đọc manifest JSON). Lấy per-file metadata từ frontmatter + manifest. Bỏ record `verify_status != VERIFIED`. Đẩy chunk + full metadata vào ChromaDB.
**Acceptance:**
- [ ] `python scripts/ingest_documents.py --source-dir data/raw/lao_dong --manifest docs/corpus/corpus_manifest.yaml --reset` chạy sạch, in số chunk/VB
- [ ] Collection `count > 0`; mỗi chunk có `clause_uid`, `effective_from`, `status`
- [ ] 0 record `UNVERIFIED`
**Verification:** chạy lệnh + `python -c` đếm + kiểm metadata 1 chunk
**Dependencies:** T9 · **Files:** `scripts/ingest_documents.py` · **Scope:** S–M

### T11: `validate_corpus.py` + mở rộng test ingestion
**Description:** `scripts/validate_corpus.py`: frontmatter đủ trường; `clause_uid` unique/version; mọi `superseded_by` trỏ `version_id` tồn tại; 0 `UNVERIFIED` trong index. Mở rộng `tests/test_ingestion.py`.
**Acceptance:**
- [ ] `python scripts/validate_corpus.py` exit 0 trên corpus đã ingest
- [ ] `tests/test_ingestion.py`: parse điều→khoản, sinh `clause_uid`, 2-version, loại `UNVERIFIED`
**Verification:** `pytest tests/test_ingestion.py tests/test_rag.py -v` + chạy validate
**Dependencies:** T10 · **Files:** `scripts/validate_corpus.py`, `tests/test_ingestion.py` · **Scope:** M

### ☑ Checkpoint C2
- [ ] Ingest thật chạy sạch; index có clause metadata
- [ ] `pytest tests/test_ingestion.py tests/test_rag.py` xanh
- [ ] `scripts/validate_corpus.py` pass
- [ ] Retriever + generator cũ vẫn hoạt động (query thử 1 câu)

---

## Phase 3a — temporal-retrieval

### T12: `src/rag/temporal.py` + test
**Description:** Hàm thuần: `versions_in_force(chunks, as_of) -> filtered`; `is_out_of_range(as_of, earliest) -> bool`; tie-break lấy `effective_from` muộn nhất + log warning.
**Acceptance:**
- [ ] Quy tắc: `effective_from <= T AND (effective_to rỗng OR effective_to > T) AND status != repealed`
- [ ] `tests/test_temporal.py`: bảng version → in-force tại nhiều T; out-of-range; tie-break
**Verification:** `pytest tests/test_temporal.py -v`
**Dependencies:** C2 · **Files:** `src/rag/temporal.py`, `tests/test_temporal.py` · **Scope:** S

### T13: `as_of_date` qua state + API
**Description:** `AgentState` thêm `as_of_date: str` + `time_out_of_range: bool`. `run_agent(..., as_of_date=None)`. `QueryRequest` (schemas.py) thêm `as_of_date: str | None = None`. Route `query.py` truyền xuống. **Chốt AgentState keys ở đây — 3b/3c phụ thuộc.**
**Acceptance:**
- [ ] `POST /query {"query":"...","as_of_date":"2025-06-01"}` không lỗi; thiếu field → mặc định hôm nay
- [ ] `as_of_date` xuất hiện trong `AgentState` khi chạy
**Verification:** `pytest tests/test_agent.py -v` + curl API
**Dependencies:** T12 · **Files:** `src/agent/graph.py`, `src/api/schemas.py`, `src/api/routes/query.py` · **Scope:** S–M

### T14: Node `do_temporal_filter` + generator
**Description:** Chèn node `do_temporal_filter` giữa `do_retrieve` và `do_grade` — gọi `versions_in_force`. Set `time_out_of_range`. Generator prompt nhận `as_of_date` (nêu mốc + chọn đúng số doc-level như lương tối thiểu). `steps` hiển thị "đang tra theo mốc <ngày>".
**Acceptance:**
- [ ] `as_of_date=2025-03-01` "lương tối thiểu vùng I" → 4.96tr; `2026-02-01` → 5.31tr
- [ ] "nghỉ thai sản" `2026-05-01` vs `2026-09-10` → 2 đáp án khác, trích đúng bản Điều 139
- [ ] `as_of_date=2010-01-01` → "corpus chỉ phủ từ 2015-01-01"
- [ ] `as_of_date` = hôm nay → hành vi retriever không đổi (regression)
**Verification:** `pytest tests/test_agent.py tests/test_temporal.py -v` + 3 truy vấn tay
**Dependencies:** T13 · **Files:** `src/agent/graph.py`, `src/rag/generator.py` · **Scope:** M

### ☑ Checkpoint C3
- [ ] 3 mốc `as_of_date` cho kết quả khác nhau (2 doc-level + 1 in-place)
- [ ] `pytest tests/test_temporal.py tests/test_agent.py tests/test_rag.py` xanh

---

## Phase 3b — scope-gate

### T15: `data/scope/topics.yaml`
**Description:** Script sinh nháp `topic → keywords` từ `dieu_tieu_de` toàn corpus + 12 loại OOC (§4 CORPUS_SPEC). Review chỉnh tay.
**Acceptance:**
- [ ] `data/scope/topics.yaml`: ≥8 topic in-scope + 12 `out_of_scope_kind`, mỗi cái có keyword list
- [ ] Keyword không mâu thuẫn (1 từ không vừa in vừa out)
**Verification:** đọc file + test load YAML
**Dependencies:** C2 · **Files:** `data/scope/topics.yaml`, `scripts/gen_scope_topics.py` · **Scope:** S

### T16: `src/agent/scope_gate.py`
**Description:** `classify_scope(inp) -> ScopeGateResult`. Tầng 1 prefilter (topics.yaml). Tầng 2 LLM (router client, temp=0, JSON out, few-shot 12 OOC + ~6 in-scope). **Fail-closed:** cả 2 tầng không kết luận → `OUT_OF_SCOPE`, `reason="classifier_unavailable"`, `confidence=0.0`, log `error`.
**Acceptance:**
- [ ] Trả `ScopeGateResult` đúng shape (spec §5.3)
- [ ] Prefilter bắt in-scope/out-of-scope chắc chắn → không gọi LLM
- [ ] Mock LLM raise + prefilter không kết luận → fail-closed đúng
**Verification:** `pytest tests/test_scope_gate.py -v`
**Dependencies:** T15 · **Files:** `src/agent/scope_gate.py`, `tests/test_scope_gate.py` · **Scope:** M

### T17: Fixture `scope_cases.json` + test (viết TRƯỚC)
**Description:** `tests/fixtures/scope_cases.json` ~35 câu gán nhãn — **viết trước khi chạy T16 trên câu thật**: 12 loại OOC (≥2/loại) + ≥8 IN_SCOPE + ≥3 NEEDS_PERSONAL_DATA + ≥3 INDIVIDUAL_ADVICE + 2 time_out_of_range. Test chấm accuracy + confusion theo `out_of_scope_kind`.
**Acceptance:**
- [ ] Fixture ≥35 câu, đủ nhãn; commit trước commit "run scope_gate trên câu thật"
- [ ] `pytest tests/test_scope_gate.py`: accuracy fixture ≥ 0.85 (LLM mock gold), prefilter-only ≥ 0.6
**Verification:** `pytest tests/test_scope_gate.py -v`
**Dependencies:** T16 · **Files:** `tests/fixtures/scope_cases.json`, `tests/test_scope_gate.py` · **Scope:** M

### T18: Graph `do_scope_gate` + `do_refusal` + gỡ `unknown`
**Description:** Node `do_scope_gate` sau `do_contextualize`, trước `router`. Node `do_refusal` (4 template: OOC, NEEDS_PERSONAL_DATA, INDIVIDUAL_ADVICE, classifier_unavailable + biến `time_out_of_range`). Conditional edge. Truyền cờ vào `AgentState` cho generator. Kiểm mọi nơi dùng `intent == "unknown"` → chuyển sang scope_gate.
**Acceptance:**
- [ ] Câu OOC → response không gọi retrieval (`chunk_count == 0`), body có phạm vi corpus + ≤3 `suggested_questions`
- [ ] classifier_unavailable → message "hệ thống tạm thời chưa phân loại", KHÔNG "ngoài phạm vi"
- [ ] Câu in-scope cũ vẫn ra câu trả lời (regression)
**Verification:** `pytest tests/test_agent.py tests/test_scope_gate.py -v` + 5 OOC + 3 in-scope tay
**Dependencies:** T17 · **Files:** `src/agent/graph.py`, `src/agent/refusal.py` · **Scope:** M

### ☑ Checkpoint C4
- [ ] OOC bị từ chối kèm phạm vi + gợi ý; fail-closed đúng message
- [ ] `pytest tests/test_scope_gate.py tests/test_agent.py` xanh
- [ ] Prefilter né được LLM cho case rõ ràng (đo qua log/mock)

---

## Phase 3c — question-library

### T19: `library.json` (40–60 câu theo tình huống)
**Description:** `scripts/gen_question_library.py` duyệt `dieu_tieu_de` → nhóm tình huống (mapping viết tay hoặc LLM-assisted + review) → sinh nháp. Review + chỉnh tay `data/questions/library.json`.
**Acceptance:**
- [ ] 40–60 câu, ≥8 tình huống, mỗi tình huống 3–8 câu
- [ ] Mỗi câu có `primary_dieu` (`clause_uid` tồn tại trong index) + `scope_topic`
- [ ] Review từng câu (không commit thẳng output LLM)
**Verification:** test cấu trúc + `scripts/validate_corpus.py` kiểm `primary_dieu` khớp index
**Dependencies:** C2 (index), tốt nhất sau T15 · **Files:** `data/questions/library.json`, `scripts/gen_question_library.py` · **Scope:** M

### T20: `question_library.py` + wire scope_gate
**Description:** `src/agent/question_library.py`: `load_library()`, `nearest_questions(query, k=3)` (embedding cosine, tái dùng embedder). Wire vào `scope_gate.suggested_questions`.
**Acceptance:**
- [ ] `nearest_questions` trả câu cùng `scope_topic` cho query mẫu; rỗng/điểm thấp cho query xa (OOC)
- [ ] `scope_gate` OOC response chứa `suggested_questions` thật từ library
**Verification:** `pytest tests/test_question_library.py tests/test_scope_gate.py -v`
**Dependencies:** T19, T16 · **Files:** `src/agent/question_library.py`, `src/agent/scope_gate.py`, `tests/test_question_library.py` · **Scope:** S–M

### ☑ Checkpoint C5
- [ ] Library 40–60 câu, mọi `primary_dieu` hợp lệ
- [ ] scope_gate trả gợi ý thật; `pytest tests/test_question_library.py` xanh

---

## Phase 4a — compliance-criteria

### T21: `criteria.json` lao động
**Description:** Viết `data/compliance/criteria.json` mới ≥6 tiêu chí (lương tối thiểu vùng I theo mốc, làm thêm 200/300h, tỷ lệ đóng BHXH 8%, trợ cấp thôi việc `formula_only`, tuổi hưu 2026, nghỉ thai sản Điều 139). Đối chiếu toàn văn `data/raw/lao_dong/` TRƯỚC; `source_url` trỏ điều cụ thể. Thêm `effective_from/to`, `criterion_kind`.
**Acceptance:**
- [ ] ≥6 tiêu chí, 0 tiêu chí banking
- [ ] Mỗi `value` + `so_hieu` + `dieu_khoan` đối chiếu toàn văn (ghi chú nguồn)
- [ ] Lương tối thiểu: ≥2 criterion cùng topic khác `effective_from`
**Verification:** đọc file + đối chiếu 3 tiêu chí với toàn văn; test load
**Dependencies:** C1, C3 · **Files:** `data/compliance/criteria.json` · **Scope:** M

### T22: Mở rộng `compliance.py`
**Description:** `match_criteria` lọc theo `as_of_date` (`effective_from/to`). `criterion_kind=formula_only` → `verdict="formula"` + `missing_fields[]`. Cập nhật `_NUMBER_NEAR_LABEL_RE` nhãn VN lao động ("lương|giờ làm thêm|tỷ lệ đóng|thâm niên|tuổi|...").
**Acceptance:**
- [ ] `check_compliance(situation, as_of_date)` lọc đúng criterion theo mốc
- [ ] `formula_only` trả `verdict="formula"` + `missing_fields`, không có con số kết quả
- [ ] Regex bắt số cạnh nhãn lao động
**Verification:** `pytest tests/test_compliance.py -v`
**Dependencies:** T21 · **Files:** `src/rag/compliance.py`, `tests/test_compliance.py` · **Scope:** M

### T23: `compliance_check_node` + render + wire NEEDS_PERSONAL_DATA
**Description:** `_render_compliance_answer` xử lý `verdict="formula"` (công thức + căn cứ + "dữ liệu còn thiếu"). scope_gate `NEEDS_PERSONAL_DATA` + criterion `formula_only` khớp → route `do_compliance`. `tests/test_compliance.py` viết lại toàn bộ (đang test banking).
**Acceptance:**
- [ ] "Làm thêm 250 giờ/năm có vượt không?" → `fail` + citation Điều 107; "+ ngành dệt may" → hỏi lại/`pass`
- [ ] "Trợ cấp thôi việc của tôi bao nhiêu?" → `verdict=formula`, liệt kê dữ liệu thiếu, KHÔNG ra số
- [ ] "lương tối thiểu vùng I" `as_of 2025-03-01` → 4.96tr; `2026-02-01` → 5.31tr
**Verification:** `pytest tests/test_compliance.py tests/test_agent.py -v` + 4 tình huống tay
**Dependencies:** T22, T18 · **Files:** `src/agent/graph.py`, `src/rag/compliance.py`, `tests/test_compliance.py` · **Scope:** M

### ☑ Checkpoint C6
- [ ] 4 tình huống (2 threshold, 1 temporal, 1 formula) đúng
- [ ] `pytest tests/test_compliance.py tests/test_agent.py` xanh

---

## Phase 4b — eval-goldset

### T24: `test_questions.json` (≥50 câu) — commit riêng TRƯỚC
**Description:** Viết gold set: ≥30 nhóm A (tra cứu tình huống, dùng lại `question-library`, ground truth viết tay từ toàn văn) + ≥12 nhóm B (12 loại OOC) + ≥8 nhóm C (temporal, cùng câu 2 `as_of_date`). Bổ sung field `source_clause`, `as_of_date`, `expected_behavior`. **Commit riêng, đứng trước commit chạy eval.**
**Acceptance:**
- [ ] ≥50 câu, đủ 3 nhóm; 0 câu banking; mọi câu A có `source_clause` tồn tại
- [ ] Ground truth viết TRƯỚC khi chạy hệ thống (không xem output)
- [ ] `tests/test_eval_goldset.py`: mọi câu đủ field, `as_of_date` hợp lệ, `source_clause` khớp index
**Verification:** `pytest tests/test_eval_goldset.py -v` + `git log` (commit gold trước commit eval)
**Dependencies:** C5, C3 · **Files:** `data/eval/test_questions.json`, `tests/test_eval_goldset.py` · **Scope:** M

### T25: `scope_eval.py` + `temporal_eval.py`
**Description:** `eval/scope_eval.py` chạy nhóm B qua `scope_gate` → precision/recall + confusion. `eval/temporal_eval.py` chạy nhóm C với `as_of_date` → % khớp ground truth đúng mốc. Optional: dòng so sánh nhóm C có/không temporal filter.
**Acceptance:**
- [ ] 2 script chạy độc lập, in metric
- [ ] Không sửa ground truth theo output
**Verification:** `python eval/scope_eval.py` + `python eval/temporal_eval.py`
**Dependencies:** T24 · **Files:** `eval/scope_eval.py`, `eval/temporal_eval.py` · **Scope:** M

### T26: Gộp báo cáo + chạy thật + failure analysis
**Description:** Mở rộng `run_evals.py`/`rag_comparison.py` gọi scope + temporal eval, gộp `reports/benchmark_results.json`. Chạy thật (cần GROQ/GEMINI key). Viết `reports/failure_analysis.md` (mọi case sai + phân loại nguyên nhân).
**Acceptance:**
- [ ] `python eval/run_evals.py --skip-ragas` chạy sạch, in retrieval + scope + temporal
- [ ] `reports/benchmark_results.json` số thật 4 strategy + scope + temporal
- [ ] hit_rate không 1.0 đồng loạt (hoặc kèm ghi chú corpus size + kế hoạch mở rộng)
- [ ] `reports/failure_analysis.md` mổ ≥1 case sai
**Verification:** chạy `eval/run_evals.py --skip-ragas` + đọc report
**Dependencies:** T25 · **Files:** `eval/run_evals.py`, `eval/rag_comparison.py`, `reports/*` · **Scope:** M

### ☑ Checkpoint C7
- [ ] `reports/benchmark_results.json` số thật, khớp cách sẽ ghi ở README
- [ ] `reports/failure_analysis.md` có nội dung
- [ ] `git log`: gold set commit trước eval commit

---

## Phase 4c — frontend-boundary-ui

### T27: API `/corpus/manifest` + `/questions/library`
**Description:** 2 endpoint GET: manifest trả `{documents:[{so_hieu,ten,ngay_hieu_luc}], updated_at, out_of_scope:[...], earliest_point_in_time}`; library trả `data/questions/library.json`.
**Acceptance:**
- [ ] `GET /corpus/manifest` + `GET /questions/library` trả JSON đúng shape
- [ ] Nội dung khớp `corpus_manifest.yaml` + `library.json`
**Verification:** `pytest tests/test_api*.py -v` (hoặc thêm) + curl
**Dependencies:** C5 · **Files:** `src/api/routes/*.py`, `src/api/schemas.py` · **Scope:** S

### T28: `CorpusManifestPanel` + `QuestionLibrary`
**Description:** Component React: panel manifest (danh sách VB + "ngoài phạm vi" collapsible + `updated_at`); thư viện câu hỏi (nhóm tình huống → chips, click điền ô hỏi). CSS theo palette hiện có. Mở rộng `main.tsx`, giữ dark-mode + streaming + citation drawer.
**Acceptance:**
- [ ] Trang chủ (chưa hỏi) hiển thị: số VB + ngày cập nhật + danh sách ngoài phạm vi + ≥1 nhóm câu hỏi
- [ ] Click câu hỏi mẫu → điền ô input
**Verification:** `cd frontend && npm run build` sạch + `npm run dev` không lỗi console
**Dependencies:** T27 · **Files:** `frontend/src/main.tsx`, `frontend/src/styles.css` · **Scope:** M

### T29: `AsOfDatePicker` + temporal badge + `RefusalCard`
**Description:** Picker `as_of_date` (mặc định hôm nay, hiện khi ≠ hôm nay); badge "Điều này đã được sửa đổi — bản có hiệu lực tại <ngày>" khi metadata chunk có `superseded_by`/`amended_by_doc`; `RefusalCard` khi response `decision != IN_SCOPE` (phạm vi + `out_of_scope_kind` + `suggested_questions` click gửi luôn).
**Acceptance:**
- [ ] `as_of_date=2025-01-01` "lương tối thiểu vùng I" → 4.96tr + nêu mốc; `2026-02-01` → 5.31tr
- [ ] Câu OOC → `RefusalCard` (phạm vi + gợi ý click được), KHÔNG câu trả lời bịa
- [ ] Điều 139 với `as_of` sau 2026-07-01 → badge "đã được sửa đổi"
**Verification:** `npm run build` + kiểm tay 5 luồng (verify trình duyệt gộp 1 lần cuối, screenshot viewport, đóng browser ngay — theo `~/.claude/CLAUDE.md`)
**Dependencies:** T28 · **Files:** `frontend/src/main.tsx`, `frontend/src/styles.css` · **Scope:** M

### ☑ Checkpoint C8
- [ ] `npm run build` sạch
- [ ] 5 luồng UI: trang chủ / in-scope + as_of / OOC refusal / temporal badge / thư viện câu hỏi — 1 lần verify trình duyệt

---

## Phase 5 — docs-rewrite

### T30: Viết lại `README.md`
**Description:** Executive overview vertical lao động + vấn đề gốc "corpus có biên giới". Key capabilities (headline = scope classifier + manifest + thư viện câu hỏi; #2 = temporal-aware retrieval + hiệu lực cấp điều/khoản). Corpus section (N văn bản, mốc 2015→nay, "ngoài phạm vi"). Benchmark section CHỈ số từ `reports/benchmark_results.json` + link EVALUATION + failure_analysis + nêu thẳng hạn chế. Cập nhật badge test count, mermaid diagram (thêm `scope_gate`, bỏ nhánh banking).
**Acceptance:**
- [ ] `grep -ri "ngân hàng\|banking\|NHNN\|SBV\|DTI\|39/2016" README.md` rỗng
- [ ] Mọi số truy về `reports/benchmark_results.json`
- [ ] Tính năng thời gian mô tả đúng tên (không "reconstruction")
**Verification:** grep + đọc toàn bộ README đối chiếu số
**Dependencies:** C7, C8 · **Files:** `README.md` · **Scope:** M

### T31: `EVALUATION.md` + `CORPUS_SPEC.md` + grep + memory
**Description:** `EVALUATION.md` viết lại (methodology gold-first, 3 nhóm câu, metrics, bảng số thật, failure analysis, section "Hạn chế"). `docs/corpus/CORPUS_SPEC.md` bỏ nhãn DRAFT, đồng bộ schema thực tế. Grep sạch toàn `docs/`. Cập nhật memory `documind-benchmark-integrity` + `documind-labor-pivot` (trạng thái "đã implement").
**Acceptance:**
- [ ] `grep -ri "ngân hàng\|banking\|NHNN\|SBV" README.md EVALUATION.md docs/` rỗng
- [ ] EVALUATION.md có section "Hạn chế"; mọi số khớp report
- [ ] Memory cập nhật
**Verification:** grep checks + đọc EVALUATION đối chiếu report
**Dependencies:** T30 · **Files:** `EVALUATION.md`, `docs/corpus/CORPUS_SPEC.md`, `~/.claude/.../memory/*.md` · **Scope:** M

### ☑ Checkpoint C9 (Complete)
- [ ] Mọi acceptance criteria toàn initiative (SPEC.md §Success criteria) đạt
- [ ] `pytest` + `ruff check src tests` + `npm run build` xanh
- [ ] grep banking rỗng; số README ↔ report khớp
- [ ] Rebase 1 commit/nhánh + PR theo `~/.claude/gitflow.md`
- [ ] Review với Ted
