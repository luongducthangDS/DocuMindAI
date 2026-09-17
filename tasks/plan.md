# Implementation Plan: DocuMind AI — Pivot Lao động / Tiền lương / BHXH

> Nguồn: `docs/spec/SPEC.md` + 9 file `docs/spec/SPEC-<module>.md`. Điểm bắt lại: memory `documind-labor-pivot` (Session 5).
> Task chi tiết (acceptance / verify / files): `tasks/todo.md`.

## Overview

Đổi DocuMind từ vertical ngân hàng sang lao động–tiền lương–BHXH khu vực tư VN, corpus dựng lại từ nguồn chính thức. Điểm bán = **corpus có biên giới** (manifest hiển thị + scope classifier fail-closed + thư viện câu hỏi theo tình huống). Tính năng thời gian làm ở mức "temporal-aware retrieval + hiệu lực cấp điều/khoản", KHÔNG bán là "point-in-time reconstruction".

9 module, 31 task, 6 phase. Build bottom-up theo dependency graph; mỗi task để hệ thống ở trạng thái chạy được; checkpoint sau mỗi 2–3 task.

## Architecture Decisions

- **Không thêm dependency.** scope_gate / temporal filter / question library tái dùng LLM client (Groq→Gemini chain) + embedder + config sẵn có.
- **Không sửa retrieval node** (`src/rag/retriever.py`). Temporal filter là node riêng `do_temporal_filter` giữa `do_retrieve` và `do_grade` — testable, đúng "tầng trên retrieval".
- **scope_gate = node mới trước router**, thay thế vai trò intent `unknown`. 2 tầng: rule prefilter (`data/scope/topics.yaml`) → LLM classifier. **Fail-closed** khi cả 2 tầng không kết luận (message "hệ thống tạm thời chưa phân loại", log `error`, không đổ "ngoài phạm vi").
- **Schema version cấp khoản** sở hữu bởi `clause-schema-ingestion`; đa số khoản 1 version, chỉ khoản sửa in-place (≈ chỉ Điều 139 k1 BLLĐ) có ≥2. Mở rộng `chunk_by_dieu`, giữ metadata cũ (backward-compat retriever/generator).
- **`as_of_date`** luồng: UI → `POST /query` body → `AgentState` → `do_temporal_filter` + generator prompt. Mặc định = hôm nay.
- **compliance engine**: mở rộng `src/rag/compliance.py` (thêm `effective_from/to` + `criterion_kind=formula_only`), giữ cơ chế match→extract→evaluate. Criteria hardcode ~6–12 mục, đối chiếu toàn văn.
- **Gold set viết + commit TRƯỚC khi chạy hệ thống** (memory `eval-methodology`). Commit riêng, đứng trước commit "run eval".
- **docs-rewrite làm cuối** — cần số thật từ `reports/benchmark_results.json`.

## Task List (index — chi tiết ở `tasks/todo.md`)

### Phase 0: Decision gate (Ted) — trước khi code
- [ ] D1: Chốt xử lý nhánh `feature/llm-openai-backup` + tạo nhánh feature cho pivot
- [ ] D2: Chốt nguồn lấy toàn văn corpus (vbpl.vn là SPA)

### Phase 1: `corpus-acquisition` (research + data, ít code)
- [ ] T1: Xoá tài liệu ngân hàng bịa + audit `data/raw/` thực tế
- [ ] T2: Tra 1 vòng 6 dòng UNVERIFIED, loại nếu fail
- [ ] T3: Đối chiếu Công báo các mốc ngày PARTIAL
- [ ] T4: Đối chiếu lược đồ vbpl.vn quan hệ sửa đổi + xác nhận điều BLLĐ bị sửa in-place
- [ ] T5: Acquire toàn văn sạch 18 VB + VBHN BLLĐ vào `data/raw/lao_dong/`
- [ ] T6: Lấy 2 bản text Điều 139 khoản 1 (2021 + 2026-07-01)
- [ ] **Checkpoint C1:** manifest VERIFIED, `data/raw/lao_dong/` đủ file toàn văn

### Phase 2: `clause-schema-ingestion`
- [ ] T7: Manifest loader YAML + bản đồ doc→hiệu lực + danh sách khoản sửa in-place
- [ ] T8: Mở rộng `chunker.py` — `clause_uid` + metadata version
- [ ] T9: Dựng 2-version cho Điều 139 k1
- [ ] T10: Mở rộng `scripts/ingest_documents.py` (đọc `data/raw/lao_dong` + YAML), loại UNVERIFIED
- [ ] T11: `scripts/validate_corpus.py` + mở rộng `tests/test_ingestion.py`
- [ ] **Checkpoint C2:** ingest chạy, index có clause metadata, `pytest tests/test_ingestion.py test_rag.py` xanh

### Phase 3a: `temporal-retrieval`
- [ ] T12: `src/rag/temporal.py` + `tests/test_temporal.py`
- [ ] T13: `AgentState` + `QueryRequest` + `run_agent` thêm `as_of_date` / `time_out_of_range`
- [ ] T14: Node `do_temporal_filter` vào graph + generator nhận `as_of_date`
- [ ] **Checkpoint C3:** 3 mốc `as_of_date` cho kết quả khác nhau; regression xanh

### Phase 3b: `scope-gate`
- [ ] T15: `data/scope/topics.yaml` (script sinh nháp + review)
- [ ] T16: `src/agent/scope_gate.py` — prefilter + LLM + fail-closed
- [ ] T17: Fixture `scope_cases.json` (~35 câu, viết TRƯỚC) + `tests/test_scope_gate.py`
- [ ] T18: Graph `do_scope_gate` + `do_refusal` + templates; gỡ vai trò intent `unknown`
- [ ] **Checkpoint C4:** OOC bị từ chối kèm gợi ý, fail-closed hoạt động, regression xanh

### Phase 3c: `question-library`
- [ ] T19: `scripts/gen_question_library.py` + review → `data/questions/library.json`
- [ ] T20: `src/agent/question_library.py` + test + wire vào `scope_gate.suggested_questions`
- [ ] **Checkpoint C5:** thư viện đủ 40–60 câu map `clause_uid` tồn tại; scope_gate trả gợi ý thật

### Phase 4a: `compliance-criteria`
- [ ] T21: Viết `data/compliance/criteria.json` lao động (≥6 tiêu chí, đối chiếu toàn văn)
- [ ] T22: Mở rộng `compliance.py` (`effective_from/to` + `formula_only`) + regex nhãn VN
- [ ] T23: `compliance_check_node` + `_render_compliance_answer` xử lý `verdict=formula`; `tests/test_compliance.py` mới
- [ ] **Checkpoint C6:** 4 tình huống (2 threshold, 1 temporal, 1 formula) đúng; regression xanh

### Phase 4b: `eval-goldset`
- [ ] T24: Viết `data/eval/test_questions.json` (≥50 câu, nhóm A/B/C) + test cấu trúc — **commit riêng**
- [ ] T25: `eval/scope_eval.py` + `eval/temporal_eval.py`
- [ ] T26: Mở rộng `run_evals.py` gộp báo cáo; chạy thật; `reports/failure_analysis.md`
- [ ] **Checkpoint C7:** `reports/benchmark_results.json` có số thật; failure analysis có ≥1 case

### Phase 4c: `frontend-boundary-ui`
- [ ] T27: API `/corpus/manifest` + `/questions/library`
- [ ] T28: `CorpusManifestPanel` + `QuestionLibrary` components
- [ ] T29: `AsOfDatePicker` + temporal badge + `RefusalCard`
- [ ] **Checkpoint C8:** `npm run build` sạch; 5 luồng UI kiểm tay (verify trình duyệt gộp 1 lần)

### Phase 5: `docs-rewrite`
- [ ] T30: Viết lại `README.md`
- [ ] T31: `EVALUATION.md` + `CORPUS_SPEC.md` + grep checks + cập nhật memory
- [ ] **Checkpoint C9:** `grep -ri "ngân hàng|banking|NHNN|SBV"` README/EVALUATION rỗng; mọi số khớp report

## Parallelization

- Sau C2: Phase 3a / 3b / 3c độc lập (chạm file khác nhau: `temporal.py` / `scope_gate.py` / `question_library.py`). `AgentState` là điểm chạm chung → làm T13 trước, chốt keys, rồi 3b/3c dùng.
- Sau C5: Phase 4a / 4b / 4c độc lập một phần. 4b (eval) cần 3a+3b+3c xong. 4c (frontend) cần API contract từ 3a (`as_of_date`) + 3b (refusal shape).
- Theo `~/.claude/CLAUDE.md`: chỉ spawn subagent khi thật sự song song; mặc định làm tuần tự inline. Nếu Ted muốn parallel, cụm 3a/3b/3c là ứng viên.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Không lấy được toàn văn sạch (vbpl.vn SPA, thuvienphapluat chặn bot) | **High** — chặn cả pipeline | D2 chốt nguồn trước; fallback: Ted tải tay vào `data/raw/lao_dong/`; ưu tiên chinhphu.vn/congbao text |
| Quan hệ sửa đổi in-place BLLĐ sai (nguồn thứ cấp) | Med — sai version Điều 139 = sai demo chính của trục thời gian | T4 đối chiếu vbpl.vn + toàn văn luật sửa; N1 đã bound ~1 case nên rủi ro giới hạn |
| scope_gate nhầm câu in-scope thành OOC (fail-closed làm nặng thêm) | Med — demo tệ khi từ chối oan | T17 fixture ≥35 câu viết trước; prefilter bắt in-scope chắc chắn để né LLM; đo precision/recall ở T25 |
| Corpus nhỏ → hit_rate eval = 1.0 đồng loạt (red flag CV) | Med | T24 thêm câu nhiễu/khó; nếu vẫn 100% → ghi rõ kích thước corpus + kế hoạch mở rộng ở EVALUATION |
| ChromaDB metadata không nhận `None`/`date` | Low | T8 chốt quy ước (`effective_to` rỗng = `""` hoặc `9999-12-31`), test ở T11 |
| RAGAS đốt quota Gemini | Low | Chỉ dùng 3.1/3.5-flash-lite (RPD 500); workflow 2 bước `--skip-ragas` rồi judge từ cache |
| Nhánh `feature/llm-openai-backup` chưa push → conflict khi tạo nhánh pivot | Low | D1 chốt push/merge trước |

## Open Questions

Từ `SPEC.md` (6) + rải trong module spec. Cần Ted trước/trong Phase tương ứng:
1. **D1 — nhánh git:** push `feature/llm-openai-backup` lên `origin` rồi tạo `feature/labor-pivot` từ đó? hay merge vào `main` trước? (repo không có `develop`).
2. **D2 — nguồn corpus:** Ted tự tải toàn văn 18 VB + VBHN vào `data/raw/lao_dong/`, hay em thử luatvietnam.vn / chinhphu.vn trước? VB lịch sử (BHXH 2014, Việc làm 2013) + VBHN lấy ở đâu?
3. **T15/T16 — ngưỡng `confidence`** prefilter↔LLM: chốt 0.85 giờ hay tinh chỉnh sau T25 eval? (đề xuất: 0.85 tạm, chỉnh sau).
4. **T19 — question library:** sinh bằng LLM (offline, review từng câu) hay viết tay 100%? 40–60 câu.
5. **T21 — compliance:** giữ hardcode `criteria.json` ~6 tiêu chí, hay mở tới ~12 (thêm nghỉ phép năm, lương thử việc 85%, BHXH một lần, mức hưởng BHTN 60%)?
6. **Deploy demo sống** (Render/Vercel): trong initiative này hay module riêng sau? (memory `portfolio-cv-selection`: demo 1 sống + 2 video).

## Definition of Done (mọi task)

- `pytest` + `ruff check src tests` xanh (test liên quan tối thiểu).
- Không thêm dependency (nếu buộc phải → dừng, hỏi Ted).
- User-facing text tiếng Việt; số/ngày pháp lý kèm nguồn VERIFIED.
- Diff review trước commit; commit `<type>(<scope>): <subject>`.
- Không báo "đã pass" nếu chưa chạy được thật.
