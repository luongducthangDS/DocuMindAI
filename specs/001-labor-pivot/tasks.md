---

description: "Task list for 001-labor-pivot"
---

# Tasks: Corpus có biên giới cho vertical Lao động – Tiền lương – BHXH

**Input**: Design documents from `/specs/001-labor-pivot/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: Có. Spec nêu tiêu chí kiểm thử cụ thể (12 loại câu ngoài phạm vi, fail-closed, gold set ≥ 50 câu) và constitution v1.0.0 nguyên tắc V buộc test xanh trước mọi tuyên bố, nên task test là bắt buộc chứ không tuỳ chọn.

**Organization**: Nhóm theo user story để từng story giao được độc lập.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Chạy song song được (khác file, không phụ thuộc task chưa xong)
- **[Story]**: US1–US4 theo spec.md

## Path Conventions

Web app: backend `src/`, frontend `frontend/src/`, test `tests/`, dữ liệu `data/`.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Chốt môi trường chạy và số liệu nền trước khi sửa gì

- [ ] T001 Ghi lại interpreter chuẩn của dự án vào `CLAUDE.md` (đo 2026-09-18: `python` mặc định của shell trỏ sang venv của project khác và thiếu `loguru`; interpreter chạy được là `C:\Users\Lenovo\AppData\Local\Programs\Python\Python310\python.exe`)
- [ ] T002 [P] Sửa `pytest.ini` để `--collect-only` chạy được mà không cần `pytest-cov` (hiện `addopts` cứng `--cov=src` làm mọi lệnh pytest hỏng nếu thiếu plugin)
- [ ] T003 Chạy `pytest -q -o addopts=""` và ghi con số + ngày vào một chỗ duy nhất làm nguồn sự thật cho tài liệu

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Phải xong trước khi bất kỳ user story nào bắt đầu

- [ ] T004 Xác minh 2 văn bản `PARTIAL` và 2 văn bản `UNVERIFIED` trong `docs/corpus/corpus_manifest.yaml`, chuyển sang VERIFIED hoặc loại khỏi corpus; gỡ nhãn "DRAFT for review" ở đầu file khi chốt
- [ ] T005 [P] Đối chiếu danh sách văn bản trong `docs/corpus/corpus_manifest.yaml` với 21 file thực tế tại `data/raw/lao_dong/`, xử lý mọi chênh lệch
- [ ] T006 Bổ sung hàm đọc manifest trong `src/ingestion/manifest.py` trả về đúng các trường ở `contracts/corpus-manifest.md` (`so_van_ban` chỉ đếm `verify_status = VERIFIED`)
- [ ] T007 [P] Viết test cho hàm đọc manifest trong `tests/test_ingestion.py`: đếm đúng VERIFIED, bỏ PARTIAL/UNVERIFIED, lỗi khi file hỏng

**Checkpoint**: Manifest là nguồn sự thật đọc được bằng code, không còn nhãn DRAFT.

---

## Phase 3: User Story 1 — Biết trước hệ thống trả lời được gì (P1)

**Goal**: Trang chủ công bố phạm vi corpus sinh từ dữ liệu thật.

**Independent Test**: Mở trang chủ khi chưa hỏi gì; số văn bản và ngày cập nhật khớp manifest; thêm/bớt một văn bản VERIFIED → số đổi theo mà không sửa frontend.

- [ ] T008 [US1] Tạo route đọc manifest tại `src/api/routes/corpus.py` theo `contracts/corpus-manifest.md`, đăng ký vào `src/api/main.py`
- [ ] T009 [P] [US1] Viết test route manifest trong `tests/test_api.py`: trả đúng `so_van_ban`, `ngay_cap_nhat`, `van_ban[]`, `ngoai_pham_vi[]`
- [ ] T010 [US1] Viết test trạng thái lỗi trong `tests/test_api.py`: khi corpus không đọc được thì trả lỗi rõ ràng, không trả số cũ
- [ ] T011 [US1] Hiển thị khối manifest ở trang chủ trong `frontend/src/main.tsx`: số văn bản VERIFIED, ngày cập nhật, mốc sớm nhất, danh sách "ngoài phạm vi" (5 mục hiện có trong `scope_out`)
- [ ] T012 [US1] Xử lý trạng thái không sẵn sàng ở `frontend/src/main.tsx`: hiển thị thông báo, tuyệt đối không dùng số đã cache
- [ ] T013 [P] [US1] Kiểm tra `frontend/src/` không còn chuỗi cứng nào về số văn bản hay ngày cập nhật (FR-002)

**Checkpoint**: US1 giao được độc lập — đã là MVP có giá trị kể cả khi chưa làm cổng phạm vi.

---

## Phase 4: User Story 2 — Bị từ chối một cách hữu ích khi hỏi ngoài phạm vi (P1)

**Goal**: Cổng phạm vi fail-closed chặn trước mọi truy hồi.

**Independent Test**: 12 loại câu ngoài phạm vi đều bị từ chối kèm đủ ba thành phần; mô phỏng lỗi bộ phân loại → `undetermined`, không trả lời.

- [ ] T014 [US2] Tạo `src/agent/scope.py`: bộ phân loại phạm vi trả về `ScopeDecision` với `verdict` ∈ `in_scope` / `out_of_scope` / `undetermined`, `ly_do`, `goi_y[]` (tối đa 3, chỉ khi `out_of_scope`) — dùng lại LLM client hiện có, không thêm dependency
- [ ] T015 [US2] Định nghĩa 12 loại câu ngoài phạm vi thành dữ liệu kiểm thử tại `tests/fixtures/out_of_scope_questions.json` (một câu đại diện mỗi loại)
- [ ] T016 [US2] Chèn node `scope_gate` vào `src/agent/graph.py` **trước** `router`, với nhánh thoát sớm cho `out_of_scope` và `undetermined`
- [ ] T017 [US2] Đảm bảo nhánh thoát sớm không gọi `do_retrieve` và không gắn nguồn nào vào phản hồi (`src/agent/graph.py`)
- [ ] T018 [P] [US2] Viết test fail-closed trong `tests/test_agent.py`: bộ phân loại ném lỗi → `verdict = undetermined`, không trả lời, không truy hồi — không được rơi về `out_of_scope`
- [ ] T019 [P] [US2] Viết test 12/12 loại ngoài phạm vi trong `tests/test_agent.py`: mỗi câu bị từ chối và thông điệp chứa đủ phạm vi + loại việc không xử lý + ≤ 3 gợi ý
- [ ] T020 [P] [US2] Viết test câu trong phạm vi vẫn đi tiếp và vẫn có trích dẫn (`tests/test_agent.py`)
- [ ] T021 [US2] Đưa `verdict`, `thong_diep`, `goi_y[]` vào phản hồi API tại `src/api/schemas.py` và `src/api/routes/query.py` theo contract
- [ ] T022 [US2] Hiển thị ba trạng thái phân biệt được ở `frontend/src/main.tsx` (đặc biệt: `undetermined` không được trông giống từ chối phạm vi)
- [ ] T023 [US2] Xử lý câu nửa trong nửa ngoài phạm vi (edge case spec): trả lời phần trong phạm vi, nêu rõ phần không xử lý

**Checkpoint**: Nguyên tắc II của constitution chuyển từ FAIL sang PASS.

---

## Phase 5: User Story 3 — Câu trả lời đúng theo thời điểm hiệu lực (P2)

**Goal**: Đáp án gắn với `as_of_date` và nói rõ điều đó.

**Independent Test**: Cùng câu hỏi, hai `as_of_date` quanh một mốc đã biết → hai đáp án khác nhau, trích đúng `version_id` tương ứng.

- [ ] T024 [US3] Đưa thời điểm hiệu lực đang áp dụng vào câu trả lời tại `src/rag/generator.py` (FR-010 — hiện chưa có)
- [ ] T025 [P] [US3] Viết test trong `tests/test_temporal.py`: câu trả lời nêu đúng `as_of_date` đã dùng
- [ ] T026 [US3] Xử lý edge case "thời điểm ngoài khoảng corpus phủ" tại `src/rag/temporal.py`: nói rõ không phủ, không trả bản gần nhất
- [ ] T027 [P] [US3] Viết test cho trường hợp trên trong `tests/test_temporal.py`
- [ ] T028 [US3] Cho người dùng chọn thời điểm trên giao diện tại `frontend/src/main.tsx`, mặc định hôm nay

---

## Phase 6: User Story 4 — Thư viện câu hỏi theo tình huống (P3)

**Goal**: 40–60 câu tĩnh gom theo tình huống, không câu nào bị chính cổng phạm vi từ chối.

**Independent Test**: Chọn một câu bất kỳ → có câu trả lời kèm trích dẫn; chạy toàn bộ thư viện qua cổng phạm vi → 0 câu bị từ chối.

- [ ] T029 [US4] Soạn thư viện 40–60 câu tại `data/questions/library.json`, mỗi mục có `noi_dung` và `tinh_huong` (gom theo tình huống đời thực, KHÔNG theo số hiệu văn bản — FR-011)
- [ ] T030 [P] [US4] Viết test bất biến trong `tests/test_agent.py`: mọi câu trong thư viện cho `verdict = in_scope` (FR-012)
- [ ] T031 [US4] Hiển thị thư viện theo nhóm tình huống ở `frontend/src/main.tsx`, chọn một câu là gửi luôn

---

## Phase 7: Gold set & tính toàn vẹn số liệu (cross-cutting, bắt buộc)

**Purpose**: Nguyên tắc III và V của constitution. Không phải polish — đây là điều kiện để công bố bất cứ con số nào.

- [ ] T032 Gộp `data/eval/test_questions.json` (25 mục) và `data/eval/temporal_questions.json` (30 mục) thành một gold set duy nhất ≥ 50 mục với trường `nhom` ∈ `tinh_huong` / `ngoai_pham_vi` / `thoi_diem`
- [ ] T033 Soạn nhóm `ngoai_pham_vi` cho gold set — nhóm này hiện hoàn toàn chưa có, mà nó mới là thứ đo được giá trị của feature
- [ ] T034 Viết `dap_an_mong_doi` và `clause_uids[]` cho toàn bộ gold set **trước khi** chạy hệ thống trên chúng (FR-014); commit gold trước, chạy sau
- [ ] T035 [P] Cập nhật `eval/run_evals.py` để báo cáo tách theo `nhom` và kèm phân tích thất bại cho mọi câu sai
- [ ] T036 Chạy eval và lưu báo cáo có tên run + ngày vào `reports/`
- [ ] T037 Sửa badge test trong `README.md` — hiện ghi "84/84 Passing (100%)", lần chạy 2026-09-18 cho 216 passed
- [ ] T038 Sửa dòng trạng thái trong `CLAUDE.md` — hiện ghi "166/166 tests passed (đo 2026-09-16)"
- [ ] T039 Viết lại `README.md` sang vertical lao động: gỡ toàn bộ mô tả ngân hàng (DTI, thẻ tín chấp, trần lãi suất NHNN, "6 văn bản banking") — FR-017
- [ ] T040 [P] Sửa phần "Corpus & Compliance" trong `CLAUDE.md`: mô tả `data/raw/banking_docs/` với 6 văn bản ngân hàng, nhưng `data/raw/` chỉ còn `lao_dong/` với 21 văn bản
- [ ] T041 [P] Cập nhật `EVALUATION.md` chỉ giữ con số có nguồn run + ngày (FR-016)
- [ ] T042 Rà `README.md` và `EVALUATION.md` lần cuối: 0 con số không nguồn, 0 tham chiếu ngân hàng (SC-006)

---

## Phase 8: Polish & Cross-Cutting

- [ ] T043 [P] Viết ADR cho quyết định cổng phạm vi fail-closed vào `docs/decisions/` theo `TEMPLATE.md`
- [ ] T044 [P] Cập nhật `docs/spec/SPEC.md` từ DRAFT sang trạng thái đã duyệt, trỏ tới `specs/001-labor-pivot/`
- [ ] T045 Chạy `pytest -q -o addopts=""` toàn bộ và ghi lại con số cuối cùng; đồng bộ vào README/CLAUDE.md
- [ ] T046 Chạy lại toàn bộ kịch bản trong `specs/001-labor-pivot/quickstart.md`

---

## Dependencies

```text
Phase 1 (Setup) ──► Phase 2 (Foundational) ──┬──► Phase 3 (US1) ──┐
                                             ├──► Phase 4 (US2) ──┼──► Phase 7 ──► Phase 8
                                             ├──► Phase 5 (US3) ──┤
                                             └──► Phase 6 (US4) ──┘
```

- US1 chỉ cần Phase 2 (hàm đọc manifest).
- US2 độc lập với US1, nhưng US4 phụ thuộc US2 (T030 cần cổng phạm vi để kiểm bất biến).
- US3 độc lập hoàn toàn — cơ chế lọc thời điểm đã có, chỉ thiếu phần nói ra.
- Phase 7 cần US2 xong mới chấm được nhóm `ngoai_pham_vi`; các task tài liệu (T037–T042) không phụ thuộc gì và làm được ngay.

## Parallel Opportunities

- Trong Phase 2: T005 và T007 song song với T004/T006.
- Trong Phase 4: T018, T019, T020 song song sau khi T016 xong.
- T037–T042 (sửa tài liệu) chạy song song với toàn bộ phần code — chúng chỉ đụng file markdown.

## Implementation Strategy

**MVP = Phase 1 + 2 + 3 (US1)**: trang chủ công bố phạm vi. Tự nó đã giải một phần vấn đề gốc ("người dùng không biết hỏi gì được") mà chưa cần đụng vào graph.

**Increment 2 = Phase 4 (US2)**: cưỡng chế biên giới. Đây là phần đưa constitution nguyên tắc II từ FAIL sang PASS và là điểm bán chính.

**Increment 3 = Phase 5 + 6**, rồi **Phase 7** trước bất kỳ lần công bố nào ra ngoài.

**Làm ngay, không chờ**: T037–T042. Tài liệu đang mô tả một hệ thống không tồn tại; chi phí sửa là vài chục phút, chi phí để nguyên là người đọc repo mất lòng tin vào mọi con số còn lại.

---

## Phase 9: Convergence

**Nguồn**: `/speckit-converge` chạy 2026-09-18 trên codebase thực tế. Chỉ chứa việc **chưa** được task nào từ Phase 1–8 phủ.

- [ ] T047 CRITICAL Gỡ hoặc thay nhánh `compliance_check` trong `src/agent/graph.py`: nó chấm pass/fail bằng `data/compliance/criteria.json` gồm 5 tiêu chí ngân hàng (`vay_tin_chap_thu_nhap`, `ty_le_dti_cho_vay`, `han_muc_the_tin_dung_tin_chap`, `lai_suat_tien_gui_khong_ky_han`, `lai_suat_tien_gui_duoi_6_thang`) trong khi corpus chỉ còn văn bản lao động — câu trả lời pass/fail không có nguồn nào trong corpus (Constitution I, Constitution II) (contradicts)
- [ ] T048 CRITICAL Sửa prompt router tại `src/agent/graph.py` dòng ~110: mô tả `compliance_check` còn ví dụ ngân hàng ("thu nhập 15 triệu/tháng có đủ điều kiện vay tín chấp không?", "hạn mức thẻ 50 triệu…") và `unknown` định nghĩa là "không liên quan đến tài liệu **ngân hàng**" — chính bộ định tuyến đang hiểu sai phạm vi hệ thống (Constitution II) (contradicts)
- [ ] T049 CRITICAL Quyết định số phận `data/compliance/criteria.json` và `src/rag/compliance.py`: viết lại theo tiêu chí lao động/BHXH có nguồn trong corpus, hay gỡ khỏi graph. Không được để nguyên trạng thái hiện tại (Constitution I) (contradicts)
- [ ] T050 Sửa mô tả "compliance testing sandbox" trong `README.md`: `frontend/src/` chỉ có `main.tsx`, `styles.css`, `vite-env.d.ts` và không chứa chuỗi `compliance` nào — tính năng UI này không tồn tại (FR-017) (contradicts)
- [ ] T051 Mở rộng `load_manifest` trong `src/ingestion/manifest.py` để đọc `meta.scope_out` (hiện chỉ đọc `DocEntry` và `earliest_point_in_time`; `scope_out` với 5 mục chưa có đường ra code) — T006 giả định phải viết loader từ đầu, thực tế chỉ thiếu trường này (FR-001) (partial)
- [ ] T052 Thêm kiểm chứng đầu ra cho thời điểm hiệu lực trong `src/rag/generator.py`: `_as_of_block` đã yêu cầu mô hình "nêu rõ mốc thời điểm này trong câu trả lời", nhưng không có gì kiểm tra mô hình có làm hay không, và khi `as_of_date` rỗng thì block rỗng hoàn toàn — T024 giả định FR-010 chưa có, thực tế là có hướng dẫn nhưng không có ràng buộc (FR-010) (partial)
- [ ] T053 Viết test hồi quy cho `is_in_force` / `versions_in_force` trong `tests/test_temporal.py` theo đúng bất biến `effective_from <= T AND (effective_to rỗng OR effective_to > T)` — cơ chế đã chạy nhưng FR-008 không có task nào bảo vệ (FR-008) (missing)
- [ ] T054 Viết test hồi quy cho quy ước `version_id = "<clause_uid>__v<effective_from>"` tại `tests/test_ingestion.py` — quy ước đã cài trong `src/ingestion/versions.py` nhưng không có test nào khoá nó lại (FR-009) (missing)
- [ ] T055 Viết test hồi quy cho chấm điểm theo `clause_uid`/`version_id` trong `tests/test_eval_metrics.py` (FR-015) (missing)
- [ ] T056 Bổ sung test cho `is_out_of_range` trong `tests/test_temporal.py`: xử lý "thời điểm ngoài khoảng corpus phủ" đã có sẵn ở `src/rag/temporal.py` và được gọi tại `src/agent/graph.py:386` — T026 giả định phải implement mới, thực tế chỉ thiếu test (spec Edge Cases) (partial)
- [ ] T057 Rà soát và justify hoặc gỡ phần không nằm trong spec: `src/report/` + `report_node` (xuất PDF), `exercises/`, `eval/scoring_ab.py` — spec 001-labor-pivot không nhắc tới chúng (unrequested)
