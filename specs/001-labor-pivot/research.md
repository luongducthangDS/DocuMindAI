# Phase 0 — Research: 001-labor-pivot

Mục đích: chốt các lựa chọn còn mở trước khi thiết kế, dựa trên hiện trạng repo (đo 2026-09-18) chứ không dựa trên mô tả trong tài liệu.

## R-01. Cổng phạm vi đặt ở đâu trong graph

**Decision**: Một node `scope_gate` chạy trước `router`, trả về một trong ba trạng thái: `in_scope`, `out_of_scope`, `undetermined`.

**Rationale**: `src/agent/graph.py` hiện đi contextualize → router → retrieve → temporal_filter → grade → answer. Đặt sau router thì một câu ngoài phạm vi vẫn được gán intent và vẫn tốn truy hồi; đặt sau retrieve thì đoạn văn ngoài phạm vi đã nằm trong ngữ cảnh và có thể rò vào câu trả lời. Đặt trước router là điểm sớm nhất còn giữ được câu hỏi đã contextualize (cần cho câu hỏi nối tiếp).

**Alternatives considered**: (a) lọc theo ngưỡng điểm truy hồi — bị loại vì điểm thấp không phân biệt được "ngoài phạm vi" với "trong phạm vi nhưng diễn đạt lạ"; (b) để prompt của mô hình sinh tự từ chối — bị loại vì đó chính là hành vi không kiểm chứng được mà nguyên tắc I cấm.

## R-02. Hành vi khi bộ phân loại lỗi

**Decision**: Fail-closed thành `undetermined` — báo "hệ thống tạm thời không phân loại được", không trả lời, không truy hồi.

**Rationale**: Hai chế độ hỏng đều tệ nhưng không đối xứng. Coi lỗi là `in_scope` thì hệ thống trả lời câu nó không nên trả lời — vi phạm nguyên tắc I và II. Coi lỗi là `out_of_scope` thì người dùng nhận một lời từ chối sai kèm mô tả phạm vi không liên quan, và không có cách nào biết đó là lỗi kỹ thuật. Trạng thái thứ ba nói thật với người dùng.

**Alternatives considered**: retry im lặng rồi mới fail — giữ được nếu rẻ, nhưng không thay thế được trạng thái thứ ba; xếp vào tầng thực thi, không phải quyết định spec.

## R-03. Nguồn sự thật cho manifest hiển thị

**Decision**: `docs/corpus/corpus_manifest.yaml`, đọc qua một endpoint backend, không nhúng số liệu vào frontend.

**Rationale**: File này đã tồn tại và đã mang đúng dữ liệu cần: `verify_status` theo từng văn bản (đếm được 21 VERIFIED, 2 PARTIAL, 2 UNVERIFIED), `as_of_baseline: 2026-09-09`, và `scope_out` gồm 5 mục viết sẵn bằng ngôn ngữ người dùng. FR-002 cấm nhập tay trong giao diện, nên frontend chỉ được hiển thị thứ backend đọc ra.

**Điểm cần lưu ý**: manifest tự ghi "DRAFT for review" ở dòng đầu, và 4 văn bản chưa VERIFIED. Trước khi trang chủ công bố con số, phải chốt: hiển thị chỉ VERIFIED (21) hay hiển thị cả trạng thái. Chọn **chỉ VERIFIED** để đúng chữ trong FR-001, và liệt kê riêng phần đang xác minh nếu cần.

**Alternatives considered**: sinh manifest từ chính vector store — bị loại vì store chứa chunk đã nạp, không mang trạng thái xác minh và không biết văn bản nào cố ý chưa nạp.

## R-04. Lọc thời điểm hiệu lực — dùng lại hay làm mới

**Decision**: Dùng lại nguyên `temporal_filter_node` và `src/rag/temporal.py` đã có; không sửa `retrieve_node`.

**Rationale**: Ràng buộc trong `docs/spec/SPEC-temporal-retrieval.md` là lọc phải nằm ở tầng trên truy hồi. Hiện trạng đã đúng như vậy: `as_of_date` đã đi từ `src/api/schemas.py` qua `src/api/routes/query.py` vào graph, và đã có test bảo vệ đường đi này (`TestQueryRouteForwardsAsOfDate`). Phần còn thiếu không phải cơ chế lọc mà là FR-010: câu trả lời chưa nói nó áp dụng cho thời điểm nào.

**Alternatives considered**: đẩy filter xuống truy vấn vector store bằng metadata — nhanh hơn nhưng khóa chặt vào provider, mâu thuẫn với nguyên tắc IV (`vector_backend.py` che provider).

## R-05. Định danh điều/khoản và phiên bản

**Decision**: Giữ quy ước đã cài trong `src/ingestion/versions.py`: `version_id = "<clause_uid>__v<effective_from>"`, kèm `effective_from` / `effective_to` / `status` trên metadata chunk.

**Rationale**: Quy ước này đã được ADR hoá (DEC-0004) và đã là đơn vị chấm điểm trong `eval/metrics.py`. FR-009 và FR-015 mô tả đúng thứ đang chạy — đây là phần hiếm hoi spec và code đã khớp.

## R-06. Gold set: mở rộng hay viết lại

**Decision**: Viết lại thành một gold set thống nhất ≥ 50 câu với trường `nhom` ∈ {tình huống, ngoài phạm vi, thời điểm}, gộp cả nội dung temporal hiện có.

**Rationale**: Đo thực tế: `data/eval/test_questions.json` có 25 mục, `data/eval/temporal_questions.json` có 30 mục, hai file rời nhau và không file nào có nhóm "ngoài phạm vi" — mà chính nhóm đó mới đo được thứ feature này bán. Giữ hai file rời thì không có chỗ nào trả lời được câu "hệ thống đúng bao nhiêu phần trăm".

**Ràng buộc bắt buộc**: đáp án phải viết trước khi chạy hệ thống (FR-014, nguyên tắc III). Sinh gold từ output hệ thống rồi sửa lại là tự chấm điểm mình.

**Alternatives considered**: giữ hai file và cộng số — bị loại vì 25 + 30 = 55 nhưng không phủ nhóm ngoài phạm vi, tức vẫn không đạt FR-013.

## R-07. Nợ tài liệu phát hiện khi khảo sát

**Decision**: Đưa vào tasks như hạng mục bắt buộc, không để "dọn sau".

**Số liệu đo được 2026-09-18**:

- `pytest -q` → **216 passed**. README.md ghi badge "84/84 Passing (100%)"; CLAUDE.md ghi "166/166 tests passed (đo 2026-09-16)". Không con số nào đúng.
- README.md mô tả toàn bộ sản phẩm là vertical **ngân hàng** (DTI, thẻ tín chấp, trần lãi suất NHNN) trong khi corpus thực tế tại `data/raw/lao_dong/` gồm 21 văn bản lao động/BHXH và `docs/spec/SPEC.md` đã chốt pivot.
- CLAUDE.md mô tả corpus ngân hàng 6 văn bản ở `data/raw/banking_docs/` — thư mục đó không còn trong `data/raw/` (chỉ còn `lao_dong/`).

**Rationale**: Nguyên tắc III cấm số liệu không nguồn và nguyên tắc V buộc con số công bố khớp lần chạy thật; FR-016 và FR-017 nói thẳng điều này. Đây cũng là rủi ro lớn nhất về mặt người đọc: repo được đọc như bằng chứng năng lực, mà ba tài liệu đang mô tả ba hệ thống khác nhau.
