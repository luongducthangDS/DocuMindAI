# Feature Specification: Corpus có biên giới cho vertical Lao động – Tiền lương – BHXH

**Feature Branch**: `001-labor-pivot`

**Created**: 2026-09-18

**Status**: Draft (reverse-engineered từ `docs/spec/SPEC.md` và các SPEC-module)

**Input**: User description: "Pivot vertical sang lao động – tiền lương – BHXH với corpus có biên giới (manifest công khai, scope_gate fail-closed), temporal-aware retrieval theo hiệu lực cấp điều/khoản, thư viện câu hỏi theo tình huống, và eval gold set >= 50 câu viết trước"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Biết trước hệ thống trả lời được gì (Priority: P1)

Người lao động / HR / kế toán tiền lương mở trang chủ và thấy ngay hệ thống bao phủ những văn bản nào, cập nhật đến ngày nào, và những loại việc nào nó không xử lý — trước khi gõ câu hỏi đầu tiên.

**Why this priority**: Vấn đề gốc không phải chất lượng retrieval mà là corpus không có biên giới. Người dùng không biết hỏi gì được nên không tin câu trả lời. Đây là điểm bán chính của định vị (a) và là thứ một trợ lý đa phòng ban chung chung không có.

**Independent Test**: Mở trang chủ khi chưa hỏi gì; đối chiếu danh sách văn bản hiển thị với manifest corpus. Tự nó đã có giá trị kể cả khi chưa làm temporal hay question library.

**Acceptance Scenarios**:

1. **Given** người dùng mở trang chủ lần đầu, **When** trang tải xong, **Then** hiển thị đúng số văn bản đã được VERIFIED, ngày cập nhật corpus, và danh sách "ngoài phạm vi" viết thẳng bằng ngôn ngữ người dùng.
2. **Given** corpus được nạp thêm hoặc bớt một văn bản, **When** người dùng tải lại trang chủ, **Then** số văn bản và ngày cập nhật đổi theo, không cần sửa nội dung giao diện thủ công.

---

### User Story 2 - Bị từ chối một cách hữu ích khi hỏi ngoài phạm vi (Priority: P1)

Người dùng hỏi một câu nằm ngoài lao động – tiền lương – BHXH. Hệ thống từ chối trả lời, nói rõ phạm vi của mình, nói rõ loại việc nó không làm, và gợi ý tối đa 3 câu gần nhất mà nó trả lời được.

**Why this priority**: Một lần trả lời bừa câu ngoài phạm vi phá hỏng lòng tin nhiều hơn mười lần trả lời đúng. Đây là phần cưỡng chế của biên giới corpus — manifest chỉ là lời hứa, gate mới là thực thi.

**Independent Test**: Chạy bộ 12 loại câu ngoài phạm vi; mỗi câu phải bị từ chối kèm đủ ba thành phần (phạm vi, loại việc không xử lý, gợi ý thay thế).

**Acceptance Scenarios**:

1. **Given** câu hỏi thuộc một trong 12 loại ngoài phạm vi, **When** người dùng gửi câu hỏi, **Then** hệ thống từ chối trước khi truy hồi tài liệu và không hiển thị nguồn nào.
2. **Given** bộ phân loại phạm vi gặp lỗi kỹ thuật, **When** người dùng gửi câu hỏi, **Then** hệ thống báo "tạm thời không phân loại được" và không trả lời — không được đoán là ngoài phạm vi, cũng không được bỏ qua gate để trả lời.
3. **Given** câu hỏi nằm trong phạm vi, **When** người dùng gửi câu hỏi, **Then** gate cho đi tiếp và câu trả lời kèm trích dẫn cấp điều/khoản.

---

### User Story 3 - Câu trả lời đúng theo thời điểm hiệu lực (Priority: P2)

Người dùng cần biết quy định áp dụng tại một thời điểm cụ thể (hôm nay, hoặc một ngày trong quá khứ khi sự việc xảy ra). Hệ thống chỉ trả lời bằng bản điều/khoản có hiệu lực tại thời điểm đó.

**Why this priority**: Sai thời điểm hiệu lực là sai nội dung, nhưng số lượng khoản đổi in-place trong corpus đã được giới hạn (~3–12 khoản, khoảng 1 câu thực sự đổi đáp án), nên đây là chiều sâu kỹ thuật chứ không phải tính năng đầu tiên người dùng chạm vào.

**Independent Test**: Hỏi cùng một câu với hai `as_of_date` khác nhau quanh một mốc hiệu lực đã biết; đáp án và trích dẫn phải khác nhau.

**Acceptance Scenarios**:

1. **Given** một câu hỏi về chế độ có mốc hiệu lực đã biết, **When** người dùng đặt `as_of_date` trước mốc và sau mốc, **Then** hệ thống trả về hai đáp án khác nhau, mỗi đáp án trích dẫn đúng phiên bản điều/khoản có hiệu lực tại thời điểm tương ứng.
2. **Given** người dùng không nêu thời điểm, **When** gửi câu hỏi, **Then** hệ thống dùng ngày hiện tại và nói rõ trong câu trả lời rằng đáp án áp dụng tại thời điểm nào.

---

### User Story 4 - Có sẵn câu hỏi mẫu theo tình huống (Priority: P3)

Người dùng chưa biết diễn đạt vấn đề của mình thành câu hỏi thì chọn từ thư viện 40–60 câu gom theo tình huống đời thực (nghỉ việc, thai sản, làm thêm giờ…), không gom theo tên văn bản.

**Why this priority**: Giảm ma sát lần đầu và cho người đọc repo thấy ngay năng lực hệ thống, nhưng không thay đổi tính đúng đắn của câu trả lời.

**Independent Test**: Chọn một câu bất kỳ trong thư viện, gửi đi, và nhận được câu trả lời có trích dẫn — không câu nào trong thư viện bị chính scope_gate từ chối.

**Acceptance Scenarios**:

1. **Given** thư viện câu hỏi hiển thị, **When** người dùng chọn một câu, **Then** câu đó được gửi như câu hỏi của họ và nhận được câu trả lời có trích dẫn.
2. **Given** thư viện gồm 40–60 câu, **When** kiểm tra toàn bộ, **Then** mọi câu đều nằm trong phạm vi corpus và được gom theo tình huống, không theo số hiệu văn bản.

---

### Edge Cases

- Câu hỏi nửa trong nửa ngoài phạm vi (ví dụ: lương tối thiểu — trong phạm vi; kèm tư vấn đầu tư khoản lương đó — ngoài phạm vi): hệ thống trả lời phần trong phạm vi và nêu rõ phần không xử lý.
- Câu hỏi về thời điểm nằm ngoài khoảng hiệu lực mà corpus bao phủ: hệ thống nói rõ corpus không phủ mốc đó thay vì trả bản gần nhất.
- Corpus rỗng hoặc vector store không mở được: trang chủ không được hiển thị số văn bản cũ đã cache; hệ thống báo trạng thái không sẵn sàng.
- Truy hồi trả về kết quả nhưng tất cả đều dưới ngưỡng liên quan: trả lời "không tìm thấy" và gợi ý bổ sung tài liệu, không gọi mô hình sinh.
- Một điều/khoản bị sửa in-place nhiều lần: mỗi phiên bản phải phân biệt được bằng định danh riêng để trích dẫn không nhập nhằng.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Hệ thống MUST công bố manifest corpus ở trang chủ gồm số văn bản đã VERIFIED, ngày cập nhật, và danh sách loại việc ngoài phạm vi.
- **FR-002**: Manifest hiển thị MUST sinh từ dữ liệu corpus thực tế, không nhập tay trong giao diện.
- **FR-003**: Hệ thống MUST phân loại phạm vi của mọi câu hỏi TRƯỚC khi định tuyến ý định và TRƯỚC khi truy hồi.
- **FR-004**: Khi câu hỏi ngoài phạm vi, hệ thống MUST từ chối kèm đủ ba thành phần: phạm vi corpus, loại việc không xử lý, và tối đa 3 câu gần nhất trả lời được.
- **FR-005**: Khi bộ phân loại phạm vi lỗi, hệ thống MUST fail-closed — báo không phân loại được và dừng, không được mặc định "ngoài phạm vi" cũng không được bỏ qua gate.
- **FR-006**: Hệ thống MUST từ chối đúng cả 12 loại câu ngoài phạm vi đã liệt kê trong bộ kiểm thử.
- **FR-007**: Mỗi truy vấn MUST có một thời điểm hiệu lực (`as_of_date`), mặc định là ngày hiện tại.
- **FR-008**: Truy hồi MUST chỉ thấy phiên bản điều/khoản có hiệu lực tại thời điểm đó.
- **FR-009**: Mỗi điều/khoản MUST có định danh ổn định và định danh phiên bản để trích dẫn và chấm điểm không phụ thuộc vị trí văn bản.
- **FR-010**: Câu trả lời MUST nêu thời điểm hiệu lực mà nó áp dụng.
- **FR-011**: Hệ thống MUST cung cấp thư viện 40–60 câu hỏi tĩnh, gom theo tình huống người dùng.
- **FR-012**: Mọi câu trong thư viện MUST nằm trong phạm vi corpus (không câu nào bị gate từ chối).
- **FR-013**: Bộ gold set đánh giá MUST có tối thiểu 50 câu, phủ ba nhóm: tình huống, ngoài phạm vi, và thời điểm hiệu lực.
- **FR-014**: Đáp án gold MUST được viết và lưu TRƯỚC khi chạy hệ thống trên chúng.
- **FR-015**: Chấm điểm truy hồi MUST dựa trên định danh điều/khoản và phiên bản, không dựa trên trùng lặp từ.
- **FR-016**: Mọi con số đánh giá xuất hiện trong tài liệu công khai MUST kèm nguồn run và ngày chạy; số không có nguồn MUST bị gỡ bỏ.
- **FR-017**: Tài liệu công khai (README, EVALUATION) MUST mô tả đúng vertical lao động và không còn tham chiếu vertical ngân hàng.

### Key Entities

- **Văn bản (Document)**: một văn bản quy phạm trong corpus; có trạng thái xác minh, nguồn chính thức, ngày ban hành, ngày hiệu lực, ngày hết hiệu lực.
- **Điều/Khoản (Clause)**: đơn vị trích dẫn nhỏ nhất; có định danh ổn định, thuộc một văn bản, có nhiều phiên bản theo thời gian.
- **Phiên bản điều/khoản (Clause Version)**: nội dung của một điều/khoản trong một khoảng hiệu lực; là thứ được truy hồi và trích dẫn.
- **Manifest corpus**: bản công bố phạm vi — danh sách văn bản VERIFIED, ngày cập nhật, và các loại việc nằm ngoài phạm vi.
- **Câu hỏi mẫu (Library Question)**: câu hỏi tĩnh thuộc một nhóm tình huống.
- **Mục gold (Gold Item)**: câu hỏi kiểm thử kèm đáp án mong đợi, nhóm (tình huống / ngoài phạm vi / thời điểm), và các định danh điều/khoản mà hệ thống phải truy hồi được.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Người dùng lần đầu xác định được hệ thống có trả lời được vấn đề của mình hay không trong vòng 30 giây kể từ khi mở trang, mà không cần gõ câu hỏi thử.
- **SC-002**: 12/12 loại câu ngoài phạm vi bị từ chối kèm đủ ba thành phần bắt buộc; 0 trường hợp trả lời bừa.
- **SC-003**: 100% trường hợp bộ phân loại lỗi đều dừng an toàn, không có trường hợp nào lọt qua gate.
- **SC-004**: Đổi thời điểm hiệu lực quanh các mốc đã biết làm đổi đáp án ở 100% số câu temporal trong gold set.
- **SC-005**: Gold set ≥ 50 câu, 100% đáp án được viết trước khi chạy, và báo cáo đánh giá kèm phân tích thất bại cho mọi câu sai.
- **SC-006**: 0 con số đánh giá không có nguồn và 0 tham chiếu vertical ngân hàng còn lại trong tài liệu công khai.
- **SC-007**: 100% câu trả lời trong phạm vi đều kèm trích dẫn cấp điều/khoản giải được về văn bản nguồn.

## Assumptions

- Corpus được xây từ nguồn chính thức và mỗi văn bản có trạng thái xác minh thủ công trước khi vào manifest; quy trình thu thập nằm ở spec corpus riêng.
- Số khoản sửa in-place trong corpus là nhỏ (~3–12) và chỉ khoảng 1 câu hỏi thực sự đổi đáp án qua cơ chế in-place; các mốc lớn còn lại xử lý được ở mức siêu dữ liệu cấp văn bản.
- Người dùng đọc tiếng Việt; giao diện và câu trả lời bằng tiếng Việt.
- Hệ thống là công cụ tra cứu, không phải tư vấn pháp lý; không có yêu cầu định danh người dùng hay lưu hồ sơ cá nhân trong phạm vi này.
- Tái dùng hạ tầng hiện có (truy hồi lai, mô hình sinh, cấu hình); initiative này không thêm phụ thuộc chạy runtime mới.
- Thư viện câu hỏi là tĩnh trong phạm vi này; sinh câu hỏi động nằm ngoài phạm vi.
