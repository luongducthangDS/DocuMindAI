# Contract: Corpus manifest (backend → frontend)

**Mục đích**: cấp cho trang chủ đúng những gì FR-001 yêu cầu công bố, sinh từ corpus thật.

**Hình thức**: một endpoint đọc, không tham số, không cần xác thực.

**Trả về**:

| Trường | Kiểu | Ý nghĩa |
|---|---|---|
| `so_van_ban` | số nguyên | Số văn bản `VERIFIED` |
| `ngay_cap_nhat` | ngày | `as_of_baseline` của corpus |
| `moc_som_nhat` | ngày | Thời điểm sớm nhất corpus trả lời được |
| `van_ban[]` | danh sách | `{so_hieu, ten, ngay_hieu_luc}` của các văn bản VERIFIED |
| `ngoai_pham_vi[]` | danh sách chuỗi | Loại việc hệ thống không xử lý |

**Quy tắc**:

- Chỉ văn bản `VERIFIED` xuất hiện trong `van_ban[]` và trong `so_van_ban`.
- Khi corpus không đọc được: trả lỗi rõ ràng. Frontend hiển thị trạng thái không sẵn sàng, KHÔNG dùng số cũ đã cache (edge case trong spec).
- Frontend không được chứa bản sao cứng của bất kỳ trường nào ở trên.

---

# Contract: Kết quả cổng phạm vi (trong câu trả lời truy vấn)

**Mục đích**: người dùng và bộ eval phân biệt được ba trạng thái.

| Trường | Kiểu | Ý nghĩa |
|---|---|---|
| `verdict` | `in_scope` / `out_of_scope` / `undetermined` | Quyết định của cổng |
| `thong_diep` | chuỗi | Với `out_of_scope`: nêu phạm vi + loại việc không xử lý. Với `undetermined`: báo không phân loại được |
| `goi_y[]` | ≤ 3 chuỗi | Câu gần nhất trả lời được; rỗng khi không phải `out_of_scope` |
| `as_of_date` | ngày | Thời điểm hiệu lực áp dụng cho câu trả lời (FR-010) |

**Quy tắc**:

- `out_of_scope` và `undetermined` MUST không kèm nguồn trích dẫn nào.
- `undetermined` MUST không bao giờ được suy ra từ lỗi kỹ thuật thành `out_of_scope`.
- `in_scope` MUST kèm trích dẫn giải được về `version_id` (FR-007 của nguyên tắc I).
