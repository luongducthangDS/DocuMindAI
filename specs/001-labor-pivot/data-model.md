# Phase 1 — Data Model: 001-labor-pivot

Mô tả thực thể ở mức khái niệm, kèm ánh xạ sang chỗ đang tồn tại trong repo.

## Document (Văn bản)

Một văn bản quy phạm trong corpus.

| Trường | Ý nghĩa | Ràng buộc |
|---|---|---|
| `so_hieu` | Số hiệu văn bản | Duy nhất trong corpus |
| `ten` | Tên đầy đủ | Bắt buộc |
| `verify_status` | VERIFIED / PARTIAL / UNVERIFIED | Chỉ VERIFIED mới được công bố ở manifest (FR-001) |
| `ngay_ban_hanh`, `ngay_hieu_luc`, `ngay_het_hieu_luc` | Mốc thời gian | `ngay_hieu_luc` bắt buộc; `ngay_het_hieu_luc` rỗng = còn hiệu lực |
| `nguon` | URL nguồn chính thức | Bắt buộc với VERIFIED |

Nguồn hiện tại: `docs/corpus/corpus_manifest.yaml` (21 VERIFIED, 2 PARTIAL, 2 UNVERIFIED).

## Clause (Điều/Khoản)

Đơn vị trích dẫn nhỏ nhất.

| Trường | Ý nghĩa | Ràng buộc |
|---|---|---|
| `clause_uid` | Định danh ổn định, không phụ thuộc vị trí | Duy nhất toàn corpus (FR-009) |
| `document` | Văn bản chứa nó | Bắt buộc |

## ClauseVersion (Phiên bản điều/khoản)

Nội dung của một điều/khoản trong một khoảng hiệu lực — đây là thứ được truy hồi và trích dẫn.

| Trường | Ý nghĩa | Ràng buộc |
|---|---|---|
| `version_id` | `<clause_uid>__v<effective_from>` | Quy ước đã cài tại `src/ingestion/versions.py` |
| `effective_from` / `effective_to` | Khoảng hiệu lực | Các khoảng của cùng `clause_uid` không được chồng lấn |
| `status` | Trạng thái tại `as_of` | Suy ra, không nhập tay |
| `noi_dung` | Văn bản điều/khoản | Bắt buộc |

**Quy tắc truy hồi**: với `as_of_date` D, chỉ các version thoả `effective_from <= D < effective_to` (hoặc `effective_to` rỗng) được thấy (FR-008).

## CorpusManifest

Bản công bố phạm vi hiển thị ở trang chủ.

| Trường | Ý nghĩa |
|---|---|
| `so_van_ban_verified` | Đếm từ Document có `verify_status = VERIFIED` |
| `as_of_baseline` | Ngày cập nhật corpus |
| `earliest_point_in_time` | Mốc sớm nhất trả lời được |
| `scope_out[]` | Danh sách loại việc ngoài phạm vi, viết bằng ngôn ngữ người dùng (hiện có 5 mục) |

Suy ra hoàn toàn từ corpus — giao diện không được chứa bản sao (FR-002).

## ScopeDecision

Kết quả của cổng phạm vi cho một truy vấn.

| Trường | Giá trị |
|---|---|
| `verdict` | `in_scope` / `out_of_scope` / `undetermined` |
| `ly_do` | Vì sao, hiển thị được cho người dùng |
| `goi_y[]` | Tối đa 3 câu gần nhất trả lời được (chỉ khi `out_of_scope`) |

`undetermined` là trạng thái fail-closed: không truy hồi, không trả lời (FR-005).

## LibraryQuestion (Câu hỏi mẫu)

| Trường | Ràng buộc |
|---|---|
| `noi_dung` | Câu hỏi hoàn chỉnh |
| `tinh_huong` | Nhóm tình huống đời thực, KHÔNG phải số hiệu văn bản (FR-011) |

Bất biến: mọi câu phải cho `verdict = in_scope` (FR-012).

## GoldItem (Mục gold)

| Trường | Ràng buộc |
|---|---|
| `cau_hoi` | Bắt buộc |
| `nhom` | `tinh_huong` / `ngoai_pham_vi` / `thoi_diem` (FR-013) |
| `dap_an_mong_doi` | Viết TRƯỚC khi chạy hệ thống (FR-014) |
| `clause_uids[]` | Điều/khoản phải truy hồi được; rỗng với nhóm `ngoai_pham_vi` |
| `as_of_date` | Bắt buộc với nhóm `thoi_diem` |

Tổng ≥ 50 mục, phủ cả ba nhóm.
