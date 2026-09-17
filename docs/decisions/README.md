# Decision log

Thư mục này là **việc chính**. `src/` là hệ quả.

## Luật đảo thứ tự

Với mọi thay đổi không tầm thường (thêm node, đổi retrieval, đổi ngưỡng, thêm
dependency), thứ tự bắt buộc:

1. **Spec** (`docs/spec/SPEC-*.md`) — làm gì, biên giới ở đâu, không làm gì.
2. **Decision** (`docs/decisions/DEC-*.md`) — chọn phương án nào, **vì sao**, **số nào
   chứng minh**, bỏ phương án nào và vì sao bỏ.
3. **Code** — agent viết cũng được, nhưng chỉ sau khi 1 và 2 đã chốt.
4. **Số đo** — quay lại điền vào mục `Bằng chứng` của DEC. DEC không có số là DEC chưa xong.

Quy tắc kiểm tra một DEC đạt chuẩn: **đọc DEC xong, tự viết lại được code**. Nếu DEC chỉ
mô tả code đã viết thì nó là changelog, không phải decision log.

Thuật ngữ dùng trong các DEC được định nghĩa ở [docs/GLOSSARY.md](../GLOSSARY.md).

## Khi nào cần một DEC

| Cần DEC | Không cần |
|---|---|
| Thêm/bỏ một node trong agent graph | Đổi tên biến, format |
| Đổi ngưỡng, `top_k`, `top_n`, `k` của RRF | Sửa typo trong prompt |
| Chọn thư viện thay vì tự viết (hoặc ngược lại) | Thêm test cho hành vi đã có |
| Đổi contract API / schema | Bump version dependency không đổi hành vi |
| Bất cứ chỗ nào sau này bị hỏi "sao lại làm thế?" | |

## Trạng thái

`Proposed` → `Accepted` → (`Superseded by DEC-XXXX` | `Rejected`).
DEC đã `Accepted` **không sửa nội dung** — sai thì viết DEC mới thay thế. Lịch sử sai
cũng là thứ đem đi phỏng vấn được.

## Danh sách

| ID | Tiêu đề | Trạng thái | Ngày |
|---|---|---|---|
| [DEC-0001](DEC-0001-temporal-filter-node.md) | Lọc hiệu lực bằng node riêng thay vì nhét vào prompt | Accepted | 2026-09-15 |
| [DEC-0002](DEC-0002-retrieval-core-tu-viet.md) | Tự viết lõi retrieval thay vì bọc LlamaIndex | Proposed | 2026-09-15 |
