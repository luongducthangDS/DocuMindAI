# DEC-0005: Đưa `feature/labor-pivot` vào `develop` bằng `merge --no-ff`, không squash

- **Trạng thái:** Accepted
- **Ngày:** 2026-09-17
- **Spec liên quan:** không có (quyết định về quy trình, không về code)
- **Ảnh hưởng tới:** lịch sử git nhánh `develop`

## 1. Bối cảnh — vấn đề gì buộc phải quyết

Quy ước git của dự án (`~/.claude/gitflow.md` §4) yêu cầu **squash mọi nhánh feature về 1 commit
trước khi merge**, để giữ lịch sử tuyến tính và sạch.

Nhánh `feature/labor-pivot` có 21 commit, trong đó hai commit này nằm liền nhau **theo đúng thứ tự**:

| Hash | Nội dung |
|---|---|
| `5723245` | `test(eval): 30-question temporal gold set written from full text (T24-C)` |
| `250c8a3` | `feat(eval): A/B harness for temporal retrieval + run on 30-question gold set (T25-C)` |

Thứ tự đó là thứ **kiểm chứng được từ bên ngoài**: bộ câu hỏi gold được viết và commit **trước**
khi hệ thống chạy trên nó lần đầu. Nghĩa là không ai có cơ hội chỉnh câu hỏi cho khớp với kết quả
mà hệ thống trả về. Đó là điều phân biệt một benchmark với một màn trình diễn, và nó được trích
dẫn trong `AUDIT.md` §3.4 và `reports/failure_analysis.md` §1 như bằng chứng của tính liêm chính.

Squash 21 commit thành 1 sẽ **nén hai commit đó vào cùng một commit**. Bằng chứng biến mất, và
không có cách nào dựng lại — đó là thuộc tính của lịch sử, không phải của nội dung file.

## 2. Các phương án đã cân nhắc

| # | Phương án | Được | Mất |
|---|---|---|---|
| A | **`merge --no-ff`, giữ nguyên 21 commit** | Giữ nguyên thứ tự gold → eval. Người ngoài verify được bằng `git log` | Lệch quy ước squash; `develop` có một merge commit và 21 commit lẻ |
| B | Squash 21 commit thành 1 | Đúng quy ước, lịch sử tuyến tính | Mất vĩnh viễn bằng chứng gold-first. Đánh đổi thứ đáng giá nhất lấy thứ dễ có nhất |
| C | Squash nhưng ghi lại hai hash trong commit message | Đúng quy ước, có "dấu vết" | Dấu vết trỏ tới commit không còn tồn tại trên nhánh nào — người ngoài không verify được, chỉ đọc được lời khẳng định. Đó là tài liệu, không phải bằng chứng |

## 3. Quyết định

Chọn **A**, một lần, chỉ cho nhánh này.

**Vì sao:** luật squash tồn tại để lịch sử dễ đọc — một tiện ích. Thứ tự gold → eval là bằng chứng
kiểm chứng được — một tài sản. Khi tiện ích và tài sản xung đột thì tài sản thắng. C bị loại vì nó
tạo ra hình thức của bằng chứng mà không có bản chất: một hash không còn reachable thì không ai
kiểm tra được.

**Biên giới của ngoại lệ:** chỉ áp dụng cho `feature/labor-pivot` → `develop`, ngày 2026-09-17.
Mọi PR feature từ đó trở đi **vẫn** rebase + squash 1 commit như quy ước. Ngoại lệ tiếp theo cần
một DEC mới, với lý do riêng.

Ngoại lệ được ghi ở đây — trong repo — chứ không sửa `~/.claude/gitflow.md`: file đó là quy ước
cá nhân dùng chung cho mọi dự án của Ted, còn lý do của ngoại lệ này chỉ thuộc về repo này.

## 4. Bằng chứng

| Kiểm chứng | Lệnh | Kết quả |
|---|---|---|
| Hai commit còn reachable từ `develop` | `git log --oneline \| grep -E "5723245\|250c8a3"` | cả hai xuất hiện |
| Gold commit đứng trước eval commit | `git log --oneline --reverse 5723245~1..250c8a3` | `5723245` rồi `250c8a3` |
| Merge commit | `774007f` | `merge(develop): dua feature/labor-pivot vao develop, giu nguyen lich su 21 commit` |

## 5. Hệ quả

- **Được:** bất kỳ ai clone repo đều tự kiểm chứng được gold-first bằng `git log`, không phải tin lời README.
- **Trả giá:** `develop` không tuyến tính hoàn toàn; `git log --oneline` của develop dài hơn 21 dòng
  so với phương án squash. Quy ước squash có một vết lõm phải giải thích cho người mới — chính là
  file này.
- **Sai khi nào:** nếu về sau bằng chứng gold-first được lưu ở nơi khác chắc chắn hơn (ví dụ tag ký
  số, hoặc timestamp có công chứng bên ngoài git), thì lịch sử không còn là nơi duy nhất giữ nó và
  ngoại lệ này hết lý do tồn tại.

## 6. Câu hỏi phỏng vấn tự đặt

1. *Sao không squash cho sạch?* — Vì lịch sử ở đây không chỉ là lịch sử, nó là bằng chứng rằng bộ
   câu hỏi đo không bị viết sau khi thấy kết quả.
2. *Sao không ghi hai hash vào commit message rồi squash?* — Hash trỏ tới commit không còn
   reachable là tài liệu, không phải bằng chứng. Người ngoài không chạy `git show` lên nó được.
3. *Vậy từ giờ bỏ luôn luật squash à?* — Không. Ngoại lệ đóng khung ở đúng một nhánh, một ngày.
   Ngoại lệ sau cần DEC mới.
