# Drill: giải thích file đóng

Bài kiểm tra tự chạy, 20 phút, mỗi ngày một file. Mục đích: tìm ra chỗ **ngập ngừng** —
vì đó đúng là chỗ sẽ ngập ngừng trước người phỏng vấn.

## Luật

1. **Bốc ngẫu nhiên**, không được chọn file mình nhớ:
   ```bash
   find src -name "*.py" ! -name "__init__.py" | shuf -n 1
   ```
2. Mở file, đọc **tối đa 3 phút**. Đóng lại. Không mở lại trong lúc trả lời.
3. Trả lời bằng cách **viết ra** (nói to cũng được, nhưng viết thì chấm được).
4. Mở file, tự chấm. Chấm thẳng tay.

## Thang chấm

| Mức | Nghĩa |
|---|---|
| **2** | Nói được *cái gì* + *tại sao viết thế* + *phương án bị loại* |
| **1** | Nói được *cái gì*, không nói được *tại sao* |
| **0** | Phải mở file mới nói được |

Mọi câu **1** hoặc **0** đi thẳng vào một dòng trong `docs/decisions/` — vì điểm 1 nghĩa là
lý do đằng sau code hiện không tồn tại ở đâu ngoài file đó.

Thuật ngữ gặp trong lúc drill mà không định nghĩa được → tra [docs/GLOSSARY.md](../GLOSSARY.md) và tự chấm mục đó.

## Bộ câu hỏi chuẩn (dùng cho mọi file)

1. File này tồn tại để làm gì? Nếu xoá đi thì cái gì hỏng?
2. Ai gọi nó? Nó gọi ai?
3. Chỉ ra **một hằng số** trong file và nói con số đó từ đâu ra.
4. Chỉ ra **một khối `try/except`** và nói nó đang phòng tình huống thật nào.
5. Chỗ nào trong file này là *quyết định*, chỗ nào chỉ là *ghép nối*?
6. Phương án nào đã bị loại khi viết file này?
7. Nếu corpus to gấp 100 lần, dòng nào hỏng trước?
8. Test nào đang che file này? Đổi dòng nào thì test vẫn xanh mà sản phẩm sai?

## Nhật ký

| Ngày | File | Điểm TB | Chỗ ngập ngừng → DEC |
|---|---|---|---|
| 2026-09-15 | [`src/rag/generator.py`](2026-09-15-generator.md) | _(tự chấm)_ | |
