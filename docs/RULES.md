# Rules

Quy tắc chi tiết hiện nằm trong `CLAUDE.md` (Claude) và `AGENTS.md` (Codex, bản sao gần như y hệt) — đọc cả mục "Các lưu ý kỹ thuật". Bài học: `docs/LESSONS.md`. File này tóm tắt các bất biến; khi gom xong (xem TASKS) thì đây là nơi duy nhất.

## Bất biến
- Phạm vi sản phẩm chỉ khai báo ở `src/config.py` và `PRODUCT` trong `main.tsx`; prompt, message từ chối, nhãn UI phải lấy từ đó.
- Mọi tiêu chí trong `data/compliance/criteria.json` phải trích văn bản CÓ trong `data/raw/lao_dong/` (có test chặn).
- Phát hiện quyết định kỹ thuật sai hoặc xử lý sai (kể cả đoán thay vì đo, báo "đã verify" khi chưa) → thêm mục vào `docs/LESSONS.md` trong cùng commit với bản sửa.
- Danh tính index là hash nội dung; không suy khớp code/index từ git log hay số chunk.
- Chạy gì trên Qdrant từ máy local phải truyền `QDRANT_COLLECTION=documind_legal_fpt`, không thì rơi về collection cũ 3072-dim.
- Đổi embedding/model thì ghi rõ cấu hình cạnh mọi con số eval; không so số giữa hai cấu hình như cùng một hệ thống.

## AI không bao giờ được
- Commit `.env`, khoá API, `data/hf_cache/`.
- Đổi env production (Render, Qdrant) khi user chưa yêu cầu.
- Push hoặc merge khi user chưa yêu cầu.
