# Design

## Giao diện (`frontend/src`)
- Nền sáng giấy: `--bg #f3f2ee`, `--rail #ffffff`, `--surface #ffffff`, viền `--border #dcd7cb`.
- Màu nhấn đỏ đậm kiểu văn bản pháp lý: `--accent #a4161a` (hover `#7f1114`, chữ trên nền nhấn `#ffffff`), viền phát sáng cùng màu.
- Font: thân `Be Vietnam Pro`, tiêu đề `Noto Serif` (biến `--display`, fallback Georgia, serif).
- Mọi nhãn mô tả phạm vi lấy từ `PRODUCT` trong `main.tsx`, không viết lại chuỗi ở chỗ khác.

## Câu trả lời
Trích dẫn theo điều/khoản và văn bản, có ngày hiệu lực; câu Có/Không phân tích điều kiện trước khi kết luận; ngoài phạm vi thì từ chối bằng message lấy từ cấu hình phạm vi; khi LLM lỗi thì rơi về trích nguyên văn.

## Báo cáo eval
Kết quả eval trong `eval/` và `docs/` (RAGAS, legal_qa_200, multi-hop), luôn ghi rõ cấu hình model/embedding/reranker vì số thay đổi theo cấu hình (FPT 0,990 so với Gemini 0,933 trên legal_qa_200).
