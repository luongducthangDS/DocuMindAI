# Câu hỏi tổng hợp — nguyên liệu, CHƯA DUYỆT

`doc2query_questions.jsonl`: 4.992 câu hỏi đời thường, 4 câu cho mỗi đoạn trong 1.249 đoạn của
index `documind_legal_fpt` (2026-10-18). FPT (`Qwen3.8-27B` / `DeepSeek-V4-Flash`) sinh bằng
[`scripts/generate_doc2query.py`](../../../scripts/generate_doc2query.py), prompt nằm trong script.

Mỗi dòng: `version_id`, `clause_uid`, `doc_id`, `so_hieu`, `dieu`, `effective_from`/`effective_to`,
`text_sha1` (hash nội dung đoạn lúc sinh), `model`, `questions`.

**Không phải gold set.** Chưa có đáp án, chưa ai đọc. Câu hỏi có thể không được đoạn nguồn trả lời
trọn, hoặc hỏi điều đoạn không nói (ví dụ đoạn về lao động nữ sinh ra câu hỏi cho lao động nam).
Muốn đưa câu nào vào eval: viết đáp án từ nguyên văn đoạn, chạy `eval/validate_gold.py`, cho người duyệt.

Dùng được cho: nguồn câu hỏi kiểu kể chuyện để chọn lọc thành gold; thử độ bền truy hồi với cách
hỏi đời thường. **Không** dùng làm nhánh truy hồi — đã đo và loại
(`reports/retrieval_experiments_2026-10-18.json`).

Cũ: `text_sha1` khác hash đoạn hiện tại là đoạn đã đổi sau lúc sinh. `python scripts/generate_doc2query.py`
chỉ sinh cho đoạn mới hoặc đã đổi; dòng cũ không tự xoá.
