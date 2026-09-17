# Drill 2026-09-15 — `src/rag/generator.py`

Bốc bằng `find src -name "*.py" ! -name "__init__.py" | shuf -n 1`. 480 dòng.

**Cách dùng tờ này:** đọc file ≤ 3 phút → đóng → viết câu trả lời vào mục "Trả lời của
tôi" → mới mở phần `Đáp án tham chiếu` để chấm. Ba câu 3/4/5 có sẵn đáp án mẫu để thấy
**độ sâu cần đạt**; các câu còn lại tự viết rồi đối chiếu với file.

---

## 1. File này tồn tại để làm gì? Xoá đi thì cái gì hỏng?

> Trả lời của tôi:

## 2. Ai gọi nó? Nó gọi ai?

> Trả lời của tôi:

## 3. Một hằng số trong file — con số đó từ đâu ra? *(có đáp án mẫu)*

> Trả lời của tôi:

## 4. Một khối `try/except` — nó phòng tình huống thật nào? *(có đáp án mẫu)*

> Trả lời của tôi:

## 5. Chỗ nào là *quyết định*, chỗ nào chỉ là *ghép nối*? *(có đáp án mẫu)*

> Trả lời của tôi:

## 6. Phương án nào đã bị loại khi viết file này?

> Trả lời của tôi:

## 7. Corpus to gấp 100 lần — dòng nào hỏng trước?

> Trả lời của tôi:

## 8. Test nào che file này? Đổi dòng nào thì test vẫn xanh mà sản phẩm sai?

> Trả lời của tôi:

---

<details>
<summary><b>Đáp án tham chiếu — chỉ mở sau khi đã viết xong</b></summary>

### Câu 3 — mức 2

`_MIN_RELEVANCE_SCORE = 0.05`. Đây **không** phải ngưỡng chung, nó được hiệu chuẩn cho
**thang điểm của cross-encoder reranker**: chunk liên quan ăn điểm rõ trên 0.05, chunk ngoài
phạm vi nằm dưới. Ba thang điểm khác nhau cùng tồn tại trong hệ thống — cross-encoder, RRF
thô (`≈1/(60+rank) ≈ 0.016`), và BM25 thô — nên cùng một con số 0.05 áp lên thang RRF sẽ
loại **mọi** chunk và hệ thống từ chối mọi câu hỏi.

Vì vậy con số không được đọc trực tiếp mà đi qua `_effective_min_score()`: hàm này trả 0.05
**chỉ khi** reranker thực sự đang chạy, và kiểm tra bằng `src.rag.retriever._reranker_active`
— trạng thái runtime — chứ không phải cờ config `settings.enable_reranker`. Lý do: reranker
có thể được *bật trong config* nhưng *nạp hỏng* lúc chạy (thiếu model cache, OOM, Render free
tier); tin vào config đã từng khiến hệ thống abstain toàn bộ dù retrieval và grading đều tìm
ra nội dung đúng.

*Điểm 2 vì:* nói được con số, thang đo nó thuộc về, ba thang cùng tồn tại, và **sự cố thật**
đã sinh ra cách viết hiện tại.

### Câu 4 — mức 2

Chuỗi `try/except` trong `generate_answer`: Groq → Gemini → OpenAI-compat → `_build_extractive_answer`.

Nó **không** phòng "lỡ có lỗi". Nó phòng ba tình huống cụ thể khác nhau:
- Groq hết quota TPD trong ngày → chuyển Gemini, đánh dấu `used_llm="gemini_fallback"`.
- Gemini rate-limit / hết key → OpenAI-compat.
- Cả ba chết → **không trả lỗi cho người dùng**, mà trích thẳng 5 chunk đầu (700 ký tự mỗi
  nguồn) thành câu trả lời trích xuất. Quyết định sản phẩm ở đây: chunk đã tìm được vẫn có
  giá trị tra cứu kể cả khi không có LLM nào diễn đạt lại.

Có một nhánh đi tắt cả chuỗi: `prefer_gemini` bỏ qua Groq hoàn toàn khi
`generator_provider=gemini` — dùng khi biết trước Groq đã cạn, để khỏi tốn một lần timeout.

### Câu 5 — mức 2

**Quyết định** (mất sẽ sai sản phẩm):
- `_effective_min_score()` — abstain gate; xem câu 3.
- `_cited_sources()` — chỉ trả về nguồn mà câu trả lời **thật sự trích** qua `[N]`. Không có
  nó, một câu từ chối do LLM tự diễn đạt ("không tìm thấy quy định này…") vẫn đính kèm đủ 8
  nguồn, khiến lời từ chối trông như câu trả lời có căn cứ. Regex `\[([\d,\s]+)\]` bắt cả
  `[1]` lẫn `[1, 2]` vì LLM không nhất quán định dạng.
- `time_out_of_range` → trả lời riêng, không gọi LLM: trả lời cho mốc ngoài phạm vi corpus
  tức là trình bày luật đời sau như thể áp dụng cho đời trước.
- `_as_of_block()` — đổi vai của prompt: chunk **đã** được lọc hiệu lực trước đó (xem
  DEC-0001), nên prompt không bắt model suy luận ngày, chỉ bắt nó **nêu** mốc và dùng đúng
  con số của bản đó.

**Ghép nối** (đổi được mà không ai chết): `_build_context` cắt 3.000 ký tự/chunk, tổng
15.000; `_EXTRACTIVE_CHARS_PER_SOURCE = 700`; thứ tự Groq/Gemini.

Ranh giới để nhớ: quyết định là chỗ **chọn giữa hai hành vi sai/đúng**; ghép nối là chỗ chọn
giữa hai con số đều chạy được.

</details>

---

## Kết luận drill

Điểm trung bình: _(điền)_

Chỗ ngập ngừng → DEC cần viết:
- [ ] _(ví dụ: "không giải thích được vì sao `_MAX_TOTAL_CHARS` là 15.000" → DEC-0003)_
