# DocuMind AI — Interview Preparation

> 5 câu hỏi khó nhất về project, với câu trả lời đầy đủ.
> Học những câu này trước khi đi interview.

---

## Q1: Tại sao dùng RRF (Reciprocal Rank Fusion) thay vì weighted sum?

**Câu trả lời:**

Dense embedding và BM25 có score ở hai scale hoàn toàn khác nhau:
- Dense (cosine): 0.0 → 1.0
- BM25: có thể là 8.2, 12.4, 100+ tùy corpus size

Nếu dùng weighted sum `α × dense_score + β × bm25_score`, phải tune α và β — và mỗi khi thay embedding model hay corpus size thay đổi, lại phải tune lại.

RRF không dùng score tuyệt đối mà chỉ dùng **rank** (thứ hạng):
```
RRF_score(d) = Σ 1 / (k + rank_i)
```
k=60 là hyperparameter chuẩn, không cần tune theo domain. Kết quả: chunk nào được nhiều nhánh xếp cao → score cao, bất kể scale của từng nhánh.

**Thêm:** Trong project tôi đo được BM25 rất quan trọng cho số hiệu văn bản ("45/2019/QH14") — dense embedding mã hóa ngữ nghĩa nhưng exact number match thì BM25 tốt hơn.

---

## Q2: `reformulate_node` được trigger khi nào? Nó làm gì?

**Câu trả lời:**

Flow: `retrieve → grade → [reformulate] → retrieve (retry)`

`reformulate_node` được trigger khi `grade_node` đánh giá chunks là **không relevant** (irrelevant), và số lần retry chưa vượt `MAX_RETRIES=1`.

**Tại sao cần?** Người dùng hay dùng từ dân dã, không khớp thuật ngữ pháp lý:
- "rút BHXH" → thuật ngữ thực: "hưởng bảo hiểm xã hội một lần"
- "đuổi việc" → "sa thải, đơn phương chấm dứt hợp đồng lao động"

Retrieval lần 1 với "rút BHXH" trả về chunk lệch. Reformulate rewrite query với phrasing pháp lý hơn → retrieve lại → tìm đúng Điều 70 Luật BHXH.

**Tại sao chỉ retry 1 lần?** Thực nghiệm cho thấy 2+ lần retry không cải thiện thêm đáng kể, nhưng mỗi vòng tốn 2 LLM calls (grade + generate). Free tier Gemini 50 RPM — đo được: 5 query liên tiếp, latency leo từ 7.5s → 80s.

---

## Q3: Recall@1 tăng từ 0.585 lên 0.790 — bạn thay đổi cái gì, đo thế nào?

**Câu trả lời:**

**Đo thế nào:** Xây dựng Golden Test Set 208 câu hỏi-đáp từ corpus thực. Mỗi câu có ground truth là `clause_uid` của chunk chứa câu trả lời. Recall@1 = tỷ lệ câu có ground truth chunk ở rank 1 sau retrieve+rerank.

**Các thay đổi theo thứ tự tác động:**

| Thay đổi | Recall@1 |
|---------|---------|
| Baseline (dense only) | 0.585 |
| + BM25 + RRF | 0.650 |
| + Reranker (BGE-v2-m3) | 0.720 |
| + Chunk heading prefix (Chương/Mục) | 0.760 |
| + Embedding model upgrade | 0.790 |

**Chunk heading prefix** là cải tiến thú vị nhất: "Điều 54. Điều kiện hưởng lương hưu" xuất hiện 2 lần — Chương III (BHXH bắt buộc) và Chương IV (tự nguyện). Embedding không phân biệt được vì text giống nhau. Thêm prefix "Chương III — BHXH BẮT BUỘC" vào đầu chunk → embedding hiểu được context khác nhau. Đo được: 18/186 câu gold bị trả về wrong article, giảm xuống ~2.

---

## Q4: Temporal filtering hoạt động thế nào khi user hỏi về luật cũ?

**Câu trả lời:**

Pháp luật VN có nhiều lớp thay thế nhau — Luật BHXH 58/2014 bị thay bởi Luật 41/2024 (hiệu lực 1/7/2025). Nếu không filter, query về "điều kiện hưởng lương hưu" sẽ trả về cả 2 phiên bản, LLM có thể dùng phiên bản cũ.

**Giải pháp pre-filter tại vector store:**

Mỗi chunk được lưu kèm:
```
effective_from_i: 20250701  (integer YYYYMMDD)
effective_to_i:   99991231  (open-ended = còn hiệu lực)
status: "in_force" | "superseded"
```

Filter: `effective_from_i ≤ as_of_date < effective_to_i`

**Tại sao integer thay vì string?** ChromaDB chỉ hỗ trợ numeric comparison (`$lte`, `$gt`) — string comparison không được hỗ trợ.

**Tại sao pre-filter thay vì post-filter?** Post-filter = reranker tốn slot cho chunks đã hết hiệu lực. Đo được: avg relevant chunks trong top-8 = 2.87 với post-filter, do reranker lãng phí 5/8 slots cho chunks superseded.

**User hỏi về luật cũ:** API nhận `as_of_date` parameter. User có thể pass `as_of_date=2024-01-01` để tra cứu theo luật tại thời điểm đó — hệ thống filter theo đúng thời điểm đó.

---

## Q5: Tại sao cần `_COSINE_GATES` khác nhau cho từng embedding model?

**Câu trả lời:**

`_COSINE_GATES` là optimization trong grader để tránh LLM call không cần thiết.

**Context:** Khi không có reranker (Render free tier), chunk score là RRF score (~0.016) — không có ngưỡng tuyệt đối có ý nghĩa. Nên grader dùng cosine score của nhánh dense thay thế.

**Vấn đề:** Mỗi embedding model có distribution cosine score khác nhau:
- `gemini-embedding-001`: gold chunks top-8 có p10 cosine = 0.755, out-of-corpus p50 = 0.737
- `Vietnamese_Embedding (FPT)`: gold chunks có p10 = 0.535, out-of-corpus p50 = 0.315

Nếu dùng cùng ngưỡng (ví dụ 0.75) cho cả 2 model: với Vietnamese_Embedding, 40% câu có đáp án sẽ bị coi là "không liên quan" và bị reformulate sai.

**Cách xác định ngưỡng:** Chạy evaluation với Golden Test Set, vẽ distribution cosine score của gold chunks và non-gold chunks → tìm điểm phân cách tốt nhất cho mỗi model.

```python
_COSINE_GATES = {
    "gemini-embedding-001": (0.75, 0.60),    # (CONFIDENT, HOPELESS)
    "Vietnamese_Embedding": (0.54, 0.42),
}
```
- `CONFIDENT`: trên ngưỡng này → relevant ngay, không hỏi LLM
- `HOPELESS`: dưới ngưỡng này → irrelevant ngay, không hỏi LLM
- Giữa 2 ngưỡng → hỏi LLM judge

---

## Câu hỏi bonus thường gặp

**"Bạn handle multi-turn conversation thế nào?"**
> `contextualize_node` rewrite câu hỏi follow-up thành câu độc lập trước khi retrieve. "Còn khoản 2 thì sao?" → "Khoản 2 Điều 54 Luật BHXH quy định gì?" Lịch sử được giữ trong `ShortTermMemory` (SQLite), 10 turns gần nhất.

**"Làm sao prevent hallucination?"**
> 3 lớp: (1) System prompt yêu cầu mandatory citations [N], (2) `validate_citations()` strip citation [N] không tương ứng với chunk nào, (3) Grader tự correction loop.

**"Tại sao không dùng LangChain agent thay vì custom LangGraph?"**
> LangChain ReAct agent không có built-in retry loop với grade/reformulate. LangGraph cho phép define state machine rõ ràng hơn — dễ debug, dễ trace từng step, dễ giới hạn số vòng lặp. Với production system, predictability quan trọng hơn flexibility.

**"Scale thế nào nếu có 1000 user đồng thời?"**
> (1) Vector store: Qdrant Cloud scale horizontally. (2) LLM: đang dùng rotation key×model để tránh rate limit. (3) Reranker: hiện local/FPT API, có thể scale với multiple worker. (4) Bottleneck thực sự: Gemini API rate limit — cần dedicated quota hoặc Azure OpenAI.

---

## Checklist trước interview

- [ ] Chạy được `python interview_demo/04_langgraph_flow.py`
- [ ] Giải thích được flow từ query đến answer trong 2 phút
- [ ] Trả lời được 5 câu hỏi trên mà không nhìn tài liệu
- [ ] Mở được ARCHITECTURE.md và walk through diagram
- [ ] Demo live trên [https://documindai-pd0r.onrender.com](https://documindai-pd0r.onrender.com)
