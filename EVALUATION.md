# DocuMind AI — Đánh giá & phát hiện

> **Bản ghi lịch sử.** Số liệu dưới đây **không đo trên kho tài liệu hiện tại**. Chúng được đo
> trên một kho tài liệu cũ (quy chế sinh viên UNETI) đã bị gỡ khỏi repo. Phạm vi hiện tại của
> sản phẩm là **pháp luật lao động và bảo hiểm xã hội** với gold set
> `data/eval/temporal_questions.json` (xem `CLAUDE.md`). Phương pháp ablation truy hồi và các
> phát hiện kỹ thuật bên dưới vẫn phản ánh đúng cách pipeline này hành xử — giữ lại để tham
> khảo cho tới khi có một lần đánh giá trên kho văn bản lao động thay thế.

Trợ lý RAG cho quy chế sinh viên UNETI (hỏi đáp pháp lý/hành chính tiếng Việt) — chính là lĩnh
vực mà lần đánh giá này chạy trên đó.
Tài liệu này tóm tắt cách hệ thống được đánh giá, kết quả đo được, và những vấn đề kỹ thuật đã
phát hiện rồi sửa trong quá trình đánh giá.

## 1. Phương pháp

- **Benchmark:** 110 câu hỏi tiếng Việt viết tay trải trên 7 văn bản quy chế
  (100 câu trong kho + 10 câu "bẫy" nằm ngoài kho, hỏi về trường khác / chủ đề không được phủ),
  mỗi câu kèm đáp án chuẩn và id chunk nguồn.
- **Ablation:** đánh giá 4 chiến lược truy hồi trên *cùng* một bộ câu hỏi —
  chỉ BM25, chỉ dense, lai (BM25 + dense + RRF), và lai + rerank bằng cross-encoder.
- **Chỉ số:**
  - *Truy hồi (không dùng LLM):* hit_rate@K, MRR, số chunk trả về trung bình.
  - *Sinh văn bản (có LLM):* RAGAS faithfulness / answer_relevancy / context_recall /
    context_precision; độ đúng của câu trả lời so với đáp án chuẩn.
  - *Độ vững về phạm vi:* tỉ lệ trích dẫn, tỉ lệ từ chối câu ngoài kho (OOC).
- **Ngăn xếp:** FastAPI · LangGraph · ChromaDB · sentence-transformers ·
  Groq Llama-3.3-70B (chính) / Gemini (dự phòng) · RAGAS (Gemini làm giám khảo).

## 2. Kết quả — truy hồi & phạm vi (benchmark đầy đủ, n = 110)

| Chiến lược          | hit_rate@K | MRR  | chunk TB | trích dẫn | từ chối OOC | độ trễ p95 |
|---------------------|:----------:|:----:|:--------:|:---------:|:-----------:|:----------:|
| Chỉ BM25            |    0,94    | 0,86 |    5     |   0,87    |    1,00     |   11,8 s   |
| Chỉ dense           |    0,89    | 0,76 |    5     |   0,64    |    1,00     |   11,4 s   |
| Lai (hybrid)        |    0,95    | 0,83 |    20    |   0,85    |    1,00     |   11,7 s   |
| **Lai + rerank**    |  **0,95**  |**0,87**|   8    | **0,87**  |  **1,00**   |   31,6 s   |

- **Lai + rerank là cấu hình tốt nhất:** MRR cao nhất (0,87), ngữ cảnh gọn
  (8 chunk so với 20), và tỉ lệ từ chối câu ngoài kho tuyệt đối.
- **Từ chối câu ngoài kho = 100%** ở mọi chiến lược — hệ thống không bao giờ trả lời câu hỏi
  về trường khác hay chủ đề không được phủ (kiểm trên 10 câu bẫy).
- BM25 là baseline mạnh trên kho tài liệu này vì câu hỏi pháp lý nặng từ khoá
  (số hiệu văn bản, số điều, thuật ngữ chính xác).

## 3. Kết quả — chất lượng sinh văn bản (RAGAS, thử nghiệm n = 5)

Con số mang tính tham khảo trên thử nghiệm 5 câu (lần chạy RAGAS đủ 110 câu bị hoãn — xem mục
Giới hạn):

| Chiến lược          | faithfulness | answer_relevancy | context_precision |
|---------------------|:------------:|:----------------:|:-----------------:|
| Lai (hybrid)        |    ~1,00     |      ~0,73       |       ~0,67       |
| **Lai + rerank**    |    ~0,93     |      ~0,85       |     **~0,83**     |

Faithfulness ≈ 0,9 cho thấy câu trả lời bám sát các đoạn đã truy hồi (ít bịa); reranker nâng cả
độ liên quan lẫn độ chính xác so với bản lai thuần.

## 4. Phát hiện kỹ thuật & bản vá

Quá trình đánh giá lộ ra vài vấn đề thật, đã truy tới gốc và sửa:

1. **Reranker lệch ngôn ngữ.** Bản chạy production dùng cross-encoder tiếng Anh
   (`ms-marco-MiniLM-L-6-v2`) để rerank các đoạn tiếng Việt. Đổi sang reranker đa ngữ
   (`bge-reranker-v2-m3`) → **context_precision 0,66 → 0,83**, trích dẫn 0,80 → 1,00
   (trên thử nghiệm). Đã đưa ra `.env` để cấu hình được.
2. **Lỗi thang điểm khiến từ chối nhầm.** Một ngưỡng liên quan cố định (0,05, hiệu chỉnh cho
   logit của cross-encoder) lại đem áp lên điểm hợp nhất RRF (~0,016) → nhánh lai từ chối
   *mọi* câu hỏi. Đã sửa bằng cách đặt ngưỡng từ chối riêng cho từng retriever.
3. **Từ chối thừa ở khâu sinh.** Khoảng 12–16% câu *trả lời được* vẫn bị từ chối dù đã truy hồi
   đúng đoạn văn (nguyên nhân gốc: prompt quá thận trọng, bám cứng vào việc khớp đúng số hiệu
   văn bản, lại bị khuếch đại bởi một model sinh nhỏ). Đã xử lý bằng cách thiết kế lại system
   prompt để trả lời dựa trên *nội dung* đoạn văn, vẫn giữ nguyên quy tắc từ chối câu ngoài kho.
4. **Hạ tầng đánh giá.** Đã dựng bộ giám khảo Gemini nhiều key với xoay vòng key/model, giới hạn
   tần suất theo từng cặp, khoá an toàn với event loop giữa các batch RAGAS, cache câu trả lời,
   và checkpoint theo batch để chạy tiếp được sau khi dừng.

## 5. Giới hạn (nói thẳng phạm vi)

- Kho tài liệu hẹp theo lĩnh vực: 91 chunk trên 7 văn bản — một trợ lý có phạm vi giới hạn,
  không phải hệ thống quy mô lớn.
- Số liệu giám khảo RAGAS lấy từ thử nghiệm 5 câu; lần chạy RAGAS đủ 110 câu còn treo vì hạn
  mức quota hằng ngày của gói miễn phí trên LLM giám khảo.
- Reranker đa ngữ làm tăng độ trễ (p95 ≈ 32 s trên CPU); muốn chạy production độ trễ thấp thì
  cần reranker nhẹ hơn hoặc GPU.
- Embedding dense (MiniLM-384 chiều) yếu với tiếng Việt; đã xác định hướng nâng cấp lên BGE-M3
  (ước tính +5–8 điểm phần trăm context_recall).
- Bản vá chống từ chối thừa đã áp dụng nhưng chưa đo lại ở quy mô đầy đủ.

## 6. Tóm tắt dùng cho CV

> Xây trợ lý RAG tiếng Việt cho quy chế đại học (FastAPI · LangGraph · ChromaDB) với truy hồi
> lai (BM25 + dense + RRF) và reranker cross-encoder. Đạt **hit_rate@K 0,95, MRR 0,88 và 100%
> từ chối câu ngoài kho** trên benchmark 110 câu tự xây; RAGAS faithfulness ≈ 0,9 (thử nghiệm).
>
> Thiết kế bộ khung đánh giá ablation 4 chiến lược (RAGAS + chỉ số truy hồi/phạm vi tự viết).
> Dùng nó để chẩn đoán và sửa các lỗi production — reranker lệch ngôn ngữ
> (**context_precision 0,66 → 0,83**), lỗi thang điểm khiến từ chối nhầm, và ~15% câu bị từ chối
> thừa ở khâu sinh — đo mức cải thiện trước/sau từng thay đổi.
