# DEC-0002: Tự viết lõi retrieval (chunk/BM25/RRF/rerank) thay vì bọc LlamaIndex

- **Trạng thái:** Proposed
- **Ngày:** 2026-09-15
- **Spec liên quan:** `exercises/retrieval_core/README.md`
- **Ảnh hưởng tới:** chưa ảnh hưởng `src/` — bước 1 chạy song song trong `exercises/`

> Đây là **mẫu viết trước khi có code**. Mục 4 đang rỗng có chủ đích: DEC này chưa
> `Accepted` được cho tới khi điền số. Đúng thứ tự là như vậy.

## 1. Bối cảnh

`src/rag/retriever.py` dài 233 dòng, trong đó phần **thuật toán** bằng 0. Toàn bộ nằm ở:

- `QueryFusionRetriever(mode="reciprocal_rerank")` — RRF, hằng số `k` không xuất hiện ở đâu trong repo.
- `BM25Retriever.from_defaults(...)` — tokenizer, `k1`, `b` đều mặc định của thư viện.
- `SentenceTransformerRerank(...)` — cross-encoder.
- `src/ingestion/chunker.py` — phần này có tự viết (tách theo `Điều`), 410 dòng.

Hệ quả cụ thể, quan sát được trong chính comment của file: ngưỡng `_MIN_RELEVANCE_SCORE =
0.05` ở `generator.py` chỉ đúng cho thang điểm cross-encoder; thang RRF là `~1/(60+rank) ≈
0.016` nên luôn dưới ngưỡng và hệ thống từ chối mọi câu. Đây là bug **xuất phát từ việc
không sở hữu con số `60`** — nó là mặc định của thư viện, phát hiện ra sau khi sản phẩm
đã sai ở môi trường thiếu reranker.

Đây cũng là phần bị hỏi nhiều nhất khi phỏng vấn RAG.

## 2. Các phương án đã cân nhắc

| # | Phương án | Được | Mất |
|---|---|---|---|
| A | Giữ nguyên, đọc source LlamaIndex để hiểu | Không tốn thời gian code | Đọc ≠ tái tạo; không có gì chứng minh khi phỏng vấn |
| B | **Tự viết lõi ~200 dòng trong `exercises/`, đối chiếu kết quả với bản đang chạy** | Sở hữu từng hằng số; có bản tham chiếu để giải thích lệch | 1–2 buổi; hai bản implementation cùng tồn tại |
| C | Tự viết rồi thay thẳng vào `src/` | Xoá hẳn dependency | Rủi ro tụt chất lượng production khi chưa có số so sánh |

## 3. Quyết định (đề xuất)

Chọn **B**, và **cấm dùng agent sinh code cho `exercises/retrieval_core/core.py`**.

**Vì sao B chứ không C:** chưa có số chứng minh bản tự viết ngang bản thư viện. Thay trước
khi đo là lặp lại đúng sai lầm đã tạo ra bug ngưỡng 0.05 — tin vào code chưa đo. C chỉ được
mở lại sau khi mục 4 điền xong và bản tự viết không tệ hơn trong ngưỡng đã định.

**Vì sao cấm agent:** mục tiêu của việc này không phải là có code chạy được — code chạy
được đã có sẵn trong `src/`. Mục tiêu là tái tạo được lý do. Agent sinh ra file đúng thì
kết quả đo được vẫn là 0.

## 4. Bằng chứng

> Điền sau khi chạy `pytest exercises/retrieval_core --no-cov` xanh và chạy so sánh trên
> gold set. DEC chuyển `Accepted` khi bảng này có số.

| Cấu hình | hit_rate@8 | MRR | context_clean | n |
|---|---|---|---|---|
| `src/rag/retriever.py` (LlamaIndex) | | | | |
| `exercises/retrieval_core` (tự viết) | | | | |

- Lệnh tái tạo: _(chưa có)_
- Ngưỡng chấp nhận đặt trước: bản tự viết được phép thấp hơn tối đa **3 điểm phần trăm**
  hit_rate@8. Thấp hơn nữa ⇒ bản tự viết sai ở đâu đó, phải tìm ra chỗ lệch chứ không nới ngưỡng.

## 5. Hệ quả (dự kiến)

- Được: mọi hằng số (`k1=1.5`, `b=0.75`, `k=60`, `top_n=8`) trở thành lựa chọn có lý do,
  không phải mặc định thừa kế.
- **Trả giá:** hai bản implementation song song, dễ phân kỳ. Giảm bằng cách để
  `exercises/` ngoài `testpaths` và không import từ `src/`.
- Quyết định này sai nếu: sau khi đo, bản tự viết lệch nhiều và không tìm ra nguyên nhân
  trong 1 buổi ⇒ dừng ở mức "đã hiểu", không theo đuổi C.

## 6. Câu hỏi phỏng vấn tự đặt

1. *RRF `k=60` ở đâu ra, đổi thành 10 thì sao?* — điền sau khi tự implement và thử.
2. *BM25 `b` làm gì, corpus toàn điều luật ngắn thì nên để bao nhiêu?* — điền sau.
3. *Sao không dùng luôn thư viện?* — có dùng ở production; bản tự viết là để biết thư viện
   đang làm gì, và nó đã bắt được bug thang điểm 0.05.
