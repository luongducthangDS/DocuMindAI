# DEC-0001: Lọc hiệu lực bằng node riêng trong graph, không nhét vào prompt

- **Trạng thái:** Accepted
- **Ngày:** 2026-09-15
- **Spec liên quan:** `docs/spec/SPEC-temporal-retrieval.md`
- **Ảnh hưởng tới:** `src/rag/temporal.py` (mới), `src/agent/graph.py`, `src/rag/generator.py`, `src/api/schemas.py`

> Đây là **mẫu đã điền đầy đủ**. Mọi số trong mục 4 lấy từ `reports/temporal_eval.json`,
> không có số nào viết từ trí nhớ.

## 1. Bối cảnh

Corpus lao động chứa nhiều **bản** của cùng một điều/khoản (lương tối thiểu vùng, tỷ lệ
đóng BHXH đổi qua các năm). Retriever hybrid không biết gì về thời gian: nó xếp hạng theo
độ giống chữ, nên bản 2020 và bản 2024 của cùng một điều đều nằm trong top-8.

Đo trên gold set 30 câu (`data/eval/temporal_questions.json`), cấu hình gốc `no_temporal`:

- `context_distractor = 76.7%` — 23/30 câu có ít nhất một bản **đã hết hiệu lực** trong context.
- `context_clean = 20.0%` — chỉ 6/30 câu có context sạch.
- `answer_accuracy = 36.7%`.

Tức là hơn 3/4 số câu, LLM phải tự đoán bản nào còn hiệu lực từ chính văn bản. Nó đoán sai.

## 2. Các phương án đã cân nhắc

| # | Phương án | Được | Mất |
|---|---|---|---|
| A | Dặn trong prompt: "chọn bản có hiệu lực tại ngày X" | Không đụng code retrieval, 10 phút | Context vẫn bẩn; đúng/sai phụ thuộc LLM; không test được bằng unit test |
| B | Metadata pre-filter ngay trong vector query (`where` của Chroma) | Lọc sớm nhất, context nhỏ nhất | Phải sửa `retriever.py` — spec cấm; BM25 không có `where`, hai nhánh fusion lệch nhau |
| C | **Node `do_temporal_filter` riêng, chạy sau `do_retrieve`** | Thuần hàm, test được không cần LLM; không đụng fusion/rerank | Thêm một node; lọc sau khi đã tốn công retrieve |
| D | Không làm gì | — | 36.7% accuracy |

## 3. Quyết định

Chọn **C**.

**Vì sao:** ràng buộc quyết định là dòng "Never sửa `src/rag/retriever.py` node retrieval /
fusion / rerank logic" trong `SPEC-temporal-retrieval.md` — retrieval đã có benchmark riêng,
trộn thêm logic thời gian vào đó là làm hỏng khả năng quy kết khi số tụt. B vi phạm trực
tiếp. A không vi phạm nhưng không **kiểm** được: không có cách viết unit test cho "LLM có
nghe lời không", mà đó chính là thứ đang sai.

C đặt quy tắc hiệu lực (`effective_from <= T AND (effective_to IS NULL OR effective_to > T)
AND status != 'repealed'`) vào một hàm thuần trong `src/rag/temporal.py`, test bằng bảng
input/output, không gọi model.

A **không bị loại hoàn toàn** — vẫn giữ `_as_of_block()` trong prompt, nhưng đổi vai: không
còn bắt model suy luận ngày, chỉ bắt nó **nêu** mốc ngày trong câu trả lời.

## 4. Bằng chứng

Gold set 30 câu, retriever giữ nguyên (hybrid BM25 + dense, RRF, rerank → top-8):

| Cấu hình | context_gold | context_distractor | context_clean | avg_chunks | answer_accuracy |
|---|---|---|---|---|---|
| `no_temporal` (trước) | 73.3% | 76.7% | 20.0% | 8.00 | 36.7% |
| `prompt_only` (phương án A) | 73.3% | 76.7% | 20.0% | 8.00 | 46.7% |
| `temporal_filter` (phương án C) | **80.0%** | **0.0%** | **80.0%** | 2.87 | **83.3%** |

Nhóm khó nhất — `clause_version` (n=4), câu hỏi về đúng bản nào của một điều:
accuracy 25% → 25% → **75%**; distractor 100% → 100% → **0%**.

- Nguồn: `reports/temporal_eval.json`. Tái tạo: `python eval/temporal_eval.py`.
- Số này đo **context có sạch không** và **câu trả lời có đúng con số của bản đúng không**.
  Nó **không** đo chất lượng diễn đạt, không đo latency, và không đo trường hợp metadata
  hiệu lực bị gán sai ngay từ khâu ingest.

Điểm đáng chú ý: A nâng accuracy 36.7% → 46.7% **mà không làm sạch context chút nào**
(distractor vẫn 76.7%). Đó là bằng chứng A chỉ chữa triệu chứng.

## 5. Hệ quả

- Được: context sạch 20% → 80%; accuracy 36.7% → 83.3%; quy tắc hiệu lực nằm ở một chỗ,
  test được offline.
- **Trả giá:** `avg_chunks` 8.00 → 2.87. Bộ lọc chạy **sau** rerank nên cắt mất hơn nửa
  context; nếu `effective_from` / `effective_to` bị gán sai ở khâu ingest thì chunk đúng bị
  vứt **im lặng** và câu trả lời thành "không tìm thấy". Rủi ro chuyển từ *trả lời sai* sang
  *từ chối oan* — dễ phát hiện hơn, nhưng vẫn là lỗi.
- Trả giá 2: top-8 bây giờ là top-8 **trước** khi lọc, nên số chunk cuối cùng không ổn định
  giữa các câu.
- Quyết định này sai khi: corpus có văn bản mà một điều sửa đổi bổ sung *một phần* điều khác
  (không thay thế trọn bản) — lúc đó lọc theo `clause_uid` bỏ mất phần gốc còn hiệu lực.
  Khi gặp ca đó → viết DEC mới, không vá node này.

## 6. Câu hỏi phỏng vấn tự đặt

1. *Sao không lọc ngay ở vector query cho rẻ?* — BM25 không nhận metadata filter, hai nhánh
   fusion sẽ lọc lệch nhau; và spec khoá `retriever.py` để giữ benchmark retrieval quy kết được.
2. *Lọc sau rerank thì phí công retrieve, sao không tăng `top_k` bù lại?* — Có, đó là hướng
   sửa tiếp theo; hiện `avg_chunks` 2.87 chưa chạm đáy recall trên gold set 30 câu nên chưa
   đổi, và đổi `top_k` phải chạy lại benchmark retrieval.
3. *Làm sao biết 83.3% không phải do may?* — n=30 nên khoảng tin cậy rộng; con số thuyết
   phục hơn là `context_distractor` 76.7% → 0%, đây là đại lượng đo trực tiếp cơ chế, không
   qua LLM.
