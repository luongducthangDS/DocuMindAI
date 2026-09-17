# DEC-0004: Chấm retrieval bằng `clause_uid` + `version_id`, bỏ ngưỡng token-F1

- **Trạng thái:** Proposed
- **Ngày:** 2026-09-17
- **Spec liên quan:** `docs/spec/SPEC-eval-metrics-v2.md`
- **Ảnh hưởng tới:** `eval/metrics.py`, `eval/rag_comparison.py`, `src/guardrails.py` (thêm hàm), `eval/scoring_ab.py` (mới)

## 1. Bối cảnh — vấn đề gì buộc phải quyết

`eval/metrics.py` coi một chunk là "trúng" khi token-F1 giữa **text của chunk** và
**`ground_truth`** ≥ 0.15:

```python
def _chunk_relevant(chunk_text, ground_truth, threshold=0.15):
    return _f1_overlap(chunk_text, ground_truth) >= threshold
```

Corpus lao động chứa **nhiều bản của cùng một điều**. Hai bản của Điều 3 NĐ lương tối thiểu
vùng khác nhau đúng một con số (4.960.000 vs 5.310.000) trên vài trăm từ giống hệt nhau — token-F1
giữa chúng gần như bằng nhau. Nghĩa là:

> Chunk **sai bản hiệu lực** vẫn được bộ đo tính là trúng.

Đó chính là lỗi mà `do_temporal_filter` (DEC-0001) sinh ra để chặn. Đo trên
`reports/temporal_eval.json`, cấu hình `no_temporal` có `context_distractor = 76.7%` — 23/30 câu
kéo về ít nhất một bản đã hết hiệu lực. Bộ đo token-F1 **không phản ánh** 23 câu đó là sai.

Hệ quả thực tế: mọi số `hit_rate`/`MRR` sinh ra từ `eval/metrics.py` đều lạc quan một cách có
hệ thống, và lạc quan nhất ở đúng nhóm câu hỏi khó nhất. Một bộ đo mù ở chỗ hệ thống dễ sai
nhất thì không dùng để gác regression được.

Điều kiện đủ để sửa: corpus **đã có** `clause_uid` và `version_id` trên 1146/1146 chunk
(`clause-schema-ingestion`, T7-T11), và `eval/temporal_eval.py::score_context` **đã** chấm theo
cách này và cho ra số phân biệt được các arm. Tức hạ tầng có sẵn, chỉ là `metrics.py` chưa dùng.

## 2. Các phương án đã cân nhắc

| # | Phương án | Được | Mất |
|---|---|---|---|
| A | **Khớp `clause_uid` + loại `version_id` là distractor** | Đo đúng thứ hệ thống hứa: đúng điều, đúng bản. Dùng lại đúng quy tắc `score_context` đã được kiểm chứng. Deterministic, không cần LLM, chạy được trong CI | Gold bắt buộc phải có `clause_uid` → câu gold cũ không có thì phải skip, không chấm được |
| B | Nâng ngưỡng token-F1 (0.15 → 0.6) | Sửa một dòng | Không giải quyết gì: hai bản của cùng một điều vẫn giống nhau >0.9. Chỉ làm metric khắt khe hơn với **mọi** chunk, không riêng chunk sai bản |
| C | Dùng LLM-judge chấm relevance | Không cần gold chi tiết | Không deterministic, không chạy được trong CI, tốn quota, và lại đi đo LLM bằng LLM |
| D | Không làm gì | 0 công | Mọi số Phase 1-5 đo bằng thước hỏng. Baseline vô nghĩa, regression gate vô dụng |

## 3. Quyết định

Chọn **A**.

**Vì sao:** thứ dự án bán là "trả lời theo đúng bản có hiệu lực tại thời điểm X". Thước đo phải
đo đúng thứ đó, nếu không thì cải tiến temporal (DEC-0001) không chứng minh được bằng số trong
bộ đo chính. B bị loại vì bản chất vấn đề không nằm ở ngưỡng mà ở **tín hiệu** — token-F1 không
mang thông tin về hiệu lực, chỉnh ngưỡng bao nhiêu cũng không tạo ra thông tin đó. C bị loại vì
biên giới đã chốt trong spec: metric phải deterministic để gác CI (không API key, không quota).
D bị loại vì đã có bằng chứng cụ thể (23/30 câu) rằng thước hiện tại sai.

Câu gold thiếu `clause_uid` thì **skip và báo số câu skip**, tuyệt đối không âm thầm rơi về
token-F1 — một bộ đo im lặng đổi thước giữa chừng còn tệ hơn bộ đo sai.

`lexical_overlap` (tên cũ `_f1_overlap`) **được giữ** cho `answer_correctness`, vì so *câu trả
lời* với *ground_truth* là việc khác so với chọn chunk. Nó bị cấm làm headline metric.

## 4. Bằng chứng (CHƯA ĐIỀN — chưa chạy được)

| Cách chấm | recall@8 | MRR@8 | n chấm được | n skip |
|---|---|---|---|---|
| token-F1 ≥ 0.15 (cũ) | | | | |
| `clause_uid` + version (mới) | | | | |

- Nguồn số: `reports/scoring_ab.json` — lệnh tái tạo: `python eval/scoring_ab.py`
- Số này đo **retrieval**, trên 30 câu gold temporal, không gọi LLM.
- Số này **không** đo chất lượng câu trả lời, không đo faithfulness.

**Vì sao còn trống (2026-09-17).** Harness đã viết xong và test xanh, nhưng không chạy được
số trên máy hiện tại — cả hai đường lấy embedding đều tắc:

| Đường | Kết quả | Chi tiết |
|---|---|---|
| `EMBEDDING_PROVIDER=local` | crash (exit 5 / segfault) | Model `AITeamVN/Vietnamese_Embedding` ~2.2GB + torch, máy chỉ còn 1.9–2.7GB RAM trống trên tổng 15.2GB |
| `EMBEDDING_PROVIDER=hf_api` | `ValueError` từ HuggingFace | `Model 'AITeamVN/Vietnamese_Embedding' doesn't support task 'feature-extraction'. Supported tasks: 'sentence-similarity'` |

Đường thứ hai là một lỗi thật, không phải giới hạn máy: `render.yaml` đang đặt
`EMBEDDING_PROVIDER=hf_api` cho production, nên **production không lấy được vector nào**.
Xem mục 0e trong kế hoạch Phase 0. Docstring `_HFInferenceAPIEmbedding` nói vector "đã được
verify khớp local" và ghi 384 chiều — đó là kiểm chứng cho model MiniLM cũ, không phải model
1024 chiều hiện tại.

**DEC này giữ trạng thái `Proposed`** cho tới khi bảng trên có số. Theo `docs/decisions/README.md`:
DEC không có số là DEC chưa xong. Code đi kèm đã merge được vì nó có test riêng chứng minh
hành vi (`tests/test_eval_metrics.py`, 30 test), nhưng **quyết định** thì chưa được chốt.

## 5. Hệ quả

- **Được:** thước đo phân biệt được "đúng điều, đúng bản" với "đúng điều, sai bản" — điều kiện
  cần để gác regression cho mọi phase sau. Chạy không cần API key.
- **Trả giá:** gold set từ nay bắt buộc có `clause_uid` cho mọi câu cần chấm retrieval. 25 câu
  gold ngân hàng cũ và bất kỳ gold nào viết kiểu "chỉ có ground_truth" đều không chấm được —
  chi phí này rơi vào Phase 1 (dựng 200 câu). Đổi lại contract `retrieve_and_answer` của 4
  strategy và format cache (v1 → v2), cache cũ phải sinh lại.
- **Sai khi nào:** nếu corpus về sau có chunk không thuộc điều/khoản nào (phụ lục, biểu mẫu,
  văn bản không đánh Điều) thì `clause_uid` rỗng và cách chấm này bỏ sót chúng. Lúc đó cần
  DEC mới về định danh gold cho chunk phi-điều-khoản.

## 6. Câu hỏi phỏng vấn tự đặt

1. *Sao không dùng LLM-as-judge cho relevance?* — Vì metric này gác CI: phải deterministic,
   không quota, chạy trong 2 phút không API key. LLM-judge để dành cho faithfulness, nơi không
   có gold chính xác.
2. *Khớp `clause_uid` thì có quá khắt khe với câu trả lời đúng nhưng trích điều khác không?* —
   Có, và đó là lý do `citation_groundedness` dùng "≥1 citation trúng gold" chứ không phải "mọi
   citation trúng gold". Recall@k thì cố ý khắt khe vì nó đo retrieval, không đo diễn đạt.
3. *Làm sao biết thước mới không phải là thước cũ đội lốt?* — Bảng ở mục 4 chấm **cùng một lần
   retrieval** bằng cả hai cách. Nếu hai cột giống nhau thì quyết định này thừa, và DEC này phải
   bị đổi trạng thái thành Rejected.
