# Failure analysis — Temporal A/B benchmark

> Nguồn số: `reports/temporal_eval.json` · Gold set: `data/eval/temporal_questions.json`
> Harness: `eval/temporal_eval.py` · Chạy ngày 2026-09-14, corpus 1146 chunk / 19 văn bản.

## 1. Thiết kế phép đo

30 câu hỏi nhạy thời điểm, viết tay từ toàn văn trong `data/raw/lao_dong/` và **commit trước
khi chạy hệ thống lần đầu** (commit `5723245`, đứng trước commit chạy eval). 14 cặp câu hỏi
giống hệt nhau về câu chữ, chỉ khác `as_of_date`, cộng 2 câu ngoài phạm vi thời gian.

| Nhóm | n | Nội dung |
|---|---|---|
| `doc_version` | 20 | Cả văn bản bị thay thế: NĐ 74/2024 ↔ 293/2025 (lương tối thiểu), Luật BHXH 2014 ↔ 2024, Luật Việc làm 2013 ↔ 2025 |
| `clause_version` | 4 | Khoản bị sửa in-place: Điều 139 khoản 1 BLLĐ (mốc 2026-07-01) |
| `control` | 4 | Quy định KHÔNG đổi giữa hai mốc (Điều 98, Điều 113 BLLĐ) — bắt lỗi lọc quá tay |
| `out_of_range` | 2 | `as_of_date` trước 2015-01-01, ngoài phạm vi corpus |

Ba nhánh dùng **chung một lần retrieval** cho mỗi câu, nên khác biệt duy nhất nằm ở tầng
trên retrieval:

| Arm | Lọc hiệu lực | Generator biết `as_of_date` | Chặn ngoài phạm vi |
|---|---|---|---|
| `no_temporal` | không | không | không |
| `prompt_only` | không | có | không |
| `temporal_filter` | có | có | có |

Chấm điểm: câu trả lời phải chứa **mọi** nhóm `expect_contains` và **không** chứa chuỗi nào
trong `expect_absent` (ví dụ mức lương của bản đã hết hiệu lực). Retrieval chấm riêng, không
cần LLM: `context_gold` / `context_distractor` / `context_clean`.

## 2. Kết quả

| Metric | no_temporal | prompt_only | temporal_filter |
|---|---|---|---|
| context_gold | 73.3% | 73.3% | **80.0%** |
| context_distractor | 76.7% | 76.7% | **0.0%** |
| context_clean | 20.0% | 20.0% | **80.0%** |
| **answer_accuracy** | **36.7%** | **46.7%** | **83.3%** |

Theo nhóm (answer_accuracy):

| Nhóm | no_temporal | prompt_only | temporal_filter |
|---|---|---|---|
| doc_version (20) | 30.0% | 45.0% | 80.0% |
| clause_version (4) | 25.0% | 25.0% | 75.0% |
| control (4) | 100% | 100% | 100% |
| out_of_range (2) | 0% | 0% | 100% |

Ba điều đọc được từ bảng trên:

1. **Chỉ nói ngày cho LLM là không đủ.** `prompt_only` chỉ hơn baseline 10 điểm
   (36.7% → 46.7%): khi ngữ cảnh chứa cả hai phiên bản, model vẫn trộn số của bản cũ và bản
   mới. Phải cắt ngữ cảnh mới dứt điểm (46.7% → 83.3%).
2. **Lọc không làm hỏng câu không nhạy thời điểm.** Nhóm `control` giữ 100% ở cả ba nhánh,
   `context_clean` 100% — bộ lọc không cắt nhầm điều còn hiệu lực.
3. **Lọc không làm giảm recall.** `context_gold` của nhánh lọc bằng hoặc cao hơn baseline ở
   mọi câu (chênh lệch duy nhất đến từ 2 câu `out_of_range`, nơi "ngữ cảnh rỗng" mới là đúng).
   Không có câu nào mất chunk đúng vì bộ lọc.

## 3. Năm câu sai còn lại (nhánh `temporal_filter`)

### 3.1 `ld_t04b` — văn bản còn hiệu lực trích lại nội dung đã bị thay thế ⚠️ lỗi thật

Hỏi số năm đóng BHXH tối thiểu để hưởng lương hưu tại `2025-09-01` (đáp án đúng: **15 năm**,
Điều 64 Luật BHXH 2024). Hệ thống trả lời **20 năm** và dẫn "Điều 54".

Nguồn của câu trả lời sai **không phải** Luật BHXH 2014 — chunk đó đã bị lọc đúng. Thủ phạm là
`45-2019-QH14__d219` — Điều 219 Bộ luật Lao động 2019 ("Sửa đổi, bổ sung một số điều của các
luật có liên quan"), trong đó **trích nguyên văn** Điều 54 Luật BHXH 2014 để sửa nó.

Đây là giới hạn có thật của cách làm hiện tại: hiệu lực được gắn cho **văn bản chứa** chunk.
BLLĐ 2019 vẫn còn hiệu lực ở mọi mốc, nên điều khoản sửa đổi nằm trong nó cũng "còn hiệu lực",
dù nội dung nó trích đã bị Luật BHXH 2024 thay thế. `cleaner.py` đã có
`strip_consolidated_quotations` xử lý trường hợp tương tự trong VBHN, nhưng luật gốc chứa điều
sửa đổi thì chưa được xử lý.

**Hướng sửa (chưa làm — ngoài phạm vi lần này):** đánh dấu các điều "sửa đổi, bổ sung luật
khác" bằng metadata riêng và gắn `effective_to` theo vòng đời của *khoản được trích*, hoặc loại
chúng khỏi ngữ cảnh trả lời và chỉ giữ vai trò lược đồ sửa đổi.

### 3.2 `ld_t12a` — model suy luận sai từ đúng ngữ cảnh ⚠️ lỗi sinh câu trả lời

Hỏi "lao động nữ sinh con thứ hai được nghỉ thai sản mấy tháng?" tại `2026-03-01`. Ngữ cảnh
đúng và sạch (`context_clean = true`, chỉ có bản khoản 1 Điều 139 hiệu lực 2021). Đáp án đúng:
**06 tháng** — quy định "con thứ hai = 07 tháng" chưa có hiệu lực trước 2026-07-01.

Hệ thống trả lời **07 tháng**, do đọc nhầm quy định kế bên: "sinh đôi trở lên thì tính từ con
thứ 02 trở đi, cứ mỗi con được nghỉ thêm 01 tháng" (dành cho sinh đôi) thành quy định cho "con
thứ hai". Đây là lỗi generation thuần tuý — bộ lọc thời gian đã làm đúng phần việc của nó.
Đáng chú ý vì con số sai lại **trùng** với con số của bản sửa đổi chưa có hiệu lực, nên nhìn
thoáng qua rất dễ tưởng là lỗi version.

### 3.3 `ld_t06a`, `ld_t07b`, `ld_t10b` — retrieval miss (không liên quan trục thời gian)

| Câu | Mốc | Chunk đúng | Kết quả |
|---|---|---|---|
| `ld_t06a` | 2024-03-01 | `58-2014-QH13__d89` (trần 20 lần lương cơ sở) | không lọt top-8 |
| `ld_t07b` | 2025-12-01 | `41-2024-QH15__d4` (chế độ BHXH tự nguyện) | không lọt top-8 |
| `ld_t10b` | 2026-05-01 | `74-2025-QH15__d38` (10 ngày làm việc) | không lọt top-8 |

Cả ba đều **trượt ngay ở retrieval thô**: `context_gold = false` ở cả ba nhánh, tức chunk đúng
không có trong top-8 ngay trước khi bộ lọc chạy. Nguyên nhân là recall của hybrid retriever
trên corpus 1146 chunk, không phải trục thời gian. Nhóm `doc_version` có `context_gold` = 70%
đều nhau ở cả ba nhánh chính là trần recall này — và nó cũng là trần của
`answer_accuracy` nhánh lọc (80% trên nhóm đó).

**Tuy nhiên, cách hỏng khác nhau rõ rệt:**

| Câu | `no_temporal` trả lời | `temporal_filter` trả lời |
|---|---|---|
| `ld_t06a` | "20 lần **mức tham chiếu**" (quy định của Luật 2024, sai thời điểm 2024-03-01) | "Tôi không tìm thấy quy định này" |
| `ld_t10b` | "**15 ngày**" (quy định Luật Việc làm 2013, sai thời điểm 2026-05-01) | "Tôi không tìm thấy quy định này" |

Cả hai đều bị chấm 0 điểm, nhưng baseline **tự tin trả lời sai theo luật của thời kỳ khác**,
còn nhánh có bộ lọc **từ chối**. Với tra cứu pháp luật, đây là khác biệt đáng kể hơn con số
accuracy thể hiện.

## 4. Sửa thước đo (ghi lại để minh bạch)

Lần chấm đầu có 6 câu sai; `ld_t09b` là **lỗi của thước đo**, không phải của hệ thống: câu trả
lời viết `ngày làm việc thứ **11**` (in đậm markdown), nên phép so chuỗi với gold `"thứ 11"`
trượt. Đã sửa `_norm()` trong `eval/temporal_eval.py` để bỏ ký tự nhấn mạnh markdown, rồi
**chấm lại từ câu trả lời đã lưu** (`--rescore`, không gọi lại LLM, không đổi một ký tự nào
của gold set). `answer_accuracy` nhánh lọc: 80.0% → 83.3%. Ground truth không bị chỉnh sau khi
nhìn output — nguyên tắc trong `SPEC-eval-goldset.md`.

## 5. Hạn chế của kết quả này

- **Corpus nhỏ và chỉ 1 khoản có 2 phiên bản thật.** Trong 4 khoản BLLĐ bị sửa in-place
  (`sua_doi_in_place` của manifest), chỉ Điều 139 khoản 1 có đủ toàn văn cả hai bản. Ba khoản
  còn lại (Điều 59 k2 điểm a, Điều 61 k3, Điều 154 k8a) chỉ có bản hiện hành — `validate_corpus.py`
  cảnh báo "point-in-time trước 2026-01-01 sẽ rơi về câu chữ hiện tại". Nhóm `clause_version`
  vì thế chỉ có 4 câu, đều xoay quanh một khoản.
- **n = 30, mọi con số có sai số lớn.** Chênh lệch 46.7% → 83.3% là lớn so với n, nhưng
  từng nhóm con (n = 2 đến 4) thì không đủ để kết luận riêng lẻ.
- **Chấm bằng so chuỗi**, không phải LLM-judge. Bắt tốt câu sai số/sai phiên bản, nhưng không
  đánh giá được diễn đạt hay tính đầy đủ của phần còn lại trong câu trả lời.
- **Trần recall 70%** ở nhóm `doc_version` là nút thắt tiếp theo: cải thiện retrieval sẽ nâng
  trần của cả ba nhánh, không riêng nhánh có bộ lọc.

## 6. Tái lập

```powershell
# Chạy lại toàn bộ (cần GROQ_API_KEY hoặc GOOGLE_API_KEY)
python eval/temporal_eval.py --output reports/temporal_eval.json

# Chỉ đo tầng ngữ cảnh, không gọi LLM (tất định)
python eval/temporal_eval.py --retrieval-only

# Chấm lại báo cáo đã có sau khi sửa quy tắc chấm
python eval/temporal_eval.py --rescore reports/temporal_eval.json --output reports/temporal_eval.json
```
