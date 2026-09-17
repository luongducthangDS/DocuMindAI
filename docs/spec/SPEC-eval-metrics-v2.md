# Spec: `eval-metrics-v2`

> Phase 0a. Phụ thuộc: `clause-schema-ingestion` (metadata `clause_uid` / `version_id`).
> Quyết định đi kèm: `docs/decisions/DEC-0004-metric-theo-clause-uid.md`.

## Objective

Metric retrieval phải chấm **đúng điều/khoản, đúng bản hiệu lực**, bằng cách so khớp
`clause_uid` + `version_id` của chunk lấy được với gold — thay cho ngưỡng token-F1 ≥ 0.15
đang dùng ở `eval/metrics.py`.

Lý do tồn tại: ngưỡng token-F1 coi một chunk là "trúng" khi nó chỉ **cùng chủ đề** với
`ground_truth`. Một chunk **sai bản hiệu lực** (NĐ 74/2024 thay vì NĐ 293/2025) có token-F1
rất cao với gold — tức chính lỗi mà `do_temporal_filter` sinh ra để chặn lại thì bộ đo
**không nhìn thấy**. Bộ đo không được mù ở đúng chỗ hệ thống dễ sai nhất.

## Contract dữ liệu

### Bản ghi chunk (`record`)

Metric nhận `list[list[dict]]` — mỗi câu hỏi một danh sách bản ghi **theo đúng thứ tự rank**:

```python
{"clause_uid": str, "version_id": str, "doc_id": str, "text": str}
```

`eval/metrics.py` **không** import `src.rag.retriever`: nó nhận dict thuần để test được mà
không cần dựng retriever. Việc chuyển `RetrievedChunk` → record nằm ở `eval/rag_comparison.py`.

### Gold

Đọc theo thứ tự ưu tiên, dừng ở cái đầu tiên có giá trị:

1. `item["gold_clause_uids"]: list[str]` — schema gold v2 (Phase 1).
2. `item["source_clause"]: str` — schema gold temporal hiện có, coi như list 1 phần tử.
3. không có → câu đó **bị bỏ qua** (skip), không rơi về token-F1.

`item["distractor_versions"]: list[str]` (tuỳ chọn) — các `version_id` sai bản hiệu lực.

## Định nghĩa "trúng" (hit)

Một record trúng khi **cả hai** điều kiện:

```
record.clause_uid ∈ gold_clause_uids
AND record.version_id ∉ distractor_versions
```

Điều kiện thứ hai là điểm khác biệt so với bộ đo cũ: **đúng điều nhưng sai bản = không trúng**.

## Metric

| Hàm | Trả về | Định nghĩa |
|---|---|---|
| `recall_at_k(items, retrieved, k)` | `float \| None` | tỉ lệ câu có ≥1 record trúng trong top-k |
| `mrr_at_k(items, retrieved, k)` | `float \| None` | trung bình `1/rank` của record trúng đầu tiên; 0 nếu không có |
| `citation_validity(answers, retrieved)` | `float \| None` | tỉ lệ câu **có citation** mà mọi `[N]` đều nằm trong `1..len(records)` |
| `citation_groundedness(items, answers, retrieved)` | `float \| None` | tỉ lệ câu có gold + có citation mà **≥1** `[N]` trỏ tới record trúng gold |
| `lexical_overlap(a, b)` | `float` | token-F1 — **proxy**, đổi tên từ `_f1_overlap` |

`None` = không có câu nào đủ điều kiện chấm. Mọi hàm trả kèm số câu bị skip qua `compute_all`.

**`citation_groundedness` dùng "≥1 citation trúng", không phải "mọi citation trúng"** — một câu
trả lời đúng vẫn được phép trích thêm chunk bối cảnh ngoài gold. Đo "mọi citation trúng" sẽ
phạt hành vi hợp lệ.

## Boundaries (delta)

- **Never:** `eval/metrics.py` không import `src.rag.*`, không gọi LLM, không gọi embedder.
- **Never:** không sửa `eval/temporal_eval.py::score_context` — logic ở đó đã đúng và đã có
  số; 0a **tái dùng cùng quy tắc**, không nhân bản hành vi khác đi.
- **Never:** không đụng RAGAS, không đụng gold set 200 câu (Phase 1), không đụng
  `src/rag/generator.py`.
- **Ask first:** đổi schema gold, đổi tên collection, xoá file report.
- `lexical_overlap` **được giữ** cho `answer_correctness` (so *câu trả lời* với *ground_truth* —
  việc khác hẳn việc chọn chunk). Nó bị cấm dùng làm headline metric trong README/DEC.

## Công việc

1. **`src/guardrails.py`:** thêm `cited_indices(answer) -> list[int]` (bắt cả `[1]` lẫn `[1, 2]`).
   **Không** đổi hành vi `validate_citations`.
2. **`eval/metrics.py`:**
   - `_f1_overlap` → `lexical_overlap` (public, docstring ghi rõ là proxy).
   - Xoá `_chunk_relevant`, `hit_rate`, `mrr` (token-F1).
   - Thêm `recall_at_k`, `mrr_at_k`, `citation_validity`, `citation_groundedness`.
   - `compute_all` nhận `retrieved_list` thay `contexts_list`, trả thêm `n_scored` / `n_skipped`.
3. **`eval/rag_comparison.py`:**
   - 4 strategy đổi contract `retrieve_and_answer` → trả `(answer, chunks: list[RetrievedChunk])`.
   - `run_strategy` derive `contexts` (text, cho RAGAS) + `records` (metadata, cho metric mới).
   - Cache lên `"format": 2`, lưu `records`; cache format < 2 bị từ chối và sinh lại.
   - Bảng in đổi nhãn `hit_rate@K` → `recall@k`, thêm dòng citation.
4. **`eval/scoring_ab.py`** (mới): chạy retrieval một lần trên gold temporal, chấm **song song**
   hai cách (token-F1 cũ vs clause_uid mới), in bảng chênh lệch → số cho DEC-0004.
5. **`tests/test_eval_metrics.py`** (mới): phủ toàn bộ hàm trên, gồm ca sai-bản-hiệu-lực.

## Success Criteria

- [ ] Chunk cùng `clause_uid` nhưng `version_id ∈ distractor_versions` → **không** tính trúng
      (có test; đây là ca mà bộ đo cũ tính trúng).
- [ ] Câu gold thiếu `clause_uid` → skip, `compute_all` báo `n_skipped`, **không** rơi về token-F1.
- [ ] `citation_validity` bắt được `[9]` khi chỉ có 8 chunk.
- [ ] `citation_groundedness` = 0 khi mọi `[N]` trỏ tới chunk ngoài gold.
- [ ] `python eval/scoring_ab.py` chạy không cần API key, in bảng 2 cách chấm.
- [ ] `pytest` xanh toàn bộ (166 test cũ + test mới).

## Verify

```powershell
pytest tests/test_eval_metrics.py -v --no-cov
pytest -q --no-cov                      # không regression
python eval/scoring_ab.py               # số cho DEC-0004
```

## Open Questions

1. Repo đang có **3** regex parse citation: `guardrails._CITATION_RE` (chỉ `[N]`),
   `generator._CITATION_BRACKET_RE` (cả `[1, 2]`), `metrics._CITATION_RE` (cả `[Điều 48]`).
   Hợp nhất chúng là việc riêng, **không** làm trong 0a vì sẽ đổi hành vi guardrail production.
2. `nDCG@k` — chưa làm. Với gold ≤ 3 clause/câu thì recall@k + MRR đã đủ phân biệt; xem lại
   khi gold v2 có câu `multi_clause` nhiều điều.
3. Ngưỡng regression gate cho CI (Phase 1) chọn bao nhiêu — cần độ nhiễu đo được trước.
