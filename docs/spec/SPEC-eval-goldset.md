# Spec: `eval-goldset`

> Module 7/9. Phụ thuộc: `question-library`, `temporal-retrieval`, `scope-gate`. Xem `SPEC.md`.
> **Ràng buộc bắt buộc:** memory `eval-methodology` + `documind-benchmark-integrity`.

## Objective

Gold set mới cho corpus lao động, thay `data/eval/test_questions.json` (đang banking). Gold **viết + commit TRƯỚC khi chạy hệ thống**. Báo accuracy **thật** + failure analysis. Không con số vô căn cứ trong README (rủi ro CV lớn nhất — memory `documind-benchmark-integrity`).

## Contract: `data/eval/test_questions.json`

```json
[
  {
    "id": "ld_qa_01",
    "question": "...",
    "ground_truth": "Theo Điều X <số hiệu>, ...",
    "category": "cho_vay|thoi_viec|thai_san|bhxh|bhtn|luong_toi_thieu|tuoi_huu|...",
    "source_doc": "45-2019-QH14",
    "source_clause": "45-2019-QH14__d139_k1",
    "as_of_date": null,
    "expected_behavior": "answer"
  }
]
```

Bổ sung so với format cũ:
- `source_clause`: `clause_uid` (dùng lại từ `question-library.primary_dieu` khi trùng).
- `as_of_date`: set cho câu temporal; null = mặc định hôm nay.
- `expected_behavior`: `answer` | `refuse_out_of_scope` | `formula_only` | `time_out_of_range`.

## Thành phần gold set (≥ 50 câu, ≥ 3 nhóm)

| Nhóm | Số câu | Nội dung |
|---|---|---|
| A. Tra cứu tình huống | ≥ 30 | từ `question-library` (dùng lại), mỗi tình huống ≥3 câu; ground truth viết tay từ toàn văn |
| B. Ngoài phạm vi (OOC) | ≥ 12 | 12 loại §4 `CORPUS_SPEC.md`, mỗi loại ≥1; `expected_behavior=refuse_out_of_scope` + `out_of_scope_kind` kỳ vọng |
| C. Temporal | ≥ 8 | cùng câu hỏi 2 `as_of_date` khác nhau → 2 ground truth khác nhau (lương tối thiểu, BHXH 2014↔2024, BHTN, Điều 139); + 1–2 câu `time_out_of_range` |

## Metrics báo cáo

- **Retrieval:** hit_rate@k, MRR — trên nhóm A (câu có `source_clause`). Kỳ vọng **< 100%** với corpus đủ lớn (nếu = 100% trên tập nhỏ → red flag, mở rộng tập / thêm câu khó).
- **Scope classification:** precision/recall trên nhóm B + phần A bị nhầm là OOC; confusion theo `out_of_scope_kind`.
- **Temporal correctness:** % câu nhóm C mà đáp án khớp ground truth của đúng `as_of_date` (không lẫn version).
- **Answer quality:** RAGAS (faithfulness, answer_relevancy, context_precision) qua `eval/run_evals.py` — bước 2, từ cache.
- **Failure analysis:** liệt kê mọi câu sai + phân loại nguyên nhân (retrieval miss / version sai / scope nhầm / generation).

## Boundaries (delta)

- **Never** chỉnh ground truth sau khi xem output hệ thống. Sai thì ghi vào failure analysis.
- **Never** đưa con số vào README trước khi `reports/benchmark_results.json` khớp.
- Gold set commit ở 1 commit **riêng, trước** commit chạy eval.
- Giữ `eval/rag_comparison.py` ablation 4 bước (BM25 → dense → hybrid → hybrid+rerank) — mở rộng cho scope + temporal, không viết lại.

## Công việc

1. Viết `data/eval/test_questions.json` (≥50 câu, 3 nhóm) — **commit trước**.
2. `eval/scope_eval.py` (mới): chạy nhóm B qua `scope_gate`, chấm precision/recall.
3. `eval/temporal_eval.py` (mới): chạy nhóm C với `as_of_date`, so ground truth.
4. Mở rộng `eval/rag_comparison.py` / `run_evals.py` gọi 2 module trên + gộp báo cáo `reports/benchmark_results.json`.
5. Chạy thật (cần GROQ/GEMINI key), lấy số, viết `reports/failure_analysis.md`.

## Success Criteria

- [ ] `data/eval/test_questions.json`: ≥50 câu, ≥30 nhóm A / ≥12 B / ≥8 C; 0 câu banking; mọi câu A có `source_clause` tồn tại.
- [ ] Test cấu trúc file gold xanh (mọi câu đủ field; `as_of_date` hợp lệ; `source_clause` khớp index).
- [ ] `python eval/run_evals.py --skip-ragas` chạy sạch, in retrieval + scope + temporal metrics.
- [ ] `reports/benchmark_results.json` có số thật cho cả 4 strategy + scope + temporal; hit_rate không phải 1.0 đồng loạt (hoặc nếu có, kèm ghi chú kích thước corpus + kế hoạch mở rộng).
- [ ] `reports/failure_analysis.md`: ≥1 case sai được mổ xẻ (kỳ vọng có vài case).
- [ ] `git log`: commit gold set đứng trước commit "run eval".

## Verify

`pytest tests/test_eval_goldset.py` (cấu trúc) + `python eval/run_evals.py --skip-ragas --limit 10` (smoke) + đọc `failure_analysis.md`.

## Open Questions

1. Corpus lao động ~18 VB × ~vài trăm điều → đủ lớn để hit_rate < 100% chưa, hay cần thêm câu nhiễu / câu khó?
2. RAGAS cần quota LLM — chạy full hay chỉ subset? (memory `gemini-free-tier-quota`: chỉ 3.1/3.5-flash-lite có RPD 500).
3. Có cần baseline "không temporal filter" để chứng minh giá trị tính năng thời gian trong báo cáo không? (đề xuất: có — 1 dòng so sánh nhóm C có/không filter).
