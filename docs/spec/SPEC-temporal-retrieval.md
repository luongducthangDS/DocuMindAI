# Spec: `temporal-retrieval`

> Module 3/9. Phụ thuộc: `clause-schema-ingestion`. Xem `SPEC.md`.

## Objective

Cho mỗi truy vấn một `as_of_date` (mặc định = hôm nay), và chỉ để retrieval thấy **bản điều/khoản có hiệu lực tại `as_of_date`**. Lọc dựng ở **tầng trên** retrieval node — **không sửa `retrieve_node` / `build_hybrid_retriever`**.

Tên gọi đúng: *"temporal-aware retrieval + hiệu lực metadata cấp điều/khoản"*. KHÔNG gọi "point-in-time reconstruction".

## Contract: `as_of_date`

- Kiểu: `date` (ISO `YYYY-MM-DD`). Nguồn: người dùng chọn ở UI (`frontend-boundary-ui`) → API `/query` body field `as_of_date` (optional) → `AgentState["as_of_date"]` → xuống filter + generator.
- Mặc định khi thiếu: hôm nay. Câu trả lời nêu rõ *"theo quy định hiện hành tại &lt;ngày&gt;"*.
- `as_of_date < meta.earliest_point_in_time` (2015-01-01): trả lời "ngoài khoảng thời gian corpus phủ", không suy đoán. (Cờ này scope_gate cũng đọc — xem `SPEC-scope-gate.md` `time_out_of_range`.)

## Quy tắc "version áp dụng tại T" (từ `CORPUS_SPEC.md` §2.3)

```
effective_from <= T
AND (effective_to IS NULL/rỗng OR effective_to > T)
AND status != 'repealed' tại T
```

## Boundaries (delta)

- **Never sửa** `src/rag/retriever.py` node retrieval / fusion / rerank logic.
- **Ask first:** thêm field vào `AgentState` hoặc body schema `/query` (`src/api/schemas.py`).
- Filter chạy như **post-processor** sau retrieval (lọc chunk trả về) HOẶC **metadata pre-filter** ở vector query (`where` clause ChromaDB). Đề xuất: pre-filter ở `retrieve_direct_chroma` + post-filter cho path LlamaIndex (không đụng fusion). Chốt cách trong plan.
- Không thêm dependency.

## Công việc

1. **`AgentState`:** thêm `as_of_date: str` (ISO) + `time_out_of_range: bool`. Set trong `run_agent()` từ tham số mới `as_of_date: str | None`.
2. **API:** `QueryRequest` (schemas.py) thêm `as_of_date: str | None = None`. `query.py` route truyền xuống `run_agent`.
3. **Filter layer** (`src/rag/temporal.py` — module mới, thuần hàm):
   - `versions_in_force(chunks, as_of) -> filtered_chunks` áp quy tắc trên.
   - `is_out_of_range(as_of, earliest) -> bool`.
   - Khi 2 version cùng `clause_uid` cùng thoả (không nên xảy ra nếu schema đúng) → lấy `effective_from` muộn nhất + log warning.
4. **Wire vào graph:** trong `retrieve_node` (chỗ nhận `chunks` xong, trước khi return) gọi `versions_in_force`. Đây là 1 dòng ở tầng node, không phải trong retriever. Nếu strict "không sửa retrieve_node" → thêm node `do_temporal_filter` giữa `do_retrieve` và `do_grade`. **Đề xuất: node riêng** `do_temporal_filter` (rõ ràng, testable, đúng "tầng trên").
5. **Generator:** truyền `as_of_date` vào prompt context để câu trả lời nêu mốc + chọn đúng số khi có nhiều mốc doc-level (vd lương tối thiểu vùng).
6. **Persist/log:** `as_of_date` vào `steps` để UI hiển thị "đang tra theo mốc &lt;ngày&gt;".

## Success Criteria

- [ ] `POST /query {"query": "...", "as_of_date": "2025-06-01"}` trả câu trả lời chỉ dựa trên VB có hiệu lực 2025-06-01 (vd BHXH → Luật 2014, không phải 2024).
- [ ] Cùng câu "chế độ nghỉ thai sản", `as_of_date=2026-05-01` vs `2026-09-10` → 2 câu trả lời khác nhau, trích đúng bản Điều 139 tương ứng.
- [ ] `as_of_date=2010-01-01` → trả lời nêu "corpus chỉ phủ từ 2015-01-01".
- [ ] Thiếu `as_of_date` → dùng hôm nay, câu trả lời có chuỗi "hiện hành tại 2026-...".
- [ ] `tests/test_temporal.py`: bảng version → tập in-force tại nhiều mốc T; out-of-range; tie-break.
- [ ] `retrieve_node` / retriever không đổi hành vi khi `as_of_date` = hôm nay (regression: `pytest tests/test_rag.py test_agent.py` xanh).

## Verify

`pytest tests/test_temporal.py tests/test_agent.py -v` + 3 truy vấn thủ công (3 mốc) qua API.

## Open Questions

1. Node riêng `do_temporal_filter` (đề xuất) vs lọc inline trong `retrieve_node`?
2. ChromaDB `where` filter theo `effective_from/to` string compare có đủ tin không, hay lọc hết ở Python sau khi lấy top-k rộng hơn (vd top-30 thay top-20)?
3. `as_of_date` cho intent `compare` (so sánh 2 mốc thời gian của cùng quy định) — có phải use case riêng cần UI khác không? (đề xuất: hoãn, ngoài scope module này).
