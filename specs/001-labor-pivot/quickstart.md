# Phase 1 — Quickstart: kiểm chứng 001-labor-pivot

Các kịch bản chạy được để chứng minh feature hoạt động đầu-cuối. Chi tiết thực thể xem [data-model.md](./data-model.md), giao kèo xem [contracts/](./contracts/).

## Điều kiện

```powershell
# Interpreter có đủ dependency (đo 2026-09-18: Python310, KHÔNG phải venv mặc định của shell)
$py = "C:\Users\Lenovo\AppData\Local\Programs\Python\Python310\python.exe"
& $py -c "import loguru, fastapi; print('deps OK')"
```

Vector store phải có corpus lao động đã nạp (`data/chroma_db/`, collection `documind_legal`).

## 1. Bộ test phải xanh trước mọi tuyên bố

```powershell
& $py -m pytest -q -o addopts=""
```

Kỳ vọng: toàn bộ pass. Ghi lại con số và ngày chạy — README/CLAUDE.md phải khớp đúng số đó (nguyên tắc V).

## 2. Manifest corpus

```powershell
# backend
& $py -m uvicorn src.api.main:app --port 8081
# rồi gọi endpoint manifest
```

Kỳ vọng: `so_van_ban` = số văn bản VERIFIED trong `docs/corpus/corpus_manifest.yaml` (hiện là 21), `ngoai_pham_vi[]` không rỗng. Thêm/bớt một văn bản VERIFIED trong manifest rồi gọi lại → số đổi theo, không sửa frontend.

## 3. Cổng phạm vi

Gửi lần lượt: (a) câu trong phạm vi, (b) câu thuộc 12 loại ngoài phạm vi, (c) mô phỏng lỗi bộ phân loại.

Kỳ vọng:
- (a) `verdict = in_scope`, có trích dẫn.
- (b) `verdict = out_of_scope`, có phạm vi + loại việc không xử lý + ≤ 3 gợi ý, **không** có nguồn.
- (c) `verdict = undetermined`, thông điệp không phân loại được, **không** trả lời.

## 4. Thời điểm hiệu lực

```powershell
& $py eval/temporal_eval.py
```

Gửi cùng một câu với `as_of_date` trước và sau một mốc đã biết → hai đáp án khác nhau, mỗi đáp án trích đúng `version_id` tương ứng, và câu trả lời nêu rõ thời điểm áp dụng.

## 5. Gold set

```powershell
& $py eval/run_evals.py --strategies dense rerank --retrieval-only
```

Kỳ vọng: gold set ≥ 50 câu, phủ cả ba nhóm; báo cáo kèm phân tích thất bại cho mọi câu sai; mọi con số đưa vào README kèm tên run và ngày.

## 6. Thư viện câu hỏi

Chạy toàn bộ câu trong thư viện qua cổng phạm vi → 0 câu bị từ chối.
