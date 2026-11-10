
## 2026-11-09: nâng cấp nền tảng (branch `feat/platform-upgrade`)

Chi tiết và cách đo ở `docs/PLATFORM.md`. Tóm tắt:
- `/upload` nhận DOCX và XLSX (`loader.load_upload`), có chặn zip bomb.
- Intent `action`: agent chỉ tạo đề xuất `create_hr_request` ở trạng thái `pending`. Duyệt qua `/api/v1/actions/{id}/approve` (admin key + `X-Approver`), thực thi đúng một lần. Bảng `agent_actions` và `hr_requests` nằm trong `sqlite_db`.
- `eval/acl_eval.py`: tỷ lệ pass kiểm soát truy cập theo phòng ban trên 3 đường (Chroma, Qdrant, BM25); hiện 528/528 mỗi đường. `temporal_eval` có thêm nDCG@3/5/8.
- `src/usage.py` và `/api/v1/metrics`: token, chi phí, p50/p95 theo request (REST và WS). Giá model lấy từ `DEFAULT_PRICES` hoặc `LLM_PRICES_JSON`.
- Gotcha: test nào patch `documents.load_pdf` phải đổi sang `documents.load_upload`.
