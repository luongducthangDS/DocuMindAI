# Quality Engineering Portfolio

## Phạm vi

DocuMindAI là ứng dụng hỏi đáp pháp luật có API FastAPI, SQLite và vector store. Bộ kiểm thử QE tập trung
vào tính đúng của hợp đồng API, observability và cách ứng dụng phản hồi khi dependency gặp lỗi. Các test
được chạy offline với database tạm và dependency giả lập; không cần API key, không gửi dữ liệu ra ngoài.

## Chiến lược kiểm thử

| Lớp | Trọng tâm | Ví dụ trong repo |
|---|---|---|
| Unit / functional | Schema, chuẩn hóa, compliance, router và guardrail | `tests/test_compliance.py`, `tests/test_guardrails.py`, `tests/test_router.py` |
| API / integration | Request-response qua FastAPI `TestClient`, kiểm tra status, JSON và persistence | `tests/test_metrics_api.py`, `tests/test_health.py`, `tests/test_degraded_flags.py` |
| Resilience | Health status khi corpus lỗi/rỗng, lỗi initialization và tín hiệu degraded | `tests/test_health.py`, `tests/test_degraded_flags.py` |
| Security / access control | Auth, upload, session isolation, origin và client IP | `tests/test_access_control.py`, `tests/test_security_hardening.py`, `tests/integration/` |
| Regression | Mỗi lỗi có test tái hiện để ngăn hồi quy | `tests/test_metrics_api.py` |
| Evaluation | Retrieval metrics và chất lượng câu trả lời trên bộ đánh giá | `eval/`, `tests/test_eval_metrics.py` |

### API test cases

| ID | Ưu tiên | Scenario | Kết quả mong đợi | Kiểm thử |
|---|---|---|---|---|
| API-MET-01 | P1 | Gọi `GET /api/v1/metrics` khi SQLite có query hợp lệ | HTTP 200; tổng query, latency trung bình (không tính latency 0), số session, error rate và corpus chunks đúng | `test_metrics_api_reports_database_aggregates` |
| API-MET-02 | P1 | Gọi metrics khi schema `query_log` thiếu/hỏng | HTTP 503 với thông báo ổn định, không trả bộ metrics toàn số 0 như kết quả thành công | `test_metrics_api_returns_service_unavailable_when_database_is_invalid` |
| API-HEALTH-01 | P0 | Vector collection không mở được hoặc rỗng | HTTP health endpoint phản ánh trạng thái không khỏe trong JSON body | `tests/test_health.py` |
| API-QUERY-01 | P1 | Một lượt query chạy ở degraded mode | Response REST/WebSocket có cờ degraded tương ứng | `tests/test_degraded_flags.py` |
| API-AUTH-01 | P0 | Request không có quyền với API được bảo vệ | Request bị từ chối, không thực hiện upload/thao tác bảo vệ | `tests/test_access_control.py`, `tests/integration/test_upload_auth_p0.py` |

## Defect lifecycle: API-MET-02

| Thuộc tính | Mô tả |
|---|---|
| Tiêu đề | Metrics API báo thành công với số liệu giả khi SQLite lỗi |
| Mức độ / ưu tiên | Medium / P1 — số 0 có thể bị hiểu nhầm là hệ thống không có query thay vì metrics không đọc được |
| Tiền điều kiện | `query_log` không tồn tại hoặc SQLite không đọc được |
| Bước tái hiện | Gọi `GET /api/v1/metrics` trên database chưa khởi tạo schema |
| Kết quả trước sửa | HTTP 200 với mọi metric bằng 0 |
| Kết quả mong đợi | HTTP 503 và thông báo lỗi không tiết lộ đường dẫn nội bộ |
| Sửa đổi | Phát tín hiệu `503 Service Unavailable` sau khi log lỗi; SQLite connection luôn đóng kể cả query thất bại |
| Regression | Hai API test dùng SQLite tạm trong `tests/test_metrics_api.py` |

## Chạy kiểm thử

Từ thư mục gốc `DocuMindAI`:

```powershell
pytest tests/test_metrics_api.py tests/test_health.py -q
pytest -q
```

Test metrics tạo database trong thư mục tạm của pytest, seed 3 query trên 2 session, và mock phần đếm
corpus/error rate. Các assertion kiểm tra trực tiếp status code và response JSON; test lỗi xác nhận không
có success-shaped fallback. Những test khác có thể cần dependency tùy chọn hoặc dữ liệu mẫu theo
`requirements-dev.txt` và fixtures hiện tại.

## Giới hạn

- Chưa có Postman collection trong repo; kiểm thử API hiện tự động hóa bằng pytest `TestClient`.
- Test dùng dependency mock không thay thế được smoke test trên deployment với Gemini/Chroma/Qdrant thật.
- Không dùng dữ liệu cá nhân hoặc hồ sơ ngân hàng thật; nội dung domain là dữ liệu pháp luật công khai.
