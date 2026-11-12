# QA agents & MCP server

Bộ tool kiểm thử và quản trị phát hành cho chính hệ thống AI này, mở ra cho agent khác qua MCP
(`src/mcp_server.py`). Code: `src/qa/`. Test: `tests/test_qa_tools.py`.

## Tool

| Tool MCP | Làm gì | LLM? | Ghi ra ngoài? |
|---|---|---|---|
| `ask_labor_law` | Hỏi đáp có trích dẫn, theo `as_of_date` — gọi HTTP tới API đang chạy | qua API | không |
| `check_labor_compliance` | Pass/fail theo 6 tiêu chí định lượng | không | không |
| `get_regression_scope` | File đổi → test, eval, câu gold bị ảnh hưởng, có cần ingest lại | không | không |
| `analyze_defects` | So 2 report eval theo câu → defect phân loại theo tầng hỏng | không | không |
| `release_scorecard` | GO/NO-GO + blocker + bảng metric | không | không |
| `draft_test_cases` | Yêu cầu → điểm mơ hồ + case nháp, kiểm hallucination, vào hàng đợi | **có** | `data/qa/review_queue.jsonl` |
| `list_pending_test_cases` / `review_test_case` | Người duyệt approve/reject | không | hàng đợi |
| `create_jira_defect` | Bug Jira từ một defect | không | Jira |
| `post_scorecard_comment` | Scorecard lên GitHub PR / GitLab MR | không | GitHub/GitLab |
| `push_test_case_to_testrail` | Case **đã duyệt** → TestRail | không | TestRail |

Mọi tool đều chạy được bằng CLI không cần MCP:
`python -m src.qa.regression_scope --base main`, `python -m src.qa.release <report> --baseline <base>`.

## Cách hoạt động

**Regression scope**: dựa trên ba quan hệ tất định. (1) Đồ thị import Python dựng bằng `ast`. (2) Tên file dữ liệu
được nhắc tới trong code, ví dụ `criteria.json` → `test_compliance.py`. (3) Một văn bản trong `data/raw/lao_dong/<doc_id>.md`
→ các câu gold có `source_doc`/`source_clause(s)`/`acceptable_clauses`/`distractor_docs` trỏ tới doc đó; các câu này
được chạy lại bằng `temporal_eval.py --ids`. Đổi `src/rag/`, corpus hoặc gold thì phải chạy eval truy hồi. Đổi
prompt, agent hoặc config thì phải chạy eval đầu cuối, vì unit test không bắt được thoái lui chất lượng.

**Defect**: một câu được tính là defect khi đúng ở baseline nhưng sai ở bản mới. Mức độ được xếp theo rủi ro cho
người dùng chứ không theo số lượng:

| Loại | Mức | Nghĩa |
|---|---|---|
| `oos_leak` | critical | Câu phải từ chối mà hệ thống lại trả lời (hallucination ngoài phạm vi) |
| `stale_law` | critical | Câu trả lời chứa số liệu/văn bản đã hết hiệu lực (`expect_absent`) |
| `wrong_verdict` | critical | Kết luận Có/Không bị ngược |
| `generation` | major | Gold có trong ngữ cảnh mà câu trả lời vẫn sai |
| `retrieval` | minor | Gold rơi khỏi top-8 |
| `infra` | info | Câu đo thiếu bước (quota, 503, trích nguyên văn) — chạy lại |

**Scorecard**: NO-GO khi pytest đỏ, khi metric tụt quá ngưỡng của `eval/ci_gate.py` (dùng chung một nguồn),
khi `answer_accuracy` giảm quá 0.02 (hai lượt cùng cấu hình lệch khoảng 0.015), khi có câu đo thiếu bước, khi gold set
khác baseline, hoặc khi có bất kỳ defect critical nào. Defect major/minor chỉ là cảnh báo và người review quyết. CI comment
scorecard lên mỗi PR (job `eval`).

**Test case + human-in-the-loop**: LLM chỉ được nháp. Bước kiểm hallucination là tất định, không dùng LLM-judge,
vì judge đã cho 1.0 cả câu sai (xem `CLAUDE.md`). Mỗi case phải có `source_quote` nguyên văn trong yêu cầu, và mọi con số
trong `expected` phải xuất hiện trong yêu cầu. Case vi phạm vẫn vào hàng đợi nhưng mang `issues`; muốn approve thì
người duyệt phải ghi lý do. Chỉ case `approved` mới đẩy được sang TestRail.

## Cài & đăng ký

```powershell
pip install -r requirements-dev.txt   # mcp==1.12.4 — bản mới hơn kéo starlette 1.x, vỡ pin FastAPI
```

- Claude Code: `.mcp.json` ở gốc repo (tự nhận).
- Kiro: `.kiro/settings/mcp.json`. `autoApprove` **chỉ** gồm tool đọc; tool ghi và `review_test_case`
  luôn hỏi người dùng.
- `ask_labor_law` cần backend chạy (`.\start.ps1`), hoặc đặt `DOCUMIND_API_URL` trỏ tới production và
  `DOCUMIND_API_KEY` để vào tenant.
- Tracker: điền `JIRA_BASE_URL`/`JIRA_EMAIL`/`JIRA_API_TOKEN`/`JIRA_PROJECT_KEY`, `GITHUB_TOKEN`,
  `GITLAB_TOKEN` (+`GITLAB_BASE_URL`), `TESTRAIL_BASE_URL`/`TESTRAIL_USER`/`TESTRAIL_API_KEY` vào `.env`
  (đã khai trong `src/config.py`). Thiếu biến nào thì tool tương ứng chạy **dry-run**: chỉ trả request sẽ gửi.

## Giới hạn

- Adapter Jira/GitLab/GitHub/TestRail mới được kiểm bằng dry-run payload, **chưa chạy trên instance thật**.
  Jira dùng REST v2 với description dạng wiki markup. TestRail giả định template "Test Case (Steps)"
  (`custom_steps_separated`); instance có custom field bắt buộc khác sẽ trả 400.
- Regression scope không thấy import động (`importlib`) và không thấy đường dẫn ghép chuỗi. Gặp các trường hợp
  đó thì chạy `pytest -q` toàn bộ. Module trung tâm như `generator.py` kéo theo phần lớn test, và đó là kết quả đúng.
- Gold bị ảnh hưởng chỉ tính ở mức văn bản, chưa tính ở mức điều: sửa một điều sẽ chạy lại mọi câu của văn bản đó.
- Defect dựa trên bộ chấm so chuỗi của `temporal_eval`, mà bộ này từng báo sai câu đúng (LESSONS). Vì vậy phải đọc
  câu trả lời trong `evidence` trước khi mở bug.
- Các thử nghiệm sinh test case đã dùng yêu cầu dạng văn bản tự do. Chưa có đo lường độ phủ hay độ chính xác trên một
  bộ yêu cầu chuẩn.
- Không có Confluence và Azure DevOps.

## Kiểm soát rủi ro

- LLM không bao giờ tự quyết bước cuối: case cần người approve, tool ghi cần credential của người dùng và cần xác nhận
  trên client MCP.
- `_report()` chỉ đọc file `.json` nằm trong repo, vì đường dẫn đến từ LLM nên là input không tin được.
- Tool QA không gửi dữ liệu ra ngoài khi chưa cấu hình tracker.
