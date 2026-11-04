# AGENTS.md — nhật ký làm việc

Hướng dẫn dự án nằm ở `CLAUDE.md`; file này chỉ ghi trạng thái công việc cho các agent khác.

## 2026-11-04 — Tách nhánh judge độ đúng

**Đã làm**
- Worktree `DocuMindAI-c4-wt`: việc dở được lưu ở commit WIP `322586b` trên `fix/output-contract-amend-invariant`.
- Phần judge chuyển sang `eval/judge-correctness` (từ `origin/main` 4d59779). Phần `generator.py`,
  `test_ingestion.py`, `test_rag.py` bị bỏ vì mọi dòng thêm đã có trên `main` (PR #10).
- `pytest`: 684 passed, 3 skipped.

**Còn lại**
- Merge PR #11 (`fix/oos-fallback-model-verdict-gold`) vào `main`, rồi rebase `eval/judge-correctness` lên đó (conflict ở `docs/LESSONS.md`):
  hai nhánh cùng sửa `eval/temporal_eval.py`, `tests/test_eval_metrics.py`, `CLAUDE.md`, `docs/LESSONS.md`.
- Chấm mù 34 câu phần C để đo judge ngoài mẫu; chưa dùng judge làm gate.

**Lưu ý**
- Worktree không có `.venv`: dùng `../DocuMindAI/.venv/Scripts/python.exe`. Venv đó không có `ruff`.
