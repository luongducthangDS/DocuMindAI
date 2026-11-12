"""MCP server: DocuMind cho agent khác dùng — tra cứu luật + bộ tool QA/release.

Chạy (stdio — Claude Code, Kiro, Cursor đều spawn lệnh này):
  .venv/Scripts/python -m src.mcp_server

Tra cứu đi qua HTTP tới API đang chạy (DOCUMIND_API_URL, mặc định local 8081) thay vì nạp index
vào tiến trình MCP: cùng một đường đi với UI, cùng ACL/tenant (DOCUMIND_API_KEY), trỏ được thẳng
production. Tool QA chạy tại chỗ, tất định, trừ draft_test_cases (LLM, có kiểm + hàng đợi duyệt).

Tool GHI ra ngoài (Jira/GitLab/GitHub/TestRail) dry-run khi thiếu credential. Tool duyệt
(review_test_case) KHÔNG được để client tự duyệt — xem .kiro/settings/mcp.json (autoApprove chỉ
gồm tool đọc).
"""
# ponytail: KHÔNG `from __future__ import annotations` — FastMCP 1.12 issubclass() trên annotation thật.
import json
import os
from pathlib import Path

import requests
from mcp.server.fastmcp import FastMCP

from src.qa import integrations, regression_scope as rs, release, testcases

ROOT = Path(__file__).resolve().parents[1]
mcp = FastMCP("documind-qa")


def _report(path: str) -> dict:
    """Chỉ đọc file JSON trong repo — client là LLM, đường dẫn là input không tin được."""
    p = (ROOT / path).resolve()
    if ROOT not in p.parents or p.suffix != ".json":
        raise ValueError(f"{path}: chỉ đọc report .json trong repo")
    return json.loads(p.read_text(encoding="utf-8"))


@mcp.tool()
def ask_labor_law(query: str, as_of_date: str | None = None) -> dict:
    """Hỏi đáp pháp luật lao động/BHXH Việt Nam có trích dẫn điều khoản. as_of_date (YYYY-MM-DD) =
    tra cứu quy định hiệu lực tại ngày đó. Trả answer, sources, degraded (rỗng = đường đầy đủ)."""
    base = os.environ.get("DOCUMIND_API_URL", "http://localhost:8081").rstrip("/")
    key = os.environ.get("DOCUMIND_API_KEY", "")
    r = requests.post(f"{base}/api/v1/query", json={"query": query, "as_of_date": as_of_date},
                      headers={"x-api-key": key} if key else {}, timeout=90)
    r.raise_for_status()
    d = r.json()
    return {k: d.get(k) for k in ("answer", "sources", "degraded", "used_llm", "compliance", "latency_ms")}


@mcp.tool()
def check_labor_compliance(situation: str, as_of_date: str | None = None) -> dict:
    """Kiểm định tuân thủ định lượng (pass/fail/insufficient_info/no_match) theo 6 tiêu chí
    lao động có trích dẫn. Tất định — không LLM."""
    from src.rag.compliance import check_compliance
    return check_compliance(situation, as_of_date)


@mcp.tool()
def get_regression_scope(files: list[str] | None = None, base: str = "main") -> dict:
    """Từ file đổi (bỏ trống = git diff với base) → test cần chạy, eval cần chạy, câu gold bị ảnh
    hưởng, có cần ingest lại corpus không. Kèm lệnh chạy sẵn."""
    return rs.regression_scope(files or rs.changed_files(base))


@mcp.tool()
def analyze_defects(report: str, baseline: str = "reports/eval_baseline.json", arm: str = "pre_filter") -> dict:
    """So hai report eval theo từng câu → defect (đúng→sai) phân loại theo tầng hỏng
    (oos_leak/stale_law/wrong_verdict/generation/retrieval/infra), mức độ, bằng chứng, lệnh tái hiện."""
    return release.analyze_defects(_report(report), _report(baseline), arm)


@mcp.tool()
def release_scorecard(report: str, baseline: str = "reports/eval_baseline.json", arm: str = "pre_filter",
                      tests_passed: bool | None = None) -> dict:
    """Quality gate phát hành: GO/NO-GO kèm blocker (metric tụt quá ngưỡng CI, câu đo thiếu bước,
    defect critical, pytest đỏ) và bảng metric markdown."""
    card = release.release_scorecard(_report(report), _report(baseline), arm, tests_passed)
    return {**card, "markdown": release.to_markdown(card)}


@mcp.tool()
def draft_test_cases(requirement: str, requirement_id: str, n: int = 8, context: str = "") -> dict:
    """Phân tích yêu cầu → điểm mơ hồ + nháp test case (LLM). Mỗi case được kiểm hallucination
    (trích nguyên văn yêu cầu, không bịa số) rồi vào hàng đợi duyệt với status=pending."""
    draft = testcases.draft_test_cases(requirement, n, context)
    return {**draft, "queued_ids": testcases.park_for_review(draft, requirement_id)}


@mcp.tool()
def list_pending_test_cases() -> list[dict]:
    """Case nháp đang chờ người duyệt."""
    return testcases.pending_cases()


@mcp.tool()
def review_test_case(case_id: str, decision: str, reviewer: str, note: str = "") -> dict:
    """NGƯỜI duyệt approve/reject một case nháp. Case có issues phải kèm note mới approve được."""
    return testcases.review_case(case_id, decision, reviewer, note)


@mcp.tool()
def create_jira_defect(report: str, defect_id: str, baseline: str = "reports/eval_baseline.json",
                       arm: str = "pre_filter", project_key: str | None = None) -> dict:
    """Tạo Bug trên Jira cho một defect từ analyze_defects. Thiếu JIRA_* = dry-run (trả payload)."""
    d = release.analyze_defects(_report(report), _report(baseline), arm)
    match = next((x for x in d["defects"] if x["id"] == defect_id), None)
    if match is None:
        raise ValueError(f"{defect_id} không phải defect giữa hai report này")
    return integrations.jira_create_defect(match, project_key)


@mcp.tool()
def post_scorecard_comment(platform: str, repo: str, number: int, report: str,
                           baseline: str = "reports/eval_baseline.json", arm: str = "pre_filter") -> dict:
    """Comment release scorecard lên GitHub PR (platform=github, repo=owner/name) hoặc GitLab MR
    (platform=gitlab, repo=project id/path, number=MR iid). Thiếu token = dry-run."""
    md = release.to_markdown(release.release_scorecard(_report(report), _report(baseline), arm))
    if platform == "github":
        return integrations.github_pr_comment(repo, number, md)
    if platform == "gitlab":
        return integrations.gitlab_mr_note(repo, number, md)
    raise ValueError("platform: github | gitlab")


@mcp.tool()
def push_test_case_to_testrail(case_id: str, section_id: int) -> dict:
    """Đẩy một case ĐÃ DUYỆT sang TestRail. Thiếu TESTRAIL_* = dry-run."""
    row = next((r for r in testcases.approved_cases() if r["id"] == case_id), None)
    if row is None:
        raise ValueError(f"{case_id} chưa được approve (hoặc không tồn tại)")
    return integrations.testrail_add_case(section_id, row)


if __name__ == "__main__":
    mcp.run()
