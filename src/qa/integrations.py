"""Đẩy kết quả QA ra tracker: Jira (defect), GitLab MR / GitHub PR (comment scorecard), TestRail (case đã duyệt).

Thiếu credential ⇒ dry-run: trả về đúng request sẽ gửi, không gọi mạng. Agent/MCP client nhờ đó
xem trước được payload; gửi thật là quyết định của người có key, không phải của LLM.
"""
from __future__ import annotations

import requests

from src.config import get_settings

_TIMEOUT = 20
_PRIORITY = {"critical": "Highest", "major": "High", "minor": "Medium", "info": "Low"}
_TR_PRIORITY = {"P0": 4, "P1": 3, "P2": 2}  # TestRail mặc định: 4 Critical, 3 High, 2 Medium


def _send(method: str, url: str, payload: dict, live: bool, **kw) -> dict:
    if not live:
        return {"dry_run": True, "method": method, "url": url, "json": payload}
    r = requests.request(method, url, json=payload, timeout=_TIMEOUT, **kw)
    r.raise_for_status()
    return {"dry_run": False, "status": r.status_code, "response": r.json() if r.content else {}}


def jira_create_defect(defect: dict, project_key: str | None = None) -> dict:
    """defect: một phần tử `defects` của release.analyze_defects."""
    s = get_settings()
    ev = defect["evidence"]
    desc = (f"*Câu hỏi:* {defect['question']}\n*Thời điểm tra cứu:* {defect.get('as_of_date')}\n"
            f"*Loại:* {defect['category']} — *mức:* {defect['severity']}\n"
            f"*gold_rank trước→sau:* {ev['gold_rank'][0]}→{ev['gold_rank'][1]}\n"
            f"*Model:* {ev.get('model')}  *degraded:* {ev.get('degraded')}\n"
            f"*Chấm:* {{code}}{ev.get('score')}{{code}}\n*Câu trả lời:*\n{{quote}}{ev.get('answer')}{{quote}}\n"
            f"*Tái hiện:* {{code}}{defect['repro']}{{code}}")
    payload = {"fields": {
        "project": {"key": project_key or s.jira_project_key or "QA"},
        "issuetype": {"name": "Bug"},
        "summary": f"[{defect['severity']}] {defect['category']}: {defect['id']} — {defect['question'][:120]}",
        "description": desc,
        "priority": {"name": _PRIORITY[defect["severity"]]},
        "labels": ["ai-eval", f"eval-{defect['category']}"],
    }}
    live = bool(s.jira_base_url and s.jira_email and s.jira_api_token)
    return _send("POST", f"{s.jira_base_url.rstrip('/')}/rest/api/2/issue", payload, live,
                 auth=(s.jira_email, s.jira_api_token))


def gitlab_mr_note(project: str, mr_iid: int, body: str) -> dict:
    s = get_settings()
    url = f"{s.gitlab_base_url.rstrip('/')}/api/v4/projects/{requests.utils.quote(str(project), safe='')}" \
          f"/merge_requests/{mr_iid}/notes"
    return _send("POST", url, {"body": body}, bool(s.gitlab_token), headers={"PRIVATE-TOKEN": s.gitlab_token})


def github_pr_comment(repo: str, pr: int, body: str) -> dict:
    s = get_settings()
    return _send("POST", f"https://api.github.com/repos/{repo}/issues/{pr}/comments", {"body": body},
                 bool(s.github_token), headers={"Authorization": f"Bearer {s.github_token}",
                                                "Accept": "application/vnd.github+json"})


def testrail_add_case(section_id: int, row: dict) -> dict:
    """row: một phần tử của testcases.approved_cases(). Case chưa duyệt bị từ chối ở đây."""
    if row.get("status") != "approved":
        raise ValueError(f"{row.get('id')}: chỉ đẩy case đã được người duyệt approve")
    s = get_settings()
    c = row["case"]
    payload = {
        "title": c["title"], "priority_id": _TR_PRIORITY.get(c["priority"], 2),
        "refs": row["requirement_id"], "custom_preconds": c.get("preconditions", ""),
        "custom_steps_separated": [{"content": st, "expected": ""} for st in c["steps"][:-1]]
                                  + [{"content": c["steps"][-1], "expected": c["expected"]}],
    }
    live = bool(s.testrail_base_url and s.testrail_user and s.testrail_api_key)
    return _send("POST", f"{s.testrail_base_url.rstrip('/')}/index.php?/api/v2/add_case/{section_id}",
                 payload, live, auth=(s.testrail_user, s.testrail_api_key))
