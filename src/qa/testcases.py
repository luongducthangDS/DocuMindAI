"""Phân tích yêu cầu → nháp test case (LLM) → kiểm hallucination (tất định) → hàng đợi người duyệt.

LLM chỉ được NHÁP. Không test case nào rời hàng đợi khi chưa có người `approve`:
  draft_test_cases(req)  → {"ambiguities": [...], "cases": [...], "issues": [...]}
  park_for_review(...)   → ghi data/qa/review_queue.jsonl, status=pending
  review_case(id, ...)   → approve / reject kèm tên người duyệt + ghi chú
  approved_cases()       → chỉ case đã duyệt (đầu vào cho TestRail/pytest)

Kiểm hallucination không dùng LLM-judge (judge cho 1.0 cả câu sai — xem CLAUDE.md):
mỗi case phải trích NGUYÊN VĂN câu yêu cầu nó kiểm (`source_quote`), và mọi con số trong
`expected` phải có trong yêu cầu. Case vi phạm vẫn vào hàng đợi nhưng mang `issues`.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from src.config import get_settings

TYPES = {"positive", "negative", "boundary", "regression", "security"}
PRIORITIES = {"P0", "P1", "P2"}
_REQUIRED = ("title", "type", "priority", "steps", "expected", "source_quote")
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")

_PROMPT = """Bạn là QA engineer. Đọc YÊU CẦU dưới đây và trả về DUY NHẤT một JSON object:
{{"ambiguities": ["điểm mơ hồ / thiếu tiêu chí chấp nhận, mỗi ý một câu"],
  "cases": [{{"title": "...", "type": "positive|negative|boundary|regression|security",
             "priority": "P0|P1|P2", "preconditions": "...", "steps": ["..."],
             "expected": "kết quả quan sát được, cụ thể", "source_quote": "câu NGUYÊN VĂN trong yêu cầu mà case này kiểm"}}]}}
Quy tắc: tối đa {n} case; có ít nhất một case negative và một case boundary nếu yêu cầu có ngưỡng;
KHÔNG bịa con số, ngưỡng, mã lỗi không có trong yêu cầu — thiếu thì ghi vào "ambiguities".
{context}
YÊU CẦU:
{requirement}"""


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _parse(raw: str) -> dict | None:
    m = re.search(r"\{.*\}", raw or "", re.S)
    try:
        data = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def validate_test_cases(cases: list[dict], requirement: str) -> list[dict]:
    """Trả về các vấn đề [{case, problem}] — rỗng = sạch."""
    req, issues, titles = _norm(requirement), [], set()
    req_nums = set(_NUM_RE.findall(requirement))
    for i, c in enumerate(cases):
        tag = c.get("title") or f"#{i}"
        if missing := [f for f in _REQUIRED if not c.get(f)]:
            issues.append({"case": tag, "problem": f"thiếu trường {', '.join(missing)}"})
            continue
        if c["type"] not in TYPES:
            issues.append({"case": tag, "problem": f"type '{c['type']}' không hợp lệ"})
        if c["priority"] not in PRIORITIES:
            issues.append({"case": tag, "problem": f"priority '{c['priority']}' không hợp lệ"})
        if _norm(c["source_quote"]) not in req:
            issues.append({"case": tag, "problem": "hallucination: source_quote không có nguyên văn trong yêu cầu"})
        if extra := sorted(set(_NUM_RE.findall(c["expected"])) - req_nums):
            issues.append({"case": tag, "problem": f"hallucination: số {', '.join(extra)} trong expected không có trong yêu cầu"})
        if _norm(c["title"]) in titles:
            issues.append({"case": tag, "problem": "trùng tiêu đề"})
        titles.add(_norm(c["title"]))
    return issues


def draft_test_cases(requirement: str, n: int = 8, context: str = "") -> dict:
    from src.rag.generator import gemini_generate

    raw = gemini_generate(
        _PROMPT.format(n=n, requirement=requirement,
                       context=f"NGỮ CẢNH (quy định/tài liệu liên quan):\n{context}\n" if context else ""),
        name="qa-draft-test-cases",
    )
    data = _parse(raw)
    if data is None:
        return {"ambiguities": [], "cases": [], "issues": [{"case": "-", "problem": "LLM không trả JSON hợp lệ"}]}
    cases = [c for c in data.get("cases", []) if isinstance(c, dict)][:n]
    return {"ambiguities": data.get("ambiguities", []), "cases": cases,
            "issues": validate_test_cases(cases, requirement)}


# ── Human-in-the-loop queue ────────────────────────────────────────────────────

def _queue() -> Path:
    return Path(get_settings().data_dir) / "qa" / "review_queue.jsonl"


def _load() -> list[dict]:
    p = _queue()
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def _save(rows: list[dict]) -> None:
    p = _queue()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def park_for_review(draft: dict, requirement_id: str) -> list[str]:
    """Đưa case nháp vào hàng đợi; id = hash(requirement_id + title) nên nạp lại không nhân bản."""
    rows, ids = _load(), []
    known = {r["id"] for r in rows}
    issues_by_case: dict[str, list[str]] = {}
    for iss in draft.get("issues", []):
        issues_by_case.setdefault(iss["case"], []).append(iss["problem"])
    for c in draft["cases"]:
        cid = "tc_" + hashlib.sha1(f"{requirement_id}|{c.get('title')}".encode()).hexdigest()[:10]
        ids.append(cid)
        if cid not in known:
            rows.append({"id": cid, "requirement_id": requirement_id, "case": c,
                         "issues": issues_by_case.get(c.get("title"), []),
                         "status": "pending", "drafted_by": "llm", "reviewed_by": None, "note": ""})
    _save(rows)
    return ids


def review_case(case_id: str, decision: str, reviewer: str, note: str = "") -> dict:
    if decision not in ("approve", "reject"):
        raise ValueError("decision phải là approve hoặc reject")
    if not reviewer.strip():
        raise ValueError("cần tên người duyệt")
    rows = _load()
    for r in rows:
        if r["id"] == case_id:
            if decision == "approve" and r["issues"] and not note.strip():
                raise ValueError("case có issues — duyệt thì phải ghi note giải thích vì sao vẫn nhận")
            r.update(status="approved" if decision == "approve" else "rejected", reviewed_by=reviewer,
                     note=note, reviewed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
            _save(rows)
            return r
    raise KeyError(case_id)


def pending_cases() -> list[dict]:
    return [r for r in _load() if r["status"] == "pending"]


def approved_cases() -> list[dict]:
    return [r for r in _load() if r["status"] == "approved"]
