"""src/qa + src/mcp_server: regression scope, defect/scorecard, test-case HITL, integrations dry-run."""
import json

import pytest

from src.qa import integrations, release, testcases
from src.qa.regression_scope import regression_scope


# ── Regression scope ──────────────────────────────────────────────────────────

def test_scope_follows_imports_and_data_references():
    s = regression_scope(["src/rag/compliance.py"])
    assert "tests/test_compliance.py" in s["tests"]
    # file dữ liệu không import được: bắt qua tên file được nhắc trong code
    assert "tests/test_compliance.py" in regression_scope(["data/compliance/criteria.json"])["tests"]


def test_scope_maps_corpus_doc_to_gold_and_reingest():
    s = regression_scope(["data/raw/lao_dong/74-2024-ND-CP.md"])
    ids = s["gold_ids"]["data/eval/legal_qa_200.json"]
    assert "ld_t01a" in ids and s["reingest"]
    assert any(e["name"] == "retrieval" for e in s["evals"])
    assert any("--ids" in e["cmd"] and "ld_t01a" in e["cmd"] for e in s["evals"])


def test_scope_prompt_change_requires_e2e_and_conftest_runs_all():
    assert any(e["name"] == "e2e" for e in regression_scope(["src/rag/generator.py"])["evals"])
    s = regression_scope(["tests/conftest.py"])
    assert s["run_all_tests"] and s["pytest_cmd"] == "pytest -q"
    assert regression_scope(["README.md"])["evals"] == []


# ── Defects + scorecard ───────────────────────────────────────────────────────

def _row(qid, correct, gold_rank=1, behavior="answer", degraded=(), **score):
    return {"id": qid, "question": f"q {qid}", "as_of_date": "2026-01-01", "expected_behavior": behavior,
            "degraded": list(degraded),
            "arms": {"pre_filter": {"context": {"gold_rank": gold_rank, "gold": gold_rank is not None},
                                    "answer": "a", "model": "m",
                                    "score": {"correct": correct, "forbidden_present": [], "wrong_verdict": False,
                                              **score}}}}


def _report(rows, acc=0.98, n_degraded=0):
    m = {"recall_at_1": 0.8, "recall_at_5": 0.93, "recall_at_8": 0.98, "mrr": 0.86, "context_clean": 0.98,
         "answer_accuracy": acc}
    return {"meta": {"gold_set": "g.json", "n_questions": len(rows), "n_degraded": n_degraded, "index": "i",
                     "git_sha": "s"}, "summary": {"overall": {"pre_filter": m}}, "rows": rows}


def test_defects_classified_by_failing_layer():
    base = _report([_row(i, True) for i in ("a", "b", "c", "d", "e", "f")])
    new = _report([
        _row("a", False, behavior="refuse"),                 # trả lời câu phải từ chối
        _row("b", False, forbidden_present=["5.310.000"]),   # số của văn bản đã hết hiệu lực
        _row("c", False, gold_rank=1),                       # gold có trong ngữ cảnh mà vẫn sai
        _row("d", False, gold_rank=None),                    # gold rơi khỏi top-k
        _row("e", False, degraded=["dense_skipped"]),        # hạ tầng
        _row("f", True),
    ])
    d = release.analyze_defects(new, base)
    cats = {x["id"]: (x["category"], x["severity"]) for x in d["defects"]}
    assert cats == {"a": ("oos_leak", "critical"), "b": ("stale_law", "critical"), "c": ("generation", "major"),
                    "d": ("retrieval", "minor"), "e": ("infra", "info")}
    assert d["defects"][0]["severity"] == "critical"
    assert "--ids a" in next(x for x in d["defects"] if x["id"] == "a")["repro"]


def test_scorecard_blocks_on_critical_degraded_and_accuracy():
    base = _report([_row("a", True), _row("b", True)])
    assert release.release_scorecard(_report([_row("a", True), _row("b", True)]), base)["decision"] == "GO"
    warn = release.release_scorecard(_report([_row("a", True), _row("b", False)]), base)
    assert warn["decision"] == "GO" and warn["warnings"]  # major = cảnh báo, người duyệt quyết
    crit = release.release_scorecard(_report([_row("a", True), _row("b", False, behavior="refuse")]), base)
    assert crit["decision"] == "NO-GO" and any("critical" in b for b in crit["blockers"])
    assert release.release_scorecard(_report([_row("a", True)], n_degraded=1), base)["decision"] == "NO-GO"
    assert release.release_scorecard(_report([_row("a", True)], acc=0.9), base)["decision"] == "NO-GO"
    assert release.release_scorecard(base, base, tests_passed=False)["decision"] == "NO-GO"
    assert "NO-GO" in release.to_markdown(crit)


# ── Test-case drafting + HITL ─────────────────────────────────────────────────

REQ = "Thời gian thử việc không quá 60 ngày đối với công việc cần trình độ cao đẳng trở lên."


def _case(**kw):
    c = {"title": "t", "type": "boundary", "priority": "P1", "steps": ["nhập 60 ngày"],
         "expected": "chấp nhận 60 ngày", "source_quote": "không quá 60 ngày"}
    return {**c, **kw}


def test_validation_flags_hallucinated_quote_and_numbers():
    assert testcases.validate_test_cases([_case()], REQ) == []
    probs = [i["problem"] for i in testcases.validate_test_cases(
        [_case(title="a", source_quote="không quá 90 ngày"), _case(title="b", expected="từ chối 61 ngày, mã E42"),
         _case(title="c", type="smoke"), {"title": "d"}], REQ)]
    assert any("source_quote" in p for p in probs)
    assert any("61" in p and "42" in p for p in probs)
    assert any("type" in p for p in probs) and any("thiếu trường" in p for p in probs)


def test_draft_park_review_flow(monkeypatch):
    out = {"ambiguities": ["chưa nói ngày lịch hay ngày làm việc"],
           "cases": [_case(title="ok"), _case(title="bịa", expected="từ chối 61 ngày")]}
    monkeypatch.setattr("src.rag.generator.gemini_generate", lambda *a, **k: "```json\n" + json.dumps(out) + "\n```")
    draft = testcases.draft_test_cases(REQ)
    assert draft["ambiguities"] and len(draft["issues"]) == 1
    ids = testcases.park_for_review(draft, "REQ-1")
    assert testcases.park_for_review(draft, "REQ-1") == ids  # nạp lại không nhân bản
    assert len(testcases.pending_cases()) == 2 and testcases.approved_cases() == []

    with pytest.raises(ValueError):
        testcases.review_case(ids[1], "approve", "lan")            # có issues mà không ghi note
    with pytest.raises(ValueError):
        testcases.review_case(ids[0], "approve", " ")              # thiếu người duyệt
    testcases.review_case(ids[0], "approve", "lan")
    testcases.review_case(ids[1], "reject", "lan", "số 61 bịa")
    assert [r["id"] for r in testcases.approved_cases()] == [ids[0]]
    assert testcases.pending_cases() == []


def test_draft_survives_non_json(monkeypatch):
    monkeypatch.setattr("src.rag.generator.gemini_generate", lambda *a, **k: "xin lỗi")
    assert testcases.draft_test_cases(REQ)["cases"] == []


# ── Integrations: dry-run khi thiếu credential ────────────────────────────────

def test_integrations_dry_run_payloads(monkeypatch):
    monkeypatch.setattr(integrations.requests, "request", lambda *a, **k: pytest.fail("không được gọi mạng"))
    base = _report([_row("a", True)])
    defect = release.analyze_defects(_report([_row("a", False, behavior="refuse")]), base)["defects"][0]
    j = integrations.jira_create_defect(defect, "QA")
    assert j["dry_run"] and j["json"]["fields"]["priority"]["name"] == "Highest"
    assert "--ids a" in j["json"]["fields"]["description"]
    assert integrations.github_pr_comment("o/r", 3, "x")["url"].endswith("/repos/o/r/issues/3/comments")
    assert "group%2Fproj" in integrations.gitlab_mr_note("group/proj", 7, "x")["url"]

    row = {"id": "tc_1", "requirement_id": "REQ-1", "status": "pending", "case": _case()}
    with pytest.raises(ValueError):
        integrations.testrail_add_case(5, row)
    tr = integrations.testrail_add_case(5, {**row, "status": "approved"})
    assert tr["json"]["custom_steps_separated"][-1]["expected"] == "chấp nhận 60 ngày"


# ── MCP server ────────────────────────────────────────────────────────────────

async def test_mcp_server_exposes_tools_and_guards_paths():
    from src import mcp_server

    names = {t.name for t in await mcp_server.mcp.list_tools()}
    assert {"ask_labor_law", "check_labor_compliance", "get_regression_scope", "analyze_defects",
            "release_scorecard", "draft_test_cases", "review_test_case", "create_jira_defect",
            "post_scorecard_comment", "push_test_case_to_testrail"} <= names
    for bad in ("../secrets.json", ".env", "C:/Windows/win.ini"):
        with pytest.raises(ValueError):
            mcp_server._report(bad)
