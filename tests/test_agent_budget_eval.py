"""eval/agent_budget_eval.py: so sánh A/B theo từng câu, không chỉ theo trung bình."""

from eval.agent_budget_eval import compare


def _report(correct: dict[str, bool], llm_mean: float) -> dict:
    return {
        "rows": [{"id": i, "correct": c} for i, c in correct.items()],
        "summary": {"accuracy": sum(correct.values()) / len(correct), "llm_calls_mean": llm_mean,
                    "citation_groundedness": None},
    }


def test_compare_reports_flips_and_deltas():
    before = _report({"a": True, "b": False, "c": True}, 3.5)
    after = _report({"a": False, "b": True, "c": True}, 1.2)
    out = compare(before, after)
    assert out["broke"] == ["a"]
    assert out["fixed"] == ["b"]
    assert out["delta"]["accuracy"] == 0.0          # trung bình đứng yên dù có 2 câu đổi
    assert out["delta"]["llm_calls_mean"] == -2.3
    assert out["delta"]["citation_groundedness"] is None  # thiếu số thì không bịa hiệu số
