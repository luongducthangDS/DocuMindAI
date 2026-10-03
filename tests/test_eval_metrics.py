"""recall@k / MRR / p95 in eval/temporal_eval.py — the numbers the README quotes."""
from types import SimpleNamespace

from eval.temporal_eval import _percentile, gold_rank, summarise


def _chunk(uid, version=""):
    return SimpleNamespace(metadata={"clause_uid": uid, "version_id": version, "doc_id": uid.split("__")[0]})


def test_gold_rank_single_clause():
    chunks = [_chunk("a__d1"), _chunk("a__d2"), _chunk("a__d3")]
    assert gold_rank(chunks, {"source_clause": "a__d2"}) == 2
    assert gold_rank(chunks, {"source_clause": "a__d9"}) is None
    assert gold_rank(chunks, {"source_clause": ""}) is None


def test_gold_rank_multi_clause_takes_worst():
    chunks = [_chunk("a__d1"), _chunk("a__d2"), _chunk("a__d3")]
    q = {"source_clause": "", "source_clauses": ["a__d3", "a__d1"]}
    assert gold_rank(chunks, q) == 3


def test_gold_rank_ignores_superseded_version():
    chunks = [_chunk("a__d1", "a__d1__v2016"), _chunk("a__d1", "a__d1__v2021")]
    q = {"source_clause": "a__d1", "distractor_versions": ["a__d1__v2016"]}
    assert gold_rank(chunks, q) == 2


def test_percentile_nearest_rank():
    assert _percentile([], 95) is None
    assert _percentile([5], 95) == 5
    assert _percentile(list(range(1, 101)), 95) == 95
    assert _percentile(list(range(1, 101)), 50) == 50


def test_summarise_recall_excludes_questions_without_gold():
    def row(rank, has_gold):
        ctx = {"gold": bool(rank), "distractor": False, "clean": bool(rank), "n_chunks": 8, "gold_rank": rank}
        return {"temporal_kind": "x", "has_gold": has_gold, "retrieve_ms": 100, "arms": {"a": {"context": ctx}}}

    rows = [row(1, True), row(4, True), row(None, True), row(None, False)]
    agg = summarise(rows, ("a",), retrieval_only=True)["overall"]["a"]
    assert agg["recall_at_1"] == 1 / 3
    assert agg["recall_at_5"] == 2 / 3
    assert abs(agg["mrr"] - (1 + 0.25) / 3) < 1e-9
    assert agg["retrieve_ms_p95"] == 100


def test_ci_gate_fails_only_beyond_tolerance():
    from eval.ci_gate import gate

    base = {"recall_at_5": 0.80, "mrr": 0.60, "retrieve_ms_p95": 900}
    ok, _ = gate({"recall_at_5": 0.79, "mrr": 0.60, "retrieve_ms_p95": 2000}, base)
    assert ok == []  # latency is reported, not gated; 0.01 drop is within tolerance
    bad, table = gate({"recall_at_5": 0.70, "mrr": 0.60}, base)
    assert len(bad) == 1 and "recall_at_5" in bad[0]
    assert "❌" in table


def test_legal_qa_200_schema():
    """DB-free guard; eval/validate_gold.py checks the same set against the live index."""
    import json
    from pathlib import Path

    from eval.validate_gold import REQUIRED_KEYS

    qs = json.loads((Path(__file__).parents[1] / "data/eval/legal_qa_200.json").read_text(encoding="utf-8"))
    assert len(qs) >= 190
    assert len({q["id"] for q in qs}) == len(qs)
    for q in qs:
        assert REQUIRED_KEYS <= q.keys(), q["id"]
        assert q["expected_behavior"] in {"answer", "refuse", "time_out_of_range"}, q["id"]
        if q["expected_behavior"] != "answer":
            assert not (q["source_clause"] or q.get("source_clauses")), q["id"]


def test_readme_eval_table_quotes_the_report_verbatim():
    """Một phương pháp, một kết quả: README chỉ được trích nguyên số trong report.

    2026-10-03: README ghi accuracy 0.763 → 0.935 (tính tay trên tập con 169 câu) trong
    khi report ghi 0.739 → 0.910 cho đúng lần chạy đó — hai bộ số trôi nổi cho một phép đo.
    """
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    summary = json.loads((root / "reports/eval_e2e.json").read_text(encoding="utf-8"))["summary"]
    readme_rows = [l for l in (root / "README.md").read_text(encoding="utf-8").splitlines()
                   if l.startswith("| ")]

    f3 = "{:.3f}".format
    expected = {  # nhãn đầu dòng README → cách định dạng số của một arm
        "Recall@1": lambda m: [f3(m["recall_at_1"])],
        "Recall@5": lambda m: [f3(m["recall_at_5"])],
        "MRR": lambda m: [f3(m["mrr"])],
        "Ngữ cảnh lẫn bản luật hết hiệu lực": lambda m: [f"{m['context_distractor'] * 100:.1f}%"],
        "Answer accuracy": lambda m: [f3(m["answer_accuracy"])],
        "Faithfulness": lambda m: [f3(m["faithfulness"]), str(m["faithfulness_n"])],
        "Latency truy hồi p95": lambda m: [f"{m['retrieve_ms_p95']:.0f} ms"],
        "Latency end-to-end": lambda m: [f"{m['e2e_ms_p50'] / 1000:.1f} s / {m['e2e_ms_p95'] / 1000:.1f} s"],
    }
    for label, fmt in expected.items():
        line = next(l for l in readme_rows if l.startswith(f"| {label}"))
        for arm in ("no_temporal", "pre_filter"):
            for value in fmt(summary["overall"][arm]):
                assert value in line, f"README '{label}' thiếu {value} ({arm})"

    doc_version = next(l for l in readme_rows if "đổi phiên bản văn bản" in l)
    for arm in ("no_temporal", "pre_filter"):
        m = summary["by_kind"]["doc_version"][arm]
        assert f"{m['answer_accuracy']:.2f}" in doc_version and f"n={m['n']}" in doc_version
