"""
Structural tests for the hard gold set (`data/eval/hard_questions.json`).

The temporal set measures one skill — picking the version in force — and its
30 questions now saturate (answer_accuracy 1.000 on both filtering arms). This
set exists to measure what that one does not: combining clauses, exact
transition days, refusing out-of-scope questions, rejecting false premises and
reaching a verdict on a concrete situation.

Every question is drafted from the verbatim corpus text, never from the system's
own output, and carries `reviewed: false` until a human checks it.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pytest

from tests.test_eval_goldset import _index_clause_uids

HARD_PATH = Path(__file__).resolve().parents[1] / "data" / "eval" / "hard_questions.json"

REQUIRED_FIELDS = {
    "id", "pair_id", "question", "as_of_date", "category", "temporal_kind",
    "expected_behavior", "ground_truth", "source_doc", "source_clause",
    "expect_contains", "expect_absent", "distractor_docs", "drafted_by", "reviewed",
}
VALID_KINDS = {"multi_clause", "table_lookup", "boundary", "out_of_scope", "false_premise", "compliance"}
VALID_BEHAVIOURS = {"answer", "refuse"}
_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@pytest.fixture(scope="module")
def hard() -> list[dict]:
    return json.loads(HARD_PATH.read_text(encoding="utf-8"))


class TestHardSetShape:
    def test_ids_unique(self, hard):
        ids = [q["id"] for q in hard]
        assert len(ids) == len(set(ids))

    def test_required_fields(self, hard):
        for q in hard:
            assert not REQUIRED_FIELDS - set(q), (q["id"], REQUIRED_FIELDS - set(q))

    def test_kinds_and_behaviours_known(self, hard):
        for q in hard:
            assert q["temporal_kind"] in VALID_KINDS, q["id"]
            assert q["expected_behavior"] in VALID_BEHAVIOURS, q["id"]

    def test_every_kind_is_represented(self, hard):
        assert {q["temporal_kind"] for q in hard} == VALID_KINDS

    def test_dates_parse(self, hard):
        for q in hard:
            assert _ISO.match(q["as_of_date"]), q["id"]
            date.fromisoformat(q["as_of_date"])

    def test_expected_and_forbidden_do_not_overlap(self, hard):
        for q in hard:
            required = {alt for group in q["expect_contains"] for alt in group}
            assert not (required & set(q["expect_absent"])), q["id"]

    def test_every_expect_group_is_non_empty(self, hard):
        for q in hard:
            assert q["expect_contains"], q["id"]
            assert all(group for group in q["expect_contains"]), q["id"]

    def test_multi_clause_names_at_least_two_clauses(self, hard):
        for q in hard:
            if q["temporal_kind"] == "multi_clause":
                assert len(q.get("source_clauses", [])) >= 2, q["id"]

    def test_refusals_have_no_source_and_expect_the_refusal_phrase(self, hard):
        for q in hard:
            if q["expected_behavior"] == "refuse":
                assert not q["source_doc"] and not q["source_clause"], q["id"]
                assert ["không tìm thấy"] in q["expect_contains"], q["id"]

    def test_boundary_pairs_straddle_one_day(self, hard):
        """A boundary pair is only a boundary test if its two dates are adjacent."""
        pairs: dict[str, list[date]] = {}
        for q in hard:
            if q["temporal_kind"] == "boundary":
                pairs.setdefault(q["pair_id"], []).append(date.fromisoformat(q["as_of_date"]))
        assert pairs
        for pid, dates in pairs.items():
            assert len(dates) == 2, pid
            assert abs((dates[1] - dates[0]).days) == 1, pid


class TestHardSetMatchesIndex:
    """Skipped when the index has not been built."""

    def test_every_named_clause_exists(self, hard):
        uids = _index_clause_uids()
        if not uids:
            pytest.skip("index chưa build — bỏ qua kiểm clause_uid")
        for q in hard:
            named = ([q["source_clause"]] if q["source_clause"] else []) + q.get("source_clauses", [])
            for uid in named:
                assert uid in uids, f"{q['id']}: clause_uid không có trong index — {uid}"
