"""
Structural tests for the temporal gold set (`data/eval/temporal_questions.json`).

The gold set is written by hand from the full text in `data/raw/lao_dong/` and
committed *before* the system is ever run against it (SPEC-eval-goldset). These
tests only check shape and that every `source_clause` really exists in the
index — they never assert on model output.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

import pytest

GOLD_PATH = Path(__file__).resolve().parents[1] / "data" / "eval" / "temporal_questions.json"

REQUIRED_FIELDS = {
    "id",
    "pair_id",
    "question",
    "as_of_date",
    "category",
    "temporal_kind",
    "expected_behavior",
    "ground_truth",
    "source_doc",
    "source_clause",
    "expect_contains",
    "expect_absent",
    "distractor_docs",
}

VALID_KINDS = {"doc_version", "clause_version", "control", "out_of_range"}
VALID_BEHAVIOURS = {"answer", "time_out_of_range"}
_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Banking vocabulary from the pre-pivot corpus — must never reappear here.
_BANKING_TERMS = ("ngân hàng", "NHNN", "tín chấp", "biểu phí", "thẻ tín dụng", "DTI")


@pytest.fixture(scope="module")
def gold() -> list[dict]:
    return json.loads(GOLD_PATH.read_text(encoding="utf-8"))


class TestGoldSetShape:
    def test_has_about_thirty_questions(self, gold):
        assert len(gold) >= 30

    def test_ids_unique(self, gold):
        ids = [q["id"] for q in gold]
        assert len(set(ids)) == len(ids)

    def test_every_question_has_required_fields(self, gold):
        for q in gold:
            missing = REQUIRED_FIELDS - set(q)
            assert not missing, f"{q.get('id')} thiếu field: {sorted(missing)}"

    def test_as_of_date_is_iso_and_parsable(self, gold):
        for q in gold:
            assert _ISO.match(q["as_of_date"]), f"{q['id']}: as_of_date sai định dạng"
            date.fromisoformat(q["as_of_date"])

    def test_kinds_and_behaviours_are_known(self, gold):
        for q in gold:
            assert q["temporal_kind"] in VALID_KINDS, q["id"]
            assert q["expected_behavior"] in VALID_BEHAVIOURS, q["id"]

    def test_expect_contains_is_list_of_alternative_groups(self, gold):
        for q in gold:
            groups = q["expect_contains"]
            assert isinstance(groups, list) and groups, q["id"]
            for group in groups:
                assert isinstance(group, list) and group, f"{q['id']}: group rỗng"
                assert all(isinstance(alt, str) and alt.strip() for alt in group)

    def test_expected_and_forbidden_strings_do_not_overlap(self, gold):
        """A string required in the answer can't also be forbidden."""
        for q in gold:
            required = {alt for group in q["expect_contains"] for alt in group}
            assert not (required & set(q["expect_absent"])), q["id"]

    def test_no_banking_vocabulary_left(self, gold):
        blob = json.dumps(gold, ensure_ascii=False).lower()
        for term in _BANKING_TERMS:
            assert term.lower() not in blob, f"còn từ vựng corpus cũ: {term}"


class TestTemporalCoverage:
    def test_every_kind_is_represented(self, gold):
        kinds = Counter(q["temporal_kind"] for q in gold)
        for kind in VALID_KINDS:
            assert kinds[kind] > 0, f"thiếu nhóm {kind}"

    def test_version_pairs_ask_the_same_question_at_two_dates(self, gold):
        """A pair is only a valid A/B probe if the wording is identical."""
        pairs: dict[str, list[dict]] = {}
        for q in gold:
            if q["temporal_kind"] in {"doc_version", "clause_version", "control"}:
                pairs.setdefault(q["pair_id"], []).append(q)
        assert pairs, "không có cặp temporal nào"
        for pair_id, items in pairs.items():
            assert len(items) == 2, f"{pair_id}: cần đúng 2 mốc, có {len(items)}"
            assert items[0]["question"] == items[1]["question"], pair_id
            assert items[0]["as_of_date"] != items[1]["as_of_date"], pair_id

    def test_version_pairs_expect_different_answers(self, gold):
        """doc_version / clause_version pairs must actually diverge."""
        pairs: dict[str, list[dict]] = {}
        for q in gold:
            if q["temporal_kind"] in {"doc_version", "clause_version"}:
                pairs.setdefault(q["pair_id"], []).append(q)
        for pair_id, (a, b) in pairs.items():
            assert a["expect_contains"] != b["expect_contains"], (
                f"{pair_id}: hai mốc kỳ vọng cùng đáp án — không đo được gì"
            )

    def test_control_pairs_expect_the_same_answer(self, gold):
        pairs: dict[str, list[dict]] = {}
        for q in gold:
            if q["temporal_kind"] == "control":
                pairs.setdefault(q["pair_id"], []).append(q)
        for pair_id, (a, b) in pairs.items():
            assert a["expect_contains"] == b["expect_contains"], pair_id

    def test_out_of_range_questions_predate_corpus_coverage(self, gold):
        from src.ingestion.manifest import corpus_earliest_point_in_time

        earliest = corpus_earliest_point_in_time()
        found = False
        for q in gold:
            if q["temporal_kind"] == "out_of_range":
                found = True
                assert q["as_of_date"] < earliest, q["id"]
                assert q["expected_behavior"] == "time_out_of_range", q["id"]
        assert found

    def test_answerable_questions_are_inside_coverage(self, gold):
        from src.ingestion.manifest import corpus_earliest_point_in_time

        earliest = corpus_earliest_point_in_time()
        for q in gold:
            if q["expected_behavior"] == "answer":
                assert q["as_of_date"] >= earliest, q["id"]


def _index_clause_uids() -> set[str] | None:
    """clause_uid values present in the built index, or None when unavailable."""
    try:
        import chromadb
    except ImportError:  # pragma: no cover - chromadb is a hard dep in practice
        return None

    db = Path(__file__).resolve().parents[1] / "data" / "chroma_db"
    if not db.exists():
        return None
    try:
        client = chromadb.PersistentClient(path=str(db))
        collection = client.get_collection("documind_legal")
        metas = collection.get(include=["metadatas"])["metadatas"] or []
    except Exception:
        return None
    return {str(m.get("clause_uid", "")) for m in metas if m.get("clause_uid")}


class TestGoldSetMatchesIndex:
    """Skipped when the index has not been built (CI without `data/chroma_db`)."""

    def test_every_source_clause_exists_in_index(self, gold):
        uids = _index_clause_uids()
        if not uids:
            pytest.skip("index chưa build — bỏ qua kiểm clause_uid")
        for q in gold:
            uid = q["source_clause"]
            if not uid:
                assert q["temporal_kind"] == "out_of_range", q["id"]
                continue
            assert uid in uids, f"{q['id']}: clause_uid không có trong index — {uid}"

    def test_distractor_versions_exist_in_index(self, gold):
        uids = _index_clause_uids()
        if not uids:
            pytest.skip("index chưa build — bỏ qua kiểm version_id")
        # version_id values, not clause_uid — fetch separately.
        import chromadb

        db = Path(__file__).resolve().parents[1] / "data" / "chroma_db"
        client = chromadb.PersistentClient(path=str(db))
        metas = client.get_collection("documind_legal").get(include=["metadatas"])["metadatas"]
        versions = {str(m.get("version_id", "")) for m in metas if m.get("version_id")}
        for q in gold:
            for vid in q.get("distractor_versions", []):
                assert vid in versions, f"{q['id']}: version_id lạ — {vid}"
