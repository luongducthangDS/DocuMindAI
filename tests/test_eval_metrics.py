"""Tests for eval/metrics.py — the clause-level retrieval metrics (DEC-0004).

The central claim these tests defend: a chunk that carries the right article in
a superseded version is NOT a hit. That is the case the old token-F1 threshold
scored as a hit, and the case `do_temporal_filter` exists to prevent.
"""

from __future__ import annotations

import pytest

from eval.metrics import (
    citation_groundedness,
    citation_validity,
    compute_all,
    gold_clause_uids,
    lexical_overlap,
    mrr_at_k,
    recall_at_k,
)


def rec(clause_uid: str = "", version_id: str = "", doc_id: str = "", text: str = "") -> dict:
    return {"clause_uid": clause_uid, "version_id": version_id, "doc_id": doc_id, "text": text}


GOLD_ITEM = {
    "id": "q1",
    "question": "Mức lương tối thiểu vùng I?",
    "gold_clause_uids": ["74-2024-ND-CP__d3"],
    "distractor_versions": ["74-2024-ND-CP__d3__v2026-01-01"],
}

GOLD_REC = rec("74-2024-ND-CP__d3", "74-2024-ND-CP__d3__v2024-07-01", "74-2024-ND-CP")
SUPERSEDED_REC = rec("74-2024-ND-CP__d3", "74-2024-ND-CP__d3__v2026-01-01", "74-2024-ND-CP")
OTHER_REC = rec("45-2019-QH14__d98", "45-2019-QH14__d98__v2021-01-01", "45-2019-QH14")


class TestGoldLookup:
    def test_prefers_gold_clause_uids(self):
        item = {"gold_clause_uids": ["a", "b"], "source_clause": "c"}
        assert gold_clause_uids(item) == ["a", "b"]

    def test_falls_back_to_source_clause(self):
        assert gold_clause_uids({"source_clause": "c"}) == ["c"]

    def test_no_gold_returns_empty(self):
        assert gold_clause_uids({"question": "x"}) == []


class TestRecallAtK:
    def test_gold_at_rank_one(self):
        assert recall_at_k([GOLD_ITEM], [[GOLD_REC, OTHER_REC]], k=8) == 1.0

    def test_gold_outside_k_is_a_miss(self):
        records = [OTHER_REC] * 8 + [GOLD_REC]
        assert recall_at_k([GOLD_ITEM], [records], k=8) == 0.0
        assert recall_at_k([GOLD_ITEM], [records], k=20) == 1.0

    def test_gold_absent(self):
        assert recall_at_k([GOLD_ITEM], [[OTHER_REC]], k=8) == 0.0

    def test_superseded_version_is_not_a_hit(self):
        """The case the old token-F1 threshold got wrong.

        The chunk is the right article and reads almost identically to gold —
        token overlap would call it a hit — but it is the version that is no
        longer in force, so retrieval did not do its job.
        """
        assert recall_at_k([GOLD_ITEM], [[SUPERSEDED_REC]], k=8) == 0.0

    def test_unlisted_version_still_counts(self):
        """Only versions named as distractors are rejected — a corpus without
        version bookkeeping must not be scored as if everything were wrong."""
        item = {"gold_clause_uids": ["74-2024-ND-CP__d3"]}
        assert recall_at_k([item], [[SUPERSEDED_REC]], k=8) == 1.0

    def test_questions_without_gold_are_skipped_not_guessed(self):
        no_gold = {"question": "x", "ground_truth": "y"}
        assert recall_at_k([no_gold], [[GOLD_REC]], k=8) is None

    def test_only_scoreable_questions_form_the_denominator(self):
        items = [GOLD_ITEM, {"question": "x"}]
        retrieved = [[GOLD_REC], [OTHER_REC]]
        assert recall_at_k(items, retrieved, k=8) == 1.0

    def test_mixed_hit_and_miss(self):
        items = [GOLD_ITEM, GOLD_ITEM]
        assert recall_at_k(items, [[GOLD_REC], [OTHER_REC]], k=8) == 0.5


class TestMrrAtK:
    def test_rank_one(self):
        assert mrr_at_k([GOLD_ITEM], [[GOLD_REC]], k=8) == 1.0

    def test_rank_three(self):
        records = [OTHER_REC, OTHER_REC, GOLD_REC]
        assert mrr_at_k([GOLD_ITEM], [records], k=8) == pytest.approx(0.3333, abs=1e-4)

    def test_absent_scores_zero(self):
        assert mrr_at_k([GOLD_ITEM], [[OTHER_REC]], k=8) == 0.0

    def test_superseded_version_does_not_take_the_rank(self):
        records = [SUPERSEDED_REC, GOLD_REC]
        assert mrr_at_k([GOLD_ITEM], [records], k=8) == 0.5

    def test_no_gold_returns_none(self):
        assert mrr_at_k([{"question": "x"}], [[GOLD_REC]], k=8) is None


class TestCitationValidity:
    def test_all_indices_exist(self):
        records = [OTHER_REC] * 8
        assert citation_validity(["Theo [1] và [3]."], [records]) == 1.0

    def test_index_beyond_chunk_count(self):
        records = [OTHER_REC] * 8
        assert citation_validity(["Theo [9]."], [records]) == 0.0

    def test_combined_bracket_form(self):
        records = [OTHER_REC] * 2
        assert citation_validity(["Theo [1, 2]."], [records]) == 1.0
        assert citation_validity(["Theo [1, 5]."], [records]) == 0.0

    def test_uncited_answers_are_not_counted(self):
        assert citation_validity(["Tôi không tìm thấy quy định này."], [[OTHER_REC]]) is None


class TestCitationGroundedness:
    def test_citation_points_at_gold(self):
        records = [OTHER_REC, GOLD_REC]
        assert citation_groundedness([GOLD_ITEM], ["Theo [2]."], [records]) == 1.0

    def test_citation_points_outside_gold(self):
        records = [OTHER_REC, GOLD_REC]
        assert citation_groundedness([GOLD_ITEM], ["Theo [1]."], [records]) == 0.0

    def test_citing_a_superseded_version_is_not_grounded(self):
        records = [SUPERSEDED_REC]
        assert citation_groundedness([GOLD_ITEM], ["Theo [1]."], [records]) == 0.0

    def test_one_good_citation_is_enough(self):
        """A correct answer may also cite supporting context outside the gold
        set; requiring every citation to be gold would measure style."""
        records = [OTHER_REC, GOLD_REC]
        assert citation_groundedness([GOLD_ITEM], ["Theo [1] và [2]."], [records]) == 1.0

    def test_out_of_range_index_ignored(self):
        records = [GOLD_REC]
        assert citation_groundedness([GOLD_ITEM], ["Theo [7]."], [records]) == 0.0

    def test_no_gold_returns_none(self):
        assert citation_groundedness([{"question": "x"}], ["Theo [1]."], [[GOLD_REC]]) is None


class TestComputeAll:
    def test_reports_coverage_so_a_silent_skip_is_impossible(self):
        items = [GOLD_ITEM, {"question": "x", "ground_truth": "y"}]
        result = compute_all(
            test_items=items,
            answers=["Theo [1].", "..."],
            retrieved_list=[[GOLD_REC], [OTHER_REC]],
            ground_truths=["gt", "y"],
        )
        assert result["coverage"] == {"n_questions": 2, "n_scored": 1, "n_skipped": 1}

    def test_retrieval_block_has_every_k(self):
        result = compute_all(
            test_items=[GOLD_ITEM],
            answers=["Theo [1]."],
            retrieved_list=[[GOLD_REC]],
            ground_truths=["gt"],
        )
        expected = {"recall@1", "recall@5", "recall@8", "recall@20", "mrr@8"}
        assert set(result["retrieval"]) == expected
        assert result["grounding"]["citation_groundedness"] == 1.0

    def test_generation_block_uses_lexical_label(self):
        result = compute_all(
            test_items=[GOLD_ITEM],
            answers=["Mức lương tối thiểu vùng I là 4.960.000 đồng."],
            retrieved_list=[[GOLD_REC]],
            ground_truths=["Mức lương tối thiểu vùng I là 4.960.000 đồng."],
        )
        assert result["generation"]["answer_correctness_lexical"] == 1.0


class TestLexicalOverlapIsAProxy:
    def test_two_versions_of_one_clause_look_almost_identical(self):
        """Why the retrieval metrics cannot be built on this function: two
        versions of the same article differ by one figure and still score far
        above the 0.15 threshold the old hit_rate used."""
        v2024 = "Mức lương tối thiểu tháng của vùng I là 4.960.000 đồng một tháng"
        v2026 = "Mức lương tối thiểu tháng của vùng I là 5.310.000 đồng một tháng"
        assert lexical_overlap(v2024, v2026) > 0.8
