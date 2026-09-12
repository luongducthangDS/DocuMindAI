"""
Tests for temporal-aware retrieval (`src/rag/temporal.py`).

The fixtures mirror the two real cases in the corpus:
  - doc-level: minimum wage, NĐ 74/2024 replaced by NĐ 293/2025 on 2026-01-01;
  - clause-level in place: Điều 139 khoản 1 BLLĐ amended by Luật Dân số
    113/2025 with effect from 2026-07-01.
"""

from dataclasses import dataclass, field

import pytest

from src.ingestion.manifest import EFFECTIVE_TO_OPEN, corpus_earliest_point_in_time
from src.rag.temporal import (
    is_in_force,
    is_out_of_range,
    today_iso,
    versions_in_force,
)


@dataclass
class FakeChunk:
    """Stand-in for RetrievedChunk — the filter only reads `.metadata`."""

    text: str = ""
    score: float = 1.0
    metadata: dict = field(default_factory=dict)


def chunk(uid, eff_from, eff_to=EFFECTIVE_TO_OPEN, status="in_force", text=""):
    return FakeChunk(
        text=text or uid,
        metadata={
            "clause_uid": uid,
            "version_id": f"{uid}__v{eff_from}",
            "effective_from": eff_from,
            "effective_to": eff_to,
            "status": status,
        },
    )


# ── is_in_force ───────────────────────────────────────────────────────────────

class TestIsInForce:
    def test_inside_range(self):
        meta = chunk("d1", "2021-01-01", "2026-01-01").metadata
        assert is_in_force(meta, "2024-05-05")

    def test_before_effective_from(self):
        meta = chunk("d1", "2026-01-01").metadata
        assert not is_in_force(meta, "2025-12-31")

    def test_on_effective_from_is_in_force(self):
        meta = chunk("d1", "2026-01-01").metadata
        assert is_in_force(meta, "2026-01-01")

    def test_on_effective_to_is_already_out(self):
        """`effective_to` is exclusive: the successor takes over that same day."""
        meta = chunk("d1", "2024-07-01", "2026-01-01").metadata
        assert not is_in_force(meta, "2026-01-01")

    def test_open_ended_sentinel_never_expires(self):
        meta = chunk("d1", "2021-01-01", EFFECTIVE_TO_OPEN).metadata
        assert is_in_force(meta, "2999-12-31")

    def test_repealed_status_excluded(self):
        meta = chunk("d1", "2015-01-01", EFFECTIVE_TO_OPEN, status="repealed").metadata
        assert not is_in_force(meta, "2024-01-01")

    def test_chunk_without_dates_is_kept(self):
        assert is_in_force({"clause_uid": "x"}, "2024-01-01")


# ── versions_in_force ─────────────────────────────────────────────────────────

class TestVersionsInForce:
    @pytest.fixture
    def luong_toi_thieu(self):
        """Doc-level replacement: two decrees, disjoint ranges."""
        return [
            chunk("74-2024-ND-CP__d3", "2024-07-01", "2026-01-01", text="4.960.000"),
            chunk("293-2025-ND-CP__d3", "2026-01-01", text="5.310.000"),
        ]

    @pytest.fixture
    def dieu_139(self):
        """Clause-level in-place amendment: same clause_uid, two versions."""
        return [
            chunk("45-2019-QH14__d139_k1", "2021-01-01", "2026-07-01", text="06 tháng"),
            chunk("45-2019-QH14__d139_k1", "2026-07-01", text="con thứ hai 07 tháng"),
        ]

    def test_doc_level_earlier_date(self, luong_toi_thieu):
        out = versions_in_force(luong_toi_thieu, "2025-03-01")
        assert [c.text for c in out] == ["4.960.000"]

    def test_doc_level_later_date(self, luong_toi_thieu):
        out = versions_in_force(luong_toi_thieu, "2026-02-01")
        assert [c.text for c in out] == ["5.310.000"]

    def test_clause_level_before_amendment(self, dieu_139):
        out = versions_in_force(dieu_139, "2026-05-01")
        assert [c.text for c in out] == ["06 tháng"]

    def test_clause_level_after_amendment(self, dieu_139):
        out = versions_in_force(dieu_139, "2026-09-10")
        assert [c.text for c in out] == ["con thứ hai 07 tháng"]

    def test_keeps_retrieval_order(self):
        chunks = [
            chunk("b", "2015-01-01", text="second-ranked"),
            chunk("a", "2015-01-01", text="first-ranked"),
        ]
        out = versions_in_force(chunks, "2024-01-01")
        assert [c.text for c in out] == ["second-ranked", "first-ranked"]

    def test_date_outside_every_range_returns_empty(self, luong_toi_thieu):
        assert versions_in_force(luong_toi_thieu, "2020-01-01") == []

    def test_tie_break_keeps_latest_effective_from(self, caplog):
        """Overlapping ranges are a corpus bug: keep the later version, warn."""
        overlapping = [
            chunk("45-2019-QH14__d139_k1", "2021-01-01", text="older"),
            chunk("45-2019-QH14__d139_k1", "2026-07-01", text="newer"),
        ]
        out = versions_in_force(overlapping, "2026-09-10")
        assert [c.text for c in out] == ["newer"]

    def test_duplicate_same_version_is_deduplicated(self):
        dup = [
            chunk("45-2019-QH14__d139_k1", "2021-01-01", text="one"),
            chunk("45-2019-QH14__d139_k1", "2021-01-01", text="one"),
        ]
        assert len(versions_in_force(dup, "2024-01-01")) == 1

    def test_chunks_without_clause_uid_are_not_deduplicated(self):
        no_uid = [
            FakeChunk(text="a", metadata={"effective_from": "2015-01-01"}),
            FakeChunk(text="b", metadata={"effective_from": "2015-01-01"}),
        ]
        assert len(versions_in_force(no_uid, "2024-01-01")) == 2

    def test_empty_as_of_falls_back_to_today(self):
        current = [chunk("x", "2015-01-01")]
        assert versions_in_force(current, "") == current

    def test_accepts_plain_dict_chunks(self):
        dicts = [{"metadata": {"clause_uid": "x", "effective_from": "2026-01-01"}}]
        assert versions_in_force(dicts, "2025-01-01") == []
        assert len(versions_in_force(dicts, "2026-05-01")) == 1


# ── is_out_of_range ───────────────────────────────────────────────────────────

class TestOutOfRange:
    def test_before_corpus_start(self):
        assert is_out_of_range("2010-01-01", "2015-01-01")

    def test_on_corpus_start_is_in_range(self):
        assert not is_out_of_range("2015-01-01", "2015-01-01")

    def test_after_corpus_start(self):
        assert not is_out_of_range("2026-09-12", "2015-01-01")

    def test_missing_values_never_flag(self):
        assert not is_out_of_range("", "2015-01-01")
        assert not is_out_of_range("2010-01-01", "")

    def test_earliest_comes_from_manifest(self):
        earliest = corpus_earliest_point_in_time("docs/corpus/corpus_manifest.yaml")
        assert earliest == "2015-01-01"
        assert is_out_of_range("2014-12-31", earliest)


def test_today_iso_format():
    assert len(today_iso()) == 10 and today_iso().count("-") == 2


# ── graph node + generator wiring (T14) ───────────────────────────────────────

class TestTemporalFilterNode:
    @staticmethod
    def _state(as_of, chunks):
        return {
            "query": "mức lương tối thiểu vùng I",
            "retrieved_chunks": chunks,
            "as_of_date": as_of,
            "steps": [],
        }

    @pytest.fixture
    def two_decrees(self):
        return [
            chunk("74-2024-ND-CP__d3", "2024-07-01", "2026-01-01", text="4.960.000"),
            chunk("293-2025-ND-CP__d3", "2026-01-01", text="5.310.000"),
        ]

    def test_node_keeps_version_in_force(self, two_decrees):
        from src.agent.graph import temporal_filter_node

        out = temporal_filter_node(self._state("2025-03-01", two_decrees))
        assert [c.text for c in out["retrieved_chunks"]] == ["4.960.000"]
        assert out["time_out_of_range"] is False
        assert "2025-03-01" in out["steps"][-1]["detail"]

    def test_node_switches_version_at_later_date(self, two_decrees):
        from src.agent.graph import temporal_filter_node

        out = temporal_filter_node(self._state("2026-02-01", two_decrees))
        assert [c.text for c in out["retrieved_chunks"]] == ["5.310.000"]

    def test_node_drops_chunks_before_corpus_coverage(self, two_decrees):
        """A 2010 question must not be answered with law indexed from 2015 on."""
        from src.agent.graph import temporal_filter_node

        out = temporal_filter_node(self._state("2010-01-01", two_decrees))
        assert out["retrieved_chunks"] == []
        assert out["time_out_of_range"] is True
        assert "2015-01-01" in out["steps"][-1]["detail"]

    def test_node_defaults_to_today_when_state_has_no_date(self, two_decrees):
        from src.agent.graph import temporal_filter_node

        state = self._state("", two_decrees)
        out = temporal_filter_node(state)
        assert out["time_out_of_range"] is False
        assert today_iso() in out["steps"][-1]["detail"]

    def test_node_sits_between_retrieve_and_grade(self):
        """Filtering must happen after retrieval and before grading.

        Grading relevance on chunks that are out of force would judge text the
        answer can never cite.
        """
        from src.agent.graph import build_graph

        edges = set(build_graph().builder.edges)
        assert ("do_retrieve", "do_temporal_filter") in edges
        assert ("do_temporal_filter", "do_grade") in edges
        assert ("do_retrieve", "do_grade") not in edges


class TestGeneratorOutOfRange:
    def test_out_of_range_answer_names_coverage_start(self):
        from src.rag.generator import generate_answer

        result = generate_answer(
            "lương tối thiểu vùng I năm 2010",
            [],
            as_of_date="2010-01-01",
            time_out_of_range=True,
            earliest_covered="2015-01-01",
        )
        assert "2015-01-01" in result["answer"]
        assert "2010-01-01" in result["answer"]
        assert result["used_llm"] == "none"
        assert result["chunk_count"] == 0

    def test_as_of_block_states_the_date(self):
        from src.rag.generator import _as_of_block

        block = _as_of_block("2026-02-01")
        assert "2026-02-01" in block
        assert _as_of_block("") == ""
