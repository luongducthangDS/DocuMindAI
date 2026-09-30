"""Mốc tra cứu (as_of) và nguồn của nó phải nằm trên trace gốc.

Không có tag này, trace của eval (gold set gán sẵn as_of_date=2024-02-01) trông y
như bug "hệ thống tự chọn sai ngày" — người đọc trace không phân biệt được.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.rag.temporal import today_iso


class _EchoGraph:
    async def ainvoke(self, state):
        return {**state, "answer": "trả lời"}


@pytest.mark.parametrize(
    "as_of_date, expected",
    [
        (None, [f"as_of:{today_iso()}", "as_of_src:today"]),
        ("  ", [f"as_of:{today_iso()}", "as_of_src:today"]),
        ("2024-02-01", ["as_of:2024-02-01", "as_of_src:request"]),
    ],
)
async def test_run_agent_tags_as_of_and_source(as_of_date, expected):
    from src.agent import graph

    tags = {}

    def fake_end_trace(*_a, extra_tags=None, **_kw):
        tags["value"] = extra_tags

    with patch.object(graph, "get_graph", return_value=_EchoGraph()), \
         patch.object(graph, "end_trace", fake_end_trace):
        await graph.run_agent("Làm thêm giờ ngày thường được trả bao nhiêu?", as_of_date=as_of_date)
    for tag in expected:
        assert tag in tags["value"]
