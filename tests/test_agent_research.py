"""src/agent/research.py — vòng agentic RAG: LLM quyết định đủ / lấy điều được dẫn chiếu / tìm thêm (LLM giả lập)."""
import json

import pytest
from llama_index.core.schema import TextNode

from src.agent import research as AR
from src.rag import legal_graph as LG
from src.rag.context import RetrievalContext
from src.rag.retriever import RetrievedChunk

BHXH = {"title": "Luật Bảo hiểm xã hội", "so_hieu": "41/2024/QH15", "effective_from": "2025-07-01",
        "effective_to": "9999-12-31"}


def art(dieu, title, text="…"):
    meta = {**BHXH, "doc_id": "41-2024-QH15", "dieu": dieu, "clause_uid": f"41-2024-QH15__d{dieu}",
            "dieu_header": f"Điều {dieu}. {title}"}
    return TextNode(text=f"Điều {dieu}. {title}\n{text}", metadata=meta, id_=f"d{dieu}")


D23 = art(23, "Chế độ đối với người lao động không đủ điều kiện hưởng lương hưu",
          "4. Mức trợ cấp hằng tháng được điều chỉnh theo quy định tại Điều 67 của Luật này; "
          "thủ tục theo Điều 22 của Luật này.")


@pytest.fixture(autouse=True)
def graph():
    LG.build_index([D23, art(22, "Các chế độ, trình tự, thủ tục"), art(67, "Điều chỉnh lương hưu",
                                                                         "trên cơ sở mức tăng của chỉ số giá tiêu dùng")])
    yield
    LG.build_index([])


def chunk(node):
    return RetrievedChunk(text=node.text, score=1.0, metadata=dict(node.metadata))


def llm(monkeypatch, *replies):
    """Giả lập LLM: trả lần lượt các reply, ghi lại prompt."""
    prompts, it = [], iter(replies)

    def fake(prompt, **kw):
        prompts.append(prompt)
        r = next(it)
        if isinstance(r, Exception):
            raise r
        return r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)

    monkeypatch.setattr("src.rag.generator.gemini_generate", fake)
    return prompts


CTX = RetrievalContext(as_of_date="2025-10-01")
Q = "Trợ cấp hằng tháng có được điều chỉnh không, trên cơ sở nào?"


def test_fetch_picks_cited_article_from_graph_candidates(monkeypatch):
    code = next(c for c, n, _, _ in AR._candidates([chunk(D23)], CTX) if n.metadata["dieu"] == 67)
    prompts = llm(monkeypatch, {"enough": False, "fetch": [code], "search": [], "reason": "thiếu Điều 67"},
                  {"enough": True, "reason": "đủ"})
    got, steps = AR.research(Q, [chunk(D23)], CTX, rounds=2)
    # LLM được thấy tiêu đề điều và câu dẫn chiếu gốc của từng ứng viên
    line = next(l for l in prompts[0].splitlines() if l.startswith(f"{code}."))
    assert "Điều 67. Điều chỉnh lương hưu" in line and "điều chỉnh theo quy định tại Điều 67" in line
    assert [c.metadata["clause_uid"] for c in got] == ["41-2024-QH15__d23", "41-2024-QH15__d67"]
    assert got[1].metadata["graph_via"] == ["41-2024-QH15__d23"] and got[1].metadata["agent_step"] == "fetch"
    assert [s["label"] for s in steps] == ["Tìm thêm", "Đánh giá đủ thông tin"]
    cands_2 = prompts[1].split("CHƯA có nội dung:")[1].split("Quy tắc:")[0]
    assert "Điều 67. Điều chỉnh" not in cands_2  # đã có thì không còn là ứng viên


def test_enough_stops_without_changes(monkeypatch):
    llm(monkeypatch, {"enough": True, "reason": "đủ"})
    got, steps = AR.research(Q, [chunk(D23)], CTX, rounds=2)
    assert [c.metadata["clause_uid"] for c in got] == ["41-2024-QH15__d23"]
    assert steps[0]["label"] == "Đánh giá đủ thông tin"


@pytest.mark.parametrize("reply", ["không phải JSON", RuntimeError("FPT 503"), '["R1"]'])
def test_llm_failure_fails_open(monkeypatch, reply):
    from src.rag.context import degraded_flags, reset_degraded, start_degraded

    llm(monkeypatch, reply)
    token = start_degraded()
    try:
        got, steps = AR.research(Q, [chunk(D23)], CTX, rounds=2)
        assert len(got) == 1 and steps == []
        assert degraded_flags() == ["agent_research_failed"]  # fail open nhưng để dấu cho eval/trace
    finally:
        reset_degraded(token)


def test_unknown_code_fetches_nothing_and_stops(monkeypatch):
    """LLM bịa mã R9 / viết thẳng tên văn bản ⇒ không lấy gì (không có đường bịa văn bản)."""
    prompts = llm(monkeypatch, {"enough": False, "fetch": ["R9", "Điều 99 Luật X"], "search": []})
    got, _ = AR.research(Q, [chunk(D23)], CTX, rounds=3)
    assert len(got) == 1 and len(prompts) == 1


def test_search_adds_only_new_passages_and_marks_them(monkeypatch):
    llm(monkeypatch, {"enough": False, "fetch": [], "search": ["chỉ số giá tiêu dùng điều chỉnh lương hưu"]})
    seen = {}

    def fake_retrieve(q, ctx):
        seen["q"], seen["ctx"] = q, ctx
        hits = [chunk(D23), chunk(art(67, "Điều chỉnh lương hưu"))] + [chunk(art(n, f"x{n}")) for n in (80, 81, 82)]
        hits[1].metadata["graph_via"] = ["khác"]
        return hits

    monkeypatch.setattr("src.rag.retriever.retrieve_with_context", fake_retrieve)
    got, steps = AR.research(Q, [chunk(D23)], CTX, rounds=1)
    assert seen["ctx"] is CTX  # cùng tenant/ACL/mốc hiệu lực
    new = got[1:]
    assert [c.metadata["clause_uid"] for c in new] == ["41-2024-QH15__d67", "41-2024-QH15__d80", "41-2024-QH15__d81"]
    assert all(c.metadata["agent_query"] for c in new) and "graph_via" not in new[0].metadata
    assert "tìm \"chỉ số giá" in steps[0]["detail"]


def test_no_cited_article_missing_means_no_llm_call(monkeypatch):
    """Ngân sách: ngữ cảnh không dẫn tới điều nào chưa có ⇒ không tốn lời gọi nào (test_llm_budget)."""
    llm(monkeypatch)
    alone = chunk(art(90, "Tiền lương", "là số tiền người sử dụng lao động trả"))
    got, steps = AR.research(Q, [alone], CTX, rounds=2)
    assert len(got) == 1 and steps == []


def test_any_internal_error_fails_open(monkeypatch):
    """Lỗi không phải LLM (đồ thị, truy hồi con…) cũng không được làm hỏng lượt hỏi — WS từng treo vì vậy."""
    monkeypatch.setattr(AR, "_candidates", lambda *a: (_ for _ in ()).throw(KeyError("boom")))
    chunks = [chunk(D23)]
    got, steps = AR.research(Q, chunks, CTX, rounds=2)
    assert got is chunks and steps == []


def test_zero_rounds_never_calls_llm(monkeypatch):
    llm(monkeypatch)  # không reply nào — gọi là StopIteration
    got, steps = AR.research(Q, [chunk(D23)], CTX, rounds=0)
    assert len(got) == 1 and steps == []


def test_budget_gives_agent_passages_room():
    from src.rag.generator import _MAX_TOTAL_CHARS, _budget

    size = _MAX_TOTAL_CHARS // 8
    base = [RetrievedChunk(text="x" * size, score=1.0, metadata={"clause_uid": f"c{i}"}) for i in range(8)]
    found = RetrievedChunk(text="y" * 100, score=0.0, metadata={"clause_uid": "s", "agent_query": "q"})
    assert [i for i, _, _ in _budget(base + [found])][-1] == 9
