"""src/rag/graphrag.py + tool versions/overview của src/agent/research.py (LLM giả lập)."""
import json
from types import SimpleNamespace

import pytest
from llama_index.core.schema import TextNode

from src.agent import research as AR
from src.rag import graphrag as GR
from src.rag import legal_graph as LG
from src.rag.context import RetrievalContext
from src.rag.retriever import RetrievedChunk

OLD = {"title": "Luật Bảo hiểm xã hội", "so_hieu": "58/2014/QH13", "doc_id": "58-2014-QH13",
       "effective_from": "2016-01-01", "effective_to": "2025-07-01"}
NEW = {"title": "Luật Bảo hiểm xã hội", "so_hieu": "41/2024/QH15", "doc_id": "41-2024-QH15",
       "effective_from": "2025-07-01", "effective_to": "9999-12-31"}


def art(base, dieu, title, text="…", **extra):
    meta = {**base, "dieu": dieu, "clause_uid": f"{base['doc_id']}__d{dieu}", "dieu_header": f"Điều {dieu}. {title}",
            "dieu_tieu_de": title, **extra}
    return TextNode(text=f"Điều {dieu}. {title}\n{text}", metadata=meta, id_=f"{base['doc_id']}{dieu}")


def chunk(node):
    return RetrievedChunk(text=node.text, score=1.0, metadata=dict(node.metadata))


O54 = art(OLD, 54, "Điều kiện hưởng lương hưu", "đủ 20 năm đóng bảo hiểm xã hội")
N64 = art(NEW, 64, "Đối tượng và điều kiện hưởng lương hưu", "từ đủ 15 năm")
N66 = art(NEW, 66, "Mức lương hưu hằng tháng")
N85 = art(NEW, 85, "Trợ cấp mai táng")
O66 = art(OLD, 66, "Trợ cấp mai táng")
SECRET = art(OLD, 99, "Điều riêng tenant khác", tenant_id="acme")


@pytest.fixture(autouse=True)
def graph(tmp_path):
    LG.build_index([O54, N64, N66, N85, O66, SECRET])
    (tmp_path / "l.json").write_text(json.dumps([
        {"old": ["58-2014-QH13", 54], "new": ["41-2024-QH15", 64], "score": 0.89, "old_title": "Điều kiện hưởng lương hưu",
         "new_title": "Đối tượng và điều kiện hưởng lương hưu"},
        {"old": ["58-2014-QH13", 99], "new": ["41-2024-QH15", 66], "score": 0.9, "old_title": "x", "new_title": "y"},
        {"old": ["58-2014-QH13", 66], "new": ["41-2024-QH15", 85], "score": 0.98, "old_title": "Trợ cấp mai táng",
         "new_title": "Trợ cấp mai táng"},
    ]), encoding="utf-8")
    (tmp_path / "c.json").write_text(json.dumps([
        {"id": "c000", "label": "Chế độ hưu trí", "summary": "Điều kiện và mức lương hưu", "docs": ["41-2024-QH15"],
         "members": [{"doc_id": "41-2024-QH15", "dieu": 64, "title": "Đối tượng và điều kiện hưởng lương hưu"},
                     {"doc_id": "58-2014-QH13", "dieu": 54, "title": "Điều kiện hưởng lương hưu"},
                     {"doc_id": "41-2024-QH15", "dieu": 66, "title": "Mức lương hưu hằng tháng"}]},
        {"id": "c001", "label": "", "summary": "", "docs": ["41-2024-QH15"],
         "members": [{"doc_id": "41-2024-QH15", "dieu": 85, "title": "Trợ cấp mai táng"}]},
    ]), encoding="utf-8")
    assert GR.load(tmp_path / "l.json", tmp_path / "c.json") == (3, 2)
    yield
    LG.build_index([])
    GR.load(tmp_path / "none", tmp_path / "none")


CTX = RetrievalContext(as_of_date="2026-11-10")


# ── Dựng offline ───────────────────────────────────────────────────────────────

def test_doc_lineage_from_manifest_both_directions():
    man = {"documents": [{"doc_id": "41-2024-QH15", "so_hieu": "41/2024/QH15", "thay_the": ["58/2014/QH13", "71/2006/QH11"]},
                         {"doc_id": "58-2014-QH13", "so_hieu": "58/2014/QH13", "bi_thay_the_boi": ["41/2024/QH15"]}]}
    # văn bản ngoài corpus (71/2006) bị bỏ; hai chiều khai cùng một cặp không nhân bản
    assert GR.doc_lineage(man, {"41-2024-QH15", "58-2014-QH13"}) == [("58-2014-QH13", "41-2024-QH15")]


def test_match_articles_mutual_best_and_threshold():
    rows = [("o1", "", {"doc_id": "A", "dieu": 1, "clause_uid": "A__d1", "dieu_tieu_de": "Trợ cấp mai táng"}),
            ("o2", "", {"doc_id": "A", "dieu": 2, "clause_uid": "A__d2", "dieu_tieu_de": "Thanh tra"}),
            ("n1", "", {"doc_id": "B", "dieu": 9, "clause_uid": "B__d9", "dieu_tieu_de": "Trợ cấp mai táng"}),
            ("n2", "", {"doc_id": "B", "dieu": 3, "clause_uid": "B__d3", "dieu_tieu_de": "Hiệu lực thi hành"})]
    vec = {"o1": [1, 0, 0], "o2": [0, 1, 0], "n1": [1, 0.1, 0], "n2": [0, 0.2, 1]}
    out = GR.match_articles(GR.first_chunks(rows), vec, [("A", "B")])
    assert [(e["old"], e["new"]) for e in out] == [(["A", 1], ["B", 9])]  # Thanh tra ⇢ Hiệu lực: điểm thấp, bỏ


def test_detect_communities_groups_by_references_and_chapter():
    nodes = [SimpleNamespace(text=f"Điều {d}. x" + (" theo Điều 2 của Luật này" if d == 1 else ""),
                             metadata={"doc_id": "41-2024-QH15", "dieu": d, "title": "Luật Bảo hiểm xã hội",
                                       "clause_uid": f"41-2024-QH15__d{d}", "chuong": "I" if d < 4 else "II",
                                       "dieu_tieu_de": f"t{d}", "so_hieu": "41/2024/QH15"}) for d in (1, 2, 3, 7, 8, 9)]
    LG.build_index(nodes)
    comms = GR.detect_communities(nodes, [], resolution=1.0)
    groups = sorted(sorted(m["dieu"] for m in c["members"]) for c in comms)
    assert groups == [[1, 2, 3], [7, 8, 9]]


def test_summary_flags_invented_numbers():
    c = {"members": [{"so_hieu": "41/2024/QH15", "dieu": 85, "title": "Trợ cấp mai táng"}]}
    ok = GR.summarize(c, lambda p: '{"label": "Mai táng", "summary": "Điều 85 của 41/2024/QH15"}')
    bad = GR.summarize(c, lambda p: '{"label": "Mai táng", "summary": "bằng 10 lần mức tham chiếu"}')
    assert ok["issues"] == [] and "10" in bad["issues"][0]


# ── Runtime ────────────────────────────────────────────────────────────────────

def test_version_candidates_skip_date_filter_but_not_tenant():
    got = GR.version_candidates([chunk(N64), chunk(N66)], CTX)
    # 58/2014 Điều 54 đã hết hiệu lực vẫn là ứng viên (so sánh cần nó); Điều 99 của tenant khác thì không
    assert [(n.metadata["clause_uid"], i["direction"]) for n, i in got] == [("58-2014-QH13__d54", "truoc")]
    assert got[0][1]["of"] == "41-2024-QH15__d64"
    back = GR.version_candidates([chunk(O54)], RetrievalContext(as_of_date="2025-03-01"))
    assert [(n.metadata["clause_uid"], i["direction"]) for n, i in back] == [("41-2024-QH15__d64", "sau")]
    assert GR.version_candidates([chunk(N64), chunk(O54)], CTX) == []  # đã có mặt


def test_community_search_and_reduce_only_in_force():
    cands = GR.community_candidates("Điều kiện hưởng lương hưu gồm những gì?")
    assert cands[0]["id"] == "c000"
    assert GR._COMMUNITIES[1]["label"] == "Trợ cấp mai táng"  # cụm chưa tóm tắt: nhãn từ tiêu đề
    arts = GR.community_articles(cands[0], "điều kiện hưởng lương hưu", CTX.allows, present=set())
    assert [n.metadata["clause_uid"] for n in arts] == ["41-2024-QH15__d64", "41-2024-QH15__d66"]  # 58/2014 hết hiệu lực


# ── Agent tools ────────────────────────────────────────────────────────────────

def llm(monkeypatch, *replies):
    prompts, it = [], iter(replies)

    def fake(prompt, **kw):
        prompts.append(prompt)
        return json.dumps(next(it), ensure_ascii=False)

    monkeypatch.setattr("src.rag.generator.gemini_generate", fake)
    return prompts


CHANGE_Q = "Số năm đóng để hưởng lương hưu theo luật 2024 thay đổi thế nào so với Luật 2014?"


def test_versions_tool_offered_only_for_change_questions(monkeypatch):
    prompts = llm(monkeypatch, {"enough": False, "versions": ["V1"], "reason": "cần luật cũ"}, {"enough": True})
    got, steps = AR.research(CHANGE_Q, [chunk(N64)], CTX, rounds=2)
    assert "V1. Điều 54. Điều kiện hưởng lương hưu — 58/2014/QH13 (phiên bản trước của Điều 64 41/2024/QH15" in prompts[0]
    old = got[1].metadata
    assert old["clause_uid"] == "58-2014-QH13__d54" and old["agent_step"] == "versions"
    assert old["lineage_of"] == "41-2024-QH15__d64" and old["in_force_at_query"] is False
    assert "phiên bản trước" in steps[0]["detail"]

    from src.rag.generator import _build_context
    ctx_block, _ = _build_context(got)
    assert "[2] (Phiên bản trước của Điều 64 41/2024/QH15 — đoạn [1]; hiệu lực 2016-01-01 → 2025-07-01; " \
           "KHÔNG áp dụng tại ngày tra cứu" in ctx_block


def test_plain_question_gets_no_new_tools_and_no_llm_call(monkeypatch):
    llm(monkeypatch)  # không reply: gọi là lỗi StopIteration ⇒ research fail open, steps rỗng
    got, steps = AR.research("Điều kiện hưởng lương hưu là gì?", [chunk(N64)], CTX, rounds=2)
    assert len(got) == 1 and steps == []


def test_overview_tool_pulls_in_force_articles_of_chosen_community(monkeypatch):
    monkeypatch.setenv("AGENT_OVERVIEW", "true")
    from src.config import get_settings
    get_settings.cache_clear()
    prompts = llm(monkeypatch, {"enough": False, "overview": ["C1"], "reason": "câu tổng quát"}, {"enough": True})
    got, steps = AR.research("Chế độ hưu trí gồm những nội dung gì?", [chunk(N85)], CTX, rounds=2)
    assert "C1. Chế độ hưu trí — Điều kiện và mức lương hưu" in prompts[0]
    new = [c.metadata["clause_uid"] for c in got[1:]]
    assert new == ["41-2024-QH15__d64", "41-2024-QH15__d66"] and all(c.metadata["agent_step"] == "overview" for c in got[1:])
    assert 'cụm "Chế độ hưu trí"' in steps[0]["detail"]


def test_enough_with_codes_still_executes_then_stops(monkeypatch):
    """gr_c04/gr_c05: Gemini trả {"enough": true, "versions": ["V1"]} = "đủ khi có thêm V1" — phải lấy V1, không hỏi vòng 2."""
    prompts = llm(monkeypatch, {"enough": True, "versions": ["V1"], "reason": "đủ với V1"})
    got, _ = AR.research(CHANGE_Q, [chunk(N64)], CTX, rounds=3)
    assert [c.metadata["clause_uid"] for c in got] == ["41-2024-QH15__d64", "58-2014-QH13__d54"] and len(prompts) == 1


def test_version_candidates_ranked_by_query_before_cut(monkeypatch):
    """gr_c11: cặp cần nằm ở đoạn hạng cuối — phải xếp theo câu hỏi trước khi cắt limit."""
    chunks = [chunk(N85), chunk(N64)]  # đoạn mai táng đứng trước ⇒ không xếp thì Điều 66 cũ chiếm chỗ
    assert GR.version_candidates(chunks, CTX, limit=1)[0][0].metadata["clause_uid"] == "58-2014-QH13__d66"
    got = GR.version_candidates(chunks, CTX, limit=1, query="điều kiện hưởng lương hưu")
    assert got[0][0].metadata["clause_uid"] == "58-2014-QH13__d54"


def test_overview_off_by_default(monkeypatch):
    llm(monkeypatch)  # không reply: có lời gọi là lỗi
    got, steps = AR.research("Chế độ hưu trí gồm những nội dung gì?", [chunk(N85)], CTX, rounds=2)
    assert len(got) == 1 and steps == []


def test_budget_keeps_the_article_being_compared():
    from src.rag.generator import _MAX_TOTAL_CHARS, _budget

    size = _MAX_TOTAL_CHARS // 8
    base = [RetrievedChunk(text="x" * size, score=1.0, metadata={"clause_uid": f"c{i}"}) for i in range(8)]
    old = RetrievedChunk(text="y" * size, score=0.0, metadata={"clause_uid": "old", "agent_step": "versions",
                                                               "lineage_of": "c7"})
    kept = {c.metadata["clause_uid"] for _, c, _ in _budget(base + [old])}
    assert {"old", "c7"} <= kept and "c6" not in kept


def test_bogus_codes_add_nothing(monkeypatch):
    llm(monkeypatch, {"enough": False, "versions": ["V7"], "overview": ["C9", "Luật X"]})
    got, _ = AR.research(CHANGE_Q, [chunk(N64)], CTX, rounds=3)
    assert len(got) == 1


# ── Dữ liệu đã dựng ───────────────────────────────────────────────────────────

def test_built_graph_files_reference_corpus_documents():
    from pathlib import Path

    import yaml

    man = yaml.safe_load(Path("docs/corpus/corpus_manifest.yaml").read_text(encoding="utf-8"))
    docs = {d["doc_id"] for d in man["documents"]}
    lineage = json.loads(GR.LINEAGE_PATH.read_text(encoding="utf-8"))
    comms = json.loads(GR.COMMUNITIES_PATH.read_text(encoding="utf-8"))
    assert lineage and all(e["old"][0] in docs and e["new"][0] in docs and e["score"] >= GR._LINEAGE_MIN for e in lineage)
    assert len({c["id"] for c in comms}) == len(comms) and all(m["doc_id"] in docs for c in comms for m in c["members"])
