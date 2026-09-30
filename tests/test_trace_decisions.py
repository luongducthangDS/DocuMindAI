"""Trace ghi QUYẾT ĐỊNH kèm lý do, không chỉ "bước đã chạy".

Tiêu chí: chọn ngẫu nhiên một câu trả lời sai, chỉ nhìn trace phải nói được nó tra
luật tại ngày nào và vì sao, lấy những điều nào, router/compliance chọn đường nào
vì sao, và trích dẫn trỏ về đâu. Và tracing không bao giờ được làm hỏng câu trả lời.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.rag.retriever import RetrievedChunk

NOW = datetime.now(timezone.utc)
BLLD_107 = {
    "so_hieu": "45/2019/QH14", "dieu": 107, "khoan": 3, "effective_from": "2021-01-01",
    "effective_to": "9999-12-31", "version_id": "blld-107-3__v2021-01-01",
}


def _capture(monkeypatch, module) -> dict:
    """Bật tracing giả cho `module` và giữ lại output của record_span cuối cùng."""
    captured: dict = {}
    monkeypatch.setattr(module, "tracing_active", lambda: True)
    monkeypatch.setattr(module, "record_span",
                        lambda name, _type, _inp, out, *_a, **_kw: captured.update({name: out}))
    return captured


async def test_canned_refusals_are_recognised_as_abstentions():
    """_ABSTAIN_OPENERS phải khớp mọi câu từ chối viết sẵn — sửa câu mà quên danh sách là test đỏ."""
    from src.rag.generator import answer_audit, generate_answer, stream_answer

    low = RetrievedChunk(text="Điều 1", score=0.01, metadata={})
    answers = [
        generate_answer("q", [])["answer"],
        generate_answer("q", [low], min_score=0.5)["answer"],
        generate_answer("q", [low], as_of_date="2010-01-01", time_out_of_range=True,
                        earliest_covered="2015-01-01")["answer"],
        "".join([t async for t in stream_answer("q", [])]),
        # Hai khuôn của rule 4 trong _SYSTEM_PROMPT (LLM viết, có khi in đậm).
        "**Câu hỏi này nằm ngoài phạm vi pháp luật lao động.**\n\n**Bạn muốn hỏi cụ thể:**",
        "Tôi chưa tìm thấy quy định khớp với câu hỏi này. Bạn đang hỏi về chế độ nào?",
    ]
    for answer in answers:
        assert answer_audit(answer, [])["flags"] == ["abstained"], answer


def test_answer_audit_maps_citations_without_chunk_text():
    from src.rag.generator import answer_audit

    source = {"index": 1, **BLLD_107, "score": 0.02}
    audit = answer_audit("Tối đa **300 giờ/năm** [1]; ngày nghỉ hằng tuần 200% [3].", [source])
    assert audit["cited"] == [1, 3]
    assert audit["unmapped"] == [3]  # [3] hiện trên UI nhưng không có thẻ nguồn
    assert audit["flags"] == ["invalid_citation"]
    assert audit["sources"] == [{
        "n": 1, "doc": "45/2019/QH14", "article": "Điều 107", "clause": 3,
        "from": "2021-01-01", "to": "9999-12-31", "id": "blld-107-3__v2021-01-01",
    }]
    assert answer_audit("Có, công ty được phép.", [])["flags"] == ["uncited"]


def test_retrieval_span_lists_each_chunk_by_reference_not_text(monkeypatch):
    from src.rag import retriever

    captured = _capture(monkeypatch, retriever)
    chunks = [
        RetrievedChunk(text="Điều 107. Làm thêm giờ ...", score=0.01639, dense_score=0.8123,
                       bm25_score=11.2468, metadata=BLLD_107),
        RetrievedChunk(text="nội dung tải lên", score=0.01613,
                       metadata={"title": "quy_che.pdf", "dieu_header": "Mục 2", "source": "user_upload"}),
    ]
    retriever.trace_retrieval("q", chunks, "hybrid", None, NOW)

    out = captured["retrieve-documents"]
    assert out["path"] == "hybrid" and out["score"] in ("rrf", "rerank")
    assert out["chunks"][0] == {"rank": 1, "score": 0.0164, "cosine": 0.812, "bm25": 11.25,
                                "doc": "45/2019/QH14", "article": "Điều 107", "clause": 3,
                                "from": "2021-01-01", "to": "9999-12-31", "id": "blld-107-3__v2021-01-01"}
    # Không có cosine lẫn bm25 = không nhánh nào ghi điểm (vd. đường direct);
    # tài liệu người dùng tải lên phải lộ nguồn gốc.
    assert out["chunks"][1] == {"rank": 2, "score": 0.0161, "doc": "quy_che.pdf",
                                "article": "Mục 2", "origin": "user_upload"}
    assert "Làm thêm giờ" not in json.dumps(out, ensure_ascii=False)


def test_tracing_never_breaks_the_request(monkeypatch):
    """Chunk lạ (không có metadata) làm hỏng payload trace — request vẫn phải chạy tiếp."""
    from src.rag import retriever

    _capture(monkeypatch, retriever)
    assert retriever.trace_retrieval("q", [SimpleNamespace(score=1.0)], "hybrid", None, NOW) is None


def test_temporal_filter_span_says_what_was_dropped_and_why(monkeypatch):
    from src.agent import graph

    captured = _capture(monkeypatch, graph)
    old = RetrievedChunk(text="4.960.000", score=0.02, metadata={
        "so_hieu": "74/2024/NĐ-CP", "clause_uid": "lttv-3", "effective_from": "2024-07-01",
        "effective_to": "2026-01-01"})
    new = RetrievedChunk(text="5.310.000", score=0.01, metadata={
        "so_hieu": "293/2025/NĐ-CP", "clause_uid": "lttv-3", "effective_from": "2026-01-01"})
    graph.temporal_filter_node({"retrieved_chunks": [old, new], "as_of_date": "2025-03-01", "steps": []})

    out = captured["temporal-filter"]
    assert out["kept"] == 1
    assert [(d["doc"], d["reason"]) for d in out["dropped"]] == [("293/2025/NĐ-CP", "out_of_force")]


LONG_Q = "Nhân viên đã làm 17 năm cho cùng một công ty thì mỗi năm được nghỉ bao nhiêu ngày?"


@pytest.mark.parametrize("query, llm, intent, route", [
    ("xin chào", None, "smalltalk", "smalltalk"),
    ("So sánh hợp đồng xác định thời hạn và hợp đồng không xác định thời hạn", None, "compare", "keyword"),
    ("Thử việc tối đa bao lâu?", None, "simple_qa", "short_query"),
    ("Người lao động được nghỉ hằng năm bao nhiêu ngày làm việc?", None, "simple_qa", "legal_term_no_digit"),
    (LONG_Q, "compliance_check", "compliance_check", "llm"),
    (LONG_Q, "tôi nghĩ là simple_qa", "simple_qa", "llm_invalid"),
    (LONG_Q, RuntimeError("503 overloaded"), "simple_qa", "llm_error"),
])
def test_router_records_why_it_chose_the_intent(monkeypatch, query, llm, intent, route):
    from src.agent.graph import router_node
    from src.rag import generator

    def fake_llm(*_a, **kw):
        assert llm is not None, "luật nhanh không được gọi LLM"
        assert kw.get("name") == "route-intent"
        if isinstance(llm, Exception):
            raise llm
        return llm

    monkeypatch.setattr(generator, "gemini_generate", fake_llm)
    out = router_node({"query": query, "steps": []})
    assert (out["intent"], out["route_source"]) == (intent, route)


@pytest.mark.usefixtures("real_criteria", "no_network")
def test_compliance_result_says_how_the_criterion_was_matched(monkeypatch):
    from src.rag import compliance

    r = compliance.check_compliance("Công ty cho nhân viên làm thêm 1.200 giờ trong năm có đúng luật không?")
    assert r["criterion_id"] == "lam_them_gio_trong_nam"
    assert r["match"]["by"] == "keyword" and r["match"]["keywords"]

    monkeypatch.setattr(compliance, "_match_by_embedding", lambda *_a: None)
    r = compliance.check_compliance("Giá vàng SJC hôm nay bao nhiêu?")
    assert r["verdict"] == "no_match" and r["match"]["by"] == "embedding"


def test_rotated_429_is_warning_only_exhaustion_is_error(monkeypatch):
    """429 rồi cặp kế tiếp trả lời được = không phải lỗi của câu hỏi (WARNING).
    Hết cả vòng xoay mới là ERROR — số ERROR trên Langfuse phải là sự cố thật."""
    from src.rag import generator

    calls: list[dict] = []
    monkeypatch.setattr(generator, "record_generation", lambda *a, **kw: calls.append({"name": a[0], **kw}))
    monkeypatch.setattr(generator, "_gemini_pairs", lambda _models=None: [("k1", "m1"), ("k2", "m2")])
    outcomes = iter([RuntimeError("429 Resource has been exhausted"),
                     SimpleNamespace(text="ok", usage_metadata=None)])

    def one_429_then_ok(*_a, **_kw):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(generator, "gemini_call", one_429_then_ok)
    assert generator.gemini_generate("p", name="route-intent") == "ok"
    assert [c["level"] for c in calls if c.get("error")] == ["WARNING"]

    def always_429(*_a, **_kw):
        raise RuntimeError("429 quota exceeded")

    monkeypatch.setattr(generator, "gemini_call", always_429)
    with pytest.raises(RuntimeError, match="429"):
        generator.gemini_generate("p", name="route-intent")
    assert [c.get("level", "ERROR") for c in calls[2:] if c.get("error")] == ["WARNING", "WARNING", "ERROR"]
    # Mọi bản ghi — thành công, lượt thử lỗi, hết vòng — mang tên mục đích của lời gọi.
    assert {c["name"] for c in calls} == {"route-intent"}


def test_every_llm_call_site_names_its_purpose():
    """Langfuse xếp mọi generation phẳng dưới trace gốc: lời gọi mới mà quên `name` sẽ lại
    hiện thành "gemini-generate" chung chung, không biết nó để làm gì."""
    import ast
    from pathlib import Path

    unnamed = []
    for path in (Path(__file__).resolve().parents[1] / "src").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            func = getattr(node, "func", None)
            called = getattr(func, "id", None) or getattr(func, "attr", None)
            if (isinstance(node, ast.Call) and called == "gemini_generate"
                    and not any(k.arg == "name" for k in node.keywords)):
                unnamed.append(f"{path.name}:{node.lineno}")
    assert unnamed == []


def test_every_span_carries_env_and_release_root_carries_index(monkeypatch):
    from src import langfuse_otel as lf

    sent: list[dict] = []
    monkeypatch.setattr(lf, "_auth", lambda: ("pk", "sk", "https://langfuse.test"))
    monkeypatch.setattr(lf, "_send_batch", sent.extend)
    monkeypatch.setattr(lf, "app_release", lambda: "abc123def456")
    monkeypatch.setattr(lf, "_index_version", "chroma/documind_legal/gemini-embedding-001/1151")

    ctx, token = lf.start_trace("t", session_id="s")
    lf.record_span("retrieve-documents", "retriever", "q", {"chunks": []}, NOW, NOW)
    lf.end_trace(ctx, token, "t", "q", "a")

    assert len(sent) == 2
    for span in sent:
        attrs = {a["key"]: a["value"].get("stringValue") for a in span["attributes"]}
        assert attrs["langfuse.environment"] == "development"
        assert attrs["langfuse.release"] == "abc123def456"
    root = next(s for s in sent if "parentSpanId" not in s)
    child = next(s for s in sent if "parentSpanId" in s)
    root_attrs = {a["key"]: a["value"].get("stringValue") for a in root["attributes"]}
    child_attrs = {a["key"]: a["value"].get("stringValue") for a in child["attributes"]}
    assert root_attrs["langfuse.trace.metadata.index"] == "chroma/documind_legal/gemini-embedding-001/1151"
    assert json.loads(child_attrs["langfuse.observation.output"]) == {"chunks": []}
