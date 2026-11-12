"""legal_graph — dẫn chiếu "Điều N của <văn bản>" giải đúng văn bản theo thời kỳ, và mở rộng ngữ cảnh."""
from types import SimpleNamespace as NS

import pytest

from src.rag import legal_graph as LG


def node(doc, dieu, start, title, text="…", so_hieu="", end="9999-12-31", uid=None):
    meta = {"doc_id": doc, "dieu": dieu, "effective_from": start, "effective_to": end, "title": title,
            "so_hieu": so_hieu, "clause_uid": uid or f"{doc}__d{dieu}"}
    return NS(text=text, metadata=meta)


@pytest.fixture(autouse=True)
def corpus():
    LG.build_index([
        node("58-2014-QH13", 54, "2016-01-01", "Luật Bảo hiểm xã hội", so_hieu="58/2014/QH13", end="2025-07-01"),
        node("41-2024-QH15", 64, "2025-07-01", "Luật Bảo hiểm xã hội", so_hieu="41/2024/QH15"),
        node("41-2024-QH15", 98, "2025-07-01", "Luật Bảo hiểm xã hội", so_hieu="41/2024/QH15"),
        node("45-2019-QH14", 35, "2021-01-01", "Bộ luật Lao động"),
        node("45-2019-QH14", 36, "2021-01-01", "Bộ luật Lao động"),
        node("45-2019-QH14", 37, "2021-01-01", "Bộ luật Lao động"),
        node("45-2019-QH14", 39, "2021-01-01", "Bộ luật Lao động"),
        node("45-2019-QH14", 21, "2021-01-01", "Bộ luật Lao động"),
        node("145-2020-ND-CP", 21, "2021-02-01", "Nghị định quy định chi tiết Bộ luật Lao động", so_hieu="145/2020/NĐ-CP"),
        node("145-2020-ND-CP", 35, "2021-02-01", "Nghị định quy định chi tiết Bộ luật Lao động", so_hieu="145/2020/NĐ-CP"),
        node("115-2015-ND-CP", 6, "2016-01-01", "Nghị định quy định chi tiết Luật BHXH", so_hieu="115/2015/NĐ-CP"),
        node("135-2020-ND-CP", 6, "2021-01-01", "Nghị định về tuổi nghỉ hưu", so_hieu="135/2020/NĐ-CP"),
    ])
    yield
    LG.build_index([])


def refs(text, doc, dieu, start, title=""):
    return LG.references(text, {"doc_id": doc, "dieu": dieu, "effective_from": start, "title": title})


def test_same_law_name_resolves_by_the_citing_text_date():
    cite = "không đủ điều kiện theo quy định tại Điều 54 của Luật Bảo hiểm xã hội"
    assert refs(cite, "115-2015-ND-CP", 6, "2016-01-01") == [("58-2014-QH13", 54)]
    assert refs(cite.replace("54", "98"), "159-2025-ND-CP", 14, "2025-07-01") == [("41-2024-QH15", 98)]


def test_self_reference_list_and_explicit_number():
    assert refs("trái quy định tại các điều 35, 36 và 37 của Bộ luật này", "45-2019-QH14", 39, "2021-01-01") == [
        ("45-2019-QH14", 35), ("45-2019-QH14", 36), ("45-2019-QH14", 37)]
    assert refs("Điều 64 Luật Bảo hiểm xã hội số 41/2024/QH15", "x", 1, "2016-01-01") == [("41-2024-QH15", 64)]


def test_reference_inside_a_described_document_resolves_at_that_documents_date():
    """NĐ 145/2020 Điều 114 bãi bỏ NĐ 41/2013 "quy định chi tiết thi hành Điều 35 của Bộ luật Lao động" — bộ luật
    2012, không có trong corpus; trước đây giải thành BLLĐ 2019 theo ngày của đoạn dẫn."""
    repeal = ("k) Nghị định số 41/2013/NĐ-CP ngày 08 tháng 5 năm 2013 của Chính phủ quy định chi tiết thi hành "
              "Điều 35 của Bộ luật Lao động về danh mục đơn vị")
    assert refs(repeal, "145-2020-ND-CP", 114, "2021-02-01") == []
    # "(sau đây được viết là …) và điểm n khoản 1 Điều 64 của Luật BHXH" là mục riêng — vẫn theo ngày đoạn dẫn.
    sep = ("Nghị định số 158/2025/NĐ-CP ngày 25 tháng 6 năm 2025 của Chính phủ quy định chi tiết (sau đây được viết "
           "là Nghị định số 158/2025/NĐ-CP) và điểm n khoản 1 Điều 64 của Luật Bảo hiểm xã hội")
    assert refs(sep, "12-2025-TT-BNV", 6, "2025-07-01") == [("41-2024-QH15", 64)]


def test_documents_outside_the_corpus_and_own_article_are_ignored():
    # Văn bản 2016 gọi "Bộ luật Lao động" là bộ luật 2012 — không có trong corpus.
    assert refs("theo Điều 35 của Bộ luật Lao động", "115-2015-ND-CP", 3, "2016-01-01") == []
    assert refs("khoản 5 Điều 328 của Bộ luật Tố tụng hình sự", "45-2019-QH14", 39, "2021-01-01") == []
    assert refs("Điều 39. Đơn phương chấm dứt … Điều 39", "45-2019-QH14", 39, "2021-01-01") == []


def test_bare_article_number_means_self_only_in_a_law():
    """Đo trên corpus: "Điều N" trơn = chính văn bản đúng 6/8 ở luật, 1/22 ở nghị định/thông tư (NĐ 145/2020
    Điều 8 "nghỉ theo Điều 112, Điều 113 …; … Điều 128 của Bộ luật Lao động" — điều của BLLĐ)."""
    bhxh = "Luật Bảo hiểm xã hội"
    assert refs("chưa đủ điều kiện theo quy định tại Điều 64", "41-2024-QH15", 98, "2025-07-01", bhxh) == [
        ("41-2024-QH15", 64)]
    # Nghị định: "Điều N" trơn lấy tên văn bản ở cuối câu, không phải chính nghị định (145/2020 cũng có Điều 21, 35).
    decree = "nghỉ theo Điều 21, Điều 35;\nthời gian tạm đình chỉ theo Điều 39 của Bộ luật Lao động. Điều 36 khác"
    assert refs(decree, "145-2020-ND-CP", 8, "2021-02-01", "Quy định chi tiết …") == [
        ("45-2019-QH14", 21), ("45-2019-QH14", 35), ("45-2019-QH14", 39)]  # Điều 36: câu sau, không tên → bỏ
    assert refs("theo khoản 2 Điều 21, quyết định tại Điều 35 Nghị định này", "145-2020-ND-CP", 8, "2021-02-01") == [
        ("145-2020-ND-CP", 21), ("145-2020-ND-CP", 35)]
    assert refs("theo Điều 21 của pháp luật về lao động", "145-2020-ND-CP", 8, "2021-02-01") == []


def test_distant_subject_is_not_guessed_as_self():
    """Câu chữ thật từng sinh cạnh sai (2026-10-19): chủ thể nêu ở câu dẫn hay phần trích, mục bên dưới chỉ ghi
    "Điều N" — BLLĐ Điều 219 sửa Luật BHXH 2014 / Bộ luật Tố tụng dân sự, biểu mẫu phụ lục của NĐ 374/2025."""
    bllđ = "Bộ luật Lao động"
    amend = ("1. Sửa đổi, bổ sung một số điều của Luật Bảo hiểm xã hội số 58/2014/QH13 đã được sửa đổi, bổ sung "
             "theo Luật số 84/2015/ QH13:\na) Sửa đổi, bổ sung Điều 35 như sau:")
    assert refs(amend, "45-2019-QH14", 219, "2021-01-01", bllđ) == []  # không phải BLLĐ Điều 35
    quoted = "a) Bổ sung khoản 1a như sau:\n“Điều 36. Tranh chấp … theo Điều 37 của Bộ luật này”"
    assert refs(quoted, "45-2019-QH14", 219, "2021-01-01", bllđ) == []
    form = "Theo đề nghị của Giám đốc Tổ chức Dịch vụ việc làm công QUYẾT ĐỊNH:\nĐiều 1. … Điều 2. … Quyết định này"
    assert refs(form, "374-2025-ND-CP", None, "2026-01-01") == []
    # Tên văn bản đứng sau cả danh sách, cách bằng ";" — trước đây là 135/2020 Điều 6.
    repeal = ("các quy định sau đây hết hiệu lực:\na) Điều 6; khoản 3 Điều 7 và khoản 2 Điều 11 của Nghị định "
              "số 115/2015/NĐ-CP ngày 11 tháng 11 năm 2015")
    assert refs(repeal, "135-2020-ND-CP", 8, "2021-01-01") == [("115-2015-ND-CP", 6)]
    assert refs("khoản 1 Điều 21 và điểm đ, e khoản 2 Điều 35 Bộ luật Lao động", "374-2025-ND-CP", 25,
                "2026-01-01") == [("45-2019-QH14", 21), ("45-2019-QH14", 35)]


def test_expand_adds_cited_article_in_force_and_skips_what_is_present():
    cited = NS(text="hưởng lương hưu theo quy định tại Điều 98 và Điều 64 của Luật Bảo hiểm xã hội",
               metadata={"doc_id": "159-2025-ND-CP", "dieu": 14, "effective_from": "2025-07-01", "clause_uid": "159__d14"})
    present = NS(text="…", metadata={"doc_id": "41-2024-QH15", "dieu": 64, "effective_from": "2025-07-01"})
    allows = lambda m: m["effective_from"] <= "2025-10-01" < m["effective_to"]
    got = LG.expand([cited, present], allows, limit=2)
    assert [(n.metadata["clause_uid"], via) for n, via in got] == [("41-2024-QH15__d98", ["159__d14"])]
    assert LG.expand([cited], lambda m: False) == []  # không version nào được phép ⇒ không thêm gì


def test_expand_prefers_specific_article_over_hub_cited_everywhere():
    """IDF: "Điều 2. Đối tượng áp dụng" bị mọi đoạn dẫn tới — một phiếu cho nó ít nghĩa hơn một phiếu cho điều hiếm."""
    hub = [node("41-2024-QH15", 98, "2025-07-01", "Luật Bảo hiểm xã hội", so_hieu="41/2024/QH15")]
    hub += [node(f"x{i}", 1, "2025-07-01", "Nghị định", text="đối tượng tại Điều 64 của Luật Bảo hiểm xã hội")
            for i in range(20)]
    hub += [node("41-2024-QH15", 64, "2025-07-01", "Luật Bảo hiểm xã hội", so_hieu="41/2024/QH15")]
    LG.build_index(hub)
    both = NS(text="theo Điều 64 và Điều 98 của Luật Bảo hiểm xã hội",
              metadata={"doc_id": "159-2025-ND-CP", "dieu": 14, "effective_from": "2025-07-01", "clause_uid": "c"})
    got = LG.expand([both], lambda m: True, limit=1)
    assert [n.metadata["clause_uid"] for n, _ in got] == ["41-2024-QH15__d98"]


def test_expansion_does_not_re_embed_when_dense_leg_hit_quota(monkeypatch):
    """Nhánh dense bị bỏ vì hết quota ⇒ mở rộng đồ thị không được embed lại (embedder ngủ chờ quota)."""
    from src.rag import retriever as R
    from src.rag.context import RetrievalContext, mark_degraded, reset_degraded, start_degraded
    from src.rag.retriever import RetrievedChunk

    from llama_index.core.schema import TextNode

    d98 = node("41-2024-QH15", 98, "2025-07-01", "Luật Bảo hiểm xã hội", so_hieu="41/2024/QH15")
    LG.build_index([TextNode(text=d98.text, metadata=d98.metadata, id_="d98")], {"d98": [1.0, 0.0]})
    # có vector ⇒ bình thường sẽ embed câu hỏi để tính cosine
    monkeypatch.setattr("src.rag.embedder.get_embedder", lambda: pytest.fail("không được embed"))
    cited = RetrievedChunk(text="theo Điều 98 của Luật Bảo hiểm xã hội", score=0.0,
                           metadata={"doc_id": "159-2025-ND-CP", "dieu": 14, "effective_from": "2025-07-01",
                                     "clause_uid": "159__d14"})
    token = start_degraded()
    try:
        mark_degraded("dense_skipped")
        got = R._graph_expansion("q", [cited], RetrievalContext(), 1)
    finally:
        reset_degraded(token)
    assert [c.metadata["clause_uid"] for c in got] == ["41-2024-QH15__d98"]
    assert got[0].metadata["graph_via"] == ["159__d14"]


def _chunk(uid, n_chars, via=None):
    from src.rag.retriever import RetrievedChunk

    meta = {"clause_uid": uid, "title": "t"}
    if via is not None:
        meta["graph_via"] = via
    return RetrievedChunk(text=uid[0] * n_chars, score=0.0, metadata=meta)


def test_budget_graph_chunk_displaces_lowest_ranked_not_itself():
    """Lần đo 2026-10-19: đoạn đồ thị nối CUỐI danh sách bị ngân sách prompt bỏ trước tiên — multi-hop k=1 không
    thêm được đoạn nào tới LLM. Giờ đoạn đồ thị được xếp chỗ trước, đoạn truy hồi hạng thấp nhất nhường chỗ."""
    from src.rag.generator import _MAX_TOTAL_CHARS, _budget

    size = _MAX_TOTAL_CHARS // 8
    retrieved = [_chunk(f"{chr(97 + i)}{i}", size) for i in range(8)]  # vừa khít ngân sách
    graph = _chunk("z98", size // 2, via=["a0"])
    kept = _budget(retrieved + [graph])
    assert [i for i, _, _ in kept] == [1, 2, 3, 4, 5, 6, 7, 9]  # hạng 8 nhường chỗ, số thứ tự gốc giữ nguyên
    assert sum(len(t) for _, _, t in kept) <= _MAX_TOTAL_CHARS
    # không có đoạn đồ thị ⇒ y như cũ
    assert [i for i, _, _ in _budget(retrieved)] == list(range(1, 9))


def test_context_tells_llm_which_passage_cites_the_graph_chunk():
    from src.rag.generator import _build_context

    ctx, _ = _build_context([_chunk("a0", 50), _chunk("b1", 50), _chunk("z98", 50, via=["b1"])])
    assert "[3] (Điều được dẫn chiếu từ đoạn [2])" in ctx
    # đoạn dẫn tới bị ngân sách bỏ ⇒ không trỏ tới số không có trong prompt
    ctx, _ = _build_context([_chunk("a0", 50), _chunk("z98", 50, via=["gone"])])
    assert "dẫn chiếu từ" not in ctx
