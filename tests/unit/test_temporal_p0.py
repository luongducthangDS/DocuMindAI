"""Q04 + Q05 (P0) — hiệu lực theo thời điểm của lương tối thiểu vùng I.

Q05: chính văn bản trong corpus (NĐ 74/2024 và NĐ 293/2025), đi qua pipeline chunk
thật (manifest → chunk_by_dieu → stamp_access_meta), rồi qua retriever thật
(dense lọc trong Chroma + BM25 lọc bằng allows()). Embedding là MockEmbedding —
không gọi Gemini, không mạng; BM25 quyết định thứ hạng.

Q04 (DATA_QC C-01): engine compliance phải chọn ngưỡng theo `as_of_date`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("no_network")

REPO = Path(__file__).resolve().parents[2]
QUERY = "mức lương tối thiểu tháng vùng I"


@pytest.fixture(scope="module")
def wage_chunks() -> list[tuple[str, str, dict]]:
    """(id, text, metadata) của 74/2024 + 293/2025, đúng như lúc ingest."""
    from scripts.ingest_documents import _chroma_safe, build_clause_chunks
    from src.ingestion.manifest import load_manifest
    from src.rag.context import stamp_access_meta

    entries = load_manifest(REPO / "docs" / "corpus" / "corpus_manifest.yaml")
    out = []
    for doc_id in ("74-2024-ND-CP", "293-2025-ND-CP"):
        for i, c in enumerate(build_clause_chunks(entries[doc_id], REPO / "data" / "raw" / "lao_dong", "2026-09-27")):
            out.append((f"{doc_id}:{i}", c.text, stamp_access_meta(_chroma_safe(c.metadata))))
    return out


@pytest.fixture
def wage_index(wage_chunks, mem_retriever):
    from llama_index.core.schema import TextNode

    return mem_retriever(TextNode(id_=i, text=t, metadata=m) for i, t, m in wage_chunks)


def _region_one_chunks(chunks):
    return [c for c in chunks if "Vùng I " in c.text]


@pytest.mark.parametrize("as_of,expected,other,so_hieu", [
    ("2025-06-01", "4.960.000", "5.310.000", "74/2024/NĐ-CP"),
    ("2026-06-01", "5.310.000", "4.960.000", "293/2025/NĐ-CP"),
])
def test_retrieval_returns_the_wage_in_force(wage_index, as_of, expected, other, so_hieu):
    from src.rag.context import RetrievalContext
    from src.rag.retriever import retrieve_with_context

    chunks = retrieve_with_context(QUERY, RetrievalContext(as_of_date=as_of))
    wage = _region_one_chunks(chunks)
    assert wage, f"không truy hồi được bảng lương tối thiểu tại {as_of}"
    assert all(expected in c.text for c in wage)
    assert not any(other in c.text for c in chunks), f"bản không còn/chưa có hiệu lực lọt vào kết quả tại {as_of}"
    assert {c.metadata["so_hieu"] for c in wage} == {so_hieu}


def test_changeover_day_is_exclusive(wage_chunks):
    """31/12/2025 còn là 74/2024; 01/01/2026 đã là 293/2025 (effective_to loại trừ)."""
    from src.rag.context import RetrievalContext

    def visible(as_of):
        return {m["so_hieu"] for _, t, m in wage_chunks
                if "Vùng I " in t and RetrievalContext(as_of_date=as_of).allows(m)}

    assert visible("2025-12-31") == {"74/2024/NĐ-CP"}
    assert visible("2026-01-01") == {"293/2025/NĐ-CP"}


# ── Q04: compliance theo thời điểm ────────────────────────────────────────────

SITUATION = "Công ty trả lương 5,0 triệu/tháng cho lao động vùng I có đúng luật không?"


@pytest.mark.usefixtures("real_criteria")
@pytest.mark.parametrize("as_of,verdict,so_hieu", [
    ("2025-03-01", "pass", "Nghị định 74/2024/NĐ-CP"),   # C-01: 5,0 ≥ 4,96
    ("2025-12-31", "pass", "Nghị định 74/2024/NĐ-CP"),
    ("2026-01-01", "fail", "Nghị định 293/2025/NĐ-CP"),  # 5,0 < 5,31
    ("2026-06-01", "fail", "Nghị định 293/2025/NĐ-CP"),
])
def test_compliance_uses_the_threshold_in_force(as_of, verdict, so_hieu):
    from src.rag.compliance import check_compliance

    result = check_compliance(SITUATION, as_of_date=as_of)
    assert result["criterion_id"] == "luong_toi_thieu_thang_vung_1"
    assert result["extracted_value"] == 5.0
    assert result["verdict"] == verdict
    assert result["citation"]["so_hieu"] == so_hieu
    assert ("4.960.000" if so_hieu.startswith("Nghị định 74") else "5.310.000") in result["explanation"]


@pytest.mark.usefixtures("real_criteria")
def test_compliance_refuses_dates_the_criteria_do_not_cover():
    """Trước 01/07/2024 corpus không có mức lương tối thiểu nào — không được phán."""
    from src.rag.compliance import check_compliance

    result = check_compliance(SITUATION, as_of_date="2024-01-01")
    assert result["verdict"] == "insufficient_info"
    assert result["citation"] == {}


@pytest.mark.usefixtures("real_criteria")
def test_every_criterion_version_cites_a_document_in_the_corpus():
    """Mở rộng TestCriteriaDataIntegrity cho các bản `versions`, và các bản không chồng khoảng."""
    import re

    from src.rag.compliance import load_criteria

    indexed = {p.stem for p in (REPO / "data" / "raw" / "lao_dong").glob("*.md")}
    for c in load_criteria():
        spans = []
        for v in (c, *c.get("versions", [])):
            number = re.search(r"\d+/\d+/[A-ZĐ0-9\-]+", v["so_hieu"]).group(0)
            assert number.replace("/", "-").replace("Đ", "D") in indexed, f"{c['id']}: {v['so_hieu']}"
            spans.append((v.get("effective_from", ""), v.get("effective_to") or "9999-12-31"))
        spans.sort()
        for (_, end), (start, _) in zip(spans, spans[1:]):
            assert end <= start, f"{c['id']}: các bản chồng khoảng hiệu lực {spans}"
