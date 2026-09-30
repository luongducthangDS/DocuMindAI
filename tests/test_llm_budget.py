"""P1.4: ngân sách lời gọi LLM mỗi câu hỏi.

Trước: câu hỏi lao động thường gặp tốn router + grade + answer (3 lời gọi generate),
câu bị grade "không liên quan" thêm reformulate + grade lần 2 (5 lời gọi). Mục tiêu:
câu hỏi rõ ràng, lượt đầu, có dấu → chỉ 1 lời gọi generate (answer) + 1 embed.
"""

from __future__ import annotations

import math
import uuid

import pytest

from src.agent import graph
from src.rag import grader
from src.rag.retriever import RetrievedChunk


def _forbid_llm(monkeypatch) -> list[str]:
    """Ghi lại mọi lời gọi Gemini; test tự quyết bao nhiêu lời gọi là hợp lệ."""
    calls: list[str] = []

    def fake_call(api_key, model_name, prompt, **_kw):
        calls.append(prompt)
        return type("R", (), {"text": "câu hỏi đã chuẩn hoá", "usage_metadata": None})()

    monkeypatch.setattr("src.rag.generator.gemini_call", fake_call)
    monkeypatch.setattr("src.rag.generator._gemini_pairs", lambda *_a, **_k: [("k", "m")])
    return calls


# ── 1. Thêm dấu + diễn giải theo ngữ cảnh: tối đa MỘT lời gọi ─────────────────

class TestNormalizeQuery:
    def test_first_turn_with_diacritics_calls_nothing(self, monkeypatch):
        calls = _forbid_llm(monkeypatch)
        q = "Mức lương tối thiểu vùng I là bao nhiêu?"
        assert graph._normalize_query(q, []) == q
        assert calls == []

    def test_first_turn_without_diacritics_calls_once(self, monkeypatch):
        calls = _forbid_llm(monkeypatch)
        graph._normalize_query("muc luong toi thieu vung mot", [])
        assert len(calls) == 1

    def test_follow_up_without_diacritics_calls_once_not_twice(self, monkeypatch):
        calls = _forbid_llm(monkeypatch)
        history = [{"role": "user", "content": "Lương tối thiểu vùng I?"},
                   {"role": "assistant", "content": "5.310.000 đồng [1]"}]
        graph._normalize_query("con vung hai thi sao", history)
        assert len(calls) == 1
        assert "dấu" in calls[0]  # một prompt làm cả hai việc

    def test_contextualize_prompt_also_restores_diacritics(self):
        assert "dấu" in graph._CONTEXTUALIZE_PROMPT


# ── 2. Router: câu có thuật ngữ lao động rõ ràng không cần LLM ────────────────

class TestRouterFastPath:
    @pytest.mark.parametrize("q", [
        "Người lao động làm việc ở vùng I thì mức lương tối thiểu tháng hiện nay là bao nhiêu tiền?",
        "Thời gian thử việc tối đa đối với công việc cần trình độ cao đẳng trở lên là bao lâu vậy?",
        "Theo quy định hiện hành thì bảo hiểm xã hội bắt buộc người lao động phải đóng bao nhiêu phần trăm?",
    ])
    def test_long_legal_question_skips_router_llm(self, monkeypatch, q):
        calls = _forbid_llm(monkeypatch)
        out = graph.router_node({"query": q, "steps": []})
        assert out["intent"] == "simple_qa"
        assert calls == []

    def test_ambiguous_long_question_still_asks_llm(self, monkeypatch):
        calls = _forbid_llm(monkeypatch)
        graph.router_node({"query": "Tôi muốn biết thêm về mấy chuyện hôm trước mình đã nói với nhau nhé", "steps": []})
        assert len(calls) == 1

    def test_legal_question_with_number_still_asks_router(self, monkeypatch):
        """Có chữ số → có thể là kiểm định tuân thủ (✅/❌) — để router LLM quyết."""
        calls = _forbid_llm(monkeypatch)
        graph.router_node({"query": "Mức đóng bảo hiểm xã hội tự nguyện là 25% thu nhập làm căn cứ đóng, có đúng không?", "steps": []})
        assert len(calls) == 1

    def test_keyword_intents_unchanged(self):
        assert graph._keyword_classify("So sánh Luật BHXH 2014 và 2024") == "compare"


# ── 3. Grade chỉ gọi LLM khi điểm dense nằm trong vùng mơ hồ ──────────────────

def _chunks(cos: float | None) -> list[RetrievedChunk]:
    return [RetrievedChunk(text="Điều 90. Tiền lương", score=0.016, metadata={}, dense_score=cos)]


@pytest.fixture
def no_reranker(monkeypatch):
    monkeypatch.setattr("src.rag.retriever._reranker_active", False)


class TestConditionalGrade:
    def test_confident_dense_score_skips_llm(self, monkeypatch, no_reranker):
        calls = _forbid_llm(monkeypatch)
        out = grader.grade_chunks("q", _chunks(grader.CONFIDENT_COSINE + 0.01))
        assert out["relevant"] is True
        assert calls == []

    def test_hopeless_dense_score_skips_llm(self, monkeypatch, no_reranker):
        calls = _forbid_llm(monkeypatch)
        out = grader.grade_chunks("q", _chunks(grader.HOPELESS_COSINE - 0.01))
        assert out["relevant"] is False
        assert calls == []

    def test_ambiguous_dense_score_asks_llm(self, monkeypatch, no_reranker):
        calls = _forbid_llm(monkeypatch)
        mid = (grader.CONFIDENT_COSINE + grader.HOPELESS_COSINE) / 2
        grader.grade_chunks("q", _chunks(mid))
        assert len(calls) == 1

    def test_unknown_dense_score_asks_llm(self, monkeypatch, no_reranker):
        """Chunk không có điểm dense (vd. chỉ BM25 tìm ra) → không đoán, hỏi LLM."""
        calls = _forbid_llm(monkeypatch)
        grader.grade_chunks("q", _chunks(None))
        assert len(calls) == 1

    def test_thresholds_are_ordered(self):
        assert -1.0 < grader.HOPELESS_COSINE < grader.CONFIDENT_COSINE < 1.0


# ── 4. Điểm dense quy về cosine, cùng thang cho Chroma lẫn Qdrant ─────────────

def test_chroma_score_converts_back_to_true_cosine():
    """llama-index Chroma trả exp(-(1-cos)) — đo trên Chroma thật, không tin công thức."""
    import chromadb
    from llama_index.core.vector_stores.types import VectorStoreQuery
    from llama_index.vector_stores.chroma import ChromaVectorStore

    from src.rag.retriever import _as_cosine

    # Settings mặc định: Chroma chỉ giữ MỘT instance ephemeral mỗi process (xem conftest).
    col = chromadb.EphemeralClient().create_collection(
        f"cos_{uuid.uuid4().hex[:8]}", metadata={"hnsw:space": "cosine"})
    a, q = [1.0, 0.0, 0.0], [0.6, 0.8, 0.0]
    col.add(ids=["a"], embeddings=[a], documents=["x"], metadatas=[{"k": "v"}])
    res = ChromaVectorStore(chroma_collection=col).query(
        VectorStoreQuery(query_embedding=q, similarity_top_k=1))
    assert math.isclose(_as_cosine(res.similarities[0], "chroma"), 0.6, abs_tol=1e-4)
    assert _as_cosine(0.6, "qdrant") == 0.6


def test_retrieve_with_context_attaches_dense_score(mem_retriever):
    from llama_index.core.schema import TextNode

    from src.rag.context import PUBLIC_CONTEXT, stamp_access_meta
    from src.rag.retriever import retrieve_with_context

    mem_retriever([TextNode(text="Điều 90. Tiền lương là số tiền người sử dụng lao động trả",
                            metadata=stamp_access_meta({"title": "BLLĐ"}))])
    chunks = retrieve_with_context("tiền lương", PUBLIC_CONTEXT)
    assert chunks and chunks[0].dense_score == pytest.approx(1.0, abs=1e-6)  # MockEmbedding: cùng vector


# ── 5. Đầu-cuối: câu rõ ràng, lượt đầu → đúng 1 lời gọi generate ──────────────

async def test_clear_first_turn_question_costs_one_generate_call(fake_gemini, mem_retriever, monkeypatch):
    from llama_index.core.schema import TextNode

    from src.rag.context import stamp_access_meta

    monkeypatch.setattr("src.rag.retriever._reranker_active", False)
    mem_retriever([TextNode(
        text="Điều 90. Tiền lương là số tiền mà người sử dụng lao động trả cho người lao động",
        metadata=stamp_access_meta({"title": "Bộ luật Lao động", "dieu_header": "Điều 90"}),
    )])
    fake_gemini.responder = lambda p, m: "Tiền lương là số tiền người sử dụng lao động trả [1]."
    result = await graph.run_agent(
        "Theo quy định của Bộ luật Lao động thì tiền lương được hiểu như thế nào vậy?",
        session_id=f"b-{uuid.uuid4().hex[:6]}",
    )
    assert len(fake_gemini.calls) == 1, [c.prompt[:60] for c in fake_gemini.calls]
    assert result["intent"] == "simple_qa"
