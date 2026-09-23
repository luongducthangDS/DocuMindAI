"""
retrieve_with_context and the tenant-facing endpoints.

The retriever tests swap the module singletons for fakes that record what they
were asked. That checks the two properties the design rests on without needing
an index: the dense leg receives the context's filters, and a forbidden BM25 hit
never survives — not into the result, not into the reranker's input.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from llama_index.core.schema import NodeWithScore, TextNode

import src.rag.retriever as r_module
from src.rag.context import RetrievalContext, stamp_access_meta


def node(node_id: str, score: float = 1.0, **meta) -> NodeWithScore:
    base = {"tenant_id": "public", "acl_label": "public",
            "effective_from": "2021-01-01", "effective_to": "9999-12-31",
            "status": "in_force", "clause_uid": node_id}
    base.update(meta)
    m = stamp_access_meta(base, tenant_id=base["tenant_id"], acl_label=base["acl_label"])
    return NodeWithScore(node=TextNode(id_=node_id, text=f"text {node_id}", metadata=m), score=score)


class FakeRetriever:
    def __init__(self, nodes):
        self.nodes = nodes

    def retrieve(self, query):
        return list(self.nodes)


class FakeIndex:
    """Records the filters the dense leg was built with."""

    def __init__(self, nodes):
        self.nodes = nodes
        self.filters_seen = []

    def as_retriever(self, similarity_top_k, filters=None):
        self.filters_seen.append(filters)
        return FakeRetriever(self.nodes)


class RecordingReranker:
    def __init__(self):
        self.received = None

    def postprocess_nodes(self, nodes, query_bundle):
        self.received = [n.node.id_ for n in nodes]
        return nodes


@pytest.fixture
def singletons(monkeypatch):
    def install(dense_nodes, bm25_nodes=None, reranker=None):
        index = FakeIndex(dense_nodes)
        monkeypatch.setattr(r_module, "_active_index", index)
        monkeypatch.setattr(r_module, "_bm25_retriever",
                            FakeRetriever(bm25_nodes) if bm25_nodes is not None else None)
        monkeypatch.setattr(r_module, "_reranker_instance", reranker)
        return index
    return install


class TestRetrieveWithContext:
    def test_dense_leg_receives_the_context_filters(self, singletons):
        index = singletons([node("a")])
        r_module.retrieve_with_context("q", RetrievalContext(tenant_id="acme"))
        keys = {f.key for f in index.filters_seen[0].filters}
        assert keys == {"tenant_id", "acl_label", "effective_from_i", "effective_to_i", "status"}

    def test_forbidden_bm25_hit_never_reaches_result_or_reranker(self, singletons):
        reranker = RecordingReranker()
        singletons(
            dense_nodes=[node("public_ok")],
            bm25_nodes=[node("public_ok"), node("globex_secret", tenant_id="globex")],
            reranker=reranker,
        )
        chunks = r_module.retrieve_with_context("q", RetrievalContext(tenant_id="acme"))
        uids = {c.metadata["clause_uid"] for c in chunks}
        assert "globex_secret" not in uids
        assert "globex_secret" not in reranker.received

    def test_superseded_bm25_hit_is_screened_by_date(self, singletons):
        singletons(
            dense_nodes=[],
            bm25_nodes=[node("old", effective_from="2015-01-01", effective_to="2026-01-01"),
                        node("new", effective_from="2026-01-01")],
        )
        got = r_module.retrieve_with_context("q", RetrievalContext(as_of_date="2026-06-01"))
        assert [c.metadata["clause_uid"] for c in got] == ["new"]

    def test_leak_from_dense_leg_is_caught_by_final_screen(self, singletons):
        """If the store ever ignored a filter, the last screen still holds."""
        singletons(dense_nodes=[node("ok"), node("leaked", tenant_id="globex")])
        got = r_module.retrieve_with_context("q", RetrievalContext(tenant_id="acme"))
        assert [c.metadata["clause_uid"] for c in got] == ["ok"]

    def test_without_reranker_truncates_to_top_n(self, singletons):
        singletons(dense_nodes=[node(f"n{i}", score=1 - i / 100) for i in range(12)])
        got = r_module.retrieve_with_context("q", RetrievalContext(), top_n=8)
        assert len(got) == 8

    def test_embedding_quota_falls_back_to_bm25_without_sleeping(self, monkeypatch):
        """Hết quota embed: server không được ngủ chờ (từng làm 1 câu hỏi mất 231s),
        mà phải trả kết quả BM25 ngay. Dùng VectorStoreIndex thật để chắc chắn
        llama_index không bọc EmbeddingUnavailable thành lỗi khác."""
        import time as time_mod
        from unittest.mock import patch

        from llama_index.core import VectorStoreIndex

        from src.rag.embedder import _GeminiAPIEmbedding

        embedder = _GeminiAPIEmbedding(model_name="gemini-embedding-001", api_keys=["k1", "k2"])
        embedder.fail_fast_queries = True
        for key in ("k1", "k2"):
            embedder._cooldown[key] = time_mod.monotonic() + 65  # vướng quota phút

        monkeypatch.setattr(r_module, "_active_index", VectorStoreIndex(nodes=[], embed_model=embedder))
        monkeypatch.setattr(r_module, "_bm25_retriever", FakeRetriever([node("bm25_hit")]))
        monkeypatch.setattr(r_module, "_reranker_instance", None)

        with patch("src.rag.embedder.time.sleep") as slept:
            got = r_module.retrieve_with_context("q", RetrievalContext())
        slept.assert_not_called()
        assert [c.metadata["clause_uid"] for c in got] == ["bm25_hit"]

    def test_no_index_falls_back_but_still_screens(self, monkeypatch):
        from src.rag.retriever import RetrievedChunk

        monkeypatch.setattr(r_module, "_active_index", None)
        leaked = node("leaked", tenant_id="globex").node.metadata
        seen_ctx = {}

        def fake_direct(query, top_k=5, ctx=None):
            seen_ctx["ctx"] = ctx
            return [RetrievedChunk(text="x", score=1.0, metadata=leaked)]

        monkeypatch.setattr(r_module, "retrieve_direct_chroma", fake_direct)
        ctx = RetrievalContext(tenant_id="acme")
        assert r_module.retrieve_with_context("q", ctx) == []
        assert seen_ctx["ctx"] is ctx


# ── Endpoints ─────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    from src.api.routes import documents, query

    app = FastAPI()
    app.include_router(query.router)
    app.include_router(documents.router)
    return TestClient(app)


class TestWhoami:
    def test_anonymous(self, client):
        body = client.get("/api/v1/whoami").json()
        assert body == {"tenant_id": "public", "acl_labels": ["public"], "authenticated": False}

    def test_known_key(self, client):
        body = client.get("/api/v1/whoami", headers={"x-api-key": "demo-acme-0000"}).json()
        assert body["tenant_id"] == "acme"
        assert body["authenticated"] is True

    def test_bad_key_is_401(self, client):
        assert client.get("/api/v1/whoami", headers={"x-api-key": "nope"}).status_code == 401


class TestDocumentListingIsTenantScoped:
    @pytest.fixture(autouse=True)
    def registry(self, monkeypatch):
        from src.api.routes import documents
        from src.api.schemas import DocumentMeta

        def meta(i, title):
            return DocumentMeta(id=i, title=title, doc_type="pdf", source="user_upload")

        monkeypatch.setattr(documents, "_doc_registry", {
            "1": meta("1", "Public doc"),
            "2": meta("2", "Acme plan"),
            "3": meta("3", "Globex layoffs"),
        })
        monkeypatch.setattr(documents, "_doc_tenant", {"1": "public", "2": "acme", "3": "globex"})

    def titles(self, client, key=None):
        headers = {"x-api-key": key} if key else {}
        return {d["title"] for d in client.get("/api/v1/documents", headers=headers).json()["documents"]}

    def test_anonymous_sees_public_only(self, client):
        assert self.titles(client) == {"Public doc"}

    def test_tenant_sees_public_and_own_not_others(self, client):
        assert self.titles(client, "demo-acme-0000") == {"Public doc", "Acme plan"}
