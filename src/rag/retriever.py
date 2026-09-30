"""
Hybrid retriever: dense vector (paraphrase-multilingual-MiniLM-L12-v2) + sparse
BM25, fused via RRF. Optional cross-encoder reranker for precision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from src.langfuse_otel import non_fatal, record_span, tracing_active
from src.rag.context import mark_degraded

# Module-level singletons — set by api/main.py at startup
_active_retriever = None
_active_index = None
# The two expensive legs, kept so a per-request context-aware retriever can
# reuse them instead of rebuilding: BM25 re-indexes the whole corpus and the
# cross-encoder loads a model — neither may run per query.
_bm25_retriever = None
_reranker_instance = None

# True only once a cross-encoder reranker actually loaded and is wrapping the
# active retriever. settings.enable_reranker alone isn't enough to gate the
# generator's score threshold — the reranker can be *requested* but still fail
# to load at runtime (missing model cache, OOM, Render free tier, etc.), in
# which case chunk scores stay on the raw RRF fusion scale (~0.016) and a
# 0.05 threshold calibrated for cross-encoder scores silently discards every
# chunk. generator._effective_min_score() reads this flag, not the config.
_reranker_active = False

if TYPE_CHECKING:
    from llama_index.core import VectorStoreIndex
    from llama_index.core.schema import NodeWithScore


@dataclass
class RetrievedChunk:
    text: str
    score: float
    metadata: dict
    # Cosine của nhánh dense (cùng thang cho Chroma/Qdrant, xem _as_cosine). `score`
    # là điểm RRF (~0.01–0.03) khi không có reranker — không nói được gì tuyệt đối;
    # grader cần một thang có nghĩa để quyết định có phải hỏi LLM hay không.
    # None = chunk chỉ BM25 tìm ra, hoặc đường truy hồi không đi qua dense.
    dense_score: float | None = None
    # Điểm BM25 thô (thang không chuẩn hoá) — chỉ để trace: RRF chỉ dùng thứ hạng, nên
    # điểm này không đi vào quyết định nào. None = nhánh BM25 không tìm ra chunk này.
    bm25_score: float | None = None

    @property
    def citation_label(self) -> str:
        title = self.metadata.get("title", "Không rõ")
        dieu = self.metadata.get("dieu_header", "")
        so_hieu = self.metadata.get("so_hieu", "")
        label = title
        if so_hieu:
            label = f"{so_hieu} — {title}"
        if dieu:
            label += f"\n    {dieu[:100]}"
        url = self.metadata.get("source_url", "")
        if url:
            label += f"\n    {url}"
        return label


def build_hybrid_retriever(
    index: "VectorStoreIndex",
    nodes: list,
    top_k: int = 20,
    rerank: bool = True,
):
    """
    QueryFusionRetriever: combines dense + BM25 with Reciprocal Rank Fusion.
    Falls back to vector-only if BM25 init fails (e.g. empty corpus).

    top_k=20: candidate pool before reranking. For a small corpus (~100 chunks),
    pulling 20 candidates gives the cross-encoder enough material to find the best 8.
    RRF fusion similarity_top_k must equal top_k (not top_k//2) so that both
    the dense and sparse lists contribute their full candidate sets to fusion.
    """
    from llama_index.core.retrievers import QueryFusionRetriever
    from llama_index.retrievers.bm25 import BM25Retriever

    vector_retriever = _DenseOrSkip(index.as_retriever(similarity_top_k=top_k))

    try:
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=top_k,
        )
        retrievers = [vector_retriever, bm25_retriever]
        global _bm25_retriever
        _bm25_retriever = bm25_retriever
        logger.info("Hybrid retriever ready (dense + BM25, top_k={})", top_k)
    except Exception as exc:
        logger.warning("BM25 init failed, using vector-only: {}", exc)
        retrievers = [vector_retriever]

    hybrid = QueryFusionRetriever(
        retrievers=retrievers,
        similarity_top_k=top_k,   # keep full pool — reranker will filter to top_n
        num_queries=1,             # no query expansion at retriever level
        mode="reciprocal_rerank",
        # Sync: retrieve_node gọi đường đồng bộ, nhưng use_async=True vẫn đẩy các
        # sub-retriever qua event loop dùng-một-lần. Với Qdrant điều đó nghĩa là
        # "Async client is not initialized" (chưa có aclient) rồi "Event loop is
        # closed" ở lượt sau — retriever im lặng rơi xuống fallback, câu trả lời
        # tụt chất lượng mà không báo lỗi. num_queries=1 nên chạy song song cũng
        # chẳng lợi gì: chỉ có dense + BM25, tuần tự là đủ.
        use_async=False,
    )

    if rerank:
        # top_n=8 balances precision vs recall; 5 was too aggressive for 20 candidates
        return _wrap_with_reranker(hybrid, top_n=8)
    # No cross-encoder to trim the pool (e.g. Render free tier): RRF fusion scores
    # are all clustered ~0.01-0.03 with no meaningful gap between relevant and
    # irrelevant hits, so a score threshold can't discriminate — truncate to RRF's
    # own rank order instead, matching the reranked top_n=8.
    return _TruncatedRetriever(hybrid, top_n=8)


def _wrap_with_reranker(base_retriever, top_n: int = 8):
    """
    Adds cross-encoder reranker on top of fusion retriever.
    Uses small but effective MiniLM model — runs locally, no API.
    """
    global _reranker_active
    # SentenceTransformerRerank không nhận cache_folder (chỉ có model/top_n/device/
    # trust_remote_code), nên ở đây buộc phải đi qua biến môi trường — khác với
    # embedder, nơi cache được truyền thẳng làm tham số.
    # Env được giữ nguyên sau khi load (khối finally cũ khôi phục lại HF_HOME trỏ về
    # G:\ đã bị gỡ): trả về một đường dẫn offline chỉ tạo ra lỗi khó hiểu cho lần
    # load model kế tiếp, không bảo vệ được gì.
    from src.hf_env import resolve_cached_model, use_local_hf_cache

    use_local_hf_cache(offline=True)

    try:
        from llama_index.core.postprocessor import SentenceTransformerRerank
        from src.config import get_settings

        model_name = get_settings().reranker_model
        # Truyền đường dẫn cache đã resolve thay cho repo id khi model có sẵn: nếu chỉ
        # đưa repo id, thư viện tra cache theo HF_HOME đã đóng băng lúc import (ổ G:\
        # offline trên máy dev) và coi như model chưa tải, dù nó nằm trong repo.
        model_ref = resolve_cached_model(model_name) or model_name
        reranker = SentenceTransformerRerank(
            model=model_ref,
            top_n=top_n,
        )
        logger.info("Cross-encoder reranker loaded: {} (top_n={})", model_name, top_n)
        _reranker_active = True
        global _reranker_instance
        _reranker_instance = reranker
        return _RerankedRetriever(base_retriever, reranker)

    except Exception as exc:
        logger.warning("Reranker unavailable, skipping: {}", exc)
        _reranker_active = False
        # Same reasoning as the rerank=False branch in build_hybrid_retriever:
        # without a cross-encoder, RRF fusion scores (~0.01-0.03) have no
        # meaningful gap to threshold on, so truncate to rank order instead
        # of returning the full top_k=20 pool.
        return _TruncatedRetriever(base_retriever, top_n=top_n)


class _RerankedRetriever:
    """Thin wrapper: retrieve from hybrid, then rerank."""

    def __init__(self, base_retriever, reranker):
        self._base = base_retriever
        self._reranker = reranker

    async def aretrieve(self, query: str) -> list["NodeWithScore"]:
        nodes = await self._base.aretrieve(query)
        from llama_index.core.schema import QueryBundle

        return self._reranker.postprocess_nodes(nodes, QueryBundle(query_str=query))

    def retrieve(self, query: str) -> list["NodeWithScore"]:
        nodes = self._base.retrieve(query)
        from llama_index.core.schema import QueryBundle

        return self._reranker.postprocess_nodes(nodes, QueryBundle(query_str=query))


class _TruncatedRetriever:
    """Thin wrapper: keep only the top-N of the RRF-fused pool (no cross-encoder
    available to trim by relevance score, so rely on RRF's own rank order)."""

    def __init__(self, base_retriever, top_n: int = 8):
        self._base = base_retriever
        self._top_n = top_n

    async def aretrieve(self, query: str) -> list["NodeWithScore"]:
        nodes = await self._base.aretrieve(query)
        return nodes[: self._top_n]

    def retrieve(self, query: str) -> list["NodeWithScore"]:
        nodes = self._base.retrieve(query)
        return nodes[: self._top_n]


def chunk_ref(meta: dict) -> dict:
    """Metadata chunk → định danh đủ để tra ngược (văn bản, điều, khoản, hiệu lực),
    KHÔNG kèm nội dung: trace giữ nhẹ, nội dung tra lại được từ `id` (version_id)."""
    from src.rag.generator import chunk_origin  # lazy: generator import module này

    ref = {"doc": meta.get("so_hieu") or meta.get("title") or "?"}
    if meta.get("dieu") not in (None, ""):
        ref["article"] = f"Điều {meta['dieu']}"
    elif meta.get("dieu_header"):
        ref["article"] = str(meta["dieu_header"])[:80]
    for key, name in (("khoan", "clause"), ("effective_from", "from"),
                      ("effective_to", "to"), ("version_id", "id")):
        if meta.get(key) not in (None, ""):
            ref[name] = meta[key]
    if chunk_origin(meta) != "official":
        ref["origin"] = chunk_origin(meta)  # tài liệu người dùng tải lên, không phải luật
    return ref


@non_fatal
def trace_retrieval(query: str, chunks: list[RetrievedChunk], path: str, ctx, started_at) -> None:
    """Span retrieve-documents: hạng, điểm, định danh từng chunk — đủ để trả lời "lỗi ở
    retrieval hay generation, có lấy phải bản hết hiệu lực không" chỉ từ trace.

    `path`: "hybrid" (dense + BM25), "direct" (hybrid rỗng → hỏi thẳng vector store),
    "error_fallback:<Lỗi>" (hybrid ném lỗi). `cosine` (dense) và `bm25` (điểm BM25 thô)
    chỉ có khi nhánh tương ứng tìm ra chunk đó — thiếu một trong hai là chỉ nhánh kia tìm ra."""
    from datetime import datetime, timezone

    if not tracing_active():
        return
    record_span("retrieve-documents", "retriever", query, {
        "path": path,
        "as_of": ctx.effective_as_of if ctx is not None else None,
        "tenant": ctx.tenant_id if ctx is not None else None,
        "score": "vector" if path != "hybrid" else ("rerank" if _reranker_active else "rrf"),
        "chunks": [
            {"rank": i, "score": round(c.score, 4),
             **({"cosine": round(c.dense_score, 3)} if c.dense_score is not None else {}),
             **({"bm25": round(c.bm25_score, 2)} if c.bm25_score is not None else {}),
             **chunk_ref(c.metadata)}
            for i, c in enumerate(chunks, 1)
        ],
    }, started_at, datetime.now(timezone.utc))


def nodes_to_chunks(nodes: list["NodeWithScore"]) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            text=n.node.text or n.node.get_content(),
            score=float(n.score or 0),
            metadata={k: v for k, v in (n.node.metadata or {}).items()
                      if not k.startswith("_")},
        )
        for n in nodes
    ]


def retrieve_direct_chroma(query: str, top_k: int = 5, ctx=None) -> list[RetrievedChunk]:
    """Fallback retrieval path that queries the vector store directly.

    Name kept as "_chroma" for backward compat with existing call sites
    (query.py, agent/graph.py) — but works against whichever provider
    VECTOR_STORE_PROVIDER points to (chroma or qdrant), via vector_backend.
    """
    try:
        from src.rag.embedder import get_embedder
        from src.rag.vector_backend import get_backend, count_chunks, direct_query

        backend = get_backend()
        count = count_chunks(backend)
        if count == 0:
            logger.error("Direct fallback ({}) found empty collection", backend.provider)
            return []

        embedder = get_embedder()
        query_embedding = embedder.get_query_embedding(query)
        where = ctx.to_where() if ctx is not None else None
        results = direct_query(backend, query_embedding, top_k=top_k, where=where)

        chunks = [
            RetrievedChunk(text=r["text"], score=r["score"], metadata=r["metadata"])
            for r in results
        ]
        logger.info("Direct fallback ({}) returned {} chunks", backend.provider, len(chunks))
        return chunks
    except Exception as exc:
        logger.error("Direct fallback failed: {}", exc)
        return []


class _DenseOrSkip:
    """Nhánh dense trả rỗng khi câu hỏi chưa embed được ngay (quota Gemini).

    Server API bật fail_fast_queries nên embedder ném EmbeddingUnavailable thay
    vì ngủ chờ quota. Khi đó RRF chỉ còn BM25: câu trả lời kém hơn một chút nhưng
    về trong vài giây, thay vì treo request tới vài phút. Lỗi khác vẫn ném ra như cũ.
    """

    def __init__(self, base_retriever):
        self._base = base_retriever
        # node_id -> điểm thô của nhánh dense ở lần gọi gần nhất; RRF ghi đè điểm này
        # trên node sau fusion nên phải giữ lại từ đây.
        self.scores: dict[str, float] = {}

    def _remember(self, nodes: list) -> list:
        self.scores = {n.node.node_id: float(n.score or 0) for n in nodes}
        return nodes

    def retrieve(self, query) -> list["NodeWithScore"]:
        from src.rag.embedder import EmbeddingUnavailable

        try:
            return self._remember(self._base.retrieve(query))
        except EmbeddingUnavailable as exc:
            logger.warning("Dense leg skipped (embedding unavailable), BM25 only: {}", exc)
            mark_degraded("dense_skipped")
            return []

    async def aretrieve(self, query) -> list["NodeWithScore"]:
        from src.rag.embedder import EmbeddingUnavailable

        try:
            return self._remember(await self._base.aretrieve(query))
        except EmbeddingUnavailable as exc:
            logger.warning("Dense leg skipped (embedding unavailable), BM25 only: {}", exc)
            mark_degraded("dense_skipped")
            return []


class _ScreenedRetriever:
    """Drops chunks the caller may not see, before they reach fusion.

    BM25 keeps one in-process index over the entire corpus and has no notion of
    a filter, so this is where its hits get screened. Screening *before* RRF
    rather than after matters twice over: a forbidden chunk never leaves the
    retriever, and fusion ranks are computed over the visible set instead of
    being skewed by neighbours the caller is not allowed to read.
    """

    def __init__(self, base_retriever, ctx):
        self._base = base_retriever
        self._ctx = ctx
        # node_id -> điểm thô của nhánh này; RRF ghi đè điểm trên node sau fusion
        # (như _DenseOrSkip.scores). Tạo mới mỗi request nên không dùng chung giữa các thread.
        self.scores: dict[str, float] = {}

    def _screen(self, nodes: list) -> list:
        visible = [n for n in nodes if self._ctx.allows(n.node.metadata or {})]
        self.scores = {n.node.node_id: float(n.score or 0) for n in visible}
        return visible

    def retrieve(self, query) -> list["NodeWithScore"]:
        return self._screen(self._base.retrieve(query))

    async def aretrieve(self, query) -> list["NodeWithScore"]:
        return self._screen(await self._base.aretrieve(query))


def retrieve_with_context(query: str, ctx, top_k: int = 20, top_n: int = 8) -> list[RetrievedChunk]:
    """Access-controlled, tenant-scoped, point-in-time retrieval.

    The dense leg pre-filters inside the vector store; the sparse leg is
    screened on the way out (see _ScreenedRetriever). Both legs therefore spend
    their whole candidate budget on chunks the caller may actually read — which
    is the difference from filtering afterwards, where the top-8 was shared with
    clauses that were never in force (reports/temporal_eval.json: avg_chunks 2.87).

    Falls back to the unfiltered singleton retriever only if no index is active,
    and screens that result too — an empty index must not mean an open door.
    """
    from llama_index.core.llms.mock import MockLLM
    from llama_index.core.retrievers import QueryFusionRetriever
    from llama_index.core.schema import QueryBundle

    index = _active_index
    if index is None:
        logger.warning("No active index — context retrieval falling back to direct query")
        chunks = retrieve_direct_chroma(query, top_k=top_n, ctx=ctx)
        return [c for c in chunks if ctx.allows(c.metadata)]

    dense = _DenseOrSkip(index.as_retriever(similarity_top_k=top_k, filters=ctx.to_llama_filters()))

    retrievers = [dense]
    sparse = _ScreenedRetriever(_bm25_retriever, ctx) if _bm25_retriever is not None else None
    if sparse is not None:
        retrievers.append(sparse)

    if len(retrievers) == 1:
        nodes = dense.retrieve(query)
    else:
        nodes = QueryFusionRetriever(
            retrievers=retrievers,
            similarity_top_k=top_k,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=False,  # same reason as build_hybrid_retriever
            # num_queries=1 never calls an LLM, but the constructor still resolves
            # one from the global Settings and falls back to OpenAI when none is
            # set. Passing MockLLM removes the hidden dependency on main.py having
            # configured Settings first (it made this path fail in isolation).
            llm=MockLLM(),
        ).retrieve(query)

    # Belt and braces: the dense leg was filtered by the store and the sparse leg
    # by _ScreenedRetriever, so this should never drop anything. It stays because
    # an ACL that only holds when every upstream layer behaves is not an ACL.
    visible = [n for n in nodes if ctx.allows(n.node.metadata or {})]
    if len(visible) != len(nodes):
        logger.error("Context screening caught {} chunk(s) that upstream filters missed",
                     len(nodes) - len(visible))

    if _reranker_instance is not None and visible:
        visible = _reranker_instance.postprocess_nodes(visible, QueryBundle(query_str=query))
    else:
        visible = visible[:top_n]

    chunks = nodes_to_chunks(visible)
    from src.config import get_settings

    provider = (get_settings().vector_store_provider or "chroma").lower()
    for chunk, node in zip(chunks, visible):
        raw = dense.scores.get(node.node.node_id)
        chunk.dense_score = _as_cosine(raw, provider) if raw is not None else None
        chunk.bm25_score = sparse.scores.get(node.node.node_id) if sparse is not None else None
    return chunks


def _as_cosine(raw: float, provider: str) -> float:
    """Điểm thô của nhánh dense → cosine, cùng một thang cho mọi provider.

    llama-index ChromaVectorStore 0.5.x trả exp(-distance), distance = 1 - cos
    (collection hnsw:space=cosine); Qdrant trả thẳng cosine. Ngưỡng của grader đặt
    trên cosine — không quy đổi thì cùng một câu hỏi qua hai provider rơi vào hai
    nhánh grade khác nhau (test_llm_budget đo trên Chroma thật).
    """
    if provider == "chroma":
        return 1.0 + math.log(raw) if raw > 0 else -1.0
    return raw
