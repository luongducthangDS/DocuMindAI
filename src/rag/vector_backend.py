"""
Provider-agnostic vector store access.

Design goal (per default-tech-stack): switching vector DB provider should be a
config change (VECTOR_STORE_PROVIDER=chroma|qdrant), not a code change scattered
across the app. Every call site that used to call ChromaDB directly should
instead call the functions here.

Verified empirically (2026-07): llama-index-vector-stores-qdrant 0.10.1 stores
each node's payload as flattened metadata + a "_node_content" key holding the
full serialized TextNode JSON (including "text"). fetch_all_chunks() and
direct_query() below parse that shape.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from loguru import logger

from src.config import get_settings


@dataclass
class Backend:
    provider: str  # "chroma" | "qdrant"
    client: object
    collection: object  # Chroma Collection object, or Qdrant collection name (str)


def get_backend() -> Backend:
    """Connect to whichever vector store VECTOR_STORE_PROVIDER points to."""
    settings = get_settings()
    provider = (settings.vector_store_provider or "chroma").lower()

    if provider == "qdrant":
        from src.rag.embedder import get_qdrant_client_and_collection

        client, collection = get_qdrant_client_and_collection()
        return Backend(provider="qdrant", client=client, collection=collection)

    from src.rag.embedder import get_chroma_collection

    client, collection = get_chroma_collection()
    return Backend(provider="chroma", client=client, collection=collection)


def make_llamaindex_vector_store(backend: Backend):
    """Return the LlamaIndex VectorStore object used to build VectorStoreIndex."""
    if backend.provider == "qdrant":
        from llama_index.vector_stores.qdrant import QdrantVectorStore

        from src.rag.embedder import get_async_qdrant_client

        return QdrantVectorStore(
            client=backend.client,
            aclient=get_async_qdrant_client(),
            collection_name=backend.collection,
        )

    from llama_index.vector_stores.chroma import ChromaVectorStore

    return ChromaVectorStore(chroma_collection=backend.collection)


def where_to_qdrant_filter(where: dict | None):
    """Translate a Chroma-style `where` dict into a Qdrant Filter.

    Only the operators RetrievalContext.to_where() actually emits are handled
    ($and / $in / $nin / $lte / $gt / $eq). Anything else raises rather than
    being dropped: this clause carries the tenant and ACL predicate, so a
    silently ignored filter is not a degraded search, it is a data leak. Failing
    loudly on an unsupported operator is the only safe default here.
    """
    if not where:
        return None
    from qdrant_client.http import models as qm

    must, must_not = [], []
    clauses = where.get("$and", [where]) if "$and" in where else [where]
    for clause in clauses:
        for key, cond in clause.items():
            if not isinstance(cond, dict):
                must.append(qm.FieldCondition(key=key, match=qm.MatchValue(value=cond)))
                continue
            for op, val in cond.items():
                if op == "$in":
                    must.append(qm.FieldCondition(key=key, match=qm.MatchAny(any=list(val))))
                elif op == "$nin":
                    must_not.append(qm.FieldCondition(key=key, match=qm.MatchAny(any=list(val))))
                elif op == "$eq":
                    must.append(qm.FieldCondition(key=key, match=qm.MatchValue(value=val)))
                elif op in ("$lte", "$lt", "$gte", "$gt"):
                    must.append(qm.FieldCondition(key=key, range=qm.Range(**{op[1:]: val})))
                else:
                    raise ValueError(
                        f"Unsupported filter operator {op!r} for Qdrant — refusing to "
                        f"run an unfiltered query with an ACL clause present."
                    )
    return qm.Filter(must=must or None, must_not=must_not or None)


def count_chunks(backend: Backend) -> int:
    if backend.provider == "qdrant":
        return backend.client.count(collection_name=backend.collection, exact=True).count
    return backend.collection.count()


def fetch_all_chunks(backend: Backend, limit: int = 10_000) -> list[tuple[str, str, dict]]:
    """Return [(id, text, metadata), ...] for every chunk in the collection.

    Used to build the BM25 corpus (BM25 needs the raw text of every chunk,
    independent of which vector DB is holding the embeddings).

    The id is the store's own id — the node_id the dense retriever returns.
    BM25 nodes must carry it: RRF fuses by node_id, so a random id makes a chunk
    found by both retrievers count as two unrelated hits (no fusion boost, and
    the duplicate eats a top-N slot).
    """
    if backend.provider == "qdrant":
        points, _ = backend.client.scroll(
            collection_name=backend.collection,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        out = []
        for p in points:
            payload = p.payload or {}
            node_content = payload.get("_node_content")
            text = ""
            if node_content:
                try:
                    text = json.loads(node_content).get("text", "")
                except (json.JSONDecodeError, TypeError):
                    text = ""
            metadata = {
                k: v
                for k, v in payload.items()
                if not k.startswith("_") and k not in ("document_id", "doc_id", "ref_doc_id")
            }
            if text:
                out.append((str(p.id), text, metadata))
        return out

    result = backend.collection.get(include=["documents", "metadatas"], limit=limit)
    ids = result.get("ids") or []
    docs = result.get("documents") or []
    metas = result.get("metadatas") or []
    return [(i, d, m or {}) for i, d, m in zip(ids, docs, metas) if d]


def direct_query(
    backend: Backend,
    query_embedding: list[float],
    top_k: int = 5,
    where: dict | None = None,
) -> list[dict]:
    """Query the vector store directly (bypassing LlamaIndex retriever).

    Used as a last-resort fallback path when the hybrid retriever is unavailable.
    Returns [{"text": ..., "metadata": ..., "score": ...}, ...].
    """
    if backend.provider == "qdrant":
        hits = backend.client.query_points(
            collection_name=backend.collection,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
            query_filter=where_to_qdrant_filter(where),
        ).points
        out = []
        for h in hits:
            payload = h.payload or {}
            node_content = payload.get("_node_content")
            text = ""
            if node_content:
                try:
                    text = json.loads(node_content).get("text", "")
                except (json.JSONDecodeError, TypeError):
                    text = ""
            metadata = {
                k: v
                for k, v in payload.items()
                if not k.startswith("_") and k not in ("document_id", "doc_id", "ref_doc_id")
            }
            if text:
                out.append({"text": text, "metadata": metadata, "score": float(h.score or 0)})
        return out

    results = backend.collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
        **({"where": where} if where else {}),
    )
    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    out = []
    for doc, metadata, distance in zip(docs, metas, distances):
        if doc:
            out.append({"text": doc, "metadata": metadata or {}, "score": 1.0 / (1.0 + float(distance or 0))})
    return out
