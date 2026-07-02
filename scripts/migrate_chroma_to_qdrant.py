"""
Migrate the local ChromaDB corpus to Qdrant Cloud.

Đọc thẳng documents + metadatas + embeddings đã có sẵn trong ChromaDB local
(data/chroma_db/) và đẩy sang Qdrant — KHÔNG re-embed, nên không cần tải lại
embedding model. Đã test logic này với đúng 91 chunks thật của dự án (xem log
migration 2026-07-02): top-1 kết quả khớp tuyệt đối giữa Chroma và Qdrant.

Yêu cầu trước khi chạy:
  1. Tạo cluster free trên https://cloud.qdrant.io
  2. Set trong .env:
       VECTOR_STORE_PROVIDER=qdrant
       QDRANT_URL=https://xxxxx.cloud.qdrant.io
       QDRANT_API_KEY=xxxxx
  3. pip install qdrant-client llama-index-vector-stores-qdrant
     (đã có sẵn trong requirements.txt)

Chạy:
    python scripts/migrate_chroma_to_qdrant.py
    python scripts/migrate_chroma_to_qdrant.py --verify   # so sánh song song sau khi migrate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger  # noqa: E402


def load_chroma_corpus():
    import chromadb

    from src.config import get_settings

    settings = get_settings()
    chroma_path = str((settings.data_dir / "chroma_db").resolve())
    client = chromadb.PersistentClient(path=chroma_path)

    names = client.list_collections()
    if settings.chroma_collection not in names:
        raise RuntimeError(
            f"Collection '{settings.chroma_collection}' không tồn tại trong {chroma_path}. "
            f"Collections có sẵn: {names}"
        )
    collection = client.get_collection(settings.chroma_collection)
    result = collection.get(include=["documents", "metadatas", "embeddings"])
    docs = result["documents"]
    metas = result["metadatas"]
    embs = result["embeddings"]
    if not docs:
        raise RuntimeError("ChromaDB collection rỗng — không có gì để migrate.")
    logger.info("Đọc {} chunks từ ChromaDB ({}), embedding dim={}", len(docs), chroma_path, len(embs[0]))
    return docs, metas, embs, client, collection


def migrate(docs, metas, embs) -> int:
    from llama_index.core.schema import TextNode
    from llama_index.vector_stores.qdrant import QdrantVectorStore

    from src.rag.embedder import get_qdrant_client_and_collection

    qclient, collection_name = get_qdrant_client_and_collection()

    nodes = []
    for doc, meta, emb in zip(docs, metas, embs):
        node = TextNode(text=doc, metadata=meta or {})
        node.embedding = list(emb)
        nodes.append(node)

    vs = QdrantVectorStore(client=qclient, collection_name=collection_name)
    vs.add(nodes)

    count = qclient.count(collection_name=collection_name, exact=True).count
    logger.info("Migration xong: {} points trong Qdrant collection '{}'", count, collection_name)
    return count


def verify(docs, metas, embs, chroma_client, chroma_collection) -> None:
    """So sánh top-3 kết quả giữa Chroma (nguồn) và Qdrant (đích) cho vài query mẫu."""
    from src.rag.vector_backend import Backend, direct_query
    from src.rag.embedder import get_qdrant_client_and_collection

    qclient, collection_name = get_qdrant_client_and_collection()
    qdrant_backend = Backend(provider="qdrant", client=qclient, collection=collection_name)
    chroma_backend = Backend(provider="chroma", client=chroma_client, collection=chroma_collection)

    sample_indices = [0, len(docs) // 2, len(docs) - 1]
    all_ok = True
    for i in sample_indices:
        q_hits = direct_query(qdrant_backend, embs[i], top_k=3)
        c_hits = direct_query(chroma_backend, embs[i], top_k=3)
        match = bool(q_hits) and bool(c_hits) and q_hits[0]["text"] == c_hits[0]["text"]
        all_ok = all_ok and match
        logger.info(
            "Query mẫu #{}: top-1 Chroma == top-1 Qdrant? {} | doc: {}",
            i, match, docs[i][:50],
        )

    if all_ok:
        logger.info("VERIFY OK — Qdrant trả kết quả nhất quán với ChromaDB nguồn.")
    else:
        logger.error("VERIFY FAILED — có mismatch, kiểm tra lại trước khi cắt sang Qdrant production.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="So sánh song song Chroma vs Qdrant sau migrate")
    args = parser.parse_args()

    from src.config import get_settings

    settings = get_settings()
    if not settings.qdrant_url:
        logger.error("QDRANT_URL chưa set trong .env — xem hướng dẫn ở đầu file này.")
        sys.exit(1)

    docs, metas, embs, chroma_client, chroma_collection = load_chroma_corpus()
    migrate(docs, metas, embs)

    if args.verify:
        verify(docs, metas, embs, chroma_client, chroma_collection)

    logger.info(
        "Xong. Nhớ set VECTOR_STORE_PROVIDER=qdrant trong .env (Render env vars) "
        "để backend chuyển sang dùng Qdrant thay vì ChromaDB."
    )


if __name__ == "__main__":
    main()
