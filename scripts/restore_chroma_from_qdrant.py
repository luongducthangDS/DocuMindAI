"""
Khôi phục ChromaDB local từ Qdrant Cloud — chiều ngược của migrate_chroma_to_qdrant.py.

Vì sao cần: chạy repo bằng Python global (chromadb 1.5.9 thay vì pin 0.6.3) làm
chromadb migrate store tại chỗ — sysdb mất config_json_str và index_metadata.pickle
của HNSW bị ghi lại thành dict, nên 0.6.3 không mở nổi segment vector. Qdrant Cloud
vẫn giữ đủ 1149 chunk kèm vector 3072-dim, nên khôi phục được mà KHÔNG re-embed:
script đọc vector + text + metadata từ Qdrant rồi ghi thẳng vào Chroma.

Nguồn metadata là `_node_content.metadata` chứ không phải payload top-level: khi
llama-index ghi sang Qdrant nó đè `doc_id`/`document_id`/`ref_doc_id` ở payload
thành chuỗi "None". Id chunk lấy từ `version_id` — đúng id mà ingest đặt trong Chroma.

Phòng tái phát: luôn chạy bằng .venv của repo (./.venv/Scripts/python.exe).

Chạy:
    python scripts/restore_chroma_from_qdrant.py --dry-run
    python scripts/restore_chroma_from_qdrant.py --yes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from loguru import logger  # noqa: E402

_BATCH = 100


def load_from_qdrant() -> tuple[list[str], list[str], list[dict], list[list[float]]]:
    """Đọc toàn bộ điểm từ Qdrant: (ids, texts, metadatas, embeddings)."""
    from qdrant_client import QdrantClient

    from src.config import get_settings

    # Client riêng thay vì get_qdrant_client_and_collection(): hàm đó timeout mặc
    # định 5s (scroll kèm vector 3072-dim vượt xa) và còn gọi Gemini để đo chiều
    # vector. Job này chỉ đọc, không cần tạo collection.
    settings = get_settings()
    if not settings.qdrant_url:
        raise RuntimeError("QDRANT_URL chưa được set trong .env")
    qclient = QdrantClient(
        url=settings.qdrant_url, api_key=settings.qdrant_api_key or None, timeout=120
    )
    collection_name = settings.qdrant_collection
    total = qclient.count(collection_name=collection_name, exact=True).count
    logger.info("Qdrant '{}' có {} points", collection_name, total)

    ids: list[str] = []
    texts: list[str] = []
    metas: list[dict] = []
    embs: list[list[float]] = []
    offset = None
    while True:
        points, offset = qclient.scroll(
            collection_name=collection_name,
            limit=64,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        for p in points:
            payload = p.payload or {}
            node = json.loads(payload.get("_node_content") or "{}")
            text = node.get("text") or ""
            metadata = node.get("metadata") or {}
            if not text:
                logger.warning("Bỏ qua point {} vì không có text", p.id)
                continue
            vector = p.vector
            if isinstance(vector, dict):  # named vectors
                vector = next(iter(vector.values()))
            ids.append(metadata.get("version_id") or str(p.id))
            texts.append(text)
            metas.append(metadata)
            embs.append(list(vector))
        logger.info("Đã đọc {}/{} chunk", len(ids), total)
        if offset is None:
            break

    if len(set(ids)) != len(ids):
        raise RuntimeError(f"Id trùng nhau: {len(ids)} chunk nhưng chỉ {len(set(ids))} id duy nhất.")
    if len(ids) != total:
        logger.warning("Đọc được {} chunk trong khi Qdrant báo {} points", len(ids), total)
    return ids, texts, metas, embs


def write_to_chroma(ids, texts, metas, embs) -> int:
    """Xoá collection Chroma hỏng rồi ghi lại từ dữ liệu Qdrant."""
    import chromadb

    from src.config import get_settings

    settings = get_settings()
    chroma_path = str((settings.data_dir / "chroma_db").resolve())
    client = chromadb.PersistentClient(path=chroma_path)
    name = settings.chroma_collection

    existing = [c if isinstance(c, str) else c.name for c in client.list_collections()]
    if name in existing:
        client.delete_collection(name)
        logger.info("Đã xoá collection cũ '{}'", name)

    collection = client.create_collection(
        name=name,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": settings.embedding_model,
            "embedding_dim": len(embs[0]),
        },
    )
    for i in range(0, len(ids), _BATCH):
        sl = slice(i, i + _BATCH)
        collection.add(
            ids=ids[sl],
            documents=texts[sl],
            metadatas=[{k: v for k, v in m.items() if v is not None} for m in metas[sl]],
            embeddings=embs[sl],
        )
        logger.info("Ghi {}/{} chunk", min(i + _BATCH, len(ids)), len(ids))

    count = collection.count()
    logger.info("Chroma '{}' giờ có {} chunk tại {}", name, count, chroma_path)
    return count


def verify(texts, embs) -> None:
    """Query lại Chroma bằng chính vector nguồn — top-1 phải là đúng chunk đó."""
    import chromadb

    from src.config import get_settings
    from src.rag.vector_backend import Backend, direct_query

    settings = get_settings()
    client = chromadb.PersistentClient(path=str((settings.data_dir / "chroma_db").resolve()))
    backend = Backend(
        provider="chroma",
        client=client,
        collection=client.get_collection(settings.chroma_collection),
    )
    ok = True
    for i in (0, len(texts) // 2, len(texts) - 1):
        hits = direct_query(backend, embs[i], top_k=1)
        match = bool(hits) and hits[0]["text"] == texts[i]
        ok = ok and match
        logger.info("Verify #{}: top-1 khớp nguồn? {} | {}", i, match, texts[i][:50])
    if not ok:
        logger.error("VERIFY FAILED — Chroma trả kết quả lệch với vector nguồn.")
        sys.exit(1)
    logger.info("VERIFY OK — Chroma khôi phục khớp Qdrant.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Chỉ đọc Qdrant, không ghi Chroma")
    parser.add_argument("--yes", action="store_true", help="Xác nhận ghi đè collection Chroma")
    args = parser.parse_args()

    ids, texts, metas, embs = load_from_qdrant()
    logger.info("Đọc {} chunk, dim={}", len(ids), len(embs[0]) if embs else 0)

    if args.dry_run:
        logger.info("dry-run: id mẫu = {}", ids[:3])
        return
    if not args.yes:
        logger.error("Cần --yes để ghi đè collection Chroma (hoặc --dry-run để xem trước).")
        sys.exit(1)

    write_to_chroma(ids, texts, metas, embs)
    verify(texts, embs)


if __name__ == "__main__":
    main()
