"""
Copy collection ChromaDB local lên Chroma Cloud (xem dashboard + eval/provider_bench.py).

Đẩy nguyên embeddings + documents + metadatas có sẵn — KHÔNG re-embed, không tốn
quota Gemini. Chroma Cloud không lưu file gốc: mỗi bản ghi là 1 chunk (text + metadata
có tên văn bản nguồn). Retrieval production KHÔNG đọc từ đây (vẫn Qdrant/Chroma local).

Cần trong .env:
    CHROMA_CLOUD_API_KEY=ck-...
    CHROMA_CLOUD_TENANT=...
    CHROMA_CLOUD_DATABASE=DocuMind

Chạy:
    python scripts/copy_chroma_to_cloud.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loguru import logger  # noqa: E402

from migrate_chroma_to_qdrant import load_chroma_corpus  # noqa: E402
from src.rag.chroma_cloud import ChromaCloud  # noqa: E402

BATCH = 100  # Chroma Cloud giới hạn số bản ghi/lần ghi


def main() -> None:
    cc = ChromaCloud()
    ids, docs, metas, embs, _, local = load_chroma_corpus()

    # ponytail: xoá rồi tạo lại cho gọn (không phải dọn chunk thừa); ổn vì không có
    # đường production nào đọc từ bản sao này.
    if cc.http.delete(f"{cc.base}/{local.name}", timeout=60).status_code not in (200, 404):
        logger.warning("Không xoá được collection cũ trên Cloud, vẫn thử tạo mới")
    col_id = cc.call("POST", json={"name": local.name, "metadata": local.metadata})["id"]

    for i in range(0, len(ids), BATCH):
        j = i + BATCH
        cc.call("POST", f"/{col_id}/add", json={
            "ids": ids[i:j],
            "documents": docs[i:j],
            "metadatas": metas[i:j],
            "embeddings": [list(map(float, e)) for e in embs[i:j]],
        })
        logger.info("Đã đẩy {}/{}", min(j, len(ids)), len(ids))

    n = cc.call("GET", f"/{col_id}/count")
    logger.info("Chroma Cloud '{}': {} chunks (local: {})", local.name, n, len(ids))
    if n != len(ids):
        sys.exit(1)


if __name__ == "__main__":
    main()
