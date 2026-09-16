"""
Re-embed corpus tại chỗ bằng model đang cấu hình (EMBEDDING_MODEL trong .env).

Vì sao không chạy lại ingest từ data/raw: chunk hiện có mang metadata temporal
(clause_uid, version_id, effective_from/to, superseded_by) mà gold set và
do_temporal_filter phụ thuộc vào. Chạy lại ingest có thể tạo ra chunk khác và
làm hỏng các liên kết đó. Script này chỉ đổi VECTOR, giữ nguyên id, text và
metadata của từng chunk — nên khác biệt đo được sau đó đúng là của model.

An toàn:
  • Sao lưu collection cũ sang tên <collection>__backup_<model cũ> trước khi ghi đè.
  • Ghi nhãn model + chiều vector vào metadata collection, để lần sau lệch model
    là báo lỗi ngay thay vì trả kết quả sai im lặng.
  • --dry-run xem trước, không ghi gì.

Chạy:
    python scripts/reembed_corpus.py --dry-run
    python scripts/reembed_corpus.py --yes
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.hf_env import use_local_hf_cache  # noqa: E402

# Cho phép tải model mới: đây chính là đường chạy khi đổi sang model chưa cache.
use_local_hf_cache(offline=False, create=True)

from loguru import logger  # noqa: E402

from src.config import get_settings  # noqa: E402
from src.rag.embedder import (  # noqa: E402
    STORE_META_DIM,
    STORE_META_MODEL,
    get_chroma_collection,
    get_embedder,
    get_embedding_dim,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _slug(model_name: str) -> str:
    """Chroma chỉ nhận [a-zA-Z0-9._-] trong tên collection."""
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name).strip("_")
    return (cleaned or "unlabeled")[:48]


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-embed corpus bằng EMBEDDING_MODEL hiện tại")
    parser.add_argument("--dry-run", action="store_true", help="chỉ xem trước, không ghi")
    parser.add_argument("--yes", action="store_true", help="xác nhận ghi đè collection")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-backup", action="store_true",
                        help="không sao lưu collection cũ (mặc định là có sao lưu)")
    args = parser.parse_args()

    settings = get_settings()
    target_model = settings.embedding_model

    # verify=False: chính script này là chỗ được phép đổi model của corpus.
    client, collection = get_chroma_collection(verify=False)
    existing_meta = collection.metadata or {}
    indexed_model = existing_meta.get(STORE_META_MODEL) or "unlabeled"

    data = collection.get(include=["documents", "metadatas"])
    ids = data.get("ids") or []
    docs = data.get("documents") or []
    metas = data.get("metadatas") or []
    keep = [(i, d, m) for i, d, m in zip(ids, docs, metas) if d]

    print()
    print(f"Collection      : {settings.chroma_collection}")
    print(f"Chunks          : {len(keep)} (bỏ qua {len(ids) - len(keep)} chunk rỗng)")
    print(f"Model đã index  : {indexed_model}")
    print(f"Model sẽ dùng   : {target_model}")
    print()

    if not keep:
        raise SystemExit("Corpus rỗng — không có gì để re-embed.")

    if args.dry_run:
        print("--dry-run: dừng ở đây, không ghi gì.")
        return

    if not args.yes:
        raise SystemExit("Cần --yes để ghi đè collection (hoặc --dry-run để xem trước).")

    embedder = get_embedder()
    dim = get_embedding_dim()
    logger.info("Model {} — vector {} chiều", target_model, dim)

    # Sao lưu trước khi đụng vào bản đang chạy.
    if not args.no_backup:
        backup_name = f"{settings.chroma_collection}__backup_{_slug(str(indexed_model))}"
        try:
            client.delete_collection(backup_name)
        except Exception:
            pass  # chưa có bản sao lưu nào — trường hợp bình thường
        backup = client.create_collection(name=backup_name, metadata=dict(existing_meta))
        old = collection.get(include=["documents", "metadatas", "embeddings"])
        backup.add(
            ids=old["ids"],
            documents=old["documents"],
            metadatas=old["metadatas"],
            embeddings=old["embeddings"],
        )
        logger.info("Đã sao lưu collection cũ → {} ({} chunks)", backup_name, len(old["ids"]))

    t0 = time.time()
    vectors: list[list[float]] = []
    for start in range(0, len(keep), args.batch_size):
        batch = keep[start:start + args.batch_size]
        vectors.extend(embedder.get_text_embedding_batch([d for _, d, _ in batch]))
        done = min(start + args.batch_size, len(keep))
        if done % 160 == 0 or done == len(keep):
            logger.info("Embed {}/{} chunks ({:.0f}s)", done, len(keep), time.time() - t0)

    if len(vectors) != len(keep):
        raise SystemExit(f"Số vector ({len(vectors)}) không khớp số chunk ({len(keep)}) — huỷ, không ghi.")
    if len(vectors[0]) != dim:
        raise SystemExit(f"Vector {len(vectors[0])} chiều, khác {dim} chiều model khai báo — huỷ, không ghi.")

    # Chroma khoá chiều vector theo collection, nên phải tạo lại collection mới
    # thay vì update tại chỗ khi số chiều đổi.
    client.delete_collection(settings.chroma_collection)
    new_collection = client.create_collection(
        name=settings.chroma_collection,
        metadata={
            "hnsw:space": "cosine",
            STORE_META_MODEL: target_model,
            STORE_META_DIM: dim,
        },
    )
    for start in range(0, len(keep), 256):
        batch = keep[start:start + 256]
        new_collection.add(
            ids=[i for i, _, _ in batch],
            documents=[d for _, d, _ in batch],
            metadatas=[m for _, _, m in batch],
            embeddings=vectors[start:start + 256],
        )

    logger.info(
        "Xong: {} chunks, {} chiều, {:.0f}s — collection đã gắn nhãn {}",
        new_collection.count(), dim, time.time() - t0, target_model,
    )


if __name__ == "__main__":
    main()
