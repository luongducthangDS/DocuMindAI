"""
Xoá chunk do người dùng upload (payload `source == "user_upload"`) khỏi Qdrant Cloud.

Trước hotfix P0, POST /api/v1/upload không cần xác thực và upload ẩn danh được
stamp tenant_id=public / acl_label=public — tức mọi người dùng đều truy hồi được,
ở mọi as_of_date. Script này gỡ các chunk đó (payload `source == "user_upload"`).

Chunk thiếu `tenant_id` cũng tính là public: RetrievalContext.allows() mặc định
tenant_id vắng = public, nên BM25 vẫn trả chúng ra.

Lọc `source` phía client, không phải phía server: Qdrant Cloud bật strict mode,
lọc trên field chưa có payload index là 400, và `source` không nằm trong
QDRANT_PAYLOAD_INDEXES. Chỉ `tenant_id` (có index) được đẩy xuống server.

    python scripts/sanitize_qdrant_corpus.py                      # xem trước (tenant public)
    python scripts/sanitize_qdrant_corpus.py --yes                # xoá (tenant public)
    python scripts/sanitize_qdrant_corpus.py --all-tenants --yes  # xoá upload của MỌI tenant

Mặc định chỉ tenant public — nơi upload ẩn danh rơi vào. `--all-tenants` xoá cả
tài liệu tenant tự upload hợp lệ; production hiện chưa có tenant nên hai cách như nhau.

Sau khi xoá: BM25 của backend đang chạy vẫn giữ text cũ trong RAM — restart
service Render (hoặc POST /api/v1/reload kèm X-Admin-Key) để nạp lại.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings  # noqa: E402
from src.rag.context import PUBLIC_TENANT  # noqa: E402

UPLOAD_SOURCE = "user_upload"
_PAGE = 256
_DELETE_BATCH = 256


def _public_filter():
    from qdrant_client.http import models as qm

    return qm.Filter(should=[
        qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=PUBLIC_TENANT)),
        qm.IsEmptyCondition(is_empty=qm.PayloadField(key="tenant_id")),
    ])


def find_poison_points(client, collection: str, all_tenants: bool = False) -> list[tuple[object, str]]:
    """[(point_id, title), ...] cho mọi point có source == user_upload (mặc định: tenant public)."""
    found: list[tuple[object, str]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            scroll_filter=None if all_tenants else _public_filter(),
            limit=_PAGE,
            offset=offset,
            with_payload=["source", "title"],
            with_vectors=False,
        )
        for p in points:
            payload = p.payload or {}
            if payload.get("source") == UPLOAD_SOURCE:
                found.append((p.id, str(payload.get("title", ""))))
        if offset is None:
            return found


def delete_points(client, collection: str, ids: list) -> int:
    from qdrant_client.http import models as qm

    for i in range(0, len(ids), _DELETE_BATCH):
        client.delete(
            collection_name=collection,
            points_selector=qm.PointIdsList(points=ids[i:i + _DELETE_BATCH]),
            wait=True,
        )
    return len(ids)


def main() -> int:
    parser = argparse.ArgumentParser(description="Xoá chunk user_upload khỏi Qdrant")
    parser.add_argument("--yes", action="store_true", help="xác nhận xoá (mặc định chỉ xem trước)")
    parser.add_argument("--all-tenants", action="store_true", help="xoá upload của mọi tenant, không chỉ public")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.qdrant_url:
        print("QDRANT_URL chưa set — không có gì để dọn.", file=sys.stderr)
        return 2

    from qdrant_client import QdrantClient

    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    collection = settings.qdrant_collection

    found = find_poison_points(client, collection, all_tenants=args.all_tenants)
    total = client.count(collection_name=collection, exact=True).count
    scope = "mọi tenant" if args.all_tenants else "tenant public"
    print(f"Collection '{collection}': {total} points, {len(found)} là upload ({scope}).")
    for title, n in Counter(t for _, t in found).most_common(20):
        print(f"  {n:5d}  {title!r}")

    if not found:
        return 0
    if not args.yes:
        print("\nXem trước — chưa xoá gì. Chạy lại với --yes để xoá.")
        return 0

    deleted = delete_points(client, collection, [pid for pid, _ in found])
    after = client.count(collection_name=collection, exact=True).count
    print(f"Đã xoá {deleted} points. Còn {after} (trước {total}).")
    print("Restart backend trên Render để BM25 trong RAM bỏ text đã xoá.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
