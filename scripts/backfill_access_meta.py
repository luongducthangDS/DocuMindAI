"""
Backfill the four access/temporal filter fields onto an already-indexed corpus.

Adds `tenant_id`, `acl_label`, `effective_from_i`, `effective_to_i` to every
chunk without touching text or embeddings — `collection.update()` with
metadatas only, so this costs no embedding API calls and keeps chunk ids.

The labour-law corpus is public by definition (published legislation), so it is
stamped tenant_id=public / acl_label=public. Tenant-private documents get their
labels at upload time instead (src/api/routes/documents.py).

    python scripts/backfill_access_meta.py            # preview
    python scripts/backfill_access_meta.py --yes      # write
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hf_env import use_local_hf_cache  # noqa: E402

use_local_hf_cache()

from src.rag.context import (  # noqa: E402
    DEFAULT_ACL_LABEL,
    PUBLIC_TENANT,
    stamp_access_meta,
)

_NEW_FIELDS = ("tenant_id", "acl_label", "effective_from_i", "effective_to_i")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill access/temporal filter metadata")
    parser.add_argument("--yes", action="store_true", help="xác nhận ghi metadata")
    parser.add_argument("--tenant", default=PUBLIC_TENANT)
    parser.add_argument("--acl-label", default=DEFAULT_ACL_LABEL)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    from src.rag.vector_backend import get_backend

    backend = get_backend()
    if backend.provider != "chroma":
        raise SystemExit(
            f"Script này chỉ chạy trên chroma (provider hiện tại: {backend.provider}). "
            "Với Qdrant: chạy lại scripts/migrate_chroma_to_qdrant.py sau khi backfill xong."
        )

    col = backend.collection
    got = col.get(include=["metadatas"])
    ids = got.get("ids") or []
    metas = got.get("metadatas") or []
    print(f"Collection: {col.name} — {len(ids)} chunk")

    todo_ids, todo_metas, unchanged = [], [], 0
    for cid, meta in zip(ids, metas):
        meta = meta or {}
        stamped = stamp_access_meta(meta, tenant_id=args.tenant, acl_label=args.acl_label)
        if all(meta.get(f) == stamped.get(f) for f in _NEW_FIELDS):
            unchanged += 1
            continue
        todo_ids.append(cid)
        todo_metas.append(stamped)

    print(f"Cần cập nhật: {len(todo_ids)} — đã đúng sẵn: {unchanged}")
    if todo_metas:
        sample = todo_metas[0]
        print("Ví dụ:", {k: sample.get(k) for k in _NEW_FIELDS},
              "| effective_to =", sample.get("effective_to"))

    # Superseded clause versions are the ones this whole exercise is about —
    # show them explicitly so a wrong sentinel is visible before writing.
    closed = [m for m in todo_metas if m.get("effective_to_i", 99991231) != 99991231]
    print(f"Khoản có mốc hết hiệu lực (effective_to_i != 99991231): {len(closed)}")
    for m in closed[:5]:
        print(f"   {m.get('clause_uid', '?')}: {m.get('effective_from')} → {m.get('effective_to')}"
              f"  [{m.get('effective_from_i')} → {m.get('effective_to_i')}]")

    if not args.yes:
        print("\n(dry-run) Thêm --yes để ghi.")
        return
    if not todo_ids:
        print("Không có gì để ghi.")
        return

    for start in range(0, len(todo_ids), args.batch_size):
        col.update(
            ids=todo_ids[start:start + args.batch_size],
            metadatas=todo_metas[start:start + args.batch_size],
        )
        print(f"  ghi {min(start + args.batch_size, len(todo_ids))}/{len(todo_ids)}")
    print("Xong.")


if __name__ == "__main__":
    main()
