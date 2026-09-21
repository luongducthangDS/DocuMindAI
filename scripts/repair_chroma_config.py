"""
Vá lại config_json_str của collection sau khi DB bị chromadb 1.x migrate.

Vì sao cần: chạy script/backend bằng Python global (chromadb 1.5.9 thay vì pin
0.6.3 của repo) làm chromadb migrate sysdb tại chỗ — nó dời config sang cột
schema_str mới và để config_json_str = "{}". Bản 0.6.3 vẫn đọc config_json_str
nên ném KeyError '_type' ngay lúc mở collection, mọi truy vấn trả 0 chunk.

Embeddings và HNSW index không bị đụng tới, chỉ hàng metadata trong sysdb hỏng,
nên chỉ cần ghi lại config đúng format 0.6.3 — không phải re-embed.

Phòng tái phát: luôn chạy bằng .venv của repo (./.venv/Scripts/python.exe).

Chạy:
    python scripts/repair_chroma_config.py --dry-run
    python scripts/repair_chroma_config.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DB = _REPO_ROOT / "data" / "chroma_db" / "chroma.sqlite3"

# Giá trị mặc định của HNSWConfigurationInternal trong chromadb 0.6.3.
_HNSW_DEFAULTS = {
    "space": "l2",
    "ef_construction": 100,
    "ef_search": 100,
    "num_threads": 16,
    "M": 16,
    "resize_factor": 1.2,
    "batch_size": 100,
    "sync_threshold": 1000,
    "_type": "HNSWConfigurationInternal",
}
# collection_metadata "hnsw:<key>" -> khoá trong HNSW config.
_HNSW_METADATA_KEYS = {
    "hnsw:space": "space",
    "hnsw:construction_ef": "ef_construction",
    "hnsw:search_ef": "ef_search",
    "hnsw:num_threads": "num_threads",
    "hnsw:M": "M",
    "hnsw:resize_factor": "resize_factor",
    "hnsw:batch_size": "batch_size",
    "hnsw:sync_threshold": "sync_threshold",
}


def _is_broken(config_json_str: str | None) -> bool:
    """0.6.3 cần khoá _type; thiếu nó là hỏng (kể cả chuỗi rỗng hay '{}')."""
    if not config_json_str:
        return True
    try:
        return "_type" not in json.loads(config_json_str)
    except json.JSONDecodeError:
        return True


def _build_config(cur: sqlite3.Cursor, collection_id: str) -> str:
    """Dựng config từ hnsw:* trong collection_metadata, phần thiếu lấy mặc định."""
    hnsw = dict(_HNSW_DEFAULTS)
    rows = cur.execute(
        "SELECT key, str_value, int_value, float_value FROM collection_metadata "
        "WHERE collection_id = ?",
        (collection_id,),
    )
    for key, str_value, int_value, float_value in rows:
        target = _HNSW_METADATA_KEYS.get(key)
        if target is None:
            continue
        value = next(v for v in (str_value, int_value, float_value) if v is not None)
        hnsw[target] = value
    return json.dumps(
        {"hnsw_configuration": hnsw, "_type": "CollectionConfigurationInternal"}
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=_DEFAULT_DB, help="Đường dẫn chroma.sqlite3")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ in ra, không ghi")
    args = parser.parse_args()

    if not args.db.exists():
        print(f"Không thấy DB: {args.db}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()
    # fetchall trước: _build_config dùng lại cursor này nên không được lặp trực tiếp.
    collections = cur.execute("SELECT id, name, config_json_str FROM collections").fetchall()
    broken = [
        (cid, name, _build_config(cur, cid))
        for cid, name, config in collections
        if _is_broken(config)
    ]

    if not broken:
        print("Mọi collection đã có config hợp lệ, không cần vá.")
        conn.close()
        return 0

    for _, name, config in broken:
        print(f"[{'dry-run' if args.dry_run else 'vá'}] {name}: {config}")

    if args.dry_run:
        conn.close()
        return 0

    backup = args.db.with_name(f"{args.db.name}.bak-{datetime.now():%Y%m%d-%H%M%S}")
    conn.close()
    shutil.copy2(args.db, backup)
    print(f"Đã sao lưu: {backup}")

    conn = sqlite3.connect(args.db)
    with conn:
        conn.executemany(
            "UPDATE collections SET config_json_str = ? WHERE id = ?",
            [(config, cid) for cid, _, config in broken],
        )
    conn.close()
    print(f"Đã vá {len(broken)} collection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
