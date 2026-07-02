"""
scripts/cleanup_chroma.py — xóa sạch mọi collection KHÔNG liên quan trong ChromaDB,
chỉ giữ lại corpus production `documind_legal` (91 chunk UNETI).

An toàn: KEEP được hard-code, script không bao giờ xóa collection trong KEEP.
Mặc định chạy ở chế độ xem trước (dry-run). Thêm --apply để xóa thật.

    python scripts/cleanup_chroma.py            # xem sẽ xóa gì
    python scripts/cleanup_chroma.py --apply    # xóa thật
"""

from __future__ import annotations

import argparse
from pathlib import Path

import chromadb

KEEP = {"documind_legal"}                 # collection duy nhất được giữ
_REPO = Path(__file__).resolve().parents[1]
_CHROMA_PATH = str((_REPO / "data" / "chroma_db").resolve())


def _count(col) -> int:
    try:
        return col.count()
    except Exception as e:
        return -1  # -1 = không đếm được (collection lỗi/dangling)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Xóa thật (mặc định chỉ xem trước)")
    args = ap.parse_args()

    client = chromadb.PersistentClient(path=_CHROMA_PATH)
    cols = client.list_collections()
    # ChromaDB >=0.6.0 trả list[str] (tên); bản cũ trả list object có .name
    names = [c if isinstance(c, str) else c.name for c in cols]

    print(f"ChromaDB: {_CHROMA_PATH}")
    print(f"Tìm thấy {len(names)} collection:\n")
    to_delete = []
    for name in names:
        n = _count(client.get_collection(name))
        keep = name in KEEP
        tag = "GIỮ " if keep else "XÓA "
        print(f"  [{tag}] {name:20s}  {n} docs")
        if not keep:
            to_delete.append(name)

    if not to_delete:
        print("\nKhông có gì để xóa. Xong.")
        return

    if not args.apply:
        print(f"\n(DRY-RUN) Sẽ xóa {len(to_delete)} collection: {to_delete}")
        print("Chạy lại với --apply để xóa thật.")
        return

    # Guard tuyệt đối: không bao giờ xóa thứ trong KEEP
    for name in to_delete:
        assert name not in KEEP, f"REFUSE to delete protected collection {name}"
        client.delete_collection(name)
        print(f"  Đã xóa: {name}")

    # Kiểm tra sau khi xóa
    remaining = client.list_collections()
    rnames = [c if isinstance(c, str) else c.name for c in remaining]
    print(f"\nCòn lại {len(rnames)} collection:")
    legal_ok = False
    for name in rnames:
        n = _count(client.get_collection(name))
        print(f"  {name:20s}  {n} docs")
        if name == "documind_legal" and n == 91:
            legal_ok = True
    print("\n✅ documind_legal vẫn nguyên 91 chunk." if legal_ok
          else "\n⚠️  KIỂM TRA LẠI: documind_legal không còn đúng 91 chunk!")


if __name__ == "__main__":
    main()
