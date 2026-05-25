"""
Fix missing so_hieu and ngay_ban_hanh in data/raw/*.json
Run: python scripts/fix_metadata.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── Regex helpers ─────────────────────────────────────────────────────────────

def _extract_so_hieu(title: str) -> str:
    """
    Extract document number from Vietnamese legal title.
    Patterns: 122/NQ-CP, 25/2026/NQ-CP, 142/2026/NĐ-CP, 21/2026/QĐ-TTg,
              777/QĐ-TTg, 07/2026/TT-NHNN, 45/2019/QH14 …
    """
    patterns = [
        r"(\d+/\d{4}/[\w\-Đđ]+)",   # XX/YYYY/TYPE-ORG
        r"(\d+/NQ\-CP)",              # XX/NQ-CP (no year)
        r"(\d+/[\w\-]+\-CP)",         # XX/TYPE-CP
        r"(\d+/[\w\-]+\-TTg)",        # XX/TYPE-TTg
        r"(\d+/[\w\-]+\-NHNN)",
        r"(\d+/[\w\-]+\-BCT)",
        r"(\d+/[\w\-]+\-BTC)",
        r"(\d+/[\w\-]+\-BGDĐT)",
        r"(\d+/[\w\-]+\-BXD)",
        r"(\d+/[\w\-]+\-BKHCN)",
        r"(\d+/[\w\-]+\-QH\d+)",     # XX/YYYY/QH14
    ]
    for pat in patterns:
        m = re.search(pat, title, re.UNICODE)
        if m:
            return m.group(1).strip(".")
    return ""


def _extract_ngay(content: str, title: str = "") -> str:
    """
    Extract the official signing date from document content.

    Strategy (in priority order):
    1. "Hà Nội, ngày DD tháng MM năm YYYY" — the signing location+date line,
       which is the most reliable indicator of the actual signing date.
    2. Last occurrence of "ngày D tháng M năm Y" in the whole document — covers
       QH laws where the signing block appears at the very end without "Hà Nội".
    """
    # Priority 1: explicit signing-location line
    m = re.search(
        r"Hà\s+Nội[,\s]+ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
        content,
        re.UNICODE,
    )
    if m:
        d, mo, y = m.group(1), m.group(2), m.group(3)
        return f"{y}-{mo.zfill(2)}-{d.zfill(2)}"

    # Priority 2: last date in document (QH laws end with the signing date)
    all_dates = re.findall(
        r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
        content,
        re.UNICODE,
    )
    if all_dates:
        d, mo, y = all_dates[-1]
        return f"{y}-{mo.zfill(2)}-{d.zfill(2)}"

    return ""


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Print changes without writing files")
    args = parser.parse_args()

    changed = 0
    skipped = 0

    for fpath in sorted(args.raw_dir.glob("*.json")):
        doc = json.loads(fpath.read_text(encoding="utf-8"))
        title = doc.get("title", "")
        content = doc.get("content", "")
        old_sh = doc.get("so_hieu", "") or ""
        old_ngay = doc.get("ngay_ban_hanh", "") or ""

        updates: dict = {}

        if not old_sh:
            new_sh = _extract_so_hieu(title)
            if new_sh:
                updates["so_hieu"] = new_sh

        if not old_ngay:
            so_hieu_ref = updates.get("so_hieu", old_sh)
            new_ngay = _extract_ngay(content, title=so_hieu_ref)
            if new_ngay:
                updates["ngay_ban_hanh"] = new_ngay

        if not updates:
            skipped += 1
            continue

        print(f"{'DRY ' if args.dry_run else ''}FIX: {title[:60]}")
        for k, v in updates.items():
            old = doc.get(k, "") or "---"
            print(f"      {k}: {old!r:25s} -> {v!r}")

        if not args.dry_run:
            doc.update(updates)
            fpath.write_text(
                json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            changed += 1

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Done: {changed} fixed, {skipped} already OK.")


if __name__ == "__main__":
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    main()
