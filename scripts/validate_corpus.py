"""
scripts/validate_corpus.py — kiểm tra corpus lao động/BHXH trước và sau khi ingest.

Kiểm 4 nhóm bất biến (SPEC-clause-schema-ingestion.md §5):
  1. Mỗi văn bản trong `locked_list_v1` có file toàn văn + frontmatter đủ trường.
  2. `clause_uid` + `version_id` không đụng nhau; mỗi `clause_uid` chỉ có các
     version khác `effective_from`.
  3. Mọi `superseded_by` trỏ tới một `version_id` có thật.
  4. Không record nào `verify_status != VERIFIED` lọt vào index.

Chạy:
    python scripts/validate_corpus.py                  # kiểm file + chunk dựng lại
    python scripts/validate_corpus.py --check-index    # kiểm thêm ChromaDB đã ingest
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_repo_root = Path(__file__).resolve().parents[1]
_local_hf = _repo_root / "data" / "hf_cache"
for _k in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "SENTENCE_TRANSFORMERS_HOME"):
    os.environ.setdefault(_k, str(_local_hf))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from loguru import logger

from src.ingestion.manifest import EFFECTIVE_TO_OPEN, load_manifest

REQUIRED_FRONTMATTER = ("doc_id", "so_hieu", "ten", "ngay_ban_hanh", "ngay_hieu_luc", "nguon")
REQUIRED_CHUNK_FIELDS = (
    "doc_id", "clause_uid", "version_id",
    "effective_from", "effective_to", "status",
)


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def render(self) -> int:
        for w in self.warnings:
            logger.warning(w)
        for e in self.errors:
            logger.error(e)
        if self.errors:
            logger.error("FAIL — {} lỗi, {} cảnh báo", len(self.errors), len(self.warnings))
            return 1
        logger.success("PASS — 0 lỗi, {} cảnh báo", len(self.warnings))
        return 0


def check_corpus_files(manifest_path: Path, corpus_dir: Path, report: Report) -> None:
    """Mỗi văn bản khoá phải có file toàn văn với frontmatter đủ trường."""
    from scripts.ingest_documents import split_frontmatter

    entries = load_manifest(manifest_path)
    if not entries:
        report.error("locked_list_v1 rỗng — không có văn bản nào để kiểm")
        return

    for doc_id, entry in entries.items():
        path = corpus_dir / entry.source_file
        if not path.exists():
            report.error(f"{doc_id}: thiếu file toàn văn {path}")
            continue
        frontmatter, body = split_frontmatter(path.read_text(encoding="utf-8"))
        missing = [f for f in REQUIRED_FRONTMATTER if not frontmatter.get(f)]
        if missing:
            report.error(f"{doc_id}: frontmatter thiếu trường {missing} ({path.name})")
        if len(body.strip()) < 500:
            report.error(f"{doc_id}: nội dung quá ngắn ({len(body)} ký tự) — nghi tóm lược")
        if not entry.is_verified:
            report.error(f"{doc_id}: verify_status = {entry.verify_status!r}, không được index")

        for amendment in entry.in_place_amended_clauses:
            if amendment.has_old_version and amendment.version_cu:
                vpath = corpus_dir / amendment.version_cu
                if not vpath.exists():
                    report.error(
                        f"{doc_id}: thiếu bản cũ {amendment.version_cu}"
                        f" của {amendment.clause_uid}"
                    )


def check_chunks(manifest_path: Path, corpus_dir: Path, as_of: str, report: Report) -> list[dict]:
    """Dựng lại chunk từ corpus và kiểm các bất biến về version."""
    from scripts.ingest_documents import discover_lao_dong

    per_doc = discover_lao_dong(manifest_path, corpus_dir, as_of)
    metas = [c.metadata for _, chunks in per_doc for c in chunks]
    if not metas:
        report.error("Không dựng được chunk nào từ corpus")
        return metas

    for meta in metas:
        missing = [f for f in REQUIRED_CHUNK_FIELDS if meta.get(f) in (None, "")]
        if missing:
            report.error(f"chunk {meta.get('clause_uid', '?')}: thiếu trường {missing}")

    dup_versions = [vid for vid, n in Counter(m["version_id"] for m in metas).items() if n > 1]
    if dup_versions:
        report.error(
            f"version_id trùng: {dup_versions[:5]} (tổng {len(dup_versions)})"
        )

    by_clause: dict[str, list[dict]] = defaultdict(list)
    for meta in metas:
        by_clause[meta["clause_uid"]].append(meta)
    for uid, versions in by_clause.items():
        dates = [v["effective_from"] for v in versions]
        if len(dates) != len(set(dates)):
            report.error(f"{uid}: hai version cùng effective_from {dates}")

    known_versions = {m["version_id"] for m in metas}
    for meta in metas:
        target = meta.get("superseded_by")
        if target and target not in known_versions:
            report.error(
                f"{meta['clause_uid']}: superseded_by trỏ version"
                f" không tồn tại ({target})"
            )

    for meta in metas:
        if meta.get("verify_status") and meta["verify_status"] != "VERIFIED":
            report.error(
                f"{meta['clause_uid']}: verify_status ="
                f" {meta['verify_status']!r} lọt vào index"
            )

    # khoảng hiệu lực phải hợp lệ
    for meta in metas:
        eff_from, eff_to = meta["effective_from"], meta["effective_to"]
        if eff_to != EFFECTIVE_TO_OPEN and eff_from >= eff_to:
            report.error(
                f"{meta['version_id']}: effective_from {eff_from}"
                f" >= effective_to {eff_to}"
            )

    multi = {uid: v for uid, v in by_clause.items() if len(v) > 1}
    logger.info(
        "Chunk: {} record / {} clause_uid / {} clause có nhiều version",
        len(metas), len(by_clause), len(multi),
    )
    for uid, versions in multi.items():
        ordered = sorted(versions, key=lambda x: x["effective_from"])
        spans = ", ".join(f"{v['effective_from']}→{v['effective_to']}" for v in ordered)
        logger.info("  {} : {}", uid, spans)
    return metas


def check_index(report: Report) -> None:
    """Kiểm collection ChromaDB đã ingest."""
    import chromadb

    from src.config import get_settings

    settings = get_settings()
    client = chromadb.PersistentClient(path=str((_repo_root / "data" / "chroma_db").resolve()))
    try:
        collection = client.get_collection(settings.chroma_collection)
    except Exception as exc:  # collection chưa tồn tại
        report.error(f"Không mở được collection {settings.chroma_collection!r}: {exc}")
        return

    count = collection.count()
    if count == 0:
        report.error("Collection rỗng — chưa ingest?")
        return

    got = collection.get(include=["metadatas"])
    metas = got["metadatas"]
    logger.info("Index: {} record trong collection {!r}", count, settings.chroma_collection)

    bad_verify = [m for m in metas if m.get("verify_status") not in ("VERIFIED", "", None)]
    if bad_verify:
        report.error(f"{len(bad_verify)} record trong index có verify_status khác VERIFIED")

    missing_clause = [m for m in metas if not m.get("clause_uid")]
    if missing_clause:
        report.warn(f"{len(missing_clause)} record trong index không có clause_uid")

    no_status = [m for m in metas if not m.get("status")]
    if no_status:
        report.error(f"{len(no_status)} record trong index thiếu status")

    banking = [
        m for m in metas
        if "NHNN" in str(m.get("so_hieu", ""))
        or "ngân hàng" in str(m.get("title", "")).lower()
    ]
    if banking:
        report.error(f"{len(banking)} record ngân hàng còn sót trong index")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("docs/corpus/corpus_manifest.yaml"))
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/raw/lao_dong"))
    parser.add_argument("--as-of", type=str, default=date.today().isoformat())
    parser.add_argument("--check-index", action="store_true", help="Kiểm thêm ChromaDB đã ingest")
    args = parser.parse_args()

    report = Report()
    check_corpus_files(args.manifest, args.corpus_dir, report)
    check_chunks(args.manifest, args.corpus_dir, args.as_of, report)
    if args.check_index:
        check_index(report)
    return report.render()


if __name__ == "__main__":
    sys.exit(main())
