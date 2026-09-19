"""
Superseded clause versions.

The corpus files under `data/raw/lao_dong/` hold the text **currently** in force
(for Bộ luật Lao động that is the consolidated VBHN). Older wording of a clause
that was amended in place lives in `data/raw/lao_dong/_versions/` and is turned
into its own chunk here, so a query with an `as_of_date` before the amendment can
still be answered with the law as it stood then.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from loguru import logger

from src.ingestion.chunker import LegalChunk, clause_status
from src.ingestion.manifest import EFFECTIVE_TO_OPEN, DocEntry

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_version_file(path: Path) -> tuple[dict, str]:
    """Split a `_versions/*.md` file into its frontmatter dict and body text."""
    raw = path.read_text(encoding="utf-8")
    m = _FRONTMATTER_RE.match(raw)
    if not m:
        return {}, raw.strip()
    meta: dict[str, str] = {}
    for line in m.group(1).split("\n"):
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta, raw[m.end():].strip()


def build_version_chunks(
    entry: DocEntry,
    corpus_dir: Path,
    doc_meta: dict,
    as_of: str | None = None,
) -> list[LegalChunk]:
    """Chunks for the superseded wording of every in-place amended clause.

    The current wording is produced by the chunker from the document text; this
    adds the older version(s) and links them with `superseded_by`, so a clause
    amended in place ends up with one record per version.
    """
    as_of = as_of or doc_meta.get("as_of") or date.today().isoformat()
    chunks: list[LegalChunk] = []

    for amendment in entry.in_place_amended_clauses:
        if not amendment.has_old_version:
            continue  # newly inserted clause — nothing existed before

        # Which side of the amendment this file holds depends on the document:
        # with a VBHN the corpus text is already the new wording and `_versions/`
        # supplies the old one; without a VBHN it is the other way round.
        corpus_text_is_old = amendment.corpus_text_is_superseded
        version_file = amendment.version_moi if corpus_text_is_old else amendment.version_cu
        if not version_file:
            logger.warning(
                "No {} text configured for {} — point-in-time {} {} "
                "will fall back to the wording in the corpus file",
                "amended" if corpus_text_is_old else "superseded",
                amendment.clause_uid,
                "from" if corpus_text_is_old else "before",
                amendment.effective_from,
            )
            continue

        path = corpus_dir / version_file
        if not path.exists():
            logger.error("Missing version file {} for {}", path, amendment.clause_uid)
            continue

        meta, body = _parse_version_file(path)
        if not body:
            logger.error("Empty version file {}", path)
            continue

        if corpus_text_is_old:
            effective_from = meta.get("effective_from") or amendment.effective_from
            effective_to = meta.get("effective_to") or EFFECTIVE_TO_OPEN
            new_version_id = ""  # this *is* the newest version — nothing supersedes it
        else:
            effective_from = meta.get("effective_from") or entry.ngay_hieu_luc
            effective_to = meta.get("effective_to") or amendment.effective_from
            new_version_id = f"{amendment.clause_uid}__v{amendment.effective_from}"

        chunks.append(
            LegalChunk(
                text=body,
                metadata={
                    # legacy fields the retriever/generator read
                    "source_url": meta.get("nguon", entry.nguon),
                    "title": entry.ten,
                    "doc_type": entry.loai,
                    "so_hieu": entry.so_hieu,
                    "ngay_ban_hanh": entry.ngay_ban_hanh,
                    "dieu_header": f"Điều {amendment.dieu}. {amendment.dieu_tieu_de}",
                    "khoan_count": 1,
                    "char_count": len(body),
                    "source": doc_meta.get("source", ""),
                    # clause/version fields
                    "doc_id": entry.doc_id,
                    "clause_uid": amendment.clause_uid,
                    # always derived, so the convention <clause_uid>__v<effective_from>
                    # holds for every record regardless of what the file declares
                    "version_id": f"{amendment.clause_uid}__v{effective_from}",
                    "dieu": amendment.dieu,
                    "dieu_tieu_de": amendment.dieu_tieu_de,
                    "khoan": amendment.khoan if amendment.khoan is not None else "",
                    "diem": amendment.diem,
                    "effective_from": effective_from,
                    "effective_to": effective_to,
                    "status": clause_status(effective_from, effective_to, as_of),
                    "superseded_by": new_version_id,
                    "amended_by_doc": meta.get("amended_by_doc", ""),
                    "consolidated_from": "",
                    "verify_status": doc_meta.get("verify_status", ""),
                },
            )
        )

    if chunks:
        logger.debug("{}: {} superseded clause version(s)", entry.doc_id, len(chunks))
    return chunks
