import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

from loguru import logger

from src.ingestion.manifest import EFFECTIVE_TO_OPEN, InPlaceAmendment


@dataclass
class LegalChunk:
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def is_valid(self) -> bool:
        return self.char_count >= 50


# Matches "Điều 1.", "Điều 12a.", "Điều 1:" etc. at the start of a line.
# Anchoring matters: an article that amends another law quotes the new wording
# ("Sửa đổi, bổ sung Điều 54 như sau: “Điều 54. ...”") and such a quoted heading
# must not open a new chunk — it belongs to the amending article.
_DIEU_RE = re.compile(
    r"^(Điều\s+\d+[a-z]?[\.\:]\s+.*?)(?=^Điều\s+\d+[a-z]?[\.\:]|\Z)",
    re.DOTALL | re.MULTILINE,
)

# Matches "Khoản 1.", "1." at line start inside an article
_KHOAN_RE = re.compile(r"(\n\d+\.\s)", re.MULTILINE)

# "Điều 139. Nghỉ thai sản" -> (139, "", "Nghỉ thai sản")
_DIEU_HEAD_RE = re.compile(r"^Điều\s+(\d+)([a-z]?)\s*[\.\:]\s*(.*)$")

# khoản boundary inside an article: "1. ", "8a. " at line start
_KHOAN_HEAD_RE = re.compile(r"^(\d+)([a-z]?)\s*\.\s")

_MAX_DIEU_CHARS = 4_000  # articles longer than this are split further

# Appendices and form templates ("Mẫu số 12", "PHỤ LỤC II") sit at the end of a
# decree and contain their own "Điều 1. ..." lines — those belong to a sample
# decision, not to the decree, so they must not be numbered as articles.
_APPENDIX_RE = re.compile(r"^(?:PHỤ LỤC|Mẫu số\s+\d+)\b", re.MULTILINE)
_APPENDIX_CHUNK_CHARS = 1_500


def split_appendix(text: str) -> tuple[str, str]:
    """Split a document into `(normative body, appendix)` at the first form marker."""
    m = _APPENDIX_RE.search(text)
    if not m:
        return text, ""
    return text[:m.start()], text[m.start():]


def parse_dieu_header(chunk_text: str) -> tuple[int | None, str, str]:
    """Return `(dieu_number, dieu_suffix, dieu_title)` from an article's first line."""
    first_line = chunk_text.split("\n")[0].strip()
    m = _DIEU_HEAD_RE.match(first_line)
    if not m:
        return None, "", ""
    return int(m.group(1)), m.group(2), m.group(3).strip()


def clause_status(effective_from: str, effective_to: str, as_of: str) -> str:
    """Lifecycle of one clause version at date `as_of` (CORPUS_SPEC §2.2)."""
    if effective_from and as_of < effective_from:
        return "not_yet_in_force"
    if effective_to and effective_to != EFFECTIVE_TO_OPEN and as_of >= effective_to:
        return "superseded"
    return "in_force"


def _version_meta(
    doc_meta: dict,
    *,
    clause_uid: str,
    dieu: int | None,
    dieu_tieu_de: str,
    khoan: int | str = "",
    diem: str = "",
    effective_from: str = "",
    amended_by_doc: str = "",
) -> dict:
    """Clause/version fields (schema: SPEC-clause-schema-ingestion.md).

    Added alongside the legacy metadata, never replacing it: the retriever and
    generator still read `source_url`, `title`, `so_hieu`, `dieu_header`, ...
    """
    as_of = doc_meta.get("as_of") or date.today().isoformat()
    eff_from = effective_from or doc_meta.get("effective_from", "")
    eff_to = doc_meta.get("effective_to", "") or EFFECTIVE_TO_OPEN
    return {
        "doc_id": doc_meta.get("doc_id", ""),
        "clause_uid": clause_uid,
        "version_id": f"{clause_uid}__v{eff_from}" if clause_uid and eff_from else "",
        "dieu": dieu if dieu is not None else "",
        "dieu_tieu_de": dieu_tieu_de,
        "khoan": khoan,
        "diem": diem,
        "effective_from": eff_from,
        "effective_to": eff_to,
        "status": clause_status(eff_from, eff_to, as_of),
        "superseded_by": "",
        "amended_by_doc": amended_by_doc,
        "consolidated_from": doc_meta.get("consolidated_from", ""),
        "verify_status": doc_meta.get("verify_status", ""),
    }


def _legacy_meta(doc_meta: dict, chunk_text: str) -> dict:
    """Metadata shape the existing retriever/generator depend on."""
    return {
        "source_url": doc_meta.get("url", ""),
        "title": doc_meta.get("title", ""),
        "doc_type": doc_meta.get("doc_type", "unknown"),
        "so_hieu": doc_meta.get("so_hieu", ""),
        "ngay_ban_hanh": doc_meta.get("ngay_ban_hanh", ""),
        "dieu_header": chunk_text.split("\n")[0][:120],
        "khoan_count": len(_KHOAN_RE.findall(chunk_text)),
        "char_count": len(chunk_text),
        "source": doc_meta.get("source", ""),
    }


def _split_amended_dieu(
    dieu_text: str,
    doc_meta: dict,
    dieu: int,
    dieu_tieu_de: str,
    amendments: Sequence[InPlaceAmendment],
) -> list[LegalChunk]:
    """Split an article amended in place into clause-level chunks.

    Amended khoản get a chunk of their own so each can carry its own
    `effective_from` and be paired with the superseded text from `_versions/`.
    The rest of the article stays in one chunk to preserve retrieval context.
    """
    wanted = {(a.khoan, a.khoan_hau_to): a for a in amendments if a.khoan is not None}
    buckets: list[tuple[InPlaceAmendment | None, list[str]]] = [(None, [])]

    for line in dieu_text.split("\n"):
        m = _KHOAN_HEAD_RE.match(line.strip())
        if m:
            key = (int(m.group(1)), m.group(2))
            if key in wanted:
                buckets.append((wanted[key], [line]))
                continue
            if buckets[-1][0] is not None:
                # a later khoản ends the amended one
                buckets.append((None, [line]))
                continue
        buckets[-1][1].append(line)

    chunks: list[LegalChunk] = []
    segment = 0
    for amendment, body in buckets:
        text = "\n".join(body).strip()
        if len(text) < 50:
            continue
        if amendment is None:
            # Un-amended remainder of the article. Pulling a khoản out can leave
            # several disjoint segments, so later ones get a suffix to keep
            # clause_uid unique (validated by scripts/validate_corpus.py).
            base_uid = f"{doc_meta.get('doc_id', '')}__d{dieu}"
            clause_uid = base_uid if segment == 0 else f"{base_uid}_s{segment}"
            segment += 1
            version = _version_meta(
                doc_meta,
                clause_uid=clause_uid,
                dieu=dieu,
                dieu_tieu_de=dieu_tieu_de,
            )
        else:
            # With a VBHN the file already carries the amended wording, so this
            # chunk *is* the new version. Without one the file is still the old
            # wording — it ends where the amendment starts, and the new text
            # comes from `version_moi` (see versions.build_version_chunks).
            corpus_text_is_old = amendment.corpus_text_is_superseded
            version = _version_meta(
                doc_meta,
                clause_uid=amendment.clause_uid,
                dieu=dieu,
                dieu_tieu_de=dieu_tieu_de or amendment.dieu_tieu_de,
                khoan=amendment.khoan if amendment.khoan is not None else "",
                diem=amendment.diem,
                effective_from="" if corpus_text_is_old else amendment.effective_from,
                amended_by_doc=amendment.amended_by_doc,
            )
            if corpus_text_is_old:
                version["effective_to"] = amendment.effective_from
                version["status"] = clause_status(
                    version["effective_from"],
                    amendment.effective_from,
                    doc_meta.get("as_of") or date.today().isoformat(),
                )
                version["superseded_by"] = f"{amendment.clause_uid}__v{amendment.effective_from}"
        meta = {**_legacy_meta(doc_meta, dieu_text), **version, "char_count": len(text)}
        chunks.append(LegalChunk(text=text, metadata=meta))
    return chunks


def chunk_by_dieu(
    text: str,
    doc_meta: dict,
    amendments: Sequence[InPlaceAmendment] = (),
) -> list[LegalChunk]:
    """
    Primary strategy: split by Điều (article).
    Each Điều becomes one chunk — preserves full legal semantics.
    Articles listed in `amendments` are split further, down to the amended khoản,
    so that clause-level versions can be tracked.
    Falls back to sliding-window if no Điều structure found.
    """
    chunks: list[LegalChunk] = []

    text, appendix = split_appendix(text)
    matches = list(_DIEU_RE.finditer(text))

    if not matches:
        logger.warning(
            "No 'Điều' structure found in '{}', using fallback chunker",
            doc_meta.get("title", "unknown"),
        )
        return _fallback_chunks(text, doc_meta)

    by_dieu: dict[int, list[InPlaceAmendment]] = {}
    for amendment in amendments:
        by_dieu.setdefault(amendment.dieu, []).append(amendment)

    for match in matches:
        chunk_text = match.group(1).strip()
        if len(chunk_text) < 50:
            continue

        dieu, _suffix, dieu_tieu_de = parse_dieu_header(chunk_text)

        # Articles amended in place are split at the amended khoản
        if dieu is not None and dieu in by_dieu:
            chunks.extend(
                _split_amended_dieu(chunk_text, doc_meta, dieu, dieu_tieu_de, by_dieu[dieu])
            )
            continue

        # Split oversized articles by khoản boundary
        if len(chunk_text) > _MAX_DIEU_CHARS:
            chunks.extend(_split_large_dieu(chunk_text, doc_meta, _MAX_DIEU_CHARS))
            continue

        version = _version_meta(
            doc_meta,
            clause_uid=f"{doc_meta['doc_id']}__d{dieu}" if doc_meta.get("doc_id") and dieu else "",
            dieu=dieu,
            dieu_tieu_de=dieu_tieu_de,
        )
        chunks.append(
            LegalChunk(
                text=chunk_text,
                metadata={**_legacy_meta(doc_meta, chunk_text), **version},
            )
        )

    chunks.extend(_appendix_chunks(appendix, doc_meta))

    logger.debug(
        "Chunked '{}' → {} chunks",
        doc_meta.get("title", "")[:40],
        len(chunks),
    )
    return chunks


def _appendix_chunks(appendix: str, doc_meta: dict) -> list[LegalChunk]:
    """Window the appendix into chunks that carry no article numbering."""
    if not appendix.strip():
        return []

    doc_id = doc_meta.get("doc_id", "")
    chunks: list[LegalChunk] = []
    paragraphs = [p.strip() for p in re.split(r"\n\n+", appendix) if p.strip()]
    buffer = ""

    def emit(body: str) -> None:
        if len(body.strip()) < 50:
            return
        uid = f"{doc_id}__pl{len(chunks) + 1}" if doc_id else ""
        version = _version_meta(
            doc_meta,
            clause_uid=uid,
            dieu=None,
            dieu_tieu_de="Phụ lục / biểu mẫu",
        )
        chunks.append(
            LegalChunk(
                text=body.strip(),
                metadata={
                    **_legacy_meta(doc_meta, body),
                    **version,
                    "char_count": len(body.strip()),
                    "chunk_strategy": "appendix",
                },
            )
        )

    for para in paragraphs:
        if len(buffer) + len(para) <= _APPENDIX_CHUNK_CHARS:
            buffer = f"{buffer}\n\n{para}" if buffer else para
        else:
            emit(buffer)
            buffer = para
    emit(buffer)
    return chunks


def _fallback_chunks(
    text: str,
    doc_meta: dict,
    size: int = 800,
    overlap: int = 100,
) -> list[LegalChunk]:
    """
    Sliding-window fallback for documents without Điều structure.
    Splits on paragraph boundaries when possible.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    chunks: list[LegalChunk] = []
    buffer = ""

    for para in paragraphs:
        if len(buffer) + len(para) <= size:
            buffer = buffer + "\n\n" + para if buffer else para
        else:
            if buffer:
                chunks.append(
                    LegalChunk(
                        text=buffer,
                        metadata={
                            **doc_meta,
                            "dieu_header": buffer[:80],
                            "char_count": len(buffer),
                            "chunk_strategy": "fallback_window",
                        },
                    )
                )
            # Keep last paragraph for overlap
            buffer = para

    if buffer:
        chunks.append(
            LegalChunk(
                text=buffer,
                metadata={
                    **doc_meta,
                    "dieu_header": buffer[:80],
                    "char_count": len(buffer),
                    "chunk_strategy": "fallback_window",
                },
            )
        )

    return chunks


def _split_large_dieu(
    dieu_text: str,
    doc_meta: dict,
    max_chars: int,
) -> list[LegalChunk]:
    """Split an oversized Điều into sub-chunks at khoản boundaries."""
    dieu, _suffix, dieu_tieu_de = parse_dieu_header(dieu_text)
    base_meta = {
        **_legacy_meta(doc_meta, dieu_text),
        **_version_meta(
            doc_meta,
            clause_uid=f"{doc_meta['doc_id']}__d{dieu}" if doc_meta.get("doc_id") and dieu else "",
            dieu=dieu,
            dieu_tieu_de=dieu_tieu_de,
        ),
    }
    parts = re.split(r"(?=\n\d+\.\s)", dieu_text)
    chunks: list[LegalChunk] = []
    buffer = ""

    def _segment_meta(length: int) -> dict:
        """Keep clause_uid/version_id unique across the segments of one article."""
        n = len(chunks)
        if n == 0:
            return {**base_meta, "char_count": length, "khoan_count": 0}
        uid = f"{base_meta.get('clause_uid', '')}_s{n}"
        eff_from = base_meta.get("effective_from", "")
        return {
            **base_meta,
            "clause_uid": uid,
            "version_id": f"{uid}__v{eff_from}" if uid and eff_from else "",
            "char_count": length,
            "khoan_count": 0,
        }

    for part in parts:
        if len(buffer) + len(part) <= max_chars:
            buffer = buffer + part
        else:
            if buffer.strip():
                chunks.append(LegalChunk(
                    text=buffer.strip()[:max_chars],
                    metadata=_segment_meta(len(buffer)),
                ))
            buffer = part

    if buffer.strip():
        chunks.append(LegalChunk(
            text=buffer.strip()[:max_chars],
            metadata=_segment_meta(len(buffer)),
        ))

    return chunks if chunks else [LegalChunk(
        text=dieu_text[:max_chars] + "…",
        metadata={**base_meta, "char_count": max_chars, "khoan_count": 0},
    )]
