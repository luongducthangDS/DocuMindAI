"""
Loader for `docs/corpus/corpus_manifest.yaml` — the single source of truth for
which documents may be indexed, when each one is in force, and which clauses
were amended in place.

`chunker`, `temporal-retrieval` and the ingest script all read the effective-date
map produced here; nothing else parses the manifest directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml
from loguru import logger

# ChromaDB metadata cannot hold None, and an open-ended range is easier to filter
# as a sentinel date than as a special case: "effective_to is empty" becomes
# "effective_to > T" for every comparison.
EFFECTIVE_TO_OPEN = "9999-12-31"

_VERIFIED_PREFIX = "VERIFIED"

# Repo-root-relative default so callers (API, graph) do not depend on cwd.
DEFAULT_MANIFEST_PATH = (
    Path(__file__).resolve().parents[2] / "docs" / "corpus" / "corpus_manifest.yaml"
)


@dataclass(frozen=True)
class InPlaceAmendment:
    """One clause of a document that was amended in place by a later document."""

    clause_uid: str
    dieu: int
    effective_from: str
    amended_by_doc: str
    khoan: int | None = None
    khoan_hau_to: str = ""
    diem: str = ""
    dieu_tieu_de: str = ""
    loai_sua: str = "sua_doi_khoan"
    version_cu: str = ""
    version_moi: str = ""

    @property
    def has_old_version(self) -> bool:
        """False for clauses that were newly inserted (no superseded text)."""
        return self.loai_sua != "bo_sung_khoan_moi"

    @property
    def corpus_text_is_superseded(self) -> bool:
        """True when the corpus file still holds the *old* wording of this clause.

        Bộ luật Lao động is indexed from its VBHN, so its file already carries
        the amended text and `_versions/` holds the superseded one. A document
        with no VBHN (Luật BHXH 2024, amended by Luật Dân số 113/2025/QH15) is
        the mirror image: the corpus file is the pre-amendment text and the new
        wording lives in `version_moi`.
        """
        return self.loai_sua == "sua_doi_khong_co_vbhn"


@dataclass(frozen=True)
class DocEntry:
    """A document from `locked_list_v1`, with its effective window."""

    doc_id: str
    so_hieu: str
    ten: str
    loai: str
    ngay_ban_hanh: str
    ngay_hieu_luc: str
    het_hieu_luc_tu: str
    trang_thai: str
    verify_status: str
    nguon: str = ""
    file_toan_van: str = ""
    file_ban_goc: str = ""
    consolidated_from: str = ""
    in_place_amended_clauses: tuple[InPlaceAmendment, ...] = field(default_factory=tuple)

    @property
    def is_verified(self) -> bool:
        """Only VERIFIED documents may enter the index (CORPUS_SPEC §2)."""
        return self.verify_status.startswith(_VERIFIED_PREFIX)

    @property
    def effective_to(self) -> str:
        """End of the document's effective window, or the open-ended sentinel."""
        return self.het_hieu_luc_tu or EFFECTIVE_TO_OPEN

    @property
    def source_file(self) -> str:
        """Markdown file under data/raw/lao_dong/ holding this document's text."""
        return self.file_toan_van or f"{self.doc_id}.md"

    def amendment_for(self, dieu: int) -> tuple[InPlaceAmendment, ...]:
        return tuple(a for a in self.in_place_amended_clauses if a.dieu == dieu)


def _as_str(value: object) -> str:
    """YAML may parse dates as `datetime.date`; metadata is stored as ISO text."""
    if value is None:
        return ""
    return str(value)


def _build_amendments(raw: list[dict] | None) -> tuple[InPlaceAmendment, ...]:
    out: list[InPlaceAmendment] = []
    for item in raw or []:
        out.append(
            InPlaceAmendment(
                clause_uid=_as_str(item.get("clause_uid")),
                dieu=int(item["dieu"]),
                effective_from=_as_str(item.get("effective_from")),
                amended_by_doc=_as_str(item.get("amended_by_doc")),
                khoan=item.get("khoan"),
                khoan_hau_to=_as_str(item.get("khoan_hau_to")),
                diem=_as_str(item.get("diem")),
                dieu_tieu_de=_as_str(item.get("dieu_tieu_de")),
                loai_sua=_as_str(item.get("loai_sua")) or "sua_doi_khoan",
                version_cu=_as_str(item.get("version_cu")),
                version_moi=_as_str(item.get("version_moi")),
            )
        )
    return tuple(out)


def locked_doc_ids(manifest: dict) -> list[str]:
    """doc_ids in `locked_list_v1` (both in-force and kept-for-point-in-time).

    `van_ban_hop_nhat` is skipped on purpose: its entries are free-text labels,
    not doc_ids — the consolidated text is attached to the document it amends
    through `file_toan_van`.
    """
    locked = manifest.get("locked_list_v1") or {}
    ids: list[str] = []
    for key in ("con_hieu_luc", "het_hieu_luc_giu_cho_point_in_time"):
        ids.extend(_as_str(x) for x in (locked.get(key) or []))
    return ids


def load_manifest(path: str | Path) -> dict[str, DocEntry]:
    """Return `{doc_id: DocEntry}` for every document in `locked_list_v1`.

    Documents present in `documents:` but outside the locked list are ignored —
    they are candidates for a later round, not part of this corpus.
    """
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    wanted = locked_doc_ids(data)
    by_id = {_as_str(d.get("doc_id")): d for d in (data.get("documents") or [])}

    entries: dict[str, DocEntry] = {}
    for doc_id in wanted:
        raw = by_id.get(doc_id)
        if raw is None:
            logger.warning("locked_list_v1 references unknown doc_id '{}'", doc_id)
            continue
        entries[doc_id] = DocEntry(
            doc_id=doc_id,
            so_hieu=_as_str(raw.get("so_hieu")),
            ten=_as_str(raw.get("ten")),
            loai=_as_str(raw.get("loai")),
            ngay_ban_hanh=_as_str(raw.get("ngay_ban_hanh")),
            ngay_hieu_luc=_as_str(raw.get("ngay_hieu_luc")),
            het_hieu_luc_tu=_as_str(raw.get("het_hieu_luc_tu")),
            trang_thai=_as_str(raw.get("trang_thai")),
            verify_status=_as_str(raw.get("verify_status")),
            nguon=_as_str(raw.get("nguon")),
            file_toan_van=_as_str(raw.get("file_toan_van")),
            file_ban_goc=_as_str(raw.get("file_ban_goc")),
            consolidated_from=_as_str(raw.get("consolidated_from")),
            in_place_amended_clauses=_build_amendments(raw.get("sua_doi_in_place")),
        )

    missing = [d for d in wanted if d not in entries]
    logger.debug(
        "Manifest: {} locked documents loaded ({} verified){}",
        len(entries),
        sum(1 for e in entries.values() if e.is_verified),
        f", {len(missing)} missing" if missing else "",
    )
    return entries


@lru_cache(maxsize=4)
def corpus_earliest_point_in_time(
    path: str | Path = DEFAULT_MANIFEST_PATH, default: str = "2015-01-01"
) -> str:
    """Earliest date the corpus can answer for (`meta.earliest_point_in_time`).

    `temporal-retrieval` compares `as_of_date` against this to tell the user the
    question falls before the corpus starts, instead of answering from law that
    was never indexed.
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    value = _as_str((data.get("meta") or {}).get("earliest_point_in_time"))
    return value or default
