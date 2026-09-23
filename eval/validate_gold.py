"""
eval/validate_gold.py — mechanical checks that a gold set is answerable from the index.

A drafted question is only ground truth if the index can back it. For every item:
  • source_clause / source_clauses exist as clause_uid in the vector store
  • as_of_date falls inside [effective_from, effective_to) of some version of each clause
  • every expect_contains group appears verbatim in that text (except derived answers)
  • ids are unique; refusals carry no gold clause

Checks against the LIVE index (Chroma or Qdrant via vector_backend), so it also catches
drift: a re-chunked corpus that renames a segment breaks the gold set loudly here
instead of silently dropping recall.

Usage:
  python eval/validate_gold.py data/eval/legal_qa_200.json
  python eval/validate_gold.py draft.json --drop-invalid --out clean.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from eval.temporal_eval import _norm  # noqa: E402  (same normaliser the scorer uses)

REQUIRED_KEYS = {
    "id", "question", "as_of_date", "category", "temporal_kind", "expected_behavior",
    "ground_truth", "source_doc", "source_clause", "expect_contains", "expect_absent",
}
# Answers computed from the clauses (12 ngày + 3 ngày thâm niên = "15 ngày", a ✅/❌
# verdict) are not verbatim in the law — only clause existence/validity is checkable.
DERIVED_KINDS = {"multi_clause", "compliance"}


def _catalog() -> dict[str, list[dict]]:
    from src.rag.vector_backend import fetch_all_chunks, get_backend

    by_uid: dict[str, list[dict]] = collections.defaultdict(list)
    for _id, text, meta in fetch_all_chunks(get_backend()):
        uid = meta.get("clause_uid")
        if uid:
            by_uid[uid].append({"text": text, **meta})
    return by_uid


def check(q: dict, catalog: dict[str, list[dict]]) -> list[str]:
    errors = [f"thiếu key {k}" for k in sorted(REQUIRED_KEYS - q.keys())]
    if errors:
        return errors
    uids = list(q.get("source_clauses") or []) or ([q["source_clause"]] if q["source_clause"] else [])
    if q["expected_behavior"] != "answer":
        # expect_contains here are refusal phrases ("không tìm thấy"), not law text.
        if uids:
            errors.append("câu từ chối/ngoài thời gian không được có gold clause")
        return errors
    if not uids:
        return errors  # doc-level question (legacy temporal set): scored by source_doc only
    as_of = int(q["as_of_date"].replace("-", ""))
    texts = []
    for uid in uids:
        versions = catalog.get(uid)
        if not versions:
            errors.append(f"clause_uid không có trong index: {uid}")
            continue
        live = [v for v in versions if v.get("effective_from_i", 0) <= as_of < v.get("effective_to_i", 99991231)]
        if not live:
            errors.append(f"{uid} không có hiệu lực tại {q['as_of_date']}")
            continue
        texts += [_norm(v["text"]) for v in live]
    if errors or q["temporal_kind"] in DERIVED_KINDS:
        return errors
    blob = "\n".join(texts)
    for group in q["expect_contains"]:
        if not any(_norm(alt) in blob for alt in group):
            errors.append(f"expect_contains không có nguyên văn trong chunk nguồn: {group}")
    return errors


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("gold", type=Path)
    p.add_argument("--drop-invalid", action="store_true")
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    questions = json.loads(args.gold.read_text(encoding="utf-8"))
    catalog = _catalog()
    dup = [i for i, n in collections.Counter(q.get("id") for q in questions).items() if n > 1]
    bad = {}
    for q in questions:
        errs = check(q, catalog)
        if errs:
            bad[q.get("id")] = errs
    for qid, errs in bad.items():
        print(f"✗ {qid}: " + "; ".join(errs))
    if dup:
        print(f"✗ id trùng: {dup}")
    print(f"{len(questions) - len(bad)}/{len(questions)} câu hợp lệ")

    if args.drop_invalid:
        kept = [q for q in questions if q.get("id") not in bad]
        (args.out or args.gold).write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
    elif bad or dup:
        sys.exit(1)


if __name__ == "__main__":
    main()
