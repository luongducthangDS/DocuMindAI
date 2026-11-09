"""
Access Control Test Pass Rate: every principal × document × date, on every enforcement path.

Department ACL = acl_label per chunk (nhan_su / ke_toan / phap_che / confidential / public); the label
set a caller holds comes from data/tenants (server-side), never from the request.

Expected visibility comes from an independent oracle written from the policy, not from the code
under test. Three paths must each agree with it:
  chroma  — RetrievalContext.to_where() run by a real (ephemeral) Chroma collection
  qdrant  — the same dict through where_to_qdrant_filter(), run by an in-memory Qdrant
  bm25    — RetrievalContext.allows(), the in-process screen for keyword hits

    python eval/acl_eval.py [--output reports/acl_eval.json]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.context import PUBLIC_TENANT, RetrievalContext, stamp_access_meta  # noqa: E402

TENANTS = [PUBLIC_TENANT, "acme", "globex"]
LABELS = ["public", "nhan_su", "ke_toan", "phap_che", "confidential"]
PRINCIPALS = {
    "anonymous": RetrievalContext(),
    "acme.nhan_su": RetrievalContext(tenant_id="acme", acl_labels={"public", "nhan_su"}),
    "acme.ke_toan": RetrievalContext(tenant_id="acme", acl_labels={"public", "ke_toan"}),
    "acme.giam_doc": RetrievalContext(tenant_id="acme", acl_labels={"public", "nhan_su", "ke_toan", "phap_che", "confidential"}),
    "globex.ke_toan": RetrievalContext(tenant_id="globex", acl_labels={"public", "ke_toan"}),
    "globex.public_only": RetrievalContext(tenant_id="globex", acl_labels={"public"}),
}
DATES = ["2024-06-01", "2026-06-01"]
LIFECYCLE = {  # name -> (effective_from, effective_to, status)
    "current": ("2021-01-01", "9999-12-31", "in_force"),
    "superseded": ("2015-01-01", "2026-01-01", "in_force"),
    "future": ("2026-01-01", "9999-12-31", "in_force"),
    "repealed": ("2015-01-01", "9999-12-31", "repealed"),
}


def documents() -> list[tuple[str, dict, dict]]:
    """(id, metadata, spec) for every tenant × label × lifecycle combination."""
    out = []
    for tenant, label, (life, (frm, to, status)) in itertools.product(TENANTS, LABELS, LIFECYCLE.items()):
        if tenant == PUBLIC_TENANT and label != "public":
            continue  # the shared legal corpus is public by definition
        spec = {"tenant": tenant, "label": label, "from": frm, "to": to, "status": status}
        meta = stamp_access_meta({"effective_from": frm, "effective_to": to, "status": status},
                                 tenant_id=tenant, acl_label=label)
        out.append((f"{tenant}__{label}__{life}", meta, spec))
    return out


def oracle(principal: RetrievalContext, spec: dict, as_of: str) -> bool:
    """The policy in plain terms, independent of RetrievalContext's implementation."""
    tenant_ok = spec["tenant"] == PUBLIC_TENANT or (principal.tenant_id != PUBLIC_TENANT and spec["tenant"] == principal.tenant_id)
    label_ok = spec["label"] in principal.acl_labels
    time_ok = spec["from"] <= as_of < spec["to"] and spec["status"] != "repealed"
    return tenant_ok and label_ok and time_ok


def chroma_path(docs):
    import chromadb

    col = chromadb.EphemeralClient().create_collection(f"acl_eval_{uuid.uuid4().hex[:8]}")
    col.add(ids=[d[0] for d in docs], documents=[d[0] for d in docs], metadatas=[d[1] for d in docs],
            embeddings=[[0.1, 0.2, 0.3]] * len(docs))
    return lambda ctx: set(col.get(where=ctx.to_where())["ids"])


def qdrant_path(docs):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    from src.rag.vector_backend import where_to_qdrant_filter

    client = QdrantClient(":memory:")
    client.create_collection("acl", vectors_config=VectorParams(size=3, distance=Distance.COSINE))
    client.upsert("acl", [PointStruct(id=i, vector=[0.1, 0.2, 0.3], payload=d[1] | {"doc": d[0]})
                          for i, d in enumerate(docs)])

    def visible(ctx):
        pts, _ = client.scroll("acl", scroll_filter=where_to_qdrant_filter(ctx.to_where()), limit=10_000)
        return {p.payload["doc"] for p in pts}
    return visible


def bm25_path(docs):
    return lambda ctx: {d[0] for d in docs if ctx.allows(d[1])}


def run() -> dict:
    docs = documents()
    paths = {"chroma": chroma_path(docs), "qdrant": qdrant_path(docs), "bm25": bm25_path(docs)}
    checks, failures = {p: 0 for p in paths}, {p: [] for p in paths}
    leaks = {p: 0 for p in paths}
    for (pname, base), as_of in itertools.product(PRINCIPALS.items(), DATES):
        ctx = RetrievalContext(tenant_id=base.tenant_id, acl_labels=base.acl_labels, as_of_date=as_of)
        expected = {d[0] for d in docs if oracle(ctx, d[2], as_of)}
        for path, visible in paths.items():
            got = visible(ctx)
            for doc_id, _, _ in docs:
                checks[path] += 1
                if (doc_id in got) != (doc_id in expected):
                    leak = doc_id in got  # shown but must not be = leak; hidden but should show = over-block
                    leaks[path] += leak
                    failures[path].append({"principal": pname, "as_of": as_of, "doc": doc_id,
                                           "kind": "leak" if leak else "over_block"})
    report = {
        "documents": len(docs), "principals": len(PRINCIPALS), "dates": DATES,
        "paths": {p: {"checks": checks[p], "failures": len(failures[p]), "leaks": leaks[p],
                      "pass_rate": 1 - len(failures[p]) / checks[p], "examples": failures[p][:10]} for p in paths},
    }
    report["pass_rate"] = min(v["pass_rate"] for v in report["paths"].values())
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("reports/acl_eval.json"))
    r = run()
    out = ap.parse_args().output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    for p, v in r["paths"].items():
        print(f"{p:7s} checks={v['checks']:4d} failures={v['failures']} leaks={v['leaks']} pass_rate={v['pass_rate']:.4f}")
    print(f"Access control pass rate (worst path): {r['pass_rate']:.4f}")
    sys.exit(0 if r["pass_rate"] == 1.0 else 1)


if __name__ == "__main__":
    main()
