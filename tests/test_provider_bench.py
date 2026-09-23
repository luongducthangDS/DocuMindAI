"""eval/provider_bench.py — ground truth phải áp đúng predicate đã gửi cho provider."""
import numpy as np

from eval.provider_bench import exact_topk, matches
from src.rag.context import RetrievalContext


def _iso(d):
    return "9999-12-31" if d == 99991231 else f"{d // 10000}-{d // 100 % 100:02d}-{d % 100:02d}"


def _meta(frm, to, status="in_force", tenant="public", acl="public"):
    # chunk thật mang cả ngày ISO (allows/temporal.py đọc) lẫn bản int (filter đẩy xuống đọc)
    return {"tenant_id": tenant, "acl_label": acl, "status": status,
            "effective_from": _iso(frm), "effective_to": _iso(to),
            "effective_from_i": frm, "effective_to_i": to}


def test_matches_follows_retrieval_context_where():
    where = RetrievalContext(as_of_date="2025-03-01").to_where()
    assert matches(where, _meta(20240701, 20260101))           # đang hiệu lực
    assert not matches(where, _meta(20260101, 99991231))       # chưa hiệu lực
    assert not matches(where, _meta(20150101, 20250301))       # hết đúng ngày → $gt loại
    assert not matches(where, _meta(20240701, 20260101, tenant="acme"))
    assert matches(None, {})


def test_matches_agrees_with_allows_on_corpus_shapes():
    ctx = RetrievalContext(as_of_date="2025-03-01")
    for m in (_meta(20240701, 20260101), _meta(20260101, 99991231), _meta(20150101, 20250301)):
        assert matches(ctx.to_where(), m) == ctx.allows(m)


def test_exact_topk_ranks_only_allowed_chunks():
    emb = np.eye(3, dtype=np.float32)
    metas = [_meta(20260101, 99991231), _meta(20240701, 20260101), _meta(20240701, 20260101)]
    q = np.array([1.0, 0.5, 0.1])
    where = RetrievalContext(as_of_date="2025-03-01").to_where()
    assert exact_topk(emb, metas, ["a", "b", "c"], q, None, k=2) == ["a", "b"]
    assert exact_topk(emb, metas, ["a", "b", "c"], q, where, k=2) == ["b", "c"]
