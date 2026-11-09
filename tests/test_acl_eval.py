"""The ACL eval must pass on the real policy AND catch a broken one (otherwise 100% means nothing)."""
import importlib.util
from pathlib import Path

import pytest

from src.rag.context import RetrievalContext

spec = importlib.util.spec_from_file_location("acl_eval", Path(__file__).resolve().parents[1] / "eval" / "acl_eval.py")
acl_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acl_eval)


def test_every_path_matches_the_policy():
    r = acl_eval.run()
    for path, v in r["paths"].items():
        assert v["pass_rate"] == 1.0, (path, v["examples"])


@pytest.mark.parametrize("path", ["bm25", "chroma"])
def test_eval_detects_a_label_leak(monkeypatch, path):
    if path == "bm25":
        # Bug: BM25 screen forgets the department label.
        monkeypatch.setattr(RetrievalContext, "allows",
                            lambda self, meta: str(meta.get("tenant_id")) in self.visible_tenants)
    else:
        # Bug: where-clause drops the acl_label predicate.
        orig = RetrievalContext.to_where
        monkeypatch.setattr(RetrievalContext, "to_where",
                            lambda self: {"$and": [c for c in orig(self)["$and"] if "acl_label" not in c]})
    r = acl_eval.run()
    assert r["paths"][path]["leaks"] > 0
    assert r["pass_rate"] < 1.0
