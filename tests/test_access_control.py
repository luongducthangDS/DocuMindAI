"""
Access control: tenant isolation, ACL labels, and point-in-time filtering.

These three share one mechanism (RetrievalContext -> a metadata predicate), so
they are tested together. The assertions that matter most are the negative
ones: a filter that returns the right rows but also returns a forbidden one is
not a partial pass, it is a leak.
"""

from __future__ import annotations

import pytest

from src.api.principal import InvalidApiKey, context_from_headers
from src.rag.context import (
    EFFECTIVE_TO_OPEN_INT,
    PUBLIC_TENANT,
    RetrievalContext,
    date_ints_for,
    stamp_access_meta,
    to_date_int,
)
from src.rag.vector_backend import where_to_qdrant_filter


def meta(**kw) -> dict:
    base = {
        "tenant_id": PUBLIC_TENANT,
        "acl_label": "public",
        "effective_from": "2021-01-01",
        "effective_to": "9999-12-31",
        "status": "in_force",
    }
    base.update(kw)
    return stamp_access_meta(base, tenant_id=base["tenant_id"], acl_label=base["acl_label"])


class TestDateInts:
    def test_iso_to_int(self):
        assert to_date_int("2021-01-01") == 20210101
        assert to_date_int("") is None
        assert to_date_int("not-a-date") is None

    def test_open_ended_uses_sentinel(self):
        assert date_ints_for({"effective_to": "9999-12-31"})["effective_to_i"] == EFFECTIVE_TO_OPEN_INT

    def test_missing_from_is_always_in_force(self):
        # Mirrors temporal.is_in_force: an undatable chunk is kept, not dropped.
        assert date_ints_for({})["effective_from_i"] == 0

    def test_int_mirror_tracks_a_closed_window(self):
        # The superseded-clause case: the mirror must not stay at the sentinel.
        m = meta(effective_to="2026-01-01")
        assert m["effective_to_i"] == 20260101


class TestTenantIsolation:
    def test_tenant_sees_public_corpus_and_its_own(self):
        ctx = RetrievalContext(tenant_id="acme")
        assert ctx.allows(meta(tenant_id=PUBLIC_TENANT))
        assert ctx.allows(meta(tenant_id="acme"))

    def test_tenant_cannot_see_another_tenant(self):
        ctx = RetrievalContext(tenant_id="acme")
        assert not ctx.allows(meta(tenant_id="globex"))

    def test_anonymous_sees_only_public(self):
        ctx = RetrievalContext()
        assert ctx.allows(meta(tenant_id=PUBLIC_TENANT))
        assert not ctx.allows(meta(tenant_id="acme"))

    def test_where_clause_scopes_tenants(self):
        clauses = RetrievalContext(tenant_id="acme").to_where()["$and"]
        tenant_clause = next(c for c in clauses if "tenant_id" in c)
        assert set(tenant_clause["tenant_id"]["$in"]) == {"public", "acme"}


class TestAclLabels:
    def test_label_not_held_is_denied(self):
        ctx = RetrievalContext(tenant_id="acme", acl_labels={"public"})
        assert not ctx.allows(meta(tenant_id="acme", acl_label="confidential"))

    def test_label_held_is_allowed(self):
        ctx = RetrievalContext(tenant_id="acme", acl_labels={"public", "confidential"})
        assert ctx.allows(meta(tenant_id="acme", acl_label="confidential"))

    def test_empty_labels_fall_back_to_public_only(self):
        # An empty grant must not mean "unrestricted".
        ctx = RetrievalContext(tenant_id="acme", acl_labels=set())
        assert ctx.acl_labels == frozenset({"public"})
        assert not ctx.allows(meta(tenant_id="acme", acl_label="internal"))

    def test_context_is_hashable(self):
        # It is used as a cache key and must keep value semantics.
        assert hash(RetrievalContext(acl_labels=["public"])) == hash(RetrievalContext())


class TestPointInTime:
    def test_superseded_clause_hidden_after_its_window(self):
        old = meta(effective_from="2015-01-01", effective_to="2026-01-01")
        assert RetrievalContext(as_of_date="2025-06-01").allows(old)
        assert not RetrievalContext(as_of_date="2026-06-01").allows(old)

    def test_future_clause_hidden_before_it_starts(self):
        new = meta(effective_from="2026-01-01")
        assert not RetrievalContext(as_of_date="2025-06-01").allows(new)
        assert RetrievalContext(as_of_date="2026-06-01").allows(new)

    def test_repealed_never_shown(self):
        assert not RetrievalContext(as_of_date="2022-01-01").allows(
            meta(status="repealed", effective_from="2015-01-01")
        )

    def test_where_and_allows_agree(self):
        """The Python predicate and the pushed-down filter must not diverge.

        They are two implementations of one rule; a mismatch means the store
        returns rows the app thinks are forbidden, or the reverse.
        """
        ctx = RetrievalContext(tenant_id="acme", acl_labels={"public", "internal"},
                               as_of_date="2025-06-01")
        as_of_i = 20250601
        rows = [
            meta(tenant_id="acme", acl_label="internal"),
            meta(tenant_id="globex"),
            meta(acl_label="confidential"),
            meta(effective_from="2015-01-01", effective_to="2026-01-01"),
            meta(effective_from="2026-01-01"),
            meta(status="repealed"),
        ]
        for row in rows:
            by_where = (
                row["tenant_id"] in {"public", "acme"}
                and row["acl_label"] in {"public", "internal"}
                and row["effective_from_i"] <= as_of_i
                and row["effective_to_i"] > as_of_i
                and row["status"] not in {"repealed"}
            )
            assert ctx.allows(row) is by_where, row


class TestPrincipalResolution:
    def test_no_key_is_anonymous_public(self):
        ctx = context_from_headers({})
        assert ctx.tenant_id == PUBLIC_TENANT
        assert ctx.acl_labels == frozenset({"public"})

    def test_known_key_resolves_to_its_tenant(self):
        ctx = context_from_headers({"x-api-key": "demo-globex-0000"})
        assert ctx.tenant_id == "globex"
        assert "confidential" in ctx.acl_labels

    def test_unknown_key_is_rejected(self):
        with pytest.raises(InvalidApiKey):
            context_from_headers({"x-api-key": "not-a-real-key"})

    def test_client_cannot_grant_itself_labels(self):
        """Headers naming labels directly must have no effect — the server decides."""
        ctx = context_from_headers({"x-api-key": "demo-acme-0000",
                                    "x-acl-labels": "confidential",
                                    "x-tenant-id": "globex"})
        assert ctx.tenant_id == "acme"
        assert "confidential" not in ctx.acl_labels


class TestQdrantTranslation:
    def test_every_operator_survives_translation(self):
        f = where_to_qdrant_filter(RetrievalContext(tenant_id="acme").to_where())
        assert len(f.must) == 4      # tenant, acl, from, to
        assert len(f.must_not) == 1  # repealed

    def test_unknown_operator_raises_instead_of_dropping(self):
        # Dropping it would run an unfiltered query with an ACL clause present.
        with pytest.raises(ValueError, match="Unsupported filter operator"):
            where_to_qdrant_filter({"$and": [{"x": {"$regex": "y"}}]})

    def test_empty_where_is_none(self):
        assert where_to_qdrant_filter(None) is None


class TestAmbientContext:
    def test_default_is_public_not_unrestricted(self):
        """A path that never sets a context must read public data, not everything."""
        from src.rag.context import current_context

        assert current_context().tenant_id == PUBLIC_TENANT
        assert current_context().acl_labels == frozenset({"public"})

    def test_set_and_reset(self):
        from src.rag.context import current_context, reset_current_context, set_current_context

        token = set_current_context(RetrievalContext(tenant_id="acme"))
        try:
            assert current_context().tenant_id == "acme"
        finally:
            reset_current_context(token)
        assert current_context().tenant_id == PUBLIC_TENANT

    @pytest.mark.asyncio
    async def test_context_survives_to_thread(self):
        """The graph runs sync nodes via to_thread — context must cross that boundary."""
        import asyncio

        from src.rag.context import current_context, reset_current_context, set_current_context

        token = set_current_context(RetrievalContext(tenant_id="globex"))
        try:
            seen = await asyncio.to_thread(lambda: current_context().tenant_id)
        finally:
            reset_current_context(token)
        assert seen == "globex"


class TestWhereClauseAgainstRealStore:
    """The filter has to hold up in the store, not just as a dict.

    chromadb 0.6.3 rejects $lte/$gt on strings, which is why effective dates are
    mirrored to integers at all — a test that only inspected the dict would have
    happily passed against a filter the database refuses to run.
    """

    @pytest.fixture
    def collection(self):
        import uuid

        import chromadb

        # EphemeralClient shares state within a process, so each test gets its
        # own collection name rather than colliding with the previous one.
        client = chromadb.EphemeralClient()
        col = client.create_collection(f"acl_test_{uuid.uuid4().hex[:8]}",
                                       metadata={"hnsw:space": "cosine"})
        rows = [
            ("public_current", meta()),
            ("public_superseded", meta(effective_from="2015-01-01", effective_to="2026-01-01")),
            ("public_future", meta(effective_from="2026-01-01")),
            ("public_repealed", meta(status="repealed")),
            ("acme_internal", meta(tenant_id="acme", acl_label="internal")),
            ("globex_confidential", meta(tenant_id="globex", acl_label="confidential")),
        ]
        col.add(
            ids=[r[0] for r in rows],
            documents=[f"nội dung {r[0]}" for r in rows],
            metadatas=[r[1] for r in rows],
            embeddings=[[0.1, 0.2, 0.3]] * len(rows),
        )
        return col

    def ids_for(self, collection, ctx) -> set[str]:
        return set(collection.get(where=ctx.to_where()).get("ids") or [])

    def test_anonymous_sees_only_public_and_current(self, collection):
        got = self.ids_for(collection, RetrievalContext(as_of_date="2025-06-01"))
        assert got == {"public_current", "public_superseded"}

    def test_tenant_sees_public_plus_own(self, collection):
        ctx = RetrievalContext(tenant_id="acme", acl_labels={"public", "internal"},
                               as_of_date="2025-06-01")
        got = self.ids_for(collection, ctx)
        assert "acme_internal" in got
        assert "globex_confidential" not in got

    def test_tenant_without_label_cannot_reach_own_confidential_doc(self, collection):
        ctx = RetrievalContext(tenant_id="globex", acl_labels={"public"},
                               as_of_date="2025-06-01")
        assert "globex_confidential" not in self.ids_for(collection, ctx)

    def test_point_in_time_flips_the_visible_set(self, collection):
        before = self.ids_for(collection, RetrievalContext(as_of_date="2025-06-01"))
        after = self.ids_for(collection, RetrievalContext(as_of_date="2026-06-01"))
        assert "public_superseded" in before and "public_superseded" not in after
        assert "public_future" not in before and "public_future" in after

    def test_repealed_is_never_visible_at_any_date(self, collection):
        for as_of in ("2016-01-01", "2025-06-01", "2026-06-01"):
            assert "public_repealed" not in self.ids_for(
                collection, RetrievalContext(as_of_date=as_of)
            )
