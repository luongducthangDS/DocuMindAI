"""Tests for eval/query_cache.py — vector câu hỏi embed sẵn.

Cache này tồn tại để eval chạy được ở nơi không load nổi model. Rủi ro đi kèm:
dùng nhầm vector của model khác mà vẫn ra điểm cosine trông hợp lệ. Phần lớn
test dưới đây canh đúng chỗ đó.
"""

from __future__ import annotations

import json

import pytest

from eval.query_cache import (
    CACHE_FORMAT,
    QueryEmbeddingCache,
    QueryEmbeddingCacheError,
    build_payload,
    question_hash,
)

MODEL = "AITeamVN/Vietnamese_Embedding"
Q1 = "Mức lương tối thiểu vùng I là bao nhiêu?"
Q2 = "Tuổi nghỉ hưu năm 2026 là bao nhiêu?"


def payload(questions=None, embeddings=None, model=MODEL):
    questions = questions or [Q1, Q2]
    embeddings = embeddings or [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    return build_payload(questions, embeddings, model)


class TestQuestionHash:
    def test_whitespace_is_normalised(self):
        assert question_hash("a  b\n c") == question_hash("a b c")

    def test_different_wording_is_a_different_question(self):
        assert question_hash(Q1) != question_hash(Q2)

    def test_empty_is_stable(self):
        assert question_hash("") == question_hash("   ")


class TestBuildPayload:
    def test_records_model_and_dim(self):
        p = payload()
        assert p["model"] == MODEL
        assert p["dim"] == 3
        assert p["format"] == CACHE_FORMAT
        assert len(p["queries"]) == 2

    def test_extra_metadata_is_kept(self):
        p = build_payload([Q1], [[0.1]], MODEL, extra={"git_commit": "abc123"})
        assert p["git_commit"] == "abc123"

    def test_count_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="2 câu hỏi nhưng 1 vector"):
            build_payload([Q1, Q2], [[0.1]], MODEL)

    def test_ragged_dimensions_are_rejected(self):
        with pytest.raises(ValueError, match="không cùng số chiều"):
            build_payload([Q1, Q2], [[0.1, 0.2], [0.3]], MODEL)


class TestCacheLoad:
    def test_lookup_returns_the_vector(self):
        cache = QueryEmbeddingCache(payload())
        assert cache.get(Q1) == [0.1, 0.2, 0.3]
        assert cache.get(Q2) == [0.4, 0.5, 0.6]

    def test_lookup_survives_whitespace_edits_in_gold(self):
        cache = QueryEmbeddingCache(payload())
        assert cache.get(f"  {Q1}  ") == [0.1, 0.2, 0.3]

    def test_wrong_model_is_refused(self):
        """Đây là ca quan trọng nhất: vector của model khác vẫn cho ra cosine
        hợp lệ, nên nếu không chặn ở đây thì sai sẽ im lặng."""
        with pytest.raises(QueryEmbeddingCacheError, match="sinh bằng"):
            QueryEmbeddingCache(payload(), expected_model="some/other-model")

    def test_matching_model_is_accepted(self):
        cache = QueryEmbeddingCache(payload(), expected_model=MODEL)
        assert cache.model == MODEL

    def test_unknown_format_is_refused(self):
        p = payload()
        p["format"] = 99
        with pytest.raises(QueryEmbeddingCacheError, match="cache format"):
            QueryEmbeddingCache(p)

    def test_empty_cache_is_refused(self):
        p = payload()
        p["queries"] = []
        with pytest.raises(QueryEmbeddingCacheError, match="rỗng"):
            QueryEmbeddingCache(p)

    def test_missing_question_names_the_fix(self):
        cache = QueryEmbeddingCache(payload())
        with pytest.raises(QueryEmbeddingCacheError, match="build_query_embedding_cache"):
            cache.get("Câu hỏi chưa từng được embed")

    def test_covers_lists_what_is_missing(self):
        cache = QueryEmbeddingCache(payload())
        missing = cache.covers([Q1, "câu mới", Q2])
        assert missing == ["câu mới"]

    def test_covers_empty_when_complete(self):
        cache = QueryEmbeddingCache(payload())
        assert cache.covers([Q1, Q2]) == []

    def test_load_from_file(self, tmp_path):
        path = tmp_path / "cache.json"
        path.write_text(json.dumps(payload(), ensure_ascii=False), encoding="utf-8")
        cache = QueryEmbeddingCache.load(path, expected_model=MODEL)
        assert len(cache) == 2
        assert cache.dim == 3

    def test_missing_file_names_the_path(self, tmp_path):
        with pytest.raises(QueryEmbeddingCacheError, match="không thấy file cache"):
            QueryEmbeddingCache.load(tmp_path / "nope.json")


class TestCachedEmbedder:
    def test_serves_queries_without_loading_a_model(self):
        from eval.query_cache import make_cached_embedder

        embedder = make_cached_embedder(QueryEmbeddingCache(payload()))
        assert embedder.get_query_embedding(Q1) == [0.1, 0.2, 0.3]

    def test_refuses_to_embed_new_documents(self):
        """Eval chỉ truy vấn corpus đã index. Nếu có gì đó đòi embed văn bản mới
        thì giả định 'không cần model' đã sai — phải nổ, không được trả vector rác."""
        from eval.query_cache import make_cached_embedder

        embedder = make_cached_embedder(QueryEmbeddingCache(payload()))
        with pytest.raises(QueryEmbeddingCacheError, match="không embed được"):
            embedder.get_text_embedding("một đoạn văn bản mới")
