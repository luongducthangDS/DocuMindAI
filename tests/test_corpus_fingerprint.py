"""corpus_fingerprint — tells two indexes with the same chunk count apart."""
import json

from src.rag.vector_backend import _parse_qdrant_payload, corpus_fingerprint

A = ("Điều 1. Phạm vi điều chỉnh", {"doc_id": "45-2019-QH14", "effective_from_i": 20210101, "status": "in_force"})
B = ("Điều 2. Đối tượng áp dụng", {"doc_id": "45-2019-QH14", "effective_from_i": 20210101, "status": "in_force"})


def test_order_does_not_matter():
    assert corpus_fingerprint([A, B]) == corpus_fingerprint([B, A])


def test_text_or_metadata_change_shows():
    base = corpus_fingerprint([A, B])
    assert corpus_fingerprint([A, (B[0] + " (sửa)", B[1])]) != base
    assert corpus_fingerprint([A, (B[0], {**B[1], "effective_from_i": 20260101})]) != base


def test_status_ignored_because_it_only_reflects_the_ingest_date():
    assert corpus_fingerprint([A, (B[0], {**B[1], "status": "superseded"})]) == corpus_fingerprint([A, B])


def test_qdrant_copy_hashes_like_the_chroma_original():
    """Migrated index must hash the same: llama-index payload shape, ints kept as ints."""
    text, meta = A
    payload = {**meta, "_node_content": json.dumps({"text": text, "metadata": meta}),
               "_node_type": "TextNode", "doc_id": "None", "ref_doc_id": "None", "document_id": "None"}
    assert corpus_fingerprint([_parse_qdrant_payload(payload)]) == corpus_fingerprint([A])
