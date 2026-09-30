"""
Shared pytest fixtures.
Uses temporary directories and mock API keys for isolation.
"""

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def patch_settings(tmp_path, monkeypatch):
    """Patch all settings to use tmp_path and dummy API keys."""
    monkeypatch.setenv("GOOGLE_API_KEY", "AIza_test_key")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("SQLITE_DB", str(tmp_path / "test.db"))
    monkeypatch.setenv("CHROMA_HOST", "localhost")
    monkeypatch.setenv("ENVIRONMENT", "development")
    # Test không bao giờ gửi trace ra ngoài: key thật trong .env máy dev từng làm
    # mỗi lượt pytest đẩy 23 batch lên Langfuse (đo 2026-09-30), trộn trace giả
    # (key Gemini giả, dữ liệu giả) vào trace thật. Env rỗng đè .env vì
    # pydantic-settings ưu tiên env; LANGCHAIN_TRACING_V2 langsmith đọc thẳng os.environ.
    for var in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGCHAIN_API_KEY"):
        monkeypatch.setenv(var, "")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")

    # Clear lru_cache so settings reload with test env
    from src.config import get_settings
    get_settings.cache_clear()

    yield

    get_settings.cache_clear()


@pytest.fixture
def sample_legal_text() -> str:
    return """
Điều 1. Phạm vi điều chỉnh
Luật này quy định về việc thành lập, tổ chức quản lý, tổ chức lại, giải thể và
hoạt động có liên quan của doanh nghiệp, bao gồm công ty trách nhiệm hữu hạn,
công ty cổ phần, công ty hợp danh và doanh nghiệp tư nhân.

Điều 2. Đối tượng áp dụng
Luật này áp dụng đối với:
1. Doanh nghiệp được thành lập, tổ chức và hoạt động tại Việt Nam.
2. Tổ chức, cá nhân liên quan đến thành lập, tổ chức, quản lý và hoạt động của doanh nghiệp.

Điều 3. Áp dụng Luật Doanh nghiệp và pháp luật có liên quan
Trường hợp luật khác có quy định đặc thù về việc thành lập, tổ chức quản lý,
tổ chức lại, giải thể và hoạt động có liên quan của doanh nghiệp thì áp dụng quy định
của luật đó.
"""


@pytest.fixture
def sample_doc_meta() -> dict:
    return {
        "title": "Luật Doanh nghiệp 2020",
        "url": "https://vbpl.vn/test",
        "doc_type": "luat",
        "source": "vbpl.vn",
        "so_hieu": "59/2020/QH14",
        "ngay_ban_hanh": "2020-06-17",
    }


# ── P0: không mạng + Gemini giả ───────────────────────────────────────────────

@pytest.fixture
def no_network(monkeypatch):
    """Mọi kết nối ra ngoài loopback đều ném lỗi (Langfuse đã tắt sẵn ở patch_settings).

    Loopback vẫn cho phép: event loop trên Windows tự nối socketpair qua 127.0.0.1.
    """
    import socket

    real_connect = socket.socket.connect

    def guarded_connect(sock, address):
        host = address[0] if isinstance(address, tuple) else address
        if sock.family == getattr(socket, "AF_UNIX", object()) or host in ("127.0.0.1", "::1", "localhost"):
            return real_connect(sock, address)
        raise RuntimeError(f"Test không được ra mạng: {address!r}")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


class _FakeStream(list):
    usage_metadata = None


class FakeGemini:
    """Thay SDK google.generativeai (configure + GenerativeModel) — không phải
    gemini_call — nên code thật phía trên (xoay key × model, cooldown, stream
    thread) vẫn chạy nguyên vẹn.

    `responder(prompt, model_name)` trả về str (câu trả lời) hoặc một exception
    (được ném ra như SDK thật). `calls` ghi lại (api_key, model, prompt, stream).
    """

    def __init__(self):
        self.calls: list[SimpleNamespace] = []
        self.responder = lambda prompt, model_name: "Câu trả lời giả [1]."
        self._key = None

    def configure(self, api_key=None, **_kw):
        self._key = api_key

    def GenerativeModel(self, model_name, **_kw):  # noqa: N802 — khớp tên SDK
        fake = self

        class _Model:
            def generate_content(self, prompt, request_options=None, stream=False, **_kw):
                # gemini_call gắn client theo key vào model (không qua configure).
                key = getattr(getattr(self, "_client", None), "api_key", fake._key)
                fake.calls.append(SimpleNamespace(
                    api_key=key, model=model_name, prompt=prompt, stream=stream))
                out = fake.responder(prompt, model_name)
                if isinstance(out, BaseException):
                    raise out
                if stream:
                    return _FakeStream(SimpleNamespace(text=w + " ") for w in out.split(" "))
                return SimpleNamespace(text=out, usage_metadata=None)

        return _Model()

    def prompts_containing(self, needle: str) -> list[str]:
        return [c.prompt for c in self.calls if needle in c.prompt]


@pytest.fixture
def fake_gemini(no_network, monkeypatch) -> FakeGemini:
    """3 key × 2 model giả, cooldown và cursor xoay vòng reset về trắng."""
    import google.generativeai as genai

    from src.config import get_settings
    from src.rag import generator

    monkeypatch.setenv("GOOGLE_API_KEY", "key-A")
    monkeypatch.setenv("GOOGLE_API_KEY_2", "key-B")
    monkeypatch.setenv("GOOGLE_API_KEY_3", "key-C")
    monkeypatch.setenv("GEMINI_GENERATION_MODELS", "model-x,model-y")
    get_settings.cache_clear()

    fake = FakeGemini()
    monkeypatch.setattr(genai, "configure", fake.configure)
    monkeypatch.setattr(genai, "GenerativeModel", fake.GenerativeModel)
    def client_for(api_key):
        fake._key = api_key  # responder của test cũ đọc _key như thời còn configure()
        return SimpleNamespace(api_key=api_key)

    monkeypatch.setattr("src.rag.embedder.genai_client", client_for)
    monkeypatch.setattr(generator, "_PAIR_DOWN_UNTIL", {})
    monkeypatch.setattr(generator, "_GEMINI_PAIR_CURSOR", 0)
    return fake


@pytest.fixture
def real_criteria(tmp_path):
    """criteria.json thật, chép vào DATA_DIR tạm của patch_settings."""
    import shutil

    from src.rag import compliance

    real = Path(__file__).resolve().parents[1] / "data" / "compliance" / "criteria.json"
    dest = tmp_path / "data" / "compliance"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(real, dest / "criteria.json")
    compliance.load_criteria.cache_clear()
    yield
    compliance.load_criteria.cache_clear()


@pytest.fixture
def mem_retriever(monkeypatch):
    """Factory: dựng Chroma in-memory + BM25 thật từ list TextNode và gắn vào
    singleton của src.rag.retriever. Embedding là MockEmbedding (hằng số) — lọc
    tenant/ACL/hiệu lực vẫn chạy thật trong Chroma; BM25 quyết định thứ hạng.

    Trả về hàm `install(nodes) -> index`; `index.p0_rebuild_bm25()` dựng lại BM25
    từ nội dung collection (thay cho _rebuild_retriever, vốn đọc backend thật).
    """
    import uuid

    import chromadb
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.core.embeddings import MockEmbedding
    from llama_index.core.schema import TextNode
    from llama_index.retrievers.bm25 import BM25Retriever
    from llama_index.vector_stores.chroma import ChromaVectorStore

    import src.rag.retriever as r_module

    def install(nodes):
        # Cấu hình mặc định: Chroma giữ MỘT instance ephemeral mỗi process và từ chối
        # instance thứ hai khác settings — test_access_control dùng mặc định.
        client = chromadb.EphemeralClient()
        col = client.create_collection(f"p0_{uuid.uuid4().hex[:8]}")
        store = ChromaVectorStore(chroma_collection=col)
        index = VectorStoreIndex.from_vector_store(
            store, embed_model=MockEmbedding(embed_dim=8),
            storage_context=StorageContext.from_defaults(vector_store=store),
        )
        index.insert_nodes(list(nodes))

        def rebuild_bm25(*_a):
            got = col.get(include=["documents", "metadatas"])
            all_nodes = [TextNode(id_=i, text=t, metadata=m)
                         for i, t, m in zip(got["ids"], got["documents"], got["metadatas"])]
            monkeypatch.setattr(r_module, "_bm25_retriever",
                                BM25Retriever.from_defaults(nodes=all_nodes, similarity_top_k=20))

        index.p0_rebuild_bm25 = rebuild_bm25
        monkeypatch.setattr(r_module, "_active_index", index)
        monkeypatch.setattr(r_module, "_reranker_instance", None)
        monkeypatch.setattr(r_module, "_reranker_active", False)
        rebuild_bm25()
        return index

    return install
