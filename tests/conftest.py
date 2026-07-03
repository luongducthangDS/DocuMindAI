"""
Shared pytest fixtures.
Uses temporary directories and mock API keys for isolation.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def patch_settings(tmp_path, monkeypatch):
    """Patch all settings to use tmp_path and dummy API keys."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")
    monkeypatch.setenv("GOOGLE_API_KEY", "AIza_test_key")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("SQLITE_DB", str(tmp_path / "test.db"))
    monkeypatch.setenv("CHROMA_HOST", "localhost")
    monkeypatch.setenv("ENVIRONMENT", "development")

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
