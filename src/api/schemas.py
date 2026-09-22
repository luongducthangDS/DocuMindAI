"""
Pydantic v2 schemas for all API request/response bodies.
Strict validation to reject malformed input at the boundary.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ── Query ──────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    # min_length=2, không phải 3 — khớp đúng min-length của WS handler
    # (query.py::websocket_stream) và với lời chào ngắn nhất trong
    # graph.py::_SMALLTALK_RE ("hi", "ok"): 3 từng chặn "hi" trước khi kịp
    # tới nhánh smalltalk, trả về lỗi validate khó hiểu ngay câu hỏi đầu tiên.
    query: str = Field(..., min_length=2, max_length=1000)
    session_id: str = Field(default="default", max_length=64)
    stream: bool = False
    # Mốc thời điểm tra cứu (ISO YYYY-MM-DD). Thiếu = quy định hiện hành hôm nay.
    as_of_date: str | None = Field(default=None, max_length=10)

    @field_validator("as_of_date")
    @classmethod
    def validate_as_of_date(cls, v: str | None) -> str | None:
        if v is None or not v.strip():
            return None
        v = v.strip()
        try:
            date.fromisoformat(v)
        except ValueError:
            raise ValueError("as_of_date must be an ISO date (YYYY-MM-DD)") from None
        return v

    @field_validator("query")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        # Strip leading/trailing whitespace
        v = v.strip()
        # Reject strings that look like prompt injection attempts
        forbidden = ["<script", "javascript:", "data:text", "\\x00"]
        for f in forbidden:
            if f.lower() in v.lower():
                raise ValueError("Query contains forbidden content")
        return v

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("session_id must be alphanumeric with _ or -")
        return v


class SourceItem(BaseModel):
    index: int
    title: str = ""
    dieu_header: str = ""
    source_url: str = ""
    score: float = 0.0


class ThinkingStep(BaseModel):
    label: str
    detail: str = ""
    ms: int = 0


class ComplianceVerdict(BaseModel):
    matched: bool
    verdict: Literal["pass", "fail", "insufficient_info", "no_match"]
    criterion_id: str = ""
    extracted_value: float | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem] = []
    used_llm: str
    chunk_count: int
    latency_ms: int
    session_id: str
    steps: list[ThinkingStep] = []
    retry_count: int = 0
    grade_reason: str = ""
    compliance: ComplianceVerdict | None = None


# ── Documents ──────────────────────────────────────────────────────────────────

class DocumentMeta(BaseModel):
    id: str
    title: str
    doc_type: str
    source: str
    url: str = ""
    so_hieu: str = ""
    ngay_ban_hanh: str = ""
    chunk_count: int = 0


class DocumentListResponse(BaseModel):
    total: int
    documents: list[DocumentMeta]


class IngestResponse(BaseModel):
    status: Literal["success", "partial", "error"]
    indexed_chunks: int
    document_title: str
    message: str = ""


# ── Report ─────────────────────────────────────────────────────────────────────

class ReportRequest(BaseModel):
    title: str = Field(..., min_length=3, max_length=200)
    query: str = Field(..., min_length=3, max_length=500)
    filename: str = Field(default="report", max_length=60)

    @field_validator("filename")
    @classmethod
    def safe_filename(cls, v: str) -> str:
        # Only allow safe characters — prevent path traversal
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", v)
        return safe[:60] or "report"


class ReportResponse(BaseModel):
    status: Literal["success", "error"]
    filename: str
    download_url: str
    message: str = ""


# ── Health ─────────────────────────────────────────────────────────────────────

class ServiceStatus(BaseModel):
    name: str
    healthy: bool
    detail: str = ""


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    services: list[ServiceStatus]
    version: str = "1.0.0"


# ── Metrics ───────────────────────────────────────────────────────────────────

class MetricsResponse(BaseModel):
    total_queries: int
    avg_latency_ms: float
    error_rate: float
    active_sessions: int
    corpus_chunks: int
