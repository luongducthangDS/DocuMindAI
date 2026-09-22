"""Gửi trace lên Langfuse qua OTLP/HTTP thuần — KHÔNG dùng SDK chính thức.

Lý do (xem thêm src.config.Settings.langfuse_host): langfuse-python (OTel-based,
>=4.x) đòi opentelemetry-api/sdk>=1.33.1; nâng hai gói đó trong venv này để lại
opentelemetry-exporter-otlp-proto-grpc==1.27.0 (dependency của chromadb) lệch
version với chúng — đã thử cài thật, ImportError phá 17 test không liên quan gì
tới LLM. Module này tự dựng JSON theo đúng OTLP property mapping Langfuse công bố
(https://langfuse.com/integrations/native/opentelemetry) và gửi bằng `requests`
(đã là dependency sẵn có) tới endpoint OTel — endpoint Legacy Ingestion API cũ
(/api/public/ingestion) sunset trên Langfuse Cloud 16/11/2026.

Một trace = một lượt gọi run_agent() (hoặc một lượt trả lời WS). start_trace()
mở context (contextvar — sống xuyên suốt cây gọi hàm/await bên trong request đó,
kể cả khi LangGraph chạy node sync qua loop.run_in_executor, vì contextvars được
Python tự propagate vào đó). Mọi record_generation()/record_span() gọi bên trong
context này tự nối vào đúng trace; gọi ngoài context (vd. eval script gọi thẳng
generate_answer()) thì tự tạo trace đứng riêng — giữ hành vi cũ.
"""

from __future__ import annotations

import contextvars
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone

from loguru import logger

from src.config import get_settings

_ctx: contextvars.ContextVar["TraceCtx | None"] = contextvars.ContextVar(
    "langfuse_trace_ctx", default=None
)

# Đủ để thấy full prompt/context RAG khi debug trong Langfuse UI, vẫn chặn payload
# phình to bất thường (vd. câu hỏi dán nguyên văn bản dài).
_MAX_FIELD_CHARS = 8_000


@dataclass
class TraceCtx:
    trace_id: str
    root_span_id: str
    session_id: str | None
    tags: list[str]
    started_at: datetime
    spans: list[dict] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add_span(self, span: dict) -> None:
        with self._lock:
            self.spans.append(span)


def _auth() -> tuple[str, str, str] | None:
    """None nếu chưa cấu hình keys — quan sát là tuỳ chọn."""
    s = get_settings()
    if not (s.langfuse_public_key and s.langfuse_secret_key):
        return None
    return s.langfuse_public_key, s.langfuse_secret_key, s.langfuse_host.rstrip("/")


def _nanos(dt: datetime) -> str:
    # OTLP JSON mã hoá uint64 (nanosecond epoch) thành string để tránh mất độ
    # chính xác khi parser JS đọc số.
    return str(int(dt.timestamp() * 1_000_000_000))


def _trunc(text: str | None) -> str:
    if not text:
        return ""
    return text if len(text) <= _MAX_FIELD_CHARS else text[:_MAX_FIELD_CHARS] + "…"


def _attr(key: str, value) -> dict:
    if isinstance(value, bool):
        v = {"boolValue": value}
    elif isinstance(value, int):
        v = {"intValue": value}
    elif isinstance(value, list):
        v = {"arrayValue": {"values": [{"stringValue": str(x)} for x in value]}}
    else:
        v = {"stringValue": str(value)}
    return {"key": key, "value": v}


def _trace_level_attrs(session_id: str | None, tags: list[str]) -> list[dict]:
    # Phải có mặt trên MỌI span trong trace (không chỉ root) để Langfuse filter/
    # aggregate theo session/tag đúng — theo property mapping docs.
    attrs = []
    if session_id:
        attrs.append(_attr("langfuse.session.id", session_id))
    if tags:
        attrs.append(_attr("langfuse.trace.tags", tags))
    return attrs


def _build_span(
    trace_id: str,
    span_id: str,
    parent_span_id: str | None,
    name: str,
    started_at: datetime,
    ended_at: datetime,
    attributes: list[dict],
) -> dict:
    span = {
        "traceId": trace_id,
        "spanId": span_id,
        "name": name,
        "kind": 1,  # SPAN_KIND_INTERNAL
        "startTimeUnixNano": _nanos(started_at),
        "endTimeUnixNano": _nanos(ended_at),
        "attributes": attributes,
        "status": {"code": 1},  # STATUS_CODE_OK
    }
    if parent_span_id:
        span["parentSpanId"] = parent_span_id
    return span


def _send_batch(spans: list[dict]) -> None:
    """Gửi trong thread riêng — network call không cộng vào latency câu trả lời
    thật; lỗi ở đây chỉ log debug, không lan ra ngoài (quan sát không được phép
    làm hỏng luồng chính)."""
    auth = _auth()
    if auth is None or not spans:
        return
    public_key, secret_key, host = auth

    def _fire() -> None:
        import requests

        payload = {
            "resourceSpans": [{
                "resource": {"attributes": []},
                "scopeSpans": [{
                    "scope": {"name": "documind-ai", "version": "1.0.0"},
                    "spans": spans,
                }],
            }]
        }
        try:
            requests.post(
                f"{host}/api/public/otel/v1/traces",
                json=payload,
                auth=(public_key, secret_key),
                headers={"x-langfuse-ingestion-version": "4"},
                timeout=5,
            )
        except Exception as exc:
            logger.debug("Langfuse OTLP ingestion failed (non-fatal): {}", exc)

    threading.Thread(target=_fire, daemon=True).start()


def start_trace(
    name: str, session_id: str | None = None, tags: list[str] | None = None
) -> tuple[TraceCtx, contextvars.Token]:
    """Mở 1 trace (vd. đầu run_agent()). Trả về (ctx, token) — token bắt buộc để
    end_trace() reset lại contextvar, không rò rỉ context sang request khác."""
    ctx = TraceCtx(
        trace_id=secrets.token_hex(16),
        root_span_id=secrets.token_hex(8),
        session_id=session_id,
        tags=list(tags or []),
        started_at=datetime.now(timezone.utc),
    )
    token = _ctx.set(ctx)
    return ctx, token


def end_trace(
    ctx: TraceCtx,
    token: contextvars.Token,
    root_name: str,
    input_text: str,
    output_text: str,
    extra_tags: list[str] | None = None,
) -> None:
    """Đóng trace: build root span (input/output = câu hỏi gốc/câu trả lời cuối,
    KHÔNG phải prompt nội bộ — đây là "meaningful trace input/output" ở mức
    trace) rồi gửi 1 batch cùng toàn bộ child spans đã ghi nhận trong lúc chạy."""
    _ctx.reset(token)
    if _auth() is None:
        return
    ended_at = datetime.now(timezone.utc)
    tags = ctx.tags + list(extra_tags or [])
    attrs = _trace_level_attrs(ctx.session_id, tags) + [
        _attr("langfuse.trace.name", root_name),
        _attr("langfuse.observation.input", _trunc(input_text)),
        _attr("langfuse.observation.output", _trunc(output_text)),
    ]
    root_span = _build_span(
        ctx.trace_id, ctx.root_span_id, None, root_name, ctx.started_at, ended_at, attrs
    )
    _send_batch([root_span] + ctx.spans)


def record_generation(
    name: str,
    model: str,
    input_text: str,
    output_text: str,
    started_at: datetime,
    ended_at: datetime,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> None:
    """Ghi 1 observation kiểu 'generation' (một lời gọi Gemini). Trong context
    của start_trace() thì nối vào trace đó làm con của root span; ngoài context
    (vd. eval script/test gọi thẳng generate_answer()) thì tự tạo 1 trace đứng
    riêng cho lần gọi này — đúng hành vi REST-thuần cũ trước khi có start_trace()."""
    if _auth() is None:
        return
    ctx = _ctx.get()
    trace_id = ctx.trace_id if ctx else secrets.token_hex(16)
    parent = ctx.root_span_id if ctx else None
    session_id = ctx.session_id if ctx else None
    tags = ctx.tags if ctx else []

    attrs = _trace_level_attrs(session_id, tags) + [
        _attr("langfuse.observation.type", "generation"),
        _attr("langfuse.observation.model.name", model),
        _attr("langfuse.observation.input", _trunc(input_text)),
        _attr("langfuse.observation.output", _trunc(output_text)),
        _attr(
            "langfuse.observation.usage_details",
            f'{{"input":{prompt_tokens},"output":{completion_tokens},'
            f'"total":{prompt_tokens + completion_tokens}}}',
        ),
    ]
    span = _build_span(trace_id, secrets.token_hex(8), parent, name, started_at, ended_at, attrs)
    if ctx is not None:
        ctx.add_span(span)
    else:
        _send_batch([span])


def record_span(
    name: str,
    type_: str,
    input_text: str,
    output_text: str,
    started_at: datetime,
    ended_at: datetime,
) -> None:
    """Ghi 1 observation không phải generation (vd. type_='retriever'/'tool').
    Chỉ có ý nghĩa lồng trong 1 trace — bỏ qua nếu gọi ngoài start_trace()."""
    if _auth() is None:
        return
    ctx = _ctx.get()
    if ctx is None:
        return
    attrs = _trace_level_attrs(ctx.session_id, ctx.tags) + [
        _attr("langfuse.observation.type", type_),
        _attr("langfuse.observation.input", _trunc(input_text)),
        _attr("langfuse.observation.output", _trunc(output_text)),
    ]
    span = _build_span(
        ctx.trace_id, secrets.token_hex(8), ctx.root_span_id, name, started_at, ended_at, attrs
    )
    ctx.add_span(span)
