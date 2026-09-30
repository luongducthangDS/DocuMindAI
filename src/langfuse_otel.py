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

import atexit
import contextvars
import functools
import json
import os
import re
import secrets
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from src.config import get_settings

_ctx: contextvars.ContextVar["TraceCtx | None"] = contextvars.ContextVar(
    "langfuse_trace_ctx", default=None
)

# Đủ để thấy full prompt/context RAG khi debug trong Langfuse UI, vẫn chặn payload
# phình to bất thường (vd. câu hỏi dán nguyên văn bản dài).
_MAX_FIELD_CHARS = 8_000

# provider/collection/model/số chunk — main._init_rag_sync đặt lúc khởi tạo RAG, để
# trace trả lời được "câu này chạy trên index nào". Rỗng = chưa khởi tạo (script eval).
_index_version = ""


def set_index_version(value: str) -> None:
    global _index_version
    _index_version = value


@functools.lru_cache(maxsize=1)
def app_release() -> str:
    """git SHA của code đang chạy → `release` của trace ("lỗi này thuộc bản nào").

    Render cấp RENDER_GIT_COMMIT lúc chạy (cả runtime Docker); máy dev hỏi git, thêm
    "-dirty" khi có thay đổi chưa commit — trace đó không tái hiện được chỉ từ SHA.
    Image Docker không có .git → "unknown"."""
    sha = os.getenv("RENDER_GIT_COMMIT", "")[:12]
    if sha:
        return sha
    root = Path(__file__).resolve().parents[1]
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=root, capture_output=True, text=True, timeout=3,
        ).stdout.strip()
        if not sha:
            return "unknown"
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=root, timeout=3).returncode != 0
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return "unknown"


def _environment() -> str:
    """Langfuse chỉ nhận environment khớp ^(?!langfuse)[a-z0-9-_]+$, ≤ 40 ký tự."""
    env = re.sub(r"[^a-z0-9_-]", "-", str(get_settings().environment).lower())[:40]
    return env if env and not env.startswith("langfuse") else "default"


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


def tracing_active() -> bool:
    """Có trace đang mở và đã cấu hình key — không thì đừng tốn công dựng payload."""
    return _ctx.get() is not None and _auth() is not None


def non_fatal(fn):
    """Cho hàm dựng payload trace từ dữ liệu của request: lỗi ở đó chỉ log debug, không
    bao giờ làm hỏng câu trả lời — cùng nguyên tắc với _send_batch. Lỗi → trả None."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            logger.debug("Trace {} failed (non-fatal): {}", fn.__name__, exc)
            return None
    return wrapper


def _nanos(dt: datetime) -> str:
    # OTLP JSON mã hoá uint64 (nanosecond epoch) thành string để tránh mất độ
    # chính xác khi parser JS đọc số.
    return str(int(dt.timestamp() * 1_000_000_000))


def _trunc(text: str | None) -> str:
    """Mọi input/output/lỗi gửi Langfuse đều qua đây — nên che PII ở đây (NĐ 13/2023)."""
    from src.guardrails import redact_pii

    if not text:
        return ""
    text = redact_pii(text)
    return text if len(text) <= _MAX_FIELD_CHARS else text[:_MAX_FIELD_CHARS] + "…"


def _text(value) -> str:
    """Input/output của observation: dict/list → JSON (Langfuse hiện dạng cây), chuỗi giữ nguyên."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)


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
    # aggregate theo session/tag đúng — theo property mapping docs. environment là
    # thuộc tính của từng observation: thiếu ở span nào, span đó rơi về "default".
    attrs = [
        _attr("langfuse.environment", _environment()),
        _attr("langfuse.release", app_release()),
    ]
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


_SEND_THREAD = "langfuse-send"
# Trần cứng cho việc gửi nốt lúc thoát — TỔNG mọi batch còn dở, không phải mỗi batch.
_EXIT_FLUSH_S = 5.0
# Lỗi tạm thời (429, 5xx, mạng) thì gửi lại sau từng khoảng chờ này (giây), rồi bỏ.
_RETRY_BACKOFF_S = (0.5, 2.0)


def _deliver(url: str, payload: dict, auth: tuple[str, str]) -> None:
    """POST một batch. Không bao giờ ném ra ngoài, nhưng batch bị bỏ phải để lại
    WARNING: requests không ném với 4xx/5xx, nên trước đây key sai = mọi trace mất mà
    không một dòng log (đo 2026-09-30: Langfuse trả 401, log rỗng)."""
    import requests

    problem = ""
    for wait in (*_RETRY_BACKOFF_S, None):
        try:
            resp = requests.post(url, json=payload, auth=auth,
                                 headers={"x-langfuse-ingestion-version": "4"}, timeout=5)
            if resp.ok:
                return
            problem = f"HTTP {resp.status_code}: {resp.text[:200]}"
            if resp.status_code != 429 and resp.status_code < 500:
                break  # 4xx (key sai, payload hỏng): gửi lại cũng vậy
        except Exception as exc:  # mạng chập chờn, timeout
            problem = f"{type(exc).__name__}: {exc}"
        if wait is not None:
            time.sleep(wait)
    logger.warning("Langfuse bỏ 1 batch trace — {}", problem)


def _send_batch(spans: list[dict]) -> None:
    """Gửi trong thread riêng — network call không cộng vào latency câu trả lời
    thật; lỗi không lan ra ngoài (quan sát không được phép làm hỏng luồng chính).

    Thread daemon + _flush_on_exit: daemon trần bị giết ngay khi script kết thúc
    (đo 2026-09-30: 3/3 generation gửi ngay trước khi thoát không tới Langfuse);
    không daemon thì Python chờ không giới hạn — DNS treo (timeout của requests không
    bao phần phân giải tên) giữ process lại. Chờ lúc thoát, có trần cứng, là đủ cả hai."""
    auth = _auth()
    if auth is None or not spans:
        return
    public_key, secret_key, host = auth
    payload = {
        "resourceSpans": [{
            "resource": {"attributes": []},
            "scopeSpans": [{
                "scope": {"name": "documind-ai", "version": "1.0.0"},
                "spans": spans,
            }],
        }]
    }
    threading.Thread(
        target=lambda: _deliver(f"{host}/api/public/otel/v1/traces", payload, (public_key, secret_key)),
        daemon=True, name=_SEND_THREAD,
    ).start()


@atexit.register
def _flush_on_exit() -> None:
    """Chờ các batch còn đang gửi, tổng tối đa _EXIT_FLUSH_S giây; quá trần thì bỏ —
    và nói ra là đã bỏ bao nhiêu, không để mất im lặng.

    atexit chạy TRƯỚC khi Python giết thread daemon (CPython Py_FinalizeEx), nên
    join ở đây vẫn kịp. threading.enumerate() chỉ trả thread đang sống — không cần
    tự giữ danh sách."""
    deadline = time.monotonic() + _EXIT_FLUSH_S
    for thread in threading.enumerate():
        if thread.name == _SEND_THREAD:
            thread.join(max(0.0, deadline - time.monotonic()))
    left = sum(t.is_alive() for t in threading.enumerate() if t.name == _SEND_THREAD)
    if left:
        logger.warning("Thoát khi còn {} batch trace chưa gửi xong sau {}s — bỏ", left, _EXIT_FLUSH_S)


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
    if _index_version:
        attrs.append(_attr("langfuse.trace.metadata.index", _index_version))
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
    error: str | None = None,
    level: str = "ERROR",
) -> None:
    """Ghi 1 observation kiểu 'generation' (một lời gọi Gemini). Trong context
    của start_trace() thì nối vào trace đó làm con của root span; ngoài context
    (vd. eval script/test gọi thẳng generate_answer()) thì tự tạo 1 trace đứng
    riêng cho lần gọi này — đúng hành vi REST-thuần cũ trước khi có start_trace().

    `error`: lần gọi thất bại (429/503/timeout...) — vẫn ghi, để trace thấy được
    thời gian đốt vào xoay vòng key×model thay vì một khoảng trống. `level`:
    WARNING cho lượt thử mà vòng xoay còn đi tiếp — không phải lỗi của câu hỏi,
    không được thổi phồng số ERROR trên Langfuse."""
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
    if error:
        attrs += [
            _attr("langfuse.observation.level", level),
            _attr("langfuse.observation.status_message", _trunc(error)),
        ]
    span = _build_span(trace_id, secrets.token_hex(8), parent, name, started_at, ended_at, attrs)
    if error and level == "ERROR":
        span["status"] = {"code": 2, "message": _trunc(error)}  # STATUS_CODE_ERROR
    if ctx is not None:
        ctx.add_span(span)
    else:
        _send_batch([span])


def record_span(
    name: str,
    type_: str,
    input_text,
    output_text,
    started_at: datetime,
    ended_at: datetime,
    level: str | None = None,
) -> None:
    """Ghi 1 observation không phải generation (vd. type_='retriever'/'tool').
    Input/output nhận chuỗi hoặc dict (ghi thành JSON). Chỉ có ý nghĩa lồng trong
    1 trace — bỏ qua nếu gọi ngoài start_trace()."""
    if _auth() is None:
        return
    ctx = _ctx.get()
    if ctx is None:
        return
    attrs = _trace_level_attrs(ctx.session_id, ctx.tags) + [
        _attr("langfuse.observation.type", type_),
        _attr("langfuse.observation.input", _trunc(_text(input_text))),
        _attr("langfuse.observation.output", _trunc(_text(output_text))),
    ]
    if level:
        attrs.append(_attr("langfuse.observation.level", level))
    span = _build_span(
        ctx.trace_id, secrets.token_hex(8), ctx.root_span_id, name, started_at, ended_at, attrs
    )
    ctx.add_span(span)
