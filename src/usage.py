"""
Per-request token and cost accounting, independent of Langfuse.

Every LLM call already goes through langfuse_otel.record_generation(); it also calls add() here.
A route opens a meter with begin(), and end() returns the totals, which log_request() stores in
SQLite `usage_log`. /api/v1/metrics aggregates that table (latency percentiles, tokens, cost).

Prices are USD per 1M tokens (input, output), overridable with LLM_PRICES_JSON. A model with no
price is counted in tokens and listed under `unpriced_models` rather than silently costed at 0.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextvars import ContextVar, Token
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.config import get_settings

DEFAULT_PRICES = {  # USD / 1M tokens, public list prices; check before relying on them
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.0-flash": (0.10, 0.40),
}
_meter: ContextVar[dict | None] = ContextVar("usage_meter", default=None)


def prices() -> dict[str, tuple[float, float]]:
    extra = os.getenv("LLM_PRICES_JSON")
    return DEFAULT_PRICES | ({k: tuple(v) for k, v in json.loads(extra).items()} if extra else {})


def begin() -> Token:
    return _meter.set({"calls": 0, "failed_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "by_model": {}})


def add(model: str, prompt_tokens: int, completion_tokens: int, failed: bool = False) -> None:
    m = _meter.get()
    if m is None:  # eval scripts / tests calling the generator directly
        return
    m["calls"] += 1
    m["failed_calls"] += int(failed)
    m["prompt_tokens"] += prompt_tokens or 0
    m["completion_tokens"] += completion_tokens or 0
    pm = m["by_model"].setdefault(model or "unknown", [0, 0])
    pm[0] += prompt_tokens or 0
    pm[1] += completion_tokens or 0


def end(token: Token) -> dict:
    m = _meter.get() or {"calls": 0, "failed_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "by_model": {}}
    _meter.reset(token)
    table, cost, unpriced = prices(), 0.0, []
    for model, (pin, pout) in m["by_model"].items():
        price = table.get(model)
        if price is None:
            if pin or pout:
                unpriced.append(model)
            continue
        cost += pin / 1e6 * price[0] + pout / 1e6 * price[1]
    return m | {"cost_usd": cost, "unpriced_models": unpriced}


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(Path(get_settings().sqlite_db)))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS usage_log (created_at TEXT, channel TEXT, intent TEXT, latency_ms INTEGER, "
        "llm_calls INTEGER, failed_calls INTEGER, prompt_tokens INTEGER, completion_tokens INTEGER, "
        "cost_usd REAL, unpriced TEXT, error INTEGER)"
    )
    return conn


def log_request(channel: str, intent: str, latency_ms: int, totals: dict, error: bool = False) -> None:
    conn = _connect()
    try:
        with conn:
            conn.execute(
                "INSERT INTO usage_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(), channel, intent or "unknown", int(latency_ms),
                 totals["calls"], totals["failed_calls"], totals["prompt_tokens"], totals["completion_tokens"],
                 totals["cost_usd"], ",".join(totals["unpriced_models"]), int(error)),
            )
    finally:
        conn.close()


def _pct(values: list[int], p: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    return float(v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))])


def summary(hours: int = 24) -> dict:
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    conn = _connect()
    try:
        rows = conn.execute("SELECT channel, intent, latency_ms, llm_calls, failed_calls, prompt_tokens, "
                            "completion_tokens, cost_usd, unpriced, error FROM usage_log WHERE created_at >= ?",
                            (since,)).fetchall()
    finally:
        conn.close()
    n = len(rows)
    lat = [r[2] for r in rows]
    out = {
        "window_hours": hours, "requests": n,
        "latency_ms": {"p50": _pct(lat, 50), "p95": _pct(lat, 95)},
        "error_rate": sum(r[9] for r in rows) / n if n else None,
        "llm_calls": sum(r[3] for r in rows), "llm_call_failures": sum(r[4] for r in rows),
        "tokens": {"prompt": sum(r[5] for r in rows), "completion": sum(r[6] for r in rows)},
        "cost_usd": round(sum(r[7] for r in rows), 6),
        "cost_per_request_usd": round(sum(r[7] for r in rows) / n, 6) if n else None,
        "unpriced_models": sorted({m for r in rows for m in (r[8] or "").split(",") if m}),
        "by_intent": {},
    }
    for intent in sorted({r[1] for r in rows}):
        sub = [r for r in rows if r[1] == intent]
        out["by_intent"][intent] = {"requests": len(sub), "p95_ms": _pct([r[2] for r in sub], 95),
                                    "cost_usd": round(sum(r[7] for r in sub), 6)}
    return out
