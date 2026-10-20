"""OP-06: httpx transport guard for the OneProvider Anthropic-compatible gateway.

The gateway occasionally answers HTTP 200 with an EMPTY message (no content,
output_tokens 0, stop_reason end_turn). The Anthropic SDK only retries 429/5xx
and connection errors, so an empty 200 reaches the caller as a silent blank
reply. This transport:

1. retries empty replies a bounded number of times (non-streaming /v1/messages only),
2. logs served-model mismatches (deduplicated per pair per hour),
3. keeps per-model latency and status statistics.

Everything here fails open: if inspection of a response raises, the response is
returned untouched. Request/response bodies and headers are never logged.

Installed only in gateway mode (see Config.anthropic_client_kwargs). This module
must NOT import agent.config at module top.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import threading
import time
from collections import deque

import anthropic
import httpx

logger = logging.getLogger("openclaw.anthropic_gateway")

_LOG_INTERVAL_S = 3600.0
_ERROR_LOG_INTERVAL_S = 600.0
_LATENCY_WINDOW = 500
_MESSAGES_SUFFIX = "/v1/messages"
_BACKOFF_BASE_S = 0.4
_DEFAULT_EMPTY_RETRIES = 2
_MAX_EMPTY_RETRIES = 3
_ENV_EMPTY_RETRIES = "CLIO_ANTHROPIC_GATEWAY_EMPTY_RETRIES"


def _pct(sorted_vals: list[int], q: float) -> int:
    n = len(sorted_vals)
    idx = min(n - 1, max(0, math.ceil(q * n) - 1))
    return int(sorted_vals[idx])


def _safe_model(value: str) -> str:
    """Keep log output to plain model-name characters, bounded in length."""
    kept = "".join(ch for ch in value if ch.isalnum() or ch in "._:/-")
    return kept[:64] or "?"


def _safe_host(value: str) -> str:
    kept = "".join(ch for ch in str(value) if ch.isalnum() or ch in ".-")
    return kept[:96] or "?"


def _safe_token(value, limit: int = 80) -> str:
    """Identifier-like token only (request ids, error types): [A-Za-z0-9_.-], bounded."""
    if not isinstance(value, str):
        return "-"
    kept = "".join(ch for ch in value if ch.isalnum() or ch in "_.-")
    return kept[:limit] or "-"


def _family_of(model: str) -> str | None:
    """Same family regex as oneprovider_balance (lazy import: keep this module light)."""
    from agent.integrations.oneprovider_balance import _family

    return _family(model)


def _error_type(raw: bytes) -> str:
    """Only error.type from a JSON error body. Message text is never returned."""
    try:
        body = json.loads(raw)
    except Exception:
        return "-"
    if not isinstance(body, dict):
        return "-"
    err = body.get("error")
    if not isinstance(err, dict):
        return "-"
    return _safe_token(err.get("type"))


class GatewayStats:
    """Thread-safe counters, status histogram and per-model latency window."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls = 0
        self.empty_retries = 0
        self.empty_final = 0
        self.mismatches = 0
        self.alias_same_family = 0
        self.family_mismatch = 0
        self._status: dict[int, int] = {}
        self._hosts: dict[str, int] = {}
        self._latency: dict[str, deque] = {}
        self._mismatch_logged: dict[tuple, float] = {}
        self._error_logged: dict[tuple[int, str], float] = {}
        self._last_log: float | None = None

    def add_empty_retry(self) -> None:
        with self._lock:
            self.empty_retries += 1

    def add_empty_final(self) -> None:
        with self._lock:
            self.empty_final += 1

    def record_call(self, model: str, latency_ms: float, status: int) -> None:
        with self._lock:
            self.calls += 1
            code = int(status)
            self._status[code] = self._status.get(code, 0) + 1
            dq = self._latency.get(model)
            if dq is None:
                dq = deque(maxlen=_LATENCY_WINDOW)
                self._latency[model] = dq
            dq.append(max(0, int(latency_ms)))

    def record_host(self, host: str) -> None:
        with self._lock:
            key = _safe_host(host)
            self._hosts[key] = self._hosts.get(key, 0) + 1

    def should_log_error(self, status: int, error_type: str, now: float) -> bool:
        """Dedupe non-200 WARNINGs per (status, error_type) per _ERROR_LOG_INTERVAL_S."""
        with self._lock:
            key = (int(status), error_type)
            last = self._error_logged.get(key)
            if last is not None and (now - last) < _ERROR_LOG_INTERVAL_S:
                return False
            self._error_logged[key] = now
            return True

    def note_mismatch(self, requested: str, served: str, now: float, family_change: bool = False) -> bool:
        """Count a mismatch (split alias vs family change); return True if this pair should be logged now."""
        with self._lock:
            self.mismatches += 1
            if family_change:
                self.family_mismatch += 1
            else:
                self.alias_same_family += 1
            key = (bool(family_change), requested, served)
            last = self._mismatch_logged.get(key)
            if last is not None and (now - last) < _LOG_INTERVAL_S:
                return False
            self._mismatch_logged[key] = now
            return True

    def _snapshot_locked(self) -> dict:
        latency = {}
        for model, dq in self._latency.items():
            vals = sorted(dq)
            if vals:
                latency[model] = {
                    "n": len(vals),
                    "p50": _pct(vals, 0.5),
                    "p95": _pct(vals, 0.95),
                }
        return {
            "calls": self.calls,
            "empty_retries": self.empty_retries,
            "empty_final": self.empty_final,
            "mismatches": self.mismatches,
            "alias_same_family": self.alias_same_family,
            "family_mismatch": self.family_mismatch,
            "status": dict(self._status),
            "hosts": dict(self._hosts),
            "latency_ms": latency,
        }

    def snapshot(self) -> dict:
        with self._lock:
            return self._snapshot_locked()

    def maybe_log(self, now: float) -> bool:
        """Emit one INFO stats line at most once per hour. Returns True if emitted."""
        with self._lock:
            if self._last_log is not None and (now - self._last_log) < _LOG_INTERVAL_S:
                return False
            self._last_log = now
            snap = self._snapshot_locked()
        _emit_stats_line(snap)
        return True

    def emit_stats_tick(self, now: float) -> None:
        """Hourly job path: emit the stats line unconditionally (also with calls=0)."""
        with self._lock:
            self._last_log = now
            snap = self._snapshot_locked()
        _emit_stats_line(snap)


def _emit_stats_line(snap: dict) -> None:
    lat = ",".join(
        f"{model}={v['p50']}/{v['p95']}" for model, v in sorted(snap["latency_ms"].items())
    ) or "-"
    status = ",".join(f"{k}:{v}" for k, v in sorted(snap["status"].items())) or "-"
    hosts = ",".join(f"{h}:{n}" for h, n in sorted(snap["hosts"].items()))
    # The prefix up to `status={...}` is parsed by scripts/morning_check.py (_GATEWAY_STATS_RE
    # and the calls/empty_retries/empty_final fields). Append-only: do not reorder.
    logger.info(
        "gateway stats: calls=%d empty_retries=%d empty_final=%d mismatches=%d "
        "latency_ms_p50/p95 by model=%s status={%s} family_mismatch=%d alias=%d hosts={%s}",
        snap["calls"],
        snap["empty_retries"],
        snap["empty_final"],
        snap["mismatches"],
        lat,
        status,
        snap["family_mismatch"],
        snap["alias_same_family"],
        hosts,
    )


_STATS = GatewayStats()


def get_stats() -> GatewayStats:
    """Process-wide stats instance."""
    return _STATS


def emit_stats_tick(now: float | None = None) -> None:
    """Emit the stats line now, regardless of calls since the last line."""
    _STATS.emit_stats_tick(time.monotonic() if now is None else now)


async def gateway_stats_tick_job(context=None) -> None:
    """PTB JobQueue callback (hourly). Emits the stats line even with calls=0. Never raises."""
    try:
        if os.getenv("CLIO_GATEWAY_MONITOR_ENABLED", "1") == "0":
            return
        emit_stats_tick()
    except Exception:
        logger.debug("gateway stats tick failed (fail-open)")


def _eligible_body(request: httpx.Request) -> dict | None:
    """Return the parsed JSON body if this request should be inspected, else None."""
    try:
        if request.method != "POST":
            return None
        if not request.url.path.endswith(_MESSAGES_SUFFIX):
            return None
        if not isinstance(request.stream, httpx.ByteStream):
            return None
        body = json.loads(request.read())
    except Exception:
        return None
    if not isinstance(body, dict):
        return None
    if body.get("stream"):
        return None
    return body


def _block_has_content(block) -> bool:
    if not isinstance(block, dict):
        return False
    btype = block.get("type")
    if btype == "text":
        text = block.get("text")
        return isinstance(text, str) and bool(text.strip())
    if btype in ("thinking", "redacted_thinking"):
        for key in ("thinking", "data"):
            value = block.get(key)
            if isinstance(value, str) and value.strip():
                return True
        return False
    # tool_use and any unknown/new block type (server_tool_use, web_search_tool_result, ...)
    # count as content: never treat an unrecognised block as an empty reply.
    return True


def _is_empty_message(payload: dict) -> bool:
    if payload.get("type") != "message":
        return False
    content = payload.get("content")
    if isinstance(content, list):
        for block in content:
            if _block_has_content(block):
                return False
    usage = payload.get("usage")
    out = usage.get("output_tokens") if isinstance(usage, dict) else None
    if not (out is None or (type(out) is int and out == 0)):
        return False
    return payload.get("stop_reason") in ("end_turn", None)


def _inspect(resp: httpx.Response) -> tuple[dict | None, bool]:
    """Return (parsed message or None, is_empty). Only 200 JSON bodies are read."""
    if resp.status_code != 200:
        return None, False
    ctype = resp.headers.get("content-type", "")
    if "json" not in ctype.lower():
        return None, False
    raw = resp.read()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        return None, False
    return payload, _is_empty_message(payload)


class GatewayTransport(httpx.BaseTransport):
    """Wraps an httpx transport; guards non-streaming POST /v1/messages only."""

    def __init__(
        self,
        inner: httpx.BaseTransport | None = None,
        *,
        retry_empty: bool = True,
        max_empty_retries: int = _DEFAULT_EMPTY_RETRIES,
        sleep=time.sleep,
        clock=time.monotonic,
        stats: GatewayStats | None = None,
        rng=random.random,
    ) -> None:
        self._inner = inner if inner is not None else httpx.HTTPTransport(retries=0)
        self._retry_empty = bool(retry_empty)
        self._max_empty_retries = max(0, int(max_empty_retries))
        self._sleep = sleep
        self._clock = clock
        self._stats = stats if stats is not None else get_stats()
        self._rng = rng

    def _note_host(self, request: httpx.Request) -> None:
        try:
            self._stats.record_host(request.url.host)
        except Exception:
            logger.debug("gateway host note skipped (fail-open)")

    def _note_status(self, request: httpx.Request, requested: str | None, resp: httpx.Response) -> None:
        """Non-200: one deduped WARNING with status/endpoint/model/request-id/error.type only.

        The body is read (the SDK reads error bodies too) but only error.type is kept;
        message text, headers and keys are never logged. Fail-open.
        """
        try:
            status = int(resp.status_code)
            if status == 200:
                return
            etype = _error_type(resp.read())
            if not self._stats.should_log_error(status, etype, self._clock()):
                return
            rid = _safe_token(resp.headers.get("x-request-id") or resp.headers.get("request-id"))
            path = "".join(ch for ch in request.url.path if ch.isalnum() or ch in "/._-")[:128] or "?"
            logger.warning(
                "gateway http_status=%d endpoint=%s model=%s request_id=%s error_type=%s",
                status,
                path,
                _safe_model(requested) if requested else "-",
                rid,
                etype,
            )
        except Exception:
            logger.debug("gateway status note skipped (fail-open)")

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self._note_host(request)
        body = _eligible_body(request)
        if body is None:
            resp = self._inner.handle_request(request)
            self._note_status(request, None, resp)
            return resp

        requested = body.get("model")
        requested = requested if isinstance(requested, str) else None
        attempts = 1 + (self._max_empty_retries if self._retry_empty else 0)

        for attempt in range(attempts):
            t0 = self._clock()
            resp = self._inner.handle_request(request)
            try:
                payload, empty = _inspect(resp)
            except Exception:
                logger.debug("gateway inspection skipped (fail-open)")
                return resp

            if empty and attempt + 1 < attempts:
                self._stats.add_empty_retry()
                try:
                    resp.close()
                except Exception:
                    logger.debug("gateway discarded-response close failed")
                self._sleep(_BACKOFF_BASE_S * (2 ** attempt) * (0.75 + 0.5 * self._rng()))
                continue

            if empty:
                self._stats.add_empty_final()
            latency_ms = (self._clock() - t0) * 1000.0
            self._note_status(request, requested, resp)
            try:
                self._observe(requested, payload, resp.status_code, latency_ms)
            except Exception:
                logger.debug("gateway observation failed (fail-open)")
            return resp

        raise RuntimeError("unreachable: gateway retry loop exited without a response")

    def _observe(self, requested: str | None, payload: dict | None, status: int, latency_ms: float) -> None:
        self._stats.record_call(_safe_model(requested or "unknown"), latency_ms, status)
        now = self._clock()
        if payload is not None and requested:
            served = payload.get("model")
            if isinstance(served, str) and served and served != requested:
                req_s, srv_s = _safe_model(requested), _safe_model(served)
                fam_req = _family_of(req_s)
                family_change = fam_req is None or fam_req != _family_of(srv_s)
                if family_change:
                    if self._stats.note_mismatch(req_s, srv_s, now, family_change=True):
                        logger.warning("gateway served family changed: requested=%s served=%s", req_s, srv_s)
                elif self._stats.note_mismatch(req_s, srv_s, now, family_change=False):
                    logger.info("gateway served model differs: requested=%s served=%s", req_s, srv_s)
        self._stats.maybe_log(now)

    def close(self) -> None:
        self._inner.close()


_shared_lock = threading.Lock()
_shared_client: anthropic.DefaultHttpxClient | None = None


def _empty_retries_from_env() -> int:
    raw = os.environ.get(_ENV_EMPTY_RETRIES)
    if raw is None:
        return _DEFAULT_EMPTY_RETRIES
    try:
        value = int(raw.strip())
    except (ValueError, AttributeError):
        return _DEFAULT_EMPTY_RETRIES
    return min(_MAX_EMPTY_RETRIES, max(0, value))


def get_shared_http_client() -> anthropic.DefaultHttpxClient:
    """Thread-safe lazy singleton: a DefaultHttpxClient carrying the gateway transport."""
    global _shared_client
    with _shared_lock:
        if _shared_client is None:
            transport = GatewayTransport(
                retry_empty=True,
                max_empty_retries=_empty_retries_from_env(),
            )
            _shared_client = anthropic.DefaultHttpxClient(transport=transport)
        return _shared_client
