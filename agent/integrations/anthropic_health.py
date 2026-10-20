"""
Anthropic API health check — detects credit-balance exhaustion and other
hard-blocker errors that silently kill downstream pipelines.

Two signals:
    1. Log-tail scan for "credit balance" / "credit_balance" strings in
       recent openclaw.log — catches already-happened 400s.
    2. Haiku ping (max_tokens=5) — proves the key is live RIGHT NOW.

Debounces to one Telegram alert per 6h per state transition. State file:
    data/anthropic_health_state.json

Flag: CLIO_ANTHROPIC_HEALTH_ENABLED (default "1"). Set to "0" to disable
the scheduled check entirely. No mutating behavior — observation only.

Runtime contract:
    check_anthropic_health() returns:
        {
            "healthy": bool,
            "recent_400_count": int,     # log lines with credit markers (legacy)
            "transport_400_count": int | None,  # R-11: gateway HTTP 400s this process (cumulative)
            "ping_ok": bool | None,      # None = ping skipped
            "ping_latency_ms": int | None,
            "reason": str | None,        # populated when unhealthy
            "checked_at": iso_utc_str,
        }
"""

import asyncio
import fcntl
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any

import anthropic

from agent.config import config

logger = logging.getLogger("openclaw.anthropic_health")

_LOG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "logs", "openclaw.log")
)
_STATE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "anthropic_health_state.json")
)
_LOCK_PATH = _STATE_PATH + ".lock"

_ALERT_DEBOUNCE_SEC = 6 * 3600
_LOG_TAIL_LINES = 500
_CREDIT_MARKERS = (
    "credit balance",
    "credit_balance",
    "insufficient balance",
    "insufficient_balance",
    "api key expired",
)
_PING_MODEL = "claude-haiku-4-5-20251001"


def _enabled() -> bool:
    return os.getenv("CLIO_ANTHROPIC_HEALTH_ENABLED", "1") == "1"


def _tail_lines_sync(path: str, n: int) -> list[str]:
    """Read last n lines of a file. Blocking — run via executor."""
    try:
        if not os.path.isfile(path):
            return []
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = 64 * 1024
            data = b""
            while size > 0 and data.count(b"\n") <= n + 1:
                read = min(block, size)
                size -= read
                f.seek(size)
                data = f.read(read) + data
        return data.decode("utf-8", errors="replace").splitlines()[-n:]
    except Exception as e:
        logger.debug(f"log tail failed: {e}")
        return []


def _count_credit_markers_sync(lines: list[str]) -> int:
    count = 0
    for line in lines:
        lo = line.lower()
        if any(marker in lo for marker in _CREDIT_MARKERS):
            count += 1
    return count


def _read_state_sync() -> dict:
    lock_fd = None
    try:
        lock_fd = open(_LOCK_PATH, "a")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        if os.path.isfile(_STATE_PATH):
            with open(_STATE_PATH, "r") as f:
                return json.load(f)
        return {"last_alert_ts": 0, "state": "unknown"}
    except Exception as e:
        logger.debug(f"state read failed: {e}")
        return {"last_alert_ts": 0, "state": "unknown"}
    finally:
        if lock_fd:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
            except Exception:
                pass


def _write_state_sync(state: dict) -> None:
    lock_fd = None
    try:
        lock_fd = open(_LOCK_PATH, "a")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        tmp = _STATE_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, _STATE_PATH)
    except Exception as e:
        logger.debug(f"state write failed: {e}")
    finally:
        if lock_fd:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
            except Exception:
                pass


# P1-17: module-level cached Anthropic client (matches sibling modules' _cached_client
# pattern) — reused across pings instead of building a new client per ping. Keeps timeout=120.
_cached_client: anthropic.Anthropic | None = None
_client_lock = threading.Lock()


def _get_client() -> anthropic.Anthropic:
    global _cached_client
    if _cached_client is None:
        with _client_lock:
            if _cached_client is None:
                _cached_client = anthropic.Anthropic(**config.anthropic_client_kwargs(timeout=120))
    return _cached_client


def _error_body_type(e: Exception) -> str:
    """Return the gateway error.type / error.code from an SDK exception body, lowercased.
    Empty string when the body is absent or not the expected shape."""
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            return f"{err.get('type') or ''} {err.get('code') or ''}".lower()
    return ""


def _ping_sync() -> tuple[bool, int | None, str | None]:
    """Send a tiny Haiku ping. Returns (ok, latency_ms, reason_if_failed).

    Reasons are FIXED WORDS (plus the generic fallback) — never a key or header.
    """
    try:
        client = _get_client()
    except Exception as e:
        return False, None, f"client_init: {e}"
    t0 = time.monotonic()
    try:
        client.messages.create(
            model=_PING_MODEL,
            max_tokens=5,
            messages=[{"role": "user", "content": "ping"}],
        )
        return True, int((time.monotonic() - t0) * 1000), None
    except anthropic.AuthenticationError:
        return False, None, "auth_invalid"
    except anthropic.PermissionDeniedError:
        return False, None, "key_forbidden_or_expired"
    except anthropic.RateLimitError as e:
        msg = str(e).lower()
        if "insufficient" in msg or "insufficient_balance" in _error_body_type(e):
            return False, None, "credit_balance_exhausted"
        return False, None, "rate_limited"
    except anthropic.BadRequestError as e:
        msg = str(e)
        if any(m in msg.lower() for m in _CREDIT_MARKERS):
            return False, None, "credit_balance_exhausted"
        return False, None, f"bad_request: {msg[:120]}"
    except (anthropic.InternalServerError, anthropic.APIConnectionError):
        # APITimeoutError is a subclass of APIConnectionError.
        return False, None, "provider_unavailable"
    except Exception as e:
        return False, None, f"{type(e).__name__}: {str(e)[:120]}"


def _transport_400_count() -> int | None:
    """R-11: HTTP 400s the gateway transport has seen in this process (status histogram of
    the shared stats; cumulative since process start). None when unavailable, never a fake 0."""
    try:
        from agent.anthropic_gateway import get_stats

        value = get_stats().snapshot()["status"].get(400, 0)
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return value
    except Exception:
        return None


async def check_anthropic_health() -> dict[str, Any]:
    """Run a full health check. Safe to call from anywhere; offloads all
    blocking work to the default executor.
    """
    loop = asyncio.get_running_loop()
    checked_at = datetime.now(timezone.utc).isoformat()

    tail = await loop.run_in_executor(None, _tail_lines_sync, _LOG_PATH, _LOG_TAIL_LINES)
    recent_400_count = await loop.run_in_executor(None, _count_credit_markers_sync, tail)
    transport_400_count = _transport_400_count()

    ping_ok, ping_latency_ms, ping_err = await loop.run_in_executor(None, _ping_sync)

    # S190 H1: healthy verdict considers both ping AND recent 400 count.
    # Previous logic returned healthy=True whenever the synthetic Haiku ping
    # succeeded, even if production traffic was hitting 400s (credit-balance-low
    # pattern). The ping is a tiny payload that can pass token accounting while
    # real workload fails — treat any 400s in the window as degraded.
    healthy = ping_ok and recent_400_count == 0
    if healthy:
        reason = None
    elif not ping_ok:
        reason = ping_err or ("credit_balance_exhausted" if recent_400_count > 0 else "ping_failed")
    else:
        reason = f"credit_balance_low ({recent_400_count} 400s in {_LOG_TAIL_LINES}-line window)"

    return {
        "healthy": healthy,
        "recent_400_count": recent_400_count,
        "transport_400_count": transport_400_count,
        "ping_ok": ping_ok,
        "ping_latency_ms": ping_latency_ms,
        "reason": reason,
        "checked_at": checked_at,
    }


async def anthropic_health_check_job(context) -> None:
    """PTB JobQueue callback. Runs the check, persists state, and sends one
    Telegram alert per state transition per 6h debounce window.
    """
    if not _enabled():
        return

    try:
        result = await check_anthropic_health()
    except Exception as e:
        logger.debug(f"health check raised: {e}")
        return

    loop = asyncio.get_running_loop()
    state = await loop.run_in_executor(None, _read_state_sync)

    prior_state = state.get("state", "unknown")
    current_state = "healthy" if result["healthy"] else "unhealthy"
    # Fail toward the alarm on a corrupt stored debounce ts (partial write / manual
    # edit / crash): float() raises ValueError/TypeError on a truthy non-numeric value
    # (the `or 0` only neutralizes falsy). This callback is registered DIRECTLY (no
    # _with_timeout broad except), so an unguarded raise here would abort before the
    # UNHEALTHY alert send AND the state rewrite, silently blacking out the monitor and
    # re-raising every tick. Treating a corrupt ts as 0.0 makes debounce_ok True (a real
    # UNHEALTHY still alerts) and the state rewrite at :221 self-heals the file.
    try:
        last_alert_ts = float(state.get("last_alert_ts", 0) or 0)
    except (ValueError, TypeError):
        logger.warning("anthropic_health: corrupt last_alert_ts, treating as 0 (fail toward alarm)")
        last_alert_ts = 0.0
    now_ts = time.time()

    state_changed = prior_state != current_state
    debounce_ok = (now_ts - last_alert_ts) > _ALERT_DEBOUNCE_SEC

    should_alert = not result["healthy"] and (state_changed or debounce_ok)

    new_state = {
        "state": current_state,
        "last_alert_ts": now_ts if should_alert else last_alert_ts,
        "last_check": result,
    }
    await loop.run_in_executor(None, _write_state_sync, new_state)

    if should_alert:
        try:
            from agent.integrations.scheduled_tasks import _get_chat_ids, _safe_send
            text = (
                f"⚠️ CLIO Anthropic health: UNHEALTHY\n"
                f"Reason: {result.get('reason') or 'unknown'}\n"
                f"Recent 400s in log: {result['recent_400_count']}\n"
                f"Ping ok: {result['ping_ok']}\n"
                f"Time: {result['checked_at']}"
            )
            for cid in _get_chat_ids():
                await _safe_send(context.bot, cid, text=text)
        except Exception as e:
            logger.debug(f"alert send failed: {e}")

    if state_changed and result["healthy"]:
        logger.info(
            f"anthropic_health: recovered — ping_latency_ms={result.get('ping_latency_ms')}"
        )
    elif not result["healthy"]:
        logger.warning(
            f"anthropic_health: unhealthy reason={result.get('reason')} "
            f"recent_400s={result['recent_400_count']}"
        )
    else:
        logger.debug("anthropic_health: healthy")
