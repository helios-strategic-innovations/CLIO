"""Process-wide Unusual Whales request pacer + 429 backoff (UW-1 / UW-2, S211; ADR-005).

Why: UW Basic ($150/mo, 40,000 req/day) rejects request *bursts*. Measured 2026-09-29: with <=3
requests in the same second 0 of 488 got a 429; with 12-14 in the same second ~31% did. CLIO's
DPOF emits 12-14 simultaneous requests every 180 s (``asyncio.gather``), and the old per-call
``asyncio.sleep(0.3)`` slept *in parallel*, so it spaced nothing. UW documents no burst limit,
so the pacer is a conservative token bucket tuned from live 429 telemetry.

Design:
  * Default is rate 3/s with burst 1 -- pure spacing. burst>1 would let a calendar second hold up
    burst+rate requests (5-6 at burst 3), above the only regime observed safe; raise it only after
    live telemetry shows headroom (CLIO_UW_BURST).
  * Slot reservation, not a queue: each ``acquire()`` takes one token under a short
    ``threading.Lock`` and, if the bucket is empty, sleeps for the computed delay. A negative
    balance is a reserved future slot, so N simultaneous callers get slots 1/rate apart (FIFO).
    No asyncio primitives -> safe across event loops and threads.
  * ``note_429()`` halves the effective rate (at most once per 5 s, floor 0.5 req/s) for
    ``penalty_s`` seconds, then the base rate is restored (AIMD-lite).
  * A cancelled waiter (DPOF wraps calls in ``asyncio.wait_for``) simply forfeits its slot.
  * ``CLIO_UW_PACER_ENABLED=0`` restores the legacy fixed sleep (rollback flag).

Pure stdlib -- importable anywhere, no agent.config.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import threading
import time
from typing import Callable, Mapping

logger = logging.getLogger("openclaw.uw_pacer")

_DEFAULT_RATE = 3.0        # requests / second (measured-safe: 0/488 429s at <=3 same-second)
_DEFAULT_BURST = 1         # pure spacing: keeps any calendar second at <=3-4 requests (the only regime observed safe)
_DEFAULT_PENALTY_S = 60.0
_MIN_RATE = 0.5            # never pace slower than one request per 2 s
_DECREASE_COOLDOWN_S = 5.0  # one multiplicative decrease per congestion event, not per 429
_LOG_INTERVAL_S = 30.0


def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    try:
        value = float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default
    if value != value:  # NaN
        return default
    return min(max(value, lo), hi)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


class UWPacer:
    def __init__(
        self,
        rate_per_s: float = _DEFAULT_RATE,
        burst: int = _DEFAULT_BURST,
        penalty_s: float = _DEFAULT_PENALTY_S,
        enabled: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.enabled = bool(enabled)
        self._base_rate = max(_MIN_RATE, float(rate_per_s))
        self._burst = max(1, int(burst))
        self._penalty_s = max(1.0, float(penalty_s))
        self._clock = clock
        self._lock = threading.Lock()
        self._tokens = float(self._burst)
        self._last = clock()
        self._factor = 1.0
        self._penalty_until = 0.0
        self._last_decrease = -1e9
        self.n_429 = 0

    # -- state (call under lock) ------------------------------------------------------------
    def _rate_locked(self, now: float) -> float:
        if self._factor < 1.0 and now >= self._penalty_until:
            self._factor = 1.0
        return max(_MIN_RATE, self._base_rate * self._factor)

    def effective_rate(self) -> float:
        with self._lock:
            return self._rate_locked(self._clock())

    def reserve(self) -> float:
        """Take one slot; return the seconds the caller must wait before sending (0 = now)."""
        with self._lock:
            now = self._clock()
            rate = self._rate_locked(now)
            self._tokens = min(float(self._burst), self._tokens + (now - self._last) * rate)
            self._last = now
            self._tokens -= 1.0
            if self._tokens >= 0.0:
                return 0.0
            return -self._tokens / rate

    async def acquire(self) -> float:
        """Wait for a slot. Returns the delay applied (0.0 when disabled or immediate)."""
        if not self.enabled:
            return 0.0
        wait = self.reserve()
        if wait > 0.0:
            await asyncio.sleep(wait)
        return wait

    def note_429(self) -> float:
        """Record a provider 429; returns the new effective rate (req/s)."""
        with self._lock:
            now = self._clock()
            self.n_429 += 1
            if now - self._last_decrease >= _DECREASE_COOLDOWN_S:
                floor_factor = min(1.0, _MIN_RATE / self._base_rate)
                self._factor = max(floor_factor, self._factor * 0.5)
                self._last_decrease = now
            self._penalty_until = now + self._penalty_s
            return max(_MIN_RATE, self._base_rate * self._factor)


# -- process singleton -------------------------------------------------------------------------
_pacer: UWPacer | None = None
_pacer_lock = threading.Lock()


def get_uw_pacer() -> UWPacer:
    global _pacer
    if _pacer is None:
        with _pacer_lock:
            if _pacer is None:
                _pacer = UWPacer(
                    rate_per_s=_env_float("CLIO_UW_RATE_PER_SEC", _DEFAULT_RATE, _MIN_RATE, 20.0),
                    burst=int(_env_float("CLIO_UW_BURST", _DEFAULT_BURST, 1, 20)),
                    penalty_s=_env_float("CLIO_UW_PENALTY_S", _DEFAULT_PENALTY_S, 1.0, 900.0),
                    enabled=_env_flag("CLIO_UW_PACER_ENABLED", True),
                )
    return _pacer


def reset_uw_pacer_for_tests() -> None:
    """Drop the singleton so the next get_uw_pacer() re-reads env. Tests only."""
    global _pacer
    with _pacer_lock:
        _pacer = None


# -- 429 handling ------------------------------------------------------------------------------
def backoff_seconds_429(headers: Mapping[str, str] | None, rand: Callable[[], float] = random.random) -> float:
    """Seconds to wait before the single retry after a 429.

    Honors a numeric ``Retry-After``; otherwise ``CLIO_UW_429_BACKOFF_S`` (default 1.5). The
    ``x-uw-req-per-minute-reset`` header is deliberately ignored: it is a constant 60000 even on
    429s (it made the old handler sleep 60 s, longer than DPOF's 8 s per-call timeout, so the
    retry never ran). Clamped to [0.5, 10] s, plus up to 0.5 s of jitter.
    """
    base = _env_float("CLIO_UW_429_BACKOFF_S", 1.5, 0.5, 10.0)
    if headers is not None:
        raw = headers.get("Retry-After") or headers.get("retry-after")
        if raw:
            try:
                base = float(raw)
            except (TypeError, ValueError):
                pass
    if base != base:  # NaN
        base = 1.5
    return min(max(base, 0.5), 10.0) + 0.5 * rand()


_last_429_log = -1e9
_suppressed_429 = 0
_log_lock = threading.Lock()


def note_429_and_log(endpoint: str, wait_s: float, clock: Callable[[], float] = time.monotonic) -> None:
    """Tell the pacer about a 429 and emit at most one WARNING per 30 s (count of the rest kept)."""
    global _last_429_log, _suppressed_429
    rate = get_uw_pacer().note_429()
    with _log_lock:
        now = clock()
        if now - _last_429_log >= _LOG_INTERVAL_S:
            suppressed, _suppressed_429, _last_429_log = _suppressed_429, 0, now
            emit = True
        else:
            _suppressed_429 += 1
            suppressed, emit = 0, False
    if emit:
        logger.warning(
            "UW rate limited (429) on %s; retry in %.1fs, pacer now %.2f req/s (suppressed=%d since last log)",
            endpoint, wait_s, rate, suppressed,
        )
    else:
        logger.debug("UW rate limited (429) on %s; retry in %.1fs", endpoint, wait_s)
