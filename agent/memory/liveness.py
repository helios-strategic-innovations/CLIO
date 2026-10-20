"""
Memory-tier liveness — S209 silent-death hardening.

The S164 working-memory deadlock stayed invisible for ~12 weeks because nothing
asserted the tier was alive and the prompt tolerated its absence silently. The
deadlock fix stops the bleeding; THIS module is what stops the *next* silent death
we haven't predicted. Two guards:

  * probe_tier_roundtrip() — a TIMEOUT-BOUNDED write→read round-trip on the REAL
    path. A hung tier (e.g. a re-introduced lock deadlock) makes the worker thread
    never finish; the bounded join returns TIMEOUT and the CALLER is not dragged
    into the hang. The probe must never inherit the failure it is meant to catch
    (a naive probe that just called the deadlocked add() would hang with it).

  * age_seconds() — tz-CONSISTENT staleness math, fail-LOUD. A naive-vs-aware
    datetime subtraction raises; if swallowed, the staleness alarm never fires
    (BC #47 — weeks of live greeks were lost to exactly that). Compute aware-vs-
    aware; on a malformed timestamp, log LOUD and return None so callers FAIL
    TOWARD the alarm — never silently treat "unknown age" as "fresh".

READ-ONLY of nothing — this module exercises the real write path on purpose.
"""

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("moltbot.memory.liveness")


def _aware(dt: datetime) -> datetime:
    """Normalize a naive datetime to UTC-aware (never a silent naive subtraction)."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def age_seconds(ts: Any) -> Optional[float]:
    """Age in seconds of an ISO string / datetime / epoch-float, tz-consistently
    vs now(UTC).

    Returns None for a genuinely-absent timestamp (None / "") AND for a malformed
    one — but a malformed value is logged at ERROR. Callers MUST treat None as
    "unknown" and fail toward the alarm; do NOT coerce None to 0/"fresh".
    """
    if ts is None or ts == "":
        return None
    try:
        if isinstance(ts, bool):  # guard: bool is an int subclass
            raise TypeError("bool is not a timestamp")
        if isinstance(ts, (int, float)):
            return max(0.0, time.time() - float(ts))
        if isinstance(ts, datetime):
            dt = _aware(ts)
        else:
            dt = _aware(datetime.fromisoformat(str(ts)))
        return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds())
    except Exception as e:
        logger.error(
            "liveness.age_seconds: unparseable timestamp %r (%s) — age UNKNOWN; "
            "callers must fail toward the alarm, not treat as fresh", ts, e,
        )
        return None


def is_stale(ts: Any, max_age_seconds: float) -> bool:
    """True if `ts` is older than `max_age_seconds`. UNKNOWN age (None) is treated
    as STALE (fail toward the alarm), never as fresh."""
    age = age_seconds(ts)
    if age is None:
        return True
    return age > max_age_seconds


def probe_tier_roundtrip(
    write_read: Callable[[], bool],
    tier_name: str,
    timeout: float = 2.0,
) -> Dict[str, str]:
    """Run a write→read round-trip closure in a daemon worker with a bounded join.

    `write_read()` must perform a real write then read it back and return True iff
    the round-trip succeeded. Returns {'status': LIVE|BROKEN|TIMEOUT, 'detail': ...}.
    A hung tier → the worker never finishes → join times out → TIMEOUT (the caller
    is NOT blocked beyond `timeout`; the worker is daemon and abandoned).
    """
    box: Dict[str, Any] = {}

    def worker():
        try:
            box["ok"] = bool(write_read())
        except Exception as e:  # noqa: BLE001 — surfaced below as BROKEN
            box["error"] = f"{type(e).__name__}: {e}"

    t = threading.Thread(target=worker, name=f"{tier_name}-liveness-probe", daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        logger.error(
            "MEMORY LIVENESS: %s probe TIMEOUT (>%.1fs) — tier HUNG "
            "(write→read round-trip never completed)", tier_name, timeout,
        )
        return {"status": "TIMEOUT", "detail": f"round-trip exceeded {timeout:.1f}s — tier hung"}
    if "error" in box:
        logger.error("MEMORY LIVENESS: %s probe BROKEN — %s", tier_name, box["error"])
        return {"status": "BROKEN", "detail": box["error"]}
    if not box.get("ok"):
        logger.error("MEMORY LIVENESS: %s probe BROKEN — write→read returned False", tier_name)
        return {"status": "BROKEN", "detail": "write→read round-trip returned False"}
    return {"status": "LIVE", "detail": "write→read round-trip ok"}


def probe_working_memory(wm, timeout: float = 2.0) -> Dict[str, str]:
    """Timeout-bounded write→read round-trip on the REAL working-memory tier.

    Writes a uniquely-marked synthetic entry (short TTL, dedicated `_liveness`
    domain), reads it back, and removes it. This is what WOULD have caught the
    S164 deadlock: against the buggy add(), the worker hangs and this returns
    TIMEOUT instead of hanging the caller.
    """
    marker = f"__wm_liveness_probe__ {time.time():.6f} round-trip integrity check entry"

    def write_read() -> bool:
        eid = wm.add(content=marker, domain="_liveness", priority=0.01, ttl_hours=0.05)
        if not eid:
            raise RuntimeError("add() rejected probe content (empty id)")
        try:
            got = wm.get(eid)
            return bool(got and got.content == marker)
        finally:
            wm._remove_entry(eid)

    return probe_tier_roundtrip(write_read, tier_name="working-memory", timeout=timeout)
