"""Account-wide Unusual Whales quota coordination.

The local SQLite/WAL ledger is the request-path admission authority shared by
Hydrinia, CLIO, and same-host datafeeds. Authoritative UW response headers are
written locally before returning to the caller, then mirrored to Supabase by a
bounded asynchronous outbox worker. Supabase is never on the provider hot path.
"""

from __future__ import annotations

import atexit
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import IntEnum
import json
import logging
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence
from uuid import uuid4
from zoneinfo import ZoneInfo

import requests


log = logging.getLogger("UWQuota")

# OUTBOX-1 (S212): consecutive failed outbox flushes before UW_SUPABASE_OUTBOX_DEGRADED is a WARNING.
_OUTBOX_WARN_AFTER = 3
_NY = ZoneInfo("America/New_York")


class QuotaPriority(IntEnum):
    P0 = 0
    P1 = 1
    P2 = 2


@dataclass(frozen=True)
class QuotaDecision:
    allowed: bool
    request_id: str
    reason_code: str
    source: str
    remaining: Optional[int] = None
    stale: bool = False


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _reorder_tolerance() -> int:
    """UW-3 (S211): how far the reported daily count may drop below the observed max before it is
    treated as an anomaly. Concurrent responses arrive out of order and each carries the server's
    count at its own processing time; on 2026-09-29 all 130 observed decreases were 1-2."""
    return max(0, _int_env("UW_QUOTA_REORDER_TOLERANCE", 5))


def _parse_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def extract_uw_headers(headers: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = {str(key).lower(): value for key, value in headers.items()}
    return {
        "daily_count": _parse_int(normalized.get("x-uw-daily-req-count")),
        "entitlement": _parse_int(normalized.get("x-uw-token-req-limit")),
        "minute_counter": _parse_int(normalized.get("x-uw-minute-req-counter")),
        "minute_remaining": _parse_int(normalized.get("x-uw-req-per-minute-remaining")),
        "minute_reset_ms": _parse_int(normalized.get("x-uw-req-per-minute-reset")),
        "retry_after": normalized.get("retry-after"),
        "raw": {
            key: str(value)
            for key, value in normalized.items()
            if key.startswith("x-uw-")
        },
    }


_DAILY_HEADER_KEYS = ("x-uw-daily-req-count", "x-uw-token-req-limit")
_MINUTE_HEADER_KEYS = ("x-uw-req-per-minute-remaining", "x-uw-req-per-minute-reset")
MINUTE_ONLY_LOG_INTERVAL_S = 3600.0


def _is_minute_only(raw: Mapping[str, Any]) -> bool:
    """True when a response carries per-minute UW headers and no daily-usage header at all.

    `raw` comes from extract_uw_headers (keys already lowercased). Any daily header present
    (even partially) means this is NOT the minute-only shape.
    """
    if any(key in raw for key in _DAILY_HEADER_KEYS):
        return False
    return any(key in raw for key in _MINUTE_HEADER_KEYS)


def quota_cycle_key(at: datetime) -> str:
    """Return UW's documented 8PM ET reset-cycle key, including DST."""
    if at.tzinfo is None:
        at = at.replace(tzinfo=timezone.utc)
    et = at.astimezone(_NY)
    return (et - timedelta(hours=20)).date().isoformat()


def _default_db_path() -> Path:
    configured = os.getenv("UW_QUOTA_SQLITE_PATH")
    if configured:
        return Path(configured)
    if Path("/home/agent").is_dir():
        return Path("/home/agent/shared/provider_ops/uw_quota.sqlite3")
    return Path(__file__).resolve().parent / "state" / "uw_quota.sqlite3"


class _SQLiteQuotaLedger:
    def __init__(
        self,
        path: Path,
        *,
        default_entitlement: int,
        stale_after_s: float,
        lease_ttl_s: float,
        clock: Callable[[], datetime],
        stale_probe: bool = True,
        stale_probe_interval_s: float = 60.0,
        minute_only_fresh: bool = True,
    ) -> None:
        self.path = path
        self.default_entitlement = default_entitlement
        self.stale_after_s = stale_after_s
        self.lease_ttl_s = lease_ttl_s
        self.clock = clock
        self.stale_probe = bool(stale_probe)
        self.stale_probe_interval_s = max(1.0, float(stale_probe_interval_s))
        # UW_QUOTA_MINUTE_ONLY_FRESH: a 200 carrying only per-minute headers refreshes observed_at (never counts).
        self.minute_only_fresh = bool(minute_only_fresh)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=5.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError:
            pass
        last_locked: Optional[Exception] = None
        for attempt in range(40):
            conn = self._connect()
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=FULL")
                conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS quota_cycles (
                    cycle_key TEXT PRIMARY KEY,
                    entitlement INTEGER NOT NULL,
                    observed_count INTEGER NOT NULL DEFAULT 0,
                    local_unconfirmed INTEGER NOT NULL DEFAULT 0,
                    observed_at TEXT,
                    minute_counter INTEGER,
                    minute_remaining INTEGER,
                    minute_reset_ms INTEGER,
                    header_schema_ok INTEGER NOT NULL DEFAULT 0,
                    last_endpoint TEXT,
                    last_caller TEXT,
                    anomaly_count INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS quota_reservations (
                    request_id TEXT PRIMARY KEY,
                    cycle_key TEXT NOT NULL REFERENCES quota_cycles(cycle_key),
                    caller TEXT NOT NULL,
                    consumer TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    endpoint TEXT NOT NULL,
                    purpose TEXT NOT NULL,
                    state TEXT NOT NULL,
                    reason_code TEXT NOT NULL,
                    reserved_at TEXT NOT NULL,
                    resolved_at TEXT,
                    response_status INTEGER
                );
                CREATE INDEX IF NOT EXISTS quota_reservations_pending_idx
                    ON quota_reservations(cycle_key, reserved_at)
                    WHERE state = 'pending';
                CREATE TABLE IF NOT EXISTS supabase_outbox (
                    event_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    synced_at TEXT,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT
                );
                CREATE INDEX IF NOT EXISTS supabase_outbox_pending_idx
                    ON supabase_outbox(created_at)
                    WHERE synced_at IS NULL;
                """
                )
                last_locked = None
                break
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() and "busy" not in str(exc).lower():
                    raise
                last_locked = exc
                time.sleep(min(0.01 * (attempt + 1), 0.25))
            finally:
                conn.close()
        if last_locked is not None:
            raise sqlite3.OperationalError(
                f"quota ledger initialization remained locked: {last_locked}"
            )
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def _now(self) -> datetime:
        now = self.clock()
        if now.tzinfo is None:
            return now.replace(tzinfo=timezone.utc)
        return now.astimezone(timezone.utc)

    @staticmethod
    def _iso(at: datetime) -> str:
        return at.astimezone(timezone.utc).isoformat()

    def _ensure_cycle(self, conn: sqlite3.Connection, cycle: str, now_iso: str) -> None:
        # UW-6 (S211): seed a new cycle from the latest header-confirmed entitlement (insert-only, no
        # schema change; the ledger file is shared with other clients), else the configured default.
        conn.execute(
            """
            INSERT INTO quota_cycles(cycle_key, entitlement, updated_at)
            VALUES (
                ?,
                COALESCE(
                    (SELECT entitlement FROM quota_cycles
                      WHERE header_schema_ok = 1 AND cycle_key < ?
                      ORDER BY cycle_key DESC LIMIT 1),
                    ?
                ),
                ?
            )
            ON CONFLICT(cycle_key) DO NOTHING
            """,
            (cycle, cycle, self.default_entitlement, now_iso),
        )

    def _expire_leases(self, conn: sqlite3.Connection, cycle: str, now: datetime) -> None:
        cutoff = self._iso(now - timedelta(seconds=self.lease_ttl_s))
        now_iso = self._iso(now)
        conn.execute(
            """
            UPDATE quota_reservations
               SET state='expired', reason_code='RESERVATION_TIMEOUT', resolved_at=?
             WHERE cycle_key=? AND state='pending' AND reserved_at < ?
            """,
            (now_iso, cycle, cutoff),
        )

    @staticmethod
    def _age_s(raw: Optional[str], now: datetime) -> float:
        if not raw:
            return float("inf")
        try:
            observed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            if observed.tzinfo is None:
                observed = observed.replace(tzinfo=timezone.utc)
            return max(0.0, (now - observed.astimezone(timezone.utc)).total_seconds())
        except (TypeError, ValueError):
            return float("inf")

    def _stale_probe_due(self, conn: sqlite3.Connection, cycle: str, now: datetime) -> bool:
        """UW-5 (S211): may ONE lower-priority request go out to refresh a stale ledger?

        A stale ledger (no header-bearing response for `stale_after_s`) admits only P0. Overnight nothing
        P0 runs, so every P1/P2 call (e.g. /news/headlines every minute) was denied until the first P0
        call at the open -- the ledger could never heal itself. A probe lets one such call through; its
        response headers refresh observed_at and normal priority rules resume. A completed probe keeps a
        `STALE_PROBE_DONE` marker (observe() would otherwise overwrite its reason with PROVIDER_RESPONSE and
        the per-interval bound could never see it).

        Bounded on purpose (a response WITHOUT usable headers would otherwise turn "stale" into "allow
        everything"): at most one probe in flight (this runs inside reserve()'s BEGIN IMMEDIATE, so it is
        single-flight across processes; a crashed probe frees after the lease TTL) and at most one per
        `stale_probe_interval_s`. The recency check reads only the newest 500 rows by rowid -- the ledger
        holds ~1.7 M rows and reason_code is not indexed.
        """
        if not self.stale_probe:
            return False
        in_flight = int(
            conn.execute(
                "SELECT count(*) FROM quota_reservations "
                "WHERE cycle_key=? AND state='pending' AND reason_code='STALE_PROBE'",
                (cycle,),
            ).fetchone()[0]
        )
        if in_flight:
            return False
        recent = conn.execute(
            """
            SELECT reserved_at FROM (
                SELECT reserved_at, reason_code FROM quota_reservations ORDER BY rowid DESC LIMIT 500
            ) WHERE reason_code IN ('STALE_PROBE', 'STALE_PROBE_DONE') ORDER BY reserved_at DESC LIMIT 1
            """
        ).fetchone()
        if recent is not None and self._age_s(recent["reserved_at"], now) < self.stale_probe_interval_s:
            return False
        return True

    def reserve(
        self,
        *,
        caller: str,
        consumer: str,
        endpoint: str,
        purpose: str,
        priority: QuotaPriority,
        p0_reserve: int,
        p1_reserve: int,
    ) -> QuotaDecision:
        now = self._now()
        now_iso = self._iso(now)
        cycle = quota_cycle_key(now)
        request_id = str(uuid4())
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            self._ensure_cycle(conn, cycle, now_iso)
            self._expire_leases(conn, cycle, now)
            row = conn.execute(
                "SELECT * FROM quota_cycles WHERE cycle_key=?", (cycle,)
            ).fetchone()
            pending = int(
                conn.execute(
                    "SELECT count(*) FROM quota_reservations WHERE cycle_key=? AND state='pending'",
                    (cycle,),
                ).fetchone()[0]
            )
            entitlement = int(row["entitlement"])
            effective = int(row["observed_count"]) + int(row["local_unconfirmed"]) + pending
            remaining = max(0, entitlement - effective)
            stale = self._age_s(row["observed_at"], now) > self.stale_after_s

            if stale:
                allowed = priority == QuotaPriority.P0 and remaining > 0
                reason = "DEGRADED_P0_LOCAL" if allowed else "DEGRADED_STALE_CACHE"
                if not allowed and remaining > 0 and self._stale_probe_due(conn, cycle, now):
                    allowed, reason = True, "STALE_PROBE"
            elif priority == QuotaPriority.P0:
                allowed = remaining > 0
                reason = "P0_RESERVED" if allowed else "ACCOUNT_LIMIT"
            elif priority == QuotaPriority.P1:
                allowed = remaining > max(0, p0_reserve)
                reason = "P1_RESERVED" if allowed else "P0_RESERVE"
            else:
                allowed = remaining > max(0, p0_reserve) + max(0, p1_reserve)
                reason = "P2_RESERVED" if allowed else "P1_P0_RESERVE"

            conn.execute(
                """
                INSERT INTO quota_reservations(
                    request_id, cycle_key, caller, consumer, priority, endpoint,
                    purpose, state, reason_code, reserved_at, resolved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    cycle,
                    caller[:96],
                    consumer[:96],
                    int(priority),
                    endpoint[:512],
                    purpose[:256],
                    "pending" if allowed else "denied",
                    reason,
                    now_iso,
                    None if allowed else now_iso,
                ),
            )
            conn.commit()
            return QuotaDecision(
                allowed=allowed,
                request_id=request_id,
                reason_code=reason,
                source="sqlite_wal",
                remaining=max(0, remaining - (1 if allowed else 0)),
                stale=stale,
            )
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def observe(
        self,
        decision: Optional[QuotaDecision],
        *,
        caller: str,
        consumer: str,
        endpoint: str,
        purpose: str,
        status_code: int,
        parsed: Mapping[str, Any],
        latency_ms: Optional[float],
    ) -> Dict[str, Any]:
        now = self._now()
        now_iso = self._iso(now)
        cycle = quota_cycle_key(now)
        schema_ok = parsed.get("daily_count") is not None and parsed.get("entitlement") is not None
        event_id = str(uuid4())
        anomaly = False
        reordered = False
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            self._ensure_cycle(conn, cycle, now_iso)
            if decision and decision.request_id:
                conn.execute(
                    """
                    UPDATE quota_reservations
                       SET state='completed',
                           reason_code=CASE WHEN reason_code='STALE_PROBE'
                                            THEN 'STALE_PROBE_DONE' ELSE 'PROVIDER_RESPONSE' END,
                           resolved_at=?, response_status=?
                     WHERE request_id=? AND state IN ('pending', 'expired')
                    """,
                    (now_iso, int(status_code), decision.request_id),
                )

            previous = conn.execute(
                "SELECT observed_count, anomaly_count FROM quota_cycles WHERE cycle_key=?",
                (cycle,),
            ).fetchone()
            previous_count = int(previous["observed_count"])
            anomaly_count = int(previous["anomaly_count"])
            if schema_ok:
                reported = int(parsed["daily_count"])
                # observed_count keeps max(), so a small out-of-order decrease never lowers usage;
                # only a drop beyond the tolerance (a real reset / regression) is an anomaly.
                anomaly = reported < previous_count - _reorder_tolerance()
                reordered = (not anomaly) and reported < previous_count
                conn.execute(
                    """
                    UPDATE quota_cycles
                       SET entitlement=?, observed_count=max(observed_count, ?),
                           local_unconfirmed=0, observed_at=?, minute_counter=?,
                           minute_remaining=?, minute_reset_ms=?, header_schema_ok=1,
                           last_endpoint=?, last_caller=?,
                           anomaly_count=?, updated_at=?
                     WHERE cycle_key=?
                    """,
                    (
                        int(parsed["entitlement"]),
                        reported,
                        now_iso,
                        parsed.get("minute_counter"),
                        parsed.get("minute_remaining"),
                        parsed.get("minute_reset_ms"),
                        endpoint[:512],
                        caller[:96],
                        anomaly_count + (1 if anomaly else 0),
                        now_iso,
                        cycle,
                    ),
                )
            else:
                # R-13: a 200 with only per-minute headers is a fresh observation of the endpoint (UW is alive
                # and answering) but carries no daily count, so observed_count/entitlement stay untouched and the
                # local usage it made stays in local_unconfirmed. 429/5xx never refresh.
                refresh_observed = (
                    self.minute_only_fresh
                    and int(status_code) == 200
                    and _is_minute_only(parsed.get("raw") or {})
                )
                conn.execute(
                    """
                    UPDATE quota_cycles
                       SET local_unconfirmed=local_unconfirmed + ?, header_schema_ok=0,
                           observed_at=COALESCE(?, observed_at),
                           last_endpoint=?, last_caller=?, updated_at=?
                     WHERE cycle_key=?
                    """,
                    (
                        1 if decision and decision.allowed else 0,
                        now_iso if refresh_observed else None,
                        endpoint[:512], caller[:96], now_iso, cycle,
                    ),
                )

            payload = {
                "event_id": event_id,
                "observed_at": now_iso,
                "caller": caller[:96],
                "consumer": consumer[:96],
                "purpose": purpose[:256],
                "endpoint": endpoint[:512],
                "status": int(status_code),
                "latency_ms": round(float(latency_ms), 1) if latency_ms is not None else None,
                "daily_count": parsed.get("daily_count"),
                "entitlement": parsed.get("entitlement"),
                "minute_counter": parsed.get("minute_counter"),
                "minute_remaining": parsed.get("minute_remaining"),
                "minute_reset_ms": parsed.get("minute_reset_ms"),
                "retry_after": str(parsed.get("retry_after"))[:128] if parsed.get("retry_after") is not None else None,
                "uw_headers": dict(parsed.get("raw") or {}),
                "header_schema_ok": bool(schema_ok),
            }
            conn.execute(
                "INSERT INTO supabase_outbox(event_id, payload_json, created_at) VALUES (?, ?, ?)",
                (event_id, json.dumps(payload, sort_keys=True, separators=(",", ":")), now_iso),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        if anomaly:
            log.error(
                "UW_HEADER_COUNT_DECREASE cycle=%s previous=%d reported=%s endpoint=%s",
                cycle,
                previous_count,
                parsed.get("daily_count"),
                endpoint,
            )
        elif reordered:
            log.debug(
                "UW_HEADER_COUNT_REORDER cycle=%s previous=%d reported=%s endpoint=%s (within tolerance)",
                cycle,
                previous_count,
                parsed.get("daily_count"),
                endpoint,
            )
        return payload

    def release(self, request_id: str, reason_code: str) -> bool:
        now = self._iso(self._now())
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                """
                UPDATE quota_reservations
                   SET state='released', reason_code=?, resolved_at=?
                 WHERE request_id=? AND state='pending'
                """,
                (reason_code[:128], now, request_id),
            )
            conn.commit()
            return cursor.rowcount == 1
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def status(self) -> Dict[str, Any]:
        now = self._now()
        cycle = quota_cycle_key(now)
        now_iso = self._iso(now)
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            self._ensure_cycle(conn, cycle, now_iso)
            self._expire_leases(conn, cycle, now)
            row = conn.execute(
                "SELECT * FROM quota_cycles WHERE cycle_key=?", (cycle,)
            ).fetchone()
            pending = int(
                conn.execute(
                    "SELECT count(*) FROM quota_reservations WHERE cycle_key=? AND state='pending'",
                    (cycle,),
                ).fetchone()[0]
            )
            outbox = int(
                conn.execute(
                    "SELECT count(*) FROM supabase_outbox WHERE synced_at IS NULL"
                ).fetchone()[0]
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        data = dict(row)
        data.update(
            {
                "pending_count": pending,
                "outbox_pending": outbox,
                "remaining": max(
                    0,
                    int(row["entitlement"])
                    - int(row["observed_count"])
                    - int(row["local_unconfirmed"])
                    - pending,
                ),
                "stale": self._age_s(row["observed_at"], now) > self.stale_after_s,
                "source": "sqlite_wal",
            }
        )
        return data

    def pending_outbox(self, limit: int) -> Sequence[sqlite3.Row]:
        conn = self._connect()
        try:
            return conn.execute(
                """
                SELECT event_id, payload_json FROM supabase_outbox
                 WHERE synced_at IS NULL ORDER BY created_at LIMIT ?
                """,
                (max(1, min(int(limit), 1000)),),
            ).fetchall()
        finally:
            conn.close()

    def mark_outbox_synced(self, event_ids: Sequence[str]) -> None:
        if not event_ids:
            return
        now = self._iso(self._now())
        placeholders = ",".join("?" for _ in event_ids)
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                f"UPDATE supabase_outbox SET synced_at=?, last_error=NULL WHERE event_id IN ({placeholders})",
                (now, *event_ids),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def mark_outbox_failed(self, event_ids: Sequence[str], error: str) -> None:
        if not event_ids:
            return
        placeholders = ",".join("?" for _ in event_ids)
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                f"""
                UPDATE supabase_outbox
                   SET attempt_count=attempt_count + 1, last_error=?
                 WHERE event_id IN ({placeholders})
                """,
                (error[:160], *event_ids),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


class _SupabaseOutboxWorker:
    def __init__(
        self,
        ledger: _SQLiteQuotaLedger,
        *,
        base_url: str,
        service_role_key: str,
        timeout_s: float,
        flush_interval_s: float,
        batch_size: int,
        start: bool,
    ) -> None:
        self.ledger = ledger
        self.base_url = base_url.rstrip("/")
        self._service_role_key = service_role_key
        self.timeout_s = timeout_s
        # UW-4 (S211): the RPC timeout (0.9 s) is sized for a fast call, not a <=100-row batch insert;
        # it caused ~52 ReadTimeouts/13 h. The POST runs on this background thread, so a longer
        # bound cannot block the event loop.
        self._post_timeout_s = max(float(timeout_s), _float_env("UW_QUOTA_SUPABASE_POST_TIMEOUT_S", 5.0))
        self.flush_interval_s = flush_interval_s
        self.batch_size = batch_size
        # OUTBOX-1 (S212): a failed flush leaves its rows unsynced and the next flush (1 s) retries them, so an
        # isolated ReadTimeout heals in seconds. Only a SUSTAINED run of failures is a WARNING.
        self._consecutive_outbox_failures = 0
        self.enabled = bool(self.base_url and self._service_role_key)
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None
        if self.enabled and start:
            self._thread = threading.Thread(
                target=self._run,
                name="uw-supabase-outbox",
                daemon=True,
            )
            self._thread.start()

    def wake(self) -> None:
        if self._thread is not None:
            self._wake.set()

    def flush_once(self) -> int:
        if not self.enabled:
            return 0
        rows = self.ledger.pending_outbox(self.batch_size)
        if not rows:
            return 0
        event_ids = [str(row["event_id"]) for row in rows]
        events = [json.loads(str(row["payload_json"])) for row in rows]
        try:
            response = requests.post(
                f"{self.base_url}/rest/v1/rpc/uw_quota_observe_batch",
                headers={
                    "apikey": self._service_role_key,
                    "Authorization": f"Bearer {self._service_role_key}",
                    "Content-Type": "application/json",
                },
                json={"p_events": events},
                timeout=self._post_timeout_s,
            )
            response.raise_for_status()
            self.ledger.mark_outbox_synced(event_ids)
            if self._consecutive_outbox_failures:
                log.info("UW outbox recovered after %d failed flush(es)", self._consecutive_outbox_failures)
                self._consecutive_outbox_failures = 0
            return len(event_ids)
        except Exception as exc:
            error = type(exc).__name__
            self.ledger.mark_outbox_failed(event_ids, error)
            self._consecutive_outbox_failures += 1
            if self._consecutive_outbox_failures >= _OUTBOX_WARN_AFTER:
                # the token the morning check and operators grep for: sustained failure only
                log.warning(
                    "UW_SUPABASE_OUTBOX_DEGRADED events=%d error=%s consecutive=%d",
                    len(event_ids), error, self._consecutive_outbox_failures,
                )
            else:
                log.info(
                    "UW outbox flush failed (events=%d error=%s, %d in a row); rows stay queued and retry",
                    len(event_ids), error, self._consecutive_outbox_failures,
                )
            return 0

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(self.flush_interval_s)
            self._wake.clear()
            self.flush_once()
        self.flush_once()

    def close(self, timeout_s: float = 2.0) -> bool:
        if self._thread is None:
            return True
        self._stop.set()
        self._wake.set()
        self._thread.join(max(0.0, timeout_s))
        return not self._thread.is_alive()


_WORKERS: Dict[str, _SupabaseOutboxWorker] = {}
_WORKERS_LOCK = threading.Lock()


def _worker_key(path: Path, base_url: str) -> str:
    return f"{path.resolve()}|{base_url.rstrip('/')}"


def _get_worker(
    ledger: _SQLiteQuotaLedger,
    *,
    base_url: str,
    service_role_key: str,
    timeout_s: float,
    start: bool,
) -> _SupabaseOutboxWorker:
    key = _worker_key(ledger.path, base_url)
    with _WORKERS_LOCK:
        worker = _WORKERS.get(key)
        if worker is None:
            worker = _SupabaseOutboxWorker(
                ledger,
                base_url=base_url,
                service_role_key=service_role_key,
                timeout_s=timeout_s,
                flush_interval_s=_float_env("UW_QUOTA_SUPABASE_FLUSH_S", 1.0),
                batch_size=_int_env("UW_QUOTA_SUPABASE_BATCH", 100),
                start=start,
            )
            _WORKERS[key] = worker
        return worker


class SupabaseUWQuotaClient:
    """Local-first quota client with asynchronous Supabase reconciliation."""

    def __init__(
        self,
        *,
        caller: str,
        base_url: Optional[str] = None,
        service_role_key: Optional[str] = None,
        db_path: Optional[str] = None,
        cache_path: Optional[str] = None,
        enabled: Optional[bool] = None,
        timeout_s: Optional[float] = None,
        stale_after_s: Optional[float] = None,
        lease_ttl_s: Optional[float] = None,
        clock: Optional[Callable[[], datetime]] = None,
        start_worker: bool = True,
        monotonic: Optional[Callable[[], float]] = None,
    ) -> None:
        self.caller = str(caller or "unknown")[:96]
        self.enabled = _bool_env("UW_QUOTA_COORDINATOR_ENABLED", True) if enabled is None else bool(enabled)
        configured_url = os.getenv("SUPABASE_URL") if base_url is None else base_url
        configured_key = (
            os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            if service_role_key is None
            else service_role_key
        )
        self.base_url = (configured_url or "").rstrip("/")
        self._service_role_key = configured_key or ""
        self.timeout_s = timeout_s if timeout_s is not None else _float_env("UW_QUOTA_RPC_TIMEOUT_S", 0.9)
        self.stale_after_s = stale_after_s if stale_after_s is not None else _float_env("UW_QUOTA_CACHE_STALE_S", 300.0)
        self.lease_ttl_s = lease_ttl_s if lease_ttl_s is not None else _float_env("UW_QUOTA_LEASE_TTL_S", 120.0)
        self.p0_reserve = _int_env("UW_QUOTA_P0_RESERVE", 5000)
        self.p1_reserve = _int_env("UW_QUOTA_P1_RESERVE", 10000)
        # UW Basic = 40,000/day (UW-6, S211; was 160000 for a former plan). Only used until the first
        # header observation of a cycle -- and new cycles inherit the last observed entitlement.
        self.default_entitlement = _int_env("UW_RATE_BUDGET_DAILY_LIMIT", 40000)
        # UW-5 (S211): let one lower-priority call refresh a stale ledger. UW_QUOTA_STALE_PROBE=0 = old behaviour.
        self.stale_probe = _bool_env("UW_QUOTA_STALE_PROBE", True)
        self.stale_probe_interval_s = _float_env("UW_QUOTA_STALE_PROBE_INTERVAL_S", 60.0)
        # UW_QUOTA_MINUTE_ONLY_FRESH (R-13): 0 = a minute-only 200 never refreshes the ledger (pre-R-13 behaviour).
        self.minute_only_fresh = _bool_env("UW_QUOTA_MINUTE_ONLY_FRESH", True)
        selected_path = Path(db_path) if db_path else None
        if selected_path is None and cache_path:
            legacy = Path(cache_path)
            selected_path = legacy if legacy.suffix in {".db", ".sqlite", ".sqlite3"} else legacy.with_suffix(".sqlite3")
        self.db_path = selected_path or _default_db_path()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        # OP-13: per-endpoint dedupe for UW_HEADER_MINUTE_ONLY INFO logs (monotonic, injectable for tests).
        self._monotonic = monotonic or time.monotonic
        self._minute_only_seen: Dict[str, float] = {}
        self._minute_only_lock = threading.Lock()
        self._ledger: Optional[_SQLiteQuotaLedger] = None
        self._worker: Optional[_SupabaseOutboxWorker] = None
        if self.enabled:
            self._ledger = _SQLiteQuotaLedger(
                self.db_path,
                default_entitlement=self.default_entitlement,
                stale_after_s=self.stale_after_s,
                lease_ttl_s=self.lease_ttl_s,
                clock=self._clock,
                stale_probe=self.stale_probe,
                stale_probe_interval_s=self.stale_probe_interval_s,
                minute_only_fresh=self.minute_only_fresh,
            )
            self._worker = _get_worker(
                self._ledger,
                base_url=self.base_url,
                service_role_key=self._service_role_key,
                timeout_s=self.timeout_s,
                start=start_worker,
            )

    def reserve(
        self,
        *,
        consumer: str,
        endpoint: str,
        purpose: str = "",
        priority: QuotaPriority | int = QuotaPriority.P1,
    ) -> QuotaDecision:
        priority = QuotaPriority(int(priority))
        if not self.enabled or self._ledger is None:
            return QuotaDecision(True, str(uuid4()), "COORDINATOR_DISABLED", "disabled")
        return self._ledger.reserve(
            caller=self.caller,
            consumer=str(consumer or "unknown"),
            endpoint=str(endpoint or "unknown"),
            purpose=str(purpose or ""),
            priority=priority,
            p0_reserve=self.p0_reserve,
            p1_reserve=self.p1_reserve,
        )

    def observe_response(
        self,
        decision: Optional[QuotaDecision],
        *,
        consumer: str,
        endpoint: str,
        purpose: str,
        status_code: int,
        headers: Mapping[str, Any],
        latency_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        parsed = extract_uw_headers(headers)
        schema_ok = parsed["daily_count"] is not None and parsed["entitlement"] is not None
        if not self.enabled or self._ledger is None:
            payload = {
                "caller": self.caller,
                "consumer": str(consumer or "unknown")[:96],
                "purpose": str(purpose or "")[:256],
                "endpoint": str(endpoint or "")[:512],
                "status": int(status_code),
                "latency_ms": round(float(latency_ms), 1) if latency_ms is not None else None,
                "retry_after": parsed["retry_after"],
                "uw_headers": parsed["raw"],
                "header_schema_ok": schema_ok,
            }
        else:
            payload = self._ledger.observe(
                decision,
                caller=self.caller,
                consumer=str(consumer or "unknown"),
                endpoint=str(endpoint or ""),
                purpose=str(purpose or ""),
                status_code=int(status_code),
                parsed=parsed,
                latency_ms=latency_ms,
            )
            if self._worker is not None:
                self._worker.wake()

        minute_only = int(status_code) == 200 and not schema_ok and _is_minute_only(parsed["raw"])
        if minute_only:
            payload["header_class"] = "minute_only"

        if status_code == 429 or status_code >= 500:
            log.warning("UW_PROVIDER_ERROR %s", json.dumps(payload, sort_keys=True, separators=(",", ":")))
        elif minute_only:
            if self._minute_only_should_log(endpoint):
                log.info("UW_HEADER_MINUTE_ONLY %s", json.dumps(payload, sort_keys=True, separators=(",", ":")))
        elif not schema_ok:
            log.warning("UW_HEADER_SCHEMA_MISSING %s", json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return payload

    def _minute_only_should_log(self, endpoint: Any) -> bool:
        key = str(endpoint or "")[:512]
        now = float(self._monotonic())
        with self._minute_only_lock:
            last = self._minute_only_seen.get(key)
            if last is not None and now - last < MINUTE_ONLY_LOG_INTERVAL_S:
                return False
            self._minute_only_seen[key] = now
            return True

    def release(self, decision: Optional[QuotaDecision], reason_code: str) -> bool:
        if not decision or not decision.request_id or not self.enabled or self._ledger is None:
            return False
        try:
            return self._ledger.release(decision.request_id, str(reason_code or "CLIENT_RELEASE"))
        except Exception as exc:
            log.warning("UW_QUOTA_RELEASE_FAILED caller=%s error=%s", self.caller, type(exc).__name__)
            return False

    def status(self) -> Dict[str, Any]:
        if not self.enabled or self._ledger is None:
            return {"source": "disabled", "stale": True, "supabase_sync_enabled": False}
        data = self._ledger.status()
        data["supabase_sync_enabled"] = bool(self._worker and self._worker.enabled)
        return data

    def flush_outbox_once(self) -> int:
        if self._worker is None:
            return 0
        return self._worker.flush_once()

    def close(self, timeout_s: float = 2.0) -> bool:
        if self._worker is None:
            return True
        return self._worker.close(timeout_s)


_CLIENTS: Dict[str, SupabaseUWQuotaClient] = {}
_CLIENTS_LOCK = threading.Lock()


def get_uw_quota_client(caller: str) -> SupabaseUWQuotaClient:
    key = str(caller or "unknown")[:96]
    with _CLIENTS_LOCK:
        client = _CLIENTS.get(key)
        if client is None:
            client = SupabaseUWQuotaClient(caller=key)
            _CLIENTS[key] = client
        return client


def close_all_uw_quota_clients(timeout_s: float = 2.0) -> bool:
    with _WORKERS_LOCK:
        workers = list(_WORKERS.values())
        _WORKERS.clear()
    deadline_per_worker = max(0.0, timeout_s / max(1, len(workers)))
    return all(worker.close(deadline_per_worker) for worker in workers)


atexit.register(close_all_uw_quota_clients)


__all__ = [
    "QuotaDecision",
    "QuotaPriority",
    "SupabaseUWQuotaClient",
    "close_all_uw_quota_clients",
    "extract_uw_headers",
    "get_uw_quota_client",
    "quota_cycle_key",
]
