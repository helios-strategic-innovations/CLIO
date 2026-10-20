"""
Intelligence-file staleness sentinel.

Reads each known data/intelligence/*.json, extracts its internal "last updated"
timestamp (probing common field names; falling back to file mtime), compares
against a per-file SLA, and produces a structured report. The digest job
sends one Telegram at 08:00 ET if any file is past its SLA.

Pure observability — does not modify any intelligence files.

Flag: CLIO_STALENESS_SENTINEL_ENABLED (default "1"). Set to "0" to disable.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

logger = logging.getLogger("openclaw.staleness_sentinel")

_INTEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "intelligence")
)

_NY = ZoneInfo("America/New_York")

# Per-file SLAs. If internal timestamp is older than this, file is flagged.
#
# Derivation (S215 / R-16): SLA = maximum expected gap between successful writes
# (the job cadence as actually scheduled, plus any gate-imposed dead time) + 12 h
# slack for run latency and for the 08:00 ET digest landing before the day's run.
# Evidence per entry is the job registration in agent/integrations/scheduled_tasks.py
# and the module's own weekday/hour gate. Keep this table in step with those.
#
# Weekend rule: files in WEEKEND_ALLOWANCE_FILES are written only on weekdays, so
# when the window since the last write spans a Saturday, WEEKEND_ALLOWANCE is added.
# Files that run on weekends (news_intel, econ_calendar) or that skip only part of
# the week with an explicit table value (bond_intel, auction_calendar) get no allowance.
SLA_MAP: dict[str, timedelta] = {
    # news_intel.run_topic_monitor every 30 min, 08:00-20:00 ET, every day
    # (intel_topic_monitor interval=1800). Overnight dead time 20:00->08:00 = 12 h,
    # + 30 min interval + 1.5 h slack.
    "news_intel_state.json": timedelta(hours=14),
    # fed_analysis: daily job anchored 07:30 ET (run_repeating 86400); module returns
    # Sat/Sun (weekday>=5). 24 h cadence + 12 h slack; weekend allowance applies.
    "fed_analysis_state.json": timedelta(hours=36),
    # commodity_review: daily job anchored 06:30 ET; module weekday gate. 24 + 12; weekend allowance.
    "commodity_research_state.json": timedelta(hours=36),
    # bond_intel: analysis only Tue/Wed/Thu (run_bond_intel weekday gate, job registered
    # 08:00 ET). Worst gap is Thu 08:00 -> Tue 08:00 = 96 h, + 12 h slack. Explicit value,
    # no allowance (a Friday write never happens, so the Friday rule does not describe it).
    "bond_intel_state.json": timedelta(hours=108),
    # EVENT-DRIVEN: written only by geopolitical_research.escalate() (triggered by escalations,
    # no job in scheduled_tasks.py), so there is no cadence to derive an SLA from and a 36 h
    # SLA flagged it stale every morning (193 h on 2026-10-12). 14 days = "no escalation at
    # all for two weeks", which is the case worth a digest line.
    "geopolitical_research_state.json": timedelta(hours=336),
    # econ_calendar.refresh: daily job anchored 06:00 ET, no weekday gate (weekends run);
    # module gate only 05:00-22:00 ET. 24 + 12 h.
    "econ_calendar.json": timedelta(hours=36),
    # macro_research: daily job anchored 07:00 ET; module weekday gate. 24 + 12; weekend allowance.
    "macro_research_state.json": timedelta(hours=36),
    # TP-1 (S211): forced policy check when trade_policy_state.json is >7 d old, run from the
    # news_intel topic monitor (08:00-20:00 ET). 168 h + 12 h overnight dead time + 6 h slack.
    "trade_policy_state.json": timedelta(hours=186),
    # earnings_intel: refresh only when earnings_calendar.refreshed_at is >3 d old
    # (_calendar_is_stale), checked on weekdays 06:00-10:00 ET (daily job anchored 07:15).
    # 72 h + 12 h slack; weekend allowance covers the Wed refresh -> Mon check worst case (120 h).
    "earnings_calendar.json": timedelta(hours=84),
    # bond_intel._refresh_auction_calendar: refreshed Monday 06:00-10:00 ET only.
    # 168 h weekly cadence + 12 h slack. Explicit value, no allowance.
    "auction_calendar.json": timedelta(hours=180),
    # analytics_fun_facts_refresh: daily job anchored 20:05 ET, market days only (module
    # returns Sat/Sun). 24 + 12; weekend allowance (Fri 20:05 -> Mon 08:00 = 60 h).
    "analytics_fun_facts.json": timedelta(hours=36),
}

# Files whose job skips Saturday and Sunday (see SLA_MAP comments for evidence).
WEEKEND_ALLOWANCE_FILES: frozenset[str] = frozenset({
    "fed_analysis_state.json",
    "commodity_research_state.json",
    "macro_research_state.json",
    "earnings_calendar.json",
    "analytics_fun_facts.json",
})

# Extra SLA granted when the window since the last write spans a Saturday (48 h = the
# dead weekend in a 24 h-cadence weekday job, Fri write -> Mon run = 72 h).
WEEKEND_ALLOWANCE = timedelta(hours=48)


def _effective_sla(filename: str, last_ts: datetime, now_utc: datetime) -> timedelta:
    """Return the SLA for one file, adding WEEKEND_ALLOWANCE when the job skipped a weekend.

    The allowance applies only to WEEKEND_ALLOWANCE_FILES, and only when the last write
    happened before the most recent Saturday 00:00 ET (so a Friday-written file is covered
    on Monday, but a Friday-written file checked on Friday is not given extra slack).
    """
    base = SLA_MAP[filename]
    if filename not in WEEKEND_ALLOWANCE_FILES:
        return base
    now_et = now_utc.astimezone(_NY)
    days_since_sat = (now_et.weekday() - 5) % 7  # Mon=0 ... Sat=5 -> 0
    last_sat_midnight = (now_et - timedelta(days=days_since_sat)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    if last_ts.astimezone(_NY) < last_sat_midnight:
        return base + WEEKEND_ALLOWANCE
    return base

# Probe these field names in order to find the internal timestamp.
# S185 second-pass: extended to cover field names actually used in production
# intelligence files (analyzed_at, researched_at, datetime_et, sent_at, updated).
# Without these, 8 of 11 files fell back to file mtime for source attribution.
_TIMESTAMP_FIELD_CANDIDATES = (
    "last_updated",
    "refreshed_at",
    "timestamp",
    "fetched_at",
    "updated_at",
    "last_run",
    "last_fetch",
    "generated_at",
    "as_of",
    "analyzed_at",
    "researched_at",
    "datetime_et",
    "sent_at",
    "updated",
)


def _enabled() -> bool:
    return os.getenv("CLIO_STALENESS_SENTINEL_ENABLED", "1") == "1"


def _extract_internal_ts(payload: Any) -> tuple[datetime | None, str | None]:
    """Return (datetime, field_used) if we can find a parseable timestamp; else (None, None).

    Probes the top-level mapping for well-known field names, then walks one level
    into dict values to find a nested "researched_at" / "last_updated" for files
    like trade_policy_state.json that wrap data under a list.
    """
    if not isinstance(payload, dict):
        return None, None

    for field in _TIMESTAMP_FIELD_CANDIDATES:
        if field in payload:
            parsed = _parse_iso(payload[field])
            if parsed is not None:
                return parsed, field

    # Nested probe: find the newest timestamp among list-valued entries.
    newest: datetime | None = None
    newest_field: str | None = None
    for key, val in payload.items():
        if isinstance(val, list) and val:
            for item in val:
                if isinstance(item, dict):
                    for field in _TIMESTAMP_FIELD_CANDIDATES:
                        if field in item:
                            parsed = _parse_iso(item[field])
                            if parsed is not None and (newest is None or parsed > newest):
                                newest = parsed
                                newest_field = f"{key}[*].{field}"
    return newest, newest_field


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _scan_one(filename: str, now_utc: datetime | None = None) -> dict:
    path = os.path.join(_INTEL_DIR, filename)
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    if not os.path.isfile(path):
        return {
            "file": filename,
            "status": "MISSING",
            "exists": False,
            "sla_hours": SLA_MAP[filename].total_seconds() / 3600,
            "age_hours": None,
            "internal_ts": None,
            "mtime": None,
            "source": None,
        }

    mtime = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
    internal_ts: datetime | None = None
    source: str = "mtime"
    field_used: str | None = None

    try:
        with open(path, "r") as f:
            payload = json.load(f)
        internal_ts, field_used = _extract_internal_ts(payload)
    except Exception as e:
        logger.debug("failed to parse %s: %s", filename, e)

    if internal_ts is not None:
        ts = internal_ts
        source = f"internal:{field_used}"
    else:
        ts = mtime

    sla = _effective_sla(filename, ts, now_utc)
    age = now_utc - ts
    fresh = age <= sla

    return {
        "file": filename,
        "status": "FRESH" if fresh else "STALE",
        "exists": True,
        "sla_hours": round(sla.total_seconds() / 3600, 2),
        "age_hours": round(age.total_seconds() / 3600, 2),
        "internal_ts": ts.isoformat(),
        "mtime": mtime.isoformat(),
        "source": source,
    }


def _scan_all_sync() -> list[dict]:
    return [_scan_one(name) for name in SLA_MAP]


async def scan_staleness() -> list[dict]:
    """Produce the full staleness report. Offloads file I/O to default executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _scan_all_sync)


async def intelligence_staleness_scan_job(context) -> None:
    """PTB JobQueue callback. Scans + logs; does NOT send Telegram (that's the digest job)."""
    if not _enabled():
        return
    try:
        report = await scan_staleness()
    except Exception as e:
        logger.warning(f"staleness scan raised: {e}")
        return

    stale = [r for r in report if r["status"] == "STALE"]
    missing = [r for r in report if r["status"] == "MISSING"]

    if stale or missing:
        logger.info(
            "staleness scan: %d stale, %d missing out of %d",
            len(stale), len(missing), len(report),
        )
        for r in stale:
            logger.info(
                "  STALE %s age=%.1fh sla=%.1fh source=%s",
                r["file"], r["age_hours"] or 0, r["sla_hours"], r["source"],
            )
        for r in missing:
            logger.warning("  MISSING %s sla=%.1fh", r["file"], r["sla_hours"])
    else:
        logger.debug("staleness scan: all %d files fresh", len(report))


async def staleness_digest_job(context) -> None:
    """PTB JobQueue callback. Sends one daily Telegram digest if anything is stale."""
    if not _enabled():
        return
    try:
        report = await scan_staleness()
    except Exception as e:
        logger.warning(f"staleness digest raised: {e}")
        return

    problems = [r for r in report if r["status"] in ("STALE", "MISSING")]
    if not problems:
        logger.debug("staleness digest: no problems, skipping send")
        return

    lines = ["🕒 CLIO intelligence-file staleness digest:"]
    for r in problems:
        if r["status"] == "MISSING":
            lines.append(f"  • MISSING: {r['file']} (SLA {r['sla_hours']:.0f}h)")
        else:
            age_h = r["age_hours"] or 0
            lines.append(
                f"  • STALE: {r['file']} — {age_h:.1f}h old (SLA {r['sla_hours']:.0f}h)"
            )
    lines.append(f"\nET time: {datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M')}")
    text = "\n".join(lines)

    try:
        from agent.integrations.scheduled_tasks import _get_chat_ids, _safe_send
        for cid in _get_chat_ids():
            await _safe_send(context.bot, cid, text=text)
    except Exception as e:
        logger.debug(f"digest send failed: {e}")
