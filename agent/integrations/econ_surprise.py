"""
CLIO intel v3 (builder E) — Economic print SURPRISE engine.

Computes actual-vs-consensus "surprise" for scheduled US economic releases
(CPI, NFP, jobless claims, GDP, PCE, ISM, JOLTS, UMich sentiment, FOMC rate
decisions, ...) and hands the decider/Jev context they can react to, without
ever placing/sizing/cancelling/modifying an order itself (ADVISORY only —
this module never imports agent.dpof.broker/orders/engine).

Sources (per recon: scratchpad/recon/ECON_SURPRISE_SOURCES.md,
ECON_ACTUALS_AND_LOCAL_REPOS.md):
  - PRIMARY: Nasdaq's public (unofficial, keyless) economic-calendar JSON,
    `GET https://api.nasdaq.com/api/calendar/economicevents?date=YYYY-MM-DD`.
    Same endpoint OpenBB's own Nasdaq provider ships against in production —
    calling the public HTTP endpoint directly (not vendoring OpenBB's AGPL
    code) is not a licensing concern. Its `gmt` field is, despite the name,
    a plain ET HH:MM local time for US rows (recon-verified against known
    release times, e.g. "08:30" for Initial Jobless Claims).
  - FALLBACK/CROSS-CHECK: TradingView's internal calendar-widget JSON,
    `GET https://economic-calendar.tradingview.com/events`, ISO8601 UTC
    timestamps, richer shape (includes a numeric `importance` and separate
    `unit` field). Used whenever Nasdaq is unavailable/unhealthy this cycle,
    and cross-checked against Nasdaq's numbers when both report an actual
    for the same release (confirmed_by).
  - OPTIONAL ground truth: BLS API v2 (keyless), used ONLY from the slow
    refresh path to *confirm* an actual already recorded from Nasdaq/TV —
    never on the fast poll path, and off by default
    (CLIO_ECON_SURPRISE_BLS_CONFIRM=1 to enable). See _bls_confirm_recent().

Both Nasdaq's and TradingView's endpoints are unofficial/undocumented (no
published SLA, no auth). Every response is shape-validated before use; on
drift/malformed output we log a WARNING at most once per calendar day per
source, mark that source unhealthy for this cycle, and fall through to the
other source or simply skip this poll — this module NEVER crashes and NEVER
fabricates a value (an unparseable/missing number is always None, never 0).

State file: <repo>/data/intelligence/econ_surprise.json (env override
CLIO_ECON_SURPRISE_FILE), written atomically (tmp + fsync + os.replace) off
the event loop (loop.run_in_executor), mirroring agent/integrations/
market_regime.py's pattern. Readers are sync, mtime-cached, and NEVER raise.

Public contract (per scratchpad/INTEL_V3_SPEC.md, contract "E"):
  async def refresh_econ_surprise(context=None) -> dict | None    PTB job callback
  def get_recent_surprises(max_age_minutes: int = 240) -> list[dict]
  def get_upcoming_releases(hours: float = 6.0) -> list[dict]
  async def bootstrap_history(...) -> dict                        optional, off by default

Cadence — ONE TICK PER INVOCATION (2026-09-25 orchestrator fix: the earlier
design ran an internal ~11-minute sleep loop per invocation, which made a
release invisible to get_recent_surprises()/Telegram until the whole window
closed, and risked a PTB job outliving systemd's graceful-shutdown budget).
The orchestrator schedules refresh_econ_surprise as a repeating job at
JOB_INTERVAL_SECONDS (15s); each invocation does AT MOST one fetch cycle and
returns immediately:
  - If a scheduled importance>=2 release (today, per econ_calendar's cache)
    is inside its [T-1min, T+10min] window: ONE fetch of both sources for
    today, process, write state (only if something actually changed) and
    send any new Telegram alert(s), THEN RETURN — the next window-tick
    15s later is a separate invocation, not an inner loop. A non-blocking
    asyncio.Lock still guards against a slow invocation overlapping the next.
  - Otherwise, if the last consensus-table refresh is stale enough (30 min
    on weekdays 04:00-20:00 ET, else every 3h — see
    _slow_refresh_interval_minutes(), 2026-09-25 pass-2 item 5): does one
    slow refresh (Nasdaq + TV, next few days).
  - Otherwise: idle. The active-window and last-slow-refresh checks above
    are themselves the only I/O on an idle tick — both are dispatched via
    loop.run_in_executor() so they never block the event loop (2026-09-25
    pass-2 item 2: econ_calendar.get_todays_events() and the mtime-cache's
    fallback read are both synchronous file I/O) — but a genuinely idle
    tick never loads or writes the full state file.
  Each invocation is bounded by asyncio.wait_for(..., JOB_TIMEOUT_SECONDS-5)
  so a hung HTTP call can never make one tick run away.

Alerts are gated by importance (2026-09-25 pass-2 item 3): only
importance>=_IMPORTANCE_MIN_FOR_FAST_POLL surprises get a Telegram alert —
ALL surprises are still recorded in state["surprises"] regardless (DPOF
context may use the low-importance ones too). When a single tick yields
more than one alertable surprise (e.g. the first slow refresh after a
restart catching several releases at once), they collapse into ONE combined
digest message instead of a burst — see _dispatch_alerts().

State growth is bounded: alerted/consensus_table entries whose release_time
(parsed from the "canonical_key|release_time_et" key) is >14 days old are
pruned on every write (unparsable keys are dropped too); bls_checked entries
are pruned the same way by their date value. See _prune_state().

Late detection: if a surprise's actual is first recorded more than 15
minutes after its release_time_et (e.g. the first tick after a restart
catches an already-released print), the entry gets `"late": true` and the
Telegram text gets an appended "(released HH:MM ET, detected late)" note —
timely entries get `"late": false`. See _LATE_DETECTION_MINUTES.

Same-name/same-time collisions (2026-09-25 pass-2 item 1, CRITICAL): Nasdaq
can legitimately carry two distinct rows for the same event name at the
same release time with different units (e.g. "Building Permits" 08:00 has
both a %-MoM row and a level-in-millions row). Grouping/history/alert keys
therefore always disambiguate on unit class ("%" vs level) for FALLBACK
(unmapped) events — curated/mapped events (CPI MoM, NFP, ...) keep their
plain canonical name since they're always a single fixed unit in practice.
See _canonical_key_for_row().

Budget: both sources share the "econ_surprise" budget (agent/config.py
budget_econ_surprise, agent/rate_limiter.py two-site registration) — a
self-imposed courtesy cap; neither vendor publishes a quota for these
endpoints, but CLIO conventions require metered-API calls to go through
agent.rate_limiter.check_and_record(). Slow refresh is 5 calls/cycle (4
Nasdaq lookahead-days + 1 TradingView range call); fast-poll ticks are
15s x 2 sources x ~11min window ~= 88 calls/event.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
import statistics
import tempfile
import threading
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from agent.rate_limiter import BudgetExhaustedError, check_and_record

logger = logging.getLogger("openclaw.econ_surprise")

_NY = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")

# ============================================================
# Paths
# ============================================================

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_FILE = os.path.join(_REPO_ROOT, "data", "intelligence", "econ_surprise.json")


def _file_path() -> str:
    """Runtime-evaluated so tests can monkeypatch the env var per-test."""
    return os.environ.get("CLIO_ECON_SURPRISE_FILE", _DEFAULT_FILE)


# ============================================================
# HTTP endpoints / headers
# ============================================================

_NASDAQ_URL = "https://api.nasdaq.com/api/calendar/economicevents"
_NASDAQ_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/",
}

_TV_URL = "https://economic-calendar.tradingview.com/events"
_TV_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Origin": "https://www.tradingview.com",
    "Referer": "https://www.tradingview.com/",
}

_BLS_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

_HTTP_TIMEOUT_SECONDS = 15.0

# ============================================================
# Cadence knobs
# ============================================================
# JOB_INTERVAL_SECONDS / JOB_TIMEOUT_SECONDS are exported for the
# orchestrator's job registration (agent/integrations/scheduled_tasks.py,
# not this builder's file): refresh_econ_surprise is meant to be scheduled
# as a repeating job at JOB_INTERVAL_SECONDS, each invocation bounded to
# JOB_TIMEOUT_SECONDS wall-clock (see refresh_econ_surprise's own
# asyncio.wait_for). ONE TICK PER INVOCATION -- see module docstring.

JOB_INTERVAL_SECONDS = 15
JOB_TIMEOUT_SECONDS = 60
_WALLCLOCK_CAP_SECONDS = JOB_TIMEOUT_SECONDS - 5  # leaves margin under the job's own timeout

_FAST_POLL_PRE_MINUTES = 1
_FAST_POLL_POST_MINUTES = 10
_SLOW_REFRESH_LOOKAHEAD_DAYS = 3

# Slow refresh is 5 budget calls/cycle (4 Nasdaq lookahead-days + 1
# TradingView range call — see _fetch_all_sources) -- 2026-09-25 pass-2 item
# 5: a flat 30min/24-7 cadence burns that unnecessarily overnight/weekends
# when nothing new is scheduled. 30min only during the release-heavy weekday
# window; every 3h otherwise (evenings + weekends) still keeps a
# Monday-morning consensus reasonably fresh. See _slow_refresh_interval_minutes().
_SLOW_REFRESH_INTERVAL_MINUTES_ACTIVE = 30
_SLOW_REFRESH_INTERVAL_MINUTES_OFF = 180
_SLOW_REFRESH_ACTIVE_START_HOUR = 4
_SLOW_REFRESH_ACTIVE_END_HOUR = 20

_HISTORY_CAP = 60
_MIN_HISTORY_FOR_Z = 8
_Z_LOOKBACK = 36
_SURPRISES_CAP = 300

_IMPORTANCE_MIN_FOR_FAST_POLL = 2
_LATE_DETECTION_MINUTES = 15

# Bounds on state growth (2026-09-25 orchestrator fix) -- alerted/
# consensus_table keys are "canonical_key|release_time_et"; entries whose
# release_time is older than this are pruned on every write. bls_checked is
# keyed by canonical_key -> ISO date string, pruned the same way by that date.
_STATE_PRUNE_TTL_DAYS = 14


# ============================================================
# Value parsing (K/M/B/T/% suffix strings, comma thousands, "&nbsp;"/blank)
# ============================================================

_UNIT_SUFFIXES = ("K", "M", "B", "T")
_BLANK_TOKENS = {"", "-", "—", "n/a", "na"}


def parse_value(raw) -> tuple[float | None, str | None]:
    """Parse a Nasdaq/TradingView-style value into (float, unit).

    unit is one of "%", "K", "M", "B", "T", or "level" (no suffix) whenever a
    number was parsed; (None, None) for anything unparseable/blank/missing —
    NEVER 0 (CLAUDE.md: "missing is never zero").

    Examples: "197K"->(197.0,"K"), "-0.3%"->(-0.3,"%"), "4.1%"->(4.1,"%"),
    "1.2M"->(1.2,"M"), "$-78.3B"->(-78.3,"B"), ""->(None,None),
    " "->(None,None) [blank consensus], "&nbsp;"->(None,None) [Nasdaq's
    blank marker for e.g. Fed-speaker rows], "1,719K"->(1719.0,"K").
    """
    if raw is None:
        return None, None
    if isinstance(raw, bool):
        return None, None
    if isinstance(raw, (int, float)):
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return None, None
        if v != v:  # NaN
            return None, None
        return v, "level"
    if not isinstance(raw, str):
        return None, None

    s = raw.replace("&nbsp;", "").strip()
    if s.lower() in _BLANK_TOKENS:
        return None, None

    unit = "level"
    if s.endswith("%"):
        unit = "%"
        s = s[:-1].strip()
    elif s and s[-1].upper() in _UNIT_SUFFIXES:
        unit = s[-1].upper()
        s = s[:-1].strip()

    s = s.replace(",", "").replace("$", "").strip()
    if not s or s in {"-", "+", ".", "-."}:
        return None, None
    try:
        val = float(s)
    except ValueError:
        return None, None
    if val != val:  # NaN guard (shouldn't happen post float(), defensive)
        return None, None
    return val, unit


# ============================================================
# Event-name normalization / category classification
# ============================================================
# Ordered (most-specific-first) so e.g. "Core CPI" is matched before the bare
# "CPI" pattern. category drives market_read's sign convention:
#   inflation / fed  -> above=hawkish, below=dovish
#   inverted         -> above=growth_down, below=growth_up (claims/unemployment:
#                       a HIGHER print is BAD for growth)
#   growth           -> above=growth_up, below=growth_down
# Anything that doesn't match falls back to (whitespace-normalized raw name,
# category=None) — still tracked (surprise/z computed) but market_read stays
# None (never guessed) since we don't know its economic sign.

_EVENT_PATTERNS: tuple[tuple[re.Pattern, str, str], ...] = (
    (re.compile(r"\bcore\s*cpi\b", re.I), "Core CPI MoM", "inflation"),
    (re.compile(r"\bcpi\b.*\byoy\b|\byoy\b.*\bcpi\b", re.I), "CPI YoY", "inflation"),
    (re.compile(r"\bcpi\b", re.I), "CPI MoM", "inflation"),
    (re.compile(r"\bcore\s*pce\b", re.I), "Core PCE MoM", "inflation"),
    (re.compile(r"\bpce\b", re.I), "PCE MoM", "inflation"),
    (re.compile(r"\bppi\b", re.I), "PPI MoM", "inflation"),
    (re.compile(r"\bnon.?farm\s*payrolls\s*private\b", re.I), "NFP Private", "growth"),
    (re.compile(r"\bnon.?farm\s*payrolls\b|\bnfp\b", re.I), "NFP", "growth"),
    (re.compile(r"\bunemployment\s*rate\b", re.I), "Unemployment Rate", "inverted"),
    (re.compile(r"\binitial\s*jobless\s*claims\b", re.I), "Initial Jobless Claims", "inverted"),
    (re.compile(r"\bcontinuing\s*(jobless\s*)?claims\b", re.I), "Continuing Jobless Claims", "inverted"),
    (re.compile(r"\bgdp\b", re.I), "GDP QoQ", "growth"),
    (re.compile(r"\bretail\s*sales\b", re.I), "Retail Sales MoM", "growth"),
    (re.compile(r"\bism\s*manufacturing\b", re.I), "ISM Manufacturing PMI", "growth"),
    (re.compile(r"\bism\s*services\b", re.I), "ISM Services PMI", "growth"),
    (re.compile(r"\bjolts\b|\bjob\s*openings\b", re.I), "JOLTS", "growth"),
    (re.compile(r"\bmichigan\b.*\bsentiment\b|\bumich\b", re.I), "UMich Sentiment", "growth"),
    (
        re.compile(r"\bfomc\b|\bfed\s*funds\s*rate\b|\bfederal\s*funds\s*rate\b|\brate\s*decision\b", re.I),
        "Fed Funds Rate decision",
        "fed",
    ),
)


def _canonical_key(name: str) -> str:
    """Map a raw event name to a canonical history/dedup key."""
    if not name or not isinstance(name, str):
        return "Unknown"
    for pattern, key, _cat in _EVENT_PATTERNS:
        if pattern.search(name):
            return key
    return " ".join(name.split())


def _event_category(name: str) -> str | None:
    if not name or not isinstance(name, str):
        return None
    for pattern, _key, cat in _EVENT_PATTERNS:
        if pattern.search(name):
            return cat
    return None


def _row_unit_class(row: dict) -> str:
    """"%" vs "level" (the level bucket covers K/M/B/T/level and rows with
    no parseable unit at all). Prefers actual_unit, then consensus_unit,
    then previous_unit; a row with no parseable value anywhere defaults to
    "level" (arbitrary but deterministic — there's nothing to disambiguate
    on a fully-blank row)."""
    for u in (row.get("actual_unit"), row.get("consensus_unit"), row.get("previous_unit")):
        if u:
            return "%" if u == "%" else "level"
    return "level"


def _canonical_key_for_row(row: dict) -> str:
    """The merge/history/table key for one normalized row (2026-09-25
    orchestrator pass-2 fix — CRITICAL same-name/same-time collision).
    Nasdaq's calendar can legitimately carry TWO distinct rows for the same
    event name at the same release time with different units (recon-
    verified against nasdaq_20260925.json: "Building Permits" 08:00 has both
    a %-MoM row AND a separate level-in-millions row; same for "New Home
    Sales" 10:00) — grouping purely by (canonical_key, release_time_et)
    silently drops one (whichever loses the dict-overwrite, i.e. row-order
    dependent) and would mix %-scale and level-scale surprises into the
    same z-score history, producing a meaningless surprise_z.

    For a CURATED/pinned event (matched by _EVENT_PATTERNS — CPI MoM, NFP,
    ... these are always a single fixed unit in practice, signaled by
    _event_category() being non-None) the key is the canonical name
    UNCHANGED, same as before ("Existing mapped keys ... should keep their
    current canonical names"). For a FALLBACK (unmapped, raw-name) event,
    the key gets a " (%)" / " (level)" suffix based on THIS row's own unit,
    so the two variants land in separate merge groups / separate history
    series / separate table_key rows deterministically, regardless of row
    order. A TradingView row carrying only one variant pairs with the
    Nasdaq row of the SAME unit class only — it can never attach to the
    other, since the key is derived purely from the row's own unit, not
    from insertion order."""
    name = row.get("event", "")
    base = _canonical_key(name)
    if _event_category(name) is not None:
        return base
    return f"{base} ({_row_unit_class(row)})"


def _classify_importance(name: str) -> int:
    """Reuse econ_calendar's TE-equivalent 1-3 classifier (lazy import per
    CLIO convention — avoids a hard top-level dependency on econ_calendar)."""
    try:
        from agent.integrations.econ_calendar import _classify_importance as _ec_classify
        return _ec_classify(name)
    except Exception as e:
        logger.debug(f"econ_surprise: _classify_importance fallback (1): {type(e).__name__}: {e}")
        return 1


# ============================================================
# Direction / market_read / z-score
# ============================================================

_TICK_EPSILON = {
    "%": 0.05,
    "K": 2.0,
    "M": 0.05,
    "B": 0.25,
    "T": 0.005,
    "level": 0.5,
}


def _direction(surprise_raw: float, unit: str | None) -> str:
    eps = _TICK_EPSILON.get(unit or "level", 0.5)
    if abs(surprise_raw) < eps:
        return "inline"
    return "above" if surprise_raw > 0 else "below"


def _market_read(category: str | None, direction: str) -> str | None:
    """None when the event's economic sign is unknown (unmapped event) —
    never guessed. "neutral" when the sign IS known but the print was inline."""
    if category is None:
        return None
    if direction == "inline":
        return "neutral"
    if category in ("inflation", "fed"):
        return "hawkish" if direction == "above" else "dovish"
    if category == "inverted":
        return "growth_down" if direction == "above" else "growth_up"
    if category == "growth":
        return "growth_up" if direction == "above" else "growth_down"
    return None


def _update_history(history: dict, canonical_key: str, surprise_raw: float) -> tuple[list[float], float | None]:
    """Appends surprise_raw to history[canonical_key] (capped _HISTORY_CAP).

    surprise_z is computed from the PRIOR history only (before this point is
    appended) — avoids a self-referential z-score on small samples. Requires
    >= _MIN_HISTORY_FOR_Z prior points (per contract: "None if <8 history
    pts"), stdev over the last _Z_LOOKBACK of those.
    """
    prior = list(history.get(canonical_key, []))
    z = None
    if len(prior) >= _MIN_HISTORY_FOR_Z:
        window = prior[-_Z_LOOKBACK:]
        try:
            sd = statistics.pstdev(window)
        except statistics.StatisticsError:
            sd = 0.0
        if sd > 1e-9:
            z = round(surprise_raw / sd, 3)
    series = prior + [surprise_raw]
    if len(series) > _HISTORY_CAP:
        series = series[-_HISTORY_CAP:]
    history[canonical_key] = series
    return series, z


def _values_agree(a: float | None, a_unit: str | None, b: float | None, b_unit: str | None,
                   rel_tol: float = 0.1, abs_tol: float = 0.5) -> bool:
    if a is None or b is None or a_unit != b_unit:
        return False
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b), 1e-9))


# ============================================================
# Nasdaq fetch + normalize
# ============================================================

def _nasdaq_no_records(payload) -> bool:
    """Nasdaq's documented empty-day answer (verified live 2026-09-27):
    {"data": null, "status": {"rCode": 200, "bCodeMessage": [{"code": 1002,
    "errorMessage": "Economic Events Calendar: No record found."}]}}."""
    if not isinstance(payload, dict) or payload.get("data") is not None:
        return False
    status = payload.get("status")
    if not isinstance(status, dict) or status.get("rCode") != 200:
        return False
    msgs = status.get("bCodeMessage")
    if not isinstance(msgs, list):
        return False
    return any(isinstance(m, dict) and m.get("code") == 1002 for m in msgs)


def _validate_nasdaq_shape(payload) -> bool:
    if not isinstance(payload, dict):
        return False
    data = payload.get("data")
    if not isinstance(data, dict):
        return False
    rows = data.get("rows")
    if not isinstance(rows, list):
        return False
    for row in rows[:5]:
        if not isinstance(row, dict):
            return False
        if not all(k in row for k in ("gmt", "country", "eventName", "actual", "consensus", "previous")):
            return False
    return True


def _mark_unhealthy(state: dict, source: str, reason: str) -> None:
    today = datetime.now(_NY).date().isoformat()
    health = state.setdefault("source_health", {}).setdefault(source, {"healthy": True, "last_warn_date": None})
    health["healthy"] = False
    if health.get("last_warn_date") != today:
        logger.warning(f"econ_surprise: {source} response shape drift/unhealthy: {reason}")
        health["last_warn_date"] = today


def _mark_healthy(state: dict, source: str) -> None:
    state.setdefault("source_health", {})[source] = {"healthy": True, "last_warn_date": None}


def _nasdaq_time_to_et(gmt: str, date_et: date) -> datetime | None:
    """Nasdaq's `gmt` field is, despite the name, a plain ET HH:MM local time
    for US rows (recon-verified: "08:30" for Initial Jobless Claims, which
    really does release at 8:30 AM ET) — NOT literal GMT/UTC. All-day rows
    ("24H") aren't a clock time and return None."""
    if not gmt or ":" not in gmt:
        return None
    try:
        hh, mm = gmt.split(":", 1)
        return datetime(date_et.year, date_et.month, date_et.day, int(hh), int(mm), tzinfo=_NY)
    except (ValueError, TypeError):
        return None


def _nasdaq_row_to_common(row: dict, date_et: date) -> dict | None:
    name = (row.get("eventName") or "").strip()
    if not name:
        return None
    dt = _nasdaq_time_to_et((row.get("gmt") or "").strip(), date_et)
    if dt is None:
        return None
    actual, actual_unit = parse_value(row.get("actual"))
    consensus, consensus_unit = parse_value(row.get("consensus"))
    previous, previous_unit = parse_value(row.get("previous"))
    return {
        "event": name,
        "release_time_et": dt.isoformat(),
        "actual": actual, "actual_unit": actual_unit,
        "consensus": consensus, "consensus_unit": consensus_unit,
        "previous": previous, "previous_unit": previous_unit,
        "importance": _classify_importance(name),
        "source": "nasdaq",
    }


async def _fetch_nasdaq_raw(date_et: date, state: dict) -> list[dict] | None:
    """Returns normalized US-only rows for one date, or None on ANY failure
    (budget/HTTP/network/shape) — never raises, never fabricates."""
    try:
        await check_and_record("econ_surprise")
    except BudgetExhaustedError:
        logger.warning("econ_surprise: budget exhausted, skipping Nasdaq fetch")
        return None

    try:
        import httpx
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_SECONDS, follow_redirects=True) as client:
            resp = await client.get(_NASDAQ_URL, headers=_NASDAQ_HEADERS, params={"date": date_et.isoformat()})
    except Exception as e:
        logger.debug(f"econ_surprise: Nasdaq fetch error: {type(e).__name__}: {e}")
        return None

    if resp.status_code != 200:
        logger.debug(f"econ_surprise: Nasdaq HTTP {resp.status_code}")
        return None

    try:
        payload = resp.json()
    except Exception as e:
        _mark_unhealthy(state, "nasdaq", f"non-JSON response ({type(e).__name__})")
        return None

    if _nasdaq_no_records(payload):
        # A day with no scheduled releases (e.g. a Sunday): a valid, empty answer —
        # not drift. Otherwise every weekend in the 3-day lookahead marks Nasdaq unhealthy.
        _mark_healthy(state, "nasdaq")
        return []
    if not _validate_nasdaq_shape(payload):
        _mark_unhealthy(state, "nasdaq", "unexpected response shape")
        return None
    _mark_healthy(state, "nasdaq")

    rows = payload["data"]["rows"]
    out = []
    for row in rows:
        if not isinstance(row, dict) or row.get("country") != "United States":
            continue
        normalized = _nasdaq_row_to_common(row, date_et)
        if normalized is not None:
            out.append(normalized)
    return out


# ============================================================
# TradingView fetch + normalize
# ============================================================

def _validate_tv_shape(payload) -> bool:
    if not isinstance(payload, dict):
        return False
    status = payload.get("status")
    if status is not None and status != "ok":
        return False
    result = payload.get("result")
    if not isinstance(result, list):
        return False
    for row in result[:5]:
        if not isinstance(row, dict):
            return False
        if not all(k in row for k in ("title", "date", "actual", "forecast", "previous")):
            return False
    return True


def _parse_tv_datetime(date_str) -> datetime | None:
    if not date_str or not isinstance(date_str, str):
        return None
    try:
        s = date_str[:-1] + "+00:00" if date_str.endswith("Z") else date_str
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_UTC)
        return dt.astimezone(_NY)
    except (ValueError, TypeError):
        return None


def _tv_numeric(raw, unit_field, title: str) -> tuple[float | None, str | None]:
    if raw is None:
        return None, None
    if isinstance(raw, bool):
        return None, None
    if isinstance(raw, (int, float)):
        val = float(raw)
        if val != val:
            return None, None
        if unit_field == "%":
            return val, "%"
        if isinstance(unit_field, str) and unit_field:
            return val, unit_field
        # TradingView omits `unit` for the payrolls-family labor-market count
        # series, which are conventionally reported in thousands (recon
        # fixture tv_nfp_20260904.json: NFP actual=162 == "162K"). Keeps
        # units consistent with Nasdaq's parallel "162K" string for the same
        # canonical series so surprise/history math never silently mixes
        # scales.
        if "payroll" in (title or "").lower():
            return val, "K"
        return val, "level"
    # defensive: string fallback via the same parser used for Nasdaq
    return parse_value(raw)


def _tv_row_to_common(row: dict) -> dict | None:
    title = (row.get("title") or row.get("indicator") or "").strip()
    if not title:
        return None
    dt = _parse_tv_datetime(row.get("date"))
    if dt is None:
        return None
    unit_field = row.get("unit")
    actual, actual_unit = _tv_numeric(row.get("actual"), unit_field, title)
    consensus, consensus_unit = _tv_numeric(row.get("forecast"), unit_field, title)
    previous, previous_unit = _tv_numeric(row.get("previous"), unit_field, title)

    imp = _classify_importance(title)
    tv_imp = row.get("importance")
    if isinstance(tv_imp, (int, float)) and tv_imp >= 1 and imp < 2:
        imp = 2  # TradingView's own numeric importance signal, per spec ("or TradingView importance")

    return {
        "event": title,
        "release_time_et": dt.isoformat(),
        "actual": actual, "actual_unit": actual_unit,
        "consensus": consensus, "consensus_unit": consensus_unit,
        "previous": previous, "previous_unit": previous_unit,
        "importance": imp,
        "source": "tradingview",
    }


async def _fetch_tradingview_raw(start_et: date, end_et: date, state: dict) -> list[dict] | None:
    """Returns normalized US rows for [start_et, end_et] inclusive, or None
    on ANY failure — never raises, never fabricates."""
    try:
        await check_and_record("econ_surprise")
    except BudgetExhaustedError:
        logger.warning("econ_surprise: budget exhausted, skipping TradingView fetch")
        return None

    start_utc = datetime(start_et.year, start_et.month, start_et.day, tzinfo=_NY).astimezone(_UTC)
    end_utc = (datetime(end_et.year, end_et.month, end_et.day, tzinfo=_NY) + timedelta(days=1)).astimezone(_UTC)
    params = {
        "from": start_utc.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "to": end_utc.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "countries": "US",
    }

    try:
        import httpx
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_SECONDS, follow_redirects=True) as client:
            resp = await client.get(_TV_URL, headers=_TV_HEADERS, params=params)
    except Exception as e:
        logger.debug(f"econ_surprise: TradingView fetch error: {type(e).__name__}: {e}")
        return None

    if resp.status_code != 200:
        logger.debug(f"econ_surprise: TradingView HTTP {resp.status_code}")
        return None

    try:
        payload = resp.json()
    except Exception as e:
        _mark_unhealthy(state, "tradingview", f"non-JSON response ({type(e).__name__})")
        return None

    if not _validate_tv_shape(payload):
        _mark_unhealthy(state, "tradingview", "unexpected response shape")
        return None
    _mark_healthy(state, "tradingview")

    out = []
    for row in payload["result"]:
        normalized = _tv_row_to_common(row) if isinstance(row, dict) else None
        if normalized is not None:
            out.append(normalized)
    return out


async def _fetch_all_sources(start_date: date, end_date: date, state: dict) -> dict[str, list[dict] | None]:
    """Nasdaq's endpoint is date-at-a-time; TradingView takes a range."""
    nasdaq_any_ok = False
    nasdaq_rows: list[dict] = []
    d = start_date
    while d <= end_date:
        raw = await _fetch_nasdaq_raw(d, state)
        if raw is not None:
            nasdaq_any_ok = True
            nasdaq_rows.extend(raw)
        d += timedelta(days=1)

    tv_rows = await _fetch_tradingview_raw(start_date, end_date, state)

    return {
        "nasdaq": nasdaq_rows if nasdaq_any_ok else None,
        "tradingview": tv_rows,
    }


# ============================================================
# Merge / surprise computation
# ============================================================

def _merge_rows_by_event(rows_by_source: dict[str, list[dict] | None]) -> dict[tuple[str, str], dict[str, dict]]:
    """Groups by (unit-class-disambiguated canonical key, release_time_et) —
    see _canonical_key_for_row()'s docstring for why the key must include
    unit class, not just the event name."""
    grouped: dict[tuple[str, str], dict[str, dict]] = {}
    for source, rows in rows_by_source.items():
        if not rows:
            continue
        for row in rows:
            key = (_canonical_key_for_row(row), row["release_time_et"])
            grouped.setdefault(key, {})[source] = row
    return grouped


def _process_rows(state: dict, rows_by_source: dict[str, list[dict] | None],
                   now: datetime | None = None) -> list[dict]:
    """Merges Nasdaq/TradingView rows, updates the consensus table, and
    computes+persists any NEW surprise (dedup via state["alerted"]). Returns
    the list of newly-recorded surprise entries (contract shape, plus the
    extra "late" bool — see _LATE_DETECTION_MINUTES) for the caller to alert
    on. Mutates `state` in place. Never raises.

    `now` defaults to datetime.now(_NY) when omitted (kept optional so the
    many existing 2-arg call sites/tests are unaffected); callers on the
    per-tick refresh path pass the tick's own `now` for determinism."""
    if now is None:
        now = datetime.now(_NY)
    grouped = _merge_rows_by_event(rows_by_source)
    new_entries: list[dict] = []

    consensus_table = state.setdefault("consensus_table", {})
    alerted = state.setdefault("alerted", {})
    history = state.setdefault("history", {})
    surprises = state.setdefault("surprises", [])

    for (canonical_key, release_time_et), by_source in grouped.items():
        primary = by_source.get("nasdaq") or by_source.get("tradingview")
        if primary is None:
            continue

        table_key = f"{canonical_key}|{release_time_et}"
        consensus_row = by_source.get("nasdaq")
        if consensus_row is None or consensus_row.get("consensus") is None:
            tv_row = by_source.get("tradingview")
            if tv_row is not None and tv_row.get("consensus") is not None:
                consensus_row = tv_row
        if consensus_row is None:
            consensus_row = primary
        consensus_table[table_key] = {
            "event": primary["event"],
            "consensus": consensus_row.get("consensus"),
            "previous": consensus_row.get("previous"),
            "importance": primary.get("importance", 1),
        }

        if table_key in alerted:
            continue  # already recorded a surprise for this exact release

        candidate = None
        for src in ("nasdaq", "tradingview"):
            row = by_source.get(src)
            if row and row.get("actual") is not None and row.get("consensus") is not None:
                candidate = row
                break
        if candidate is None:
            continue  # no usable actual+consensus pair yet this poll

        unit = candidate.get("consensus_unit")
        if candidate.get("actual_unit") != unit:
            logger.debug(
                f"econ_surprise: unit mismatch for {canonical_key!r} "
                f"(actual_unit={candidate.get('actual_unit')!r} consensus_unit={unit!r}) — skipping"
            )
            continue

        actual = candidate["actual"]
        consensus = candidate["consensus"]
        surprise_raw = round(actual - consensus, 6)
        direction = _direction(surprise_raw, unit)
        category = _event_category(primary["event"])
        market_read = _market_read(category, direction)
        _series, z = _update_history(history, canonical_key, surprise_raw)

        confirmed_by = [candidate["source"]]
        other_src = "tradingview" if candidate["source"] == "nasdaq" else "nasdaq"
        other_row = by_source.get(other_src)
        if other_row and other_row.get("actual") is not None:
            if _values_agree(actual, candidate.get("actual_unit"), other_row["actual"], other_row.get("actual_unit")):
                confirmed_by.append(other_src)

        late = False
        try:
            release_dt = datetime.fromisoformat(release_time_et)
            if release_dt.tzinfo is None:
                release_dt = release_dt.replace(tzinfo=_NY)
            late = (now - release_dt).total_seconds() > _LATE_DETECTION_MINUTES * 60
        except (ValueError, TypeError):
            pass  # unparsable release_time_et: leave late=False rather than guess

        entry = {
            "event": primary["event"],
            "release_time_et": release_time_et,
            "actual": actual,
            "consensus": consensus,
            "previous": candidate.get("previous"),
            "unit": unit,
            "surprise_raw": surprise_raw,
            "surprise_z": z,
            "direction": direction,
            "market_read": market_read,
            "importance": primary.get("importance", 1),
            "source": candidate["source"],
            "confirmed_by": confirmed_by,
            "as_of": now.isoformat(),
            "late": late,
        }
        surprises.append(entry)
        if len(surprises) > _SURPRISES_CAP:
            del surprises[: len(surprises) - _SURPRISES_CAP]
        alerted[table_key] = True
        new_entries.append(entry)

    return new_entries


# ============================================================
# State load/save (atomic, off-loop)
# ============================================================

def _default_state() -> dict:
    return {
        "surprises": [],
        "history": {},
        "alerted": {},
        "consensus_table": {},
        "source_health": {
            "nasdaq": {"healthy": True, "last_warn_date": None},
            "tradingview": {"healthy": True, "last_warn_date": None},
        },
        "bls_checked": {},
        "last_slow_refresh_at": None,
        "as_of": None,
    }


_state_cache_lock = threading.Lock()
_state_cache: dict[str, tuple[tuple[int, int, int], dict]] = {}


def _read_state_cached(path: str | None = None) -> dict | None:
    path = path or _file_path()
    try:
        st = os.stat(path)
    except OSError:
        return None
    # Inode too: writers replace the file atomically (new inode per write), and two writes of the
    # same size can share an mtime tick -- (mtime, size) alone then serves the stale content.
    key = (st.st_ino, st.st_mtime_ns, st.st_size)
    with _state_cache_lock:
        hit = _state_cache.get(path)
        if hit is not None and hit[0] == key:
            return hit[1]
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        logger.debug(f"econ_surprise: unreadable state file {path} ({type(e).__name__}: {e})")
        return None
    if not isinstance(data, dict):
        return None
    with _state_cache_lock:
        _state_cache[path] = (key, data)
    return data


def _load_state_for_write() -> dict:
    """BLOCKING. Loads (or defaults) the state, backfilling any missing
    top-level keys so older/partial files never KeyError downstream."""
    raw = _read_state_cached()
    merged = _default_state()
    if isinstance(raw, dict):
        for k in merged:
            if k in raw:
                merged[k] = raw[k]
    for nested_key in ("history", "alerted", "consensus_table", "source_health", "bls_checked"):
        if not isinstance(merged.get(nested_key), dict):
            merged[nested_key] = {}
    if not isinstance(merged.get("surprises"), list):
        merged["surprises"] = []
    return merged


def _get_last_slow_refresh_at() -> str | None:
    """Cheap check for the idle-path decision: reads via the mtime-cached
    reader (no full _load_state_for_write() merge, no executor hop) so an
    idle 15s tick costs one os.stat() (typically already satisfied from the
    in-process cache) instead of a full state load."""
    state = _read_state_cached()
    if not isinstance(state, dict):
        return None
    val = state.get("last_slow_refresh_at")
    return val if isinstance(val, str) else None


def _release_time_from_key(key: str) -> datetime | None:
    """alerted/consensus_table keys are "canonical_key|release_time_et_iso".
    Returns the parsed release_time_et, or None if the key can't be parsed
    (callers treat None as droppable — see _prune_state)."""
    if not isinstance(key, str) or "|" not in key:
        return None
    _prefix, _sep, ts = key.rpartition("|")
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_NY)
        return dt
    except (ValueError, TypeError):
        return None


def _prune_state(state: dict, now: datetime) -> None:
    """Bounds state growth (2026-09-25 orchestrator fix): the 3-day-lookahead
    slow refresh adds alerted/consensus_table keys forever otherwise. Drops
    entries whose release_time (parsed from the key) is older than
    _STATE_PRUNE_TTL_DAYS, and unparsable keys unconditionally (never let a
    key that can't be aged out accumulate indefinitely either). bls_checked
    is pruned the same way by its date VALUE (it's keyed by canonical_key,
    not "key|time"). Mutates `state` in place. Never raises."""
    try:
        cutoff = now - timedelta(days=_STATE_PRUNE_TTL_DAYS)
        for field in ("alerted", "consensus_table"):
            table = state.get(field)
            if not isinstance(table, dict):
                continue
            drop = []
            for key in table:
                dt = _release_time_from_key(key)
                if dt is None or dt < cutoff:
                    drop.append(key)
            for key in drop:
                del table[key]

        bls_checked = state.get("bls_checked")
        if isinstance(bls_checked, dict):
            bls_cutoff = (now - timedelta(days=_STATE_PRUNE_TTL_DAYS)).date()
            drop_bls = []
            for key, val in bls_checked.items():
                try:
                    d = date.fromisoformat(val)
                except (ValueError, TypeError):
                    drop_bls.append(key)
                    continue
                if d < bls_cutoff:
                    drop_bls.append(key)
            for key in drop_bls:
                del bls_checked[key]
    except Exception as e:  # noqa: BLE001 — pruning must never break a write
        logger.debug(f"econ_surprise: _prune_state failed (non-critical): {type(e).__name__}: {e}")


def _write_state_blocking(state: dict) -> bool:
    """BLOCKING. tmp + fsync + os.replace. Never raises; returns False (and
    leaves the previous file untouched) on any failure. FD-leak-safe (per
    CLAUDE.md's documented pattern, 2026-09-25 pass-2 item 6): `fd` is set
    to -1 the moment os.fdopen() hands ownership to the file object `f` (so
    the `with` block's own close is the only close on the success path);
    if mkstemp succeeds but fdopen itself raises before that point, `fd`
    still holds the raw descriptor and the `finally` closes it directly."""
    path = _file_path()
    d = os.path.dirname(path)
    fd = -1
    tmp_path = None
    try:
        os.makedirs(d, exist_ok=True)
        payload = dict(state)
        payload["as_of"] = datetime.now(_NY).isoformat()
        data_str = json.dumps(payload, indent=2, default=str)  # serialize BEFORE mkstemp
        fd, tmp_path = tempfile.mkstemp(dir=d, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                fd = -1  # file object now owns the fd -- don't close it again below
                f.write(data_str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
            tmp_path = None
        finally:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        return True
    except OSError as e:
        logger.warning(f"econ_surprise: atomic write failed, keeping previous file ({type(e).__name__}: {e})")
        return False


# ============================================================
# Cadence orchestration
# ============================================================

def _active_release_windows(now: datetime) -> list[dict]:
    """Today's importance>=2 econ_calendar events whose fast-poll window
    [t-1min, t+10min] currently contains `now`."""
    try:
        from agent.integrations import econ_calendar as _ec
        todays = _ec.get_todays_events()
    except Exception as e:
        logger.debug(f"econ_surprise: econ_calendar.get_todays_events() failed: {type(e).__name__}: {e}")
        return []

    active = []
    for ev in todays:
        try:
            if ev.get("importance", 1) < _IMPORTANCE_MIN_FOR_FAST_POLL:
                continue
            dt = datetime.fromisoformat(ev["datetime_et"])
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_NY)
        except (KeyError, ValueError, TypeError):
            continue
        window_start = dt - timedelta(minutes=_FAST_POLL_PRE_MINUTES)
        window_end = dt + timedelta(minutes=_FAST_POLL_POST_MINUTES)
        if window_start <= now <= window_end:
            active.append(ev)
    return active


def _slow_refresh_interval_minutes(now: datetime) -> int:
    """30 min on weekdays 04:00-20:00 ET (keeps the fast-reacting consensus
    table fresh during the release-heavy window); every 3h otherwise
    (evenings + weekends) — see the module-level constants' comment for the
    5-calls/cycle budget rationale (2026-09-25 pass-2 item 5)."""
    is_weekday = now.weekday() < 5
    in_active_window = _SLOW_REFRESH_ACTIVE_START_HOUR <= now.hour < _SLOW_REFRESH_ACTIVE_END_HOUR
    if is_weekday and in_active_window:
        return _SLOW_REFRESH_INTERVAL_MINUTES_ACTIVE
    return _SLOW_REFRESH_INTERVAL_MINUTES_OFF


def _should_run_slow_refresh(last: str | None, now: datetime) -> bool:
    """`last` is the raw last_slow_refresh_at string (or None) — callers on
    the cheap idle-check path pass _get_last_slow_refresh_at()'s result
    directly, without loading the full state dict."""
    if not last:
        return True
    try:
        last_dt = datetime.fromisoformat(last)
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=_NY)
    except (ValueError, TypeError):
        return True
    return (now - last_dt).total_seconds() >= _slow_refresh_interval_minutes(now) * 60


async def _run_fast_tick(state: dict, now: datetime) -> list[dict]:
    """ONE fetch of both sources for today, processed once. No internal
    sleep/loop — the orchestrator's JOB_INTERVAL_SECONDS-cadence job IS the
    poll loop now (2026-09-25 fix: the old in-invocation ~11min sleep loop
    delayed state writes/alerts until the whole window closed and risked
    outliving systemd's graceful-shutdown budget). Idempotent via
    state["alerted"] dedup, same as before."""
    date_et = now.date()
    rows_by_source = await _fetch_all_sources(date_et, date_et, state)
    return _process_rows(state, rows_by_source, now)


async def _slow_refresh(state: dict, now: datetime) -> list[dict]:
    start = now.date()
    end = start + timedelta(days=_SLOW_REFRESH_LOOKAHEAD_DAYS)
    rows_by_source = await _fetch_all_sources(start, end, state)
    new_entries = _process_rows(state, rows_by_source, now)
    state["last_slow_refresh_at"] = now.isoformat()

    if os.environ.get("CLIO_ECON_SURPRISE_BLS_CONFIRM", "0") == "1":
        try:
            await _bls_confirm_recent(state, now)
        except Exception as e:
            logger.debug(f"econ_surprise: BLS confirm pass failed (non-critical): {type(e).__name__}: {e}")

    return new_entries


# ============================================================
# Telegram alert (dedup via state["alerted"], set before this is called)
# ============================================================

def _fmt_num(v) -> str:
    if v is None:
        return "N/A"
    try:
        if float(v).is_integer():
            return str(int(v))
        return f"{v:g}"
    except (TypeError, ValueError):
        return str(v)


def _format_alert_line(entry: dict) -> str:
    """One entry's alert text — shared by the single-alert and digest paths
    so both render identically."""
    unit = entry.get("unit") or ""
    unit_sfx = "%" if unit == "%" else (unit if unit and unit != "level" else "")
    z = entry.get("surprise_z")
    market_read = entry.get("market_read")
    if z is not None and market_read:
        z_txt = f" ({z:+.1f}σ, {market_read})"
    elif market_read:
        z_txt = f" ({market_read})"
    else:
        z_txt = ""

    confirmed = entry.get("confirmed_by") or []
    if len(confirmed) >= 2:
        src_txt = f"{confirmed[0].capitalize()}, {confirmed[1].capitalize()} agrees"
    elif confirmed:
        src_txt = confirmed[0].capitalize()
    else:
        src_txt = str(entry.get("source", "")).capitalize()

    line = (
        f"{entry.get('event')} {_fmt_num(entry.get('actual'))}{unit_sfx} "
        f"vs {_fmt_num(entry.get('consensus'))}{unit_sfx} consensus{z_txt} "
        f"— source {src_txt}"
    )

    if entry.get("late"):
        try:
            rt = datetime.fromisoformat(entry["release_time_et"])
            if rt.tzinfo is None:
                rt = rt.replace(tzinfo=_NY)
            line += f" (released {rt.strftime('%H:%M')} ET, detected late)"
        except (KeyError, ValueError, TypeError):
            line += " (detected late)"
    return line


async def _send_alert(context, entry: dict) -> None:
    try:
        msg = _format_alert_line(entry)
        if context is None or getattr(context, "bot", None) is None:
            logger.info(f"econ_surprise: alert (no Telegram context, log-only): {msg}")
            return

        from agent.integrations.scheduled_tasks import _get_chat_ids, _safe_send
        for cid in _get_chat_ids():
            await _safe_send(context.bot, cid, text=msg)
    except Exception as e:
        logger.debug(f"econ_surprise: alert send failed (non-critical): {type(e).__name__}: {e}")


async def _send_digest_alert(context, entries: list[dict]) -> None:
    """Combines >1 alertable surprises from a SINGLE tick into ONE Telegram
    message (e.g. the first slow refresh after a restart catching several
    already-released prints at once) instead of a burst of separate sends
    (2026-09-25 orchestrator pass-2 fix, item 3)."""
    try:
        lines = [_format_alert_line(e) for e in entries]
        msg = f"{len(entries)} economic surprises:\n" + "\n".join(lines)
        if context is None or getattr(context, "bot", None) is None:
            logger.info(f"econ_surprise: digest alert (no Telegram context, log-only): {msg}")
            return

        from agent.integrations.scheduled_tasks import _get_chat_ids, _safe_send
        for cid in _get_chat_ids():
            await _safe_send(context.bot, cid, text=msg)
    except Exception as e:
        logger.debug(f"econ_surprise: digest alert send failed (non-critical): {type(e).__name__}: {e}")


async def _dispatch_alerts(context, new_entries: list[dict]) -> None:
    """Alerts ONLY on importance>=_IMPORTANCE_MIN_FOR_FAST_POLL surprises
    (same threshold as fast-poll window activation) — ALL surprises are
    already recorded in state["surprises"] regardless of importance (DPOF
    context may still use the low-importance ones; see item 3). >1
    alertable entry from one tick collapses into a single digest message."""
    alertable = [e for e in new_entries if e.get("importance", 1) >= _IMPORTANCE_MIN_FOR_FAST_POLL]
    if not alertable:
        return
    if len(alertable) == 1:
        await _send_alert(context, alertable[0])
    else:
        await _send_digest_alert(context, alertable)


# ============================================================
# Optional: BLS confirmation (slow path only, off by default)
# ============================================================

_BLS_SERIES_MAP = {
    "CPI MoM": "CUSR0000SA0",
    "Core CPI MoM": "CUSR0000SA0L1E",
    "NFP": "CES0000000001",
    "Unemployment Rate": "LNS14000000",
    "PPI MoM": "WPSFD4",
}


async def _bls_confirm_recent(state: dict, now: datetime) -> None:
    """OPTIONAL, off by default (CLIO_ECON_SURPRISE_BLS_CONFIRM=1). Never
    called from the fast-poll path. For each BLS-coverable curated series
    with a surprise recorded in the last 3 days that hasn't been BLS-checked
    today, fetches BLS's latest published value (keyless API v2) and, if
    it's within tolerance of our recorded `actual`, appends "bls" to that
    entry's confirmed_by list. Any per-series failure is a no-op — never
    raises, never fabricates a confirmation."""
    bls_checked = state.setdefault("bls_checked", {})
    today = now.date().isoformat()
    cutoff = now - timedelta(days=3)

    try:
        import httpx
    except ImportError:
        return

    for entry in state.get("surprises", []):
        key = _canonical_key(entry.get("event", ""))
        series_id = _BLS_SERIES_MAP.get(key)
        if not series_id:
            continue
        if "bls" in (entry.get("confirmed_by") or []):
            continue
        if bls_checked.get(key) == today:
            continue
        try:
            rt = datetime.fromisoformat(entry["release_time_et"])
            if rt.tzinfo is None:
                rt = rt.replace(tzinfo=_NY)
        except (KeyError, ValueError, TypeError):
            continue
        if rt < cutoff:
            continue

        bls_checked[key] = today  # mark checked regardless of outcome — respects the 25/day keyless cap
        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_SECONDS, follow_redirects=True) as client:
                resp = await client.get(_BLS_URL + series_id)
        except Exception as e:
            logger.debug(f"econ_surprise: BLS confirm fetch failed for {key}: {type(e).__name__}: {e}")
            continue
        if resp.status_code != 200:
            continue
        try:
            payload = resp.json()
            series_list = payload.get("Results", {}).get("series", [])
            points = series_list[0].get("data", []) if series_list else []
            if not points:
                continue
            latest_val = float(points[0].get("value"))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError):
            continue

        actual = entry.get("actual")
        if actual is None:
            continue
        if abs(latest_val - actual) <= max(0.5, 0.02 * abs(actual)):
            confirmed = entry.setdefault("confirmed_by", [])
            if "bls" not in confirmed:
                confirmed.append("bls")


# ============================================================
# Public: refresh_econ_surprise (PTB job callback)
# ============================================================

_refresh_lock = asyncio.Lock()


async def _do_refresh(context) -> dict:
    """ONE TICK. Cheapest path first: check for an active fast-poll window
    (no full state load needed — only econ_calendar, dispatched off-loop via
    run_in_executor since econ_calendar.get_todays_events() does a blocking
    open()+json.load()). If none, cheaply check whether a slow refresh is
    due (the mtime-cached reader, also off-loop — it can fall through to a
    blocking read on a cache miss). Only when there's actual work to do do
    we load the full state, and even then the write is skipped when nothing
    in it actually changed (deepcopy diff — new fetch data, health flips,
    and consensus-table updates all count as "changed"; a re-poll that
    finds nothing new does not). Every genuinely idle tick (no active
    window, slow refresh not yet due) costs exactly those two off-loop
    reads — no state load, no write (see module docstring)."""
    now = datetime.now(_NY)
    loop = asyncio.get_event_loop()

    active = await loop.run_in_executor(None, _active_release_windows, now)
    if active:
        state = await loop.run_in_executor(None, _load_state_for_write)
        before = copy.deepcopy(state)
        new_entries = await _run_fast_tick(state, now)
        if state != before:
            _prune_state(state, now)
            await loop.run_in_executor(None, _write_state_blocking, state)
        await _dispatch_alerts(context, new_entries)
        return {
            "mode": "fast_poll",
            "active_windows": len(active),
            "new_surprises": len(new_entries),
            "events": [e["event"] for e in new_entries],
        }

    last_slow = await loop.run_in_executor(None, _get_last_slow_refresh_at)
    if _should_run_slow_refresh(last_slow, now):
        state = await loop.run_in_executor(None, _load_state_for_write)
        new_entries = await _slow_refresh(state, now)  # always changes last_slow_refresh_at -> always write
        _prune_state(state, now)
        await loop.run_in_executor(None, _write_state_blocking, state)
        await _dispatch_alerts(context, new_entries)
        return {
            "mode": "slow_refresh",
            "active_windows": 0,
            "new_surprises": len(new_entries),
            "events": [e["event"] for e in new_entries],
        }

    return {"mode": "idle", "active_windows": 0, "new_surprises": 0, "events": []}


async def refresh_econ_surprise(context=None) -> dict | None:
    """PTB job callback. The orchestrator schedules this as a repeating job
    at JOB_INTERVAL_SECONDS (15s), ALL DAY. ONE TICK PER INVOCATION (see
    module docstring / _do_refresh) — this function decides internally
    whether this tick is inside a live release window, due for a slow
    consensus refresh (cadence varies — see _slow_refresh_interval_minutes()),
    or idle, does AT MOST one fetch cycle, and returns immediately; it never
    runs an internal sleep loop.
    Never raises — any failure logs and returns None. An asyncio.Lock
    prevents a slow tick from overlapping the next one; each tick is itself
    bounded by asyncio.wait_for(JOB_TIMEOUT_SECONDS-5) so a hung HTTP call
    can't run away."""
    if _refresh_lock.locked():
        logger.debug("econ_surprise: refresh already in progress, skipping this tick")
        return None
    async with _refresh_lock:
        try:
            return await asyncio.wait_for(_do_refresh(context), timeout=_WALLCLOCK_CAP_SECONDS)
        except asyncio.TimeoutError:
            logger.warning("econ_surprise: tick exceeded wall-clock cap, aborting")
            return None
        except Exception as e:
            logger.error(f"econ_surprise: refresh failed: {type(e).__name__}: {e}")
            return None


# ============================================================
# Public: readers (sync, mtime-cached, never raise)
# ============================================================

def get_recent_surprises(max_age_minutes: int = 240) -> list[dict]:
    """Sync, mtime-cached, NEVER raises. [] when the file is missing/corrupt
    or nothing recent enough exists."""
    try:
        state = _read_state_cached()
        if not state:
            return []
        now = datetime.now(_NY)
        out = []
        for entry in state.get("surprises", []):
            as_of = entry.get("as_of")
            if not isinstance(as_of, str):
                continue
            try:
                dt = datetime.fromisoformat(as_of)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=_NY)
            except (TypeError, ValueError):
                continue
            age_min = (now - dt).total_seconds() / 60.0
            if age_min <= max_age_minutes:
                out.append(entry)
        out.sort(key=lambda e: e.get("release_time_et", ""), reverse=True)
        return out
    except Exception as e:  # noqa: BLE001 — public read API: must never raise
        logger.debug(f"econ_surprise: get_recent_surprises failed: {type(e).__name__}: {e}")
        return []


def get_upcoming_releases(hours: float = 6.0) -> list[dict]:
    """Sync, mtime-cached, NEVER raises. Base schedule comes from
    econ_calendar's own cache (already the canonical UW/FRED/FOMC-fallback
    calendar); consensus/previous are overlaid from this module's own
    Nasdaq/TradingView-derived consensus table when available (richer,
    already-parsed numbers) and fall back to econ_calendar's own
    forecast/previous strings (parsed via parse_value) otherwise."""
    try:
        now = datetime.now(_NY)
        try:
            from agent.integrations import econ_calendar as _ec
            cal_events = _ec.get_upcoming_events(hours=hours)
        except Exception as e:
            logger.debug(f"econ_surprise: econ_calendar.get_upcoming_events() failed: {type(e).__name__}: {e}")
            cal_events = []

        state = _read_state_cached()
        consensus_table = (state or {}).get("consensus_table", {}) if isinstance(state, dict) else {}

        out = []
        for ev in cal_events:
            try:
                dt = datetime.fromisoformat(ev["datetime_et"])
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=_NY)
            except (KeyError, ValueError, TypeError):
                continue
            name = ev.get("name", "")
            base_key = _canonical_key(name)
            # econ_calendar's event names carry no parsed unit, so we can't
            # reproduce _canonical_key_for_row()'s exact disambiguated key —
            # pinned/curated events never get a suffix (unchanged lookup);
            # fallback events try both variants with a deterministic
            # preference ("%" first, then "level") per the 2026-09-25
            # orchestrator pass-2 fix (item 1).
            if _event_category(name) is not None:
                candidate_keys = [base_key]
            else:
                candidate_keys = [f"{base_key} (%)", f"{base_key} (level)"]
            table_entry = None
            for ck in candidate_keys:
                table_entry = consensus_table.get(f"{ck}|{dt.isoformat()}")
                if table_entry:
                    break
            if table_entry:
                event_name = table_entry.get("event", name)
                consensus = table_entry.get("consensus")
                previous = table_entry.get("previous")
                importance = table_entry.get("importance", ev.get("importance", 1))
            else:
                event_name = name
                consensus, _u = parse_value(ev.get("forecast"))
                previous, _u2 = parse_value(ev.get("previous"))
                importance = ev.get("importance", 1)
            out.append({
                "event": event_name,
                "release_time_et": dt.isoformat(),
                "importance": importance,
                "consensus": consensus,
                "previous": previous,
                "minutes_until": round((dt - now).total_seconds() / 60.0, 1),
            })
        out.sort(key=lambda e: e.get("release_time_et", ""))
        return out
    except Exception as e:  # noqa: BLE001 — public read API: must never raise
        logger.debug(f"econ_surprise: get_upcoming_releases failed: {type(e).__name__}: {e}")
        return []


# ============================================================
# Optional: bootstrap_history — off by default, never auto-invoked
# ============================================================

async def bootstrap_history(canonical_key: str | None = None, lookback_days: int = 730,
                             max_points: int = _HISTORY_CAP) -> dict:
    """OPTIONAL, bounded, OFF by default — never called from
    refresh_econ_surprise or any scheduled path. Seeds state["history"] from
    TradingView's historical range endpoint (the only one of the two sources
    that accepts an arbitrary from/to window — Nasdaq's is date-at-a-time,
    too many calls for a multi-year backfill). Intended for a one-off,
    operator-invoked call (a script or a future admin command), e.g. to warm
    up z-score history for a newly-added canonical series faster than 8
    live releases would.

    canonical_key=None bootstraps every canonical series found in the range;
    passing one key scopes it to just that series — note the key must match
    _canonical_key_for_row()'s disambiguated shape for a fallback (unmapped)
    event, e.g. "Building Permits (%)", to match what the live pipeline
    writes to state["history"] (2026-09-25 orchestrator pass-2 fix, item 1).
    Returns {"seeded": {key: n_points, ...}, "source": "tradingview"}.
    """
    loop = asyncio.get_event_loop()
    state = await loop.run_in_executor(None, _load_state_for_write)
    now = datetime.now(_NY)
    start = (now - timedelta(days=max(0, lookback_days))).date()
    end = now.date()

    seeded: dict[str, int] = {}
    tv_raw = await _fetch_tradingview_raw(start, end, state)
    if tv_raw is not None:
        by_key: dict[str, list[tuple[str, float]]] = {}
        for row in tv_raw:
            if row.get("actual") is None or row.get("consensus") is None:
                continue
            if row.get("actual_unit") != row.get("consensus_unit"):
                continue
            key = _canonical_key_for_row(row)
            if canonical_key is not None and key != canonical_key:
                continue
            surprise = row["actual"] - row["consensus"]
            by_key.setdefault(key, []).append((row["release_time_et"], surprise))

        history = state.setdefault("history", {})
        for key, points in by_key.items():
            points.sort(key=lambda p: p[0])
            values = [p[1] for p in points][-max_points:]
            history[key] = values
            seeded[key] = len(values)

    await loop.run_in_executor(None, _write_state_blocking, state)
    return {"seeded": seeded, "source": "tradingview"}
