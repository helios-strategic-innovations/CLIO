"""
OpenClaw/Moltbot — Economic Calendar Integration

Pulls the weekly economic calendar from Unusual Whales /market/economic-calendar,
with a FRED-backed fallback (releases/dates + a curated release-id table) and a
hardcoded FOMC decision-date list, so DPOF's economic-event blackout gate
(agent/dpof/data.py's _check_econ_gate, CALENDAR_MAX_AGE_SECONDS-gated) survives a
UW outage instead of going unhealthy. Provides get_upcoming_events() for
event-triggered news intelligence searches.

S187 (2026-04-30): Migrated from Trading Economics (TE) to UW. TE had a 100/month
cap that got exhausted mid-month, cascading into stale econ_calendar → all 3
downstream intel modules' silent exits. UW economic-calendar endpoint is part
of the existing 5K/day UW budget (~7-8 calls/day for daily refresh = 0.16% utilization).
Mirrors Hydrinia's CC-UW-3 swap (uw_market_analyzer.py:893).

2026-09-25 (DPOF econ-blackout resilience): fetch_calendar_with_fallback() adds
FRED (agent/dpof brief STEP1: CLIO's config already carries FRED_API_KEY —
agent/config.py's fred_api_key — but until this change nothing in agent/ actually
called the FRED API; Hydrinia's FREDClient in hydrinia.py:14389 was the only
live consumer of that key anywhere in this org) as a second source and a static
FOMC decision-date table as a third. run_calendar_refresh() now saves (and thus
refreshes CALENDAR_CACHE_PATH's mtime — the only signal DPOF's file-presence
health check can see, per _econ_calendar_read_sync()'s docstring in data.py)
whenever AT LEAST ONE live source (UW or FRED) produced a genuine response this
cycle, even an empty one — not only when UW specifically had events. It is
withheld only when BOTH live sources failed/were unavailable, so a stale cache
still ages toward DPOF-unhealthy exactly as before ("missing is never zero" —
see agent/dpof/models.py and data.py's module docstring). The static FOMC list
never flips that signal by itself: see fetch_calendar_with_fallback()'s docstring.

Cache file shape unchanged for downstream consumers (macro_research, fed_analysis,
news_intel via get_upcoming_events); a `source` field ("uw"|"fred"|"fomc_static")
is additive, mirroring how `reported_period` was added in the TE→UW migration.
"""

import json
import logging
import os
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from agent.config import config
from agent.rate_limiter import check_and_record, BudgetExhaustedError

logger = logging.getLogger("openclaw.econ_calendar")

_NY = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "intelligence",
)
CALENDAR_CACHE_PATH = os.path.join(_DATA_DIR, "econ_calendar.json")
CALENDAR_MAX_AGE_DAYS = 7

# UW economic-calendar endpoint (mirrors Hydrinia's CC-UW-3 pattern)
_UW_BASE_URL = "https://api.unusualwhales.com/api"
_UW_CAL_PATH = "/market/economic-calendar"

# FRED fallback endpoint. NOTE: /fred/releases/dates (plural "releases") takes
# no release_id param at all -- it returns dates for ALL ~330+ FRED releases in
# the realtime window (verified against fred.stlouisfed.org/docs/api/fred/
# releases_dates.html, 2026-09-25); the per-release singular /fred/release/dates
# endpoint (not used here) is the one that filters server-side. We fetch the
# broad list and filter client-side to _FRED_RELEASE_TABLE below -- this keeps
# it to 1 HTTP call/refresh and avoids feeding macro_research/fed_analysis/
# news_intel hundreds of irrelevant regional/international release rows.
_FRED_BASE_URL = "https://api.stlouisfed.org/fred"
_FRED_RELEASES_DATES_PATH = "/releases/dates"
_FRED_CALL_TIMEOUT_SECONDS = 15.0

# release_id -> {name, time_et, importance}. release_ids verified 2026-09-25 via
# each release's fred.stlouisfed.org/release?rid=<id> page (web search/fetch,
# not the live API -- CLIO has no FRED key in this dev sandbox). `name` is
# deliberately phrased to ALSO pass _classify_importance()'s keyword matcher
# (defense in depth in case a future caller re-derives importance from the name
# string instead of reading the `importance` field) -- but `importance` below is
# always set explicitly from this table, never derived from `name`, because
# FRED's own official release names ("Consumer Price Index", "Advance Monthly
# Sales for Retail and Food Services", "Personal Income and Outlays") do NOT
# reliably contain _classify_importance()'s substrings (e.g. "Consumer Price
# Index" contains neither "cpi" nor "inflation rate" nor "core cpi").
# HIGH=3 set matches _classify_importance()'s _HIGH_IMPACT_KEYWORDS docstring
# contract: FOMC/CPI/PPI/NFP/GDP/PCE/Retail Sales.
#
# GAP: ISM Manufacturing/Services (PMI) is NOT available from FRED -- all 22
# ISM series were removed from the FRED database/API in 2016 (St. Louis Fed
# news.research.stlouisfed.org/2016/06/institute-for-supply-management-data-to
# -be-removed-from-fred/, confirmed 2026-09-25). No FRED release_id exists for
# it; if UW is down, ISM prints are simply not covered by this fallback.
_FRED_RELEASE_TABLE: dict[int, dict] = {
    10:  {"name": "CPI (Consumer Price Index)", "time_et": "08:30", "importance": 3},
    46:  {"name": "PPI (Producer Price Index)", "time_et": "08:30", "importance": 3},
    50:  {"name": "Employment Situation (Non-Farm Payrolls)", "time_et": "08:30", "importance": 3},
    53:  {"name": "GDP (Gross Domestic Product)", "time_et": "08:30", "importance": 3},
    54:  {"name": "PCE (Personal Income and Outlays)", "time_et": "08:30", "importance": 3},
    9:   {"name": "Retail Sales (Advance Monthly Sales)", "time_et": "08:30", "importance": 3},
    # FRED's own "FOMC Press Release" release (id 101) exists but its dated
    # coverage isn't reliable enough to depend on alone -- kept here as a bonus
    # (dedup will fold it into the hardcoded _FOMC_DECISION_DATES_ET row for the
    # same day if both are present) but never the SOLE FOMC source; see
    # fetch_fomc_static(). R-14 (2026-10-12): FRED returned a row for every day
    # of a week (phantom 14:00 "statements" 10-12..10-18), so release 101 rows
    # are accepted only on a static decision day -- see _is_static_fomc_day().
    101: {"name": "FOMC Press Release / Statement", "time_et": "14:00", "importance": 3},
    180: {"name": "Initial Jobless Claims", "time_et": "08:30", "importance": 2},
    192: {"name": "JOLTS (Job Openings and Labor Turnover)", "time_et": "10:00", "importance": 2},
    91:  {"name": "Consumer Sentiment (Michigan)", "time_et": "10:00", "importance": 2},
    13:  {"name": "Industrial Production and Capacity Utilization", "time_et": "09:15", "importance": 2},
    27:  {"name": "Housing Starts (New Residential Construction)", "time_et": "08:30", "importance": 2},
}
_FRED_FOMC_RELEASE_ID = 101

# Hardcoded FOMC decision-day dates (third source, STEP2) -- the second day of
# each regularly scheduled 2-day meeting, when the post-meeting statement/rate
# decision is announced at 14:00 ET. Verified 2026-09-25 against
# federalreserve.gov/monetarypolicy/fomccalendars.htm (cross-checked against a
# second independent source the same day) -- UNVERIFIED beyond that single
# verification pass; re-check against the live page if this list is still in
# use past 2027, and extend annually (mirrors the CLAUDE.md note that
# Hydrinia's market_events half-day/holiday sets are "2026-only").
_FOMC_DECISION_DATES_ET: tuple[str, ...] = (
    # 2026
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
    # 2027
    "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-09",
    "2027-07-28", "2027-09-15", "2027-10-27", "2027-12-08",
)
# TODO(operator): add 2027 FOMC decision dates when the Fed publishes them


# ============================================================
# Importance Classifier (TE-equivalent 1-3 scale)
# ============================================================
# UW does not return an `impact` field on this endpoint (all None as of S187).
# Derive importance from event-name keywords. Mirrors TE's high-impact set so
# downstream filters (macro_research uses importance >= 3) keep behaving as before.

_HIGH_IMPACT_KEYWORDS = (
    "fomc", "fed funds", "rate decision", "fed chair", "powell",
    "cpi", "inflation rate", "core cpi",
    "ppi", "producer price",
    "non-farm", "nonfarm", "employment situation", "payrolls", "unemployment rate",
    "gdp ", "gdp,", "gross domestic",
    "pce", "personal consumption expenditures",
    "ism manufacturing", "ism services",
    "retail sales",
)
_MEDIUM_IMPACT_KEYWORDS = (
    "consumer sentiment", "consumer confidence",
    "industrial production", "capacity utilization",
    "trade balance",
    "housing starts", "building permits", "existing home", "new home",
    "durable goods",
    "jolts", "job opening", "adp employment", "initial jobless",
    "philly fed", "empire state", "chicago pmi", "kansas city fed",
    "case-shiller", "case shiller",
    "michigan",
    "auction",  # Treasury auctions
    "fed speak", "fed speech", "fed governor", "fed president", "fed board",
)


def _classify_importance(event_name: str) -> int:
    """Map event name to TE-compatible importance level (1=low, 2=medium, 3=high).

    Replaces TE's `Importance` field which UW doesn't provide. Conservative:
    unknown events default to 1 so they don't trigger high-impact downstream
    workflows (macro_research filters importance>=3 only).
    """
    if not event_name or not isinstance(event_name, str):
        return 1
    name_lower = event_name.lower()
    if any(kw in name_lower for kw in _HIGH_IMPACT_KEYWORDS):
        return 3
    if any(kw in name_lower for kw in _MEDIUM_IMPACT_KEYWORDS):
        return 2
    return 1


# ============================================================
# UW Date Parsing
# ============================================================

def _parse_uw_time(time_str: str) -> datetime | None:
    """Parse UW's ISO8601 UTC timestamp to ET-localized datetime."""
    if not time_str:
        return None
    try:
        # UW format: "2026-05-08T14:00:00Z"
        s = str(time_str).rstrip("Z") + "+00:00" if str(time_str).endswith("Z") else str(time_str)
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_UTC)
        return dt.astimezone(_NY)
    except (ValueError, TypeError):
        return None


def _normalize_event(raw: dict) -> dict | None:
    """Normalize a UW economic-calendar event to our standard cache format.

    Returns None if the event can't be parsed.
    Output shape preserved from TE migration so consumers don't change.
    """
    dt = _parse_uw_time(raw.get("time", ""))
    if dt is None:
        return None

    name = (raw.get("event") or "").strip()
    if not name:
        return None

    # UW's `prev` and `forecast` may be None or numeric/percent strings
    forecast = raw.get("forecast")
    previous = raw.get("prev")
    period = raw.get("reported_period") or ""

    return {
        "name": name,
        "category": (raw.get("type") or "").strip(),  # UW: "report", "speech", etc.
        "datetime_et": dt.isoformat(),
        "importance": _classify_importance(name),
        "forecast": "" if forecast is None else str(forecast),
        "previous": "" if previous is None else str(previous),
        "actual": "",  # UW doesn't return actuals on this endpoint; downstream
                       # didn't rely on it for upcoming events
        "country": "United States",
        "reported_period": period,  # additive; TE didn't have this
        "source": "uw",  # additive (2026-09-25); see _normalize_fred_release / fetch_fomc_static
    }


def _is_static_fomc_day(dt: datetime) -> bool:
    """True iff dt's ET calendar date is a hardcoded FOMC decision day.

    Used to gate FRED release 101 (R-14): FRED's FOMC release rows are only
    trusted on the real decision dates in _FOMC_DECISION_DATES_ET.
    """
    return dt.astimezone(_NY).date().isoformat() in _FOMC_DECISION_DATES_ET


def _longest_consecutive_run(days: set[date]) -> int:
    """Length of the longest run of consecutive calendar days in `days`."""
    best = run = 0
    prev = None
    for d in sorted(days):
        run = run + 1 if prev is not None and (d - prev).days == 1 else 1
        best = max(best, run)
        prev = d
    return best


def _normalize_fred_release(item: dict) -> dict | None:
    """Normalize one FRED /releases/dates item to our standard cache format.

    Returns None for anything not in _FRED_RELEASE_TABLE (client-side filter —
    see the module-level comment on _FRED_RELEASES_DATES_PATH for why: the
    plural endpoint returns ALL releases, not just the majors we care about)
    or with an unparseable date. importance/time_et come from the table, never
    from FRED's own release_name (see _FRED_RELEASE_TABLE docstring comment).
    """
    if not isinstance(item, dict):
        return None
    try:
        release_id = int(item.get("release_id"))
    except (TypeError, ValueError):
        return None
    meta = _FRED_RELEASE_TABLE.get(release_id)
    if meta is None:
        return None
    date_str = item.get("date")
    if not date_str:
        return None
    try:
        d = date.fromisoformat(str(date_str))
    except (ValueError, TypeError):
        return None
    try:
        hh, mm = meta["time_et"].split(":")
        dt = datetime(d.year, d.month, d.day, int(hh), int(mm), tzinfo=_NY)
    except (ValueError, TypeError):
        return None

    if release_id == _FRED_FOMC_RELEASE_ID and not _is_static_fomc_day(dt):
        logger.debug(f"FRED FOMC release row {d.isoformat()} is not a static decision day; dropped (R-14)")
        return None

    return {
        "name": meta["name"],
        "category": "report",
        "datetime_et": dt.isoformat(),
        "importance": meta["importance"],
        "forecast": "",  # FRED /releases/dates carries no forecast/actual figures
        "previous": "",
        "actual": "",
        "country": "United States",
        "reported_period": "",
        "source": "fred",
    }


def _dedup_key(ev: dict) -> tuple:
    """(ET date, ET HH:MM, importance bucket) — see _merge_events docstring.

    The release TIME is part of the key: the DPOF blackout is a +/- window
    around each event's time, so two same-day importance-3 events at different
    times (e.g. CPI 08:30 and FOMC 14:00) must both survive the merge.
    """
    dt_str = ev.get("datetime_et", "") or ""
    try:
        dt = datetime.fromisoformat(dt_str)
        d, t = dt.date().isoformat(), dt.strftime("%H:%M")
    except (ValueError, TypeError):
        d, t = str(dt_str)[:10], str(dt_str)[11:16]
    imp = ev.get("importance")
    bucket = "high" if imp == 3 else ("medium" if imp == 2 else "low")
    return (d, t, bucket)


def _merge_events(primary: list[dict], secondary: list[dict]) -> list[dict]:
    """Merge two normalized event lists, keeping `primary`'s row whenever both
    have an event at the same ET date + HH:MM + importance bucket
    (high/medium/low) — "prefer UW rows when fresh". Events at different times
    are always kept, so every blackout window survives the merge (a same-day
    CPI 08:30 and FOMC 14:00 are two rows). Two distinct releases at the SAME
    minute and bucket collapse to one row — harmless for the blackout (same
    window), only informational detail is reduced.
    """
    seen = {_dedup_key(ev) for ev in primary}
    merged = list(primary)
    for ev in secondary:
        key = _dedup_key(ev)
        if key in seen:
            continue
        seen.add(key)
        merged.append(ev)
    return merged


# ============================================================
# API Client
# ============================================================

async def _fetch_uw_calendar_raw(days_ahead: int = 7) -> tuple[list[dict] | None, bool]:
    """Core UW fetch + normalize. Returns (events, ok).

    ok=False means UW did not produce a usable answer this cycle (no key,
    budget exhausted, HTTP/network/parse failure) — events is None in that
    case. ok=True means UW gave a genuine response, which may legitimately be
    an empty list (a real "nothing scheduled" week, not a failure).

    The ok=True/False distinction hinges on uw_flow_intel._uw_get's actual
    contract (read 2026-09-25, agent/integrations/uw_flow_intel.py:169-314):
    EVERY failure branch in _uw_get (no key, budget/quota denied, non-200,
    exception) returns a bare `{}` with no "data" key; only a genuine 200
    response returns resp.json(), which for this endpoint is `{"data": [...]}`
    with the key always present (possibly `[]`). So "data" in payload is a
    reliable success signal — `payload.get("data", [])` (the pre-fallback
    code) could NOT tell a real empty week from a failed call, which is
    exactly the ambiguity fetch_calendar_with_fallback() needs resolved to
    decide whether to try FRED and whether the refresh counts as "a live
    source worked" for DPOF's cache-freshness health signal.
    """
    api_key = config.uw_api_key or ""
    if not api_key:
        logger.debug("UW API key not configured, skipping calendar fetch")
        return None, False

    # Budget gate
    try:
        await check_and_record("unusual_whales")
    except BudgetExhaustedError:
        logger.warning("UW budget exhausted, skipping calendar fetch")
        return None, False

    # P1-11: reuse the shared UW singleton client. uw_flow_intel._uw_get owns the
    # module-level httpx.AsyncClient (connection pooling + 429 handling) — this used to
    # be a 3rd independent UW HTTP path opening a fresh AsyncClient per call. Same
    # endpoint (_UW_BASE_URL + _UW_CAL_PATH == _uw_get's _API_BASE + path) and same
    # Bearer auth. skip_moltbot_budget=True: the UW budget was already charged via
    # check_and_record above, so _uw_get must not double-count. On any non-200 _uw_get
    # returns {} → treated as failure below (see docstring).
    try:
        from agent.integrations.uw_flow_intel import _uw_get
        payload = await _uw_get(_UW_CAL_PATH, skip_moltbot_budget=True)
    except Exception as e:
        # Avoid exc_info=True per S118 W-075 (auth header may end up in trace)
        logger.error(f"UW economic-calendar fetch failed: {type(e).__name__}: {str(e)[:200]}")
        return None, False

    if not isinstance(payload, dict) or "data" not in payload:
        # _uw_get's uniform {} failure signature (see docstring) — NOT the
        # same as a genuine {"data": []} empty-week response.
        logger.debug("econ_calendar: UW call unavailable this cycle (no 'data' key)")
        return None, False

    raw_events = payload.get("data")
    if not isinstance(raw_events, list):
        logger.error(f"UW economic-calendar returned unexpected type: {type(raw_events)}")
        return None, False

    events = []
    cutoff = datetime.now(_NY) + timedelta(days=days_ahead)
    for raw in raw_events:
        if not isinstance(raw, dict):
            continue
        ev = _normalize_event(raw)
        if ev is None:
            continue
        # Filter to days_ahead window for caller contract parity
        try:
            ev_dt = datetime.fromisoformat(ev["datetime_et"])
            if ev_dt.tzinfo is None:
                ev_dt = ev_dt.replace(tzinfo=_NY)
            if ev_dt > cutoff:
                continue
        except (ValueError, TypeError):
            # S190 L3: silent-skip would hide UW schema drift. Log at DEBUG
            # (not WARNING — single-event parse failure is benign; multiple
            # in a row will show up in DEBUG-level digests if operator
            # bumps log level for investigation).
            logger.debug("econ_calendar: unparseable datetime_et %r on event %r",
                         ev.get("datetime_et"), ev.get("name", "?"))
        events.append(ev)

    logger.info(f"Calendar fetch: {len(events)} events for next {days_ahead} days (UW)")
    return events, True


async def fetch_calendar(days_ahead: int = 7) -> list[dict]:
    """Call UW /market/economic-calendar. Returns normalized event list, []
    on any failure. Back-compat wrapper around _fetch_uw_calendar_raw() —
    UW-only, no FRED/FOMC fallback. Prefer fetch_calendar_with_fallback() for
    the scheduled refresh job (STEP2, 2026-09-25); this function is kept for
    any caller that specifically wants UW-only behavior.

    `days_ahead` parameter is retained for API stability with prior TE caller
    contract; UW returns its own forward window (typically ~7d) and we filter
    client-side.
    """
    events, ok = await _fetch_uw_calendar_raw(days_ahead)
    return events if ok and events is not None else []


async def fetch_fred_calendar(days_ahead: int = 7) -> list[dict] | None:
    """Call FRED /releases/dates and normalize the curated majors. Second
    source in the fallback chain (STEP2).

    Returns None if the fallback source itself didn't work this cycle (no
    api key, network/HTTP/parse failure) — distinct from a successful call
    that legitimately found zero curated-release dates in the window
    (returns [] in that case). Mirrors _fetch_uw_calendar_raw's None-vs-[]
    contract so fetch_calendar_with_fallback() can tell "FRED is down too"
    from "FRED says nothing major is scheduled".

    Budget: 1 count from the 'fred_api' budget (agent/rate_limiter.py) — a
    self-imposed courtesy cap; FRED's free tier is ~120 req/min and this
    fires ~1x/day under the existing daily refresh cadence (no new scheduled
    job added).
    """
    api_key = config.fred_api_key or ""
    if not api_key:
        logger.debug("FRED API key not configured, econ calendar fallback unavailable")
        return None

    try:
        await check_and_record("fred_api")
    except BudgetExhaustedError:
        logger.warning("FRED budget exhausted, skipping calendar fallback fetch")
        return None

    today = datetime.now(_NY).date()
    end = today + timedelta(days=days_ahead)

    try:
        import httpx
    except ImportError:
        logger.debug("httpx not available, FRED calendar fallback skipped")
        return None

    try:
        async with httpx.AsyncClient(timeout=_FRED_CALL_TIMEOUT_SECONDS, follow_redirects=True) as client:
            resp = await client.get(
                f"{_FRED_BASE_URL}{_FRED_RELEASES_DATES_PATH}",
                params={
                    "api_key": api_key,
                    "file_type": "json",
                    "realtime_start": today.isoformat(),
                    "realtime_end": end.isoformat(),
                    "include_release_dates_with_no_data": "true",
                    "limit": 1000,
                },
            )
    except Exception as e:
        # Avoid exc_info=True per S118 W-075 (api_key rides in the query string)
        logger.error(f"FRED econ-calendar fallback fetch failed: {type(e).__name__}: {str(e).split('?')[0][:200]}")
        return None

    if resp.status_code != 200:
        logger.warning(f"FRED econ-calendar fallback HTTP {resp.status_code}")
        return None

    try:
        payload = resp.json()
    except Exception as e:
        logger.warning(f"FRED econ-calendar fallback: bad JSON: {type(e).__name__}")
        return None

    raw_dates = payload.get("release_dates") if isinstance(payload, dict) else None
    if not isinstance(raw_dates, list):
        logger.warning(f"FRED econ-calendar fallback: unexpected 'release_dates' shape {type(raw_dates).__name__}")
        return None

    # Observability only (R-14): record the raw (pre-guard) release days per
    # curated release id so an artefact like a week of daily rows is visible.
    release_days: dict[int, set[date]] = {}
    for item in raw_dates:
        if not isinstance(item, dict):
            continue
        try:
            rid = int(item.get("release_id"))
            rday = date.fromisoformat(str(item.get("date")))
        except (TypeError, ValueError):
            continue
        if rid in _FRED_RELEASE_TABLE:
            release_days.setdefault(rid, set()).add(rday)
    for rid, days in sorted(release_days.items()):
        run = _longest_consecutive_run(days)
        if run > 3:
            logger.info(
                f"FRED release {rid} ({_FRED_RELEASE_TABLE[rid]['name']}) has {run} consecutive "
                f"release days in the window; rows are not dropped by this check"
            )

    events = []
    cutoff = datetime.now(_NY) + timedelta(days=days_ahead)
    for item in raw_dates:
        ev = _normalize_fred_release(item)
        if ev is None:
            continue
        try:
            ev_dt = datetime.fromisoformat(ev["datetime_et"])
        except (ValueError, TypeError):
            continue
        if ev_dt > cutoff:
            continue
        events.append(ev)

    logger.info(f"FRED calendar fallback: {len(events)} curated-release dates for next {days_ahead} days")
    return events


def fetch_fomc_static(days_ahead: int = 7) -> list[dict]:
    """Hardcoded FOMC decision-day dates — third source in the fallback chain
    (STEP2). Pure, no network, never fails; always merged in regardless of
    whether UW/FRED succeeded (see fetch_calendar_with_fallback()). See
    _FOMC_DECISION_DATES_ET for verification provenance/date range.
    """
    now = datetime.now(_NY)
    cutoff = now + timedelta(days=days_ahead)
    floor = now - timedelta(days=1)
    events = []
    for date_str in _FOMC_DECISION_DATES_ET:
        try:
            d = date.fromisoformat(date_str)
        except ValueError:
            continue
        dt = datetime(d.year, d.month, d.day, 14, 0, tzinfo=_NY)
        if dt > cutoff or dt < floor:
            continue
        events.append({
            "name": "FOMC Rate Decision / Statement",
            "category": "report",
            "datetime_et": dt.isoformat(),
            "importance": 3,
            "forecast": "",
            "previous": "",
            "actual": "",
            "country": "United States",
            "reported_period": "",
            "source": "fomc_static",
        })
    return events


async def fetch_calendar_with_fallback(days_ahead: int = 7) -> tuple[list[dict], bool]:
    """Orchestrates UW -> FRED -> FOMC-static (STEP2). Returns (events, source_ok).

    FRED is tried when UW "fails or returns nothing" (UW unavailable this
    cycle, OR a genuine-but-empty UW response) — literally STEP2's trigger
    condition. The static FOMC list is always merged in last, regardless of
    whether UW/FRED worked.

    source_ok is True iff at least one LIVE source (UW or FRED) produced a
    genuine response this cycle — even an empty one. It is False only when
    BOTH UW and FRED failed/were unavailable. The hardcoded FOMC list never
    flips source_ok True by itself: it can't fail (it's not a live call), so
    on its own it must not mask "both real sources are down" — run_calendar_
    refresh() uses source_ok to decide whether to refresh CALENDAR_CACHE_PATH
    (and therefore its mtime, the only signal agent/dpof/data.py's
    _econ_calendar_read_sync() can see — "missing is never zero").
    """
    uw_events, uw_ok = await _fetch_uw_calendar_raw(days_ahead)
    events = uw_events if uw_ok and uw_events is not None else []

    fred_ok = False
    if not uw_ok or not events:
        fred_events = await fetch_fred_calendar(days_ahead)
        if fred_events is not None:
            fred_ok = True
            events = _merge_events(events, fred_events)
        elif not uw_ok:
            logger.warning("econ_calendar: both UW and FRED fallback unavailable this cycle")

    fomc_events = fetch_fomc_static(days_ahead)
    events = _merge_events(events, fomc_events)

    return events, (uw_ok or fred_ok)


# ============================================================
# Cache Management
# ============================================================

def save_calendar(events: list[dict]) -> None:
    """Atomic write to calendar cache."""
    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
        data = {
            "fetched_at": datetime.now(_NY).isoformat(),
            "event_count": len(events),
            "events": events,
        }
        tmp = CALENDAR_CACHE_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, CALENDAR_CACHE_PATH)
        logger.info(f"Calendar cache saved: {len(events)} events")
    except Exception as e:
        logger.error(f"Failed to save calendar cache: {e}")


def load_calendar() -> list[dict]:
    """Read cached calendar. Returns [] if missing/corrupt."""
    if not os.path.isfile(CALENDAR_CACHE_PATH):
        return []
    try:
        with open(CALENDAR_CACHE_PATH, "r") as f:
            data = json.load(f)
        events = data.get("events", [])
        if not isinstance(events, list):
            return []

        # Warn if cache is stale
        fetched_str = data.get("fetched_at", "")
        if fetched_str:
            try:
                fetched = datetime.fromisoformat(fetched_str)
                if fetched.tzinfo is None:
                    fetched = fetched.replace(tzinfo=_NY)
                age_days = (datetime.now(_NY) - fetched).days
                if age_days > CALENDAR_MAX_AGE_DAYS:
                    logger.warning(f"Calendar cache is {age_days} days old (max {CALENDAR_MAX_AGE_DAYS})")
            except (ValueError, TypeError):
                pass

        return events
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load calendar cache: {e}")
        return []


def get_upcoming_events(hours: float = 24) -> list[dict]:
    """Return events in next N hours from cached calendar."""
    events = load_calendar()
    if not events:
        return []

    now = datetime.now(_NY)
    cutoff = now + timedelta(hours=hours)
    upcoming = []

    for ev in events:
        try:
            dt = datetime.fromisoformat(ev["datetime_et"])
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_NY)
            if now <= dt <= cutoff:
                upcoming.append(ev)
        except (ValueError, TypeError, KeyError):
            continue

    return upcoming


def get_todays_events() -> list[dict]:
    """Convenience: events with datetime_et matching today's date."""
    events = load_calendar()
    if not events:
        return []

    today = datetime.now(_NY).date()
    result = []

    for ev in events:
        try:
            dt = datetime.fromisoformat(ev["datetime_et"])
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_NY)
            if dt.date() == today:
                result.append(ev)
        except (ValueError, TypeError, KeyError):
            continue

    return result


# ============================================================
# Scheduled Task Callback
# ============================================================

async def run_calendar_refresh(context) -> None:
    """Scheduled task: fetch economic calendar from UW, with a FRED + hardcoded-
    FOMC fallback so a UW outage alone doesn't strand the cache (2026-09-25;
    see fetch_calendar_with_fallback()).

    S187 (2026-04-30) migrated this away from Trading Economics; S190 L1 refreshes
    the stale docstring. UW leg counts against the shared UW 5K/day budget
    (~7-8 calls/day); FRED fallback leg counts against its own small 'fred_api'
    budget, only spent when UW fails or returns nothing.

    Runs daily at 6:00 AM ET. Skips outside 5 AM - 10 PM ET window.
    """
    now = datetime.now(_NY)

    # Time guard: skip outside 5 AM - 10 PM ET
    if now.hour < 5 or now.hour >= 22:
        return

    logger.info("Starting economic calendar refresh...")

    try:
        events, source_ok = await fetch_calendar_with_fallback(days_ahead=7)

        if source_ok:
            # At least one LIVE source (UW or FRED) answered this cycle, even
            # if with zero curated events — save unconditionally so the cache
            # file's mtime advances (this is the only signal DPOF's
            # _econ_calendar_read_sync() health check reads). Only withholding
            # the save when BOTH live sources failed lets a genuinely stale
            # cache still age toward DPOF-unhealthy — "missing is never zero".
            save_calendar(events)

            used_fallback = any(ev.get("source") == "fred" for ev in events)

            # Notify via Telegram (fire-and-forget) — only when there's
            # something to report, same as before the fallback change.
            if events:
                try:
                    from agent.integrations.scheduled_tasks import _get_chat_ids, _safe_send
                    chat_ids = _get_chat_ids()
                    today_count = 0
                    for ev in events:
                        try:
                            if datetime.fromisoformat(ev.get("datetime_et", "")).date() == now.date():
                                today_count += 1
                        except (ValueError, TypeError):
                            pass
                    high_imp = sum(1 for ev in events if ev.get("importance", 0) >= 3)
                    fallback_note = " [UW down, used FRED fallback]" if used_fallback else ""
                    msg = (
                        f"Economic calendar updated: {len(events)} events this week "
                        f"({today_count} today, {high_imp} high-importance){fallback_note}"
                    )
                    for cid in chat_ids:
                        await _safe_send(context.bot, cid, text=msg)
                except Exception as e:
                    logger.debug(f"Fire-and-forget calendar notification failed: {e}")

            logger.info(
                f"Calendar refresh complete: {len(events)} events"
                + (" (FRED fallback used)" if used_fallback else "")
            )
        else:
            # Both UW and FRED unavailable this cycle — keep cached data,
            # exactly the pre-fallback behavior on any failure.
            cached = load_calendar()
            if cached:
                logger.warning(f"Calendar refresh: UW and FRED both unavailable, keeping {len(cached)} cached")
            else:
                logger.warning("Calendar refresh: UW and FRED both unavailable, no cache available")

    except BudgetExhaustedError:
        logger.warning("Calendar budget exhausted, skipping calendar refresh")
    except Exception as e:
        # S118 W-075: Avoid exc_info=True — stack trace may expose API keys in URL
        logger.error(f"Calendar refresh failed: {type(e).__name__}: {str(e).split('?')[0]}")
