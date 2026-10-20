"""
newsflash_lite — CLIO's own breaking-news detector/classifier (contract "N", INTEL_V3_SPEC.md).

ADVISORY ONLY. This module NEVER places, sizes, cancels or modifies an order and NEVER
imports anything from agent.dpof.broker / agent.dpof.orders / agent.dpof.engine. It only
ever writes: (1) a local JSON snapshot (data/intelligence/newsflash_state.json) that
agent/dpof/data.py or any other reader consumes via get_breaking_state(), and (2) an
optional plain-text Telegram alert for high-severity events. The flow signal only
NOMINATES context for the decision model (Opus) + Jev — it is never a mechanical trade
trigger. (Grep-provable: no "agent.dpof.broker"/"agent.dpof.orders"/"agent.dpof.engine"
imports, no order/position/EMERGENCY_STOP/atlas_team tokens anywhere in this file — see
tests/test_newsflash_lite_advisory.py::test_no_trading_or_order_code.)

Design credit: the tier/dedupe *methodology* (not the code) is ported from the operator's
own Hydrinia repo, agent-owned `/home/user/hydrinia/newsflash_engine.py` (NewsFlashEngine's
classify_headline / MD5-hash+time-window dedup) and `flow_confirm_engine.py` (concept only:
a lightweight multi-source-corroboration confidence bump, no actual UW-confluence call in
v1 — see NEWSFLASH_PORT.md "optional v2"). Unlike NewsFlashEngine, this module NEVER builds
or calls anything resembling `_execute_portfolio_actions` — see NEWSFLASH_PORT.md §3 "What
should NOT be ported". The advisory-only write-path discipline (grep-provable, no
risk/order/position write) mirrors agent/integrations/social_intel_divergence.py.

Sources (each optional; a failed/absent source never blocks the others — see
_refresh_pass()):
  - UW news headlines (`/news/headlines`, verified against docs/api/uw.txt +
    scratchpad uw_openapi.json "Headline News" schema) via the existing
    `agent.integrations.uw_flow_intel._uw_get()` helper — shares the existing
    "unusual_whales" BudgetTracker budget (mirrors agent/integrations/econ_calendar.py's
    `_fetch_uw_calendar_raw()`: `check_and_record("unusual_whales")` then
    `_uw_get(path, skip_moltbot_budget=True)`).
  - Polygon general market news (`/v2/reference/news`, direct httpx call with
    config.polygon_api_key — NOT via agent/integrations/polygon.py, which only exposes a
    pre-formatted/sanitized text string via `_get_news()`, unsuitable for structured
    re-parsing; NEWSFLASH_PORT.md explicitly sanctions "direct httpx with its key" as the
    alternative). Benzinga earnings/guidance/ratings endpoints exist on the same Polygon
    plan (agent/integrations/polygon.py:430-533) but are NOT wired in v1 — the general news
    feed's `tickers`/`title`/`description` are sufficient for the tier classifier and adding
    3 more polled endpoints would multiply request volume for marginal signal; flagged as a
    v2 candidate in the build report. If the plan lacks the news entitlement, the source
    self-disables after one 401/403 with a single INFO log (see `_polygon_entitlement_ok`).
  - CLIO's existing Lighthouse 55-feed RSS scan. This module NEVER calls
    `agent.integrations.global_news_feeds.scan_current_window()` itself — see "Lighthouse
    multi-consumer starvation fix" below — it only reads the shared, non-consuming
    `get_recent_headlines()` buffer that scan_current_window() (driven by
    market_events.py / learning_engine / DPOF's Jev refresher on their own cadences)
    populates as a side effect. Lighthouse pre-wraps title/summary in a
    sanitize_ingested_content() envelope; _unwrap_ingested() strips it back to raw text
    (this module stores RAW text for display per the build spec — see "Sanitization"
    below).
  - DDG search is intentionally NOT wired (NEWSFLASH_PORT.md §3: CLIO already has
    DDG-based news_intel.py/news_update.py; a third DDG consumer would be redundant spend
    for the same signal). UW WebSocket is intentionally NOT wired (v1 stays on polling; a
    persistent WS connection is a v2 candidate only if 60-120s polling proves too slow).

Lighthouse multi-consumer starvation fix (ORCH-7Q4K-DPOF follow-up, 2026-09-25; extended
follow-up #2, same day): `global_news_feeds.HeadlineManager._seen_keys` is a
process-global "not yet returned to ANY caller" filter shared by every
`scan_current_window()` caller (learning_engine, market_events.check_rss_breaking_news,
DPOF's Jev headline refresher) — even a caller that only wants the SIDE EFFECT (drive a
scan / populate the buffer) and ignores the return value still "steals" headlines,
because that return value is what marks them seen. Builder N owns
agent/integrations/global_news_feeds.py + agent/integrations/market_events.py for this
fix: global_news_feeds.py gained a non-consuming `get_recent_headlines()` /
`get_last_scan_time()` buffer (populated as a pure side effect of scan_current_window(),
whose own return value and behavior for existing callers are unchanged) PLUS a
side-effect-only driver, `refresh_recent_headlines()`, that fetches/filters identically
but never touches `_seen_keys` — see tests/test_global_news_feeds_buffer.py.
market_events.check_rss_breaking_news now calls `refresh_recent_headlines()` (not
scan_current_window()) to drive the scan, and builds its crisis cluster from
`get_recent_headlines()`. This module reads `get_recent_headlines()` exclusively (see
_fetch_lighthouse_events()) — it never calls scan_current_window() OR
refresh_recent_headlines() itself (it has no need to drive a scan; it just reads
whatever the market-hours driver already populated). (Orchestrator: DPOF data.py's Jev
refresher switches its own driver call to refresh_recent_headlines() next; not touched
here.)

Budgets: UW news reuses the existing shared "unusual_whales" BudgetTracker budget (already
accounted by every UW consumer — see agent/rate_limiter.py). Polygon news is registered as
its own BudgetTracker budget, "newsflash_polygon" (agent/config.py
`budget_newsflash_polygon`, default 800/day, env `BUDGET_NEWSFLASH_POLYGON`; agent/
rate_limiter.py `_register_budgets()`), checked via `check_and_record("newsflash_polygon")`
before each request, alongside the existing 60s min-interval gate
(_POLYGON_MIN_INTERVAL_S). (Earlier revision self-imposed a local daily counter instead,
since this builder's file scope didn't originally include config.py/rate_limiter.py; the
orchestrator has since expanded that scope specifically for this registration.)

Sanitization: per the build spec, this module stores RAW text (capped length) for display
and NEVER calls sanitize_ingested_content() — this module makes no LLM calls itself, so the
"only where text goes to an LLM" condition never applies here; any consumer that puts this
module's headline text into an LLM prompt (e.g. G's DPOF integration) is responsible for
sanitizing it itself, exactly mirroring agent/dpof/data.py's existing precedent ("jev.py is
the single sanitization point ... sanitizing here too would double-wrap the envelope").
"""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import httpx

from agent.config import config
from agent.rate_limiter import BudgetExhaustedError, check_and_record

logger = logging.getLogger("openclaw.newsflash_lite")

_NY = ZoneInfo("America/New_York")

# ============================================================
# State file (contract N — INTEL_V3_SPEC.md)
# ============================================================

_STATE_FILENAME = "newsflash_state.json"
_READ_CACHE: dict = {"mtime": None, "data": None}


def _state_path() -> str:
    override = os.environ.get("CLIO_NEWSFLASH_FILE")
    if override:
        return override
    # 3x dirname from agent/integrations/newsflash_lite.py -> repo root (matches
    # agent/integrations/social_intel_divergence.py's _DATA_DIR resolution).
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, "data", "intelligence", _STATE_FILENAME)


def _load_state() -> dict:
    path = _state_path()
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data.setdefault("events", [])
            data.setdefault("alerts", {"sent_ids": [], "timestamps": []})
            return data
    except FileNotFoundError:
        pass
    except Exception as e:  # noqa: BLE001 — advisory: a corrupt state file starts fresh, never crashes
        logger.warning("newsflash: state load failed, starting fresh: %s", e)
    return {"events": [], "active": False, "max_severity": 0, "as_of": None,
            "alerts": {"sent_ids": [], "timestamps": []}}


def _save_state_sync(state: dict) -> None:
    """Blocking I/O — must run off the event loop (see _refresh_pass's run_in_executor)."""
    path = _state_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp, path)
    except Exception as e:  # noqa: BLE001 — advisory: log, never raise into the caller
        logger.error("newsflash: state save failed: %s", e)


def _public_event(ev: dict) -> dict:
    """Contract shape: {id, headline, category, tier, severity, confidence, source, ts,
    tickers, direction_hint} plus a bonus `sources` list (corroborating source names)."""
    return {
        "id": ev.get("id"),
        "headline": ev.get("headline"),
        "category": ev.get("category"),
        "tier": ev.get("tier"),
        "severity": ev.get("severity"),
        "confidence": ev.get("confidence"),
        "source": ev.get("source"),
        "ts": ev.get("ts"),
        "tickers": ev.get("tickers") or [],
        "direction_hint": ev.get("direction_hint"),
        "sources": ev.get("sources") or [],
        "market_linked": ev.get("market_linked"),
    }


def get_breaking_state(max_age_seconds: int = 900) -> dict:
    """Sync, never raises, mtime-cached. `as_of` staleness (default 900s) is a JOB-LIVENESS
    signal ("has the background refresh run recently"), NOT a per-event TTL — individual
    events are evicted from the persisted list on their own _EVENT_RETENTION_SECONDS window
    each refresh cycle. active=False/events=[]/stale=True on any missing/stale/corrupt state
    ("missing is never zero")."""
    empty = {"active": False, "max_severity": 0, "events": [], "as_of": None, "stale": True}
    try:
        path = _state_path()
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return empty
        if _READ_CACHE.get("mtime") == mtime and _READ_CACHE.get("data") is not None:
            data = _READ_CACHE["data"]
        else:
            with open(path, "r") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return empty
            _READ_CACHE["mtime"] = mtime
            _READ_CACHE["data"] = data
        as_of = data.get("as_of")
        age = None
        if isinstance(as_of, str):
            try:
                dt = datetime.fromisoformat(as_of)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                age = (datetime.now(timezone.utc) - dt).total_seconds()
            except ValueError:
                age = None
        if age is None or age > max_age_seconds:
            return {"active": False, "max_severity": 0, "events": [], "as_of": as_of, "stale": True}
        raw_events = data.get("events")
        events = [_public_event(ev) for ev in raw_events if isinstance(ev, dict)] if isinstance(raw_events, list) else []
        return {"active": bool(data.get("active")), "max_severity": data.get("max_severity", 0) or 0,
                "events": events, "as_of": as_of, "stale": False}
    except Exception as e:  # noqa: BLE001 — reader contract: never raise
        logger.debug("newsflash: get_breaking_state failed: %s", e)
        return empty


# ============================================================
# Ingested-text unwrap (Lighthouse pre-wraps via sanitize_ingested_content)
# ============================================================

_INGESTED_WRAP_RE = re.compile(
    r'^<ingested_data[^>]*>\n.*?\n---\n(.*)\n---\n</ingested_data>$', re.DOTALL
)


def _unwrap_ingested(text: str) -> str:
    if not text:
        return ""
    m = _INGESTED_WRAP_RE.match(text.strip())
    return m.group(1) if m else text


# ============================================================
# Dedupe (normalized-title hash + fuzzy near-duplicate within a time window)
# ============================================================

_DEDUP_WINDOW_SECONDS = 45 * 60  # 45 min — within the spec's 30-60 min range
_FUZZY_THRESHOLD = 0.80  # empirically separates cross-source paraphrases (~0.81-0.92)
                          # from genuinely distinct headlines (~0.36-0.45) on real examples


def _normalize_title(title: str) -> str:
    t = title.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _title_hash(norm_title: str) -> str:
    return hashlib.md5(norm_title.encode()).hexdigest()[:16]


def _find_match(norm: str, h: str, events: list[dict], now_ts: float) -> dict | None:
    for ev in events:
        if now_ts - ev.get("last_seen_ts", 0) > _DEDUP_WINDOW_SECONDS:
            continue
        if ev.get("_hash") == h:
            return ev
        if difflib.SequenceMatcher(None, norm, ev.get("_norm", "")).ratio() >= _FUZZY_THRESHOLD:
            return ev
    return None


# ============================================================
# Classification — tiered regex patterns, SPY/QQQ macro-relevance tuned
#
# ORCH follow-up #2 items 2/3 (2026-09-25): the original pattern set used un-conjugated
# verb stems (`misses?` matches "miss"/"misses" but NOT "missed"/"missing"; `crashes?`
# matches "crashes" but NOT bare "crash"; etc.) which silently made several categories
# (earnings_shock especially — real headlines almost never use bare tickers like "NVDA",
# they say "Nvidia") near-unreachable, and left several common headline shapes (CPI/PPI
# "hotter/cooler than expected", named Fed-official rate commentary, BOJ/ECB/BOE/PBOC
# surprise moves, bare-index-name crashes with a magnitude cue, oil-jump-with-%, an
# attack on oil infrastructure) unclassified. Every verb below is conjugated with an
# explicit suffix group; `_pattern_matches()` additionally supports an AND-group (a
# tuple of independent regexes that must ALL match somewhere in the text, order-
# agnostic) for multi-condition rules like "central bank name AND surprise-word AND
# rate-move", which would otherwise need fragile unordered lookahead chains.
# ============================================================

def _c(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


def _pattern_matches(text: str, spec) -> bool:
    """spec is a compiled regex, a predicate callable(text) -> bool, or a tuple of
    those that must ALL match somewhere in text (independently, any order) — an
    AND-group."""
    if isinstance(spec, tuple):
        return all(_pattern_matches(text, p) for p in spec)
    if isinstance(spec, re.Pattern):
        return bool(spec.search(text))
    return bool(spec(text))


# Magnitude gates (orchestrator pass-2 probe): a bare "%"/"points" cue let routine
# moves through ("Nasdaq sinks 0.3%" -> Tier 1, "Oil climbs 1%" -> Tier 2). Moves are
# now thresholded numerically.
_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent\b)", re.IGNORECASE)
_POINTS_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*points?\b", re.IGNORECASE)
_INDEX_POINT_KEYS = (
    (re.compile(r"\bdow\b(?!\s+chemical)", re.IGNORECASE), "dow"),
    (re.compile(r"\bnasdaq\b", re.IGNORECASE), "nasdaq"),
    (re.compile(r"\b(?:s&p(?:\s*500)?|spx)\b", re.IGNORECASE), "spx"),
)


def _max_number(rx: re.Pattern, text: str) -> float | None:
    vals = []
    for m in rx.finditer(text):
        try:
            vals.append(float(m.group(1).replace(",", "")))
        except ValueError:
            continue
    return max(vals) if vals else None


def _index_move_at_least(pct_min: float, points_min: dict) -> "callable":
    """True when the headline states an index move of >= pct_min percent, or >= the
    per-index point threshold for an index it names (dow / nasdaq / spx)."""
    def _pred(text: str) -> bool:
        pct = _max_number(_PCT_RE, text)
        if pct is not None and pct >= pct_min:
            return True
        pts = _max_number(_POINTS_RE, text)
        if pts is None:
            return False
        return any(rx.search(text) and pts >= points_min[key] for rx, key in _INDEX_POINT_KEYS)
    return _pred


def _pct_at_least(pct_min: float) -> "callable":
    def _pred(text: str) -> bool:
        pct = _max_number(_PCT_RE, text)
        return pct is not None and pct >= pct_min
    return _pred


def _no_small_pct(pct_max_small: float) -> "callable":
    """False when the headline states a move and it is below pct_max_small (e.g.
    "oil surges 1%"); True when no percentage is given or it is large enough."""
    def _pred(text: str) -> bool:
        pct = _max_number(_PCT_RE, text)
        return pct is None or pct >= pct_max_small
    return _pred


_HISTORIC_YEAR_CRISIS_RE = re.compile(r"\b(?:19|20)\d{2}(?:s)?\s+(?:banking|financial|credit)\s+crisis\b", re.IGNORECASE)


def _not_historic_crisis(text: str) -> bool:
    return not _HISTORIC_YEAR_CRISIS_RE.search(text)


_CRASH_T1_MAGNITUDE = _index_move_at_least(3.0, {"dow": 1200, "nasdaq": 600, "spx": 150})
_SELLOFF_T2_MAGNITUDE = _index_move_at_least(1.5, {"dow": 600, "nasdaq": 300, "spx": 75})
_INDEX_NAME_RE = re.compile(r"\b(?:dow(?!\s+chemical)|nasdaq|s&p\s*500|s&p|spx)\b", re.IGNORECASE)
_INDEX_DOWN_VERB_RE = re.compile(
    r"\b(?:tumbl\w*|sinks?|sinking|sank|sunk|slid\w*|slumps?|slumped|slumping|plung\w*|tanks?|tanked|tanking|"
    r"falls?|fell|drops?|dropped|crash\w*|plummet\w*|collaps\w*|crater\w*)\b", re.IGNORECASE)


# Checked FIRST on every headline. If any of these match, Tier 1/2 classification is
# skipped for that headline (defense-in-depth false-positive guard — see
# tests/test_newsflash_lite.py::test_false_positive_guards / test_negative_controls).
_DEMOTION_PATTERNS = [
    _c(r"\bprice\s+war\b"), _c(r"\bbidding\s+war\b"), _c(r"\bwar\s+chest\b"),
    _c(r"\bwar\s+of\s+words\b"), _c(r"\bculture\s+war\b"), _c(r"\bturf\s+war\b"),
    _c(r"\bwar\s+on\s+(?:drugs|terror|poverty|crime)\b"),
    _c(r"\bcrash\s+test\b"), _c(r"\bcrash\s+bandicoot\b"), _c(r"\bcar\s+crash\b"),
    _c(r"\bcrash\s+course\b"),
    # retrospective / historical framing is never a live shock
    _c(r"\blessons?\s+(?:from|of)\b"), _c(r"\brevisit(?:s|ed|ing)?\b"), _c(r"\banniversary\b"),
    _c(r"\bdocumentary\b"), _c(r"\bretrospective\b"), _c(r"\blook(?:s|ing)?\s+back\b"),
]

# (category, patterns) — Tier 1: market-wide shock. severity=3. Each pattern is a
# compiled regex OR a tuple-of-regexes (AND-group — see _pattern_matches()).
_TIER1_PATTERNS: list[tuple[str, list]] = [
    ("fed_emergency", [
        _c(r"\b(?:fed|federal reserve|fomc)\b.{0,50}\b(?:emergency|unscheduled|inter-?meeting|surprise)\b.{0,50}\b(?:meeting|rate\s*(?:cuts?|hikes?|decisions?)|action|announcement)\b"),
        _c(r"\b(?:emergency|unscheduled|surprise)\b.{0,50}\b(?:fed|federal reserve|fomc)\b.{0,50}\b(?:meeting|rate\s*(?:cuts?|hikes?)|action)\b"),
    ]),
    ("war_attack", [
        _c(r"\bdeclar(?:es?|ed|ing)\s+war\b"),
        _c(r"\b(?:outbreak|start|eruption)\s+of\s+war\b"),
        _c(r"\b(?:go(?:es)?|went)\s+to\s+war\b"),
        _c(r"\bfull-?scale\s+(?:war|invasion)\b"),
        _c(r"\bworld\s+war\b"),
        _c(r"\bmilitary\s+(?:invasion|invad(?:es?|ed|ing)|offensive|strikes?|attacks?)\b"),
        _c(r"\b(?:missile|air|drone|nuclear)\s+strikes?\b"),
        _c(r"\btroops?\s+(?:cross(?:es|ed|ing)?|invad(?:es?|ed|ing)|storm(?:s|ed|ing)?)\b"),
        _c(r"\binvad(?:es?|ed|ing)\b.{0,40}\b(?:territory|border|nation|country)\b"),
        # attack on oil/energy infrastructure — market-wide-shock severity per the
        # 2019 Abqaiq-style precedent (ORCH follow-up #2 item 3).
        _c(r"\b(?:drone|missile|air|rocket)\s+(?:attack|strike)\b.{0,60}\b(?:oil|energy|refiner(?:y|ies)|pipeline|tanker)\b"),
        _c(r"\battack(?:s|ed|ing)?\b.{0,40}\b(?:oil\s+facilit(?:y|ies)|refiner(?:y|ies)|pipeline|tanker|strait\s+of\s+hormuz)\b"),
    ]),
    ("bank_failure", [
        _c(r"\bbank\s+(?:collaps(?:e[sd]?|ing)|failure|fail(?:s|ed|ing)?|seiz(?:es?|ed|ing)|(?:go(?:es)?|went)\s+under)\b"),
        _c(r"\brun\s+on\s+.{0,30}\bbank\b"),
        _c(r"\bbank\s+run\b"),
        _c(r"\bfdic\s+(?:seiz(?:es?|ed|ing)|(?:takes?|took)\s+over)\b"),
        # A bare "banking crisis" mention is NOT a concrete failure (it matched
        # "lessons from the 2008 banking crisis") -- it is Tier-2 credit_stress now.
    ]),
    ("exchange_halt", [
        _c(r"\b(?:nyse|nasdaq|cboe)\b.{0,40}\b(?:halt(?:s|ed|ing)?|suspend(?:s|ed|ing)?\s+trading)\b"),
        _c(r"\bcircuit\s+breakers?\s+(?:triggered|tripped|hit)\b"),
        _c(r"\btrading\s+halted\b.{0,40}\b(?:market-?wide|exchange-?wide)\b"),
        _c(r"\bmarket-?wide\s+trading\s+halt\b"),
    ]),
    ("sovereign_default", [
        _c(r"\bsovereign\s+default\b"),
        _c(r"\b(?:nation|country)\s+default(?:s|ed|ing)?\s+on\b"),
        _c(r"\bdebt\s+default\b"),
        _c(r"\bdefault(?:s|ed|ing)?\s+on\b.{0,30}\b(?:sovereign|national)\s+debt\b"),
    ]),
    ("pandemic", [
        _c(r"\bpandemic\s+declared\b"),
        _c(r"\bnew\s+pandemic\b"),
        _c(r"\bglobal\s+health\s+emergency\b"),
        _c(r"\bwho\s+declar(?:es?|ed|ing)\b.{0,30}\b(?:pandemic|public\s+health\s+emergency)\b"),
    ]),
    ("market_crash", [
        # inherently severe verbs (crash/plummet/collapse/crater) — magnitude-independent,
        # the word alone already implies a major move.
        _c(r"\b(?:dow(?!\s+chemical)|s&p\s*500|s&p500|spx|nasdaq|russell\s*2000|stock\s+market|wall\s+street|equities|global\s+markets?)\b.{0,40}\b(?:crash(?:es|ed|ing)?|plummet(?:s|ed|ing)?|collaps(?:e[sd]?|ing)|crater(?:s|ed|ing)?)\b"),
        _c(r"\b(?:crash(?:es|ed|ing)?|collaps(?:e[sd]?|ing)|plummet(?:s|ed|ing)?)\b.{0,40}\b(?:dow(?!\s+chemical)|s&p\s*500|nasdaq|stock\s+market|wall\s+street|major\s+ind(?:ex(?:es)?|ices))\b"),
        # milder/ambiguous-magnitude verbs (tumble/sink/slide/slump/plunge/tank — these
        # describe anything from a routine -0.5% dip to a real rout) — bare index name +
        # verb + an explicit magnitude cue (points or %) required, so "S&P 500 futures
        # slide" alone (no number) does NOT trigger Tier1. The negative lookahead still
        # excludes "Dow Chemical" (ORCH follow-up #2 item 3 + negative control).
        # Tier 1 needs a crash-sized move: >= 3% or Dow 1200 / Nasdaq 600 / S&P 150 pts.
        (_INDEX_NAME_RE, _INDEX_DOWN_VERB_RE, _CRASH_T1_MAGNITUDE),
    ]),
]

_MEGACAP_NAME_TO_TICKER = {
    "nvidia": "NVDA", "nvda": "NVDA",
    "apple": "AAPL", "aapl": "AAPL",
    "microsoft": "MSFT", "msft": "MSFT",
    "amazon": "AMZN", "amzn": "AMZN",
    "meta": "META", "facebook": "META",
    "alphabet": "GOOGL", "google": "GOOGL", "googl": "GOOGL",
    "tesla": "TSLA", "tsla": "TSLA",
    "broadcom": "AVGO", "avgo": "AVGO",
}
_MEGACAP_TICKERS = frozenset(_MEGACAP_NAME_TO_TICKER.values())
_MEGACAP_NAME_OR_TICKER_RE = _c(
    r"\b(?:nvidia|nvda|apple|aapl|microsoft|msft|amazon|amzn|meta|facebook|"
    r"alphabet|google|googl|tesla|tsla|broadcom|avgo)\b"
)
# ORCH follow-up #2 item 2: real headlines use company NAMES, not bare tickers — the
# original ticker-only pattern made earnings_shock effectively unreachable. Verb suffix
# forms fixed throughout (miss/missed/missing, crash/crashed/crashing, etc.).
_EARNINGS_VERB_RE = _c(
    r"\b(?:miss(?:es|ed|ing)?|beats?|"
    r"(?:cuts?|slash(?:es|ed|ing)?|lower(?:s|ed|ing)?|rais(?:es?|ed|ing)?)\s*.{0,25}\b(?:guidance|outlook|forecast)\b|"
    r"slash(?:es|ed|ing)?|warns?|plung(?:e[sd]?|ing)|crash(?:es|ed|ing)?|surg(?:e[sd]?|ing)|"
    r"soars?|craters?(?:ed|ing)?|tumbl(?:e[sd]?|ing)|slump(?:s|ed|ing)?|falls?|drops?)\b"
)
_BULLISH_EARNINGS_VERB_RE = _c(r"\b(?:beats?|surg(?:e[sd]?|ing)|soars?|rais(?:es?|ed|ing)?\s*.{0,25}\b(?:guidance|outlook|forecast)\b)\b")
_BEARISH_EARNINGS_VERB_RE = _c(
    r"\b(?:miss(?:es|ed|ing)?|"
    r"(?:cuts?|slash(?:es|ed|ing)?|lower(?:s|ed|ing)?)\s*.{0,25}\b(?:guidance|outlook|forecast)\b|"
    r"slash(?:es|ed|ing)?|warns?|plung(?:e[sd]?|ing)|crash(?:es|ed|ing)?|craters?(?:ed|ing)?|"
    r"tumbl(?:e[sd]?|ing)|slump(?:s|ed|ing)?|falls?|drops?)\b"
)


def _extract_megacap_tickers(text: str, api_tickers: list[str] | None) -> set[str]:
    found: set[str] = set()
    for m in _MEGACAP_NAME_OR_TICKER_RE.finditer(text):
        ticker = _MEGACAP_NAME_TO_TICKER.get(m.group(0).lower())
        if ticker:
            found.add(ticker)
    if isinstance(api_tickers, list):
        for t in api_tickers:
            if isinstance(t, str) and t.upper() in _MEGACAP_TICKERS:
                found.add(t.upper())
    return found


def _check_earnings_shock(text: str, api_tickers: list[str] | None) -> dict | None:
    """Dedicated check (not a plain regex list entry) because it must cross-reference
    api_tickers (ORCH follow-up #2 item 2: "also accept api-supplied tickers in the
    classification decision") in addition to company-name/ticker text mentions."""
    tickers = _extract_megacap_tickers(text, api_tickers)
    if not tickers or not _EARNINGS_VERB_RE.search(text):
        return None
    return {"category": "earnings_shock", "tier": 2, "severity": 2,
            "direction_hint": _direction_hint("earnings_shock", text),
            "tickers": sorted(tickers | set(_extract_tickers(text, api_tickers)))}


# (category, patterns) — Tier 2: macro/policy/geopolitical movers, big-cap shocks.
# severity=2. earnings_shock is handled by _check_earnings_shock() (see above), not a
# plain pattern list entry here.
_TIER2_PATTERNS: list[tuple[str, list]] = [
    # A sharp but not crash-sized index drop: >= 1.5% or Dow 600 / Nasdaq 300 / S&P 75 pts.
    ("market_selloff", [
        (_INDEX_NAME_RE, _INDEX_DOWN_VERB_RE, _SELLOFF_T2_MAGNITUDE),
    ]),
    # Live banking/credit stress without a concrete failure (never a dated/historic crisis).
    ("credit_stress", [
        (_c(r"\b(?:banking\s+(?:crisis|turmoil|contagion|panic)|bank(?:ing)?\s+stress|credit\s+crunch)\b"),
         _not_historic_crisis),
    ]),
    ("macro_policy", [
        _c(r"\b(?:fed|federal reserve|fomc)\b.{0,50}\b(?:hik(?:es?|ed|ing)|cuts?|rais(?:es?|ed|ing)|lower(?:s|ed|ing)?)\b.{0,30}\brates?\b"),
        _c(r"\brate\s+(?:cuts?|hikes?|increases?|decreases?)\b"),
        _c(r"\binflation\b.{0,40}\b(?:surg(?:es?|ed|ing)|spik(?:es?|ed|ing)|hotter\s+than\s+expected|cooler\s+than\s+expected)\b"),
        _c(r"\b(?:cpi|ppi|pce|non-?farm\s+payrolls|nfp|payrolls|gdp|jobs?\s+report)\b.{0,40}\b(?:miss(?:es|ed|ing)?|beats?|surprise\w*|shock\w*|hotter\s+than\s+expected|cooler\s+than\s+expected|stronger\s+than\s+expected|weaker\s+than\s+expected|comes?\s+in\s+(?:hot|cold|hotter|cooler))\b"),
        _c(r"\bhawkish\b"), _c(r"\bdovish\b"),
        # a named Fed official commenting on rate policy (e.g. "Fed's Waller says rate
        # cuts likely appropriate") — ORCH follow-up #2 item 3.
        _c(r"\bfed'?s\s+\w+\b.{0,60}\brate\s*(?:cuts?|hikes?|increases?|decreases?)\b"),
        _c(r"\b(?:fed|federal reserve)\b.{0,20}(?:chair|governor|president|official)\b.{0,60}\brates?\b"),
        # BOJ/ECB/BOE/PBOC SURPRISE/UNEXPECTED rate moves only — a routine scheduled
        # hold/decision falls through to the Tier-3 bare central-bank-name catch below
        # (ORCH follow-up #2 item 3: "routine scheduled holds -> T3").
        (
            _c(r"\b(?:bank\s+of\s+japan|boj|ecb|european\s+central\s+bank|bank\s+of\s+england|boe|pboc|people'?s\s+bank\s+of\s+china)\b"),
            _c(r"\b(?:unexpectedly|surprise\w*|shock\w*|in\s+an?\s+surprise)\b"),
            _c(r"\b(?:rais(?:es?|ed|ing)|hik(?:es?|ed|ing)|cuts?|cutting|lower(?:s|ed|ing)?)\b.{0,30}\b(?:rates?|interest\s+rates?)\b"),
        ),
    ]),
    ("geopolitical_mover", [
        _c(r"\btaiwan\s+strait\b.{0,40}\b(?:tensions?|escalat\w*|incursion)\b"),
        _c(r"\bsanctions?\b.{0,40}\b(?:new|imposed|escalat\w*|expanded|tightened)\b"),
        _c(r"\bceasefire\b.{0,40}\b(?:collaps(?:e[sd]?|ing)|breaks?\s+down|fails?)\b"),
        _c(r"\bceasefire\b.{0,40}\b(?:agreed|reached|signed|holds?)\b"),
        _c(r"\bembassy\b.{0,40}\b(?:attack(?:ed)?|stormed|bombed)\b"),
        _c(r"\bmartial\s+law\b.{0,40}\b(?:declared|imposed)\b"),
        _c(r"\bmilitary\s+coup\b"),
        _c(r"\bgovernment\s+(?:overthrown|toppled)\b"),
    ]),
    ("tariffs", [
        _c(r"\btariffs?\b.{0,50}\b(?:impose[sd]?|announc(?:es?|ed|ing)|threaten(?:s|ed|ing)?|rais(?:es?|ed|ing)|new)\b"),
        _c(r"\b(?:impose[sd]?|announc(?:es?|ed|ing)|threaten(?:s|ed|ing)?|rais(?:es?|ed|ing)|new)\b.{0,50}\btariffs?\b"),
        _c(r"\btariffs?\b.{0,50}\b(?:removed|rolled\s+back|eased|lifted|suspended|paused|rescinded)\b"),
        _c(r"\btrade\s+war\b"),
        _c(r"\bexport\s+controls?\b.{0,40}\b(?:new|expanded|tightened|imposed)\b"),
    ]),
    ("oil_shock", [
        (_c(r"\b(?:oil|crude|brent|wti)\b.{0,40}\b(?:spik(?:es?|ed|ing)|surg(?:es?|ed|ing)|soars?|plung(?:es?|ed|ing)|collaps(?:es?|ed|ing)|crater(?:s|ed|ing)?)\b"),
         _no_small_pct(3.0)),
        # jump/climb specifically require a magnitude cue (%) — these are milder verbs
        # than spike/surge/soar and would otherwise over-trigger on routine moves
        # (ORCH follow-up #2 item 3).
        (
            _c(r"\b(?:oil|crude|brent|wti)\b"),
            _c(r"\b(?:jump(?:s|ed|ing)?|climb(?:s|ed|ing)?)\b"),
            _pct_at_least(4.0),
        ),
        _c(r"\bopec\+?\b.{0,40}\b(?:surprise\s+cut|emergency\s+meeting|cuts?\s+production|production\s+cut)\b"),
        _c(r"\bstrait\s+of\s+hormuz\b.{0,40}\b(?:closed|blocked|threat(?:ened)?)\b"),
    ]),
]

# Tier 3 fallback: broad SPY/QQQ macro relevance keywords (multi-word/scoped to avoid the
# "bare word ambushes plain English" trap — e.g. NOT bare "fed", which would match "I fed
# the dog"). severity=1.
_TIER3_PATTERNS: list[re.Pattern] = [
    _c(r"\bfederal\s+reserve\b"), _c(r"\bfomc\b"), _c(r"\bthe\s+fed\b"),
    _c(r"\bfed\s+(?:chair|officials?|policy|rate|meeting|minutes)\b"),
    _c(r"\binflation\b"), _c(r"\bunemployment\b"), _c(r"\bjobs?\s+report\b"), _c(r"\brecession\b"),
    _c(r"\bearnings\b"), _c(r"\bguidance\b"), _c(r"\btariffs?\b"), _c(r"\bopec\b"),
    _c(r"\bcrude\s+oil\b"), _c(r"\boil\s+prices?\b"), _c(r"\bgeopolitical\b"), _c(r"\btaiwan\b"),
    _c(r"\bsanctions?\b"), _c(r"\btreasury\s+yields?\b"), _c(r"\bbond\s+market\b"), _c(r"\bvix\b"),
    _c(r"\bs&p\s*500\b"), _c(r"\bnasdaq\b"), _c(r"\bdow\s+jones\b"),
    # ORCH follow-up #2 item 3 additions:
    _c(r"\bcpi\b"), _c(r"\bppi\b"), _c(r"\bpce\b"), _c(r"\bnfp\b"), _c(r"\bpayrolls?\b"),
    _c(r"\bdow\b"),
    _c(r"\bbank\s+of\s+japan\b"), _c(r"\bboj\b"), _c(r"\becb\b"), _c(r"\beuropean\s+central\s+bank\b"),
    _c(r"\bbank\s+of\s+england\b"), _c(r"\bboe\b"), _c(r"\bpboc\b"), _c(r"\bpeople'?s\s+bank\s+of\s+china\b"),
]

_WATCH_TICKERS = ("SPY", "QQQ", "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "GOOG", "TSLA")
_TICKER_RE = {t: re.compile(r"\b" + re.escape(t) + r"\b") for t in _WATCH_TICKERS}
# ORCH follow-up #2 item 5: an ALL-CAPS headline (some feeds format alerts this way)
# makes a case-sensitive ticker scan ambiguous — "SPY" the ticker becomes
# indistinguishable from "spy" the common word once everything is uppercased. Skip the
# free-text scan (api_tickers still apply) when the text is mostly uppercase.
_ALLCAPS_LETTER_MIN = 10
_ALLCAPS_RATIO_THRESHOLD = 0.60


def _is_mostly_uppercase(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if len(letters) < _ALLCAPS_LETTER_MIN:
        return False
    return (sum(1 for c in letters if c.isupper()) / len(letters)) > _ALLCAPS_RATIO_THRESHOLD


# Direction-hint helper regexes (checked against the raw match text, case-insensitive).
_DOVISH_RE = _c(r"\b(?:cuts?|cutting|lower(?:s|ed|ing)?|eas(?:es?|ed|ing)|dovish|stimulus)\b")
_HAWKISH_RE = _c(r"\b(?:hik(?:es?|ed|ing)|rais(?:es?|ed|ing)|tighten(?:s|ed|ing)?|hawkish)\b")
_DEESCALATION_RE = _c(r"\b(?:ceasefire\s+(?:agreed|reached|holds?)|tensions?\s+eas\w*|de-?escalat\w*|peace\s+deal)\b")
_TARIFF_EASE_RE = _c(r"\b(?:removed|rolled\s+back|eased|lifted|suspended|paused)\b")
_OIL_SPIKE_RE = _c(r"\b(?:spik(?:es?|ed|ing)|surg(?:es?|ed|ing)|soars?|jump(?:s|ed|ing)?|climb(?:s|ed|ing)?)\b")
_OIL_PLUNGE_RE = _c(r"\b(?:plung(?:es?|ed|ing)|collaps(?:es?|ed|ing)|crater(?:s|ed|ing)?)\b")

_RISK_OFF_ALWAYS = {"war_attack", "bank_failure", "exchange_halt", "sovereign_default", "pandemic", "market_crash",
                    "market_selloff", "credit_stress"}


def _direction_hint(category: str, text: str) -> str | None:
    if category in _RISK_OFF_ALWAYS:
        return "risk_off"
    if category in ("fed_emergency", "macro_policy"):
        d, h = _DOVISH_RE.search(text), _HAWKISH_RE.search(text)
        if d and not h:
            return "risk_on"
        if h and not d:
            return "risk_off"
        return "mixed"
    if category == "geopolitical_mover":
        return "risk_on" if _DEESCALATION_RE.search(text) else "risk_off"
    if category == "earnings_shock":
        b, r = _BULLISH_EARNINGS_VERB_RE.search(text), _BEARISH_EARNINGS_VERB_RE.search(text)
        if b and not r:
            return "risk_on"
        if r and not b:
            return "risk_off"
        return "mixed"
    if category == "tariffs":
        return "risk_on" if _TARIFF_EASE_RE.search(text) else "risk_off"
    if category == "oil_shock":
        if _OIL_SPIKE_RE.search(text) and not _OIL_PLUNGE_RE.search(text):
            return "risk_off"
        return "mixed"
    return "mixed"  # notable (tier 3) / unrecognized category


def _extract_tickers(raw_text: str, api_tickers: list[str] | None = None) -> list[str]:
    """Case-sensitive scan against the watchlist (avoids "meta"-the-word vs META-the-ticker
    style false positives) plus any API-supplied tickers already in the watchlist. Skips
    the free-text scan on mostly-uppercase input (see _is_mostly_uppercase) — api_tickers
    still apply regardless."""
    found: set[str] = set()
    if isinstance(api_tickers, list):
        for t in api_tickers:
            if isinstance(t, str) and t.upper() in _WATCH_TICKERS:
                found.add(t.upper())
    if not _is_mostly_uppercase(raw_text):
        for t, rx in _TICKER_RE.items():
            if rx.search(raw_text):
                found.add(t)
    return sorted(found)


# R-18 (S215): a headline is market-linked when it names a ticker or a market channel (oil, rates, dollar, ...).
# Advisory only: DPOF reads it for CLIO_DPOF_NEWSFLASH_REQUIRE_LINK; classification and thresholds do not use it.
_MARKET_LINK_RE = _c(
    r"\b(?:oil|crude|brent|wti|hormuz|gulf|saudi|opec|taiwan|semiconductors?|spx|s&p|nasdaq|fed|"
    r"yields?|treasur(?:y|ies)|tariffs?|dollar|gold)\b"
)


def classify_headline(title: str, summary: str = "", api_tickers: list[str] | None = None) -> dict | None:
    """Deterministic regex-only classifier (no LLM — see module docstring). Returns
    {category, tier, severity, direction_hint, tickers, market_linked} or None if the headline is not
    SPY/QQQ macro-relevant at all. market_linked (R-18) is True when a ticker or a market channel is named."""
    result = _classify_headline_core(title, summary, api_tickers)
    if result is None:
        return None
    text = f"{title} {summary}".strip()
    result["market_linked"] = bool(result.get("tickers")) or _MARKET_LINK_RE.search(text) is not None
    return result


def _classify_headline_core(title: str, summary: str = "", api_tickers: list[str] | None = None) -> dict | None:
    """Classification body behind classify_headline() (unchanged by R-18)."""
    text = f"{title} {summary}".strip()
    if not text:
        return None
    demoted = any(_pattern_matches(text, p) for p in _DEMOTION_PATTERNS)
    if not demoted:
        for category, patterns in _TIER1_PATTERNS:
            if any(_pattern_matches(text, p) for p in patterns):
                return {"category": category, "tier": 1, "severity": 3,
                        "direction_hint": _direction_hint(category, text),
                        "tickers": _extract_tickers(text, api_tickers)}
        earnings = _check_earnings_shock(text, api_tickers)
        if earnings is not None:
            return earnings
        for category, patterns in _TIER2_PATTERNS:
            if any(_pattern_matches(text, p) for p in patterns):
                return {"category": category, "tier": 2, "severity": 2,
                        "direction_hint": _direction_hint(category, text),
                        "tickers": _extract_tickers(text, api_tickers)}
    if any(_pattern_matches(text, p) for p in _TIER3_PATTERNS):
        return {"category": "notable", "tier": 3, "severity": 1,
                "direction_hint": "mixed", "tickers": _extract_tickers(text, api_tickers)}
    return None


# ============================================================
# Confidence (multi-source corroboration raises it; single low-authority source lowers it)
# ============================================================

_TIER_BASE_CONFIDENCE = {1: 0.75, 2: 0.55, 3: 0.32}
_AUTHORITY_BASE = {"unusual_whales": 0.95, "polygon_news": 0.90}
_LIGHTHOUSE_TIER_AUTHORITY = {
    "T1_VERIFIED": 0.90, "T2_SEMI_INDEPENDENT": 0.72, "T3_STATE_AFFILIATED": 0.48,
}
_CORROBORATION_STEP = 0.08
_CORROBORATION_CAP = 0.25
_SINGLE_LOW_AUTHORITY_THRESHOLD = 0.55
_SINGLE_LOW_AUTHORITY_PENALTY = 0.75


def _authority_for_item(item: dict) -> float:
    kind = item.get("source_kind")
    if kind == "lighthouse":
        return _LIGHTHOUSE_TIER_AUTHORITY.get(item.get("reliability_tier"), 0.6)
    return _AUTHORITY_BASE.get(kind, 0.6)


def _compute_confidence(tier: int, authorities: list[float]) -> float:
    base = _TIER_BASE_CONFIDENCE.get(tier, 0.3)
    if not authorities:
        return round(base * 0.6, 3)
    primary = max(authorities)
    conf = base * primary
    extra = len(authorities) - 1
    if extra > 0:
        conf += min(extra * _CORROBORATION_STEP, _CORROBORATION_CAP)
    elif primary <= _SINGLE_LOW_AUTHORITY_THRESHOLD:
        conf *= _SINGLE_LOW_AUTHORITY_PENALTY
    return round(min(max(conf, 0.0), 1.0), 3)


# ============================================================
# Ingest — UW news (shares the existing "unusual_whales" BudgetTracker budget)
# ============================================================

_UW_NEWS_PATH = "/news/headlines"
_UW_MIN_INTERVAL_S = 90.0  # within the spec's 60-120s UW cadence range
_uw_last_fetch_ts = 0.0


async def _fetch_uw_news() -> list[dict]:
    global _uw_last_fetch_ts
    if not config.uw_api_key:
        return []
    now_ts = time.time()
    if now_ts - _uw_last_fetch_ts < _UW_MIN_INTERVAL_S:
        return []
    try:
        await check_and_record("unusual_whales")
    except BudgetExhaustedError:
        logger.info("newsflash: UW budget exhausted, skipping this cycle")
        return []
    except Exception as e:  # noqa: BLE001 — fail-open on tracker error (mirrors uw_flow_intel._uw_get)
        logger.debug("newsflash: UW budget check failed (proceeding without enforcement): %s", e)
    _uw_last_fetch_ts = now_ts
    try:
        from agent.integrations.uw_flow_intel import _uw_get
        payload = await _uw_get(_UW_NEWS_PATH, {"limit": 50}, skip_moltbot_budget=True)
    except Exception as e:  # noqa: BLE001 — isolate this source's failure from the others
        logger.warning("newsflash: UW news fetch failed: %s", e)
        return []
    if not isinstance(payload, dict) or "data" not in payload:
        return []
    raw = payload.get("data")
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        headline = item.get("headline")
        if not headline:
            continue
        out.append({
            "title": str(headline)[:500], "summary": "",
            "source": "unusual_whales", "source_kind": "unusual_whales",
            "published": item.get("created_at"),
            "tickers": [t for t in (item.get("tickers") or []) if isinstance(t, str)],
        })
    return out


# ============================================================
# Ingest — Polygon general news (direct httpx; registered "newsflash_polygon" budget)
# ============================================================

_POLYGON_NEWS_URL = "https://api.polygon.io/v2/reference/news"
_POLYGON_MIN_INTERVAL_S = 60.0
_POLYGON_TIMEOUT_S = 10.0
_polygon_last_fetch_ts = 0.0
_polygon_entitlement_ok: bool | None = None  # None=untested, True/False=known


async def _fetch_polygon_news() -> list[dict]:
    global _polygon_last_fetch_ts, _polygon_entitlement_ok
    api_key = config.polygon_api_key
    if not api_key:
        return []
    if _polygon_entitlement_ok is False:
        return []
    now_ts = time.time()
    if now_ts - _polygon_last_fetch_ts < _POLYGON_MIN_INTERVAL_S:
        return []
    try:
        await check_and_record("newsflash_polygon")
    except BudgetExhaustedError:
        logger.info("newsflash: polygon budget exhausted, skipping this cycle")
        return []
    except Exception as e:  # noqa: BLE001 — fail-open on tracker error (mirrors uw_flow_intel._uw_get)
        logger.debug("newsflash: polygon budget check failed (proceeding without enforcement): %s", e)
    _polygon_last_fetch_ts = now_ts
    try:
        async with httpx.AsyncClient(timeout=_POLYGON_TIMEOUT_S, follow_redirects=True) as client:
            resp = await client.get(_POLYGON_NEWS_URL, params={
                "limit": 20, "order": "desc", "sort": "published_utc", "apiKey": api_key,
            })
    except Exception as e:  # noqa: BLE001 — isolate this source's failure from the others
        logger.debug("newsflash: polygon news fetch failed: %s", e)
        return []

    if resp.status_code in (401, 403):
        if _polygon_entitlement_ok is not False:
            logger.info("newsflash: Polygon news source disabled (HTTP %d) — check plan entitlement",
                        resp.status_code)
        _polygon_entitlement_ok = False
        return []
    if resp.status_code != 200:
        logger.debug("newsflash: polygon news HTTP %d", resp.status_code)
        return []
    _polygon_entitlement_ok = True

    try:
        payload = resp.json()
    except Exception as e:  # noqa: BLE001
        logger.debug("newsflash: polygon news JSON parse failed: %s", e)
        return []
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        return []
    out = []
    for r in results:
        if not isinstance(r, dict):
            continue
        title = r.get("title")
        if not title:
            continue
        publisher = r.get("publisher")
        pub_name = publisher.get("name") if isinstance(publisher, dict) else None
        out.append({
            "title": str(title)[:500], "summary": str(r.get("description") or "")[:500],
            "source": f"polygon_news:{pub_name}" if pub_name else "polygon_news",
            "source_kind": "polygon_news",
            "published": r.get("published_utc"),
            "tickers": [t for t in (r.get("tickers") or []) if isinstance(t, str)],
        })
    return out


# ============================================================
# Ingest — Lighthouse (reads the shared recent-headlines buffer; NEVER calls
# scan_current_window() itself — see module docstring "Lighthouse multi-consumer
# starvation fix". get_recent_headlines() is a cheap, sync, non-network, non-consuming
# read, so no executor/timeout wrapping or cadence throttle is needed here — that
# machinery only made sense for a live network fetch, which this module no longer does.)
# ============================================================

_LIGHTHOUSE_LOOKBACK_SECONDS = 3600  # how far back into the shared buffer to look each cycle


async def _fetch_lighthouse_events() -> list[dict]:
    try:
        from agent.integrations import global_news_feeds
        headlines = global_news_feeds.get_recent_headlines(max_age_seconds=_LIGHTHOUSE_LOOKBACK_SECONDS)
    except Exception as e:  # noqa: BLE001 — isolate this source's failure from the others
        logger.debug("newsflash: lighthouse buffer read failed: %s", e)
        return []
    if not isinstance(headlines, list):
        return []
    out = []
    for hl in headlines:
        title = getattr(hl, "title", None)
        if not title:
            continue
        published = getattr(hl, "published_utc", None)
        source_id = getattr(hl, "source_id", None) or "lighthouse"
        out.append({
            "title": _unwrap_ingested(title),
            "summary": _unwrap_ingested(getattr(hl, "summary", "") or ""),
            "source": f"lighthouse:{source_id}", "source_kind": "lighthouse",
            "reliability_tier": getattr(hl, "reliability_tier", None),
            "published": published.isoformat() if isinstance(published, datetime) else None,
            "tickers": [],
        })
    return out


# ============================================================
# refresh_newsflash — the scheduled job callback (contract N)
# ============================================================

_HEADLINE_MAX_CHARS = 300
_EVENT_RETENTION_SECONDS = 6 * 3600
_MAX_EVENTS = 60
_ALERT_RATE_LIMIT_PER_HOUR = 6

# ORCH follow-up #2 item 1b: classify_headline() + _find_match()'s difflib fuzzy scan
# against every retained event was measured at ~8ms/item — at buffered-headline volume
# (Lighthouse buffer can hold up to 1000 items, each surfaced again on every ~60s cycle
# for as long as it stays within _LIGHTHOUSE_LOOKBACK_SECONDS) that reprocessed the SAME
# headline dozens of times before it aged out. _seen_item_cache remembers each distinct
# headline's hash -> the event id it resolved to (or None if it was irrelevant), so a
# repeat sighting within _SEEN_ITEM_TTL_SECONDS only does an O(1) liveness-refresh, never
# a full re-classify + re-fuzzy-scan. Module memory (not persisted) is sufficient per the
# orchestrator's instruction ("TTL ~2h, persisted in state or module memory") — a restart
# just costs one reprocessing burst, not a correctness issue.
_SEEN_ITEM_TTL_SECONDS = 2 * 3600
_seen_item_cache: dict[str, dict] = {}  # hash -> {"event_id": str | None, "expires_at": float}

# Exported for the orchestrator's scheduler registration (agent/integrations/
# scheduled_tasks.py, NOT edited by this builder). JOB_INTERVAL_SECONDS matches the
# spec's "~60s" cadence; per-source min-interval gates (UW 90s, Polygon 60s) are
# independently enforced inside this module regardless of how often the job itself
# fires. JOB_TIMEOUT_SECONDS bounds a single refresh_newsflash() call — generous
# relative to typical per-source timeouts (UW ~15s via _uw_get, Polygon 10s, Lighthouse
# is now a cheap in-memory read) so a single slow HTTP call can't stall the job queue.
JOB_INTERVAL_SECONDS = 60
JOB_TIMEOUT_SECONDS = 45

_run_lock = asyncio.Lock()


def _coerce_ts(candidate, now_iso: str) -> str:
    if isinstance(candidate, str):
        try:
            dt = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except ValueError:
            pass
    return now_iso


async def _send_alert(context, ev: dict) -> bool:
    """Returns True iff this event should be marked alerted/handled (Telegram sent OR
    intentionally log-only-suppressed). False means NOT handled (e.g. no bot context yet)
    — the caller must not mark it sent, so a later cycle with a real context can retry."""
    if os.environ.get("CLIO_NEWSFLASH_TELEGRAM_LOG_ONLY", "0") in ("1", "true", "yes"):
        logger.info("NEWSFLASH [LOG_ONLY] tier-%d %s: %s", ev["tier"], ev["category"], ev["headline"][:150])
        return True
    if context is None or getattr(context, "bot", None) is None:
        return False
    try:
        from agent.integrations.scheduled_tasks import _get_chat_ids, _safe_send
        ticker_str = f" | Tickers: {', '.join(ev['tickers'])}" if ev.get("tickers") else ""
        msg = (
            f"NEWSFLASH — Tier {ev['tier']} ({ev['category'].replace('_', ' ')})\n\n"
            f"{ev['headline'][:280]}\n\n"
            f"Sources: {ev['source']} | Confidence: {ev['confidence']:.0%} | "
            f"Direction: {ev.get('direction_hint') or 'unclear'}{ticker_str}\n\n"
            f"Advisory only — context for review, not a trade signal."
        )
        sent_any = False
        for cid in _get_chat_ids():
            ok = await _safe_send(context.bot, cid, text=msg)
            sent_any = sent_any or ok
        return sent_any
    except Exception as e:  # noqa: BLE001 — advisory: never fail the pass on a send error
        logger.warning("newsflash: alert send failed: %s", e)
        return False


def _merge_source_into_event(ev: dict, item: dict, classification: dict | None = None) -> None:
    if item["source"] not in ev["sources"]:
        ev["sources"].append(item["source"])
        ev.setdefault("authorities", []).append(_authority_for_item(item))
        ev["source"] = ",".join(ev["sources"])
        ev["confidence"] = _compute_confidence(ev["tier"], ev["authorities"])
    if classification:
        ev["tickers"] = sorted(set(ev.get("tickers") or []) | set(classification["tickers"]))


def _cleanup_seen_cache(now_ts: float) -> None:
    expired = [h for h, v in _seen_item_cache.items() if v.get("expires_at", 0) < now_ts]
    for h in expired:
        del _seen_item_cache[h]


def _process_ingest_sync(raw_items: list[dict], state: dict, now_ts: float, now_iso: str) -> tuple[dict, list[str]]:
    """CPU-bound classify+dedupe+merge, split out of _refresh_pass so it can run in a
    thread executor (ORCH follow-up #2 item 1a). Reads/writes the module-level
    _seen_item_cache (item 1b — classify each distinct headline once per TTL window, not
    every ~60s cycle). Pure function over its arguments + that cache; the single-flight
    _run_lock in refresh_newsflash() guarantees no concurrent _refresh_pass, so no extra
    locking is needed around the cache from this executor thread."""
    _cleanup_seen_cache(now_ts)
    events = [ev for ev in (state.get("events") or [])
              if now_ts - ev.get("last_seen_ts", 0) <= _EVENT_RETENTION_SECONDS]
    events_by_hash = {ev.get("_hash"): ev for ev in events if ev.get("_hash")}

    new_event_ids: list[str] = []
    for item in raw_items:
        title = (item.get("title") or "").strip()
        if not title:
            continue
        norm = _normalize_title(title)
        if not norm:
            continue
        h = _title_hash(norm)

        cached = _seen_item_cache.get(h)
        if cached is not None:
            # Seen within the TTL window already — skip the full classify + O(n)
            # fuzzy-match scan entirely, just refresh liveness/corroboration.
            cached["expires_at"] = now_ts + _SEEN_ITEM_TTL_SECONDS
            ev = events_by_hash.get(cached.get("event_id"))
            if ev is not None:
                _merge_source_into_event(ev, item)
                ev["last_seen_ts"] = now_ts
            continue

        match = _find_match(norm, h, events, now_ts)
        classification = classify_headline(title, item.get("summary", ""), item.get("tickers"))
        if match is None:
            if classification is None:
                # irrelevant to SPY/QQQ macro — discard, but remember so we don't
                # reclassify the exact same headline every cycle.
                _seen_item_cache[h] = {"event_id": None, "expires_at": now_ts + _SEEN_ITEM_TTL_SECONDS}
                continue
            authority = _authority_for_item(item)
            ev = {
                "id": h, "_hash": h, "_norm": norm,
                "headline": title[:_HEADLINE_MAX_CHARS],
                "category": classification["category"], "tier": classification["tier"],
                "severity": classification["severity"],
                "source": item["source"], "sources": [item["source"]], "authorities": [authority],
                "confidence": _compute_confidence(classification["tier"], [authority]),
                "ts": _coerce_ts(item.get("published"), now_iso),
                "last_seen_ts": now_ts,
                "tickers": classification["tickers"],
                "direction_hint": classification["direction_hint"],
                "market_linked": classification.get("market_linked"),
            }
            events.append(ev)
            events_by_hash[h] = ev
            new_event_ids.append(h)
            _seen_item_cache[h] = {"event_id": h, "expires_at": now_ts + _SEEN_ITEM_TTL_SECONDS}
        else:
            _merge_source_into_event(match, item, classification)
            match["last_seen_ts"] = now_ts
            _seen_item_cache[h] = {"event_id": match.get("_hash"), "expires_at": now_ts + _SEEN_ITEM_TTL_SECONDS}

    # Cap by severity-then-recency so a burst of Tier3 noise cannot crowd out Tier1/2 events.
    events.sort(key=lambda e: (-e.get("severity", 0), -e.get("last_seen_ts", 0)))
    events = events[:_MAX_EVENTS]

    max_severity = max((ev["severity"] for ev in events), default=0)
    active = any(ev["severity"] >= 2 for ev in events)

    state["events"] = events
    state["active"] = active
    state["max_severity"] = max_severity
    state["as_of"] = now_iso
    return state, new_event_ids


async def _refresh_pass(context) -> dict:
    loop = asyncio.get_running_loop()
    now_ts = time.time()
    now_iso = datetime.now(timezone.utc).isoformat()

    raw_items: list[dict] = []
    for fetch_fn, label in (
        (_fetch_uw_news, "unusual_whales"),
        (_fetch_polygon_news, "polygon"),
        (_fetch_lighthouse_events, "lighthouse"),
    ):
        try:
            raw_items.extend(await fetch_fn())
        except Exception as e:  # noqa: BLE001 — one source's failure never blocks the others
            logger.warning("newsflash: %s source failed: %s", label, e)

    state = await loop.run_in_executor(None, _load_state)

    # ORCH follow-up #2 item 1a: classify+dedupe is CPU-bound (~8ms/item measured;
    # difflib fuzzy match against every retained event) — at buffered-headline volume
    # this blocks the event loop for seconds, and wait_for() cannot preempt CPU work.
    # Run it in the thread executor.
    state, new_event_ids = await loop.run_in_executor(
        None, _process_ingest_sync, raw_items, state, now_ts, now_iso
    )
    events = state["events"]
    max_severity = state["max_severity"]
    active = state["active"]

    alert_state = state.setdefault("alerts", {"sent_ids": [], "timestamps": []})
    alert_state["timestamps"] = [t for t in alert_state.get("timestamps", []) if now_ts - t <= 3600.0]
    sent_ids = set(alert_state.get("sent_ids") or [])
    eligible = [
        ev for ev in events
        if ev["id"] not in sent_ids and (ev["tier"] == 1 or (ev["tier"] == 2 and len(ev.get("sources", [])) >= 2))
    ]

    sent_count = 0
    for ev in eligible:
        if len(alert_state["timestamps"]) >= _ALERT_RATE_LIMIT_PER_HOUR:
            break
        handled = await _send_alert(context, ev)
        if handled:
            alert_state.setdefault("sent_ids", []).append(ev["id"])
            alert_state["timestamps"].append(time.time())
            sent_count += 1
    alert_state["sent_ids"] = (alert_state.get("sent_ids") or [])[-200:]

    await loop.run_in_executor(None, _save_state_sync, state)

    result = {"status": "ok", "events": len(events), "new_events": len(new_event_ids),
              "active": active, "max_severity": max_severity, "alerts_sent": sent_count,
              "as_of": now_iso}
    logger.info("newsflash: events=%d new=%d active=%s max_severity=%d alerts_sent=%d",
                len(events), len(new_event_ids), active, max_severity, sent_count)
    return result


async def refresh_newsflash(context=None) -> dict | None:
    """PTB scheduled-job callback (contract N). Single-flight; never raises."""
    if os.environ.get("CLIO_NEWSFLASH_ENABLED", "1") not in ("1", "true", "yes"):
        return {"status": "disabled"}
    if _run_lock.locked():
        return {"status": "skipped_concurrent"}
    async with _run_lock:
        try:
            return await _refresh_pass(context)
        except Exception as e:  # noqa: BLE001 — scheduled context: report, never raise
            logger.error("newsflash: refresh pass failed: %s", e, exc_info=True)
            return {"status": "error", "error": str(e)[:300]}


if __name__ == "__main__":
    # CLI: one pass, print the resulting public state. No Telegram (context=None).
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(refresh_newsflash())
    print(json.dumps(get_breaking_state(), indent=2, default=str))
