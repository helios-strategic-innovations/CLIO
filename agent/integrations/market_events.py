"""
OpenClaw/Moltbot — Breaking News / Market Event Detection

Polls Polygon for SPY and VIX snapshots every 5 minutes during market hours.
When thresholds are exceeded, activates breaking news mode — the learning
engine suspends normal rotation and focuses on event-specific research.

Auto-expires after 4 learning cycles (1 hour) or when conditions normalize.
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

logger = logging.getLogger("openclaw.market_events")

_NY = ZoneInfo("America/New_York")

# ============================================================
# Thresholds (Moltbot-internal, not in .env)
# ============================================================

SPY_MOVE_PCT = 2.0          # Absolute SPY day change %
VIX_LEVEL = 30.0            # VIX level threshold
VIX_SPIKE_PTS = 5.0         # VIX day change in points
NEWS_MODE_CYCLES = 4        # Auto-expire after this many learning cycles
COOLDOWN_HOURS = 4          # Don't re-trigger within this window


# ============================================================
# Market Hours
# ============================================================

# NYSE full-close dates. Cross-checked against the published NYSE holiday rules
# (nth-weekday + weekend-observance) and Good Friday via the Gregorian computus —
# see the fix commit message. 2027 dates additionally cross-checked against the
# NYSE Group 2026/2027/2028 holiday-and-early-closings press release (ICE IR,
# Nov 2024) via WebSearch on 2026-09-24. Extend annually.
MARKET_HOLIDAYS = frozenset({
    # --- 2026 ---
    date(2026, 1, 1),    # New Year's Day
    date(2026, 1, 19),   # Martin Luther King Jr. Day (3rd Mon Jan)
    date(2026, 2, 16),   # Washington's Birthday / Presidents' Day (3rd Mon Feb)
    date(2026, 4, 3),    # Good Friday
    date(2026, 5, 25),   # Memorial Day (last Mon May)
    date(2026, 6, 19),   # Juneteenth
    date(2026, 7, 3),    # Independence Day (observed; Jul 4 is a Saturday)
    date(2026, 9, 7),    # Labor Day (1st Mon Sep)
    date(2026, 11, 26),  # Thanksgiving (4th Thu Nov)
    date(2026, 12, 25),  # Christmas Day
    # --- 2027 ---
    date(2027, 1, 1),    # New Year's Day (Friday)
    date(2027, 1, 18),   # Martin Luther King Jr. Day (3rd Mon Jan)
    date(2027, 2, 15),   # Washington's Birthday / Presidents' Day (3rd Mon Feb)
    date(2027, 3, 26),   # Good Friday
    date(2027, 5, 31),   # Memorial Day (last Mon May)
    date(2027, 6, 18),   # Juneteenth (observed; Jun 19 is a Saturday)
    date(2027, 7, 5),    # Independence Day (observed; Jul 4 is a Sunday)
    date(2027, 9, 6),    # Labor Day (1st Mon Sep)
    date(2027, 11, 25),  # Thanksgiving (4th Thu Nov)
    date(2027, 12, 24),  # Christmas Day (observed; Dec 25 is a Saturday)
})
# Half-days — NYSE closes early at 13:00 ET.
MARKET_EARLY_CLOSE = frozenset({
    date(2026, 11, 27),  # Day after Thanksgiving
    date(2026, 12, 24),  # Christmas Eve
    date(2027, 11, 26),  # Day after Thanksgiving
    # 2027 has no other scheduled early close per the NYSE Group press release
    # above (no Jul 2 or Dec 23 early close that year — confirmed, not assumed).
})


def _is_market_hours() -> bool:
    """Check if current time is within US market hours (9:30-16:00 ET, weekdays),
    honoring the NYSE holiday calendar (full closes) and early-close half-days
    (13:00 ET). See MARKET_HOLIDAYS / MARKET_EARLY_CLOSE."""
    now = datetime.now(_NY)
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    today = now.date()
    if today in MARKET_HOLIDAYS:
        return False
    close_hour = 13 if today in MARKET_EARLY_CLOSE else 16
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=close_hour, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close


# ============================================================
# Event Query Generation
# ============================================================

def _generate_event_queries(spy_pct: float, vix_level: float, vix_change: float) -> list[str]:
    """Generate targeted search queries based on the market event type."""
    now = datetime.now(_NY)
    date_str = now.strftime("%B %d %Y")
    queries = []

    if spy_pct <= -SPY_MOVE_PCT:
        queries.extend([
            f"stock market selloff {date_str} causes analysis",
            "market crash hedging strategies portfolio protection",
            "sector rotation during market decline historical analysis",
            "S&P 500 drawdown recovery historical patterns",
        ])
    elif spy_pct >= SPY_MOVE_PCT:
        queries.extend([
            f"stock market rally {date_str} drivers analysis",
            "market rally sustainability momentum analysis risks",
            "momentum trading strategies strong market conditions",
        ])

    if vix_level >= VIX_LEVEL:
        queries.extend([
            "VIX above 30 historical analysis trading outcomes",
            "high volatility options strategies portfolio protection",
            "volatility regime change portfolio adjustment strategies",
        ])

    if vix_change >= VIX_SPIKE_PTS:
        queries.extend([
            "VIX spike analysis market fear indicator historical",
            "volatility explosion options hedging strategies",
        ])

    # If no specific threshold matched (shouldn't happen), generic event query
    if not queries:
        queries.append(f"stock market major move {date_str} analysis")

    # Deduplicate while preserving order, cap at 5
    seen = set()
    unique = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    return unique[:5]


# ============================================================
# Operation Lighthouse v2.0 — RSS Breaking News Detection (CC-LH-3)
# ============================================================

# S148 H-1: Feature flags moved to runtime os.environ.get() at call site

_CRISIS_KEYWORDS = {
    "war", "invasion", "attack", "strike", "bombed", "bombing", "missile strike",
    "carrier struck", "declaration of war", "nuclear", "strait closed", "blockade",
    "ceasefire collapse", "martial law", "coup", "emergency summit", "oil embargo",
    "sanctions emergency", "currency crisis", "bank run", "default",
}

_RSS_BREAKING_T1_THRESHOLD = 2
_RSS_BREAKING_CLUSTER_WINDOW = 3600


def check_rss_breaking_news() -> dict | None:
    """Check RSS headlines for breaking geopolitical news warranting event mode.

    Requires 2+ T1 sources reporting crisis keywords within 1 hour.
    Returns event dict or None. CC-LH-3.
    """
    if os.environ.get("CLIO_RSS_BREAKING_ENABLED", "1") != "1":
        return None
    # S148-B W2-2: defense-in-depth — skip outside market hours (stale data on weekends)
    if not _is_market_hours():
        return None

    try:
        from agent.integrations.global_news_feeds import refresh_recent_headlines, get_recent_headlines
    except ImportError:
        return None

    # S233 CR-2: the underlying feed fetch uses sync httpx.Client.
    # This function is sync, called from async check_market_events via _with_timeout.
    # The timeout wrapper prevents indefinite blocking.
    #
    # ORCH follow-up "Lighthouse headline starvation" + follow-up #2 item 4 ("residual
    # starvation for learning_engine"): this used to call scan_current_window() purely
    # to drive the scan (its RETURN VALUE was already unused here — see
    # get_recent_headlines() below). But scan_current_window()'s return value is
    # computed via HeadlineManager.process(), which marks every item "seen" in
    # _seen_keys regardless of whether the caller looks at what came back — so even
    # driving-only calls silently stole headlines from scan_current_window()'s one
    # remaining return-value consumer, learning_engine. refresh_recent_headlines() is a
    # side-effect-only driver: it fetches the same feeds and applies the same
    # age/relevance filtering, but NEVER touches _seen_keys, so learning_engine's own
    # scan_current_window() call still sees every headline afterwards. Reading the
    # shared buffer via get_recent_headlines() (rather than any driver's return value)
    # is what makes headlines "consumed" by any OTHER caller still visible here too.
    refresh_recent_headlines()
    headlines = get_recent_headlines(max_age_seconds=_RSS_BREAKING_CLUSTER_WINDOW)
    if not headlines:
        return None

    now_utc = datetime.now(ZoneInfo("UTC"))
    cutoff = now_utc - timedelta(seconds=_RSS_BREAKING_CLUSTER_WINDOW)

    crisis = []
    for hl in headlines:
        if hl.published_utc < cutoff or hl.reliability_tier != "T1_VERIFIED":
            continue
        combined = f"{hl.title} {hl.summary or ''}".lower()
        kw_matches = [kw for kw in _CRISIS_KEYWORDS if re.search(r'\b' + re.escape(kw) + r'\b', combined)]
        if kw_matches:
            crisis.append({"headline": hl, "keywords": kw_matches})

    if len(crisis) < _RSS_BREAKING_T1_THRESHOLD:
        return None

    sources = sorted({c["headline"].source_name for c in crisis})
    keywords = sorted({kw for c in crisis for kw in c["keywords"]})
    regions = sorted({c["headline"].source_region for c in crisis})

    if os.environ.get("CLIO_RSS_BREAKING_LOG_ONLY", "1") == "1":
        logger.info(
            "LIGHTHOUSE BREAKING [LOG_ONLY] %d T1 crisis sources: %s — %s",
            len(crisis), ", ".join(sources), ", ".join(keywords),
        )
        return None

    logger.warning(
        "LIGHTHOUSE BREAKING: %d T1 crisis sources: %s — %s — activating event mode",
        len(crisis), ", ".join(sources), ", ".join(keywords),
    )
    return {
        "trigger": "rss_breaking_news",
        "sources": sources,
        "keywords": keywords,
        "regions": regions,
        "headline_count": len(crisis),
        "sample_title": crisis[0]["headline"].title[:120],
    }


# ============================================================
# Snapshot Fetching
# ============================================================

async def _fetch_snapshot(symbol: str) -> dict | None:
    """
    Fetch raw snapshot from Polygon for a single symbol.
    Returns parsed dict with key fields, or None on error.
    """
    try:
        from agent.integrations.market_snapshot import get_snapshot_raw
        return await get_snapshot_raw(symbol)
    except Exception as e:
        # SEC-1 (S211): str(e) of an httpx error embeds the request URL, and Polygon auth is a
        # query parameter -- always scrub. POLY-1: the 403 breaker already logged once per window.
        from agent.log_scrub import scrub_secrets
        if type(e).__name__ == "PolygonNotEntitled":
            logger.debug("Market snapshot skipped for %s (Polygon not entitled; breaker open)", symbol)
        else:
            logger.warning(f"Market snapshot failed for {symbol}: {type(e).__name__}: {scrub_secrets(str(e))}")
        return None


# ============================================================
# Scheduled Job
# ============================================================

async def check_market_events(context) -> None:
    """
    Scheduled job: check for major market events via Polygon snapshots.
    Runs every 5 minutes. Only active during market hours.
    """
    if not _is_market_hours():
        return

    try:
        from agent.integrations.learning_engine import (
            _load_learning_state,
            _save_learning_state,
        )

        # S233: flock-based I/O must not run on event loop thread (same pattern as HI-5)
        _loop = asyncio.get_running_loop()
        state = await _loop.run_in_executor(None, _load_learning_state)
        breaking = state.get("breaking_news", {})

        # Already in breaking news mode — let the learning engine handle it
        if breaking.get("active"):
            return

        # Cooldown check
        cooldown_until = breaking.get("cooldown_until", "")
        if cooldown_until:
            try:
                cooldown_dt = datetime.fromisoformat(cooldown_until)
                if cooldown_dt.tzinfo is None:
                    cooldown_dt = cooldown_dt.replace(tzinfo=_NY)
                if datetime.now(_NY) < cooldown_dt:
                    return
            except (ValueError, TypeError):
                pass

        # --- Lighthouse v2.0: RSS pre-check (runs even without Polygon) ---
        _rss_precheck_event = None
        try:
            loop = asyncio.get_running_loop()
            _rss_precheck_event = await loop.run_in_executor(None, check_rss_breaking_news)
        except Exception as e:
            logger.debug("Lighthouse RSS pre-check failed: %s", e)

        # Fetch SPY and VIX snapshots
        spy = await _fetch_snapshot("SPY")
        vix = await _fetch_snapshot("VIX")

        if not spy:
            # If RSS detected a crisis, proceed without SPY data
            if _rss_precheck_event:
                spy_pct = 0
            else:
                logger.debug("Market events: SPY snapshot unavailable, skipping check")
                return
        else:
            spy_pct = spy.get("change_pct")
            if spy_pct is None:
                spy_pct = 0

        # VIX handling — may fail (Polygon sometimes returns empty for VIX)
        vix_level = 0.0
        vix_change = 0.0
        if vix:
            vix_level = vix.get("last_price")
            if vix_level is None:
                vix_level = vix.get("day_close")
            if vix_level is None:
                vix_level = 0
            vix_change = vix.get("change")
            if vix_change is None:
                vix_change = 0

        # Threshold evaluation
        is_event = (
            abs(spy_pct) >= SPY_MOVE_PCT
            or vix_level >= VIX_LEVEL
            or vix_change >= VIX_SPIKE_PTS
        )

        # --- CC-CLIO-UW-4: UW REGIME-BASED TRIGGERS v2.0 (7 triggers) ---
        event_type = None  # Set by UW triggers or SPY/VIX logic below
        _uw_events_enabled = os.environ.get("CLIO_UW_EVENTS_ENABLED",
                                             os.environ.get("MOLTBOT_UW_EVENTS_ENABLED", "1")) == "1"
        _uw_events_log_only = os.environ.get("CLIO_UW_EVENTS_LOG_ONLY",
                                              os.environ.get("MOLTBOT_UW_EVENTS_LOG_ONLY", "1")) == "1"
        if _uw_events_enabled and not is_event:
            try:
                from agent.integrations.uw_signal_bridge import get_uw_bridge
                bridge = get_uw_bridge()
                if bridge.is_fresh():
                    vol = bridge.get_vol_regime()
                    vol_regime = vol.get("regime", "CALM")

                    # Trigger 1: Vol regime escalation (HIGH or CRISIS)
                    if vol_regime in ("HIGH", "CRISIS"):
                        logger.warning(
                            "CLIO-UW-4 VOL_REGIME trigger: %s (IV rank %s, term %s)",
                            vol_regime, vol.get("iv_rank_spy"), vol.get("term_structure"),
                        )
                        if not _uw_events_log_only:
                            is_event = True
                            if vol_regime == "CRISIS":
                                event_type = "uw_vol_crisis"
                            else:
                                event_type = "uw_vol_high"

                    # Trigger 2: Event risk + imminent economic calendar
                    if vol.get("event_risk_flag"):
                        mc = bridge.get_market_context()
                        econ_next = mc.get("economic_calendar_next", {})
                        econ_date = econ_next.get("date", "")
                        if econ_date:
                            try:
                                econ_dt = datetime.fromisoformat(econ_date) if "T" in econ_date else datetime.strptime(econ_date, "%Y-%m-%d").replace(tzinfo=_NY)
                                if econ_dt.tzinfo is None:
                                    econ_dt = econ_dt.replace(tzinfo=_NY)
                                hours_until = (econ_dt - datetime.now(_NY)).total_seconds() / 3600
                                if 0 < hours_until < 24:
                                    logger.warning(
                                        "CLIO-UW-4 EVENT_RISK: backwardation + %s in %.0fh",
                                        econ_next.get("event", "?"), hours_until,
                                    )
                                    if not _uw_events_log_only and not is_event:
                                        is_event = True
                                        event_type = "uw_event_risk"
                            except (ValueError, TypeError):
                                pass

                    # Build held position set once for overlap checks (Triggers 3, 4, 5).
                    # CLIO's own DPOF book (Hydrinia is permanently off; its
                    # position_log.json is frozen and no longer reflects reality).
                    held = set()
                    try:
                        from agent.dpof import readmodel
                        held = set(readmodel.open_underlyings())
                    except Exception as _he:
                        logger.debug("CLIO-UW-4 held position load failed: %s", _he)

                    # Trigger 3: Unusual ticker overlap with open positions
                    unusual = bridge.get_unusual_tickers()
                    if unusual:
                        overlap = set(unusual) & held
                        if overlap:
                            logger.warning("CLIO-UW-4 UNUSUAL_OVERLAP: %s (open positions)", overlap)

                    # Trigger 4: Earnings proximity for held positions (NEW — CLIO-UW-4)
                    earnings = bridge.get_earnings()
                    upcoming = earnings.get("upcoming", [])
                    if upcoming and held:
                        for earn_item in upcoming[:10]:
                            e_ticker = earn_item.get("ticker", "")
                            e_date = earn_item.get("date", "")
                            if e_ticker in held and e_date:
                                try:
                                    e_dt = datetime.fromisoformat(e_date) if "T" in e_date else datetime.strptime(e_date, "%Y-%m-%d").replace(tzinfo=_NY)
                                    if e_dt.tzinfo is None:
                                        e_dt = e_dt.replace(tzinfo=_NY)
                                    hours_until = (e_dt - datetime.now(_NY)).total_seconds() / 3600
                                    if 0 < hours_until < 48:
                                        logger.warning(
                                            "CLIO-UW-4 EARNINGS_PROXIMITY: %s earnings in %.0fh (held position)",
                                            e_ticker, hours_until,
                                        )
                                except (ValueError, TypeError):
                                    pass

                    # Trigger 5: Prediction unusual spike on held/watchlist (NEW — CLIO-UW-4)
                    predictions = bridge.get_predictions()
                    unusual_finance = predictions.get("unusual_finance", [])
                    if unusual_finance and held:
                        for pred_item in unusual_finance:
                            p_ticker = pred_item.get("ticker", "")
                            try:
                                p_score = float(pred_item.get("activity_score", 0))
                            except (ValueError, TypeError):
                                p_score = 0.0
                            if p_ticker in held and p_score > 7.0:
                                logger.warning(
                                    "CLIO-UW-4 PREDICTION_SPIKE: %s activity_score=%.1f (held position)",
                                    p_ticker, p_score,
                                )

                    # Trigger 6: ETF tide divergence (NEW — CLIO-UW-4)
                    sector_enh = bridge.get_sector_enhanced()
                    etf_tide = sector_enh.get("etf_tide", {})
                    if isinstance(etf_tide, dict) and etf_tide.get("divergence_from_equity"):
                        logger.warning(
                            "CLIO-UW-4 ETF_DIVERGENCE: ETF tide=%s, divergence from equity detected",
                            etf_tide.get("direction", "?"),
                        )
                        if not _uw_events_log_only and vol_regime in ("HIGH", "CRISIS") and not is_event:
                            is_event = True
                            event_type = "uw_etf_divergence"

                    # Trigger 7: Correlation breakdown (NEW — CLIO-UW-4)
                    regime_enh = bridge.get_regime_enhanced()
                    corr_changes = regime_enh.get("correlation_changes", {})
                    if isinstance(corr_changes, dict):
                        corr_alerts = [pair for pair, data in corr_changes.items()
                                       if isinstance(data, dict) and data.get("alert")]
                        if corr_alerts:
                            logger.warning(
                                "CLIO-UW-4 CORRELATION_BREAK: %s — cross-asset regime shift detected",
                                ", ".join(corr_alerts),
                            )
                            if not _uw_events_log_only and not is_event:
                                is_event = True
                                event_type = "uw_correlation_break"

            except Exception as e:
                logger.debug("CLIO-UW-4 event enrichment failed: %s", e)
        # --- END CC-CLIO-UW-4 ---

        # --- Lighthouse v2.0: RSS breaking news check (reuse pre-check result) ---
        if not is_event and _rss_precheck_event:
            is_event = True
            event_type = "rss_breaking_news"
            logger.warning("LIGHTHOUSE: RSS breaking news -> event mode: %s", _rss_precheck_event.get("sample_title", ""))

        if not is_event:
            return

        # Determine event type for logging (skip if UW triggers already set it)
        if event_type is None:
            if spy_pct <= -SPY_MOVE_PCT:
                event_type = "spy_crash"
            elif spy_pct >= SPY_MOVE_PCT:
                event_type = "spy_rally"
            elif vix_change >= VIX_SPIKE_PTS:
                event_type = "vix_spike"
            else:
                event_type = "vix_elevated"

        queries = _generate_event_queries(spy_pct, vix_level, vix_change)

        # CC-CLIO-UW-4: Inject UW-specific queries when triggered by vol regime or regime shifts
        if event_type is not None and event_type.startswith("uw_"):
            now_str = datetime.now(_NY).strftime("%B %d %Y")
            uw_queries = [
                "options implied volatility regime change trading strategies 2026",
                "VIX term structure backwardation event risk hedging strategies",
                f"market volatility spike analysis {now_str} options positioning",
                "gamma exposure dealer hedging market impact analysis 2026",
            ]
            if event_type == "uw_correlation_break":
                uw_queries.insert(0, f"cross-asset correlation breakdown regime shift trading {now_str}")
            elif event_type == "uw_etf_divergence":
                uw_queries.insert(0, f"ETF options flow divergence equity market regime {now_str}")
            # Prepend UW queries (higher priority), keep SPY/VIX queries as fallback
            queries = uw_queries + queries
            # S159 L-2: Explicit dedup (replaces set.add() anti-pattern)
            _seen = set()
            _deduped = []
            for q in queries:
                if q not in _seen:
                    _seen.add(q)
                    _deduped.append(q)
            queries = _deduped[:5]

        # S135 U-5: Inject RSS-specific queries when triggered by Lighthouse breaking news
        if event_type == "rss_breaking_news" and _rss_precheck_event:
            now_str = datetime.now(_NY).strftime("%B %d %Y")
            crisis_kws = ", ".join(_rss_precheck_event.get("keywords", [])[:4])
            regions = ", ".join(_rss_precheck_event.get("regions", [])[:3])
            sample = _rss_precheck_event.get("sample_title", "")[:60]
            rss_queries = [
                f"geopolitical crisis {crisis_kws} market impact {now_str}",
                f"{sample} financial markets analysis",
                f"{crisis_kws} commodity oil energy impact trading {now_str}",
                f"{regions} geopolitical risk portfolio hedging strategies 2026",
            ]
            queries = rss_queries + queries
            # S159 L-2: Explicit dedup (replaces set.add() anti-pattern)
            _seen = set()
            _deduped = []
            for q in queries:
                if q not in _seen:
                    _seen.add(q)
                    _deduped.append(q)
            queries = _deduped[:5]

        now = datetime.now(_NY)
        breaking_state = {
            "active": True,
            "triggered_at": now.isoformat(),
            "event_type": event_type,
            "spy_pct": round(spy_pct, 2),
            "vix_level": round(vix_level, 1),
            "vix_change": round(vix_change, 1),
            "queries": queries,
            "query_index": 0,
            "cycles_remaining": NEWS_MODE_CYCLES,
            "cooldown_until": (now + timedelta(hours=COOLDOWN_HOURS)).isoformat(),
        }

        state["breaking_news"] = breaking_state
        await _loop.run_in_executor(None, _save_learning_state, state)  # S233: off event loop

        logger.warning(
            f"BREAKING NEWS MODE ACTIVATED: {event_type} | "
            f"SPY {spy_pct:+.1f}% | VIX {vix_level:.1f} ({vix_change:+.1f}pts) | "
            f"{len(queries)} queries queued"
        )

        # Telegram notification
        if context and hasattr(context, "bot"):
            try:
                from agent.integrations.scheduled_tasks import _safe_send, _get_chat_ids
                for chat_id in _get_chat_ids():
                    await _safe_send(
                        context.bot, chat_id,
                        text=(
                            f"Breaking News Mode ACTIVATED\n"
                            f"Event: {event_type.replace('_', ' ').upper()}\n"
                            f"SPY: {spy_pct:+.1f}% | VIX: {vix_level:.1f} ({vix_change:+.1f})\n"
                            f"Learning engine switching to event-focused research "
                            f"for {NEWS_MODE_CYCLES} cycles."
                        ),
                    )
            except Exception as e:
                logger.debug(f"Breaking news Telegram notification failed: {e}")

    except Exception as e:
        logger.error(f"Market event check failed: {e}", exc_info=True)
