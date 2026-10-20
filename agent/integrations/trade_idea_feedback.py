"""
Trade Idea Feedback Loop — Feature 6

Polls Hydrinia's trade debrief directory, matches debriefs to original Moltbot trade ideas,
and feeds outcomes into episodic + meta memory for calibration learning.

Closes the Moltbot -> Hydrinia -> Moltbot feedback loop.
"""

import asyncio
import fcntl
import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from typing import Optional, Tuple

from agent.config import config

logger = logging.getLogger("openclaw.trade_idea_feedback")

# S156: Dedup for UW calibration episodic records (per-session, prevents re-evaluation spam)
# S159 M-8: Changed from set to dict with timestamps for time-based expiry
_uw_cal_recorded: dict[str, float] = {}

# Gap Signal Bridge — conviction gating from PHOENIX tier + CRUCIBLE concentration


def _get_gap_bridge():
    """Lazy import to avoid circular dependency."""
    try:
        from agent.integrations.gap_signal_bridge import get_gap_bridge
        return get_gap_bridge()
    except ImportError:
        return None


def _normalize_conviction_value(value) -> Optional[float]:
    """Return a finite 0..1 conviction value, or None when the value is absent."""
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return max(0.0, min(1.0, numeric))


def _select_idea_conviction(idea: dict) -> Tuple[Optional[float], Optional[str], Optional[float], bool]:
    """Prefer conviction over confidence, without inventing a neutral default."""
    for key in ("conviction", "confidence"):
        if key not in idea:
            continue
        raw = idea.get(key)
        normalized = _normalize_conviction_value(raw)
        if normalized is None:
            continue
        raw_float = float(raw)
        return normalized, key, raw_float, normalized != raw_float
    return None, None, None, False


def _current_adjusted_conviction(idea: dict) -> Tuple[Optional[float], Optional[str]]:
    for key in ("gap_adjusted_conviction", "conviction", "confidence"):
        if key not in idea:
            continue
        normalized = _normalize_conviction_value(idea.get(key))
        if normalized is not None:
            return normalized, key
    return None, None


def _append_conviction_bounds_warning(idea: dict, warning: str) -> None:
    warnings = list(idea.get("conviction_bounds_warnings") or [])
    warnings.append(warning)
    idea["conviction_bounds_warnings"] = warnings
    idea["conviction_bounds_warning"] = warnings[0]


_DEBRIEF_DIR = os.path.join(config.tradingbot_dir, "moltbot_knowledge", "trade_debriefs")
_TRADE_IDEAS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "knowledge", "trade_ideas",
)
_STATE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "knowledge", "trade_idea_feedback_state.json",
)

_MAX_PER_RUN = 20
_STATE_LOCK = _STATE_FILE + ".lock"


def _load_state() -> dict:
    lock_fd = None
    try:
        lock_fd = open(_STATE_LOCK, "a")
        fcntl.flock(lock_fd, fcntl.LOCK_SH)
        with open(_STATE_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"processed_debriefs": [], "total_processed": 0, "wins": 0, "losses": 0}
    finally:
        if lock_fd:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
            except Exception as e:
                logger.debug(f"Lock release failed in _load_state: {e}")


def _save_state(state: dict) -> None:
    lock_fd = None
    try:
        lock_fd = open(_STATE_LOCK, "a")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        tmp = _STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, _STATE_FILE)
    finally:
        if lock_fd:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
            except Exception as e:
                logger.debug(f"Lock release failed in _save_state: {e}")


def _load_trade_ideas() -> dict:
    """Load trade ideas indexed by both ID and ticker for matching.

    Returns dict with:
      "by_id": {idea_id: idea_dict}  — exact match via trade_id
      "by_ticker": {ticker: [idea_dict, ...]}  — S160-4: list per ticker, newest first
    """
    by_id: dict[str, dict] = {}
    by_ticker: dict[str, list] = {}
    if not os.path.isdir(_TRADE_IDEAS_DIR):
        return {"by_id": by_id, "by_ticker": by_ticker}
    for fname in os.listdir(_TRADE_IDEAS_DIR):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(_TRADE_IDEAS_DIR, fname), "r") as f:
                idea = json.load(f)
            idea_id = idea.get("id", "")
            if idea_id:
                by_id[idea_id] = idea
            ticker = idea.get("ticker", "")
            if not ticker:
                continue
            by_ticker.setdefault(ticker, []).append(idea)
        except Exception as e:
            logger.debug(f"Failed to parse trade idea file: {e}")
            continue
    # Sort each ticker's ideas newest first
    for _tk in by_ticker:
        by_ticker[_tk].sort(key=lambda x: x.get("created", ""), reverse=True)
    return {"by_id": by_id, "by_ticker": by_ticker}


def _match_debrief_to_idea(debrief: dict, ideas: dict) -> Optional[Tuple[dict, int]]:
    """Match debrief to trade idea. Multi-layer: ID → moltbot_idea_id → ticker+direction+time.

    S160-4: Rewritten to fix trade_id namespace mismatch (dpof_* vs ti_*).
    S190 H3: Returns (idea, match_layer) tuple. match_layer ∈ {1, 2, 3} corresponds
    to Layer 1 / Layer 1b / Layer 2 respectively. Returns None on no match. Layer
    1 is strongest (direct trade_id match); Layer 2 (ticker+direction+time) is
    weakest and may produce false positives that operator can audit post-hoc via
    the new match_layer field on each SEAL reward record.
    """
    # Layer 1: Direct ID match (works if Hydrinia preserves moltbot idea ID as trade_id)
    trade_id = debrief.get("trade_id", "")
    if trade_id and trade_id in ideas["by_id"]:
        return (ideas["by_id"][trade_id], 1)
    # Layer 1b: Check moltbot_idea_id field (HI-9 — Hydrinia stamps via _generate_debrief)
    moltbot_id = debrief.get("moltbot_idea_id", "")
    if moltbot_id and moltbot_id in ideas["by_id"]:
        return (ideas["by_id"][moltbot_id], 2)

    # S185 W2 — observability canary for HI-9 chain health (S185 UL-2.1: INFO not DEBUG
    # so it actually surfaces in production at default log level — see logging_config.py
    # console+file handlers at INFO; root_logger filters DEBUG before reaching handlers).
    # If Hydrinia drops the field or stamps an unknown id, we want a visible trace so
    # a regression doesn't masquerade as healthy Layer-2 fallback matching.
    # Note: also distinguish "Layer 1 trade_id was set but did not match" vs absent
    # (S2-1 second-pass clarity feedback).
    if not moltbot_id:
        logger.info(
            "Layer 1b miss: debrief lacks moltbot_idea_id (symbol=%s, trade_id=%r) "
            "— falling through to Layer 2",
            debrief.get("symbol", "?"), trade_id or "(absent)",
        )
    elif moltbot_id not in ideas.get("by_id", {}):
        logger.info(
            "Layer 1b miss: idea_id=%r not in known ideas (symbol=%s) "
            "— falling through to Layer 2",
            moltbot_id, debrief.get("symbol", "?"),
        )

    # Layer 2: Ticker + direction + time-window matching
    symbol = debrief.get("symbol", "").upper()
    if not symbol:
        return None
    candidates = ideas.get("by_ticker", {}).get(symbol, [])
    if not candidates:
        return None

    # Parse debrief timestamp for proximity matching
    try:
        _match_window_h = int(os.environ.get("TRADE_IDEA_MATCH_WINDOW_HOURS", "72"))
    except (ValueError, TypeError):
        _match_window_h = 72
    _match_window_s = _match_window_h * 3600

    debrief_ts = debrief.get("timestamp", "")
    debrief_dt = None
    if debrief_ts:
        try:
            if isinstance(debrief_ts, str):
                debrief_dt = datetime.fromisoformat(debrief_ts)
            else:
                debrief_dt = datetime.fromtimestamp(float(debrief_ts), tz=timezone.utc)
            if debrief_dt.tzinfo is None:
                debrief_dt = debrief_dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError, OSError):
            debrief_dt = None

    # Normalize debrief direction
    _raw_dir = debrief.get("direction", debrief.get("side", "")).upper()
    debrief_dir = "LONG" if _raw_dir in ("BUY", "LONG", "BULLISH") else (
        "SHORT" if _raw_dir in ("SELL", "SHORT", "BEARISH") else _raw_dir
    )

    best_match = None
    best_gap = _match_window_s + 1  # Upper bound for gap tracking

    for idea in candidates:
        _idea_dir = idea.get("direction", "").upper()
        _idea_dir_norm = "LONG" if _idea_dir in ("BUY", "LONG", "BULLISH") else (
            "SHORT" if _idea_dir in ("SELL", "SHORT", "BEARISH") else _idea_dir
        )
        # Direction must match
        if _idea_dir_norm != debrief_dir:
            continue

        # Time proximity check
        if debrief_dt:
            idea_ts = idea.get("created", "")
            try:
                idea_dt = datetime.fromisoformat(idea_ts)
                if idea_dt.tzinfo is None:
                    idea_dt = idea_dt.replace(tzinfo=timezone.utc)
                gap = abs((debrief_dt - idea_dt).total_seconds())
                if gap > _match_window_s:
                    continue
                if gap < best_gap:
                    best_gap = gap
                    best_match = idea
            except (ValueError, TypeError):
                if best_match is None:
                    best_match = idea
        else:
            # No debrief timestamp — accept first direction match (newest, already sorted)
            if best_match is None:
                best_match = idea

    # S162: Diagnostic logging for SEAL reward pipeline tracing
    if best_match is None:
        logger.info(
            f"SEAL match failed: symbol={symbol} dir={debrief_dir} "
            f"candidates={len(candidates)} debrief_ts={debrief_ts[:19] if debrief_ts else 'none'}"
        )
        return None
    else:
        logger.debug(
            f"SEAL match OK (Layer 2): symbol={symbol} -> idea={best_match.get('id', '?')[:30]} "
            f"cat={best_match.get('source_category', '?')}"
        )
    return (best_match, 3)  # S190 H3: Layer 2 = weakest signal, may FP


def _apply_gap_conviction_filter(idea: dict) -> dict:
    """Apply PHOENIX tier conviction gating and CRUCIBLE concentration awareness.

    Modifies the idea dict in-place with additional metadata:
      - 'gap_conviction_warning': str (first warning) or None
      - 'gap_warnings': list of all warning strings
      - 'gap_filtered': bool (True if idea below PHOENIX minimum)
      - 'gap_adjusted_conviction': float (original or boosted/penalized)

    Returns the modified idea dict.
    """
    gap_enabled = os.environ.get("MOLTBOT_GAP_CONVICTION_ENABLED", "1") == "1"
    gap_log_only = os.environ.get("MOLTBOT_GAP_CONVICTION_LOG_ONLY", "1") == "1"

    if not gap_enabled:
        return idea

    bridge = _get_gap_bridge()
    if not bridge:
        return idea

    # S156 H-A: Skip gap filtering if bridge data is stale (fail-open)
    try:
        _drawdown = bridge.get_drawdown_state()
        if _drawdown and _drawdown.get("_is_stale"):
            logger.info("MB-GAP-3: Bridge data stale — skipping gap conviction filter")
            return idea
    except Exception as e:
        logger.debug(f"MB-GAP-3: Stale check failed: {e}")

    # CR-7 S128: Shallow copy to avoid mutating the shared idea dict from lookup index
    idea = dict(idea)

    idea_ticker = (idea.get("ticker") or idea.get("symbol") or "UNKNOWN")
    idea_sector = (idea.get("sector") or "Unknown")

    warnings = []
    idea_conviction, conviction_source, conviction_raw, conviction_clamped = _select_idea_conviction(idea)
    if idea_conviction is None:
        warning = (
            "CONVICTION_BOUNDS: missing/invalid conviction and confidence; "
            "dropping GAP evaluation instead of defaulting"
        )
        warnings.append(warning)
        idea["gap_filtered"] = True
        idea["gap_adjusted_conviction"] = None
        idea["gap_warnings"] = warnings
        idea["gap_conviction_warning"] = warning
        _append_conviction_bounds_warning(idea, warning)
        if gap_log_only:
            logger.info("MB-GAP-3: [LOG_ONLY] %s for %s", warning, idea_ticker)
        else:
            logger.info("MB-GAP-3: %s for %s", warning, idea_ticker)
        return idea

    idea["conviction_bounds_source"] = conviction_source
    idea["conviction_bounds_applied"] = False
    if conviction_clamped:
        warning = (
            f"CONVICTION_BOUNDS: {conviction_source} {conviction_raw:.4f} "
            f"clamped to {idea_conviction:.4f}"
        )
        warnings.append(warning)
        idea["conviction_bounds_applied"] = True
        idea["conviction_bounds_original"] = conviction_raw
        _append_conviction_bounds_warning(idea, warning)

    # ── PHOENIX Tier Conviction Gate ──
    min_conviction = bridge.get_min_conviction()
    current_tier = bridge.get_current_tier()

    if idea_conviction < min_conviction and current_tier != "NORMAL":
        warning = (f"PHOENIX {current_tier}: Idea conviction {idea_conviction:.2f} "
                   f"below tier minimum {min_conviction:.2f} — would be rejected by Hydrinia")
        warnings.append(warning)
        idea["gap_filtered"] = True

        if gap_log_only:
            logger.info(f"MB-GAP-3: [LOG_ONLY] {warning} for {idea_ticker}")
        else:
            logger.info(f"MB-GAP-3: {warning} for {idea_ticker}")
    else:
        idea["gap_filtered"] = False

    # ── Leveraged ETF Gate ──
    # Hedge tickers take priority over leveraged classification during drawdown tiers
    _HEDGE_TICKERS = {"UVIX", "UGL", "BOIL", "SH", "PSQ", "SPXU", "SQQQ"}
    is_leveraged = idea.get("is_leveraged_etf", False)
    _KNOWN_LEVERAGED = {"SOXL", "TQQQ", "UVIX", "UGL", "BOIL", "NRGU", "AMDL",
                         "AMZU", "GGLL", "METU", "NFXL", "MSFU", "AAPU", "TSLL"}
    if idea_ticker in _KNOWN_LEVERAGED:
        is_leveraged = True

    # Skip leveraged block if ticker is a hedge AND we're in a drawdown tier
    _is_hedge_in_drawdown = (idea_ticker in _HEDGE_TICKERS
                             and current_tier in ("CAUTION", "RECOVERY", "EMERGENCY"))

    if is_leveraged and not bridge.is_leveraged_etf_allowed() and not _is_hedge_in_drawdown:
        warning = f"PHOENIX {current_tier}: Leveraged ETF {idea_ticker} entry BLOCKED"
        warnings.append(warning)
        idea["gap_filtered"] = True

        if gap_log_only:
            logger.info(f"MB-GAP-3: [LOG_ONLY] {warning}")
        else:
            logger.info(f"MB-GAP-3: {warning}")

    # ── CRUCIBLE Concentration Check ──
    sector_pcts = bridge.get_sector_concentration()
    if sector_pcts and idea_sector in sector_pcts:
        # S170B P2-1: preserve legitimate 0.0 — `or 0` masked the actual value.
        _sp = sector_pcts.get(idea_sector)
        current_pct = _sp if isinstance(_sp, (int, float)) and not isinstance(_sp, bool) else 0
        if current_pct > 35:
            warning = (f"CRUCIBLE: {idea_sector} at {current_pct:.0f}% concentration — "
                       f"adding {idea_ticker} increases risk")
            warnings.append(warning)

            if gap_log_only:
                logger.info(f"MB-GAP-3: [LOG_ONLY] Concentration warning: {warning}")
            else:
                logger.info(f"MB-GAP-3: Concentration warning: {warning}")

    # ── CRUCIBLE Hedge Encouragement ──
    # _HEDGE_TICKERS defined above (leveraged ETF gate section)
    if idea_ticker in _HEDGE_TICKERS and current_tier in ("CAUTION", "RECOVERY", "EMERGENCY"):
        hedge_boost = 0.10
        idea["gap_adjusted_conviction"] = min(1.0, idea_conviction + hedge_boost)
        note = (f"PHOENIX {current_tier}: Hedge idea {idea_ticker} gets +{hedge_boost:.2f} "
                f"conviction boost ({idea_conviction:.2f} → {idea['gap_adjusted_conviction']:.2f})")
        warnings.append(note)

        if gap_log_only:
            logger.info(f"MB-GAP-3: [LOG_ONLY] {note}")
        else:
            logger.info(f"MB-GAP-3: {note}")
    else:
        idea["gap_adjusted_conviction"] = idea_conviction

    # Store warnings
    idea["gap_warnings"] = warnings
    idea["gap_conviction_warning"] = warnings[0] if warnings else None

    # ── SEAL Reward Emission ──
    if not gap_log_only and warnings:
        try:
            from agent.integrations.seal_reward_engine import get_reward_engine
            re = get_reward_engine()
            aligned = not idea.get("gap_filtered", False)
            re.record_binary_reward(
                idea_ticker,
                aligned,
                0.0,
                idea.get("trade_idea_id", "unknown"),
                "trading"  # S134: was "portfolio_awareness" (orphan, not in learning categories)
            )
        except Exception as e:
            logger.debug(f"SEAL gap reward emission skipped: {e}")

    return idea


# ============================================================
# CC-CLIO-FD-2: Fundamental Quality Gate
# ============================================================

def _fd_fundamental_check(ticker: str) -> dict:
    """Score fundamental quality before approving a trade idea.

    Sync — callers use run_in_executor(). Budget check is caller's responsibility.
    Returns: {"enabled": bool, "score": float, "grade": str, "factors": list}
    """
    if os.environ.get("CLIO_FD_TRADE_IDEA_FUNDAMENTALS", "0") != "1":
        return {"enabled": False}
    try:
        from agent.integrations.fd_client import get_fd_client
        result = get_fd_client().fundamental_score(ticker)
        logger.info("CC-CLIO-FD-2: %s fundamental score: %s (%s) — %s",
                     ticker, result["grade"], result["score"], result["factors"])
        return {"enabled": True, **result}
    except Exception as e:
        logger.warning("CC-CLIO-FD-2: Fundamental check failed for %s: %s", ticker, e)
        return {"enabled": True, "score": 0.5, "grade": "N/A", "factors": [str(e)]}


async def process_trade_idea_feedback(context) -> None:
    """Scheduled job: poll debrief directory, match to trade ideas, record outcomes."""
    # Hydrinia is permanently off (2026-09-24) — this whole module polls its
    # moltbot_knowledge/trade_debriefs/ bridge dir, which nothing writes to any
    # more. Defense in depth beyond the job-registration gate.
    try:
        from agent.hydrinia_bridge import bridge_enabled
        if not bridge_enabled():
            return
    except Exception as e:
        logger.debug(f"Bridge check failed (treating as disabled): {e}")
        return

    if not os.path.isdir(_DEBRIEF_DIR):
        return

    # S170B P0-NEW: flock-bearing _load_state must not block the asyncio event loop
    _loop = asyncio.get_event_loop()
    state = await _loop.run_in_executor(None, _load_state)
    processed = set(state.get("processed_debriefs", []))

    # List debrief files
    try:
        debrief_files = sorted(
            f for f in os.listdir(_DEBRIEF_DIR)
            if f.endswith(".json") and f not in processed
        )
    except OSError as e:
        logger.warning(f"Cannot list debrief directory: {e}")
        return

    if not debrief_files:
        return

    debrief_files = debrief_files[:_MAX_PER_RUN]
    ideas = _load_trade_ideas()

    # ── S129 Pass 1: Load debriefs, match to ideas, group by idea ID ──
    # Multiple debriefs for the same position (partial fills/adjustments) must be
    # aggregated into a single event to avoid poisoning SEAL bandit statistics.
    # Key = idea ID (or "unmatched_{symbol}") → list of (fname, debrief, idea)
    # S144 H-4: Per-ticker daily rate limit with persistence (was per-run in S143)
    _DEBRIEF_MAX_PER_TICKER_PER_DAY = 5
    _today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    _daily_data = state.get("daily_ticker_counts", {})
    if _daily_data.get("date") != _today_str:
        _daily_data = {"date": _today_str, "counts": {}}
    _ticker_counts: dict[str, int] = _daily_data.get("counts", {})
    _rate_limited_count = 0

    matched_groups: dict[str, list[tuple[str, dict, Optional[dict]]]] = {}
    # S190 H3: parallel dict tracking which matcher Layer (1/2/3) matched each group.
    # Used to tag the downstream SEAL reward record so operator can audit Layer-2
    # false-positive rate post-hoc. Layer 0 = unmatched group.
    group_match_layers: dict[str, int] = {}
    for fname in debrief_files:
        try:
            with open(os.path.join(_DEBRIEF_DIR, fname), "r") as f:
                debrief = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read debrief {fname}: {e}")
            processed.add(fname)
            continue

        symbol = debrief.get("symbol", "UNKNOWN")

        # S142-B W2-A: Rate limit per ticker
        _ticker_counts[symbol] = _ticker_counts.get(symbol, 0) + 1
        if _ticker_counts[symbol] > _DEBRIEF_MAX_PER_TICKER_PER_DAY:
            _rate_limited_count += 1
            processed.add(fname)  # Mark as processed so we don't retry
            continue

        # S190 H3: matcher returns (idea, match_layer) tuple now. Track layer per
        # group so the SEAL reward emission downstream can tag the reward record.
        _matched = _match_debrief_to_idea(debrief, ideas)
        if _matched is None:
            idea = None
            _match_layer = 0  # no match
        else:
            idea, _match_layer = _matched
        group_key = idea.get("id", f"unmatched_{symbol}") if idea else f"unmatched_{symbol}"
        matched_groups.setdefault(group_key, []).append((fname, debrief, idea))
        # First entry into the group sets the layer; subsequent entries should
        # match via the same layer (same idea object identity).
        if group_key not in group_match_layers:
            group_match_layers[group_key] = _match_layer

    if _rate_limited_count > 0:
        logger.info(f"Debrief rate-limited: {_rate_limited_count} files skipped (>{_DEBRIEF_MAX_PER_TICKER_PER_DAY}/ticker/day)")

    # ── S129 Pass 2: Process each group as a single aggregated event ──
    results = []
    for group_key, entries in matched_groups.items():
        # Sort by filename (chronological) — use latest debrief for metadata
        entries.sort(key=lambda x: x[0])
        latest_fname, latest_debrief, idea = entries[-1]
        symbol = latest_debrief.get("symbol", "UNKNOWN")

        # Aggregate P&L across all debriefs in the group
        total_pnl = 0.0
        pnl_pct_sum = 0.0
        for _, d, _ in entries:
            try:
                total_pnl += float(d.get("pnl", 0) or 0)
            except (ValueError, TypeError):
                pass
            try:
                pnl_pct_sum += float(d.get("pnl_pct", 0) or 0)
            except (ValueError, TypeError):
                pass
        pnl = total_pnl
        pnl_pct = pnl_pct_sum / len(entries) if entries else 0.0
        is_win = pnl > 0

        # Mark ALL files in the group as processed
        for fn, _, _ in entries:
            processed.add(fn)

        if len(entries) > 1:
            logger.info(
                f"Debrief dedup: {len(entries)} debriefs for {symbol} "
                f"(idea={group_key}) aggregated into single event "
                f"(total_pnl=${pnl:+.0f}, avg_pnl_pct={pnl_pct*100:+.1f}%)"
            )

        # S151-B M-2: Write outcome back to trade idea JSON file
        if idea and idea.get("id"):
            try:
                _idea_path = os.path.join(_TRADE_IDEAS_DIR, f"{idea['id']}.json")
                if os.path.exists(_idea_path):
                    with open(_idea_path, "r") as _if:
                        _idea_data = json.load(_if)
                    _idea_data["outcome"] = "win" if is_win else "loss"
                    _idea_data["outcome_pnl_pct"] = round(pnl_pct, 6)
                    _idea_data["outcome_pnl"] = round(pnl, 2)
                    _idea_data["closed_at"] = datetime.now(timezone.utc).isoformat()
                    _idea_data["debrief_matched"] = True
                    _tmp_idea = _idea_path + ".tmp"
                    with open(_tmp_idea, "w") as _of:
                        json.dump(_idea_data, _of, indent=2)
                    os.replace(_tmp_idea, _idea_path)
            except (OSError, json.JSONDecodeError) as _oe:
                logger.debug(f"Trade idea outcome write failed: {_oe}")

        # MB-GAP-3: Apply portfolio-aware conviction gating (once per group)
        if idea:
            try:
                idea = _apply_gap_conviction_filter(idea)
                if idea.get("gap_filtered") and os.environ.get("MOLTBOT_GAP_CONVICTION_LOG_ONLY", "1") != "1":
                    logger.info(f"MB-GAP-3: Trade idea {idea.get('ticker', 'UNKNOWN')} "
                                f"filtered by gap constraints — skipping evaluation")
            except Exception as e:
                logger.error(f"MB-GAP-3: Filter failed: {e}", exc_info=True)

        # CC-CLIO-FD-2: Fundamental quality overlay (sync → run_in_executor)
        if idea:
            try:
                from agent.rate_limiter import budget_tracker
                _fd_ok, _ = await budget_tracker.check_budget("financial_datasets")
                if not _fd_ok:
                    raise Exception("financial_datasets budget exhausted")
                _loop = asyncio.get_running_loop()
                _idea_ticker = idea.get("ticker") or idea.get("symbol") or symbol
                # S170B P2-8: outer wait_for bounds the worst case where multiple
                # FD calls inside fundamental_score each hit per-request timeout.
                try:
                    fd_check = await asyncio.wait_for(
                        _loop.run_in_executor(None, _fd_fundamental_check, _idea_ticker),
                        timeout=90.0,
                    )
                except asyncio.TimeoutError:
                    logger.warning("CC-CLIO-FD-2: fundamental_score timed out for %s", _idea_ticker)
                    fd_check = {"enabled": True, "score": 0.5, "grade": "N/A", "factors": ["timeout"]}
                if fd_check.get("enabled") and fd_check.get("grade") != "N/A":
                    await budget_tracker.record_usage("financial_datasets")
                    _adj, _adj_source = _current_adjusted_conviction(idea)
                    if _adj is None:
                        _warn = "CONVICTION_BOUNDS: FD overlay skipped; no valid conviction field"
                        _append_conviction_bounds_warning(idea, _warn)
                        logger.info("CC-CLIO-FD-2: %s", _warn)
                    elif fd_check["grade"] in ("D", "F"):
                        idea["gap_adjusted_conviction"] = _adj * 0.85  # 15% penalty
                        logger.info("CC-CLIO-FD-2: %s fundamental grade %s — conviction %.2f → %.2f",
                                    _idea_ticker, fd_check["grade"], _adj, idea["gap_adjusted_conviction"])
                    elif fd_check["grade"] == "A":
                        idea["gap_adjusted_conviction"] = min(1.0, _adj * 1.05)  # 5% bonus
                        logger.info("CC-CLIO-FD-2: %s fundamental grade %s — conviction %.2f → %.2f",
                                    _idea_ticker, fd_check["grade"], _adj, idea["gap_adjusted_conviction"])
                    idea["fd_fundamental_grade"] = fd_check["grade"]
                    idea["fd_fundamental_score"] = fd_check["score"]
            except Exception as e:
                logger.debug("CC-CLIO-FD-2: Fundamental overlay failed: %s", e)

        # Record episodic memory (fire-and-forget) — single event for the group
        # S146: Dedup check to prevent 10-12x duplication per trade
        try:
            from agent.memory.memory_engine import get_memory_engine
            mem = get_memory_engine()

            # Check if this trade outcome was already recorded
            _existing = mem.episodic.search(
                event_type="trade", keyword=symbol,
                category="trade_idea_outcome", limit=30, months_back=2,
            )
            _debrief_key = latest_fname
            _already_logged = any(
                (e.get("metadata") or {}).get("debrief_file") == _debrief_key
                for e in _existing
            )
            if _already_logged:
                logger.debug(f"Episodic trade dedup: {symbol} {_debrief_key} already logged")
            else:
                content = (
                    f"Trade outcome: {symbol} — {'WIN' if is_win else 'LOSS'} "
                    f"({pnl_pct * 100:+.1f}%, ${pnl:+.0f}). "
                    f"Exit: {latest_debrief.get('exit_reason', 'unknown')}."
                )
                if len(entries) > 1:
                    content += f" ({len(entries)} partial fills aggregated.)"
                if idea:
                    # S159 M-7: Note pre-boost so calibration isn't skewed by UW adjustments
                    content += f" Original idea confidence: {idea.get('confidence', idea.get('conviction', '?'))} (pre-UW-boost)."
                mem.episodic.record(
                    event_type="trade",
                    content=content,
                    category="trade_idea_outcome",
                    outcome="success" if is_win else "failure",
                    importance=0.7 if idea else 0.5,
                    domain="trade_ideas",
                    metadata={
                        "symbol": symbol,
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "matched_idea": idea.get("id") if idea else None,
                        "debrief_file": latest_fname,
                        "debrief_count": len(entries),
                    },
                )
        except Exception as e:
            logger.warning("Episodic record failed: %s: %s", type(e).__name__, e, exc_info=True)

        # Record meta calibration if matched idea has confidence (fire-and-forget, dedup by pred_id)
        if idea and (idea.get("confidence") is not None or idea.get("conviction") is not None):
            try:
                from agent.memory.memory_engine import get_memory_engine
                mem = get_memory_engine()
                pred_id = idea.get('id') or f"ti_{symbol}"
                existing = mem.meta.get_calibration_by_prediction_id(pred_id)
                _calibration_confidence, _, _, _ = _select_idea_conviction(idea)
                if not existing and _calibration_confidence is not None:
                    mem.meta.record_prediction(
                        prediction_id=pred_id,
                        domain="trade_ideas",
                        stated_confidence=_calibration_confidence,
                        description=f"{symbol} {idea.get('direction', '?')} — {idea.get('thesis', '')[:100]}",
                    )
                    mem.meta.record_outcome(pred_id, is_win)
            except Exception as e:
                logger.debug(f"Meta calibration failed: {e}")

        # --- SEAL BINARY REWARD HOOK (once per group, not per debrief) ---
        try:
            from agent.integrations.seal_reward_engine import (
                get_reward_engine,
            )
            if os.environ.get("MOLTBOT_SEAL_REWARDS", "1") == "1" and idea:
                # S162: Fall back to category if source_category missing
                _seal_source_cat = idea.get("source_category", "") or idea.get("category", "")
                get_reward_engine().record_binary_reward(
                    ticker=symbol,
                    won=is_win,
                    pnl_pct=pnl_pct,
                    trade_idea_id=idea.get("id", ""),
                    source_category=_seal_source_cat,
                    idea_timestamp=idea.get("created", ""),
                    # S190 H3: tag with matcher Layer so operator can post-hoc
                    # filter Layer 2 (weakest signal) when auditing bandit signal.
                    match_layer=group_match_layers.get(group_key, 1),
                )
            elif os.environ.get("MOLTBOT_SEAL_REWARDS", "1") == "1" and not idea:
                # S162: Log when reward skipped due to no matching trade idea
                logger.info(
                    f"SEAL reward skipped: no matching trade idea for {symbol}"
                )
        except Exception as e:
            logger.debug(f"SEAL reward hook failed (non-fatal): {e}")
        # --- END SEAL BINARY REWARD HOOK ---

        # --- CC-CLIO-UW-3: UW CONVICTION MODIFIERS v2.0 (8 factors) ---
        _uw_conviction_enabled = os.environ.get("CLIO_UW_CONVICTION_ENABLED",
                                                 os.environ.get("MOLTBOT_UW_CONVICTION_ENABLED", "1")) == "1"
        _uw_conviction_log_only = os.environ.get("CLIO_UW_CONVICTION_LOG_ONLY",
                                                  os.environ.get("MOLTBOT_UW_CONVICTION_LOG_ONLY", "1")) == "1"
        if _uw_conviction_enabled and idea:
            try:
                from agent.integrations.uw_signal_bridge import get_uw_bridge
                bridge = get_uw_bridge()
                # S159-B M-1: Use stale snapshot during off-hours rather than
                # returning all-zeros. Stale data from last market session is
                # more informative than no data.
                _has_snapshot = bool(bridge._read_snapshot())
                if bridge.is_fresh() or _has_snapshot:
                    idea_direction = (idea.get("direction") or "").upper()
                    if idea_direction not in ("LONG", "BUY", "SHORT", "SELL"):
                        idea_direction = "LONG"  # default assumption
                    factors = bridge.get_conviction_factors(symbol, idea_direction, allow_stale=True)

                    total_boost = (factors["gex_boost"] + factors["flow_boost"]
                                   + factors["nope_boost"] + factors["max_pain_penalty"]
                                   + factors["oi_sr_boost"] + factors["predictions_boost"]
                                   + factors["technical_boost"])
                    total_boost = max(-0.50, min(0.50, total_boost))  # Cap

                    uw_detail = (
                        f"gex={factors['gex_boost']:+.2f} flow={factors['flow_boost']:+.2f} "
                        f"nope={factors['nope_boost']:+.2f} maxpain={factors['max_pain_penalty']:+.2f} "
                        f"oi_sr={factors['oi_sr_boost']:+.2f} pred={factors['predictions_boost']:+.2f} "
                        f"tech={factors['technical_boost']:+.2f} | "
                        f"total_boost={total_boost:+.2f} vol_sizing={factors['vol_sizing']:.2f}"
                    )

                    if _uw_conviction_log_only:
                        logger.info(
                            "CLIO-UW-3 [LOG_ONLY] conviction: %s | ticker=%s dir=%s | win=%s",
                            uw_detail, symbol, idea_direction, is_win,
                        )
                    else:
                        # S156: Apply UW conviction boost (active mode only)
                        if abs(total_boost) > 0.001:
                            _old_conv, _conv_src, _conv_raw, _conv_clamped = _select_idea_conviction(idea)
                            if _old_conv is None:
                                _warn = (
                                    "CONVICTION_BOUNDS: UW conviction boost skipped; "
                                    "no valid conviction field"
                                )
                                idea["uw_conviction_skipped"] = True
                                _append_conviction_bounds_warning(idea, _warn)
                                logger.info("CLIO-UW-3: %s for %s", _warn, symbol)
                            else:
                                if _conv_clamped:
                                    _warn = (
                                        f"CONVICTION_BOUNDS: {_conv_src} {_conv_raw:.4f} "
                                        f"clamped to {_old_conv:.4f} before UW boost"
                                    )
                                    _append_conviction_bounds_warning(idea, _warn)
                                idea["conviction"] = max(0.10, min(0.95, _old_conv + total_boost))
                                idea["uw_boost_applied"] = round(total_boost, 4)
                                logger.info(
                                    "UW conviction boost: %.2f -> %.2f (%+.3f) for %s",
                                    _old_conv, idea["conviction"], total_boost, symbol,
                                )
                                # Persist boost fields to trade idea JSON (audit trail)
                                if idea.get("id"):
                                    try:
                                        _bp = os.path.join(_TRADE_IDEAS_DIR, f"{idea['id']}.json")
                                        if os.path.exists(_bp):
                                            with open(_bp, "r") as _bf:
                                                _bd = json.load(_bf)
                                            _bd["uw_boost_applied"] = idea["uw_boost_applied"]
                                            _bd["uw_adjusted_conviction"] = idea["conviction"]
                                            _bt = _bp + ".tmp"
                                            with open(_bt, "w") as _bw:
                                                json.dump(_bd, _bw, indent=2)
                                            os.replace(_bt, _bp)
                                    except (OSError, json.JSONDecodeError) as _be:
                                        logger.debug("UW boost persist failed: %s", _be)
                        logger.info(
                            "CLIO-UW-3 conviction: %s | ticker=%s dir=%s | win=%s",
                            uw_detail, symbol, idea_direction, is_win,
                        )
                        # S156: Dedup — skip if already recorded for this idea+symbol
                        _uw_dedup_key = f"{idea.get('id', '')}_{symbol}"
                        # S159 M-8: Time-based expiry (7 days) instead of size-only
                        _now = time.time()
                        _uw_cal_recorded_clean = {k: v for k, v in _uw_cal_recorded.items() if _now - v < 7 * 86400}
                        _uw_cal_recorded.clear()
                        _uw_cal_recorded.update(_uw_cal_recorded_clean)
                        if _uw_dedup_key not in _uw_cal_recorded:
                            try:
                                from agent.memory.memory_engine import get_memory_engine
                                _mem = get_memory_engine()
                                _mem.episodic.record(
                                    event_type="uw_conviction",
                                    content=(
                                        f"UW conviction v2 at outcome: {symbol} "
                                        f"{'WIN' if is_win else 'LOSS'} ({pnl_pct * 100:+.1f}%) — "
                                        f"{uw_detail}"
                                    ),
                                    category="trade_idea_uw_calibration",
                                    outcome="success" if is_win else "failure",
                                    importance=0.6,
                                    domain="uw_calibration",
                                    metadata={
                                        "symbol": symbol,
                                        "factors": factors,
                                        "total_boost": total_boost,
                                        "is_win": is_win,
                                        "pnl_pct": pnl_pct,
                                    },
                                )
                                _uw_cal_recorded[_uw_dedup_key] = _now
                                # S159 M-8: Cap dict size as fallback
                                if len(_uw_cal_recorded) > 500:
                                    _uw_cal_recorded_trimmed = dict(list(_uw_cal_recorded.items())[-500:])
                                    _uw_cal_recorded.clear()
                                    _uw_cal_recorded.update(_uw_cal_recorded_trimmed)
                            except Exception as _e:
                                logger.warning("CLIO-UW-3 episodic record failed: %s: %s", type(_e).__name__, _e, exc_info=True)
                        else:
                            logger.debug("CLIO-UW-3: skipping duplicate calibration for %s", _uw_dedup_key)
            except Exception as e:
                logger.debug("CLIO-UW-3 conviction scoring failed: %s", e)
        # --- END CC-CLIO-UW-3 ---

        _exit = latest_debrief.get("exit_reason", "unknown")
        results.append({
            "symbol": symbol,
            "win": is_win,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "matched": idea is not None,
            "exit_reason": _exit,
        })

    if not results:
        return

    # Update state
    wins = sum(1 for r in results if r["win"])
    losses = len(results) - wins
    state["processed_debriefs"] = sorted(processed)[-200:]  # Cap at 200
    state["total_processed"] = state.get("total_processed", 0) + len(results)
    state["wins"] = state.get("wins", 0) + wins
    state["losses"] = state.get("losses", 0) + losses
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    state["daily_ticker_counts"] = {"date": _today_str, "counts": _ticker_counts}  # S144 H-4: persist
    # Rationale for this batch
    _symbols = [r["symbol"] for r in results]
    _exits = [r.get("exit_reason", "?") for r in results]
    from collections import Counter
    _exit_counts = Counter(_exits)
    _top_exit = _exit_counts.most_common(1)[0][0] if _exit_counts else "?"
    state["last_batch_rationale"] = (
        f"{wins}W/{losses}L on {', '.join(_symbols[:5])}{'...' if len(_symbols) > 5 else ''}. "
        f"Primary exit: {_top_exit}."
    )
    # S170B P0-NEW: flock-bearing _save_state must not block the asyncio event loop
    await _loop.run_in_executor(None, _save_state, state)

    # Send Telegram notification
    matched = sum(1 for r in results if r["matched"])
    try:
        from agent.integrations.scheduled_tasks import _safe_send, _get_chat_ids
        total_pnl = sum(r["pnl"] or 0 for r in results)
        msg = (
            f"Trade Idea Feedback: {len(results)} debriefs processed\n"
            f"Wins: {wins} | Losses: {losses} | Matched to ideas: {matched}\n"
            f"Batch P&L: ${total_pnl:+,.0f}\n"
            f"Cumulative: {state['total_processed']} processed "
            f"({state['wins']}W / {state['losses']}L)"
        )
        if context and hasattr(context, "bot"):
            for cid in _get_chat_ids():
                await _safe_send(context.bot, cid, text=msg)
    except Exception as e:
        logger.warning(f"Failed to send feedback notification: {e}")

    logger.info(
        f"Trade idea feedback: {len(results)} debriefs ({wins}W/{losses}L), "
        f"{matched} matched to ideas"
    )
