"""
OpenClaw/Moltbot — Trade Outcome Feedback Loop

Reads Hydrinia's closed positions from position_log.json and adjusts
learning category weights so the engine researches more about areas
where the bot is losing money.

Runs every 2 hours via JobQueue. Zero API budget — reads local JSON only.

S205-B2b: Hydrinia is permanently off (operator direction, 2026-09-24). This
module's symbol -> sector/category mapping (_symbol_to_categories, SYMBOL_SECTOR_MAP)
has no meaningful equivalent for CLIO's own DPOF engine (SPY/QQQ long options only,
no sector rotation), so there is no CLIO-native source to repoint this to.
analyze_trade_feedback() (the scheduled job entry point) early-returns a no-op when
agent.hydrinia_bridge.bridge_enabled() is False -- this IS the gate (the scheduler
registration itself is not conditional; see agent/integrations/scheduled_tasks.py).
"""

import asyncio
import fcntl
import json
import logging
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from agent.config import config

logger = logging.getLogger("openclaw.trade_feedback")

_NY = ZoneInfo("America/New_York")

POSITION_LOG_PATH = os.path.join(config.tradingbot_dir, "state", "position_log.json")

# S189 A2': SEAL feedback freeze visibility — module-level dedup per ADR-003 P1.
# When upstream (Hydrinia / position_log) provides no new closed positions, the
# trade-feedback loop exits silently at DEBUG. Promote to INFO with 6h dedup so
# operator can see the freeze. Pattern justified: 6h window >> ~2 restarts/day.
_FEEDBACK_PAUSED_LOG_INTERVAL_SEC = 6 * 3600
_last_feedback_paused_log_ts = 0.0


def _log_feedback_paused(reason: str, history: list | None = None) -> None:
    """Emit a deduped INFO log when SEAL feedback can't advance.

    `reason` is a short human tag (e.g. "no positions in log"). `history` is the
    `feedback_history` list from learning_state when available, so the log can
    surface when the last successful feedback recording happened.
    """
    global _last_feedback_paused_log_ts
    now = time.time()
    if now - _last_feedback_paused_log_ts < _FEEDBACK_PAUSED_LOG_INTERVAL_SEC:
        logger.debug(f"Trade feedback paused: {reason} (dedup window active)")
        return
    _last_feedback_paused_log_ts = now
    last_fb = ""
    if history:
        try:
            last_fb = f"; last recorded entry: {history[-1].get('timestamp', '?')[:19]}"
        except (IndexError, AttributeError):
            pass
    pos_age = ""
    try:
        if os.path.isfile(POSITION_LOG_PATH):
            age_h = (now - os.path.getmtime(POSITION_LOG_PATH)) / 3600
            pos_age = f"; position_log mtime {age_h:.0f}h ago"
    except OSError:
        pass
    logger.info(
        f"SEAL feedback paused: {reason}{pos_age}{last_fb}. "
        f"Likely upstream (Hydrinia) inactive — bandit weights drifting unverified."
    )


# ============================================================
# Symbol → Sector Mapping
# ============================================================

# Covers Hydrinia's 20-symbol intraday universe + common strategic bank tickers.
# Unknown symbols fall back to "equity".
SYMBOL_SECTOR_MAP = {
    # Technology
    "AAPL": "technology", "MSFT": "technology", "NVDA": "technology",
    "GOOGL": "technology", "GOOG": "technology", "META": "technology",
    "AMZN": "technology", "TSM": "technology", "AVGO": "technology",
    "AMD": "technology", "INTC": "technology", "CRM": "technology",
    "ORCL": "technology", "CSCO": "technology", "QCOM": "technology",
    "MU": "technology", "MRVL": "technology", "ANET": "technology",
    # Financial
    "JPM": "financial", "BAC": "financial", "GS": "financial",
    "MS": "financial", "WFC": "financial", "C": "financial",
    "BLK": "financial", "SCHW": "financial", "COF": "financial",
    # Healthcare
    "JNJ": "healthcare", "UNH": "healthcare", "PFE": "healthcare",
    "ABBV": "healthcare", "LLY": "healthcare", "MRK": "healthcare",
    "BMY": "healthcare", "AMGN": "healthcare", "GILD": "healthcare",
    # Energy
    "XOM": "energy", "CVX": "energy", "COP": "energy",
    "SLB": "energy", "OXY": "energy", "XLE": "energy",
    "USO": "energy", "XOP": "energy",
    # Defense/Aerospace
    "LMT": "defense", "RTX": "defense", "NOC": "defense",
    "BA": "defense", "GD": "defense", "LHX": "defense",
    # Consumer
    "TSLA": "consumer", "NKE": "consumer", "SBUX": "consumer",
    "DIS": "consumer", "HD": "consumer", "WMT": "consumer",
    "COST": "consumer", "TGT": "consumer", "MCD": "consumer",
    # Materials / Mining
    "FCX": "materials", "NEM": "materials", "AA": "materials",
    "X": "materials", "CLF": "materials",
    # Commodities (ETFs)
    "GLD": "commodities_etf", "SLV": "commodities_etf",
    "GDX": "commodities_etf", "GDXJ": "commodities_etf",
    # Crypto proxies
    "COIN": "crypto", "MSTR": "crypto", "BITO": "crypto",
    # Indices / Broad ETFs
    "SPY": "broad_market", "QQQ": "broad_market", "IWM": "broad_market",
    "DIA": "broad_market", "VIX": "broad_market",
    # Bonds ETFs
    "TLT": "bonds_etf", "HYG": "bonds_etf", "LQD": "bonds_etf",
    "SHY": "bonds_etf", "IEF": "bonds_etf",
}

# Sector → learning categories
SECTOR_TO_CATEGORIES = {
    "technology":     ["equity", "trading"],
    "financial":      ["finance", "bonds"],
    "healthcare":     ["equity", "investing"],
    "energy":         ["commodities", "equity"],
    "defense":        ["geopolitical", "us_domestic_political"],
    "consumer":       ["equity", "psychology"],
    "materials":      ["commodities", "supply_chain"],
    "commodities_etf":["commodities", "futures"],
    "crypto":         ["crypto", "trading"],
    "broad_market":   ["equity", "trading", "strategy"],
    "bonds_etf":      ["bonds", "finance"],
}

DEFAULT_CATEGORIES = ["equity", "trading"]


# ============================================================
# Core Logic
# ============================================================

def _symbol_to_categories(symbol: str, is_options: bool) -> list[str]:
    """Map a ticker symbol to relevant learning categories."""
    cats = []
    sector = SYMBOL_SECTOR_MAP.get(symbol.upper())
    if sector:
        cats = list(SECTOR_TO_CATEGORIES.get(sector, DEFAULT_CATEGORIES))
    else:
        cats = list(DEFAULT_CATEGORIES)

    # Options trades always boost the options category
    if is_options and "options" not in cats:
        cats.append("options")

    return cats


def _load_positions() -> dict:
    """Load position_log.json from Hydrinia state dir."""
    if not os.path.isfile(POSITION_LOG_PATH):
        return {}
    lock_path = POSITION_LOG_PATH + ".lock"
    try:
        import fcntl
        with open(lock_path, "a") as lf:
            fcntl.flock(lf, fcntl.LOCK_SH)
            try:
                with open(POSITION_LOG_PATH, "r") as f:
                    data = json.load(f)
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)
        # Handle both {positions: {}} and direct {} formats
        if isinstance(data, dict) and "positions" in data:
            return data["positions"]
        if isinstance(data, dict):
            return data
        return {}
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to read position_log.json: {e}")
        return {}


def calculate_category_weights(
    positions: dict,
    last_feedback_run: str,
    existing_weights: dict[str, float],
) -> tuple[dict[str, float], list[dict]]:
    """
    Calculate learning priority weights from recently closed positions.

    Returns (updated_weights, feedback_entries) where feedback_entries
    is a list of dicts describing what was analyzed (for logging/knowledge save).
    """
    from agent.integrations.learning_engine import LEARNING_CATEGORIES

    # Start from existing weights with time decay (10% toward 1.0)
    weights: dict[str, float] = {}
    for cat in LEARNING_CATEGORIES:
        name = cat["name"]
        old_w = existing_weights.get(name, 1.0)
        weights[name] = old_w + (1.0 - old_w) * 0.10  # Decay toward 1.0

    # Filter to positions closed since last feedback run
    cutoff_ts = 0.0
    if last_feedback_run:
        try:
            cutoff_dt = datetime.fromisoformat(last_feedback_run)
            if cutoff_dt.tzinfo is None:
                cutoff_dt = cutoff_dt.replace(tzinfo=_NY)
            cutoff_ts = cutoff_dt.timestamp()
        except (ValueError, TypeError):
            pass

    feedback_entries = []
    for pos_id, pos in positions.items():
        if pos.get("status") != "closed":
            continue
        close_time = pos.get("close_time")
        if not close_time:
            continue
        # Normalize close_time to float for comparison with cutoff_ts
        if isinstance(close_time, str):
            try:
                close_time = datetime.fromisoformat(close_time)
                if close_time.tzinfo is None:
                    close_time = close_time.replace(tzinfo=_NY)
                close_time = close_time.timestamp()
            except (ValueError, TypeError):
                continue
        if close_time <= cutoff_ts:
            continue

        symbol = pos.get("symbol", "")
        pnl = pos.get("pnl")
        pnl_pct = pos.get("pnl_pct")
        if pnl_pct is None:
            pnl_pct = 0
        is_options = pos.get("is_options", False)

        if pnl is None:
            continue

        related = _symbol_to_categories(symbol, is_options)
        is_loss = pnl < 0

        if is_loss:
            boost = min(abs(pnl_pct) * 5.0, 2.0)
        else:
            boost = min(abs(pnl_pct) * 1.0, 0.5)

        for cat_name in related:
            if cat_name in weights:
                weights[cat_name] += boost

        feedback_entries.append({
            "symbol": symbol,
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct * 100, 1),
            "is_loss": is_loss,
            "boost": round(boost, 3),
            "categories": related,
        })

    # Clamp to [0.5, 2.0]
    for name in weights:
        weights[name] = max(0.5, min(2.0, weights[name]))

    return weights, feedback_entries


# ============================================================
# Perspective Impact Analysis (Feature 1)
# ============================================================

IMPACT_LOG_PATH = os.path.join(config.tradingbot_dir, "state", "moltbot_impact_log.json")
FEEDBACK_FILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "knowledge", "hydrinia_feedback.json",
)
FEEDBACK_LOCK_PATH = FEEDBACK_FILE_PATH + ".lock"

MIN_TRADES_FOR_REGIME_STATS = 5   # Minimum sample before reporting per regime


def analyze_perspective_impact() -> str:
    """Analyze the Moltbot impact log to correlate perspective regimes with outcomes.

    Returns a report string (empty if no data or insufficient samples).
    """
    if not os.path.isfile(IMPACT_LOG_PATH):
        return ""

    try:
        with open(IMPACT_LOG_PATH, "r") as f:
            entries = json.load(f)
    except (json.JSONDecodeError, OSError):
        return ""

    # Filter to entries with outcomes
    closed = [e for e in entries if e.get("outcome") is not None]
    if len(closed) < MIN_TRADES_FOR_REGIME_STATS:
        return ""

    # Group by perspective_regime
    regime_stats: dict[str, dict] = {}
    for e in closed:
        regime = e.get("perspective_regime") or "unknown"
        if regime not in regime_stats:
            regime_stats[regime] = {"wins": 0, "losses": 0, "total_pnl": 0.0}
        if e.get("outcome") == "WIN":
            regime_stats[regime]["wins"] += 1
        else:
            regime_stats[regime]["losses"] += 1
        regime_stats[regime]["total_pnl"] += e.get("pnl", 0) or 0

    # Build report for regimes with sufficient samples
    lines = [f"Perspective Impact ({len(closed)} closed trades):"]
    for regime, stats in sorted(regime_stats.items()):
        total = stats["wins"] + stats["losses"]
        if total < MIN_TRADES_FOR_REGIME_STATS:
            continue
        win_rate = stats["wins"] / total * 100
        avg_pnl = stats["total_pnl"] / total
        lines.append(
            f"  '{regime}': {win_rate:.0f}% win rate ({total} trades), "
            f"avg P&L ${avg_pnl:+.0f}"
        )

    return "\n".join(lines) if len(lines) > 1 else ""


# ============================================================
# Loss-Concentration Detection (EN5)
# ============================================================

def detect_loss_concentration(positions: dict | None = None) -> str:
    """Detect whether losses are concentrated in a few tickers.

    Only considers positions closed within the last 30 days to avoid
    stale warnings repeating identically on every run.

    Flags:
      - Any single ticker accounting for >20% of total losses
      - Top 3 tickers accounting for >50% of total losses

    Returns a warning string (empty if no concentration detected or
    insufficient data).
    """
    if positions is None:
        positions = _load_positions()
    if not positions:
        return ""

    # Only consider positions closed within the last 30 days
    thirty_days_ago = time.time() - (30 * 86400)

    # Gather closed-position losses grouped by symbol
    loss_by_symbol: dict[str, float] = {}
    for pos in positions.values():
        if pos.get("status") != "closed":
            continue
        # Filter by close timestamp (epoch floats in position_log.json)
        close_time = pos.get("close_time")
        if not close_time:
            continue
        if isinstance(close_time, (int, float)) and close_time < thirty_days_ago:
            continue
        if isinstance(close_time, str):
            try:
                ct_dt = datetime.fromisoformat(close_time)
                if ct_dt.tzinfo is None:
                    ct_dt = ct_dt.replace(tzinfo=_NY)
                ct = ct_dt.timestamp()
                if ct < thirty_days_ago:
                    continue
            except (ValueError, TypeError):
                continue
        pnl = pos.get("pnl")
        if pnl is None or pnl >= 0:
            continue
        symbol = pos.get("symbol", "UNKNOWN")
        loss_by_symbol[symbol] = loss_by_symbol.get(symbol, 0.0) + abs(pnl)

    total_loss = sum(loss_by_symbol.values())
    if total_loss <= 0 or len(loss_by_symbol) < 2:
        return ""

    # Sort descending by loss amount
    ranked = sorted(loss_by_symbol.items(), key=lambda x: x[1], reverse=True)

    warnings: list[str] = []

    # Check single-ticker concentration (>20%)
    for symbol, amount in ranked:
        pct = amount / total_loss * 100
        if pct > 20:
            warnings.append(
                f"  {symbol} accounts for {pct:.0f}% of all losses (${amount:.0f})"
            )

    # Check top-3 concentration (>50%)
    top3 = ranked[:3]
    top3_total = sum(amt for _, amt in top3)
    top3_pct = top3_total / total_loss * 100
    if top3_pct > 50:
        top3_names = ", ".join(s for s, _ in top3)
        warnings.append(
            f"  Top 3 ({top3_names}) account for {top3_pct:.0f}% of all losses"
        )

    if not warnings:
        return ""

    header = "Loss Concentration Warning — consider reducing exposure:"
    return header + "\n" + "\n".join(warnings)


# ============================================================
# Usefulness Feedback Processing (Feature 4)
# ============================================================

def process_usefulness_feedback(
    existing_weights: dict[str, float],
) -> tuple[dict[str, float], str]:
    """Read Hydrinia's feedback file, correlate regime/themes with outcomes,
    and apply conservative weight adjustments.

    Returns (adjusted_weights, report_text).
    """
    if not os.path.isfile(FEEDBACK_FILE_PATH):
        return existing_weights, ""

    # Single LOCK_EX for entire read-modify-write (eliminates TOCTOU race — Fix S104-E)
    lock_fd = None
    try:
        lock_fd = open(FEEDBACK_LOCK_PATH, "a")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)

        with open(FEEDBACK_FILE_PATH, "r") as f:
            entries = json.load(f)

        # Filter to unprocessed entries within last 30 days
        cutoff = datetime.now(_NY).timestamp() - 30 * 86400
        relevant = [
            e for e in entries
            if e.get("result") and not e.get("processed_by_moltbot")
            and (e.get("timestamp", 0) or 0) > cutoff
        ]

        if len(relevant) < MIN_TRADES_FOR_REGIME_STATS:
            return existing_weights, ""

        # Aggregate by regime
        regime_stats: dict[str, dict] = {}
        for e in relevant:
            regime = e.get("perspective_regime") or "unknown"
            if regime not in regime_stats:
                regime_stats[regime] = {"wins": 0, "losses": 0}
            if e.get("result") == "WIN":
                regime_stats[regime]["wins"] += 1
            else:
                regime_stats[regime]["losses"] += 1

        weights = dict(existing_weights)
        report_lines: list[str] = ["Usefulness Feedback:"]
        adjusted = False

        for regime, stats in regime_stats.items():
            total = stats["wins"] + stats["losses"]
            if total < MIN_TRADES_FOR_REGIME_STATS:
                continue
            win_rate = stats["wins"] / total

            # Conservative adjustment: ±0.1 max
            # win_rate > 0.6 → slight negative (research is fine, maintain)
            # win_rate < 0.4 → slight positive (research needs improvement)
            adjustment = 0.1 * (0.5 - win_rate)  # Ranges from -0.05 to +0.05

            if abs(adjustment) > 0.01:
                for cat in weights:
                    weights[cat] = max(0.5, min(2.0, weights[cat] + adjustment))
                adjusted = True

            report_lines.append(
                f"  '{regime}': {win_rate:.0%} win rate ({total} trades) → "
                f"weight adj {adjustment:+.3f}"
            )

        # Mark entries as processed and write back (still inside lock scope)
        if adjusted:
            for e in entries:
                if not e.get("processed_by_moltbot") and e.get("result"):
                    e["processed_by_moltbot"] = True
            tmp = FEEDBACK_FILE_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump(entries, f, indent=2, default=str)
            os.replace(tmp, FEEDBACK_FILE_PATH)

        report = "\n".join(report_lines) if len(report_lines) > 1 else ""
        return weights, report

    except (json.JSONDecodeError, OSError) as e:
        logger.debug(f"Feedback processing failed: {e}")
        return existing_weights, ""
    finally:
        if lock_fd:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
            except Exception as e:
                logger.debug(f"Lock release failed: {e}")


# ============================================================
# Scheduled Job
# ============================================================

async def analyze_trade_feedback(context) -> None:
    """
    Scheduled job: analyze Hydrinia's closed trades and adjust learning weights.
    Runs every 2 hours. Zero API budget consumed.

    S205-B2b: Hydrinia is permanently off (operator direction, 2026-09-24) and this
    module has no CLIO-native (DPOF) equivalent to fall back to, so this is a no-op
    unless the bridge is explicitly re-enabled.
    """
    try:
        from agent.hydrinia_bridge import bridge_enabled
        if not bridge_enabled():
            logger.debug("Trade feedback: Hydrinia bridge disabled, skipping (no-op)")
            return
    except Exception as e:
        logger.debug(f"Trade feedback: bridge check failed, skipping ({e})")
        return

    try:
        from agent.integrations.learning_engine import (
            _load_learning_state,
            _save_learning_state,
        )

        positions = _load_positions()
        if not positions:
            # S189 A2': promote to INFO via deduped helper (was silent DEBUG).
            # state hasn't been loaded yet at this point — pass None.
            _log_feedback_paused("no position_log data", None)
            return

        # S233-V2: flock-based I/O off event loop
        _loop = asyncio.get_running_loop()
        state = await _loop.run_in_executor(None, _load_learning_state)

        async def _save_state_async(s):
            await _loop.run_in_executor(None, _save_learning_state, s)
        last_run = state.get("last_feedback_run", "")
        existing_weights = state.get("category_weights", {})

        weights, entries = calculate_category_weights(positions, last_run, existing_weights)

        if not entries:
            # No new closed positions — still apply decay
            state["category_weights"] = weights
            state["last_feedback_run"] = datetime.now(_NY).isoformat()
            await _save_state_async(state)
            # S189 A2': promote to INFO via deduped helper (was silent DEBUG).
            # Decay was applied; visibility is about WHY feedback_history isn't advancing.
            _log_feedback_paused("no new closed positions since last run", state.get("feedback_history"))
            return

        # Persist weights
        state["category_weights"] = weights
        state["last_feedback_run"] = datetime.now(_NY).isoformat()

        # Keep a compact history (last 20 entries)
        history = state.get("feedback_history", [])
        adjustments = {}
        for e in entries:
            for cat in e["categories"]:
                sign = "+" if not e["is_loss"] else "-"
                adjustments[cat] = f"{sign}{e['boost']}"
        history.append({
            "timestamp": datetime.now(_NY).isoformat(),
            "positions_analyzed": len(entries),
            "adjustments": adjustments,
        })
        state["feedback_history"] = history[-20:]

        await _save_state_async(state)

        # Log summary
        non_default = {k: round(v, 2) for k, v in weights.items() if abs(v - 1.0) > 0.05}
        logger.info(
            f"Trade feedback: analyzed {len(entries)} closed positions. "
            f"Adjusted weights: {non_default or 'all near 1.0'}"
        )

        # Save a knowledge entry summarizing the feedback
        try:
            from agent.tools.knowledge_tools import handle_save_knowledge

            symbols = [e["symbol"] for e in entries]
            losses = [e for e in entries if e["is_loss"]]
            wins = [e for e in entries if not e["is_loss"]]

            # Process perspective impact analysis (Feature 1)
            impact_report = ""
            try:
                impact_report = analyze_perspective_impact()
            except Exception as ie:
                logger.debug(f"Perspective impact analysis failed (non-critical): {ie}")

            # Loss-concentration detection (EN5)
            concentration_warning = ""
            try:
                concentration_warning = detect_loss_concentration(positions)
            except Exception as ce:
                logger.debug(f"Loss concentration detection failed (non-critical): {ce}")

            # Process usefulness feedback (Feature 4)
            feedback_report = ""
            try:
                weights, feedback_report = process_usefulness_feedback(weights)
                if feedback_report:
                    # Re-persist adjusted weights
                    state["category_weights"] = weights
                    await _save_state_async(state)
            except Exception as ue:
                logger.debug(f"Usefulness feedback processing failed (non-critical): {ue}")

            content = (
                f"Trade Feedback Analysis — {datetime.now(_NY).strftime('%b %d %Y')}\n\n"
                f"Positions analyzed: {len(entries)}\n"
                f"Wins: {len(wins)}, Losses: {len(losses)}\n"
                f"Symbols: {', '.join(symbols)}\n\n"
                f"Weight adjustments:\n"
                + "\n".join(f"  {k}: {v:.2f}" for k, v in sorted(weights.items()) if abs(v - 1.0) > 0.05)
                + "\n\nSource: Hydrinia position_log.json automated analysis"
            )
            if concentration_warning:
                content += f"\n\n{concentration_warning}"
            if impact_report:
                content += f"\n\n{impact_report}"
            if feedback_report:
                content += f"\n\n{feedback_report}"
            await handle_save_knowledge({
                "title": f"Trade Feedback — {datetime.now(_NY).strftime('%b %d')}",
                "category": "trading_bot_status",
                "content": content,
                "tags": ["trade_feedback", "auto_learning", "weight_adjustment"],
                "source": "trade_feedback_pipeline",
            })
        except Exception as e:
            logger.debug(f"Trade feedback knowledge save failed (non-critical): {e}")

    except Exception as e:
        logger.error(f"Trade feedback analysis failed: {e}", exc_info=True)
