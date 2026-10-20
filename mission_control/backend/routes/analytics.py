"""Analytics routes — performance metrics, Sharpe, Sortino, benchmarks, fun facts."""

import json
import logging
import math
import os
import random
import time
from datetime import datetime

import numpy as np
import httpx
from fastapi import APIRouter, Depends, Query

from mission_control.backend.auth import require_auth
from mission_control.backend.config import mc_config
from mission_control.backend.deps import run_in_executor

router = APIRouter(prefix="/api/analytics", tags=["analytics"])
logger = logging.getLogger("mission_control.analytics")

RF_RATE = 0.04  # Annual risk-free rate (Fed funds)


def _fetch_alpaca_history(period: str = "3M") -> dict:
    """Fetch portfolio history from Alpaca API."""
    key = os.getenv("ALPACA_PAPER_KEY_ID", "")
    secret = os.getenv("ALPACA_PAPER_SECRET_KEY", "")
    base = "https://paper-api.alpaca.markets"
    try:
        resp = httpx.get(
            f"{base}/v2/account/portfolio/history",
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
            params={"period": period, "timeframe": "1D"},
            timeout=15,
            follow_redirects=True,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"Alpaca portfolio history failed: {e}")
        return {}


def _compute_metrics(returns: list, equities: list, dates: list) -> dict:
    """Compute risk-adjusted metrics from daily returns."""
    if len(returns) < 2 or len(equities) < 2:
        return {
            "sharpe": None, "sortino": None, "calmar": None, "max_dd_pct": None,
            "max_dd_peak": None, "max_dd_trough": None,
            "total_return_pct": None, "ann_return_pct": None, "ann_vol_pct": None,
            "best_day_pct": None, "worst_day_pct": None,
            "up_days": sum(1 for value in returns if value > 0),
            "down_days": sum(1 for value in returns if value < 0),
            "avg_daily_pct": None,
            "start_equity": round(equities[0], 2) if equities else None,
            "end_equity": round(equities[-1], 2) if equities else None,
            "days": len(returns),
            "source": "alpaca_portfolio_history",
            "as_of": dates[-1] if dates else None,
            "stale": True,
            "reason_code": "INSUFFICIENT_HISTORY",
        }

    r = np.array(returns)
    rf_daily = RF_RATE / 252
    excess = r - rf_daily

    # Sharpe
    std = np.std(excess, ddof=1)
    sharpe = float((np.mean(excess) / std) * math.sqrt(252)) if std > 0 else None

    # Sortino
    downside = excess[excess < 0]
    down_std = np.std(downside, ddof=1) if len(downside) > 0 else 0
    sortino = float((np.mean(excess) / down_std) * math.sqrt(252)) if down_std > 0 else None

    # Max drawdown
    peak = equities[0]
    max_dd = 0
    dd_peak_date = dates[0] if dates else ""
    dd_trough_date = dates[0] if dates else ""
    for i, eq in enumerate(equities):
        if eq > peak:
            peak = eq
            dd_peak_date = dates[i] if i < len(dates) else ""
        dd = (peak - eq) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
            dd_trough_date = dates[i] if i < len(dates) else ""

    # Calmar
    total_ret = equities[-1] / equities[0] - 1 if equities[0] > 0 else 0
    n = len(returns)
    ann_ret = (1 + total_ret) ** (252 / max(n, 1)) - 1
    calmar = ann_ret / max_dd if max_dd > 0 else None

    return {
        "sharpe": round(sharpe, 2) if sharpe is not None else None,
        "sortino": round(sortino, 2) if sortino is not None else None,
        "calmar": round(calmar, 1) if calmar is not None else None,
        "max_dd_pct": round(max_dd * 100, 1),
        "max_dd_peak": dd_peak_date,
        "max_dd_trough": dd_trough_date,
        "total_return_pct": round(total_ret * 100, 1),
        "ann_return_pct": round(ann_ret * 100, 1),
        "ann_vol_pct": round(float(np.std(r, ddof=1) * math.sqrt(252) * 100), 1),
        "best_day_pct": round(float(max(r) * 100), 2),
        "worst_day_pct": round(float(min(r) * 100), 2),
        "up_days": int(sum(1 for x in r if x > 0)),
        "down_days": int(sum(1 for x in r if x < 0)),
        "avg_daily_pct": round(float(np.mean(r) * 100), 3),
        "start_equity": round(equities[0], 2),
        "end_equity": round(equities[-1], 2),
        "days": n,
        "source": "alpaca_portfolio_history",
        "as_of": dates[-1] if dates else None,
        "stale": False,
        "reason_code": "FRESH",
    }


# Industry benchmarks (approximate historical Sharpe ratios)
BENCHMARKS = [
    {"name": "CLIO (Alpaca paper)", "sharpe": 0, "sortino": 0, "color": "#c0a050", "ours": True},
    {"name": "Renaissance Medallion*", "sharpe": 2.80, "sortino": 5.20, "color": "#9b59b6"},
    {"name": "Two Sigma Absolute Return", "sharpe": 2.10, "sortino": 3.80, "color": "#3498db"},
    {"name": "Citadel Wellington", "sharpe": 1.90, "sortino": 3.20, "color": "#2ecc71"},
    {"name": "Bridgewater Pure Alpha", "sharpe": 1.40, "sortino": 2.30, "color": "#e67e22"},
    {"name": "S&P 500 (10yr avg)", "sharpe": 0.90, "sortino": 1.20, "color": "#808080"},
    {"name": "60/40 Portfolio", "sharpe": 0.50, "sortino": 0.70, "color": "#555555"},
]

FUN_FACTS = [
    "Hydrinia (the old 12-agent trading system) is permanently retired as of Sep 24 2026 — CLIO now trades natively.",
    "CLIO's DPOF engine trades long options only (calls and puts) on SPY/QQQ — no shorting, no leverage.",
    "DPOF's entry signal is a 4-pillar composite: NOFA, options premium flow, dark-pool prints, and GEX regime.",
    "A dark-pool reading that contradicts the composite beyond threshold triggers a deterministic veto — no override.",
    "DPOF requires price confirmation (VWAP side + 15-minute momentum) to agree with the flow signal before entering.",
    "Every DPOF exit is deterministic — stop-loss, profit tiers, breakeven-after-tier-1, or thesis invalidation. Jev (TypeSafe) is never consulted on exits.",
    "Jev, DPOF's optional fast-veto layer, can block an entry but can never initiate, upsize, or gate an exit.",
    "DPOF flattens every open position by 3:50 PM ET on a normal session — no overnight option risk.",
    "An operator HALT file (or Telegram '/dpof halt') blocks new DPOF entries from the next cycle without touching open positions.",
    "DPOF's position sizing scales with signal conviction, from 1% to 3% of equity at risk per trade.",
    "Daily and weekly loss limits, plus a max-drawdown circuit breaker off the high-water mark, can hard-halt DPOF — only an operator '/dpof resume' clears it.",
    "CLIO trades on its own Alpaca paper account — 'paper mode IS the soak,' per operator directive; there's no separate shadow period.",
    "DPOF's ledger writes are atomic (temp file + fsync + os.replace) so any reader always sees a complete positions.json.",
    "CLIO's Mission Control dashboard and the DPOF trading engine run as separate processes, talking only through the read-only DPOF ledger.",
]

# --------------------------------------------------------------------------- #
# Fun facts loader — shared file with fallback to hardcoded list
# --------------------------------------------------------------------------- #
_FUN_FACTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "data", "intelligence", "analytics_fun_facts.json",
)
_fun_facts_cache: list[str] = []
_fun_facts_cache_time: float = 0.0
_FUN_FACTS_CACHE_TTL = 300  # 5 minutes


def _load_fun_facts() -> list[str]:
    """Load fun facts from shared JSON file, fallback to hardcoded FUN_FACTS."""
    global _fun_facts_cache, _fun_facts_cache_time  # noqa: PLW0603
    now = time.time()
    if _fun_facts_cache and (now - _fun_facts_cache_time) < _FUN_FACTS_CACHE_TTL:
        return _fun_facts_cache

    try:
        if os.path.exists(_FUN_FACTS_PATH):
            mtime = os.path.getmtime(_FUN_FACTS_PATH)
            # Reject if >48h stale
            if (now - mtime) > 48 * 3600:
                logger.debug("Fun facts file >48h stale, using hardcoded fallback")
                _fun_facts_cache = list(FUN_FACTS)
                _fun_facts_cache_time = now
                return _fun_facts_cache

            with open(_FUN_FACTS_PATH, "r") as f:
                data = json.load(f)
            facts = data.get("facts", [])
            if facts and isinstance(facts, list) and len(facts) >= 10:
                _fun_facts_cache = facts
                _fun_facts_cache_time = now
                return _fun_facts_cache
    except Exception as e:
        logger.debug(f"Failed to load fun facts file: {e}")

    _fun_facts_cache = list(FUN_FACTS)
    _fun_facts_cache_time = now
    return _fun_facts_cache


@router.get("")
async def get_analytics(_=Depends(require_auth)):
    """Full performance analytics with metrics, benchmarks, and fun facts."""
    raw = await run_in_executor(_fetch_alpaca_history, "3M")
    timestamps = raw.get("timestamp", [])
    equities = raw.get("equity", [])
    profit_loss_pct = raw.get("profit_loss_pct", [])

    if not timestamps or not equities:
        return {"error": "No portfolio history available"}

    dates = [datetime.fromtimestamp(ts).strftime("%Y-%m-%d") for ts in timestamps]

    # Build equity curve for chart
    equity_curve = [{"date": d, "equity": round(eq, 2)} for d, eq in zip(dates, equities)]

    # Filter to active trading period (skip flat $100K days)
    active_idx = 0
    for i, eq in enumerate(equities):
        if i > 0 and abs(equities[i] - equities[i - 1]) > 1:
            active_idx = i
            break

    active_equities = equities[active_idx:]
    active_dates = dates[active_idx:]
    active_returns = [profit_loss_pct[i] for i in range(active_idx, len(profit_loss_pct))]

    # Full period metrics
    full_metrics = _compute_metrics(active_returns, active_equities, active_dates)

    # Phase breakdown — find transition points
    # Pre-live: first trade → Mar 9
    # Post-live: Mar 10 → latest
    # CRISIS: Mar 12 → latest
    phases = {}
    for label, start_date, end_date in [
        ("Pre-Live (Feb 5 \u2192 Mar 9)", "2026-02-05", "2026-03-09"),
        ("Post-Live (Mar 10+)", "2026-03-10", "2099-12-31"),
        ("CRISIS Regime (Mar 12+)", "2026-03-12", "2099-12-31"),
    ]:
        idx = [(i, d, equities[i], profit_loss_pct[i])
               for i, d in enumerate(dates) if start_date <= d <= end_date]
        if len(idx) >= 2:
            p_eq = [x[2] for x in idx]
            p_ret = [x[3] for x in idx]
            p_dates = [x[1] for x in idx]
            phases[label] = _compute_metrics(p_ret, p_eq, p_dates)

    # Build benchmarks with our actual Sharpe
    benchmarks = []
    for b in BENCHMARKS:
        entry = dict(b)
        if entry.get("ours"):
            entry["sharpe"] = full_metrics["sharpe"]
            entry["sortino"] = full_metrics["sortino"]
        benchmarks.append(entry)
    benchmarks.sort(
        key=lambda x: x["sharpe"] if isinstance(x.get("sharpe"), (int, float)) else float("-inf"),
        reverse=True,
    )

    # Pick 10 random fun facts (client rotates through them)
    all_facts = _load_fun_facts()
    facts = random.sample(all_facts, min(10, len(all_facts)))

    return {
        "equity_curve": equity_curve,
        "metrics": full_metrics,
        "phases": phases,
        "benchmarks": benchmarks,
        "fun_facts": facts,
        "total_fun_facts": len(all_facts),
        "updated": datetime.now().isoformat(),
    }
