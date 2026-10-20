"""Briefing API — single-call consolidated state for external AI tools."""

import json
import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends

from mission_control.backend.auth import require_auth
from mission_control.backend.deps import get_trading_client, run_in_executor
from agent.integrations.alpaca_values import AlpacaValueError, normalize_position_side

logger = logging.getLogger("mission_control.briefing")
router = APIRouter(prefix="/api/briefing", tags=["briefing"])

_NY = ZoneInfo("America/New_York")

_OPENCLAW = Path("/home/agent/openclaw")
_TRADINGBOT = Path("/home/agent/tradingbot")


def _safe_json(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


@router.get("/full")
async def full_briefing(_=Depends(require_auth)):
    """
    Everything Perplexity Computer needs in one call:
    - Portfolio state (positions, equity, P&L)
    - Market regime and risk
    - Bot health (services, budgets, errors)
    - Active directives
    - Recent intelligence
    - Knowledge stats
    - System resources
    """
    import psutil

    now = datetime.now(_NY)
    dpof_mode = "unknown"
    try:
        from agent.dpof.config import load_dpof_config

        dpof_mode = load_dpof_config().mode.value
    except Exception as e:
        logger.debug(f"Briefing: DPOF mode unavailable: {e}")
    briefing = {
        "timestamp": now.isoformat(),
        "system": "Atlas Strategies / CLIO (Moltbot) — DPOF native trading engine",
        "mode": f"DPOF {dpof_mode.upper()} (Hydrinia retired 2026-09-24)",
    }

    # ---- Portfolio ----
    try:
        client = get_trading_client()
        acct = await run_in_executor(lambda: client.get_account())
        briefing["portfolio"] = {
            "equity": float(acct.equity or 0),
            "cash": float(acct.cash or 0),
            "buying_power": float(acct.buying_power or 0),
            "portfolio_value": float(acct.portfolio_value or 0),
            "daily_pnl": float(acct.equity or 0) - float(acct.last_equity or 0),
        }
    except Exception as e:
        logger.error(f"Briefing portfolio fetch failed: {e}", exc_info=True)
        briefing["portfolio"] = {"error": "Internal server error"}

    # ---- Positions ----
    try:
        client = get_trading_client()
        positions = await run_in_executor(lambda: client.get_all_positions())
        briefing["positions"] = []
        briefing["position_rejections"] = []
        for p in positions:
            try:
                side = normalize_position_side(getattr(p, "side", None))
                briefing["positions"].append({
                    "symbol": p.symbol,
                    "qty": float(p.qty or 0),
                    "side": side,
                    "avg_entry": float(p.avg_entry_price or 0),
                    "current_price": float(p.current_price or 0),
                    "unrealized_pnl": float(p.unrealized_pl or 0),
                    "unrealized_pnl_pct": float(p.unrealized_plpc or 0) * 100,
                    "market_value": float(p.market_value or 0),
                    "asset_class": str(getattr(p, "asset_class", "us_equity")),
                })
            except AlpacaValueError as exc:
                briefing["position_rejections"].append({
                    "symbol": str(getattr(p, "symbol", "")),
                    "reason_code": "UNKNOWN_POSITION_SIDE",
                    "detail": str(exc),
                })
                continue
            except Exception:
                continue
    except Exception as e:
        logger.error(f"Briefing positions fetch failed: {e}", exc_info=True)
        briefing["positions"] = {"error": "Internal server error"}

    # ---- Local file / psutil-derived state (batched off the event loop) ----
    def _gather_local() -> dict:
        out: dict = {}

        # ---- Market regime (CLIO-native; Hydrinia daily_metrics fallback only if bridge enabled) ----
        try:
            from agent.integrations.market_regime import get_market_regime

            native_regime = get_market_regime()
        except Exception as e:
            logger.debug(f"Briefing: market_regime module unavailable: {e}")
            native_regime = None
        if native_regime:
            out["market_regime"] = native_regime
        else:
            out["market_regime"] = {"regime": "UNKNOWN"}
            try:
                from agent.hydrinia_bridge import bridge_enabled
                if bridge_enabled():
                    dm = _safe_json(str(_TRADINGBOT / "state" / "daily_metrics.json"))
                    mr = dm.get("market_regime", "UNKNOWN")
                    out["market_regime"] = mr if isinstance(mr, dict) else {"regime": str(mr)}
            except Exception:
                pass

        # ---- Active directives (Hydrinia voting-influence bridge — bridge-enabled only) ----
        try:
            from agent.hydrinia_bridge import bridge_enabled
            active = []
            if bridge_enabled():
                directives_dir = _TRADINGBOT / "moltbot_knowledge" / "directives"
                if directives_dir.is_dir():
                    for f in directives_dir.iterdir():
                        if f.suffix == ".json" and not f.name.startswith("."):
                            d = _safe_json(str(f))
                            if d:
                                active.append({
                                    "type": d.get("type", "UNKNOWN"),
                                    "command": d.get("command", "")[:100],
                                    "symbol": d.get("symbol"),
                                    "expires_at": d.get("expires_at"),
                                })
            out["active_directives"] = active
        except Exception:
            out["active_directives"] = []

        # ---- Recent intel (last 5) — CLIO's own data/intelligence/ (openclaw-native) ----
        try:
            intel_dir = _OPENCLAW / "data" / "intelligence"
            intel_items = []
            if intel_dir.is_dir():
                files = sorted(intel_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)[:5]
                for f in files:
                    d = _safe_json(str(f))
                    if d:
                        intel_items.append({
                            "topic": d.get("topic", f.stem),
                            "summary": (d.get("summary") or d.get("analysis", ""))[:300],
                            "timestamp": d.get("timestamp", ""),
                        })
            out["recent_intel"] = intel_items
        except Exception:
            out["recent_intel"] = []

        # ---- Knowledge stats ----
        try:
            ki = _safe_json(str(_OPENCLAW / "data" / "knowledge" / "knowledge_index.json"))
            entries = ki.get("entries", [])
            categories = {}
            for e in entries:
                cat = e.get("knowledge_category", "unknown")
                categories[cat] = categories.get(cat, 0) + 1
            out["knowledge"] = {
                "total_entries": len(entries),
                "categories": categories,
            }
        except Exception:
            out["knowledge"] = {}

        # ---- Budgets (top 10 by usage %) ----
        try:
            rl = _safe_json(str(_OPENCLAW / "data" / "rate_limits.json"))
            budgets = []
            for name, info in rl.items():
                if isinstance(info, dict) and "max_requests" in info:
                    mx = max(info.get("max_requests", 1), 1)
                    used = info.get("used", 0)
                    budgets.append({"name": name, "used": used, "max": mx, "pct": round(100 * used / mx, 1)})
            budgets.sort(key=lambda x: x["pct"], reverse=True)
            out["budgets_top10"] = budgets[:10]
        except Exception:
            out["budgets_top10"] = []

        # ---- System health (emergency_stop is CLIO-native DPOF halt state) ----
        try:
            mem = psutil.virtual_memory()
            emergency_stop = False
            try:
                from agent.dpof import readmodel

                risk = readmodel.load_risk()
                emergency_stop = bool(readmodel.halt_file_present() or risk.hard_halted)
            except Exception as e:
                logger.debug(f"Briefing: DPOF halt state unavailable: {e}")
            out["system_health"] = {
                "cpu_percent": psutil.cpu_percent(interval=0.3),
                "memory_percent": mem.percent,
                "disk_percent": psutil.disk_usage("/").percent,
                "emergency_stop": emergency_stop,
            }
        except Exception:
            out["system_health"] = {}

        # ---- Trade ideas (pending) ----
        try:
            ti_dir = _OPENCLAW / "data" / "knowledge" / "trade_ideas"
            if ti_dir.is_dir():
                ideas = []
                for f in sorted(ti_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)[:5]:
                    d = _safe_json(str(f))
                    if d:
                        ideas.append({
                            "ticker": d.get("ticker"),
                            "direction": d.get("direction"),
                            "confidence": d.get("confidence"),
                            "thesis": (d.get("thesis") or "")[:150],
                            "created": d.get("created"),
                        })
                out["pending_trade_ideas"] = ideas
        except Exception:
            out["pending_trade_ideas"] = []

        # ---- Living perspective summary ----
        try:
            lps_path = _OPENCLAW / "data" / "knowledge" / "living_perspective_summary.txt"
            if lps_path.exists():
                out["living_perspective_summary"] = lps_path.read_text()[:2000]
        except Exception:
            pass

        # ---- Current priors ----
        try:
            priors_path = _OPENCLAW / "data" / "knowledge" / "current_priors.md"
            if priors_path.exists():
                out["current_priors"] = priors_path.read_text()[:1500]
        except Exception:
            pass

        return out

    # R6 PORT-WITH-FIXES: each of the 9 sections above has its own inner try/except
    # (fault isolation preserved per-section), but the outer executor call itself was
    # previously unguarded — an exception escaping all 9 inner try blocks (e.g. thread-
    # pool exhaustion) would 500 the whole endpoint instead of degrading section-by-
    # section, unlike the old sequential-inline code this batching replaced. Restore that
    # parity: on a total _gather_local failure, degrade to the same per-section fallback
    # defaults the inner try/excepts already use, rather than raising.
    try:
        briefing.update(await run_in_executor(_gather_local))
    except Exception as e:
        logger.error(f"Briefing local-state gather failed: {e}", exc_info=True)
        briefing.setdefault("market_regime", "UNKNOWN")
        briefing.setdefault("active_directives", [])
        briefing.setdefault("recent_intel", [])
        briefing.setdefault("knowledge", {})
        briefing.setdefault("budgets_top10", [])
        briefing.setdefault("system_health", {})
        briefing.setdefault("pending_trade_ideas", [])

    return briefing
