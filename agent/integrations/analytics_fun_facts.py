"""
Analytics Fun Facts Generator — reads CLIO's own DPOF book + Alpaca account and
produces a mix of static architecture facts + dynamic data-driven facts.

Repointed off Hydrinia (2026-09-24): Hydrinia (/home/agent/tradingbot) is
permanently off, so this module no longer reads its state/ files. Dynamic facts
now come from agent/dpof/readmodel.py (CLIO's own trade book) and CLIO's own
Alpaca paper account (agent/integrations/alpaca.py) — the same account CLIO
trades on.

Output: /home/agent/openclaw/data/intelligence/analytics_fun_facts.json
Consumed by: mission_control/backend/routes/analytics.py
Schedule: 8:05 PM ET every market day (via scheduled_tasks.py)
"""

import json
import logging
import os
import tempfile
from datetime import datetime, timezone

logger = logging.getLogger("openclaw.analytics_fun_facts")

_OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "intelligence", "analytics_fun_facts.json",
)

# --------------------------------------------------------------------------- #
# Static architecture facts — CLIO-native (2026-09-24). Kept few and factual;
# no invented counts/numbers.
# --------------------------------------------------------------------------- #

STATIC_FACTS = [
    "CLIO's DPOF engine trades SPY and QQQ long options — single calls/puts and debit verticals, no complex spreads.",
    "Every DPOF entry passes through Jev (TypeSafe), an LLM-based veto layer that can block — but never force — an entry.",
    "DPOF flattens all open positions ahead of the close, by 3:50 PM ET.",
    "DPOF's signal pillars combine NOFA, premium, dark pool, and GEX regime from Unusual Whales flow data.",
    "The operator HALT file lets Mission Control stop new DPOF entries from its own process — the engine checks it every cycle.",
    "The autonomous knowledge learning pipeline runs web-research cycles roughly every 15 minutes, gated by a quality score.",
    "News intelligence covers 10 structured topics plus 6 deep-analysis modules: geopolitical, Fed, earnings, commodity, bond, and trade policy.",
    "CLIO trades on its own Alpaca paper account — the same account this dashboard's portfolio and P&L figures come from.",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _safe_float(val, default: float = 0.0) -> float:
    """Safely coerce to float."""
    if isinstance(val, (int, float)):
        return float(val)
    return default


def _safe_int(val, default: int = 0) -> int:
    """Safely coerce to int."""
    if isinstance(val, (int, float)):
        return int(val)
    return default


# --------------------------------------------------------------------------- #
# Dynamic fact generation
# --------------------------------------------------------------------------- #

async def _generate_dynamic_facts() -> list[str]:
    """Generate data-driven fun facts from CLIO's own DPOF book + Alpaca account."""
    facts: list[str] = []

    # --- DPOF book (agent/dpof/readmodel.py) ---
    try:
        from agent.dpof import readmodel
        snap = readmodel.snapshot()

        open_count = _safe_int(snap.get("open_count"))
        if open_count > 0:
            facts.append(
                f"Currently holding {open_count} open DPOF position(s) across SPY/QQQ options."
            )

        closed_today = _safe_int(snap.get("closed_today_count"))
        if closed_today > 0:
            wins = _safe_int(snap.get("closed_today_wins"))
            realized = _safe_float(snap.get("realized_pnl_today"))
            sign = "+" if realized >= 0 else "-"
            facts.append(
                f"{closed_today} DPOF trade(s) closed today ({wins} win(s)), "
                f"realized {sign}${abs(realized):,.2f}."
            )

        realized_week = _safe_float(snap.get("realized_pnl_week"))
        if realized_week != 0:
            sign = "+" if realized_week >= 0 else "-"
            facts.append(f"DPOF realized P&L this week: {sign}${abs(realized_week):,.2f}.")

        if snap.get("halt_file") or snap.get("hard_halted"):
            reason = snap.get("halt_reason") or "operator halt"
            facts.append(f"DPOF entries are currently HALTED ({reason}).")

        mode = snap.get("mode")
        if mode and mode != "unknown":
            facts.append(f"DPOF is running in {mode} mode.")
    except Exception as e:
        logger.debug(f"DPOF readmodel fact generation skipped: {e}")

    # --- CLIO's own Alpaca paper account (equity + daily P&L) ---
    try:
        from agent.integrations.alpaca import _get_trading_client, _retry_alpaca_call

        client = _get_trading_client()
        account = await _retry_alpaca_call(client.get_account)
        # Alpaca SDK account fields (equity/last_equity) come back as STRINGS, not
        # floats — _safe_float() (built for already-parsed JSON) would silently
        # zero them out. Coerce via float() directly, same pattern as the sibling
        # `_sf()` helper in hourly_debrief.py.
        def _alpaca_float(val, default=0.0):
            try:
                return float(val) if val is not None else default
            except (TypeError, ValueError):
                return default

        equity = _alpaca_float(getattr(account, "equity", None))
        last_equity = _alpaca_float(getattr(account, "last_equity", None))
        if equity > 0:
            facts.append(f"Current CLIO paper account equity: ${equity:,.0f}.")
        if equity > 0 and last_equity > 0:
            daily_pnl = equity - last_equity
            if daily_pnl != 0:
                sign = "+" if daily_pnl > 0 else "-"
                facts.append(f"Today's account P&L: {sign}${abs(daily_pnl):,.2f}.")
    except Exception as e:
        logger.debug(f"Alpaca account fact generation skipped: {e}")

    return facts


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

async def generate_fun_facts() -> list[str]:
    """Generate combined static + dynamic fun facts list."""
    dynamic = await _generate_dynamic_facts()
    # Combine: all static + all dynamic
    combined = list(STATIC_FACTS) + dynamic
    return combined


async def refresh_analytics_fun_facts(context) -> None:
    """Scheduled job entry point — generate and write fun facts to shared JSON file."""
    try:
        facts = await generate_fun_facts()
        static_count = len(STATIC_FACTS)
        dynamic_count = len(facts) - static_count

        payload = {
            "facts": facts,
            "updated": datetime.now(timezone.utc).isoformat(),
            "static_count": static_count,
            "dynamic_count": dynamic_count,
        }

        # Atomic write
        os.makedirs(os.path.dirname(_OUTPUT_PATH), exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=os.path.dirname(_OUTPUT_PATH), suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, _OUTPUT_PATH)
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        logger.info(
            f"Analytics fun facts refreshed: {len(facts)} total "
            f"({static_count} static, {dynamic_count} dynamic)"
        )
    except Exception as e:
        logger.error(f"Fun facts refresh failed: {e}", exc_info=True)
        raise
