"""
OpenClaw/Moltbot — Hourly Trade Debrief (Session 85)

Runs every hour from 9:30 AM to 4:30 PM ET on weekdays.
Produces a concise summary of all trades within that hour, delivered to
Telegram (+ the Hydrinia file bridge, only if explicitly re-enabled via
CLIO_HYDRINIA_BRIDGE_ENABLED — see agent/hydrinia_bridge.py).

This describes CLIO's own DPOF book (SPY/QQQ long options on the CLIO Alpaca
paper account, agent/dpof/) — never Hydrinia's frozen state files.

Data sources (priority order):
  1. Alpaca orders — all filled orders in the hour window (trims, opens, closes)
  2. Alpaca positions — live unrealized P&L, current prices
  3. Alpaca account — today's total P&L
  4. agent/dpof/readmodel.py — direction/structure context, matched by OCC
     contract symbol against Alpaca's order/position symbols
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import anthropic

from agent.config import config
from agent.integrations.alpaca_values import (
    AlpacaValueError,
    normalize_order_side,
    normalize_position_side,
)
from agent.prompts.system_prompt import cached_system_block
from agent.rate_limiter import budget_tracker
from agent.security import sanitize_ingested_content


def _get_athena():
    """Lazy import for Athena bridge (Phase 1 integration)."""
    try:
        from agent.integrations.athena_bridge import get_athena_bridge
        return get_athena_bridge()
    except Exception:
        return None


# Previous market tide for shift detection (module-level, survives across runs)
# Only mutated from the hourly debrief scheduled task (single event loop, no threading race)
_prev_tide: dict | None = None

logger = logging.getLogger("openclaw.hourly_debrief")

_NY = ZoneInfo("America/New_York")

REPORTS_DIR = os.path.join(config.tradingbot_dir, "moltbot_knowledge", "reports", "hourly")

_cached_client: anthropic.Anthropic | None = None
_client_lock = __import__("threading").Lock()


def _get_client() -> anthropic.Anthropic:
    global _cached_client
    if _cached_client is None:
        with _client_lock:
            if _cached_client is None:
                _cached_client = anthropic.Anthropic(**config.anthropic_client_kwargs(timeout=120))
    return _cached_client


HOURLY_SYSTEM_PROMPT = """You are a concise trading desk analyst giving an hourly portfolio update.
Be direct, insightful, and waste no words. Every sentence must deliver value.

Write in plain text only — NO markdown formatting, NO headers with #, NO bold/italic markers.

Format your update exactly like this:

HOUR: {time_range}
TRADES: {total order count} orders ({buys} buys, {sells} sells)

[For each significant order or group of orders:]
  {BUY/SELL}: {TICKER} {qty} contracts @ ${avg_price} | {context if available}

[For full position exits, note them explicitly:]
  EXIT: {TICKER} | P&L: ${amount} | held {duration}

ACTIVE POSITIONS: {count}
NET: unrealized ${amount} | today's total P&L ${amount}

INSIGHT: 1-2 sentences with a pattern observation, risk note, or market read. If FLOW SHIFT data is present, incorporate it — note whether options flow confirms or contradicts the current portfolio bias.

If no orders this hour: give a brief "quiet hour" summary with current position status and any notable price movements."""


# ============================================================
# Data Loading
# ============================================================

def _load_position_context() -> dict[str, dict]:
    """DPOF context (direction/structure/id) keyed by OCC leg symbol.

    Alpaca's option order/position `symbol` field IS the OCC contract symbol,
    so matching on it directly ties each fill/position back to the DPOF
    position that owns it. CLIO-native (agent/dpof/readmodel.py) — replaces
    the old position_log.json (Hydrinia) lookup.
    """
    context: dict[str, dict] = {}
    try:
        from agent.dpof import readmodel
        for p in readmodel.open_positions():
            for leg in (p.legs or []):
                sym = leg.get("symbol") if isinstance(leg, dict) else None
                if sym:
                    context[sym] = {
                        "direction": p.direction,
                        "structure": p.structure,
                        "underlying": p.underlying,
                        "dpof_id": p.id,
                        "entry_price": p.entry_price or 0,
                    }
    except Exception as e:
        logger.debug(f"DPOF readmodel unavailable for position context: {e}")
    return context


async def _get_alpaca_data() -> tuple[list[dict], list[dict], dict]:
    """Fetch Alpaca orders (last 65 min), positions, and account. Returns (orders, positions, account)."""
    try:
        from agent.integrations.alpaca import _get_trading_client, _retry_alpaca_call
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        client = _get_trading_client()

        # Recent filled orders (last 65 min)
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=65)
        order_req = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            after=cutoff,
            limit=50,
        )
        raw_orders = await _retry_alpaca_call(lambda: client.get_orders(filter=order_req)) or []

        def _sf(obj, attr):
            """Safe float extraction from Alpaca objects."""
            try:
                return float(getattr(obj, attr, 0) or 0)
            except (ValueError, TypeError):
                return 0.0

        orders = []
        for o in raw_orders:
            status = str(getattr(o, "status", "")).lower()
            if "filled" not in status:
                continue
            try:
                side = normalize_order_side(getattr(o, "side", None))
            except AlpacaValueError:
                logger.warning("Hourly debrief skipped order with unknown side: %s", getattr(o, "id", ""))
                continue
            orders.append({
                "symbol": getattr(o, "symbol", ""),
                "side": side,
                "qty": _sf(o, "filled_qty"),
                "avg_price": _sf(o, "filled_avg_price"),
                "filled_at": str(getattr(o, "filled_at", "")),
            })

        # Current positions
        raw_positions = await _retry_alpaca_call(client.get_all_positions) or []
        positions = []
        for pos in raw_positions:
            try:
                side = normalize_position_side(getattr(pos, "side", None))
            except AlpacaValueError:
                logger.warning("Hourly debrief skipped position with unknown side: %s", getattr(pos, "symbol", ""))
                continue
            positions.append({
                "symbol": getattr(pos, "symbol", ""),
                "qty": _sf(pos, "qty"),
                "side": side,
                "unrealized_pl": _sf(pos, "unrealized_pl"),
                "current_price": _sf(pos, "current_price"),
                "avg_entry_price": _sf(pos, "avg_entry_price"),
                "change_today": _sf(pos, "change_today"),
                "market_value": _sf(pos, "market_value"),
            })

        # Account summary
        raw_account = await _retry_alpaca_call(client.get_account)
        account = {}
        if raw_account:
            account = {
                "equity": _sf(raw_account, "equity"),
                "cash": _sf(raw_account, "cash"),
                "pnl_today": _sf(raw_account, "equity") - _sf(raw_account, "last_equity"),
            }

        return orders, positions, account

    except Exception as e:
        logger.error(f"Alpaca data fetch failed: {e}", exc_info=True)
        return [], [], {}


# ============================================================
# Data Formatting
# ============================================================

def _get_flow_shift() -> str:
    """Compare current market tide with previous hour's to detect flow regime shifts."""
    global _prev_tide
    try:
        from agent.integrations.uw_flow_intel import get_cached_tide
        current = get_cached_tide()
        if not current:
            return ""

        shift_parts = []
        if _prev_tide:
            prev_net = _prev_tide.get("net_flow", 0)
            curr_net = current.get("net_flow", 0)
            prev_ratio = _prev_tide.get("ratio", 1.0)
            curr_ratio = current.get("ratio", 1.0)

            # Detect significant premium shift (>$5M swing)
            delta = curr_net - prev_net
            if abs(delta) > 5_000_000:
                direction = "bullish" if delta > 0 else "bearish"
                shift_parts.append(f"Net premium shifted ${delta:+,.0f} ({direction})")

            # Detect call/put ratio flip (ratio > 1.0 = calls dominate, < 1.0 = puts dominate)
            if (prev_ratio < 1.0 and curr_ratio >= 1.0) or (prev_ratio >= 1.0 and curr_ratio < 1.0):
                label = "calls dominating" if curr_ratio >= 1.0 else "puts dominating"
                shift_parts.append(f"Call/put ratio flipped to {curr_ratio:.2f} ({label})")

        _prev_tide = dict(current)  # Store for next hour
        return " | ".join(shift_parts) if shift_parts else ""
    except Exception as e:
        logger.debug(f"Flow shift detection failed: {e}")
        return ""


def _get_position_forecasts(positions: list[dict]) -> str:
    """
    Fetch Athena vault forecasts for open positions.
    Returns formatted string or empty string if unavailable.
    """
    try:
        if os.environ.get("CLIO_ATHENA_ENABLED", "0") != "1":
            return ""

        athena = _get_athena()
        if athena is None or not athena.is_available():
            return ""

        forecast_lines = ["ATHENA POSITION FORECASTS (TimesFM vault):"]

        for pos in positions[:5]:
            ticker = pos.get("symbol", "")
            if not ticker:
                continue
            forecast = athena.forecast_ticker(ticker)
            if forecast is None:
                continue

            label = forecast.get("directional_label", "NEUTRAL")
            alignment = forecast.get("directional_alignment", 0.0)
            implied_vol = forecast.get("implied_vol", 0.0)
            scenarios = forecast.get("scenarios", {})
            worst = scenarios.get("worst", 0.0)
            best = scenarios.get("best", 0.0)

            if isinstance(alignment, (int, float)) and isinstance(implied_vol, (int, float)):
                line = f"  {ticker}: {label} (alignment={alignment:+.2f}, IV={implied_vol:.1%}"
                if isinstance(worst, (int, float)) and isinstance(best, (int, float)):
                    line += f", worst={worst:+.1%}, best={best:+.1%}"
                line += ")"
                forecast_lines.append(line)

        if len(forecast_lines) <= 1:
            return ""

        forecast_text = "\n".join(forecast_lines)

        if os.environ.get("CLIO_ATHENA_LOG_ONLY", "1") == "1":
            logger.info("[ATHENA LOG-ONLY] Position forecasts:\n%s", forecast_text)
            return ""  # Don't inject in log-only mode

        return forecast_text

    except Exception as e:
        logger.debug("_get_position_forecasts() failed: %s", e)
        return ""


def _dpof_label(symbol: str) -> str:
    """OCC contract symbol -> readable label (e.g. 'SPY 2026-09-25 580C'); passes through anything else."""
    try:
        from agent.dpof.readmodel import contract_label
        return contract_label(symbol)
    except Exception:
        return symbol


def _format_trade_data(orders: list[dict], positions: list[dict],
                       account: dict, pos_context: dict[str, dict]) -> str:
    """Format all data for the Claude prompt."""
    now = datetime.now(_NY)
    hour_start = (now - timedelta(hours=1)).strftime("%I:%M %p")
    hour_end = now.strftime("%I:%M %p ET")
    time_range = f"{hour_start} - {hour_end}"

    lines = [f"Time range: {time_range}\n"]

    # Orders this hour
    buys = [o for o in orders if "buy" in o["side"].lower()]
    sells = [o for o in orders if "sell" in o["side"].lower()]
    lines.append(f"ORDERS THIS HOUR: {len(orders)} total ({len(buys)} buys, {len(sells)} sells)")

    if orders:
        # Group orders by symbol for readability
        by_symbol: dict[str, list[dict]] = {}
        for o in orders:
            by_symbol.setdefault(o["symbol"], []).append(o)

        for symbol, sym_orders in sorted(by_symbol.items()):
            ctx = pos_context.get(symbol, {})
            direction = ctx.get("direction", "")
            structure = ctx.get("structure", "")
            ctx_str = f" | {direction} {structure}".strip(" |") if (direction or structure) else ""
            label = _dpof_label(symbol)

            for o in sym_orders:
                side = o["side"].upper()
                qty = o["qty"]
                price = o["avg_price"]
                lines.append(f"  {side}: {label} {qty:.0f} contracts @ ${price:.2f}{ctx_str}")
    else:
        lines.append("  No orders this hour")

    # Active positions with live data
    lines.append(f"\nACTIVE POSITIONS ({len(positions)}, CLIO's DPOF book):")
    total_unrealized = 0
    for pos in positions:
        symbol = pos["symbol"]
        qty = pos["qty"]
        side = "SHORT" if qty < 0 else "LONG"
        unrealized = pos["unrealized_pl"]
        price = pos["current_price"]
        change = pos["change_today"] * 100
        total_unrealized += unrealized

        ctx = pos_context.get(symbol, {})
        direction = ctx.get("direction", "")
        structure = ctx.get("structure", "")
        ctx_str = f" | {direction} {structure}".strip(" |") if (direction or structure) else ""
        label = _dpof_label(symbol)

        lines.append(
            f"  {label} {side} {abs(qty):.0f} contracts @ ${price:.2f} "
            f"({change:+.1f}% today) | unrealized: ${unrealized:+,.2f}{ctx_str}"
        )

    # Account-level P&L
    pnl_today = account.get("pnl_today", 0)
    equity = account.get("equity", 0)
    lines.append(f"\nPORTFOLIO: equity ${equity:,.2f} | today's P&L: ${pnl_today:+,.2f}")
    lines.append(f"NET UNREALIZED: ${total_unrealized:+,.2f}")

    # Options flow shift detection (fire-and-forget)
    flow_shift = _get_flow_shift()
    if flow_shift:
        lines.append(f"\nFLOW SHIFT: {flow_shift}")

    return "\n".join(lines)


# ============================================================
# Report Generation & Delivery
# ============================================================

async def _generate_debrief(trade_data: str) -> str | None:
    """Call Claude Sonnet to generate the hourly debrief."""
    allowed, msg = await budget_tracker.check_budget("hourly_debrief")
    if not allowed:
        logger.warning(f"Hourly debrief budget exhausted: {msg}")
        return None

    try:
        client = _get_client()
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, lambda: client.messages.create(
            model=config.fast_model,  # Sonnet for concise
            max_tokens=2048,
            system=cached_system_block(HOURLY_SYSTEM_PROMPT),
            messages=[{"role": "user", "content": trade_data}],
            output_config={"effort": "medium"},
        ))

        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

        result = text.strip() if text.strip() else None
        if result:
            await budget_tracker.record_usage("hourly_debrief")
        return result

    except Exception as e:
        logger.error(f"Hourly debrief Claude call failed: {e}", exc_info=True)
        return None


def _write_to_bridge(report_text: str, order_count: int, position_count: int) -> None:
    """Write report to the Hydrinia file bridge — no-op unless explicitly re-enabled."""
    try:
        from agent.hydrinia_bridge import bridge_enabled
        if not bridge_enabled():
            return
    except Exception as e:
        logger.debug(f"Hydrinia bridge check failed, skipping bridge write: {e}")
        return
    try:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        timestamp = datetime.now(_NY).strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_hourly.json"

        entry = {
            "id": f"report_{timestamp}_hourly",
            "type": "hourly_debrief",
            "created": datetime.now(_NY).isoformat(),
            "content": report_text,
            "orders_this_hour": order_count,
            "active_positions": position_count,
        }

        file_path = os.path.join(REPORTS_DIR, filename)
        tmp = file_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(entry, f, indent=2, default=str)
        os.replace(tmp, file_path)

        logger.info(f"Hourly debrief written to bridge: {filename}")
    except Exception as e:
        logger.error(f"Failed to write hourly debrief to bridge: {e}")


async def _send_report(context, report_text: str) -> None:
    """Send report to Telegram."""
    try:
        from agent.integrations.scheduled_tasks import _get_chat_ids, _safe_send
        chat_ids = _get_chat_ids()
        if not chat_ids:
            return

        header = f"HOURLY DEBRIEF — {datetime.now(_NY).strftime('%I:%M %p ET')}\n\n"
        full_text = header + report_text

        if len(full_text) <= 4000:
            for cid in chat_ids:
                await _safe_send(context.bot, cid, text=full_text)
        else:
            chunks = [full_text[i:i + 4000] for i in range(0, len(full_text), 4000)]
            for cid in chat_ids:
                for chunk in chunks:
                    await _safe_send(context.bot, cid, text=chunk)

    except Exception as e:
        logger.warning(f"Failed to send hourly debrief to Telegram: {e}")


# ============================================================
# Main Entry Point
# ============================================================

async def run_hourly_debrief(context) -> None:
    """Main entry point: concise hourly trade debrief.

    Called by JobQueue every hour. Skips weekends and outside market hours.
    Active window: 9:30 AM - 4:30 PM ET weekdays.
    """
    now = datetime.now(_NY)

    # Skip weekends
    if now.weekday() >= 5:
        return

    # Skip outside market hours (9:30 AM - 4:30 PM ET)
    if now.hour < 9 or (now.hour == 9 and now.minute < 30):
        return
    if now.hour > 16 or (now.hour == 16 and now.minute > 30):
        return

    logger.info("Starting hourly debrief...")

    try:
        pos_context = _load_position_context()
        orders, positions, account = await _get_alpaca_data()
        trade_data = _format_trade_data(orders, positions, account, pos_context)

        # Athena position forecasts (Phase 1 — fire-and-forget)
        athena_forecasts = _get_position_forecasts(positions)
        if athena_forecasts:
            trade_data += f"\n\n{athena_forecasts}"

        trade_data = sanitize_ingested_content(trade_data, source="alpaca_positions")

        report = await _generate_debrief(trade_data)
        if not report:
            logger.warning("Hourly debrief: no report generated (budget exhausted or API error)")
            return

        _write_to_bridge(report, len(orders), len(positions))
        await _send_report(context, report)

        logger.info(
            f"Hourly debrief complete: {len(orders)} orders, "
            f"{len(positions)} positions, P&L today: ${account.get('pnl_today', 0):+,.2f}"
        )

    except Exception as e:
        logger.error(f"Hourly debrief failed: {e}", exc_info=True)
