"""
OpenClaw/Moltbot — Alpaca Trading Integration
Handles: execute_trade, get_portfolio, get_account_info, get_market_data (fallback)
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

_NY = ZoneInfo("America/New_York")

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest, StopLimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest, StockSnapshotRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from agent.config import config
from agent.integrations.alpaca_values import AlpacaValueError, normalize_order_side
from agent.security import sanitize_ingested_content

logger = logging.getLogger("openclaw.integrations.alpaca")

# Initialize clients
_trading_client = None
_data_client = None
_trading_lock = __import__("threading").Lock()
_data_lock = __import__("threading").Lock()


def _get_trading_client() -> TradingClient:
    global _trading_client
    if _trading_client is None:
        with _trading_lock:
            if _trading_client is None:
                _trading_client = TradingClient(
                    api_key=config.alpaca_api_key,
                    secret_key=config.alpaca_secret_key,
                    paper=config.alpaca_paper,
                    url_override=config.alpaca_base_url,
                )
    return _trading_client


def _get_data_client() -> StockHistoricalDataClient:
    global _data_client
    if _data_client is None:
        with _data_lock:
            if _data_client is None:
                _data_client = StockHistoricalDataClient(
                    api_key=config.alpaca_api_key,
                    secret_key=config.alpaca_secret_key,
                )
    return _data_client


# --- Timeframe mapping ---
TIMEFRAME_MAP = {
    "1m": TimeFrame(1, TimeFrameUnit.Minute),
    "5m": TimeFrame(5, TimeFrameUnit.Minute),
    "15m": TimeFrame(15, TimeFrameUnit.Minute),
    "1h": TimeFrame(1, TimeFrameUnit.Hour),
    "1d": TimeFrame(1, TimeFrameUnit.Day),
    "1w": TimeFrame(1, TimeFrameUnit.Week),
}

TIF_MAP = {
    "day": TimeInForce.DAY,
    "gtc": TimeInForce.GTC,
    "ioc": TimeInForce.IOC,
    "fok": TimeInForce.FOK,
}


async def _retry_alpaca_call(func, *args, max_attempts=3, base_delay=1.0, **kwargs):
    """Retry an Alpaca SDK call with exponential backoff on transient errors."""
    from alpaca.common.exceptions import APIError

    loop = asyncio.get_running_loop()
    last_exception = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
        except APIError as e:
            status = getattr(e, "status_code", None)
            if status in {429, 500, 502, 503, 504} and attempt < max_attempts:
                delay = min(base_delay * (2 ** (attempt - 1)) * 2.0, 30.0)
                logger.warning(f"Alpaca retry {attempt}/{max_attempts}: HTTP {status}, sleeping {delay:.1f}s")
                last_exception = e
                await asyncio.sleep(delay)
                continue
            raise
        except Exception as e:
            if attempt < max_attempts and _is_transient_error(e):
                delay = min(base_delay * (2 ** (attempt - 1)) * 2.0, 30.0)
                logger.warning(f"Alpaca retry {attempt}/{max_attempts}: {type(e).__name__}, sleeping {delay:.1f}s")
                last_exception = e
                await asyncio.sleep(delay)
                continue
            raise
    # S118 W-009: Guard against last_exception being None
    if last_exception is not None:
        raise last_exception
    raise RuntimeError(f"Alpaca retry exhausted after {max_attempts} attempts")


def _is_transient_error(e: Exception) -> bool:
    """Check if an exception looks like a transient network error."""
    name = type(e).__name__
    return any(kw in name for kw in ("Timeout", "Connection", "Reset", "BrokenPipe"))


async def execute_trade(params: dict[str, Any]) -> str:
    """Place a trade via Alpaca. Supports extended hours (pre/post market) via limit orders."""
    symbol = params["symbol"]
    try:
        side_value = normalize_order_side(params.get("side"))
    except AlpacaValueError as exc:
        return f"ERROR: Invalid order side ({exc})"
    side = OrderSide.BUY if side_value == "buy" else OrderSide.SELL
    qty = float(params["qty"])
    order_type = params["order_type"]
    extended_hours = bool(params.get("extended_hours", False))
    trade_id = params.get("trade_id") or None  # Alpaca client_order_id for position tracking

    # Extended hours validation: Alpaca only supports limit orders in pre/post market
    if extended_hours and order_type != "limit":
        return (
            "ERROR: Extended hours trading requires limit orders only. "
            "Market, stop, and stop_limit orders are not supported pre/post market. "
            "Please resubmit as order_type='limit' with a limit_price."
        )

    # Extended hours forces time_in_force=DAY (Alpaca requirement)
    if extended_hours:
        tif = TimeInForce.DAY
        logger.info(f"[EXTENDED HOURS] Order: {side.value} {qty} {symbol} — forcing TIF=DAY")
    else:
        tif = TIF_MAP.get(params.get("time_in_force", "day"), TimeInForce.DAY)

    client = _get_trading_client()

    try:
        if order_type == "market":
            request = MarketOrderRequest(
                symbol=symbol, qty=qty, side=side, time_in_force=tif,
                client_order_id=trade_id,
            )
        elif order_type == "limit":
            limit_price = params.get("limit_price")
            if limit_price is None:
                return "ERROR: limit_price required for limit orders"
            request = LimitOrderRequest(
                symbol=symbol, qty=qty, side=side, time_in_force=tif,
                limit_price=float(limit_price),
                extended_hours=extended_hours,
                client_order_id=trade_id,
            )
        elif order_type == "stop":
            stop_price = params.get("stop_price")
            if stop_price is None:
                return "ERROR: stop_price required for stop orders"
            request = StopOrderRequest(
                symbol=symbol, qty=qty, side=side, time_in_force=tif,
                stop_price=float(stop_price),
                client_order_id=trade_id,
            )
        elif order_type == "stop_limit":
            limit_price = params.get("limit_price")
            stop_price = params.get("stop_price")
            if limit_price is None or stop_price is None:
                return "ERROR: limit_price and stop_price required for stop_limit orders"
            request = StopLimitOrderRequest(
                symbol=symbol, qty=qty, side=side, time_in_force=tif,
                limit_price=float(limit_price), stop_price=float(stop_price),
                client_order_id=trade_id,
            )
        else:
            return f"ERROR: Unknown order type: {order_type}"

        order = await _retry_alpaca_call(client.submit_order, request)

        ext_tag = " [EXTENDED HOURS]" if extended_hours else ""
        logger.info(f"Order submitted{ext_tag}: {order.id} {side.value} {qty} {symbol}")

        tid_tag = f"\n  Trade ID: {trade_id}" if trade_id else ""
        result = (
            f"Order submitted successfully!{ext_tag}\n"
            f"  ID: {order.id}\n"
            f"  Symbol: {order.symbol}\n"
            f"  Side: {order.side.value}\n"
            f"  Qty: {order.qty}\n"
            f"  Type: {order.order_type.value}\n"
            f"  TIF: {order.time_in_force.value}\n"
            f"  Extended Hours: {'YES — Pre/Post Market' if extended_hours else 'NO — Regular Session'}\n"
            f"  Status: {order.status.value}\n"
            f"  Submitted: {order.submitted_at}"
            f"{tid_tag}"
        )
        return sanitize_ingested_content(result, "alpaca:trade")
    except Exception as e:
        logger.error(f"Trade failed: {e}", exc_info=True)
        return f"ERROR: Trade failed — {e}"


async def get_portfolio(params: dict[str, Any]) -> str:
    """Get current portfolio positions."""
    client = _get_trading_client()
    try:
        positions = await _retry_alpaca_call(client.get_all_positions)
        if not positions:
            return "No open positions."

        equity_positions = []
        option_positions = []
        for pos in positions:
            # S159-B H-1: AssetClass is a str-Enum — str() gives "AssetClass.US_OPTION".
            raw_class = getattr(pos, "asset_class", None)
            asset_class = raw_class.value if hasattr(raw_class, "value") else str(raw_class)
            if asset_class == "us_option":
                option_positions.append(pos)
            else:
                equity_positions.append(pos)

        lines = ["Current Positions:\n"]
        total_unrealized = 0.0
        total_market_value = 0.0

        if equity_positions:
            lines.append("--- Equities ---")
            for pos in equity_positions:
                try:
                    # S121 W-22: Use `is None` check instead of `or 0` falsy chain
                    unrealized = float(0 if pos.unrealized_pl is None else pos.unrealized_pl)
                    market_val = float(0 if pos.market_value is None else pos.market_value)
                    pct = float(0 if pos.unrealized_plpc is None else pos.unrealized_plpc) * 100
                    entry_price = float(0 if pos.avg_entry_price is None else pos.avg_entry_price)
                    cur_price = float(0 if pos.current_price is None else pos.current_price)
                except (TypeError, ValueError):
                    unrealized = market_val = pct = entry_price = cur_price = 0.0
                total_unrealized += unrealized
                total_market_value += market_val
                lines.append(
                    f"  {pos.symbol}: {pos.qty} shares @ ${entry_price:.2f}\n"
                    f"    Current: ${cur_price:.2f} | "
                    f"Mkt Value: ${market_val:,.2f} | "
                    f"P&L: ${unrealized:+,.2f} ({pct:+.2f}%)"
                )

        if option_positions:
            lines.append("\n--- Options ---")
            for pos in option_positions:
                try:
                    # S121 W-22: Use `is None` check instead of `or 0` falsy chain
                    unrealized = float(0 if pos.unrealized_pl is None else pos.unrealized_pl)
                    market_val = float(0 if pos.market_value is None else pos.market_value)
                    pct = float(0 if pos.unrealized_plpc is None else pos.unrealized_plpc) * 100
                    entry_price = float(0 if pos.avg_entry_price is None else pos.avg_entry_price)
                    cur_price = float(0 if pos.current_price is None else pos.current_price)
                except (TypeError, ValueError):
                    unrealized = market_val = pct = entry_price = cur_price = 0.0
                total_unrealized += unrealized
                total_market_value += market_val
                # Try to show human-readable OCC format
                try:
                    from agent.integrations.alpaca_options import format_occ_human
                    display = format_occ_human(str(pos.symbol))
                except Exception as e:
                    logger.debug(f"OCC format failed for {pos.symbol}: {e}")
                    display = str(pos.symbol)
                lines.append(
                    f"  {display}: {pos.qty} contracts @ ${entry_price:.2f}\n"
                    f"    Current: ${cur_price:.2f} | "
                    f"Mkt Value: ${market_val:,.2f} | "
                    f"P&L: ${unrealized:+,.2f} ({pct:+.2f}%)"
                )

        lines.append(f"\nTotal Market Value: ${total_market_value:,.2f}")
        lines.append(f"Total Unrealized P&L: ${total_unrealized:+,.2f}")
        return sanitize_ingested_content("\n".join(lines), "alpaca:portfolio")
    except Exception as e:
        logger.error(f"Portfolio fetch failed: {e}", exc_info=True)
        return f"ERROR: Failed to fetch portfolio — {e}"


async def get_account_info(params: dict[str, Any]) -> str:
    """Get account balance and buying power."""
    client = _get_trading_client()
    try:
        account = await _retry_alpaca_call(client.get_account)
        result = (
            f"Account Info ({'Paper' if config.alpaca_paper else 'Live'} Trading):\n"
            f"  Status: {account.status.value}\n"
            f"  Cash: ${float(account.cash):,.2f}\n"
            f"  Portfolio Value: ${float(account.portfolio_value):,.2f}\n"
            f"  Buying Power: ${float(account.buying_power):,.2f}\n"
            f"  Equity: ${float(account.equity):,.2f}\n"
            f"  Long Market Value: ${float(account.long_market_value):,.2f}\n"
            f"  Short Market Value: ${float(account.short_market_value):,.2f}\n"
            f"  Day Trade Count: {account.daytrade_count}\n"
            f"  PDT Restricted: {account.pattern_day_trader}"
        )
        return sanitize_ingested_content(result, "alpaca:account")
    except Exception as e:
        logger.error(f"Account fetch failed: {e}", exc_info=True)
        return f"ERROR: Failed to fetch account — {e}"


async def get_market_data(params: dict[str, Any]) -> str:
    """Fetch market data from Alpaca (fallback if Polygon unavailable)."""
    symbol = params["symbol"]
    data_type = params.get("data_type", "quote")
    client = _get_data_client()

    try:
        if data_type == "quote":
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = await _retry_alpaca_call(client.get_stock_latest_quote, request)
            q = quotes[symbol]
            result = (
                f"Latest Quote for {symbol}:\n"
                f"  Ask: ${q.ask_price:.2f} x {q.ask_size}\n"
                f"  Bid: ${q.bid_price:.2f} x {q.bid_size}\n"
                f"  Timestamp: {q.timestamp}"
            )
            return sanitize_ingested_content(result, "alpaca:market_data")

        elif data_type == "bars":
            tf_str = params.get("timeframe", "1d")
            timeframe = TIMEFRAME_MAP.get(tf_str, TimeFrame(1, TimeFrameUnit.Day))
            limit = params.get("limit", 10)

            # Default lookback based on timeframe
            lookback_days = {"1m": 1, "5m": 2, "15m": 5, "1h": 10, "1d": 30, "1w": 180}
            days = lookback_days.get(tf_str, 30)

            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=datetime.now(_NY) - timedelta(days=days),
                limit=limit,
            )
            bars = await _retry_alpaca_call(client.get_stock_bars, request)
            bar_list = bars[symbol]

            lines = [f"Bars for {symbol} ({tf_str}, last {len(bar_list)}):"]
            for bar in bar_list[-limit:]:
                lines.append(
                    f"  {bar.timestamp.strftime('%Y-%m-%d %H:%M')} | "
                    f"O: ${bar.open:.2f} H: ${bar.high:.2f} "
                    f"L: ${bar.low:.2f} C: ${bar.close:.2f} V: {bar.volume:,}"
                )
            return sanitize_ingested_content("\n".join(lines), "alpaca:market_data")

        elif data_type == "snapshot":
            request = StockSnapshotRequest(symbol_or_symbols=symbol)
            snapshots = await _retry_alpaca_call(client.get_stock_snapshot, request)
            snap = snapshots[symbol]

            result = (
                f"Snapshot for {symbol}:\n"
                f"  Latest Trade: ${snap.latest_trade.price:.2f} ({snap.latest_trade.size} shares)\n"
                f"  Latest Quote: Ask ${snap.latest_quote.ask_price:.2f} / Bid ${snap.latest_quote.bid_price:.2f}\n"
                f"  Daily Bar: O ${snap.daily_bar.open:.2f} H ${snap.daily_bar.high:.2f} "
                f"L ${snap.daily_bar.low:.2f} C ${snap.daily_bar.close:.2f} V {snap.daily_bar.volume:,}\n"
                f"  Prev Daily Bar: O ${snap.previous_daily_bar.open:.2f} C ${snap.previous_daily_bar.close:.2f}"
            )
            return sanitize_ingested_content(result, "alpaca:market_data")

        return f"Unknown data_type: {data_type}"
    except Exception as e:
        logger.error(f"Market data fetch failed: {e}", exc_info=True)
        return f"ERROR: Market data failed for {symbol} — {e}"
