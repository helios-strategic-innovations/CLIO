"""
OpenClaw/Moltbot — Alpaca Options Integration
Handles: get_option_chain, get_option_quote, execute_options_trade
Mid-point limit pricing for options orders.
"""

import asyncio
import logging
import re
import threading
from datetime import datetime
from typing import Any, Optional

from zoneinfo import ZoneInfo

from agent.config import config
from agent.integrations.alpaca import _get_trading_client, _retry_alpaca_call
from agent.security import sanitize_ingested_content

logger = logging.getLogger("openclaw.integrations.alpaca_options")

_NY = ZoneInfo("America/New_York")

# --- OCC Symbol Regex ---
# Format: 1-6 char root + 6 digit date (YYMMDD) + C/P + 8 digit strike (strike*1000 zero-padded)
_OCC_REGEX = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")

# --- Thread-safe singletons ---
_option_data_client = None
_option_data_lock = threading.Lock()
# S121 W-23/24: _get_trading_client imported from alpaca.py (shared singleton)


def _get_option_data_client():
    """Thread-safe singleton for OptionHistoricalDataClient."""
    global _option_data_client
    if _option_data_client is None:
        with _option_data_lock:
            if _option_data_client is None:
                from alpaca.data.historical.option import OptionHistoricalDataClient
                _option_data_client = OptionHistoricalDataClient(
                    api_key=config.alpaca_api_key,
                    secret_key=config.alpaca_secret_key,
                )
    return _option_data_client


# --- OCC Symbol Utilities ---

def format_occ_symbol(underlying: str, expiration: str, option_type: str, strike: float) -> str:
    """
    Build OCC option symbol from components.
    ('SPY', '2025-03-21', 'call', 590.0) -> 'SPY250321C00590000'
    """
    dt = datetime.strptime(expiration, "%Y-%m-%d")
    date_part = dt.strftime("%y%m%d")
    type_char = "C" if option_type.lower() == "call" else "P"
    strike_int = int(strike * 1000)
    return f"{underlying.upper()}{date_part}{type_char}{strike_int:08d}"


def parse_occ_symbol(occ: str) -> Optional[dict]:
    """
    Parse OCC option symbol into components.
    'SPY250321C00590000' -> {underlying, expiration, type, strike}
    Returns None if invalid.
    """
    m = _OCC_REGEX.match(occ)
    if not m:
        return None
    underlying, date_str, type_char, strike_str = m.groups()
    try:
        dt = datetime.strptime(date_str, "%y%m%d")
        expiration = dt.strftime("%Y-%m-%d")
    except ValueError:
        return None
    return {
        "underlying": underlying,
        "expiration": expiration,
        "type": "call" if type_char == "C" else "put",
        "strike": int(strike_str) / 1000.0,
    }


def format_occ_human(occ: str) -> str:
    """Format OCC symbol for human display: 'SPY Mar 21 $590.00 Call'"""
    parsed = parse_occ_symbol(occ)
    if not parsed:
        return occ
    try:
        dt = datetime.strptime(parsed["expiration"], "%Y-%m-%d")
        date_str = dt.strftime("%b %d")
    except ValueError:
        date_str = parsed["expiration"]
    return f"{parsed['underlying']} {date_str} ${parsed['strike']:.2f} {parsed['type'].capitalize()}"


# --- Mid-point Pricing ---

def _compute_options_limit_price(bid: float, ask: float, side: str,
                                  spread_adjustment: float = 0.02) -> float:
    """
    Compute limit price for options using mid-point strategy.

    spread_adjustment = 0.02 means move 2% of the spread width toward the taker side.
    For a $1.00/$1.10 option (spread = $0.10):
      - Mid = $1.05
      - Buy: $1.05 + 0.10 * 0.02 = $1.052 -> $1.05
      - Sell: $1.05 - 0.10 * 0.02 = $1.048 -> $1.05
    For a $2.00/$2.40 option (spread = $0.40):
      - Mid = $2.20
      - Buy: $2.20 + 0.40 * 0.02 = $2.208 -> $2.21
      - Sell: $2.20 - 0.40 * 0.02 = $2.192 -> $2.19
    """
    if bid <= 0 or ask <= 0 or ask < bid:
        raise ValueError(f"Invalid bid/ask: {bid}/{ask}")
    spread = ask - bid
    mid = (bid + ask) / 2
    if side == "buy":
        price = mid + spread * spread_adjustment
    else:
        price = mid - spread * spread_adjustment
    return round(price, 2)


# S121 W-25: _retry_alpaca_call imported from alpaca.py (shared implementation)


# --- Safe float helper ---

def _sf(val, default=0.0) -> float:
    """Safe float conversion for external API values."""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# --- Option Chain Retrieval ---

async def get_option_chain(params: dict[str, Any]) -> str:
    """Fetch option chain for an underlying symbol."""
    symbol = params.get("underlying", "").upper()
    if not symbol:
        return "ERROR: underlying symbol is required"

    try:
        from alpaca.trading.requests import GetOptionContractsRequest
        from alpaca.trading.enums import ContractType

        req_params = {"underlying_symbols": [symbol]}

        expiration = params.get("expiration")
        if expiration:
            req_params["expiration_date"] = expiration

        exp_gte = params.get("expiration_gte")
        if exp_gte:
            req_params["expiration_date_gte"] = exp_gte

        exp_lte = params.get("expiration_lte")
        if exp_lte:
            req_params["expiration_date_lte"] = exp_lte

        option_type = params.get("option_type", "").lower()
        if option_type == "call":
            req_params["type"] = ContractType.CALL
        elif option_type == "put":
            req_params["type"] = ContractType.PUT

        strike_min = params.get("strike_min")
        if strike_min is not None:
            req_params["strike_price_gte"] = str(strike_min)

        strike_max = params.get("strike_max")
        if strike_max is not None:
            req_params["strike_price_lte"] = str(strike_max)

        try:
            limit = min(int(params.get("limit", 20)), 50)
        except (TypeError, ValueError):
            limit = 20
        req_params["limit"] = limit

        request = GetOptionContractsRequest(**req_params)
        client = _get_trading_client()
        response = await _retry_alpaca_call(client.get_option_contracts, request)

        contracts = response.option_contracts or []
        if not contracts:
            return f"No option contracts found for {symbol} with given filters."

        # Get snapshots for quotes/Greeks
        contract_symbols = [c.symbol for c in contracts[:limit]]
        snapshots = {}
        try:
            from alpaca.data.requests import OptionSnapshotRequest
            opt_client = _get_option_data_client()
            # Feed parameter goes on the request, not the method call
            snap_kwargs: dict[str, Any] = {"symbol_or_symbols": contract_symbols}
            try:
                from alpaca.data.enums import OptionsFeed
                feed_val = config.alpaca_options_feed.upper()
                snap_kwargs["feed"] = OptionsFeed.OPRA if feed_val == "OPRA" else OptionsFeed.INDICATIVE
            except (ImportError, AttributeError):
                pass
            snap_req = OptionSnapshotRequest(**snap_kwargs)
            snapshots = await _retry_alpaca_call(opt_client.get_option_snapshot, snap_req)
        except Exception as e:
            logger.debug(f"Option snapshot fetch failed (non-fatal): {e}")

        # Build output
        lines = [f"Option Chain for {symbol} ({len(contracts[:limit])} contracts):\n"]
        lines.append(f"{'Symbol':<25} {'Exp':>10} {'Strike':>8} {'Type':>5} {'Bid':>8} {'Ask':>8} {'Mid':>8} {'Vol':>7} {'OI':>7} {'IV':>7} {'Delta':>7}")
        lines.append("-" * 120)

        for c in contracts[:limit]:
            sym = c.symbol
            exp = str(getattr(c, "expiration_date", ""))
            strike = _sf(getattr(c, "strike_price", 0))
            ctype = "CALL" if str(getattr(c, "type", "")).upper() in ("CALL", "C") else "PUT"

            snap = snapshots.get(sym)
            bid = ask = mid = last_price = 0.0
            vol = oi = 0
            iv = delta = 0.0

            if snap:
                q = getattr(snap, "latest_quote", None)
                if q:
                    bid = _sf(getattr(q, "bid_price", 0))
                    ask = _sf(getattr(q, "ask_price", 0))
                    mid = round((bid + ask) / 2, 2) if bid > 0 and ask > 0 else 0.0

                t = getattr(snap, "latest_trade", None)
                if t:
                    last_price = _sf(getattr(t, "price", 0))

                greeks = getattr(snap, "greeks", None)
                if greeks:
                    delta = _sf(getattr(greeks, "delta", 0))

                iv = _sf(getattr(snap, "implied_volatility", 0))

            oi = int(_sf(getattr(c, "open_interest", 0)))
            # Volume from snapshot daily bar if available
            if snap:
                db = getattr(snap, "daily_bar", None)
                if db:
                    vol = int(_sf(getattr(db, "volume", 0)))

            lines.append(
                f"{sym:<25} {exp:>10} {strike:>8.2f} {ctype:>5} "
                f"${bid:>7.2f} ${ask:>7.2f} ${mid:>7.2f} "
                f"{vol:>7,} {oi:>7,} {iv:>6.1%} {delta:>7.3f}"
            )

        return sanitize_ingested_content("\n".join(lines), "alpaca:option_chain")
    except Exception as e:
        logger.error(f"Option chain fetch failed: {e}", exc_info=True)
        return f"ERROR: Option chain failed for {symbol} — {e}"


async def get_option_quote(params: dict[str, Any]) -> str:
    """Get latest quote for a specific option contract."""
    option_symbol = params.get("occ_symbol", "").upper()
    if not option_symbol or not _OCC_REGEX.match(option_symbol):
        return f"ERROR: Invalid option symbol: {option_symbol}. Expected OCC format (e.g., SPY250321C00590000)"

    try:
        opt_client = _get_option_data_client()

        # Get snapshot (with feed parameter matching chain endpoint)
        from alpaca.data.requests import OptionSnapshotRequest
        snap_kwargs: dict[str, Any] = {"symbol_or_symbols": [option_symbol]}
        try:
            from alpaca.data.enums import OptionsFeed
            feed_val = config.alpaca_options_feed.upper()
            snap_kwargs["feed"] = OptionsFeed.OPRA if feed_val == "OPRA" else OptionsFeed.INDICATIVE
        except (ImportError, AttributeError):
            pass
        snap_req = OptionSnapshotRequest(**snap_kwargs)
        snapshots = await _retry_alpaca_call(opt_client.get_option_snapshot, snap_req)

        snap = snapshots.get(option_symbol)
        if not snap:
            return f"No quote data found for {option_symbol}"

        parsed = parse_occ_symbol(option_symbol)
        human_name = format_occ_human(option_symbol) if parsed else option_symbol

        q = getattr(snap, "latest_quote", None)
        bid = _sf(getattr(q, "bid_price", 0)) if q else 0.0
        ask = _sf(getattr(q, "ask_price", 0)) if q else 0.0
        mid = round((bid + ask) / 2, 2) if bid > 0 and ask > 0 else 0.0
        spread = round(ask - bid, 2) if bid > 0 and ask > 0 else 0.0

        t = getattr(snap, "latest_trade", None)
        last_price = _sf(getattr(t, "price", 0)) if t else 0.0
        last_size = int(_sf(getattr(t, "size", 0))) if t else 0

        iv = _sf(getattr(snap, "implied_volatility", 0))

        greeks = getattr(snap, "greeks", None)
        delta = _sf(getattr(greeks, "delta", 0)) if greeks else 0.0
        gamma = _sf(getattr(greeks, "gamma", 0)) if greeks else 0.0
        theta = _sf(getattr(greeks, "theta", 0)) if greeks else 0.0
        vega = _sf(getattr(greeks, "vega", 0)) if greeks else 0.0
        rho = _sf(getattr(greeks, "rho", 0)) if greeks else 0.0

        result = (
            f"Option Quote: {human_name}\n"
            f"  Symbol: {option_symbol}\n"
            f"  Bid: ${bid:.2f}  |  Ask: ${ask:.2f}  |  Mid: ${mid:.2f}  |  Spread: ${spread:.2f}\n"
            f"  Last: ${last_price:.2f} x {last_size}\n"
            f"  IV: {iv:.1%}\n"
            f"  Greeks:\n"
            f"    Delta: {delta:+.4f}  Gamma: {gamma:.4f}  Theta: {theta:+.4f}  Vega: {vega:.4f}  Rho: {rho:+.4f}"
        )
        return sanitize_ingested_content(result, "alpaca:option_quote")
    except Exception as e:
        logger.error(f"Option quote failed: {e}", exc_info=True)
        return f"ERROR: Option quote failed for {option_symbol} — {e}"


# --- Options Trade Execution ---

# Position intent mapping
_INTENT_MAP = {
    "buy_to_open": "BUY_TO_OPEN",
    "buy_to_close": "BUY_TO_CLOSE",
    "sell_to_open": "SELL_TO_OPEN",
    "sell_to_close": "SELL_TO_CLOSE",
    "bto": "BUY_TO_OPEN",
    "btc": "BUY_TO_CLOSE",
    "sto": "SELL_TO_OPEN",
    "stc": "SELL_TO_CLOSE",
}


async def execute_options_trade(params: dict[str, Any]) -> str:
    """
    Place an options trade via Alpaca using mid-point limit pricing.

    Always submits as a limit order. No market orders for options.
    """
    option_symbol = params.get("occ_symbol", "").upper()
    side_str = params.get("side", "").lower()
    qty = params.get("qty", 0)
    intent_str = params.get("position_intent", "").lower()
    manual_limit = params.get("limit_price_override")
    spread_adj = _sf(params.get("spread_adjustment", 0.02), 0.02)
    tif_str = params.get("time_in_force", "day").lower()
    trade_id = params.get("trade_id")

    # Validate
    if not _OCC_REGEX.match(option_symbol):
        return f"ERROR: Invalid option symbol: {option_symbol}. Expected OCC format."

    if side_str not in ("buy", "sell"):
        return f"ERROR: Invalid side: {side_str}. Must be 'buy' or 'sell'."

    try:
        qty = int(qty)
    except (TypeError, ValueError):
        return f"ERROR: Invalid qty: {qty}"
    if qty < 1 or qty > 100:
        return f"ERROR: qty must be 1-100 contracts (got {qty})"

    # Default intent based on side when not provided
    if not intent_str:
        intent_str = "buy_to_open" if side_str == "buy" else "sell_to_close"
    intent_key = _INTENT_MAP.get(intent_str)
    if not intent_key:
        return f"ERROR: Invalid position_intent: {intent_str}. Use buy_to_open, buy_to_close, sell_to_open, sell_to_close."

    # Determine pricing mode: explicit limit override or mid-point
    if manual_limit is not None:
        try:
            limit_price = max(0.01, round(float(manual_limit), 2))
        except (TypeError, ValueError):
            return f"ERROR: Invalid limit_price_override: {manual_limit}"
        price_mode = "limit"
    else:
        limit_price = None
        price_mode = "mid"

    # Market hours check — options only trade during regular hours
    now = datetime.now(_NY)
    if now.weekday() >= 5:
        return "ERROR: Options cannot be traded on weekends."
    hour_dec = now.hour + now.minute / 60.0
    if hour_dec < 9.5 or hour_dec >= 16.0:
        return "ERROR: Options can only be traded during market hours (9:30 AM – 4:00 PM ET)."

    # Step 1: Fetch current quote for pricing
    try:
        opt_client = _get_option_data_client()
        from alpaca.data.requests import OptionSnapshotRequest
        snap_req = OptionSnapshotRequest(symbol_or_symbols=[option_symbol])
        snapshots = await _retry_alpaca_call(opt_client.get_option_snapshot, snap_req)
        snap = snapshots.get(option_symbol)
        if not snap:
            return f"ERROR: No quote data available for {option_symbol}. Cannot determine pricing."
    except Exception as e:
        return f"ERROR: Failed to fetch option quote for pricing — {e}"

    q = getattr(snap, "latest_quote", None)
    bid = _sf(getattr(q, "bid_price", 0)) if q else 0.0
    ask = _sf(getattr(q, "ask_price", 0)) if q else 0.0

    if bid <= 0 or ask <= 0:
        return f"ERROR: Invalid quote for {option_symbol} — bid=${bid:.2f}, ask=${ask:.2f}. Market may be closed."

    if ask < bid:
        return f"ERROR: Crossed market for {option_symbol} — bid=${bid:.2f} > ask=${ask:.2f}. Try again."

    mid = round((bid + ask) / 2, 2)
    spread = round(ask - bid, 2)

    # Spread sanity check — reject illiquid options
    # S190 M1: defense-in-depth. The upstream `bid <= 0 or ask <= 0` guard at line ~414
    # already catches the both-zero illiquid case (mid would be 0). Tightening this
    # secondary check to `mid > 0.01` adds a margin for sub-penny edge cases that
    # Alpaca shouldn't produce but downstream protection doesn't rely on the upstream.
    if mid > 0.01 and spread / mid > 0.50:
        return (
            f"ERROR: Option spread too wide for safe execution.\n"
            f"  {option_symbol}: bid=${bid:.2f}, ask=${ask:.2f}, spread=${spread:.2f} ({spread/mid:.0%} of mid)\n"
            f"  Max allowed: 50% of mid. This option is too illiquid."
        )

    # Step 2: Compute limit price (with $0.01 floor)
    if limit_price is None:
        limit_price = _compute_options_limit_price(bid, ask, side_str, spread_adj)
    limit_price = max(0.01, limit_price)

    # Step 3: Submit order
    try:
        from alpaca.trading.requests import LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        tif = TimeInForce.GTC if tif_str == "gtc" else TimeInForce.DAY

        # Build order kwargs — position_intent as enum if available
        order_kwargs: dict[str, Any] = {
            "symbol": option_symbol,
            "qty": qty,
            "side": OrderSide.BUY if side_str == "buy" else OrderSide.SELL,
            "time_in_force": tif,
            "limit_price": limit_price,
        }
        if trade_id:
            order_kwargs["client_order_id"] = trade_id
        try:
            from alpaca.trading.enums import PositionIntent
            order_kwargs["position_intent"] = PositionIntent(intent_key.lower())
        except (ImportError, ValueError) as e:
            logger.debug(f"PositionIntent enum not available, using string: {e}")
            order_kwargs["position_intent"] = intent_key

        order_request = LimitOrderRequest(**order_kwargs)

        client = _get_trading_client()
        order = await _retry_alpaca_call(client.submit_order, order_request)

        human_name = format_occ_human(option_symbol)
        logger.info(
            f"Options order submitted: {order.id} {side_str.upper()} {qty}x {option_symbol} "
            f"@ ${limit_price:.2f} ({price_mode})"
        )

        result = (
            f"Options order submitted successfully!\n"
            f"  Contract: {human_name}\n"
            f"  Symbol: {option_symbol}\n"
            f"  Side: {side_str.upper()}\n"
            f"  Intent: {intent_str.upper()}\n"
            f"  Qty: {qty} contracts\n"
            f"  Limit Price: ${limit_price:.2f} (mode: {price_mode})\n"
            f"  Market: bid=${bid:.2f} / ask=${ask:.2f} / mid=${mid:.2f} / spread=${spread:.2f}\n"
            f"  TIF: {tif_str.upper()}\n"
            f"  Order ID: {order.id}\n"
            f"  Status: {str(order.status)}"
        )
        return sanitize_ingested_content(result, "alpaca:options_trade")
    except Exception as e:
        logger.error(f"Options trade failed: {e}", exc_info=True)
        return f"ERROR: Options trade failed — {e}"
