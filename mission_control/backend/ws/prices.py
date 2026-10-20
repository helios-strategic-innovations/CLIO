"""WebSocket stream — live price updates via PriceStreamManager + Polygon fallback."""

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from mission_control.backend.auth import decode_token
from mission_control.backend.ws.manager import ws_manager

router = APIRouter(tags=["ws"])
logger = logging.getLogger("mission_control.ws.prices")


@router.websocket("/ws/prices")
async def ws_prices(websocket: WebSocket, token: str = Query("")):
    if not decode_token(token):
        await websocket.close(code=4001, reason="Unauthorized")
        return

    await ws_manager.connect("prices", websocket)
    subscribed_symbols: set[str] = set()
    _closed = False

    # Try to use PriceStreamManager for real-time push
    _stream_mgr = None
    _stream_callback = None
    try:
        from agent.integrations.price_stream import get_stream_manager
        _stream_mgr = get_stream_manager()
    except Exception:
        pass

    # Get the running event loop for cross-thread push
    _loop = asyncio.get_running_loop()

    def _on_quote(symbol: str, quote) -> None:
        """Callback from PriceStreamManager (fires from stream thread)."""
        if symbol not in subscribed_symbols or _closed:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                websocket.send_json({
                    "type": "price",
                    "symbol": symbol,
                    "bid": quote.bid,
                    "ask": quote.ask,
                    "mid": quote.mid,
                    "last": quote.last_trade_price,
                    "source": quote.source,
                }),
                _loop,
            )
        except Exception:
            pass

    if _stream_mgr and _stream_mgr.is_running:
        _stream_callback = _on_quote
        _stream_mgr.register_callback(_stream_callback)

    async def poll_fallback():
        """Periodic poll: push stream cache or fall back to Polygon."""
        while not _closed:
            await asyncio.sleep(5)
            if not subscribed_symbols:
                continue
            for symbol in list(subscribed_symbols):
                # Check stream cache first
                if _stream_mgr:
                    cached = _stream_mgr.get_quote(symbol)
                    if cached:
                        try:
                            await websocket.send_json({
                                "type": "price",
                                "symbol": symbol,
                                "bid": cached.bid,
                                "ask": cached.ask,
                                "mid": cached.mid,
                                "last": cached.last_trade_price,
                                "source": cached.source,
                            })
                        except Exception:
                            pass
                        continue
                # Polygon fallback for symbols without stream data
                try:
                    from agent.integrations.polygon import get_snapshot_raw
                    snap = await get_snapshot_raw(symbol)
                    day = snap.get("day", {})
                    prev = snap.get("prevDay", {})
                    last = snap.get("lastTrade", {})
                    price = float(last.get("p", 0)) or float(day.get("c", 0))
                    prev_close = float(prev.get("c", 0))
                    change = price - prev_close if price and prev_close else 0
                    await websocket.send_json({
                        "type": "price",
                        "symbol": symbol,
                        "price": price,
                        "change": round(change, 2),
                        "change_pct": round(change / prev_close * 100, 2) if prev_close else 0,
                        "volume": int(day.get("v", 0)),
                        "source": "polygon",
                    })
                except Exception as e:
                    logger.debug(f"Price poll error for {symbol}: {e}")

    poller = asyncio.create_task(poll_fallback())
    try:
        while True:
            msg = await websocket.receive_text()
            try:
                data = json.loads(msg)
                if "subscribe" in data:
                    for s in data["subscribe"]:
                        sym = s.upper()
                        subscribed_symbols.add(sym)
                        # Auto-subscribe to stream if manager is running
                        if _stream_mgr and _stream_mgr.is_running:
                            if any(c.isdigit() for c in sym[:6]):
                                asyncio.create_task(_safe_subscribe_option(sym))
                            else:
                                asyncio.create_task(_safe_subscribe_stock(sym))
                if "unsubscribe" in data:
                    for s in data["unsubscribe"]:
                        subscribed_symbols.discard(s.upper())
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        pass
    finally:
        _closed = True
        poller.cancel()
        if _stream_mgr and _stream_callback:
            _stream_mgr.unregister_callback(_stream_callback)
        await ws_manager.disconnect("prices", websocket)


async def _safe_subscribe_stock(symbol: str):
    try:
        from agent.integrations.price_stream import get_stream_manager
        await get_stream_manager().subscribe_stock(symbol)
    except Exception:
        pass


async def _safe_subscribe_option(occ: str):
    try:
        from agent.integrations.price_stream import get_stream_manager
        await get_stream_manager().subscribe_option(occ)
    except Exception:
        pass
