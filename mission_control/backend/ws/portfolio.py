"""WebSocket stream — live portfolio updates."""

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from mission_control.backend.auth import decode_token
from mission_control.backend.ws.manager import ws_manager
from mission_control.backend.deps import get_trading_client

router = APIRouter(tags=["ws"])
logger = logging.getLogger("mission_control.ws.portfolio")


@router.websocket("/ws/portfolio")
async def ws_portfolio(websocket: WebSocket, token: str = Query("")):
    if not decode_token(token):
        await websocket.close(code=4001, reason="Unauthorized")
        return

    await ws_manager.connect("portfolio", websocket)

    async def poll():
        while True:
            await asyncio.sleep(30)
            try:
                client = get_trading_client()
                loop = asyncio.get_running_loop()
                account, positions = await loop.run_in_executor(
                    None, lambda: (client.get_account(), client.get_all_positions())
                )
                data = {
                    "type": "portfolio_update",
                    "equity": float(account.equity or 0),
                    "cash": float(account.cash or 0),
                    "day_pl": float(account.equity or 0) - float(account.last_equity or 0),
                    "positions": [
                        {
                            "symbol": p.symbol,
                            "qty": float(p.qty or 0),
                            "unrealized_pl": float(p.unrealized_pl or 0),
                            "current_price": float(p.current_price or 0),
                        }
                        for p in positions
                    ],
                }
                await websocket.send_json(data)
            except Exception as e:
                logger.debug(f"Portfolio poll error: {e}")

    poller = asyncio.create_task(poll())
    try:
        while True:
            await websocket.receive_text()  # Keep alive
    except WebSocketDisconnect:
        pass
    finally:
        poller.cancel()
        await ws_manager.disconnect("portfolio", websocket)
