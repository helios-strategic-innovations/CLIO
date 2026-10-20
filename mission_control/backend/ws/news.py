"""WebSocket stream — news push."""

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from mission_control.backend.auth import decode_token
from mission_control.backend.ws.manager import ws_manager
from mission_control.backend.deps import read_json, run_in_executor
from mission_control.backend.config import mc_config

import os

router = APIRouter(tags=["ws"])
logger = logging.getLogger("mission_control.ws.news")

_INTEL_STATE = os.path.join(mc_config.openclaw_root, "data", "intelligence", "news_intel_state.json")


@router.websocket("/ws/news")
async def ws_news(websocket: WebSocket, token: str = Query("")):
    if not decode_token(token):
        await websocket.close(code=4001, reason="Unauthorized")
        return

    await ws_manager.connect("news", websocket)
    last_count = 0

    async def poll():
        nonlocal last_count
        while True:
            await asyncio.sleep(60)  # Check every minute
            try:
                intel = await run_in_executor(lambda: read_json(_INTEL_STATE))
                if intel:
                    items = intel.get("items", [])
                    if len(items) > last_count:
                        new_items = items[last_count:]
                        for item in new_items:
                            await websocket.send_json({
                                "type": "news",
                                "title": item.get("title", item.get("headline", "")),
                                "topic": item.get("topic", ""),
                                "timestamp": item.get("sent_at", ""),
                            })
                        last_count = len(items)
            except Exception as e:
                logger.debug(f"News poll error: {e}")

    poller = asyncio.create_task(poll())
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        poller.cancel()
        await ws_manager.disconnect("news", websocket)
