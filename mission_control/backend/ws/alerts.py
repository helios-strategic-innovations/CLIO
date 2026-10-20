"""WebSocket stream — Hydrinia alerts push."""

import asyncio
import logging
import os

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from mission_control.backend.auth import decode_token
from mission_control.backend.ws.manager import ws_manager
from mission_control.backend.deps import read_json, run_in_executor
from mission_control.backend.config import mc_config

router = APIRouter(tags=["ws"])
logger = logging.getLogger("mission_control.ws.alerts")

# Hydrinia is permanently retired (operator direction, 2026-09-24); its alerts.json
# goes stale on disk, so this stream only sources it when the bridge is explicitly
# re-enabled. Otherwise the socket stays connected (no reconnect storm) but is a
# silent no-op — there is no CLIO-native alert feed wired here yet.
_ALERTS_PATH = os.path.join(mc_config.tradingbot_root, "state", "alerts.json")
_bridge_gap_logged = False


def _hydrinia_bridge_enabled() -> bool:
    try:
        from agent.hydrinia_bridge import bridge_enabled

        return bridge_enabled()
    except Exception as e:
        logger.debug(f"ws.alerts: bridge_enabled() unavailable ({type(e).__name__}: {e})")
        return False


@router.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket, token: str = Query("")):
    if not decode_token(token):
        await websocket.close(code=4001, reason="Unauthorized")
        return

    await ws_manager.connect("alerts", websocket)
    last_count = 0

    async def poll():
        nonlocal last_count
        global _bridge_gap_logged
        while True:
            await asyncio.sleep(15)  # Check every 15s
            if not _hydrinia_bridge_enabled():
                if not _bridge_gap_logged:
                    logger.info("ws.alerts: Hydrinia bridge disabled — alert stream is a no-op")
                    _bridge_gap_logged = True
                continue
            try:
                alerts = await run_in_executor(lambda: read_json(_ALERTS_PATH)) or []
                if isinstance(alerts, list) and len(alerts) > last_count:
                    new_alerts = alerts[last_count:]
                    for alert in new_alerts:
                        await websocket.send_json({"type": "alert", **alert})
                    last_count = len(alerts)
            except Exception as e:
                logger.debug(f"Alert poll error: {e}")

    poller = asyncio.create_task(poll())
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        poller.cancel()
        await ws_manager.disconnect("alerts", websocket)
