"""WebSocket stream — live log tailing."""

import asyncio
import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from mission_control.backend.auth import decode_token
from mission_control.backend.ws.manager import ws_manager
from mission_control.backend.config import mc_config

router = APIRouter(tags=["ws"])
logger = logging.getLogger("mission_control.ws.logs")

# Hydrinia is permanently retired (operator direction, 2026-09-24); its unit is
# only a valid tail target when the bridge is explicitly re-enabled.
_ALLOWED_UNITS = {"moltbot", "moltbot-voice", "moltbot-control"}
_BRIDGE_ONLY_UNITS = {"hydrinia"}


def _hydrinia_bridge_enabled() -> bool:
    try:
        from agent.hydrinia_bridge import bridge_enabled

        return bridge_enabled()
    except Exception as e:
        logger.debug(f"ws.logs: bridge_enabled() unavailable ({type(e).__name__}: {e})")
        return False


@router.websocket("/ws/logs")
async def ws_logs(
    websocket: WebSocket,
    token: str = Query(""),
    unit: str = Query("moltbot"),
    lines: int = Query(200, ge=10, le=2000),
):
    # Auth check
    if not decode_token(token):
        await websocket.close(code=4001, reason="Unauthorized")
        return

    allowed = _ALLOWED_UNITS | (_BRIDGE_ONLY_UNITS if _hydrinia_bridge_enabled() else set())
    if unit not in allowed:
        await websocket.close(code=4002, reason=f"Unknown unit: {unit}")
        return

    channel = f"logs:{unit}"
    await ws_manager.connect(channel, websocket)

    try:
        # Start journalctl streaming
        proc = await asyncio.create_subprocess_exec(
            "journalctl", "-f", "-u", unit, "--no-pager", "-n", str(lines),
            "--output=short-precise",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

        async def read_logs():
            _line_count = 0
            _last_reset = time.monotonic()
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                _now = time.monotonic()
                if _now - _last_reset >= 1.0:
                    _line_count = 0
                    _last_reset = _now
                _line_count += 1
                if _line_count > 50:  # Max 50 lines/sec
                    await asyncio.sleep(0.1)
                text = line.decode("utf-8", errors="replace").rstrip()
                try:
                    await websocket.send_json({"line": text})
                except Exception:
                    break

        # Run log reader and listen for client messages concurrently
        reader_task = asyncio.create_task(read_logs())

        try:
            while True:
                # Keep connection alive, handle client commands
                msg = await websocket.receive_text()
                # Client can send "pause"/"resume" — for now just keep alive
        except WebSocketDisconnect:
            pass
        finally:
            reader_task.cancel()
            proc.terminate()
            try:
                await proc.wait()
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"WS logs error: {e}")
    finally:
        await ws_manager.disconnect(channel, websocket)
