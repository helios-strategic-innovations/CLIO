"""WebSocket connection manager — handles subscriptions and fan-out."""

import asyncio
import json
import logging
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger("mission_control.ws")

_MAX_PER_CHANNEL = 10
_MAX_TOTAL = 50


class ConnectionManager:
    """Manages WebSocket connections grouped by channel."""

    def __init__(self):
        self._connections: dict[str, list[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, channel: str, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            total = sum(len(c) for c in self._connections.values())
            if total >= _MAX_TOTAL:
                logger.warning("S159 C-3: Global WS connection limit reached")
                await ws.close(code=1008, reason="Connection limit reached")
                return
            if channel not in self._connections:
                self._connections[channel] = []
            if len(self._connections[channel]) >= _MAX_PER_CHANNEL:
                logger.warning(f"S159 C-3: Channel {channel} WS connection limit reached")
                await ws.close(code=1008, reason="Channel limit reached")
                return
            self._connections[channel].append(ws)
        logger.info(f"WS connected: {channel} (total: {len(self._connections.get(channel, []))})")

    async def disconnect(self, channel: str, ws: WebSocket):
        async with self._lock:
            if channel in self._connections:
                try:
                    self._connections[channel].remove(ws)
                except ValueError:
                    pass
                if not self._connections[channel]:
                    del self._connections[channel]

    async def broadcast(self, channel: str, data: Any):
        """Send data to all connections on a channel."""
        async with self._lock:
            connections = list(self._connections.get(channel, []))

        dead = []
        for ws in connections:
            try:
                if isinstance(data, str):
                    await ws.send_text(data)
                else:
                    await ws.send_json(data)
            except Exception:
                dead.append(ws)

        # Clean up dead connections
        for ws in dead:
            await self.disconnect(channel, ws)

    def connection_count(self, channel: str = None) -> int:
        if channel:
            return len(self._connections.get(channel, []))
        return sum(len(conns) for conns in self._connections.values())


ws_manager = ConnectionManager()
