"""
OpenClaw/Moltbot — Persistent Conversation Memory (SQLite)

Replaces the in-memory ConversationManager with SQLite-backed storage.
Conversations survive restarts. Uses aiosqlite for async access.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger("openclaw.memory")

DB_PATH = Path("/home/agent/openclaw/data/conversations.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp REAL NOT NULL,
    token_estimate INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_conv_id ON conversations(conversation_id);
CREATE INDEX IF NOT EXISTS idx_conv_ts ON conversations(conversation_id, timestamp);
"""


class PersistentConversationManager:
    """SQLite-backed conversation manager with same interface as ConversationManager."""

    # Claude API max input is ~200k tokens; leave headroom for system prompt + current turn
    MAX_INPUT_TOKENS = 150_000

    def __init__(self, max_history: int = 50):
        self.max_history = max_history
        self._db: aiosqlite.Connection | None = None
        self._initialized = False
        self._db_lock = asyncio.Lock()

    async def _ensure_db(self) -> aiosqlite.Connection:
        if self._db is not None:
            return self._db
        async with self._db_lock:
            if self._db is not None:
                return self._db
            DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            _db = await aiosqlite.connect(str(DB_PATH))
            await _db.execute("PRAGMA journal_mode=WAL")
            await _db.execute("PRAGMA busy_timeout=5000")
            _db.row_factory = aiosqlite.Row
            if not self._initialized:
                await _db.executescript(_SCHEMA)
                await _db.commit()
                self._initialized = True
                logger.info(f"Conversation DB initialized: {DB_PATH}")
            self._db = _db  # S153: assign after schema commit to prevent uninitialized access
        return self._db

    async def get_history(self, cid: str) -> list[dict]:
        """Load last N messages for a conversation, respecting token budget.

        Loads newest messages first, stops when cumulative tokens approach
        MAX_INPUT_TOKENS. This prevents the 'input too long' API error
        that occurs with many tool-result-heavy messages.
        """
        try:
            db = await self._ensure_db()
            cursor = await db.execute(
                "SELECT id, role, content, token_estimate FROM conversations "
                "WHERE conversation_id = ? ORDER BY timestamp DESC LIMIT ?",
                (cid, self.max_history),
            )
            rows = await cursor.fetchall()

            # Walk newest→oldest, accumulate tokens, stop at budget
            budget = self.MAX_INPUT_TOKENS
            kept = []
            for row in rows:
                est = row["token_estimate"] if row["token_estimate"] is not None else (len(row["content"]) // 4)
                if budget - est < 0 and kept:
                    # Would exceed budget — stop (but always keep at least 1 message)
                    logger.info(f"Token budget hit: keeping {len(kept)}/{len(rows)} messages "
                                f"(~{self.MAX_INPUT_TOKENS - budget:,} tokens)")
                    break
                budget -= est
                kept.append(row)

            # Reverse to chronological order
            result = []
            for row in reversed(kept):
                try:
                    content = json.loads(row["content"])
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping malformed conversation history row: rowid=%s",
                        row["id"],
                    )
                    continue
                result.append({"role": row["role"], "content": content})
            return result
        except Exception as e:
            logger.error("Failed to load history: %s", type(e).__name__, exc_info=True)
            return []

    async def add_message(self, cid: str, role: str, content: Any) -> None:
        """Store a message. Content is JSON-serialized to handle tool_use blocks."""
        try:
            db = await self._ensure_db()
            serialized = json.dumps(content, default=str)
            token_estimate = len(serialized) // 4
            await db.execute(
                "INSERT INTO conversations (conversation_id, role, content, timestamp, token_estimate) "
                "VALUES (?, ?, ?, ?, ?)",
                (cid, role, serialized, time.time(), token_estimate),
            )
            await db.commit()
            await self._prune(cid)
        except Exception as e:
            logger.error(f"Failed to save message for {cid}: {e}")

    async def _prune(self, cid: str) -> None:
        """Keep only the most recent max_history messages per conversation."""
        db = await self._ensure_db()
        cursor = await db.execute(
            "SELECT COUNT(*) as cnt FROM conversations WHERE conversation_id = ?",
            (cid,),
        )
        row = await cursor.fetchone()
        count = row["cnt"] if row else 0
        if count > self.max_history + 10:  # Prune with some slack
            await db.execute(
                "DELETE FROM conversations WHERE conversation_id = ? AND id NOT IN "
                "(SELECT id FROM conversations WHERE conversation_id = ? ORDER BY timestamp DESC LIMIT ?)",
                (cid, cid, self.max_history),
            )
            await db.commit()

    async def clear(self, cid: str) -> None:
        """Delete all messages for a conversation."""
        try:
            db = await self._ensure_db()
            await db.execute(
                "DELETE FROM conversations WHERE conversation_id = ?", (cid,)
            )
            await db.commit()
            logger.info(f"Cleared conversation: {cid}")
        except Exception as e:
            logger.error(f"Failed to clear {cid}: {e}")

    async def get_last_timestamp(self, cid: str) -> float | None:
        """Return the epoch timestamp of the most recent message in a conversation, or None."""
        try:
            db = await self._ensure_db()
            cursor = await db.execute(
                "SELECT MAX(timestamp) as ts FROM conversations WHERE conversation_id = ?",
                (cid,),
            )
            row = await cursor.fetchone()
            return row["ts"] if row and row["ts"] is not None else None
        except Exception as e:
            logger.error(f"Failed to get last timestamp for {cid}: {e}")
            return None

    async def get_summary(self, cid: str) -> dict:
        """Return message count and estimated tokens for a conversation."""
        try:
            db = await self._ensure_db()
            cursor = await db.execute(
                "SELECT COUNT(*) as cnt, COALESCE(SUM(token_estimate), 0) as tokens "
                "FROM conversations WHERE conversation_id = ?",
                (cid,),
            )
            row = await cursor.fetchone()
            return {"messages": row["cnt"], "token_estimate": row["tokens"]}
        except Exception as e:
            logger.error(f"Failed to get summary for {cid}: {e}")
            return {"messages": 0, "token_estimate": 0}

    @property
    def conversations(self) -> dict:
        """Compatibility property for /status command (returns empty dict — use get_summary instead)."""
        return {}

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def checkpoint(self) -> None:
        """WAL checkpoint to keep WAL file small (S148-B W3-1)."""
        try:
            db = await self._ensure_db()
            await db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            logger.debug("WAL checkpoint completed")
        except Exception as e:
            logger.debug(f"WAL checkpoint failed: {e}")
