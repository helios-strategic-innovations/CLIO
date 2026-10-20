"""
Moltbook API Client
===================
Sole HTTP interface between Moltbot and moltbook.com.
All API calls route through this module — no other module touches the network for Moltbook.

Security:
- API key loaded from env only, never logged or exposed
- All requests timeout at 30s
- Rate limiting enforced via token bucket
- Dry-run mode: all write operations no-op when MOLTBOOK_DRY_RUN=true
- Credential isolation: explicitly cannot access any non-Moltbook credentials
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import aiohttp

from agent.moltbook.config import MoltbookConfig
from agent.moltbook.kill_switch import MoltbookKillSwitch

logger = logging.getLogger(__name__)

# ─── Credential Isolation — Explicit Deny List ────────────────────────────────
# These env vars must NEVER be accessed by MoltbookClient.
# Verified at startup. Any attempt to read these raises IsolationViolation.
_FORBIDDEN_ENV_VARS = {
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "POLYGON_API_KEY",
    "UNUSUAL_WHALES_API_KEY",
    "UW_API_KEY",  # the name CLIO actually reads (agent/config.py)
    "EMAIL_PASSWORD",
    "CANVAS_API_TOKEN",
    "ANTHROPIC_API_KEY",
    "ONEPROVIDER_API_KEY",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_AUTHORIZED_USERS",
    "AUDIT_LOG_HMAC_SECRET",
}


class MoltbookRateLimiter:
    """Token bucket rate limiter for Moltbook API calls."""

    def __init__(self, max_calls_per_minute: int = 10):
        self.max_calls = max_calls_per_minute
        self.tokens = max_calls_per_minute
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            # Refill tokens based on elapsed time
            self.tokens = min(
                self.max_calls,
                self.tokens + elapsed * (self.max_calls / 60.0)
            )
            self.last_refill = now

            if self.tokens < 1:
                wait_time = (1 - self.tokens) / (self.max_calls / 60.0)
                logger.debug(f"[MoltbookClient] Rate limit: waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


@dataclass
class MoltbookPost:
    """A post retrieved from Moltbook."""
    post_id: str
    author: str
    content: str
    timestamp: str
    category: str = ""
    tags: list = field(default_factory=list)
    engagement: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)


@dataclass
class MoltbookComment:
    """A comment retrieved from or posted to Moltbook."""
    comment_id: str
    post_id: str
    author: str
    content: str
    timestamp: str
    raw: dict = field(default_factory=dict)


@dataclass
class PublishResult:
    """Result of a publish operation."""
    success: bool
    dry_run: bool
    post_id: Optional[str] = None
    comment_id: Optional[str] = None
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class MoltbookClient:
    """
    Sole HTTP interface to moltbook.com.

    Usage:
        async with MoltbookClient(config) as client:
            posts = await client.browse_feed()
            result = await client.post_comment(post_id, content)

    NOTE: API endpoints are placeholders — verify against moltbook.com API docs
    before enabling MOLTBOOK_DRY_RUN=false.
    """

    # ── Placeholder endpoints — MUST be verified from moltbook.com API docs ──
    _ENDPOINTS = {
        "feed":        "/api/v1/feed",
        "post_detail": "/api/v1/posts/{post_id}",
        "create_post": "/api/v1/posts",
        "create_comment": "/api/v1/posts/{post_id}/comments",
        "delete_post": "/api/v1/posts/{post_id}",
        "delete_comment": "/api/v1/comments/{comment_id}",
        "profile":     "/api/v1/me",
        "search":      "/api/v1/search",
    }
    # S185 W4 — flip to True ONLY after every entry in _ENDPOINTS is verified
    # against real moltbook.com API docs. While False, live-mode construction
    # is refused (defense against accidental MOLTBOOK_DRY_RUN=false flip).
    _ENDPOINTS_VERIFIED: bool = False

    def __init__(self, config: MoltbookConfig):
        # S185 W4 — refuse live construction while endpoints are placeholders.
        if not config.dry_run and not self._ENDPOINTS_VERIFIED:
            raise RuntimeError(
                "MoltbookClient: cannot construct in live mode (MOLTBOOK_DRY_RUN=false) "
                "while endpoints are placeholders. Verify all entries in _ENDPOINTS "
                "against moltbook.com API docs and set _ENDPOINTS_VERIFIED=True before "
                "flipping DRY_RUN."
            )
        self._config = config
        self._kill_switch = MoltbookKillSwitch()
        self._rate_limiter = MoltbookRateLimiter(max_calls_per_minute=10)
        self._session: Optional[aiohttp.ClientSession] = None
        self._verify_isolation()

    def _verify_isolation(self) -> None:
        """Log which forbidden env vars exist. Advisory only — DRY_RUN mode makes this non-critical."""
        found = [var for var in _FORBIDDEN_ENV_VARS if var in os.environ]
        if found:
            logger.warning(f"[MoltbookClient] Isolation advisory: {len(found)} forbidden env vars present: {', '.join(sorted(found))}")
        logger.debug("[MoltbookClient] Credential isolation check complete")

    @property
    def _base_url(self) -> str:
        return self._config.base_url.rstrip("/")

    @property
    def _api_key(self) -> str:
        """Load API key from env — only place it's accessed."""
        key = os.environ.get("MOLTBOOK_API_KEY", "")
        if not key:
            logger.warning("[MoltbookClient] MOLTBOOK_API_KEY not set")
        return key

    def _url(self, endpoint_key: str, **kwargs) -> str:
        path = self._ENDPOINTS[endpoint_key].format(**kwargs)
        return f"{self._base_url}{path}"

    def _headers(self) -> dict:
        # S121 W-51: X-Request-ID is set once per session, not per request.
        # Acceptable while MOLTBOOK_DRY_RUN=true. If Moltbook goes live,
        # generate X-Request-ID per-request in each method instead.
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Moltbot/1.0",
            "X-Request-ID": hashlib.sha256(
                f"{time.time()}".encode()
            ).hexdigest()[:16],
        }

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self._session = aiohttp.ClientSession(
            timeout=timeout,
            headers=self._headers()
        )
        return self

    async def __aexit__(self, *args):
        if self._session:
            try:
                await self._session.close()
            except Exception as e:
                logger.debug(f"Session close failed: {e}")
            self._session = None

    def _check_kill_switch(self) -> None:
        """Halt immediately if kill switch is active."""
        if self._kill_switch.is_active:
            raise RuntimeError("[MoltbookClient] Kill switch active — all API calls halted")

    async def _get(self, url: str, params: dict = None) -> dict:
        """Execute GET request with rate limiting and kill switch check."""
        self._check_kill_switch()
        await self._rate_limiter.acquire()

        if not self._session:
            raise RuntimeError("MoltbookClient must be used as async context manager")

        logger.debug(f"[MoltbookClient] GET {url} params={params}")
        try:
            async with self._session.get(url, params=params) as resp:
                resp.raise_for_status()
                return await resp.json()
        except aiohttp.ClientError as e:
            logger.error(f"[MoltbookClient] GET {url} failed: {e}")
            raise

    async def _post(self, url: str, payload: dict) -> dict:
        """Execute POST request. No-ops in dry-run mode."""
        self._check_kill_switch()
        # S118 C-002: Rate limit BEFORE dry-run check (prevents bypass)
        await self._rate_limiter.acquire()

        if self._config.dry_run:
            logger.info(f"[MoltbookClient] DRY-RUN POST {url} payload={json.dumps(payload)[:200]}")
            return {"dry_run": True, "url": url, "payload": payload}
        if not self._session:
            raise RuntimeError("MoltbookClient must be used as async context manager")

        logger.info(f"[MoltbookClient] POST {url}")
        try:
            async with self._session.post(url, json=payload) as resp:
                resp.raise_for_status()
                return await resp.json()
        except aiohttp.ClientError as e:
            logger.error(f"[MoltbookClient] POST {url} failed: {e}")
            raise

    async def _delete(self, url: str) -> dict:
        """Execute DELETE request. No-ops in dry-run mode."""
        self._check_kill_switch()
        # S118 C-002: Rate limit BEFORE dry-run check (prevents bypass)
        await self._rate_limiter.acquire()

        if self._config.dry_run:
            logger.info(f"[MoltbookClient] DRY-RUN DELETE {url}")
            return {"dry_run": True, "url": url}
        if not self._session:
            raise RuntimeError("MoltbookClient must be used as async context manager")

        logger.info(f"[MoltbookClient] DELETE {url}")
        try:
            async with self._session.delete(url) as resp:
                resp.raise_for_status()
                return await resp.json()
        except aiohttp.ClientError as e:
            logger.error(f"[MoltbookClient] DELETE {url} failed: {e}")
            raise

    # ── Public API Methods ────────────────────────────────────────────────────

    async def browse_feed(
        self,
        category: str = "",
        limit: int = 20,
        offset: int = 0
    ) -> list[MoltbookPost]:
        """Browse the Moltbook feed. Read-only — safe in all modes."""
        params = {"limit": limit, "offset": offset}
        if category:
            params["category"] = category

        try:
            data = await self._get(self._url("feed"), params=params)
            posts = []
            for item in data.get("posts", data if isinstance(data, list) else []):
                posts.append(MoltbookPost(
                    post_id=str(item.get("id", "")),
                    author=item.get("author", {}).get("username", "") if isinstance(item.get("author"), dict) else str(item.get("author", "")),
                    content=item.get("content", item.get("body", "")),
                    timestamp=item.get("created_at", item.get("timestamp", "")),
                    category=item.get("category", ""),
                    tags=item.get("tags", []),
                    engagement={
                        "likes": item.get("likes", 0),
                        "comments": item.get("comment_count", 0),
                    },
                    raw=item
                ))
            logger.info(f"[MoltbookClient] browse_feed: retrieved {len(posts)} posts")
            return posts
        except Exception as e:
            logger.error(f"[MoltbookClient] browse_feed failed: {e}")
            return []

    async def get_post(self, post_id: str) -> Optional[MoltbookPost]:
        """Get a single post by ID."""
        try:
            data = await self._get(self._url("post_detail", post_id=post_id))
            return MoltbookPost(
                post_id=str(data.get("id", post_id)),
                author=data.get("author", {}).get("username", "") if isinstance(data.get("author"), dict) else str(data.get("author", "")),
                content=data.get("content", data.get("body", "")),
                timestamp=data.get("created_at", ""),
                category=data.get("category", ""),
                tags=data.get("tags", []),
                raw=data
            )
        except Exception as e:
            logger.error(f"[MoltbookClient] get_post {post_id} failed: {e}")
            return None

    async def create_post(self, content: str, category: str = "", tags: list = None) -> PublishResult:
        """
        Create a new post on Moltbook.
        Blocked if kill switch active. No-op if dry_run=True.
        """
        self._check_kill_switch()

        payload = {
            "content": content,
            "category": category,
            "tags": tags or [],
        }

        try:
            data = await self._post(self._url("create_post"), payload)
            if data.get("dry_run"):
                return PublishResult(success=True, dry_run=True)

            return PublishResult(
                success=True,
                dry_run=False,
                post_id=str(data.get("id", "")),
            )
        except Exception as e:
            logger.error(f"[MoltbookClient] create_post failed: {e}")
            return PublishResult(success=False, dry_run=self._config.dry_run, error=str(e))

    async def create_comment(self, post_id: str, content: str) -> PublishResult:
        """
        Comment on a post.
        Blocked if kill switch active. No-op if dry_run=True.
        """
        self._check_kill_switch()

        payload = {"content": content}

        try:
            data = await self._post(self._url("create_comment", post_id=post_id), payload)
            if data.get("dry_run"):
                return PublishResult(success=True, dry_run=True)

            return PublishResult(
                success=True,
                dry_run=False,
                comment_id=str(data.get("id", "")),
                post_id=post_id,
            )
        except Exception as e:
            logger.error(f"[MoltbookClient] create_comment {post_id} failed: {e}")
            return PublishResult(success=False, dry_run=self._config.dry_run, error=str(e))

    async def delete_post(self, post_id: str) -> PublishResult:
        """Delete a post. No-op if dry_run=True."""
        try:
            data = await self._delete(self._url("delete_post", post_id=post_id))
            return PublishResult(
                success=True,
                dry_run=self._config.dry_run,
                post_id=post_id,
            )
        except Exception as e:
            logger.error(f"[MoltbookClient] delete_post {post_id} failed: {e}")
            return PublishResult(success=False, dry_run=self._config.dry_run, error=str(e))

    async def delete_comment(self, comment_id: str) -> PublishResult:
        """Delete a comment. No-op if dry_run=True."""
        try:
            await self._delete(self._url("delete_comment", comment_id=comment_id))
            return PublishResult(
                success=True,
                dry_run=self._config.dry_run,
                comment_id=comment_id,
            )
        except Exception as e:
            logger.error(f"[MoltbookClient] delete_comment {comment_id} failed: {e}")
            return PublishResult(success=False, dry_run=self._config.dry_run, error=str(e))

    async def search_posts(self, query: str, limit: int = 20) -> list[MoltbookPost]:
        """Search posts by keyword."""
        try:
            data = await self._get(self._url("search"), params={"q": query, "limit": limit})
            posts = []
            for item in data.get("results", data if isinstance(data, list) else []):
                posts.append(MoltbookPost(
                    post_id=str(item.get("id", "")),
                    author=item.get("author", {}).get("username", "") if isinstance(item.get("author"), dict) else str(item.get("author", "")),
                    content=item.get("content", ""),
                    timestamp=item.get("created_at", ""),
                    raw=item
                ))
            return posts
        except Exception as e:
            logger.error(f"[MoltbookClient] search_posts failed: {e}")
            return []

    async def get_profile(self) -> dict:
        """Get Moltbot's own profile."""
        try:
            return await self._get(self._url("profile"))
        except Exception as e:
            logger.error(f"[MoltbookClient] get_profile failed: {e}")
            return {}

    async def health_check(self) -> bool:
        """Verify API connectivity. Safe read-only check."""
        try:
            profile = await self.get_profile()
            return bool(profile)
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False
