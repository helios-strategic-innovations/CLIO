"""
OpenClaw/Moltbot — Web Search & Content Fetching

DuckDuckGo search + httpx content fetching + BeautifulSoup text extraction.
All fetched content is treated as UNTRUSTED and sanitized before use.
"""

import asyncio
import ipaddress
import logging
import os
import random
import socket
import time as _time
from urllib.parse import urlparse
from typing import Any

import httpx
from bs4 import BeautifulSoup

from agent.rate_limiter import check_and_record
from agent.security import sanitize_ingested_content

logger = logging.getLogger("openclaw.web_search")

# S122 2A: In-memory search result cache (2h TTL, max 200 entries)
# Cache hits skip budget check — saves API calls and reduces Brave 429 exposure.
import hashlib
from datetime import datetime as _dt, timedelta as _td
_search_cache: dict[str, tuple[list[dict], _dt]] = {}
_CACHE_TTL = _td(hours=2)
_CACHE_MAX = 200

# S146: Global rate limiter — minimum interval between searches (all 21 callers share this)
_search_lock = asyncio.Lock()
_last_search_ts: float = 0.0
_MIN_SEARCH_INTERVAL = 2.5  # seconds (S165: was 1.5, increased to reduce 429s)

# S165: Backend rotation — distribute load across DDG backends
_BACKENDS = ["duckduckgo", "auto"]
_backend_idx = 0

# S165: 403 log dedup — only log first occurrence per domain per session
_blocked_domains_logged: set = set()

# R-26: transient network errors (ConnectError / connection reset) are retried with a
# short backoff, then fail over to the other backend once. Classified by exception class
# name and errno text only; the message body is never logged.
_TRANSIENT_NET_CLASSES = frozenset({
    "ConnectError", "ConnectionResetError", "ConnectTimeout",
    "ReadTimeout", "RemoteProtocolError",
})
_TRANSIENT_NET_MARKERS = ("Errno 104", "Connection reset by peer")


def _transient_class(exc: BaseException) -> "str | None":
    """Class name of a transient network error in exc or its __cause__ chain, else None."""
    cur = exc
    for _ in range(5):
        if cur is None:
            break
        name = type(cur).__name__
        if name in _TRANSIENT_NET_CLASSES or any(m in str(cur) for m in _TRANSIENT_NET_MARKERS):
            return name
        cur = cur.__cause__
    return None


# S185 W3-1: Suppress upstream-engine bot-detection noise from the `ddgs` library.
# Google/Brave/Mojeek/Yahoo routinely return 403/429 to headless search traffic;
# `ddgs` already retries across backends, but it WARNINGs each failure. That
# produced ~374 waste log lines in 72h. Filter drops 403/429 WARNINGs only — real
# errors (ERROR level, parse failures, etc) still propagate.
_SUPPRESS_UPSTREAM_NOISE = os.environ.get("CLIO_WEB_SEARCH_QUIET", "1") == "1"


class _UpstreamNoiseFilter(logging.Filter):
    _BOT_MARKERS = ("403", "429", "forbidden", "rate limit", "blocked")

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno > logging.WARNING:
            return True  # always keep ERROR/CRITICAL
        msg = record.getMessage().lower()
        return not any(m in msg for m in self._BOT_MARKERS)


if _SUPPRESS_UPSTREAM_NOISE:
    for _name in ("ddgs", "primp", "duckduckgo_search"):
        try:
            logging.getLogger(_name).addFilter(_UpstreamNoiseFilter())
        except Exception as e:
            logger.debug("web_search: failed to attach noise filter to %r logger: %s", _name, e)

# ============================================================
# Constants
# ============================================================

MAX_SEARCH_RESULTS = 10
MAX_RESPONSE_BYTES = 512_000          # 500KB max HTTP response body
MAX_EXTRACTED_CHARS = 50_000          # 50K chars max extracted text
FETCH_TIMEOUT = httpx.Timeout(connect=15.0, read=30.0, write=10.0, pool=10.0)

# User-Agent for HTTP requests (honest identification)
USER_AGENT = "MoltBot/1.0 (Research Agent; +https://github.com/openclaw)"

# Domain blocklist — no social media, forums, file hosting, URL shorteners
BLOCKED_DOMAINS = {
    # Social media
    "facebook.com", "www.facebook.com", "instagram.com", "www.instagram.com",
    "tiktok.com", "www.tiktok.com", "snapchat.com",
    "twitter.com", "x.com",  # We have dedicated Twitter tools
    # Forums (we have dedicated Reddit tools)
    "reddit.com", "www.reddit.com", "old.reddit.com",
    "quora.com", "www.quora.com",
    # File hosting / URL shorteners
    "bit.ly", "t.co", "goo.gl", "tinyurl.com",
    "dropbox.com", "drive.google.com", "mega.nz",
    # Potentially dangerous
    "localhost", "127.0.0.1", "0.0.0.0",
}

# Tags to remove from HTML before text extraction
STRIP_TAGS = {"script", "style", "nav", "footer", "header", "aside", "iframe", "noscript", "svg"}


# ============================================================
# URL Validation
# ============================================================

def _is_ip_address(hostname: str) -> bool:
    """Check if hostname is any form of IP address (IPv4, IPv6, bracketed)."""
    clean = hostname.strip("[]")
    try:
        ipaddress.ip_address(clean)
        return True
    except ValueError:
        return False


def _resolves_to_private(hostname: str) -> bool:
    """DNS-resolve hostname and check if it points to a private/reserved IP."""
    try:
        results = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for family, _, _, _, sockaddr in results:
            ip_str = sockaddr[0]
            ip = ipaddress.ip_address(ip_str)
            if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local:
                return True
    except (socket.gaierror, OSError, ValueError):
        return True  # DNS failure — fail closed (block for SSRF defense)
    return False


def _is_allowed_url(url: str) -> tuple[bool, str]:
    """Validate URL for safety. Returns (allowed, reason)."""
    try:
        parsed = urlparse(url)
    except Exception as e:
        logger.debug(f"URL parse failed: {e}")
        return False, "Invalid URL format"

    # Must be HTTPS (or HTTP)
    if parsed.scheme not in ("http", "https"):
        return False, f"Unsupported scheme: {parsed.scheme}"

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return False, "No hostname in URL"

    # Block all IP addresses — IPv4, IPv6, octal, hex, decimal, bracketed
    if _is_ip_address(hostname):
        return False, "IP addresses not allowed"

    # Block hostnames that resolve to private/reserved IPs (DNS rebinding defense)
    if _resolves_to_private(hostname):
        return False, "Hostname resolves to private/reserved IP"

    # Check domain blocklist (check base domain and full hostname)
    for blocked in BLOCKED_DOMAINS:
        if hostname == blocked or hostname.endswith("." + blocked):
            return False, f"Domain blocked: {blocked}"

    return True, "OK"


async def _is_allowed_url_async(url: str) -> tuple[bool, str]:
    """Async URL validation — runs DNS check in thread executor to avoid blocking."""
    try:
        parsed = urlparse(url)
    except Exception as e:
        logger.debug(f"Async URL parse failed: {e}")
        return False, "Invalid URL format"

    if parsed.scheme not in ("http", "https"):
        return False, f"Unsupported scheme: {parsed.scheme}"

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return False, "No hostname in URL"

    if _is_ip_address(hostname):
        return False, "IP addresses not allowed"

    # DNS check in executor (the blocking part)
    loop = asyncio.get_running_loop()
    is_private = await loop.run_in_executor(None, _resolves_to_private, hostname)
    if is_private:
        return False, "Hostname resolves to private/reserved IP"

    for blocked in BLOCKED_DOMAINS:
        if hostname == blocked or hostname.endswith("." + blocked):
            return False, f"Domain blocked: {blocked}"

    return True, "OK"


# ============================================================
# DuckDuckGo Search
# ============================================================

def shutdown_search_executors(wait: bool = False) -> bool:
    """Best-effort cleanup for third-party search executors before process exit."""
    try:
        from ddgs import DDGS

        executor = getattr(DDGS, "_executor", None)
        if executor is None:
            logger.info("DDGS executor shutdown skipped: no executor was created")
            return False

        try:
            executor.shutdown(wait=wait, cancel_futures=True)
        except TypeError:
            executor.shutdown(wait=wait)
        setattr(DDGS, "_executor", None)
        logger.info("DDGS executor shutdown requested (wait=%s)", wait)
        return True
    except Exception as e:
        logger.warning("DDGS executor shutdown failed: %s", str(e)[:200])
        return False


async def search_web(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """
    Search DuckDuckGo for web results.

    Returns list of dicts with keys: title, url, snippet.
    Runs synchronous DDGS in thread executor to avoid blocking.
    S146/S165: Global rate limiter (2.5s min between searches) + retry with backoff on 429.
    """
    global _last_search_ts
    max_results = max(1, min(int(max_results), MAX_SEARCH_RESULTS))

    # S122 2A: Cache check BEFORE budget — cache hits are free
    _cache_key = hashlib.md5(f"{query}:{max_results}".encode()).hexdigest()
    _now = _dt.now()
    _cached = _search_cache.get(_cache_key)
    if _cached and (_now - _cached[1]) < _CACHE_TTL:
        logger.debug(f"Search cache hit: '{query[:50]}'")
        return _cached[0]

    await check_and_record("web_search")

    # S146: Global rate limiter — prevents burst requests across all 21 callers
    async with _search_lock:
        _elapsed = _time.time() - _last_search_ts
        if _elapsed < _MIN_SEARCH_INTERVAL:
            await asyncio.sleep(_MIN_SEARCH_INTERVAL - _elapsed + random.uniform(0, 1.0))
        _last_search_ts = _time.time()

    def _search_on(_primary, _fallback):
        from ddgs import DDGS
        _backend_used = _primary
        for _attempt in range(3):
            try:
                with DDGS() as ddgs:
                    try:
                        raw_results = list(ddgs.text(query, max_results=max_results, backend=_primary))
                    except Exception as _dbe:
                        # S159-B L-2: DDG backend may raise "No results" instead of
                        # returning []. Fall back to other backend before giving up.
                        if "no results" in str(_dbe).lower():
                            logger.debug("DDG backend '%s' raised 'No results' for '%s', trying '%s'",
                                         _primary, query[:50], _fallback)
                            raw_results = list(ddgs.text(query, max_results=max_results, backend=_fallback))
                            _backend_used = _fallback
                        else:
                            raise
                    # S148-B W1-2: Fallback to other backend if primary returns empty
                    if not raw_results:
                        logger.debug(f"DDG backend '{_primary}' empty for '{query[:50]}', retrying with '{_fallback}'")
                        raw_results = list(ddgs.text(query, max_results=max_results, backend=_fallback))
                        _backend_used = _fallback
                results = []
                for r in raw_results:
                    url = r.get("href", r.get("url", ""))
                    allowed, _ = _is_allowed_url(url)
                    if not allowed:
                        continue
                    results.append({
                        "title": r.get("title", "(no title)"),
                        "url": url,
                        "snippet": r.get("body", r.get("snippet", ""))[:500],
                    })
                return results, _backend_used
            except Exception as e:
                if _attempt < 2 and ("429" in str(e) or "rate" in str(e).lower()):
                    _time.sleep(2 ** _attempt + random.uniform(0, 1))
                    logger.debug(f"Search retry {_attempt+1}/3 for: {query[:50]}...")
                    continue
                raise

    def _sync_search():
        global _backend_idx
        # S165: Rotate primary backend to distribute load
        _primary = _BACKENDS[_backend_idx % len(_BACKENDS)]
        _fallback = _BACKENDS[(_backend_idx + 1) % len(_BACKENDS)]
        _backend_idx += 1
        # R-26: primary gets the initial attempt + 2 retries (1 s, 2 s backoff) on transient
        # network errors; then the other backend gets exactly one attempt. Give up with [].
        for _t_attempt in range(3):
            try:
                return _search_on(_primary, _fallback)
            except Exception as e:
                if _transient_class(e) is None:
                    raise
                if _t_attempt < 2:
                    _time.sleep(1 if _t_attempt == 0 else 2)
        try:
            return _search_on(_fallback, _primary)
        except Exception as e:
            _cls = _transient_class(e)
            if _cls is None:
                raise
        logger.warning("web_search transient network error backend=%s class=%s", _fallback, _cls)
        return [], _fallback

    try:
        loop = asyncio.get_running_loop()
        results, _backend_used = await loop.run_in_executor(None, _sync_search)
        logger.info(f"Web search: '{query[:40]}' → {len(results)} results (backend={_backend_used})")
        # S122 2A: Cache successful results (don't cache empty — allow retry)
        if results:
            _search_cache[_cache_key] = (results, _now)
            if len(_search_cache) > _CACHE_MAX:
                _oldest = min(_search_cache, key=lambda k: _search_cache[k][1])
                del _search_cache[_oldest]
        return results
    except Exception as e:
        # S159-B L-2: "No results" is common for site-scoped queries — demote, no traceback
        # S186 W4.1: Yahoo backend RequestErrors are recovered by primp's auto-rotation
        # (see Apr 29 logs: 24x errors followed by next-backend successes). Demote so they
        # don't pollute ERROR-level monitoring. ERROR reserved for non-recoverable cases.
        _es = str(e)
        _es_lower = _es.lower()
        if "no results" in _es_lower:
            logger.warning("DuckDuckGo search: no results for query=%s", query[:80])
        elif "requesterror" in _es_lower and ("yahoo" in _es_lower or "duckduckgo" in _es_lower):
            logger.warning("DuckDuckGo search: backend rotation error (recovered downstream): %s", _es[:200])
        else:
            logger.error(f"DuckDuckGo search error: {e}", exc_info=True)
        return []


# ============================================================
# Content Fetching
# ============================================================

async def fetch_page_content(url: str, max_chars: int = MAX_EXTRACTED_CHARS) -> str:
    """
    Fetch a URL and extract clean text content.

    Returns sanitized plain text (wrapped in sanitize_ingested_content).
    Returns error string on failure.
    """
    # P1-12: validate the URL BEFORE charging budget — a blocked/SSRF-filtered URL
    # must not drain the web_fetch budget.
    allowed, reason = await _is_allowed_url_async(url)
    if not allowed:
        return f"ERROR: URL blocked — {reason}"

    await check_and_record("web_fetch")

    max_chars = max(1000, min(int(max_chars), MAX_EXTRACTED_CHARS))

    try:
        async with httpx.AsyncClient(
            timeout=FETCH_TIMEOUT,
            follow_redirects=False,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            # Manual redirect loop with validation (max 5 hops)
            current_url = url
            for _redirect_hop in range(5):
                response = await client.get(current_url)

                if response.is_redirect:
                    location = response.headers.get("location", "")
                    if not location:
                        return "ERROR: Redirect with no Location header"
                    # Resolve relative redirects
                    if location.startswith("/"):
                        parsed_current = urlparse(current_url)
                        location = f"{parsed_current.scheme}://{parsed_current.netloc}{location}"
                    redirect_ok, redirect_reason = await _is_allowed_url_async(location)
                    if not redirect_ok:
                        return f"ERROR: Redirect to blocked URL — {redirect_reason}"
                    current_url = location
                    continue

                response.raise_for_status()

                # Check content type — only process HTML/text
                content_type = response.headers.get("content-type", "")
                if not any(ct in content_type.lower() for ct in ("text/html", "text/plain", "application/xhtml")):
                    return f"ERROR: Unsupported content type: {content_type}"

                # Check size (Content-Length pre-check if available)
                cl_header = response.headers.get("content-length")
                if cl_header:
                    try:
                        if int(cl_header) > MAX_RESPONSE_BYTES:
                            return f"ERROR: Response too large ({cl_header} bytes, max {MAX_RESPONSE_BYTES})"
                    except ValueError:
                        pass

                content_bytes = response.content
                if len(content_bytes) > MAX_RESPONSE_BYTES:
                    return f"ERROR: Response too large ({len(content_bytes)} bytes, max {MAX_RESPONSE_BYTES})"

                html = response.text
                break
            else:
                return "ERROR: Too many redirects"

    except httpx.TimeoutException:
        return f"ERROR: Timeout fetching {url}"
    except httpx.HTTPStatusError as e:
        # S165: Dedup 403 logging per domain to reduce noise
        if e.response.status_code == 403:
            _domain = urlparse(url).netloc
            if _domain not in _blocked_domains_logged:
                _blocked_domains_logged.add(_domain)
                logger.warning(f"Fetch blocked (403): {_domain}")
        return f"ERROR: HTTP {e.response.status_code} for {url}"
    except Exception as e:
        logger.error(f"Fetch error for {url}: {e}", exc_info=True)
        return f"ERROR: Failed to fetch {url}: {str(e)[:200]}"

    # Extract text with BeautifulSoup
    try:
        text = _extract_article_text(html)
    except Exception as e:
        logger.error(f"Parse error for {url}: {e}", exc_info=True)
        return f"ERROR: Failed to parse content from {url}"

    if not text or len(text.strip()) < 100:
        return f"ERROR: No meaningful text content extracted from {url}"

    # Cap at max_chars
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[TRUNCATED]"

    # Sanitize as untrusted external content
    return sanitize_ingested_content(text, f"web:{url[:100]}")


def _extract_article_text(html: str) -> str:
    """Extract main article content from HTML. Prefers <article> or <main> tags."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted tags
    for tag in soup.find_all(STRIP_TAGS):
        tag.decompose()

    # Try to find main content container (in order of preference)
    main_content = None
    for selector in ["article", "main", '[role="main"]', ".post-content", ".article-content", ".entry-content"]:
        main_content = soup.select_one(selector)
        if main_content:
            break

    # Fall back to body
    if not main_content:
        main_content = soup.find("body")
    if not main_content:
        main_content = soup

    # Extract text
    text = main_content.get_text(separator="\n", strip=True)

    # Clean up whitespace
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            lines.append(line)

    return "\n".join(lines)
