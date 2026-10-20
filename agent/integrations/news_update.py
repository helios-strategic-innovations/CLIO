"""
OpenClaw/Moltbot — Half-Hourly News Update (Session 85)

Runs every 30 minutes from 8:15 AM to 8:15 PM ET, every day (including weekends).
Delivers punchy news headlines with dedup to avoid repeating stories, to
Telegram (+ the Hydrinia file bridge, only if explicitly re-enabled via
CLIO_HYDRINIA_BRIDGE_ENABLED — see agent/hydrinia_bridge.py).

Ticker coverage tracks CLIO's own DPOF book (SPY/QQQ long options on the CLIO
Alpaca paper account, agent/dpof/) — never Hydrinia's frozen state files.
"""

import asyncio
import fcntl
import json
import logging
import os
import re
from datetime import datetime, timedelta
from urllib.parse import urlparse, urlunparse
from zoneinfo import ZoneInfo

import anthropic

from agent.config import config
from agent.prompts.system_prompt import cached_system_block
from agent.rate_limiter import budget_tracker, BudgetExhaustedError
from agent.security import sanitize_ingested_content

logger = logging.getLogger("openclaw.news_update")

_NY = ZoneInfo("America/New_York")

REPORTS_DIR = os.path.join(config.tradingbot_dir, "moltbot_knowledge", "reports", "news")
DEDUP_STATE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "knowledge", "news_update_state.json",
)

# ETF -> underlying asset news query mapping
# When searching for news about ETF positions, search for the underlying asset instead
_ETF_NEWS_QUERIES = {
    "BOIL": "natural gas", "KOLD": "natural gas", "UNG": "natural gas",
    "TQQQ": "NASDAQ tech stocks", "SQQQ": "NASDAQ tech stocks", "QQQ": "NASDAQ tech stocks",
    "UPRO": "S&P 500", "SPXU": "S&P 500", "SPXL": "S&P 500", "SPY": "S&P 500",
    "UVIX": "VIX volatility", "VIXY": "VIX volatility", "SVXY": "VIX volatility",
    "GLD": "gold", "SLV": "silver", "AGQ": "silver", "IAU": "gold", "UGL": "gold",
    "TLT": "treasury bonds", "TBT": "treasury bonds",
    "SOXL": "semiconductor stocks SMH", "SOXS": "semiconductor stocks SMH",
    "XLE": "energy sector oil", "XLF": "financial sector banks",
    "XLK": "technology sector", "XLV": "healthcare sector",
    "IWM": "Russell 2000 small cap",
}

_cached_client: anthropic.Anthropic | None = None
_client_lock = __import__("threading").Lock()


def _get_client() -> anthropic.Anthropic:
    global _cached_client
    if _cached_client is None:
        with _client_lock:
            if _cached_client is None:
                _cached_client = anthropic.Anthropic(**config.anthropic_client_kwargs(timeout=120))
    return _cached_client


NEWS_SYSTEM_PROMPT = """You are a punchy financial news wire editor. Write a brief, concise news update.

CRITICAL FRESHNESS RULE:
- You MUST only report news from TODAY. The current date/time is provided below.
- If a search result describes events from yesterday or earlier, DO NOT include it.
- If the data in a search result contradicts what you know about current market direction, SKIP IT — the data is likely stale.
- When in doubt about freshness, OMIT the story. It is FAR better to report nothing than to report wrong/stale information.
- NEVER fabricate price movements or percentages — only report numbers explicitly stated in the provided search results.

Rules:
- No fluff, no filler — every sentence must be informative
- Include source URLs where available
- Mark updates to previously reported stories with [UPDATE]
- Focus on: positions we hold, major market-moving news, geopolitical events

Write in plain text only — NO markdown formatting, NO headers with #, NO bold/italic markers.

Format:

NEWS UPDATE — {time ET}

[For each story, 1-3 lines max:]
HEADLINE: {concise headline}
{1-2 sentence summary with key numbers/facts}
{URL if available}

[If an update to a previous story:]
[UPDATE] HEADLINE: {headline}
{What changed since last report}
{URL}

If no significant new developments: "No major updates since last report. Markets steady."
"""


# ============================================================
# Dedup State Management
# ============================================================

_DEDUP_STATE_LOCK = DEDUP_STATE_PATH + ".lock"


def _load_dedup_state() -> dict:
    """Load news dedup state from disk."""
    default = {"sent_stories": [], "sent_event_keys": [], "last_update": ""}
    if not os.path.isfile(DEDUP_STATE_PATH):
        return default
    lock_fd = None
    try:
        lock_fd = open(_DEDUP_STATE_LOCK, "a")
        fcntl.flock(lock_fd, fcntl.LOCK_SH)
        with open(DEDUP_STATE_PATH, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return default
        # Migrate: add sent_event_keys if missing
        if "sent_event_keys" not in data:
            data["sent_event_keys"] = []
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load news dedup state: {e}")
        return default
    finally:
        if lock_fd:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
            except Exception:
                pass


def _save_dedup_state(state: dict) -> None:
    """Save news dedup state atomically with file locking."""
    lock_fd = None
    try:
        os.makedirs(os.path.dirname(DEDUP_STATE_PATH), exist_ok=True)
        lock_fd = open(_DEDUP_STATE_LOCK, "a")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        tmp = DEDUP_STATE_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp, DEDUP_STATE_PATH)
    except Exception as e:
        logger.error(f"Failed to save news dedup state: {e}")
    finally:
        if lock_fd:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
            except Exception:
                pass


def _prune_dedup_state(state: dict) -> dict:
    """Remove stories older than 48 hours and cap at 200."""
    cutoff = datetime.now(_NY) - timedelta(hours=48)
    stories = state.get("sent_stories", [])

    pruned = []
    for story in stories:
        first_sent = story.get("first_sent", "")
        if not first_sent:
            continue  # Drop entries without timestamp
        try:
            dt = datetime.fromisoformat(first_sent)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_NY)
            if dt >= cutoff:
                pruned.append(story)
        except (ValueError, TypeError):
            continue  # Drop entries with unparseable dates

    # Cap at 200 (oldest first — keep newest)
    if len(pruned) > 200:
        pruned = pruned[-200:]

    state["sent_stories"] = pruned

    # Prune event keys with same 48h window
    event_keys = state.get("sent_event_keys", [])
    pruned_keys = []
    for ek in event_keys:
        first_sent = ek.get("first_sent", "")
        if first_sent:
            try:
                dt = datetime.fromisoformat(first_sent)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=_NY)
                if dt >= cutoff:
                    pruned_keys.append(ek)
                    continue
            except (ValueError, TypeError):
                continue  # Drop entries with unparseable dates
    if len(pruned_keys) > 200:
        pruned_keys = pruned_keys[-200:]
    state["sent_event_keys"] = pruned_keys

    return state


def _get_sent_headlines(state: dict) -> str:
    """Format previously sent stories for the Claude prompt."""
    stories = state.get("sent_stories", [])
    if not stories:
        return "No previously reported stories."

    # Show last 20 to keep prompt concise
    recent = stories[-20:]
    lines = []
    for s in recent:
        headline = s.get("headline", "")
        ticker = s.get("ticker_context", "")
        ctx = f" [{ticker}]" if ticker else ""
        lines.append(f"  - {headline}{ctx}")
    return "\n".join(lines)


# ============================================================
# Data Gathering
# ============================================================

def _get_open_positions() -> list[dict]:
    """Tickers CLIO's news update should track: DPOF holdings + trading universe.

    CLIO-native (agent/dpof/readmodel.py) — replaces Hydrinia's position_log.json.
    Union of DPOF's currently-open underlyings (held=True) and the configured
    DPOF universe (SPY/QQQ by default, held=False), so coverage continues even
    when CLIO is flat. Kept small and deduped.
    """
    held: set[str] = set()
    try:
        from agent.dpof import readmodel
        held = set(readmodel.open_underlyings())
    except Exception as e:
        logger.debug(f"DPOF readmodel unavailable for open positions: {e}")

    universe: list[str] = ["SPY", "QQQ"]
    try:
        from agent.dpof.config import load_dpof_config
        universe = list(load_dpof_config().underlyings) or universe
    except Exception as e:
        logger.debug(f"DPOF config unavailable, using default universe: {e}")

    tickers: list[str] = []
    seen: set[str] = set()
    for t in list(held) + universe:
        if t and t not in seen:
            seen.add(t)
            tickers.append(t)

    return [{"symbol": t, "direction": "", "held": t in held} for t in tickers]


def _freshness_cutoff() -> datetime:
    """Return the freshness cutoff: 8 PM ET the prior day.

    Any news older than this is considered stale and must not be reported.
    """
    now = datetime.now(_NY)
    yesterday_8pm = now.replace(hour=20, minute=0, second=0, microsecond=0) - timedelta(days=1)
    return yesterday_8pm


def _url_is_stale(url: str) -> bool:
    """Reject URLs containing dates older than 8 PM ET the prior day."""
    if not url:
        return False
    now = datetime.now(_NY)

    # Check for full date patterns: /YYYY/MM/DD/ or /YYYY-MM-DD
    full_date = (
        re.search(r'/(\d{4})/(\d{1,2})/(\d{1,2})/', url) or
        re.search(r'/(\d{4})-(\d{1,2})-(\d{1,2})', url)
    )
    if full_date:
        try:
            url_date = datetime(
                int(full_date.group(1)), int(full_date.group(2)), int(full_date.group(3)),
                hour=23, minute=59, tzinfo=now.tzinfo,
            )
            if url_date < _freshness_cutoff():
                return True
        except (ValueError, OverflowError):
            pass
        return False

    # Check for month-only patterns: /YYYY/MM/ (no day)
    month_only = re.search(r'/(\d{4})/(\d{1,2})/', url) or re.search(r'/(\d{4})-(\d{1,2})(?!-\d)', url)
    if month_only:
        try:
            url_year, url_month = int(month_only.group(1)), int(month_only.group(2))
            # If month is more than 1 month behind current, it's stale
            url_date = datetime(url_year, url_month, 1, tzinfo=now.tzinfo)
            if (now - url_date).days > 45:
                return True
        except (ValueError, OverflowError):
            pass

    return False


def _snippet_is_stale(title: str, snippet: str) -> bool:
    """Check if title/snippet contains date references older than freshness cutoff.

    Catches patterns like 'March 1, 2026', 'Mar 1 2026', 'February 28' etc.
    """
    text = (title + " " + snippet).lower()
    now = datetime.now(_NY)
    cutoff = _freshness_cutoff()

    # Match "Month DD, YYYY" or "Month DD YYYY" or "Mon DD, YYYY"
    _MONTHS = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    for pattern in [
        r'(\w+)\s+(\d{1,2}),?\s+(\d{4})',  # March 1, 2026
        r'(\d{1,2})\s+(\w+)\s+(\d{4})',     # 1 March 2026
    ]:
        for m in re.finditer(pattern, text):
            groups = m.groups()
            try:
                if groups[0].isdigit():
                    day, month_str, year = int(groups[0]), groups[1], int(groups[2])
                else:
                    month_str, day, year = groups[0], int(groups[1]), int(groups[2])
                month_num = _MONTHS.get(month_str.lower())
                if month_num and 2020 <= year <= now.year + 1:
                    article_date = datetime(year, month_num, day, hour=23, minute=59, tzinfo=now.tzinfo)
                    if article_date < cutoff:
                        return True
            except (ValueError, OverflowError):
                pass

    # "January 2026", "Feb 2026" — month + year without day number.
    # Catches stale economic data period references (e.g., "January 2026 Core PCE"
    # when today's release covers February 2026). 45-day threshold matches _url_is_stale
    # month-only check — if the referenced month started >45 days ago, it's stale.
    for m in re.finditer(r'\b(\w+)\s+(\d{4})\b', text):
        month_str, year_str = m.group(1), m.group(2)
        month_num = _MONTHS.get(month_str.lower())
        if month_num:
            try:
                year = int(year_str)
                if 2020 <= year <= now.year + 1:
                    ref_date = datetime(year, month_num, 1, tzinfo=now.tzinfo)
                    if (now - ref_date).days > 45:
                        return True
            except (ValueError, OverflowError):
                pass

    return False


def _normalize_url(url: str) -> str:
    """Normalize URL for dedup: strip query params, trailing slash, www., force https."""
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        scheme = "https"
        host = parsed.hostname or ""
        host = host.lower()
        if host.startswith("www."):
            host = host[4:]
        path = parsed.path.rstrip("/") or ""
        return urlunparse((scheme, host, path, "", "", ""))
    except Exception as e:
        logger.debug(f"URL normalization failed: {e}")
        return url.lower().strip()


def _get_sent_urls(state: dict) -> set[str]:
    """Extract normalized URLs from sent_stories for URL-level dedup."""
    urls = set()
    for story in state.get("sent_stories", []):
        raw = story.get("url", "")
        if raw:
            urls.add(_normalize_url(raw))
    return urls


# Irrelevant title patterns — reject non-financial noise
_IRRELEVANT_TITLE_RE = re.compile(
    r"(?i)\b(?:wordle|crossword|horoscope|recipe|cricket|t20|playlist|"
    r"blood\s+boil|sudoku|astrology|weather\s+forecast|obituar|"
    r"nfl\s+draft|fantasy\s+football|movie\s+review|tv\s+show|"
    r"world\s+cup\s+(?:schedule|cricket|t20)|"
    r"dinar|iraqi\s+dinar|nesara|gesara|quantum\s+financial)\b"
)

# Domain blocklist — conspiracy, disinformation, and low-quality sites
_BLOCKED_DOMAINS = {
    "dinarchronicles.com", "dinarrecaps.com", "inteldinarchronicles.blogspot.com",
    "beforeitsnews.com", "naturalnews.com", "zerohedge.com",
    "infowars.com", "thegatewaypundit.com",
}


def _url_is_blocked_domain(url: str) -> bool:
    """Check if URL is from a blocked domain."""
    try:
        from urllib.parse import urlparse
        host = urlparse(url).hostname or ""
        host = host.lower().removeprefix("www.")
        return host in _BLOCKED_DOMAINS
    except Exception as e:
        logger.debug(f"Blocked domain check failed: {e}")
        return False


def _title_is_irrelevant(title: str) -> bool:
    """Check if a title matches known irrelevant patterns."""
    return bool(_IRRELEVANT_TITLE_RE.search(title))


# Relevance filter for search results — prevents off-topic results from polluting news
_FINANCIAL_KEYWORDS = {
    "stock", "shares", "market", "etf", "fund", "price", "trading",
    "investor", "earnings", "dividend", "portfolio", "equity", "bond",
    "treasury", "fed", "inflation", "recession", "gdp", "tariff",
    "sanctions", "crude", "oil", "gold", "bitcoin", "crypto", "ipo",
    "sec", "hedge", "options", "futures", "rally", "selloff", "sell-off",
    "downturn", "bull", "bear", "volatility", "vix", "s&p", "nasdaq",
    "dow", "index", "sector", "yield", "rate", "bank",
}


def _is_relevant(title: str, snippet: str, ticker: str) -> bool:
    """Check if a search result is relevant to finance/markets or the target ticker."""
    text = (title + " " + snippet).lower()
    return ticker.lower() in text or any(re.search(r'\b' + re.escape(kw) + r'\b', text) for kw in _FINANCIAL_KEYWORDS)


async def _fetch_news(positions: list[dict], sent_urls: set[str] | None = None) -> list[dict]:
    """Fetch news via web search for held tickers + broad market.

    Args:
        positions: Open positions list.
        sent_urls: Set of normalized URLs already sent (for cross-cycle dedup).
    """
    results_all = []
    seen_urls = set()  # Within-cycle URL dedup
    now = datetime.now(_NY)
    today_str = now.strftime("%B %d %Y")

    # Broad market queries (always) — include specific date for freshness
    queries = [
        (f"market news today {today_str}", "market"),
        (f"breaking financial news {today_str}", "market"),
    ]

    # Ticker-specific queries (top 5 positions)
    # For ETFs, search for the underlying asset instead of the ETF ticker
    tickers = [p.get("symbol", "") for p in positions if p.get("symbol")]
    for ticker in tickers[:5]:
        search_term = _ETF_NEWS_QUERIES.get(ticker, f"{ticker} stock")
        queries.append((f"{search_term} news {today_str}", ticker))

    for query, context in queries:
        try:
            allowed, msg = await budget_tracker.check_budget("report_web_search")
            if not allowed:
                break

            from agent.integrations.web_search import search_web
            results = await search_web(query, max_results=2)
            await budget_tracker.record_usage("report_web_search")

            for r in (results or []):
                title = r.get("title", "")
                snippet = r.get("snippet", r.get("body", ""))
                url = r.get("url", r.get("href", ""))
                if not title:
                    continue
                # URL-level dedup: normalize and check against sent + within-cycle
                norm_url = _normalize_url(url)
                if norm_url and norm_url in seen_urls:
                    continue
                if norm_url and sent_urls and norm_url in sent_urls:
                    logger.debug(f"URL dedup filtered: {url[:80]}")
                    continue
                if norm_url:
                    seen_urls.add(norm_url)
                # Irrelevant title filter
                if _title_is_irrelevant(title):
                    logger.debug(f"Irrelevant title filtered: {title[:80]}")
                    continue
                # Blocked domain filter
                if _url_is_blocked_domain(url):
                    logger.debug(f"Blocked domain filtered: {url[:80]}")
                    continue
                if _url_is_stale(url):
                    logger.debug(f"Stale URL filtered: {url[:80]}")
                    continue
                if _snippet_is_stale(title, snippet):
                    logger.debug(f"Stale snippet filtered: {title[:80]}")
                    continue
                if _is_relevant(title, snippet, context):
                    results_all.append({
                        "title": title,
                        "snippet": snippet,
                        "url": url,
                        "ticker_context": context,
                    })

        except Exception as e:
            logger.warning(f"News fetch failed for '{query}': {e}")

        await asyncio.sleep(0.3)

    return results_all


def _format_news_for_prompt(news: list[dict]) -> str:
    """Format fetched news for the Claude prompt."""
    if not news:
        return "No fresh news results available."

    lines = []
    for item in news:
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        url = item.get("url", "")
        ctx = item.get("ticker_context", "")
        line = f"- [{ctx}] {title}"
        if snippet:
            line += f": {snippet}"
        if url:
            line += f" ({url})"
        lines.append(line)

    return sanitize_ingested_content("\n".join(lines), source="ddg_search")


# ============================================================
# Report Generation
# ============================================================

async def _generate_update(positions: list[dict], news: list[dict],
                           sent_headlines: str) -> str | None:
    """Call Claude Sonnet to generate the news update."""
    allowed, msg = await budget_tracker.check_budget("news_update")
    if not allowed:
        logger.warning(f"News update budget exhausted: {msg}")
        return None

    news_text = _format_news_for_prompt(news)
    now_str = datetime.now(_NY).strftime("%I:%M %p ET, %A %B %d")

    held_str = ", ".join(p.get("symbol", "") for p in positions if p.get("symbol") and p.get("held")) or "None (flat)"
    watched_str = ", ".join(p.get("symbol", "") for p in positions if p.get("symbol") and not p.get("held")) or "None"

    user_message = (
        f"Current time: {now_str}\n"
        f"CLIO DPOF positions held: {held_str}\n"
        f"CLIO DPOF trading universe watched (not necessarily held): {watched_str}\n\n"
        f"FRESH NEWS RESULTS:\n{news_text[:5000]}\n\n"
        f"PREVIOUSLY REPORTED STORIES (do NOT repeat these unless there's an update):\n"
        f"{sent_headlines}"
    )

    try:
        client = _get_client()
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, lambda: client.messages.create(
            model=config.fast_model,  # Sonnet for concise
            max_tokens=1500,
            system=cached_system_block(NEWS_SYSTEM_PROMPT),
            messages=[{"role": "user", "content": user_message}],
            output_config={"effort": "medium"},
        ))

        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

        result = text.strip() if text.strip() else None
        if result:
            await budget_tracker.record_usage("news_update")
        return result

    except BudgetExhaustedError as e:
        logger.warning(f"Budget exhausted during news update: {e}")
        return None
    except Exception as e:
        logger.error(f"News update Claude call failed: {e}", exc_info=True)
        return None


def _compute_event_key(title: str, ticker_context: str) -> str:
    """Compute a short event key from title + ticker for event-level dedup.

    Extracts the 4 most significant lowercase words (length >= 4) from the title
    and combines with the ticker context + date prefix. This catches rephrased
    headlines about the same event while allowing same-topic coverage across days.
    """
    # Extract meaningful words (4+ chars, lowercase, alpha only)
    words = re.findall(r'[a-z]{4,}', title.lower())
    # Remove very common filler words
    _STOP = {"this", "that", "with", "from", "have", "been", "will", "would",
             "could", "should", "about", "their", "these", "those", "after",
             "before", "today", "report", "says", "said", "according", "latest",
             "update", "breaking", "news", "just"}
    significant = [w for w in words if w not in _STOP][:4]
    ticker = ticker_context.lower().strip() if ticker_context else "market"
    date_prefix = datetime.now(_NY).strftime("%m%d")
    return f"{ticker}:{date_prefix}_{'_'.join(sorted(significant))}"


def _is_event_duplicate(event_key: str, state: dict) -> bool:
    """Check if an event key already exists in the dedup state."""
    for ek in state.get("sent_event_keys", []):
        if ek.get("key") == event_key:
            return True
    return False


def _extract_headlines_from_news(news: list[dict], state: dict) -> list[dict]:
    """Extract headline entries for dedup state from fetched news.

    Also records event keys for event-level dedup across rephrased headlines.
    """
    now_iso = datetime.now(_NY).isoformat()
    entries = []
    new_event_keys = []
    seen = set()
    for item in news:
        title = item.get("title", "").strip()
        if not title or title in seen:
            continue
        seen.add(title)

        # Event-level dedup: skip if a similar event was already reported
        event_key = _compute_event_key(title, item.get("ticker_context", ""))
        if _is_event_duplicate(event_key, state):
            logger.debug(f"Event dedup: skipping '{title[:60]}' (key={event_key})")
            continue

        entries.append({
            "headline": title[:200],
            "url": item.get("url", ""),
            "first_sent": now_iso,
            "ticker_context": item.get("ticker_context", ""),
        })
        new_event_keys.append({
            "key": event_key,
            "first_sent": now_iso,
        })

    # Append new event keys to state
    state.setdefault("sent_event_keys", []).extend(new_event_keys)

    return entries


# ============================================================
# Delivery
# ============================================================

def _write_to_bridge(report_text: str, positions_count: int) -> None:
    """Write report to the Hydrinia file bridge — no-op unless explicitly re-enabled."""
    try:
        from agent.hydrinia_bridge import bridge_enabled
        if not bridge_enabled():
            return
    except Exception as e:
        logger.debug(f"Hydrinia bridge check failed, skipping bridge write: {e}")
        return
    try:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        timestamp = datetime.now(_NY).strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_news.json"

        entry = {
            "id": f"report_{timestamp}_news",
            "type": "news_update",
            "created": datetime.now(_NY).isoformat(),
            "content": report_text,
            "positions_tracked": positions_count,
        }

        file_path = os.path.join(REPORTS_DIR, filename)
        tmp = file_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(entry, f, indent=2, default=str)
        os.replace(tmp, file_path)

        logger.info(f"News update written to bridge: {filename}")
    except Exception as e:
        logger.error(f"Failed to write news update to bridge: {e}")


async def _send_report(context, report_text: str) -> None:
    """Send report to Telegram."""
    try:
        from agent.integrations.scheduled_tasks import _get_chat_ids, _safe_send
        chat_ids = _get_chat_ids()
        if not chat_ids:
            return

        if len(report_text) <= 4000:
            for cid in chat_ids:
                await _safe_send(context.bot, cid, text=report_text)
        else:
            chunks = [report_text[i:i + 4000] for i in range(0, len(report_text), 4000)]
            for cid in chat_ids:
                for chunk in chunks:
                    await _safe_send(context.bot, cid, text=chunk)

    except Exception as e:
        logger.warning(f"Failed to send news update to Telegram: {e}")


# ============================================================
# Main Entry Point
# ============================================================

async def run_news_update(context) -> None:
    """Main entry point: half-hourly news update.

    Called by JobQueue every 30 minutes. Runs every day including weekends.
    Active window: 8:15 AM - 8:15 PM ET.
    """
    now = datetime.now(_NY)

    # Skip outside active window (8:15 AM - 8:15 PM ET)
    if now.hour < 8 or (now.hour == 8 and now.minute < 15):
        return
    if now.hour > 20 or (now.hour == 20 and now.minute > 15):
        return

    logger.info("Starting news update...")

    try:
        # Load dedup state
        dedup_state = _load_dedup_state()
        dedup_state = _prune_dedup_state(dedup_state)
        _raw_headlines = _get_sent_headlines(dedup_state)
        try:
            from agent.security import sanitize_ingested_content
            sent_headlines = sanitize_ingested_content(_raw_headlines, source="dedup_headlines")
        except ImportError:
            sent_headlines = _raw_headlines

        # Gather data — pass sent URLs for cross-cycle dedup
        sent_urls = _get_sent_urls(dedup_state)
        positions = _get_open_positions()
        news = await _fetch_news(positions, sent_urls=sent_urls)

        # Generate report
        report = await _generate_update(positions, news, sent_headlines)

        if not report:
            logger.warning("News update: no report generated (budget exhausted or API error)")
            return

        # Update dedup state with new headlines (also updates event keys in-place)
        new_entries = _extract_headlines_from_news(news, dedup_state)
        dedup_state["sent_stories"].extend(new_entries)
        dedup_state["last_update"] = datetime.now(_NY).isoformat()
        _save_dedup_state(dedup_state)

        # Deliver
        _write_to_bridge(report, len(positions))
        await _send_report(context, report)

        logger.info(f"News update complete: {len(news)} stories processed, {len(positions)} positions tracked")

    except Exception as e:
        logger.error(f"News update failed: {e}", exc_info=True)
