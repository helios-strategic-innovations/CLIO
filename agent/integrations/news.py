"""
OpenClaw/Moltbot — News Integration
Primary: Polygon/Benzinga news (via Polygon Massive tier)
Fallback: NewsAPI.org for headline aggregation from 80,000+ sources.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from agent.config import config
from agent.rate_limiter import retry_on_transient
from agent.security import sanitize_ingested_content

logger = logging.getLogger("openclaw.integrations.news")

NEWSAPI_BASE_URL = "https://newsapi.org/v2"


@retry_on_transient(max_attempts=3, base_delay=1.0)
async def _newsapi_get(endpoint: str, params: dict, headers: dict) -> dict:
    """Make GET request to NewsAPI with retry on transient errors."""
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        resp = await client.get(f"{NEWSAPI_BASE_URL}{endpoint}", params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()


async def get_news_summary(params: dict[str, Any]) -> str:
    """Fetch news — tries Polygon/Benzinga first, falls back to NewsAPI."""
    topic = params.get("topic", "markets")
    sources = params.get("sources", ["all"])
    timeframe = params.get("timeframe", "1d")
    symbol = params.get("symbol")

    # If a specific symbol is given and Polygon is available, try that first
    if symbol and config.polygon_api_key:
        try:
            from agent.integrations.polygon import get_market_data
            result = await get_market_data({"symbol": symbol, "data_type": "news", "limit": 10})
            if result and "ERROR" not in result and "No news" not in result:
                return result
        except Exception as e:
            logger.debug(f"Polygon news fallthrough for {symbol}: {e}")

    # Fall back to NewsAPI
    if not config.news_api_key:
        return f"ERROR: No news sources available. Configure POLYGON_API_KEY or NEWS_API_KEY."

    try:
        articles = await _fetch_newsapi(sources, topic, timeframe)
        if not articles:
            # Try top-headlines as fallback
            articles = await _fetch_top_headlines(topic)

        if not articles:
            return f"No news found for '{topic}'."

        lines = [f"News Summary — '{topic}' ({len(articles)} articles):"]
        for i, article in enumerate(articles[:15], 1):
            title = article.get("title", "No title")
            source = article.get("source", {}).get("name", "Unknown")
            published = article.get("publishedAt", "")[:10]
            desc = article.get("description", "")
            url = article.get("url", "")

            source_name = str(source) if isinstance(source, str) else "Unknown"
            safe_title = sanitize_ingested_content(title, f"news:{source_name}")
            safe_desc = sanitize_ingested_content(desc, f"news:{source_name}") if desc else ""

            lines.append(
                f"\n{i}. [{source}] {published}\n"
                f"   {safe_title}\n"
                f"   {safe_desc}\n"
                f"   {url}"
            )

        return "\n".join(lines)

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            return "ERROR: NEWS_API_KEY is invalid."
        if e.response.status_code == 429:
            return "ERROR: NewsAPI rate limit exceeded."
        if e.response.status_code == 426:
            return "ERROR: NewsAPI requires a paid plan for this query. Try a shorter timeframe or /top-headlines."
        logger.error(f"NewsAPI HTTP error: {e}", exc_info=True)
        return f"ERROR: NewsAPI error — {e.response.status_code}"
    except Exception as e:
        logger.error(f"News fetch failed: {e}", exc_info=True)
        return f"ERROR: News fetch failed — {e}"


async def _fetch_newsapi(sources: list[str], topic: str, timeframe: str) -> list[dict]:
    """Fetch articles from NewsAPI /everything endpoint."""
    hours_map = {"1h": 1, "4h": 4, "1d": 24, "1w": 168}
    hours = hours_map.get(timeframe, 24)
    # NewsAPI only accepts date format YYYY-MM-DD (not full ISO timestamps)
    from_date = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%d")

    headers = {"X-Api-Key": config.news_api_key}

    params: dict[str, Any] = {
        "q": topic,
        "from": from_date,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": 20,
    }

    # Add specific source IDs if requested
    source_map = {
        "bloomberg": "bloomberg",
        "reuters": "reuters",
        "cnbc": "cnbc",
        "wsj": "the-wall-street-journal",
        "ft": "financial-times",
        "bbc": "bbc-news",
        "cnn": "cnn",
    }

    if sources and "all" not in sources:
        source_ids = [source_map[s.lower()] for s in sources if s.lower() in source_map]
        if source_ids:
            params["sources"] = ",".join(source_ids)

    # Remove None values
    params = {k: v for k, v in params.items() if v is not None}

    data = await _newsapi_get("/everything", params, headers)
    return data.get("articles", [])


async def _fetch_top_headlines(topic: str) -> list[dict]:
    """Fallback: fetch from /top-headlines (works on free tier without date restrictions)."""
    headers = {"X-Api-Key": config.news_api_key}

    params: dict[str, Any] = {
        "category": "business",
        "language": "en",
        "pageSize": 20,
        "country": "us",
    }

    # If topic is specific enough, use it as a query instead of category
    general_topics = {"markets", "general", "business", "finance", "stocks"}
    if topic.lower() not in general_topics:
        params["q"] = topic
        del params["category"]
        del params["country"]

    data = await _newsapi_get("/top-headlines", params, headers)
    return data.get("articles", [])
