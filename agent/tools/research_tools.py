"""
OpenClaw/Moltbot — Research Tool Handlers

Interactive web search and content fetching tools for Claude tool-use loop.
These are invoked when the user asks Moltbot to research something via Telegram.
The scheduled learning engine calls web_search functions directly.
"""

import logging

from agent.integrations.web_search import search_web, fetch_page_content
from agent.security import sanitize_ingested_content

logger = logging.getLogger("openclaw.research_tools")


async def handle_web_search(p: dict) -> str:
    """Search the web via DuckDuckGo. Returns formatted results."""
    query = p.get("query", "").strip()
    if not query:
        return "ERROR: 'query' parameter is required."

    if len(query) > 500:
        query = query[:500]

    try:
        max_results = max(1, min(int(p.get("max_results", 5)), 10))
    except (ValueError, TypeError):
        max_results = 5

    results = await search_web(query, max_results=max_results)

    if not results:
        return f"No results found for: {query}"

    lines = [f"Found {len(results)} result(s) for: {query}\n"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"{i}. {r.get('title', '(no title)')}\n"
            f"   URL: {r.get('url', r.get('href', ''))}\n"
            f"   {r.get('snippet', r.get('body', ''))}\n"
        )

    return sanitize_ingested_content("\n".join(lines), source=f"web_search:{query}")


async def handle_fetch_and_analyze(p: dict) -> str:
    """Fetch a URL and return extracted text content for analysis."""
    url = p.get("url", "").strip()
    if not url:
        return "ERROR: 'url' parameter is required."

    if len(url) > 2000:
        return "ERROR: URL too long (max 2000 characters)."

    if not url.startswith(("http://", "https://")):
        return "ERROR: URL must start with http:// or https://"

    try:
        max_chars = max(1000, min(int(p.get("max_chars", 10_000)), 50_000))
    except (ValueError, TypeError):
        max_chars = 10_000

    content = await fetch_page_content(url, max_chars=max_chars)
    return content  # fetch_page_content already sanitizes via sanitize_ingested_content
