"""
OpenClaw/Moltbot — Research Tool Definitions

Tool schemas for web search and content fetching.
"""


def get_research_tools() -> list[dict]:
    return [
        {
            "name": "web_search",
            "description": "Search the internet via DuckDuckGo. Returns titles, URLs, and snippets for web results. Use for research, news lookup, or finding references.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'algorithmic trading strategies', 'SPY options flow analysis').",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max results to return (default 5, max 10).",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "fetch_and_analyze",
            "description": "Fetch a web page URL and extract its text content for analysis. Returns the page's main text content. Use after web_search to read a specific result.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full URL to fetch (http or https).",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Max characters to extract (default 10000, max 50000).",
                    },
                },
                "required": ["url"],
            },
        },
    ]
