"""
OpenClaw/Moltbot — Knowledge Management Tool Schemas

4 tools: save_knowledge, recall_knowledge, get_knowledge_summary, update_priors
"""

# 16 categories mapped to directory paths
KNOWLEDGE_CATEGORIES = [
    "trading_strategies",
    "trading_trades",
    "trading_market_analysis",
    "trading_conversations",
    "trading_bot_status",
    "personal_preferences",
    "personal_conversations",
    "personal_notes",
    "email_gmail_personal",
    "email_atlas_personal",
    "email_atlas_bot",
    "email_gmail_bot",
    "academic_courses",
    "academic_research",
    "career_applications",
    "career_networking",
]


def get_knowledge_tools() -> list[dict]:
    return [
        {
            "name": "save_knowledge",
            "description": (
                "Save important information to persistent knowledge base. "
                "Use for strategies, trade analysis, market insights, personal preferences, "
                "email summaries, academic notes, career contacts, etc. "
                "PROACTIVELY save anything worth remembering across conversations. "
                "Trading knowledge is automatically mirrored to /home/agent/tradingbot/moltbot_knowledge/."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short descriptive title (e.g., 'SPY Gamma Squeeze Analysis').",
                    },
                    "category": {
                        "type": "string",
                        "enum": KNOWLEDGE_CATEGORIES,
                        "description": "Knowledge category determining storage location.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full knowledge content to persist.",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Searchable tags (e.g., ['SPY', 'gamma', 'options']).",
                    },
                    "source": {
                        "type": "string",
                        "description": "Origin of the knowledge (e.g., 'conversation', 'email', 'analysis').",
                    },
                    "update_existing": {
                        "type": "boolean",
                        "description": "If true, updates existing entry with same title instead of creating new. Use for canonical single-entry concepts (bot status, trait roster, active directives).",
                    },
                },
                "required": ["title", "category", "content"],
            },
        },
        {
            "name": "recall_knowledge",
            "description": (
                "Search persistent knowledge base by category, keywords, or tags. "
                "Returns matching entries with metadata. Use include_content=true to read full content."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Filter by category (exact match or prefix, e.g., 'trading' matches all trading_*).",
                    },
                    "keywords": {
                        "type": "string",
                        "description": "Search keywords matched against titles and tags.",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter entries that have any of these tags.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default 10).",
                    },
                    "include_content": {
                        "type": "boolean",
                        "description": "If true, include full content of matching entries (capped at 5).",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "get_knowledge_summary",
            "description": (
                "Get overview of all stored knowledge: total entries, counts per category, "
                "and the 3 most recent entries per category."
            ),
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "update_priors",
            "description": (
                "Update your current operational priors — a living document of your strongest "
                "current beliefs about markets, bot state, and active problems. This is injected "
                "into every system prompt. Use for operational, specific, actionable beliefs. "
                "NOT for macro/thematic views (that's the Living Perspective). "
                "This REPLACES the entire document (not appends). Keep concise — max 1500 chars."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Full content of current priors. This REPLACES the entire document.",
                    }
                },
                "required": ["content"],
            },
        },
    ]
