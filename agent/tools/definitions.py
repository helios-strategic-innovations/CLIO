"""
OpenClaw/Moltbot Tool Definitions
"""

import functools


def _hydrinia_bridge_enabled() -> bool:
    """Hydrinia is permanently off (operator direction, 2026-09-24). Its
    monitoring/control tools are only advertised to the LLM when the operator
    explicitly re-enables the bridge (agent/hydrinia_bridge.py)."""
    try:
        from agent.hydrinia_bridge import bridge_enabled
        return bridge_enabled()
    except Exception:
        return False


# Cache: get_all_tools() is invoked multiple times per API call (initial +
# retries) in core.py, and rebuilding the full ~40-tool static list on every
# call is wasted work — none of these sub-lists depend on anything that
# changes mid-process. Cached here via a bridge-INDEPENDENT helper.
#
# The Hydrinia bridge flag is intentionally kept OUT of this cache: it is a
# startup-only env var in production (a restart is required to flip it,
# which clears this cache along with everything else) but tests toggle it
# mid-process with monkeypatch and expect get_all_tools() to reflect that
# immediately. get_all_tools() itself therefore stays uncached and re-checks
# the bridge on every call; only the bridge-independent bulk is memoized.
@functools.lru_cache(maxsize=1)
def _bridge_independent_tools() -> list[dict]:
    from agent.tools.file_definitions import get_file_tools
    from agent.tools.knowledge_definitions import get_knowledge_tools
    from agent.tools.research_definitions import get_research_tools
    return [
        *_market_tools(), *_trading_tools(),
        *get_file_tools(),
        *_communication_tools(),
        *get_knowledge_tools(), *get_research_tools(),
        *_dpof_tools(),
    ]


def get_all_tools() -> list[dict]:
    from agent.tools.hydrinia_definitions import get_hydrinia_tools
    _bridge = _hydrinia_bridge_enabled()
    return [
        *_bridge_independent_tools(),
        *_code_tools(_bridge),
        *(get_hydrinia_tools() if _bridge else []),
        *_bot_control_tools(_bridge),
    ]


def _market_tools() -> list[dict]:
    return [
        {"name": "get_news_summary", "description": "Fetch and summarize latest news from connected media sources.", "input_schema": {"type": "object", "properties": {"sources": {"type": "array", "items": {"type": "string", "enum": ["bloomberg", "economist", "japantimes", "scmp", "all"]}}, "topic": {"type": "string"}, "symbol": {"type": "string", "description": "Stock symbol for symbol-specific news (uses Polygon/Benzinga)"}, "timeframe": {"type": "string", "enum": ["1h", "4h", "1d", "1w"]}}, "required": ["sources"]}},
        {"name": "get_market_data", "description": "Fetch current market data for a symbol.", "input_schema": {"type": "object", "properties": {"symbol": {"type": "string"}, "data_type": {"type": "string", "enum": ["quote", "bars", "snapshot", "news", "details", "earnings", "ratings", "guidance", "calendar"]}, "timeframe": {"type": "string", "enum": ["1m", "5m", "15m", "1h", "1d", "1w"]}, "limit": {"type": "integer"}}, "required": ["symbol"]}},
        {"name": "get_options_flow", "description": "Monitor unusual options activity.", "input_schema": {"type": "object", "properties": {"symbol": {"type": "string"}, "min_premium": {"type": "integer"}, "flow_type": {"type": "string", "enum": ["all", "sweeps", "blocks", "unusual_volume"]}}, "required": []}},
        {"name": "get_darkpool_data", "description": "Fetch dark pool trading data.", "input_schema": {"type": "object", "properties": {"symbol": {"type": "string"}, "timeframe": {"type": "string", "enum": ["1d", "1w", "1m"]}}, "required": ["symbol"]}},
        {"name": "get_market_tide", "description": "Get market-wide options flow sentiment (bullish/bearish).", "input_schema": {"type": "object", "properties": {}, "required": []}},
        {"name": "get_net_flow_timeline", "description": "Get per-ticker net options flow in 5-min buckets (call vs put premium over the day). Like a text version of bigshort.com flow chart.", "input_schema": {"type": "object", "properties": {"symbol": {"type": "string", "description": "Stock ticker (default: SPY)"}, "bucket_minutes": {"type": "integer", "description": "Bucket size in minutes (default: 5)"}, "date": {"type": "string", "description": "Trading date YYYY-MM-DD (default: today)"}}, "required": []}},
        {"name": "get_congress_trades", "description": "Get recent congressional trading activity.", "input_schema": {"type": "object", "properties": {"symbol": {"type": "string", "description": "Filter by stock symbol"}}, "required": []}},
        {"name": "get_prediction_consensus", "description": "Get Kalshi prediction market consensus for major macro events (Fed rate, CPI, GDP, etc.). Returns market-implied probability distributions.", "input_schema": {"type": "object", "properties": {"topic": {"type": "string", "enum": ["fed", "cpi", "gdp", "payrolls", "recession", "rates"], "description": "Macro topic to query"}}, "required": ["topic"]}},
        {"name": "get_prediction_markets", "description": "Search/browse Kalshi prediction markets by keywords or series.", "input_schema": {"type": "object", "properties": {"keywords": {"type": "string", "description": "Search keywords (e.g., 'tariff', 'recession')"}, "series": {"type": "string", "description": "Kalshi series ticker (e.g., 'KXFED', 'KXCPI')"}, "limit": {"type": "integer", "description": "Max results (default 10)"}}, "required": []}},
        {"name": "get_prediction_event", "description": "Get detailed probability distribution for a specific Kalshi event.", "input_schema": {"type": "object", "properties": {"event_ticker": {"type": "string", "description": "Kalshi event ticker (e.g., 'KXFED-26MAR')"}}, "required": ["event_ticker"]}},
    ]


def _trading_tools() -> list[dict]:
    return [
        {"name": "execute_trade", "description": "Place a trade via Alpaca. REQUIRES confirmation.", "input_schema": {"type": "object", "properties": {"symbol": {"type": "string"}, "side": {"type": "string", "enum": ["buy", "sell"]}, "qty": {"type": "number"}, "order_type": {"type": "string", "enum": ["market", "limit", "stop", "stop_limit"]}, "limit_price": {"type": "number"}, "stop_price": {"type": "number"}, "time_in_force": {"type": "string", "enum": ["day", "gtc", "ioc", "fok"]}, "extended_hours": {"type": "boolean", "description": "Submit order for pre/post market trading (4AM-9:30AM, 4PM-8PM ET). LIMIT ORDERS ONLY. Forces time_in_force=day."}, "trade_id": {"type": "string", "description": "Optional ID linking this trade to a trade idea (e.g. ti_20260301_NVDA). Alpaca stores this as client_order_id for position tracking."}}, "required": ["symbol", "side", "qty", "order_type"]}},
        {"name": "get_portfolio", "description": "Get current portfolio positions and P&L.", "input_schema": {"type": "object", "properties": {"include_closed": {"type": "boolean"}}, "required": []}},
        {"name": "get_account_info", "description": "Get account balance and buying power.", "input_schema": {"type": "object", "properties": {}, "required": []}},
        {"name": "get_option_chain", "description": "Fetch options chain for a symbol with strikes, expirations, Greeks, and bid/ask.", "input_schema": {"type": "object", "properties": {"underlying": {"type": "string", "description": "Underlying stock symbol (e.g. SPY, AAPL)"}, "expiration": {"type": "string", "description": "Expiration date YYYY-MM-DD (optional — shows nearest if omitted)"}, "option_type": {"type": "string", "enum": ["call", "put", "both"], "description": "Filter by call/put (default: both)"}, "strike_min": {"type": "number", "description": "Minimum strike price filter"}, "strike_max": {"type": "number", "description": "Maximum strike price filter"}, "limit": {"type": "integer", "description": "Max contracts to return (default 20)"}}, "required": ["underlying"]}},
        {"name": "get_option_quote", "description": "Get real-time quote + Greeks for a specific option contract by OCC symbol.", "input_schema": {"type": "object", "properties": {"occ_symbol": {"type": "string", "description": "OCC option symbol (e.g. SPY250321C00590000)"}}, "required": ["occ_symbol"]}},
        {"name": "execute_options_trade", "description": "Trade an option contract via Alpaca. Uses mid-point limit pricing. REQUIRES confirmation.", "input_schema": {"type": "object", "properties": {"occ_symbol": {"type": "string", "description": "OCC option symbol (e.g. SPY250321C00590000)"}, "side": {"type": "string", "enum": ["buy", "sell"], "description": "Buy or sell"}, "qty": {"type": "integer", "description": "Number of contracts (1-100)"}, "position_intent": {"type": "string", "enum": ["buy_to_open", "buy_to_close", "sell_to_open", "sell_to_close", "bto", "btc", "sto", "stc"], "description": "Position intent (default: buy_to_open for buy, sell_to_close for sell)"}, "limit_price_override": {"type": "number", "description": "Override computed mid-point limit price (optional)"}, "spread_adjustment": {"type": "number", "description": "Fraction of spread to adjust from mid toward taker side (default 0.02 = 2%)"}, "time_in_force": {"type": "string", "enum": ["day", "gtc", "ioc", "fok"], "description": "Time in force (default: day)"}, "trade_id": {"type": "string", "description": "Optional ID linking this trade to a trade idea"}}, "required": ["occ_symbol", "side", "qty"]}},
    ]


def _code_tools(bridge_enabled: bool = False) -> list[dict]:
    if not bridge_enabled:
        # Hydrinia is permanently off — hydrinia.py isn't advertised to the LLM.
        return []
    return [
        {"name": "modify_hydrinia", "description": "Read, edit, or manage hydrinia.py. Edit replaces entire file (provide complete new content). Edits are git-committed. REQUIRES confirmation.", "input_schema": {"type": "object", "properties": {"action": {"type": "string", "enum": ["read", "list_files", "edit", "git_log", "git_diff"]}, "file_path": {"type": "string"}, "changes": {"type": "string", "description": "Complete new file content (replaces entire file)"}, "commit_message": {"type": "string"}}, "required": ["action"]}},
    ]


def _dpof_tools() -> list[dict]:
    return [
        {
            "name": "get_dpof_status",
            "description": "Get CLIO's own DPOF long-options engine status: engine/gates/equity status, open positions, recent decisions, or config. Read-only.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "enum": ["status", "positions", "decisions", "config"],
                        "description": "Which view to return (default: status).",
                    },
                },
                "required": [],
            },
        },
    ]


def _communication_tools() -> list[dict]:
    return [
        {
            "name": "read_emails",
            "description": "Read emails from a configured account. Supports folder selection, search, and unread filtering.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "account": {
                        "type": "string",
                        "enum": ["personal_gmail", "personal_atlas", "bot_atlas", "bot_gmail"],
                        "description": "Email account to read from. Default: personal_gmail.",
                    },
                    "folder": {
                        "type": "string",
                        "enum": ["inbox", "sent", "flagged"],
                        "description": "Mailbox folder (default: inbox).",
                    },
                    "limit": {"type": "integer", "description": "Max emails to return (default 10, max 20)."},
                    "unread_only": {"type": "boolean", "description": "Only return unread emails."},
                    "search_query": {
                        "type": "string",
                        "description": "Search by subject or sender keyword.",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "send_email",
            "description": "Send an email. REQUIRES confirmation. Default sender: bot@example.org.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient(s), comma-separated."},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                    "cc": {"type": "string", "description": "CC recipients, comma-separated."},
                    "from_account": {
                        "type": "string",
                        "enum": ["bot_atlas", "bot_gmail"],
                        "description": "Sending account (default: bot_atlas = bot@example.org).",
                    },
                    "reply_to_id": {"type": "string", "description": "Email ID to reply to."},
                },
                "required": ["to", "subject", "body"],
            },
        },
        {
            "name": "email_trade_summary",
            "description": "Send a CLIO DPOF trading summary email with account equity/day P&L and DPOF positions/decisions. REQUIRES confirmation.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient(s), comma-separated."},
                    "summary_type": {
                        "type": "string",
                        "enum": ["daily", "trade"],
                        "description": "Summary type: 'daily' for full report, 'trade' for last trade.",
                    },
                    "from_account": {
                        "type": "string",
                        "enum": ["bot_atlas", "bot_gmail"],
                        "description": "Sending account (default: bot_atlas).",
                    },
                },
                "required": ["to"],
            },
        },
    ]


def _bot_control_tools(bridge_enabled: bool = False) -> list[dict]:
    tools = [
        {
            "name": "get_bot_processes",
            "description": "List running bot-related processes (openclaw, moltbot).",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    ]
    if bridge_enabled:
        # Hydrinia is permanently off by default — start/stop/restart is not
        # advertised to the LLM unless the operator explicitly re-enables the bridge.
        tools.insert(0, {
            "name": "control_hydrinia",
            "description": "Start, stop, restart, or emergency-stop the Hydrinia trading bot. REQUIRES confirmation.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["start", "stop", "restart", "emergency_stop", "status"]},
                },
                "required": ["action"],
            },
        })
    return tools
