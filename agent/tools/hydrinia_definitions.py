"""
OpenClaw/Moltbot — Hydrinia Monitoring Tool Definitions

Tools for reading Hydrinia logs, state files, status, portfolio, and performance.
"""


def get_hydrinia_tools() -> list[dict]:
    return [
        {
            "name": "get_hydrinia_logs",
            "description": "Read recent Hydrinia trading bot log entries. Supports filtering by level and keyword.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "lines": {"type": "integer", "description": "Number of recent lines to read (default: 100, max: 500)"},
                    "level": {"type": "string", "enum": ["ERROR", "WARNING", "INFO", "DEBUG"], "description": "Filter by log level"},
                    "keyword": {"type": "string", "description": "Filter lines containing this keyword"},
                },
                "required": [],
            },
        },
        {
            "name": "get_hydrinia_state",
            "description": "Read a Hydrinia state file (JSON). Available: position_log, daily_metrics, agent_memory, learning_log, knowledge_base, reserve_state, overnight_queue, trade_outcomes, role_elections, employee_of_month_state.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file": {"type": "string", "description": "State file name (e.g., 'position_log.json', 'daily_metrics.json')"},
                },
                "required": ["file"],
            },
        },
        {
            "name": "get_hydrinia_status",
            "description": "Check if Hydrinia is running, its PID, uptime, and last activity.",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "get_hydrinia_portfolio",
            "description": "Get current Hydrinia portfolio positions from state files.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "detail": {"type": "boolean", "description": "Include per-position details (default: true)"},
                },
                "required": [],
            },
        },
        {
            "name": "get_hydrinia_performance",
            "description": "Get Hydrinia performance metrics: daily P&L, win rate, trade count from state files.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "description": "Number of days to look back (default: 7)"},
                },
                "required": [],
            },
        },
    ]
