"""
OpenClaw/Moltbot Tool Executor
Routes tool calls to real integrations or stubs.
"""

import asyncio
import os
import subprocess
import logging
from typing import Any

from agent.config import config
from agent.security import log_security_event

logger = logging.getLogger("openclaw.tools")

# S190 M13: env-tunable subprocess timeout for git ops. Default 30s matches the
# previous hardcoded value (preserved to keep cold-repo first-pulls working).
# Operators with fast local-only repos can lower via CLIO_SUBPROCESS_TIMEOUT=10.
# Range clamped to [5, 120] to prevent foot-guns.
try:
    _SUBPROCESS_TIMEOUT = max(5, min(120, int(os.environ.get("CLIO_SUBPROCESS_TIMEOUT", "30"))))
except (TypeError, ValueError):
    _SUBPROCESS_TIMEOUT = 30

# Hydrinia is permanently off (operator direction, 2026-09-24). These tool names
# are gated on agent.hydrinia_bridge.bridge_enabled() at the dispatch choke point
# below — a call reaching execute_tool() for one of these while the bridge is
# disabled NEVER reaches its handler; it gets a clear retired-tool string instead.
# (Definitions.py also omits them from the list the LLM sees when the bridge is
# off — this is the second, independent gate for anything that still names them,
# e.g. stale conversation history.)
_HYDRINIA_BRIDGE_GATED_TOOLS = frozenset({
    "get_hydrinia_logs", "get_hydrinia_state", "get_hydrinia_status",
    "get_hydrinia_portfolio", "get_hydrinia_performance",
    "control_hydrinia", "modify_hydrinia",
})


def _hydrinia_bridge_enabled() -> bool:
    try:
        from agent.hydrinia_bridge import bridge_enabled
        return bridge_enabled()
    except Exception as e:
        logger.debug(f"Hydrinia bridge check failed, defaulting to disabled: {e}")
        return False


async def execute_tool(tool_name: str, tool_input: dict[str, Any]) -> str:
    if tool_name in _HYDRINIA_BRIDGE_GATED_TOOLS and not _hydrinia_bridge_enabled():
        return (
            f"RETIRED: Hydrinia is permanently off — '{tool_name}' is unavailable. "
            "CLIO trades its own SPY/QQQ book via the DPOF engine now; use get_dpof_status, "
            "get_portfolio, or get_account_info instead."
        )

    from agent.tools.file_tools import (
        handle_read_file, handle_write_file, handle_edit_file,
        handle_list_files, handle_search_files,
        handle_git_status, handle_git_diff, handle_run_command,
    )
    from agent.tools.hydrinia_tools import (
        handle_hydrinia_logs, handle_hydrinia_state, handle_hydrinia_status,
        handle_hydrinia_portfolio, handle_hydrinia_performance,
    )
    from agent.tools.bot_control import handle_control_hydrinia, handle_get_bot_processes
    from agent.tools.dpof_tools import handle_dpof_status
    from agent.tools.email_tools import handle_read_emails, handle_send_email, handle_email_trade_summary
    from agent.tools.knowledge_tools import handle_save_knowledge, handle_recall_knowledge, handle_get_knowledge_summary, handle_update_priors
    from agent.tools.research_tools import handle_web_search, handle_fetch_and_analyze

    handlers = {
        "get_news_summary": _handle_news,
        "get_market_data": _handle_market_data,
        "get_options_flow": _handle_options_flow,
        "get_darkpool_data": _handle_darkpool,
        "get_market_tide": _handle_market_tide,
        "get_net_flow_timeline": _handle_net_flow_timeline,
        "get_congress_trades": _handle_congress_trades,
        "get_prediction_consensus": _handle_prediction_consensus,
        "get_prediction_markets": _handle_prediction_markets,
        "get_prediction_event": _handle_prediction_event,
        "execute_trade": _handle_trade,
        "get_portfolio": _handle_portfolio,
        "get_account_info": _handle_account,
        "get_option_chain": _handle_option_chain,
        "get_option_quote": _handle_option_quote,
        "execute_options_trade": _handle_options_trade,
        "modify_hydrinia": _handle_code,
        # File tools
        "read_file": handle_read_file,
        "write_file": handle_write_file,
        "edit_file": handle_edit_file,
        "list_files": handle_list_files,
        "search_files": handle_search_files,
        "git_status": handle_git_status,
        "git_diff": handle_git_diff,
        "run_command": handle_run_command,
        # Hydrinia monitoring tools
        "get_hydrinia_logs": handle_hydrinia_logs,
        "get_hydrinia_state": handle_hydrinia_state,
        "get_hydrinia_status": handle_hydrinia_status,
        "get_hydrinia_portfolio": handle_hydrinia_portfolio,
        "get_hydrinia_performance": handle_hydrinia_performance,
        # Bot control tools
        "control_hydrinia": handle_control_hydrinia,
        "get_bot_processes": handle_get_bot_processes,
        # CLIO-native DPOF status (read-only)
        "get_dpof_status": handle_dpof_status,
        # Communication
        "read_emails": handle_read_emails,
        "send_email": handle_send_email,
        "email_trade_summary": handle_email_trade_summary,
        # Knowledge management
        "save_knowledge": handle_save_knowledge,
        "recall_knowledge": handle_recall_knowledge,
        "get_knowledge_summary": handle_get_knowledge_summary,
        "update_priors": handle_update_priors,
        # Research / web search
        "web_search": handle_web_search,
        "fetch_and_analyze": handle_fetch_and_analyze,
    }
    handler = handlers.get(tool_name)
    if not handler:
        return f"ERROR: No handler for '{tool_name}'"
    from agent.rate_limiter import BudgetExhaustedError
    try:
        return await handler(tool_input)
    except BudgetExhaustedError as e:
        logger.info(f"Budget exhausted for {tool_name}: {e}")
        return f"BUDGET EXHAUSTED: {e}"
    except Exception as e:
        logger.error(f"Tool error ({tool_name}): {e}", exc_info=True)
        return f"ERROR: {tool_name} failed: {str(e)}"


# ============================================================
# LIVE INTEGRATIONS
# ============================================================

async def _handle_news(p: dict) -> str:
    from agent.integrations.news import get_news_summary
    return await get_news_summary(p)


async def _handle_market_data(p: dict) -> str:
    # Prefer Polygon, fall back to Alpaca
    if config.polygon_api_key:
        from agent.integrations.polygon import get_market_data
        return await get_market_data(p)
    elif config.alpaca_api_key:
        from agent.integrations.alpaca import get_market_data
        return await get_market_data(p)
    return "ERROR: No market data provider configured (need POLYGON_API_KEY or ALPACA_API_KEY)."


async def _handle_options_flow(p: dict) -> str:
    if config.uw_api_key:
        from agent.integrations.unusual_whales import get_options_flow
        return await get_options_flow(p)
    return "ERROR: UW_API_KEY not configured for options flow."


async def _handle_darkpool(p: dict) -> str:
    if config.uw_api_key:
        from agent.integrations.unusual_whales import get_darkpool_data
        return await get_darkpool_data(p)
    return "ERROR: UW_API_KEY not configured for dark pool data."


async def _handle_market_tide(p: dict) -> str:
    if config.uw_api_key:
        from agent.integrations.unusual_whales import get_market_tide
        return await get_market_tide(p)
    return "ERROR: UW_API_KEY not configured for market tide."


async def _handle_net_flow_timeline(p: dict) -> str:
    if config.uw_api_key:
        from agent.integrations.unusual_whales import get_net_flow_timeline
        return await get_net_flow_timeline(p)
    return "ERROR: UW_API_KEY not configured for net flow timeline."


async def _handle_congress_trades(p: dict) -> str:
    if config.uw_api_key:
        from agent.integrations.unusual_whales import get_congress_trades
        return await get_congress_trades(p)
    return "ERROR: UW_API_KEY not configured for congress trades."


async def _handle_prediction_consensus(p: dict) -> str:
    from agent.integrations import kalshi
    if kalshi.is_available():
        return await kalshi.get_prediction_consensus(p)
    return "ERROR: Kalshi unavailable — API key and/or private-key pem not configured."


async def _handle_prediction_markets(p: dict) -> str:
    from agent.integrations import kalshi
    if kalshi.is_available():
        return await kalshi.get_prediction_markets(p)
    return "ERROR: Kalshi unavailable — API key and/or private-key pem not configured."


async def _handle_prediction_event(p: dict) -> str:
    from agent.integrations import kalshi
    if kalshi.is_available():
        return await kalshi.get_prediction_event(p)
    return "ERROR: Kalshi unavailable — API key and/or private-key pem not configured."


async def _handle_trade(p: dict) -> str:
    if config.alpaca_api_key:
        from agent.integrations.alpaca import execute_trade
        return await execute_trade(p)
    return "ERROR: ALPACA_API_KEY not configured for trading."


async def _handle_portfolio(p: dict) -> str:
    if config.alpaca_api_key:
        from agent.integrations.alpaca import get_portfolio
        return await get_portfolio(p)
    return "ERROR: ALPACA_API_KEY not configured."


async def _handle_account(p: dict) -> str:
    if config.alpaca_api_key:
        from agent.integrations.alpaca import get_account_info
        return await get_account_info(p)
    return "ERROR: ALPACA_API_KEY not configured."


async def _handle_option_chain(p: dict) -> str:
    if config.alpaca_api_key:
        from agent.integrations.alpaca_options import get_option_chain
        return await get_option_chain(p)
    return "ERROR: ALPACA_API_KEY not configured for options."


async def _handle_option_quote(p: dict) -> str:
    if config.alpaca_api_key:
        from agent.integrations.alpaca_options import get_option_quote
        return await get_option_quote(p)
    return "ERROR: ALPACA_API_KEY not configured for options."


async def _handle_options_trade(p: dict) -> str:
    if config.alpaca_api_key:
        from agent.integrations.alpaca_options import execute_options_trade
        return await execute_options_trade(p)
    return "ERROR: ALPACA_API_KEY not configured for options trading."


# ============================================================
# STUBS (not yet implemented)
# ============================================================

def _handle_code_sync(p: dict) -> str:
    """Synchronous modify_hydrinia implementation. Runs in thread executor."""
    action = p.get("action", "read")
    base_dir = "/home/agent/tradingbot"

    if action == "list_files":
        try:
            _MAX_LIST_FILES = 500
            files = []
            _truncated = False
            for root, dirs, filenames in os.walk(base_dir):
                for f in filenames:
                    files.append(os.path.relpath(os.path.join(root, f), base_dir))
                    if len(files) >= _MAX_LIST_FILES:
                        _truncated = True
                        break
                if _truncated:
                    break
            result = "Files:\n" + "\n".join(files) if files else "No files found."
            if _truncated:
                result += f"\n\n[truncated at {_MAX_LIST_FILES} files]"
            return result
        except Exception as e:
            return f"ERROR: {e}"

    elif action == "read":
        file_path = p.get("file_path", "hydrinia.py")
        full_path = os.path.join(base_dir, file_path)
        real_path = os.path.realpath(full_path)
        if not real_path.startswith(os.path.realpath(base_dir) + os.sep):
            return "ERROR: Path traversal detected."
        try:
            with open(real_path, "r") as f:
                content = f.read()
            from agent.security import sanitize_ingested_content
            return sanitize_ingested_content(f"Contents of {file_path}:\n```\n{content}\n```", f"hydrinia:{file_path}")
        except FileNotFoundError:
            return f"File not found: {file_path}"

    elif action == "edit":
        # NOTE: "edit" is a FULL FILE REPLACEMENT — 'changes' contains the complete new
        # file content, not a diff or patch. The entire file is overwritten and committed.
        file_path = p.get("file_path", "hydrinia.py")
        changes = p.get("changes", "")
        commit_msg = p.get("commit_message", f"Moltbot: edit {file_path}")
        full_path = os.path.join(base_dir, file_path)
        real_path = os.path.realpath(full_path)
        if not real_path.startswith(os.path.realpath(base_dir) + os.sep):
            return "ERROR: Path traversal detected."
        try:
            tmp_path = real_path + ".tmp"
            with open(tmp_path, "w") as f:
                f.write(changes)
            os.replace(tmp_path, real_path)
            subprocess.run(["git", "add", "--", file_path], cwd=base_dir, check=True, timeout=_SUBPROCESS_TIMEOUT, shell=False)
            subprocess.run(["git", "commit", "-m", commit_msg], cwd=base_dir, check=True, timeout=_SUBPROCESS_TIMEOUT, shell=False)
            return f"Edited {file_path} and committed: {commit_msg}"
        except subprocess.TimeoutExpired:
            return f"ERROR: Command timed out after {_SUBPROCESS_TIMEOUT}s"
        except Exception as e:
            return f"ERROR: {e}"

    elif action == "git_log":
        try:
            from agent.security import sanitize_ingested_content
            r = subprocess.run(["git", "log", "--oneline", "-20"], cwd=base_dir, capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT, shell=False)
            output = f"Git log:\n{r.stdout}" if r.stdout else "No history."
            return sanitize_ingested_content(output, source="git_log")
        except subprocess.TimeoutExpired:
            return f"ERROR: Command timed out after {_SUBPROCESS_TIMEOUT}s"
        except Exception as e:
            return f"ERROR: {e}"

    elif action == "git_diff":
        try:
            from agent.security import sanitize_ingested_content
            r = subprocess.run(["git", "diff"], cwd=base_dir, capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT, shell=False)
            output = f"Diff:\n{r.stdout}" if r.stdout else "No changes."
            return sanitize_ingested_content(output, source="git_diff")
        except subprocess.TimeoutExpired:
            return f"ERROR: Command timed out after {_SUBPROCESS_TIMEOUT}s"
        except Exception as e:
            return f"ERROR: {e}"

    return f"Unknown action: {action}"


async def _handle_code(p: dict) -> str:
    """Async wrapper — runs blocking file/subprocess ops in thread executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _handle_code_sync, p)

