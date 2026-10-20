"""
OpenClaw/Moltbot — Hydrinia Monitoring Tool Handlers

Read Hydrinia logs, state files, process status, portfolio, and performance.

Hydrinia is permanently off (operator direction, 2026-09-24) and its files on
disk are frozen/stale. Every public handle_* here is gated on
agent.hydrinia_bridge.bridge_enabled() (default False) so it degrades to a
clear retired-tool string instead of reading stale files or contacting a dead
process. This is a SECOND, independent gate on top of agent/tools/executor.py's
dispatch-level gate — at least one caller (agent/integrations/scheduled_tasks.py)
imports these handlers directly, bypassing execute_tool().
"""

import json
import logging
import os
import time
from collections import deque
from pathlib import Path

from agent.security import sanitize_ingested_content
from agent.integrations.alpaca_values import AlpacaValueError, normalize_position_side

logger = logging.getLogger("openclaw.hydrinia_tools")

TRADINGBOT_DIR = "/home/agent/tradingbot"
STATE_DIR = os.path.join(TRADINGBOT_DIR, "state")
LOG_DIR = os.path.join(TRADINGBOT_DIR, "state", "logs")  # Hydrinia logs at state/logs/, not logs/

_RETIRED_MSG = (
    "RETIRED: Hydrinia is permanently off — this tool is unavailable. "
    "CLIO trades its own SPY/QQQ book via the DPOF engine now; use get_dpof_status, "
    "get_portfolio, or get_account_info instead."
)


def _bridge_enabled() -> bool:
    try:
        from agent.hydrinia_bridge import bridge_enabled
        return bridge_enabled()
    except Exception as e:
        logger.debug(f"Hydrinia bridge check failed, defaulting to disabled: {e}")
        return False

ALLOWED_STATE_FILES = {
    "position_log.json",
    "daily_metrics.json",
    "agent_memory.json",
    "learning_log.json",
    "knowledge_base.json",
    "reserve_state.json",
    "overnight_queue.json",
    "trade_outcomes.json",
    "role_elections.json",
    "employee_of_month_state.json",
}


def _tail_file_bytes(file_path: str, chunk_size: int = 131072) -> list[str]:
    """Efficiently read the last chunk of a large file using byte-seeking.

    Reads the last `chunk_size` bytes (default 128KB) of the file — enough
    to cover ~500-1000 typical log lines without scanning the full 30MB.
    Returns a list of decoded lines (no trailing newline).
    """
    try:
        file_size = os.path.getsize(file_path)
        seek_pos = max(0, file_size - chunk_size)
        with open(file_path, "rb") as f:
            f.seek(seek_pos)
            raw = f.read()
        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()
        # If we seeked mid-file, the first line is likely a partial — discard it
        if seek_pos > 0 and lines:
            lines = lines[1:]
        return lines
    except Exception as e:
        logger.debug(f"Failed to tail file {file_path}: {e}")
        return []


async def handle_hydrinia_logs(p: dict) -> str:
    """Read recent Hydrinia log entries with optional filtering.

    Uses byte-seek tail for large files (30MB+) — reads only the last 128KB
    (~500-1000 lines) rather than scanning the entire file.
    """
    if not _bridge_enabled():
        return _RETIRED_MSG
    log_file = os.path.join(LOG_DIR, "hydrinia.log")
    if not os.path.isfile(log_file):
        # Also try errors-only log as fallback info
        errors_log = os.path.join(LOG_DIR, "hydrinia_errors.log")
        if os.path.isfile(errors_log):
            return (
                "Main log not found at expected path. "
                f"Errors log exists: {errors_log}\n"
                "Try get_hydrinia_logs with keyword='ERROR' after Moltbot restart."
            )
        return "No Hydrinia log file found. Bot may not have run yet."

    try:
        max_lines = min(int(p.get("lines", 100)), 500)
    except (ValueError, TypeError):
        max_lines = 100
    level_filter = p.get("level")
    keyword = p.get("keyword")

    # For filtered reads, fetch a larger tail to have enough candidates
    chunk_size = 131072  # 128KB default — covers ~500-1000 lines
    if level_filter or keyword:
        chunk_size = 524288  # 512KB for filtered reads to ensure enough matches

    try:
        raw_lines = _tail_file_bytes(log_file, chunk_size=chunk_size)
        if not raw_lines:
            return "Log file exists but could not be read."

        filtered = []
        for line in reversed(raw_lines):
            line = line.rstrip()
            if not line:
                continue
            if level_filter and f" | {level_filter}" not in line and f"[{level_filter}]" not in line:
                continue
            if keyword and keyword.lower() not in line.lower():
                continue
            filtered.append(line)
            if len(filtered) >= max_lines:
                break

        # Restore chronological order
        filtered.reverse()

        if not filtered:
            msg = "No log entries found"
            if level_filter:
                msg += f" with level={level_filter}"
            if keyword:
                msg += f" containing '{keyword}'"
            msg += " in last 512KB of log. Try increasing lines or checking hydrinia_errors.log."
            return msg

        file_size_mb = os.path.getsize(log_file) / 1_048_576
        header = f"Hydrinia Logs ({len(filtered)} lines, log={file_size_mb:.1f}MB"
        if level_filter:
            header += f", level={level_filter}"
        if keyword:
            header += f", keyword='{keyword}'"
        header += "):\n"

        return sanitize_ingested_content(header + "\n".join(filtered), "hydrinia:logs")
    except Exception as e:
        return f"ERROR reading logs: {e}"


async def handle_hydrinia_state(p: dict) -> str:
    """Read a Hydrinia state file."""
    if not _bridge_enabled():
        return _RETIRED_MSG
    file_name = p.get("file", "")
    if not file_name:
        return f"ERROR: file is required. Available: {', '.join(sorted(ALLOWED_STATE_FILES))}"

    if file_name not in ALLOWED_STATE_FILES:
        return f"ERROR: '{file_name}' not allowed. Available: {', '.join(sorted(ALLOWED_STATE_FILES))}"

    file_path = os.path.join(STATE_DIR, file_name)
    if not os.path.isfile(file_path):
        return f"State file not found: {file_name} (bot may not have created it yet)"

    try:
        with open(file_path, "r") as f:
            content = f.read()

        # Try to pretty-print JSON
        try:
            data = json.loads(content)
            formatted = json.dumps(data, indent=2, default=str)
            if len(formatted) > 8000:
                formatted = formatted[:8000] + "\n...(truncated)"
            return sanitize_ingested_content(f"State file: {file_name}\n{formatted}", f"hydrinia:state:{file_name}")
        except json.JSONDecodeError:
            if len(content) > 8000:
                content = content[:8000] + "\n...(truncated)"
            return sanitize_ingested_content(f"State file: {file_name}\n{content}", f"hydrinia:state:{file_name}")
    except Exception as e:
        return f"ERROR reading state file: {e}"


async def handle_hydrinia_status(p: dict) -> str:
    """Check if Hydrinia is running, PID, uptime, and last activity."""
    if not _bridge_enabled():
        return _RETIRED_MSG
    lock_file = os.path.join(STATE_DIR, "hydrinia.lock")
    emergency_file = os.path.join(STATE_DIR, "EMERGENCY_STOP")

    status_parts = []

    # Check process
    if os.path.isfile(lock_file):
        try:
            with open(lock_file, "r") as f:
                lock_data = json.loads(f.read())
            pid = lock_data.get("pid")
            started = lock_data.get("started", "unknown")
            start_ts = lock_data.get("timestamp", 0)

            try:
                os.kill(pid, 0)  # Check if process is alive
                uptime_sec = time.time() - start_ts if start_ts else 0
                hours = int(uptime_sec // 3600)
                minutes = int((uptime_sec % 3600) // 60)
                status_parts.append(f"Status: RUNNING")
                status_parts.append(f"PID: {pid}")
                status_parts.append(f"Started: {started}")
                status_parts.append(f"Uptime: {hours}h {minutes}m")
            except (ProcessLookupError, PermissionError):
                status_parts.append(f"Status: NOT RUNNING (stale lock, PID {pid})")
                status_parts.append(f"Last started: {started}")
        except Exception as e:
            status_parts.append(f"Status: UNKNOWN (lock file error: {e})")
    else:
        status_parts.append("Status: NOT RUNNING (no lock file)")

    # Check emergency stop
    if os.path.isfile(emergency_file):
        try:
            with open(emergency_file, "r") as f:
                estop = json.loads(f.read())
            reason = estop.get("reason", "unknown")
            status_parts.append(f"EMERGENCY STOP: Active — {reason}")
        except Exception as e:
            logger.debug(f"ESTOP file parse failed: {e}")
            status_parts.append("EMERGENCY STOP: Active (file exists)")
    else:
        status_parts.append("Emergency Stop: Not active")

    # Check last log activity
    log_file = os.path.join(LOG_DIR, "hydrinia.log")
    if os.path.isfile(log_file):
        try:
            mtime = os.path.getmtime(log_file)
            age_min = (time.time() - mtime) / 60
            if age_min < 1:
                status_parts.append("Last log: < 1 min ago")
            elif age_min < 60:
                status_parts.append(f"Last log: {int(age_min)} min ago")
            else:
                status_parts.append(f"Last log: {int(age_min / 60)}h {int(age_min % 60)}m ago")
        except Exception as e:
            logger.debug(f"Failed to read log mtime: {e}")

    return sanitize_ingested_content("\n".join(status_parts), source="hydrinia_status")


async def _get_alpaca_positions() -> list:
    """Fetch live positions from Alpaca. Returns list of position objects, or empty list on failure."""
    try:
        from agent.integrations.alpaca import _get_trading_client, _retry_alpaca_call
        client = _get_trading_client()
        return await _retry_alpaca_call(client.get_all_positions) or []
    except Exception as e:
        logger.debug(f"Alpaca position fetch failed (will fall back to position_log): {e}")
        return []


async def handle_hydrinia_portfolio(p: dict) -> str:
    """Show current portfolio with live Alpaca data (primary) and position_log context (fallback)."""
    if not _bridge_enabled():
        return _RETIRED_MSG
    # Try Alpaca first — it has live prices, all asset classes, and correct P&L
    alpaca_positions = await _get_alpaca_positions()

    if alpaca_positions:
        return _format_alpaca_portfolio(alpaca_positions, p)

    # Fallback: parse position_log.json (no live prices available)
    return _format_position_log_portfolio(p)


def _format_alpaca_portfolio(alpaca_positions: list, p: dict) -> str:
    """Format portfolio from live Alpaca position data."""
    detail = p.get("detail", True)

    # Separate equities and options
    equities = []
    options = []
    for pos in alpaca_positions:
        # S159-B H-1: AssetClass is a str-Enum — use .value for reliable comparison
        raw_class = getattr(pos, "asset_class", None)
        asset_class = (raw_class.value if hasattr(raw_class, "value") else str(raw_class)).lower()
        if "option" in asset_class:
            options.append(pos)
        else:
            equities.append(pos)

    total_count = len(alpaca_positions)
    lines = [f"Portfolio: {total_count} position(s) ({len(equities)} equity, {len(options)} options)\n"]
    total_pnl = 0.0
    total_market_value = 0.0

    # Equities
    if equities:
        lines.append("Equities:")
        for pos in equities:
            total_pnl, total_market_value = _append_position_line(
                lines, pos, detail, total_pnl, total_market_value
            )

    # Options
    if options:
        lines.append("\nOptions:")
        for pos in options:
            total_pnl, total_market_value = _append_position_line(
                lines, pos, detail, total_pnl, total_market_value
            )

    lines.append(f"\nTotal Market Value: ${total_market_value:+,.2f}")
    lines.append(f"Total Unrealized P&L: ${total_pnl:+,.2f}")
    return sanitize_ingested_content("\n".join(lines), source="alpaca_portfolio")


def _append_position_line(
    lines: list, pos, detail: bool, total_pnl: float, total_market_value: float
) -> tuple:
    """Append a formatted position line. Returns updated (total_pnl, total_market_value)."""
    symbol = getattr(pos, "symbol", "???")
    try:
        # S162 H-5: Avoid falsy chain — 0 is valid for all fields
        qty = float(pos.qty) if pos.qty is not None else 0.0
        entry = float(pos.avg_entry_price) if pos.avg_entry_price is not None else 0.0
        current = float(pos.current_price) if pos.current_price is not None else 0.0
        market_val = float(pos.market_value) if pos.market_value is not None else 0.0
        pnl = float(pos.unrealized_pl) if pos.unrealized_pl is not None else 0.0
        pnl_pct_raw = pos.unrealized_plpc
        pnl_pct = float(pnl_pct_raw) * 100 if pnl_pct_raw is not None else 0.0
    except (ValueError, TypeError):
        lines.append(f"  {symbol}: Error reading position data")
        return total_pnl, total_market_value
    try:
        side = normalize_position_side(getattr(pos, "side", None))
    except AlpacaValueError:
        lines.append(f"  {symbol}: Unknown broker position side; excluded")
        return total_pnl, total_market_value

    total_pnl += pnl
    total_market_value += abs(market_val)

    display_qty = abs(qty)
    # Show integer qty if whole number
    qty_str = str(int(display_qty)) if display_qty == int(display_qty) else f"{display_qty:.2f}"

    if detail:
        lines.append(
            f"  {symbol} ({side}): {qty_str} @ ${entry:.2f}"
            f" → ${current:.2f} | Val: ${abs(market_val):,.2f}"
            f" | P&L: ${pnl:+,.2f} ({pnl_pct:+.1f}%)"
        )
    else:
        lines.append(f"  {symbol}: {qty_str} ({side}) P&L: ${pnl:+,.2f} ({pnl_pct:+.1f}%)")

    return total_pnl, total_market_value


def _format_position_log_portfolio(p: dict) -> str:
    """Fallback: format portfolio from position_log.json when Alpaca is unreachable."""
    position_file = os.path.join(STATE_DIR, "position_log.json")
    if not os.path.isfile(position_file):
        return "No position_log.json found. Bot may not have any positions yet."

    try:
        with open(position_file, "r") as f:
            data = json.loads(f.read())

        if isinstance(data, dict):
            positions = data.get("positions", data.get("open_positions", []))
            if isinstance(positions, dict):
                positions = list(positions.values())
        elif isinstance(data, list):
            positions = data
        else:
            return f"Unexpected position_log format: {type(data).__name__}"

        open_positions = [pos for pos in positions if pos.get("status") == "open"]
        if not open_positions:
            return "Portfolio: No open positions."

        detail = p.get("detail", True)
        lines = [f"Portfolio: {len(open_positions)} open position(s) (live prices unavailable)\n"]

        for pos in open_positions:
            symbol = pos.get("symbol", "???")
            contract = pos.get("contract", symbol)
            qty = pos.get("quantity", pos.get("qty", 0))
            entry = pos.get("entry_price", 0)
            is_short = pos.get("is_short", False)
            is_options = pos.get("is_options", False)
            side = "short" if is_short else ("short" if pos.get("direction") == "bearish" else "long")
            display_name = contract if is_options and contract != symbol else symbol

            if detail:
                lines.append(f"  {display_name} ({side}): {qty} @ ${entry:.2f} → N/A")
            else:
                lines.append(f"  {display_name}: {qty} ({side})")

        lines.append("\n(Note: Live prices unavailable — connect Alpaca for current data)")
        return sanitize_ingested_content("\n".join(lines), source="hydrinia_strategy")
    except Exception as e:
        return f"ERROR reading portfolio: {e}"


async def handle_hydrinia_performance(p: dict) -> str:
    """Read daily metrics and performance data."""
    if not _bridge_enabled():
        return _RETIRED_MSG
    metrics_file = os.path.join(STATE_DIR, "daily_metrics.json")
    if not os.path.isfile(metrics_file):
        return "No daily_metrics.json found. Bot may not have generated metrics yet."

    try:
        with open(metrics_file, "r") as f:
            data = json.loads(f.read())

        days = p.get("days", 7)

        # Handle various formats
        if isinstance(data, list):
            recent = data[-days:]
        elif isinstance(data, dict):
            if "history" in data:
                recent = data["history"][-days:]
            else:
                # Single day metrics
                recent = [data]
        else:
            return f"Unexpected metrics format: {type(data).__name__}"

        if not recent:
            return "No performance data available."

        lines = [f"Performance ({len(recent)} day(s)):\n"]

        total_pnl = 0.0
        total_trades = 0
        wins = 0
        losses = 0

        for day in recent:
            date = day.get("date", "?")
            pnl = day.get("pnl")
            if pnl is None:
                pnl = day.get("total_pnl")
            if pnl is None:
                pnl = 0
            try:
                pnl = float(pnl)
            except (ValueError, TypeError):
                pnl = 0.0
            trades = day.get("trades")
            if trades is None:
                trades = day.get("trade_count")
            if trades is None:
                trades = 0
            try:
                trades = int(trades)
            except (ValueError, TypeError):
                trades = 0
            wr = day.get("win_rate")
            if wr is None:
                wr = 0
            try:
                wr = float(wr)
            except (ValueError, TypeError):
                wr = 0.0

            total_pnl += pnl
            total_trades += trades
            try:
                wins += int(day.get("wins", 0))
            except (TypeError, ValueError) as e:
                logger.debug(f"Failed to parse wins for {date}: {e}")
            try:
                losses += int(day.get("losses", 0))
            except (TypeError, ValueError) as e:
                logger.debug(f"Failed to parse losses for {date}: {e}")

            # win_rate may be 0-1 fraction or 0-100 percentage; normalize
            wr_display = wr if wr <= 1.0 else wr / 100.0
            lines.append(f"  {date}: P&L ${pnl:+.2f} | {trades} trades | WR {wr_display:.0%}")

        lines.append(f"\nSummary:")
        lines.append(f"  Total P&L: ${total_pnl:+.2f}")
        lines.append(f"  Total Trades: {total_trades}")
        if wins + losses > 0:
            lines.append(f"  Win Rate: {wins/(wins+losses):.0%} ({wins}W/{losses}L)")

        return sanitize_ingested_content("\n".join(lines), source="hydrinia_performance")
    except Exception as e:
        return f"ERROR reading performance: {e}"
