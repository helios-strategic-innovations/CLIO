"""
OpenClaw/Moltbot — Scheduled Monitoring Tasks

Periodic checks on Hydrinia status, error logs, and portfolio.
Uses python-telegram-bot's built-in JobQueue (APScheduler backend).
Alert-based: only sends Telegram messages when something changes.
"""

import asyncio
import fcntl
import logging
import os
import json
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from telegram.error import BadRequest, Forbidden
from telegram.ext import Application

from agent.config import config
from agent.hydrinia_bridge import bridge_enabled

logger = logging.getLogger("openclaw.scheduled")

# ============================================================
# State tracking (module-level, survives across job runs)
# ============================================================

_last_status: str = "unknown"       # "running" | "stopped" | "unknown"
_last_log_size: int = 0             # Track log file size to detect new errors
_last_log_ino: int = 0              # Track inode to detect rename-rotation regardless of size
_last_emergency_stop: bool = False  # Track EMERGENCY_STOP file state (initialized below after _ESTOP_FILE defined)
_unreachable_chats: set[str] = set()  # Chat IDs that returned "Chat not found" / Forbidden
_heartbeat_task: asyncio.Task | None = None  # Stored for cancellation on shutdown
_status_lock = threading.Lock()     # S159 M-3: Protects _last_status, _last_self_check_state across threads
_LOG_SCAN_SIGNAL_KEYS: set[tuple[str, str]] = set()  # (path, error type) already warned this process lifetime

# Padding-tolerant pipe-delimited level match, e.g. "... | ERROR    | Hydrinia | ..."
_LEVEL_LINE_PATTERN = re.compile(r"\|\s*(ERROR|CRITICAL)\s*\|")

# Paths (derived from config for portability)
_STATE_DIR = os.path.join(config.tradingbot_dir, "state")
_LOG_FILE = os.path.join(config.tradingbot_dir, "state", "logs", "hydrinia_errors.log")
_LOCK_FILE = os.path.join(_STATE_DIR, "hydrinia.lock")
_ESTOP_FILE = os.path.join(_STATE_DIR, "EMERGENCY_STOP")

# Initialize emergency stop state from file so restarts don't re-alert on stale stops
_last_emergency_stop = os.path.isfile(_ESTOP_FILE)

# Intervals (seconds)
STATUS_CHECK_INTERVAL = 300     # 5 minutes
ERROR_SCAN_INTERVAL = 300       # 5 minutes
PORTFOLIO_INTERVAL = 1800       # 30 minutes
LEARNING_INTERVAL = 900         # 15 minutes
FEEDBACK_INTERVAL = 7200        # 2 hours
MARKET_EVENT_INTERVAL = 300     # 5 minutes
BOOK_INTERVAL = 5400            # 90 minutes
RETENTION_INTERVAL = 86400      # 24 hours (daily)

# First run delays (seconds after startup)
STATUS_FIRST_RUN = 30
ERROR_FIRST_RUN = 60
PORTFOLIO_FIRST_RUN = 120
LEARNING_FIRST_RUN = 180        # 3 min after startup
FEEDBACK_FIRST_RUN = 600        # 10 min after startup
MARKET_EVENT_FIRST_RUN = 240    # 4 min after startup
BOOK_FIRST_RUN = 900            # 15 min after startup
RETENTION_FIRST_RUN = 1200      # 20 min after startup (let other systems warm up)
SELF_CHECK_INTERVAL = 600       # 10 minutes (S142-B W1-B: was 180)
SELF_CHECK_FIRST_RUN = 90       # 1.5 min after startup
HEALTH_REPORT_INTERVAL = 3600   # 1 hour
HEALTH_REPORT_FIRST_RUN = 300   # 5 min after startup
ATHENA_HEALTH_INTERVAL = 1800   # 30 min (Phase 1 Athena bridge health check)
ATHENA_HEALTH_FIRST_RUN = 600   # 10 min delay before first check

# Athena stale-vault alert (S179 deferred → S184 followup)
# Default 6h: vault is normally updated after each idea-deliberation cycle (~hourly).
# 6h staleness = half-day silence, suggests Hydrinia writer stopped or hung.
# Operator-tunable via CLIO_ATHENA_STALE_HOURS env var.
_athena_stale_alert_dedup: dict = {}   # key: "vault_stale" → last alert epoch (sec)

# E.1 enrichment alert dedup (drop-rate / credit / cap) — module-level is fine per ADR-003:
# the 6h alert window dominates the ~daily restart rate; duplicate-on-restart is tolerable.
_enrich_alert_dedup: dict = {}         # key: alert kind → last alert epoch (sec)

# Service start time (for uptime tracking, W1-A)
_SERVICE_START_TIME = time.time()

# Daily task dedup (S142-B W4-A) — prevents duplicate runs on restart
_DAILY_TASK_RAN_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "daily_task_runs.json"))


def _daily_task_already_ran(task_name: str) -> bool:
    """Check if a daily task already ran today (prevents duplicate runs on restart)."""
    try:
        today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        if os.path.isfile(_DAILY_TASK_RAN_FILE):
            with open(_DAILY_TASK_RAN_FILE, "r") as f:
                runs = json.load(f)
        else:
            runs = {}
        return runs.get(task_name) == today
    except Exception as e:
        logger.debug(f"Daily task state read: {e}")
        return False


_DAILY_TASK_LOCK_FILE = _DAILY_TASK_RAN_FILE + ".lock"


def _mark_daily_task_ran(task_name: str) -> None:
    """Mark a daily task as having run today. File-locked to prevent concurrent clobber."""
    lock_fd = None
    try:
        today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        lock_fd = open(_DAILY_TASK_LOCK_FILE, "a")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        if os.path.isfile(_DAILY_TASK_RAN_FILE):
            with open(_DAILY_TASK_RAN_FILE, "r") as f:
                runs = json.load(f)
        else:
            runs = {}
        runs[task_name] = today
        tmp = _DAILY_TASK_RAN_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(runs, f)
        os.replace(tmp, _DAILY_TASK_RAN_FILE)
    except Exception as e:
        logger.debug("Failed to mark daily task %s: %s", task_name, e)
    finally:
        if lock_fd:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
            except Exception as e:
                logger.debug(f"Lock cleanup: {e}")


def _capture_mtime(path: str) -> float:
    """Return mtime of path, or 0.0 if missing/unreadable. Baseline for progress checks."""
    try:
        return os.path.getmtime(path) if os.path.exists(path) else 0.0
    except Exception:
        return 0.0


# R-31 (S215): a daily task that returned but left its state file un-advanced is escalated
# after _DAILY_TASK_MISS_ESCALATE_AT consecutive misses on days it was EXPECTED to advance.
# Counters are in memory (a restart resets them: acceptable). One alert per task per ET day.
_DAILY_TASK_MISS_ESCALATE_AT = 2
_MARKET_WEEKDAYS = frozenset(range(5))
# Weekdays each daily task is expected to advance on, taken from its own time guard:
# macro/fed/earnings/commodity return on weekend days; bond_intel runs Tue-Thu only;
# analytics_fun_facts is Mon-Fri; econ_calendar has no weekday gate (every day).
_DAILY_TASK_EXPECTED_WEEKDAYS: dict[str, frozenset] = {
    "calendar_refresh": frozenset(range(7)),
    "macro_research": _MARKET_WEEKDAYS,
    "fed_analysis": _MARKET_WEEKDAYS,
    "earnings_intel": _MARKET_WEEKDAYS,
    "commodity_review": _MARKET_WEEKDAYS,
    "bond_intel": frozenset({1, 2, 3}),
    "analytics_fun_facts": _MARKET_WEEKDAYS,
}
_daily_task_consecutive_misses: dict[str, int] = {}
_daily_task_alert_day: dict[str, str] = {}
_background_alert_tasks: set = set()


def _daily_task_et_now() -> datetime:
    """Current ET time for the escalation gate (patchable in tests)."""
    return datetime.now(ZoneInfo("America/New_York"))


def _daily_task_expected_today(task_name: str) -> bool:
    """True when task_name is expected to advance its state file on today's ET weekday."""
    weekdays = _DAILY_TASK_EXPECTED_WEEKDAYS.get(task_name, frozenset(range(7)))
    return _daily_task_et_now().weekday() in weekdays


async def _send_daily_task_alert(bot, text: str) -> None:
    """Plain-text alert to every notify chat (never parse_mode)."""
    try:
        for cid in _get_chat_ids():
            await _safe_send(bot, cid, text=text)
    except Exception as e:
        logger.debug("daily task alert send failed: %s", e)


def _escalate_daily_task_miss(task_name: str, state_file: str, misses: int, bot) -> None:
    """Log ERROR on every escalated miss; send at most one Telegram alert per task per ET day."""
    logger.error(
        "daily task '%s' has not advanced %s for %d consecutive expected runs",
        task_name,
        os.path.basename(state_file),
        misses,
    )
    if bot is None:
        return
    today = _daily_task_et_now().strftime("%Y-%m-%d")
    if _daily_task_alert_day.get(task_name) == today:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    _daily_task_alert_day[task_name] = today
    msg = (
        f"Daily task '{task_name}' has not advanced {os.path.basename(state_file)} "
        f"for {misses} consecutive expected runs. Check the scheduled task logs."
    )
    alert = loop.create_task(_send_daily_task_alert(bot, msg))
    _background_alert_tasks.add(alert)
    alert.add_done_callback(_background_alert_tasks.discard)


def _mark_daily_task_if_state_progressed(
    task_name: str, state_file: str, baseline_mtime: float, bot=None
) -> bool:
    """Only stamp the daily-run marker if state_file's mtime advanced past baseline.

    Closes the silent-failure loop where a task returns cleanly (catches its own
    Anthropic 400, etc.) but produces no state output. Without this gate, the
    next cycle sees "already ran today" and skips — leaving the state file stale
    for days while the operator has no signal.
    """
    try:
        current = os.path.getmtime(state_file) if os.path.exists(state_file) else 0.0
        if current > baseline_mtime:
            _mark_daily_task_ran(task_name)
            _daily_task_consecutive_misses[task_name] = 0
            return True
        logger.warning(
            "daily task '%s' returned but %s did not advance "
            "(baseline=%.0f current=%.0f) — NOT marking as ran",
            task_name,
            os.path.basename(state_file),
            baseline_mtime,
            current,
        )
        if _daily_task_expected_today(task_name):
            misses = _daily_task_consecutive_misses.get(task_name, 0) + 1
            _daily_task_consecutive_misses[task_name] = misses
            if misses >= _DAILY_TASK_MISS_ESCALATE_AT:
                _escalate_daily_task_miss(task_name, state_file, misses, bot)
        return False
    except Exception as e:
        logger.debug("mtime check for %s: %s", task_name, e)
        return False


def _commodity_review_was_noop(result) -> bool:
    """CM-1 (S211): True only for a run that reviewed nothing BECAUSE nothing was due.

    run_weekly_commodity_review skips every commodity analysed within its 6-day dedup window and then
    returns without touching commodity_research_state.json. The S185/S205 state-advance gate read that
    as a silent failure and warned every day. A real failure (empty research, no analysis, an exception,
    an unknown/None result) still goes through the gate.
    """
    if not isinstance(result, dict):
        return False
    return (
        result.get("reviewed") == 0
        and result.get("failed") == 0
        and not result.get("error")
        and result.get("skipped_recent", 0) > 0
    )


async def _run_commodity_review_guarded(context, real_fn) -> None:
    """Daily commodity review with the dedup marker + state-advance gate (S122 4D, S185 W1-2, CM-1)."""
    if _daily_task_already_ran("commodity_review"):
        logger.info("commodity_review already ran today, skipping")
        return
    state_file = _STATE_FILE_FOR_TASK["commodity_review"]
    baseline = _capture_mtime(state_file)
    result = await real_fn(context)
    if _commodity_review_was_noop(result):
        _mark_daily_task_ran("commodity_review")
        _daily_task_consecutive_misses["commodity_review"] = 0
        logger.info("commodity_review: nothing due -- every commodity was analysed within the dedup window")
        return
    _mark_daily_task_if_state_progressed(
        "commodity_review", state_file, baseline, bot=getattr(context, "bot", None)
    )


# Canonical state-file paths used by mtime-gated daily task dedup (S185 W1-2)
_INTEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "intelligence")
)
_STATE_FILE_FOR_TASK = {
    "macro_research": os.path.join(_INTEL_DIR, "macro_research_state.json"),
    "fed_analysis": os.path.join(_INTEL_DIR, "fed_analysis_state.json"),
    "earnings_intel": os.path.join(_INTEL_DIR, "earnings_calendar.json"),
    "commodity_review": os.path.join(_INTEL_DIR, "commodity_research_state.json"),
    "bond_intel": os.path.join(_INTEL_DIR, "bond_intel_state.json"),
    "analytics_fun_facts": os.path.join(_INTEL_DIR, "analytics_fun_facts.json"),
    "calendar_refresh": os.path.join(_INTEL_DIR, "econ_calendar.json"),
}


# ============================================================
# Helpers
# ============================================================

def _dpof_eod_now() -> datetime:
    """Current ET time for the DPOF EOD digest's own gating (patchable in tests)."""
    return datetime.now(ZoneInfo("America/New_York"))


def _get_chat_ids() -> list[str]:
    """Return all configured notification chat IDs, excluding unreachable ones."""
    return [cid for cid in config.telegram_notify_chat_ids if cid not in _unreachable_chats]


async def _safe_send(bot, chat_id: str, **kwargs) -> bool:
    """Send a message, suppressing Chat-not-found / Forbidden errors.

    Returns True if sent successfully, False if the chat is unreachable.
    Logs a warning only once per chat_id, then silently skips on future calls.
    """
    try:
        await bot.send_message(chat_id=chat_id, **kwargs)
        return True
    except Forbidden as e:
        if chat_id not in _unreachable_chats:
            _unreachable_chats.add(chat_id)
            logger.warning(
                f"Chat {chat_id} is unreachable ({e}); suppressing future notifications"
            )
        return False
    except BadRequest as e:
        err_msg = str(e).lower()
        if "chat not found" in err_msg or "bot was blocked" in err_msg:
            if chat_id not in _unreachable_chats:
                _unreachable_chats.add(chat_id)
                logger.warning(
                    f"Chat {chat_id} is unreachable ({e}); suppressing future notifications"
                )
            return False
        logger.warning(f"BadRequest sending to {chat_id}: {e}")
        return False


def _is_hydrinia_running() -> tuple[bool, dict]:
    """Check if Hydrinia process is alive. Returns (running, lock_data)."""
    if not os.path.isfile(_LOCK_FILE):
        return False, {}
    lock_data = {}
    try:
        with open(_LOCK_FILE) as f:
            lock_data = json.loads(f.read())
        pid = lock_data.get("pid")
        if pid:
            os.kill(pid, 0)  # Check alive (raises ProcessLookupError if dead)
            return True, lock_data
    except (ProcessLookupError, ValueError, json.JSONDecodeError):
        pass
    except OSError:
        pass
    return False, lock_data


# ============================================================
# Task 1: Status Check (every 5 min)
# ============================================================

async def check_hydrinia_status(context) -> None:
    """Alert on Hydrinia status transitions (running ↔ stopped, emergency stop)."""
    global _last_status, _last_emergency_stop

    chat_ids = _get_chat_ids()
    if not chat_ids:
        return

    try:
        running, lock_data = _is_hydrinia_running()
        current_status = "running" if running else "stopped"

        # S159 M-3: Read/write _last_status under lock
        with _status_lock:
            previous_status = _last_status
            _last_status = current_status

        # Detect transitions
        if previous_status != "unknown" and current_status != previous_status:
            if current_status == "stopped":
                pid = lock_data.get("pid", "?")
                for chat_id in chat_ids:
                    await _safe_send(
                        context.bot, chat_id,
                        text=f"Hydrinia has stopped!\nPrevious PID: {pid}\nDetected at scheduled check.",
                    )
                logger.warning(f"Scheduled check: Hydrinia stopped (was PID {pid})")
            else:
                pid = lock_data.get("pid", "?")
                started = lock_data.get("started", "?")
                for chat_id in chat_ids:
                    await _safe_send(
                        context.bot, chat_id,
                        text=f"Hydrinia is back online\nPID: {pid}\nStarted: {started}",
                    )
                logger.info(f"Scheduled check: Hydrinia back online (PID {pid})")

        # Emergency stop detection
        estop_active = os.path.isfile(_ESTOP_FILE)
        if estop_active and not _last_emergency_stop:
            reason = "unknown"
            try:
                with open(_ESTOP_FILE) as f:
                    estop_data = json.loads(f.read())
                reason = estop_data.get("reason", "unknown")
            except Exception as e:
                logger.debug(f"Fire-and-forget ESTOP file read failed: {e}")
            for chat_id in chat_ids:
                await _safe_send(
                    context.bot, chat_id,
                    text=f"EMERGENCY STOP DETECTED\nReason: {reason}",
                )
            logger.warning(f"Scheduled check: Emergency stop detected — {reason}")
        elif not estop_active and _last_emergency_stop:
            for chat_id in chat_ids:
                await _safe_send(
                    context.bot, chat_id,
                    text="✅ Emergency stop cleared.",
                )

        _last_emergency_stop = estop_active

        # Memory health check — log warnings when tiers degrade (Fix S103-2C)
        # S159 M-4: health_check() is sync — run in executor to avoid blocking event loop
        try:
            from agent.memory.memory_engine import get_memory_engine
            loop = asyncio.get_running_loop()
            mem_health = await loop.run_in_executor(None, get_memory_engine().health_check)
            if not mem_health.get("overall_healthy", True):
                logger.warning(f"Memory health degraded: {mem_health}")
        except Exception as e:
            logger.debug(f"Fire-and-forget memory health check failed: {e}")

    except Exception as e:
        logger.error(f"Scheduled status check failed: {e}", exc_info=True)


# ============================================================
# Task 2: Error Log Scanner (every 5 min)
# ============================================================

def _emit_log_scan_signal(path_text: str, error: Exception) -> None:
    """Warn once per (path, error type) instead of silently skipping a missing/unreadable target."""
    reason = type(error).__name__
    key = (path_text, reason)
    if key in _LOG_SCAN_SIGNAL_KEYS:
        return
    _LOG_SCAN_SIGNAL_KEYS.add(key)
    logger.warning(
        "Hydrinia error-log scan target unavailable; error scanning disabled until it recovers; path=%s reason=%s",
        path_text,
        reason,
    )


async def scan_hydrinia_errors(context) -> None:
    """Scan for new ERROR/CRITICAL-level log entries since last check."""
    global _last_log_size, _last_log_ino

    chat_ids = _get_chat_ids()
    if not chat_ids:
        return

    try:
        try:
            stat_result = os.stat(_LOG_FILE)
        except OSError as e:
            _emit_log_scan_signal(_LOG_FILE, e)
            return

        current_size = stat_result.st_size
        current_ino = stat_result.st_ino

        if current_ino != _last_log_ino:
            # Rename-rotation (or first observation) — the path now points at a
            # different inode, so the offset we tracked is meaningless. Scan the
            # new file from the beginning regardless of its current size.
            _last_log_size = 0
            _last_log_ino = current_ino
        elif current_size < _last_log_size:
            # In-place truncation (e.g. copytruncate) on the same inode.
            _last_log_size = 0

        # Skip if file hasn't changed
        if current_size == _last_log_size:
            return

        # Read only the new portion of the log
        new_errors = []
        try:
            with open(_LOG_FILE, "r", errors="replace") as f:
                f.seek(_last_log_size)
                for line in f:
                    line = line.rstrip()
                    if _LEVEL_LINE_PATTERN.search(line):
                        new_errors.append(line)
        except OSError as e:
            # Do not advance _last_log_size here — leave the offset alone so a
            # transient read failure doesn't silently drop the unread span.
            _emit_log_scan_signal(_LOG_FILE, e)
            return

        _last_log_size = current_size

        if not new_errors:
            return

        # Format and send (cap at 10 errors, 4000 chars for Telegram)
        header = f"⚠️ {len(new_errors)} new error(s) in Hydrinia logs:\n"
        body = "\n".join(new_errors[:10])
        if len(new_errors) > 10:
            body += f"\n... and {len(new_errors) - 10} more"

        message = header + body
        if len(message) > 4000:
            message = message[:3997] + "..."

        for chat_id in chat_ids:
            await _safe_send(
                context.bot, chat_id, text=message,
            )
        logger.info(f"Scheduled check: {len(new_errors)} new errors reported")

    except Exception as e:
        logger.error(f"Scheduled error scan failed: {e}", exc_info=True)


# ============================================================
# Task 3: Portfolio Snapshot (every 30 min)
# ============================================================

async def check_portfolio_snapshot(context) -> None:
    """Send periodic portfolio snapshot if Hydrinia is running with positions."""
    chat_ids = _get_chat_ids()
    if not chat_ids:
        return

    try:
        running, _ = _is_hydrinia_running()
        if not running:
            return

        from agent.tools.hydrinia_tools import handle_hydrinia_portfolio
        portfolio = await handle_hydrinia_portfolio({})

        if "No open positions" in portfolio or "No position_log" in portfolio:
            return

        message = f"📊 Portfolio Snapshot\n{portfolio}"
        if len(message) > 4000:
            message = message[:3997] + "..."

        for chat_id in chat_ids:
            await _safe_send(
                context.bot, chat_id, text=message,
            )

    except Exception as e:
        logger.error(f"Scheduled portfolio check failed: {e}", exc_info=True)


# ============================================================
# Task: Self-Check (every 10 min) — directives, memory, tasks
# ============================================================

_last_self_check_state: dict = {}  # S142-B W1-B change detection

async def run_self_check(context) -> None:
    """Lightweight self-audit: check pending directives, memory state, bridge files.

    Runs every 10 minutes (S142-B W1-B: was 3). Only logs when state changes.
    Sends Telegram alert if something requires operator attention.
    """
    global _last_self_check_state
    chat_ids = _get_chat_ids()
    alerts = []

    # 1. Check for active/pending override directives from Telegram
    try:
        from agent.integrations.directive_delivery import get_active_directives
        active = get_active_directives()
        if active:
            directive_ids = [d.get("id", "?")[:30] for d in active[:5]]
            logger.info(f"Self-check: {len(active)} active directive(s): {directive_ids}")
            for d in active:
                d_id = d.get("id", "?")
                d_type = d.get("type", "?")
                logger.debug(f"  Active directive: {d_id} type={d_type}")
        else:
            logger.debug("Self-check: no active directives")
    except Exception as e:
        logger.debug(f"Self-check directive scan failed: {e}")

    # 2. Check for unprocessed directive responses (Hydrinia → CLIO) — Hydrinia is
    # permanently off; this bridge directory only fills again if it's ever revived.
    if bridge_enabled():
        try:
            responses_dir = os.path.join(config.tradingbot_dir, "moltbot_knowledge", "directive_responses")
            if os.path.isdir(responses_dir):
                unprocessed = [
                    f for f in os.listdir(responses_dir)
                    if f.endswith(".json") and ".notified." not in f
                ]
                if unprocessed:
                    logger.info(f"Self-check: {len(unprocessed)} unprocessed directive response(s)")
        except Exception as e:
            logger.debug(f"Self-check directive responses scan failed: {e}")

    # 3. Memory health evaluation
    # S159 M-4: health_check() is sync — run in executor to avoid blocking event loop
    try:
        from agent.memory.memory_engine import get_memory_engine
        me = get_memory_engine()
        loop = asyncio.get_running_loop()
        health = await loop.run_in_executor(None, me.health_check)
        if not health.get("overall_healthy", True):
            degraded = []
            for tier in ("working", "episodic", "procedural", "meta"):
                tier_data = health.get(tier, {})
                if not tier_data.get("healthy", True):
                    degraded.append(tier)
            if degraded:
                alerts.append(f"Memory degraded: {', '.join(degraded)}")
                logger.warning(f"Self-check: memory degraded tiers: {degraded}")
        else:
            logger.debug("Self-check: memory healthy")

        # Check working memory entry count
        working_total = health.get("working", {}).get("total", 0)
        if working_total > 200:
            logger.info(f"Self-check: working memory high ({working_total} entries)")
    except Exception as e:
        logger.debug(f"Self-check memory eval failed: {e}")

    # 4. Check for urgent bridge files (newsflash, gap signals)
    try:
        urgent_dir = os.path.join(config.tradingbot_dir, "moltbot_knowledge", "urgent")
        if os.path.isdir(urgent_dir):
            urgent_files = [f for f in os.listdir(urgent_dir) if f.endswith(".json")]
            if urgent_files:
                alerts.append(f"{len(urgent_files)} urgent bridge file(s)")
                logger.debug(f"Self-check: {len(urgent_files)} urgent bridge file(s)")
    except Exception as e:
        logger.debug(f"Self-check urgent bridge scan failed: {e}")

    # 5. Check meta memory for corrections/gaps
    try:
        from agent.memory.memory_engine import get_memory_engine
        me = get_memory_engine()
        meta_counts = me.meta.get_counts()
        corrections = meta_counts.get("corrections", 0)
        gaps = meta_counts.get("gaps", 0)
        if corrections > 0 or gaps > 0:
            logger.debug(f"Self-check: meta memory — {corrections} corrections, {gaps} gaps")
    except Exception as e:
        logger.debug(f"Self-check procedural eval failed: {e}")

    # 6. Check EMERGENCY_STOP — Hydrinia's file; a stale copy from before it was
    # retired must not keep re-appearing in alerts once the bridge is off.
    if bridge_enabled():
        try:
            if os.path.isfile(_ESTOP_FILE):
                alerts.append("EMERGENCY STOP is active")
        except Exception as e:
            logger.debug(f"Self-check estop check failed: {e}")

    # S142-B W1-B: Change detection — skip logging if state unchanged
    # S159 M-3: Read/write _last_self_check_state under lock
    _current_state = {"alerts": sorted(alerts)}
    with _status_lock:
        if _current_state == _last_self_check_state and not alerts:
            return  # No change since last check, skip
        _last_self_check_state = _current_state

    # Send alerts only if there's something requiring attention
    if alerts and chat_ids:
        msg = "CLIO Self-Check Alert:\n" + "\n".join(f"- {a}" for a in alerts)
        for chat_id in chat_ids:
            await _safe_send(context.bot, chat_id, text=msg)


# ============================================================
# Task: System Health Report (hourly)
# ============================================================

async def run_system_health_report(context) -> None:
    """Comprehensive system health report sent to Telegram every hour.

    Covers: memory tiers, API budgets, Hydrinia status, personality,
    learning engine, SEAL, kill switch, RSS feeds.
    """
    chat_ids = _get_chat_ids()
    if not chat_ids:
        return

    lines = ["CLIO System Health Report", "=" * 30, ""]

    # 1. CLIO's own DPOF book + Trading mode + Uptime (S142-B W1-A; repointed off
    # Hydrinia 2026-09-24 — Hydrinia is permanently off, CLIO trades its own book)
    try:
        from agent.dpof import readmodel
        _dpof_snap = readmodel.snapshot()
        lines.append(
            f"DPOF: mode={_dpof_snap.get('mode', 'unknown')} | "
            f"open {_dpof_snap.get('open_count', 0)} | "
            f"closed today {_dpof_snap.get('closed_today_count', 0)} "
            f"({_dpof_snap.get('closed_today_wins', 0)} wins) | "
            f"realized today ${_dpof_snap.get('realized_pnl_today', 0.0):+,.2f} | "
            f"HALT file: {'yes' if _dpof_snap.get('halt_file') else 'no'}"
        )
        if _dpof_snap.get("hard_halted"):
            lines.append(f"  HARD HALTED: {_dpof_snap.get('halt_reason', 'no reason recorded')}")
    except Exception as e:
        lines.append(f"DPOF: check failed ({e})")
    if bridge_enabled() and os.path.isfile(_ESTOP_FILE):
        lines.append("  Hydrinia EMERGENCY STOP file present (bridge re-enabled)")

    # Trading mode
    _paper = config.alpaca_paper if hasattr(config, 'alpaca_paper') else True
    lines.append(f"Trading Mode: {'PAPER' if _paper else 'LIVE'}")

    # Service uptime
    _uptime_s = int(time.time() - _SERVICE_START_TIME)
    _up_h, _up_m = divmod(_uptime_s // 60, 60)
    lines.append(f"Uptime: {_up_h}h {_up_m}m")

    # 2. Memory health
    # S159 M-4: health_check() is sync — run in executor to avoid blocking event loop
    try:
        from agent.memory.memory_engine import get_memory_engine
        me = get_memory_engine()
        loop = asyncio.get_running_loop()
        health = await loop.run_in_executor(None, me.health_check)
        # S180: Trust the engine's overall_healthy signal. Working-memory=0 after 72h
        # of conversational idle is expected (WORKING_MEMORY_TTL_HOURS=72), not degradation.
        # Prior logic (`_wm_count == 0 -> DEGRADED`) produced false alarms on idle.
        overall_healthy = health.get("overall_healthy", True)
        _ep_stale = health.get("episodic", {}).get("stale", True)
        _mem_degraded = (not overall_healthy) or _ep_stale
        overall = "DEGRADED" if _mem_degraded else "OK"
        lines.append(f"Memory: {overall}")

        working = health.get("working", {})
        lines.append(f"  Working: {working.get('total', '?')} entries")

        episodic = health.get("episodic", {})
        ep_status = "stale (>24h)" if episodic.get("stale", True) else "fresh"
        lines.append(f"  Episodic: {ep_status}")

        procedural = health.get("procedural", {})
        lines.append(f"  Procedural: {procedural.get('patterns', '?')} patterns, triggered={procedural.get('any_triggered', False)}")

        meta = health.get("meta", {})
        lines.append(f"  Meta: {meta.get('corrections', 0)} corrections, {meta.get('gaps', 0)} gaps")
    except Exception as e:
        lines.append(f"Memory: check failed ({e})")

    # 3. API budgets (top consumers only)
    try:
        from agent.rate_limiter import budget_tracker
        status = await budget_tracker.get_status()
        lines.append("")
        lines.append("API Budgets:")
        # Show top 8 most-used budgets
        sorted_budgets = sorted(status.items(), key=lambda x: x[1].get("used", 0), reverse=True)
        for name, info in sorted_budgets[:8]:
            used = info.get("used", 0)
            max_req = info.get("max", 0)
            pct = (used / max_req * 100) if max_req > 0 else 0
            indicator = "!!" if pct >= 90 else "!" if pct >= 70 else ""
            if used > 0:  # Only show budgets with usage
                lines.append(f"  {name}: {used}/{max_req} ({pct:.0f}%) {indicator}")
    except Exception as e:
        lines.append(f"API Budgets: check failed ({e})")

    # 4. Active directives
    try:
        from agent.integrations.directive_delivery import get_active_directives
        active = get_active_directives()
        lines.append("")
        lines.append(f"Active Directives: {len(active)}")
        for d in active[:3]:
            d_id = d.get("id", "?")[:30]
            d_type = d.get("type", "?")
            lines.append(f"  {d_id} ({d_type})")
    except Exception as e:
        lines.append(f"Directives: check failed ({e})")

    # 5. Learning engine
    try:
        from agent.integrations.learning_engine import _get_learning_scheduler
        le = _get_learning_scheduler()
        le_status = le.get_status()
        queue_depth = le_status.get("queue_depth", 0)
        surgical = le_status.get("surgical_count", 0)
        cumulative = le_status.get("cumulative_count", 0)
        lines.append("")
        _uptime_s_le = int(time.time() - _SERVICE_START_TIME)
        _up_h_le = _uptime_s_le // 3600
        lines.append(
            f"Learning Engine: queue={queue_depth}, surgical={surgical}, "
            f"session_cycles={cumulative} (since restart {_up_h_le}h ago)"
        )
        # S142-B W1-A: Cumulative cycle count from state file
        # S185 W3 — moved IO to executor; on filesystem stall the event loop must not block.
        try:
            from agent.integrations.learning_engine import LEARNING_STATE_FILE
            def _read_ls_sync():
                with open(LEARNING_STATE_FILE, "r") as _lf:
                    return json.load(_lf)
            _loop_ls = asyncio.get_running_loop()
            _ls = await _loop_ls.run_in_executor(None, _read_ls_sync)
            lines.append(f"  Total cycles (all-time): {_ls.get('cycle_count', '?')}")
        except Exception as e:
            logger.debug(f"Learning state read: {e}")
    except Exception as e:
        lines.append(f"Learning Engine: check failed ({e})")

    # 5b. Knowledge stats (S142-B W1-A)
    try:
        from agent.tools.knowledge_tools import _load_index
        _idx = _load_index()
        _entry_count = len(_idx.get("entries", []))
        lines.append(f"Knowledge: {_entry_count} entries")
    except Exception as e:
        logger.debug(f"Knowledge index read: {e}")

    # 6. Personality
    try:
        _personality_enabled = os.environ.get("PERSONALITY_ENABLED", "").lower() == "true"
        if _personality_enabled:
            from agent.personality.engine import get_personality_engine
            pe = get_personality_engine()
            confirmed = len(pe.registry.get_confirmed_traits())
            proposed = len(pe.registry.get_proposed_traits())
            lines.append(f"Personality: active ({confirmed} confirmed, {proposed} proposed)")
        else:
            lines.append("Personality: disabled")
    except Exception as e:
        lines.append(f"Personality: check failed ({e})")

    # 7. Kill switch
    try:
        from agent.moltbook.kill_switch import MoltbookKillSwitch
        ks = MoltbookKillSwitch()
        lines.append(f"Moltbook Kill Switch: {'ACTIVE' if ks.is_active else 'inactive'}")
    except Exception as e:
        lines.append(f"Kill Switch: check failed ({e})")

    # 8. SEAL bandit
    try:
        _seal_enabled = os.environ.get("MOLTBOT_SEAL_ENABLED", "1") == "1"
        if _seal_enabled:
            from agent.integrations.seal_category_bandit import get_category_bandit
            bandit = get_category_bandit()
            b_status = bandit.get_status()
            # S180: Include total_rewards so 0-reward state is obvious (diagnoses
            # broken trade_idea_id linkage before it gets masked by pull growth).
            _rewards = b_status.get('total_rewards', 0)
            lines.append(
                f"SEAL Bandit: {b_status.get('total_pulls', 0)} pulls, "
                f"{_rewards} rewards, {b_status.get('buckets', 0)} buckets, "
                f"win_rate={b_status.get('overall_win_rate', 0)}"
            )
        else:
            lines.append("SEAL: disabled")
    except Exception as e:
        lines.append(f"SEAL Bandit: check failed ({e})")

    # 9. RSS/Lighthouse feed status
    try:
        _lh_enabled = os.environ.get("CLIO_LIGHTHOUSE_ENABLED", "1") == "1"
        if _lh_enabled:
            from agent.integrations.global_news_feeds import get_feed_stats
            feed_stats = get_feed_stats()
            total_feeds = feed_stats.get("total_feeds", 0)
            active_feeds = feed_stats.get("active_feeds", 0)
            window = feed_stats.get("active_window", "none")
            _lh_log_only = os.environ.get("CLIO_LIGHTHOUSE_LOG_ONLY", "1") == "1"
            _lh_qualifier = " [LOG_ONLY]" if _lh_log_only else ""
            # S180: Show disabled count so "missing" feeds aren't mistaken for broken ones.
            # Upstream-dead feeds are permanently disabled in FEED_REGISTRY (S135).
            disabled_count = feed_stats.get("disabled_feeds", 0)
            disabled_qualifier = f" ({disabled_count} disabled upstream)" if disabled_count > 0 else ""
            lines.append(
                f"RSS Feeds: {active_feeds}/{total_feeds} active{disabled_qualifier}"
                f"{_lh_qualifier}, window: {window}"
            )
        else:
            lines.append("RSS Feeds: disabled")
    except Exception as e:
        # Lighthouse may not be active — just skip
        logger.debug(f"Health report RSS check skipped: {e}")

    # 10. CLIO idea outcomes (S182 M1+H3): last 20 consumed ideas, agreement rate,
    # and attribution rate (credited via HI-9 direction match). Hydrinia is
    # permanently off — consumed_by_hydrinia can never be set again — so this
    # section is bridge-gated rather than always showing a frozen/empty result.
    if bridge_enabled():
      try:
        # S185 W3 — moved glob + 30 file reads to executor (was blocking event loop).
        import glob as _g_ideas
        _ideas_dir = os.path.join(config.tradingbot_dir, "moltbot_knowledge", "trade_ideas")
        def _scan_consumed_ideas_sync():
            _result = []
            for _idf in sorted(_g_ideas.glob(os.path.join(_ideas_dir, "ti_*.json")))[-30:]:
                try:
                    with open(_idf) as _fh:
                        _d = json.load(_fh)
                    if _d.get("consumed_by_hydrinia"):
                        _result.append(_d)
                except (OSError, json.JSONDecodeError):
                    continue
            return _result
        _loop_ci = asyncio.get_running_loop()
        _consumed = await _loop_ci.run_in_executor(None, _scan_consumed_ideas_sync)
        if _consumed:
            _agreed = sum(1 for d in _consumed if d.get("hydrinia_direction_agreement") == "agreed")
            _flipped = sum(1 for d in _consumed if d.get("hydrinia_direction_agreement") == "flipped")
            _hold = sum(1 for d in _consumed if d.get("hydrinia_direction_agreement") == "hold")
            # Older ideas (pre-S182 L1) won't have the field — count as unknown
            _unknown = len(_consumed) - _agreed - _flipped - _hold
            lines.append("")
            lines.append(f"CLIO Ideas (last {len(_consumed)} consumed): "
                         f"agreed={_agreed}, flipped={_flipped}, hold={_hold}" +
                         (f", unknown={_unknown}" if _unknown else ""))
            # Attribution rate: agreed = potentially credited via HI-9
            if len(_consumed) > 0:
                _attr_rate = 100.0 * _agreed / len(_consumed)
                lines.append(f"  Agreement rate: {_attr_rate:.0f}% (agreed ideas are HI-9 credit candidates)")
            # Most recent 3
            for d in _consumed[-3:]:
                _ag = d.get("hydrinia_direction_agreement", "?")
                lines.append(f"  {d.get('ticker','?')} {d.get('direction','?')}@{d.get('confidence',0)}: "
                             f"{d.get('hydrinia_result','?')} ({_ag})")
      except Exception as _e_ideas:
        logger.debug(f"Health report CLIO outcomes skipped: {_e_ideas}")

    # 11. S183 F3 — moltbot execution shadow stats (LOG-ONLY observation window).
    # moltbot_execution_shadow.jsonl lives in the Hydrinia bridge dir and nothing
    # writes it any more now that Hydrinia is off — bridge-gated for the same
    # reason as section 10 above.
    if bridge_enabled():
      try:
        _shadow_path = os.path.join(
            config.tradingbot_dir, "moltbot_knowledge", "moltbot_execution_shadow.jsonl"
        )
        if os.path.isfile(_shadow_path):
            # S185 W3 — moved JSONL parse to executor (was blocking event loop on
            # potentially long files; shadow journal grows unbounded).
            def _read_shadow_sync():
                _entries = []
                with open(_shadow_path) as _sh:
                    for _line in _sh:
                        _line = _line.strip()
                        if not _line:
                            continue
                        try:
                            _entries.append(json.loads(_line))
                        except json.JSONDecodeError:
                            continue
                return _entries
            _loop_sh = asyncio.get_running_loop()
            _shadow_entries = await _loop_sh.run_in_executor(None, _read_shadow_sync)
            if _shadow_entries:
                # S170B P2-3: parse cutoff and entry ts in the same epoch frame
                # to survive backward NTP correction; reject "future" entries.
                from datetime import datetime as _dt_sh, timezone as _tz_sh, timedelta as _td_sh
                _now_utc = _dt_sh.now(_tz_sh.utc)
                _cutoff_dt = _now_utc - _td_sh(days=7)
                _recent = []
                for _s in _shadow_entries:
                    try:
                        _entry_dt = _dt_sh.fromisoformat(_s.get("timestamp", ""))
                        if _entry_dt.tzinfo is None:
                            _entry_dt = _entry_dt.replace(tzinfo=_tz_sh.utc)
                        if _cutoff_dt <= _entry_dt <= _now_utc:
                            _recent.append(_s)
                    except (ValueError, TypeError):
                        _recent.append(_s)
                _would_exec = sum(1 for s in _recent if s.get("would_execute"))
                _blocked = len(_recent) - _would_exec
                _live = sum(
                    1 for s in _recent
                    if (not s.get("log_only_flag", True)) and s.get("enabled_flag", False)
                )
                _mode = "LIVE" if _live > 0 else "LOG-ONLY"
                lines.append("")
                lines.append(f"Moltbot Exec Shadow ({_mode}, last 7d): "
                             f"would_execute={_would_exec}, blocked={_blocked}, total={len(_recent)}")
                # Top block reasons
                if _blocked > 0:
                    _blocks_by_reason: dict[str, int] = {}
                    for _s in _recent:
                        if not _s.get("would_execute"):
                            _by = _s.get("gate_result", {}).get("blocked_by") or []
                            _key = ",".join(_by) if _by else "unknown"
                            _blocks_by_reason[_key] = _blocks_by_reason.get(_key, 0) + 1
                    _top = sorted(_blocks_by_reason.items(), key=lambda x: -x[1])[:3]
                    if _top:
                        _top_str = ", ".join(f"{k}={v}" for k, v in _top)
                        lines.append(f"  Block reasons: {_top_str}")
      except Exception as _e_shadow:
        logger.debug(f"Health report moltbot shadow stats skipped: {_e_shadow}")

    # Build and send
    lines.append("")
    _ny = ZoneInfo("America/New_York")
    lines.append(f"Report time: {datetime.now(_ny).strftime('%I:%M %p ET')}")

    message = "\n".join(lines)
    if len(message) > 4000:
        message = message[:3997] + "..."

    logger.debug("System health report:\n%s", message)

    for chat_id in chat_ids:
        await _safe_send(context.bot, chat_id, text=message)

    logger.info("System health report sent")


# ============================================================
# Shutdown
# ============================================================

async def shutdown_background_tasks():
    """Cancel background tasks on graceful shutdown."""
    logger.info("CLIO shutdown stage ENTER: shutdown_background_tasks.body")
    global _heartbeat_task
    if _heartbeat_task and not _heartbeat_task.done():
        _heartbeat_task.cancel()
        try:
            await _heartbeat_task
        except asyncio.CancelledError:
            pass
        logger.info("Moltbook heartbeat task cancelled")

    # Stop price stream manager (unconditionally — manager tracks its own state)
    try:
        from agent.integrations.price_stream import get_stream_manager
        mgr = get_stream_manager()
        if mgr.is_running:
            await mgr.stop()
            logger.info("Price stream manager stopped")
    except Exception as e:
        logger.debug(f"Price stream shutdown error: {e}")

    # Stop the DPOF engine (closes its jev client + data feed; no-op if never started)
    try:
        from agent.dpof.engine import shutdown_dpof
        await shutdown_dpof()
        logger.info("DPOF engine shutdown complete")
    except Exception as e:
        logger.debug(f"DPOF engine shutdown error: {e}")

    # Close httpx clients on shutdown
    for _mod, _label in [
        ("agent.integrations.uw_flow_intel", "UW flow intel"),
        ("agent.integrations.unusual_whales", "Unusual Whales"),
        ("agent.integrations.polygon", "Polygon"),
    ]:
        try:
            _m = __import__(_mod, fromlist=["close_client"])
            await _m.close_client()
        except Exception as e:
            logger.debug(f"{_label} client shutdown error: {e}")

    # Flush the local-first UW quota outbox and stop its daemon worker. This is
    # bounded and never waits on Supabase indefinitely.
    try:
        from agent.integrations.uw_quota_client import close_all_uw_quota_clients
        timeout_s = float(os.environ.get("CLIO_UW_QUOTA_SHUTDOWN_TIMEOUT_S", "5"))
        loop = asyncio.get_running_loop()
        quota_closed = await loop.run_in_executor(
            None,
            lambda: close_all_uw_quota_clients(timeout_s=max(0.1, min(timeout_s, 30.0))),
        )
        logger.info("UW quota worker shutdown complete=%s", quota_closed)
    except Exception as e:
        logger.warning("UW quota worker shutdown error: %s", e)

    # Stop personality executors so interpreter exit does not wait on non-daemon pools.
    for _mod, _func, _label in [
        ("agent.personality.reranker_semantic", "shutdown_executor", "semantic reranker"),
        ("agent.personality.retrieval", "shutdown_shadow_executor", "personality shadow"),
    ]:
        try:
            _m = __import__(_mod, fromlist=[_func])
            getattr(_m, _func)(wait=False)
            logger.info(f"{_label} executor shutdown requested")
        except Exception as e:
            logger.debug(f"{_label} executor shutdown error: {e}")

    logger.info("CLIO shutdown stage EXIT: shutdown_background_tasks.body")


# ============================================================
# Task: Bridge File Cleanup (daily)
# ============================================================

async def _cleanup_bridge_files(context) -> None:
    """Remove bridge files older than 7 days from moltbot_knowledge/ subdirectories.

    Cleans: market_analysis, strategies, bot_status, trades,
            reports/hourly, reports/news, reports/premarket, reports/postmarket.
    Does NOT clean: personality/, working_memory/, directives/, urgent/, adaptations/.
    """
    # Hydrinia (/home/agent/tradingbot) is permanently off — moltbot_knowledge/ is its
    # bridge dir and nothing writes to it any more. Skip pruning it, but keep the
    # CLIO-native cleanup below (trade ideas, SEAL impl files, WAL checkpoint) running.
    if bridge_enabled():
        bridge_dir = os.path.join(config.tradingbot_dir, "moltbot_knowledge")
        cleanup_subdirs = [
            "market_analysis",
            "strategies",
            "bot_status",
            "trades",
            os.path.join("reports", "hourly"),
            os.path.join("reports", "news"),
            os.path.join("reports", "premarket"),
            os.path.join("reports", "postmarket"),
        ]

        cutoff = time.time() - (7 * 86400)  # 7 days ago
        total_removed = 0

        for subdir in cleanup_subdirs:
            full_path = os.path.join(bridge_dir, subdir)
            if not os.path.isdir(full_path):
                continue
            try:
                for fname in os.listdir(full_path):
                    fpath = os.path.join(full_path, fname)
                    if not os.path.isfile(fpath):
                        continue
                    try:
                        if os.path.getmtime(fpath) < cutoff:
                            os.remove(fpath)
                            total_removed += 1
                    except OSError:
                        pass
            except OSError as e:
                logger.warning(f"Bridge cleanup failed for {subdir}: {e}")

        # Extended cleanup with shorter TTLs (S142-B W4-B)
        _extended_dirs = {
            "urgent": 2 * 86400,          # 2 days
            "intelligence": 7 * 86400,     # 7 days
            "trade_ideas": 7 * 86400,      # 7 days
            "trade_debriefs": 14 * 86400,  # 14 days (keep longer for feedback)
            "uw_signals": 3 * 86400,       # 3 days
            "gap_signals": 3 * 86400,      # 3 days
        }
        for _ext_subdir, _ext_ttl in _extended_dirs.items():
            _ext_path = os.path.join(bridge_dir, _ext_subdir)
            if not os.path.isdir(_ext_path):
                continue
            _ext_cutoff = time.time() - _ext_ttl
            try:
                for fname in os.listdir(_ext_path):
                    fpath = os.path.join(_ext_path, fname)
                    if os.path.isfile(fpath):
                        try:
                            if os.path.getmtime(fpath) < _ext_cutoff:
                                os.remove(fpath)
                                total_removed += 1
                        except OSError:
                            pass
            except OSError as e:
                logger.debug(f"Bridge cleanup error in {_ext_subdir}: {e}")

        if total_removed > 0:
            logger.info(f"Bridge file cleanup: removed {total_removed} files")

    # S156: Prune CLIO trade idea files older than 14 days
    _proj_root_cleanup = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _ti_dir = os.path.join(_proj_root_cleanup, "data", "knowledge", "trade_ideas")
    if os.path.isdir(_ti_dir):
        _ti_cutoff = time.time() - (14 * 86400)
        _ti_pruned = 0
        try:
            for _f in os.listdir(_ti_dir):
                if _f.startswith("ti_") and _f.endswith(".json"):
                    _fp = os.path.join(_ti_dir, _f)
                    try:
                        if os.path.getmtime(_fp) < _ti_cutoff:
                            os.remove(_fp)
                            _ti_pruned += 1
                    except OSError:
                        pass
        except OSError as e:
            logger.debug(f"Trade idea pruning error: {e}")
        if _ti_pruned:
            logger.info(f"Pruned {_ti_pruned} trade idea files older than 14 days")

    # S156: Prune SEAL impl files older than 30 days
    _ma_dir_cleanup = os.path.join(_proj_root_cleanup, "data", "knowledge", "trading", "market_analysis")
    if os.path.isdir(_ma_dir_cleanup):
        _impl_cutoff = time.time() - (30 * 86400)
        _impl_pruned = 0
        try:
            for _f in os.listdir(_ma_dir_cleanup):
                if _f.startswith("seal_impl_") and _f.endswith(".json"):
                    _fp = os.path.join(_ma_dir_cleanup, _f)
                    try:
                        if os.path.getmtime(_fp) < _impl_cutoff:
                            os.remove(_fp)
                            _impl_pruned += 1
                    except OSError:
                        pass
        except OSError as e:
            logger.debug(f"SEAL impl pruning error: {e}")
        if _impl_pruned:
            logger.info(f"Pruned {_impl_pruned} SEAL impl files older than 30 days")

    # S148-B W3-1: WAL checkpoint (daily alongside bridge cleanup)
    try:
        from agent.core import conversation_manager
        await conversation_manager.checkpoint()
    except Exception as e:
        logger.debug(f"WAL checkpoint in bridge cleanup failed: {e}")


# S186 W4.2: Per-task timeout counter (resets daily) for monitoring without alerting overhead
# Module-level dict keyed by task_name → {date_iso, count}. Resets when local date changes.
# S190 M2: in-memory by design — per CLAUDE.md anti-pattern #18, module-level state is
# acceptable when "the alert window dominates restart rate AND duplicate alerts on restart
# are operationally tolerable." Counter resets on every service restart; absolute count is
# advisory, not forensic. If you need cross-restart persistence, write to state/ and atomic-
# swap; do not enable persistence in this file without operator sign-off (see ADR-003 P3).
_timeout_counter: dict[str, dict] = {}
_timeout_counter_lock = threading.Lock()


def _record_timeout(task_name: str) -> int:
    """Increment timeout count for task_name; reset if new day. Returns today's count."""
    # CLAUDE.md gotcha: scheduled_tasks.py has no module-level _NY — define inline
    _ny = ZoneInfo("America/New_York")
    today_iso = datetime.now(_ny).date().isoformat()
    with _timeout_counter_lock:
        rec = _timeout_counter.get(task_name)
        if rec is None or rec.get("date") != today_iso:
            rec = {"date": today_iso, "count": 0}
        rec["count"] += 1
        _timeout_counter[task_name] = rec
        return rec["count"]


# S215 W6A: per-job duration logging. Successful runs at or above the INFO floor
# log a job_duration line; a run at or above 80% of its timeout logs WARNING instead.
_JOB_DURATION_INFO_S = 5.0
_JOB_DURATION_WARN_FRAC = 0.8
_clock = time.monotonic  # injectable for tests


# S233 HI-1: Timeout wrapper for scheduled tasks — prevents any single hung task
# from stalling the entire job queue (S162/S163-B class of bug).
def _with_timeout(coro_fn, timeout_s: int, name: str):
    """Wrap an async scheduled task callback with asyncio.wait_for timeout."""
    async def _wrapped(context):
        _t0 = _clock()
        try:
            await asyncio.wait_for(coro_fn(context), timeout=timeout_s)
        except asyncio.TimeoutError:
            # S186 W4.2: track per-task daily timeout count for operator visibility
            _today_count = _record_timeout(name)
            logger.error(
                "Scheduled task '%s' timed out after %ds (today_count=%d)",
                name, timeout_s, _today_count,
            )
        except Exception as e:
            logger.error(f"Scheduled task '{name}' failed: {e}", exc_info=True)
        else:
            _dur = _clock() - _t0
            if _dur >= _JOB_DURATION_WARN_FRAC * timeout_s:
                logger.warning("job_duration name=%s s=%.1f timeout=%ds (>=80%% of the timeout)", name, _dur, timeout_s)
            elif _dur >= _JOB_DURATION_INFO_S:
                logger.info("job_duration name=%s s=%.1f", name, _dur)
    _wrapped.__name__ = f"{name}_timeout_wrapped"
    return _wrapped


# Registration
# ============================================================

def _register_oneprovider_monitor(jq) -> bool:
    """OP-10: register the OneProvider balance/canary monitor (6h) only when the
    gateway is on. Returns True when registered. Never raises."""
    try:
        from agent.config import config as _gw_config
        if not _gw_config.anthropic_gateway:
            return False
        from agent.integrations.oneprovider_balance import oneprovider_monitor_job
        jq.run_repeating(
            oneprovider_monitor_job,
            interval=21600,
            first=240,
            name="oneprovider_monitor",
        )
        logger.info("OneProvider monitor registered (21600s interval, first=240s)")
        return True
    except Exception as e:
        logger.error(f"Failed to register OneProvider monitor: {e}", exc_info=True)
        return False


def _register_gateway_stats_tick(jq) -> bool:
    """R-09: hourly gateway stats line (first at 3600s), gateway mode only. Never raises."""
    try:
        from agent.config import config as _gw_config
        if not _gw_config.anthropic_gateway:
            return False
        from agent.anthropic_gateway import gateway_stats_tick_job
        jq.run_repeating(
            gateway_stats_tick_job,
            interval=3600,
            first=3600,
            name="gateway_stats_tick",
        )
        return True
    except Exception as e:
        logger.error(f"Failed to register gateway stats tick: {e}", exc_info=True)
        return False


def register_scheduled_jobs(app: Application) -> None:
    """Register all periodic monitoring jobs with the bot's JobQueue."""
    global _last_log_size, _last_log_ino

    jq = app.job_queue
    has_chat_ids = bool(_get_chat_ids())

    # Hydrinia (/home/agent/tradingbot) is permanently off (operator direction, 2026-09-24).
    # Every job that only reads/writes Hydrinia's process, logs, or bridge files is gated
    # here; CLIO-native jobs (DPOF, market regime, learning, etc.) run unconditionally.
    _bridge_on = bridge_enabled()
    _skipped_hydrinia_jobs: list[str] = []

    # Telegram-dependent monitoring jobs (require at least one chat_id)
    if has_chat_ids:
        if _bridge_on:
            # Initialize log size/inode tracker to current state (don't report old errors on startup)
            try:
                _init_stat = os.stat(_LOG_FILE)
                _last_log_size = _init_stat.st_size
                _last_log_ino = _init_stat.st_ino
            except OSError:
                pass

            jq.run_repeating(
                check_hydrinia_status,
                interval=STATUS_CHECK_INTERVAL,
                first=STATUS_FIRST_RUN,
                name="hydrinia_status_check",
            )
            jq.run_repeating(
                scan_hydrinia_errors,
                interval=ERROR_SCAN_INTERVAL,
                first=ERROR_FIRST_RUN,
                name="hydrinia_error_scan",
            )
            jq.run_repeating(
                _with_timeout(check_portfolio_snapshot, 60, "portfolio_snapshot"),  # S233
                interval=PORTFOLIO_INTERVAL,
                first=PORTFOLIO_FIRST_RUN,
                name="hydrinia_portfolio_snapshot",
            )
            logger.info(
                f"Telegram monitoring jobs registered: "
                f"status_check ({STATUS_CHECK_INTERVAL}s), "
                f"error_scan ({ERROR_SCAN_INTERVAL}s), "
                f"portfolio ({PORTFOLIO_INTERVAL}s)"
            )
        else:
            _skipped_hydrinia_jobs.extend([
                "hydrinia_status_check", "hydrinia_error_scan", "hydrinia_portfolio_snapshot",
            ])

        # S185 W1-1: Anthropic credit/health monitor (every 30 min)
        try:
            from agent.integrations.anthropic_health import anthropic_health_check_job
            jq.run_repeating(
                anthropic_health_check_job,
                interval=1800,
                first=120,
                name="anthropic_health_check",
            )
            logger.info("Anthropic health check registered (1800s interval, first=120s)")
        except Exception as e:
            logger.error(f"Failed to register anthropic health check: {e}", exc_info=True)

        # OP-10: OneProvider gateway balance/expiry monitor + served-model canary (6h)
        _register_oneprovider_monitor(jq)
        _register_gateway_stats_tick(jq)  # R-09 hourly gateway stats line

        # S185 W1-3: Intelligence-file staleness sentinel (hourly scan + 08:00 ET digest)
        try:
            from agent.integrations.staleness_sentinel import (
                intelligence_staleness_scan_job,
                staleness_digest_job,
            )
            jq.run_repeating(
                intelligence_staleness_scan_job,
                interval=3600,
                first=300,
                name="intelligence_staleness_scan",
            )
            _ny = ZoneInfo("America/New_York")
            _now = datetime.now(_ny)
            _digest_target = _now.replace(hour=8, minute=0, second=0, microsecond=0)
            if _now >= _digest_target:
                _digest_target += timedelta(days=1)
            _digest_first = (_digest_target - _now).total_seconds()
            jq.run_repeating(
                staleness_digest_job,
                interval=86400,
                first=_digest_first,
                name="staleness_digest",
            )
            logger.info(
                f"Staleness sentinel registered (scan every 3600s, "
                f"digest first in {_digest_first / 3600:.1f}h)"
            )
        except Exception as e:
            logger.error(f"Failed to register staleness sentinel: {e}", exc_info=True)
    else:
        logger.warning("TELEGRAM_NOTIFY_CHAT_ID not set — Telegram monitoring jobs disabled")

    # Autonomous learning pipeline (does NOT require chat_id — runs independently)
    try:
        from agent.integrations.learning_engine import run_learning_cycle
        jq.run_repeating(
            _with_timeout(run_learning_cycle, 900, "learning_cycle"),  # S170B: 15 min timeout (was 300s; SEAL multigen legitimately exceeds 300s on heavy load)
            interval=LEARNING_INTERVAL,
            first=LEARNING_FIRST_RUN,
            name="learning_cycle",
        )
        logger.info(f"Learning pipeline registered ({LEARNING_INTERVAL}s interval)")
    except Exception as e:
        logger.error(f"Failed to register learning pipeline: {e}", exc_info=True)

    # Trade outcome feedback loop (adjusts learning category weights)
    try:
        from agent.integrations.trade_feedback import analyze_trade_feedback
        jq.run_repeating(
            _with_timeout(analyze_trade_feedback, 120, "trade_feedback"),  # S233
            interval=FEEDBACK_INTERVAL,
            first=FEEDBACK_FIRST_RUN,
            name="trade_feedback",
        )
        logger.info(f"Trade feedback pipeline registered ({FEEDBACK_INTERVAL}s interval)")
    except Exception as e:
        logger.error(f"Failed to register trade feedback: {e}", exc_info=True)

    # Market event detection (breaking news mode for learning engine)
    try:
        from agent.integrations.market_events import check_market_events
        jq.run_repeating(
            _with_timeout(check_market_events, 60, "market_events"),  # S233
            interval=MARKET_EVENT_INTERVAL,
            first=MARKET_EVENT_FIRST_RUN,
            name="market_event_check",
        )
        logger.info(f"Market event detection registered ({MARKET_EVENT_INTERVAL}s interval)")
    except Exception as e:
        logger.error(f"Failed to register market event detection: {e}", exc_info=True)

    # CLIO DPOF long-options engine (self-gates on CLIO_DPOF_MODE; default off -> immediate no-op)
    try:
        from agent.dpof.engine import run_dpof_cycle, DPOF_JOB_INTERVAL, DPOF_JOB_TIMEOUT
        jq.run_repeating(
            _with_timeout(run_dpof_cycle, DPOF_JOB_TIMEOUT, "dpof_cycle"),
            interval=DPOF_JOB_INTERVAL,
            first=60,
            name="dpof_cycle",
        )
        logger.info(f"DPOF engine registered ({DPOF_JOB_INTERVAL}s interval)")
    except Exception as e:
        logger.error(f"Failed to register DPOF engine: {e}", exc_info=True)

    # CLIO-native market regime refresh (replaces Hydrinia daily_metrics.json["market_regime"])
    try:
        from agent.integrations.market_regime import refresh_market_regime
        jq.run_repeating(
            _with_timeout(refresh_market_regime, 60, "market_regime_refresh"),
            interval=1800,
            first=60,
            name="market_regime_refresh",
        )
        logger.info("Market regime refresh registered (1800s interval)")
    except Exception as e:
        logger.warning(f"Market regime module unavailable — skipping registration: {e}")

    # Intel v3 (2026-09-25) — ADVISORY context for the DPOF decider/Jev, review triggers and
    # Telegram alerts. None of these jobs places, sizes or cancels an order.
    # Vol surface (SPY/QQQ): self-gates to weekdays 04:00-20:00 ET, so a 30-min interval also
    # gives the premarket runs.
    try:
        from agent.integrations.vol_surface import refresh_vol_surface
        jq.run_repeating(
            _with_timeout(refresh_vol_surface, 120, "vol_surface_refresh"),
            interval=1800,
            first=90,
            name="vol_surface_refresh",
        )
        logger.info("Vol surface refresh registered (1800s interval)")
    except Exception as e:
        logger.warning(f"Vol surface module unavailable — skipping registration: {e}")

    # Econ surprise (actual vs consensus): one short tick per invocation — fast-polls only
    # inside importance>=2 release windows, otherwise a <=30-min consensus refresh or idle.
    try:
        from agent.integrations import econ_surprise as _es
        _es_interval = getattr(_es, "JOB_INTERVAL_SECONDS", 15)
        _es_timeout = getattr(_es, "JOB_TIMEOUT_SECONDS", 60)
        jq.run_repeating(
            _with_timeout(_es.refresh_econ_surprise, _es_timeout, "econ_surprise_refresh"),
            interval=_es_interval,
            first=45,
            name="econ_surprise_refresh",
        )
        logger.info(f"Econ surprise refresh registered ({_es_interval}s interval)")
    except Exception as e:
        logger.warning(f"Econ surprise module unavailable — skipping registration: {e}")

    # NewsFlash-lite (breaking-news classifier over UW / Polygon / Lighthouse headlines).
    try:
        from agent.integrations import newsflash_lite as _nf
        _nf_interval = getattr(_nf, "JOB_INTERVAL_SECONDS", 60)
        _nf_timeout = getattr(_nf, "JOB_TIMEOUT_SECONDS", 60)
        jq.run_repeating(
            _with_timeout(_nf.refresh_newsflash, _nf_timeout, "newsflash_refresh"),
            interval=_nf_interval,
            first=75,
            name="newsflash_refresh",
        )
        logger.info(f"NewsFlash-lite registered ({_nf_interval}s interval)")
    except Exception as e:
        logger.warning(f"NewsFlash-lite module unavailable — skipping registration: {e}")

    # Book corpus pipeline (Master Research Agenda — 22 foundational books)
    try:
        from agent.integrations.book_pipeline import run_book_cycle
        jq.run_repeating(
            _with_timeout(run_book_cycle, 300, "book_corpus"),  # S233
            interval=BOOK_INTERVAL,
            first=BOOK_FIRST_RUN,
            name="book_corpus",
        )
        logger.info(f"Book corpus pipeline registered ({BOOK_INTERVAL}s interval)")
    except Exception as e:
        logger.error(f"Failed to register book corpus pipeline: {e}", exc_info=True)

    # Moltbook heartbeat (long-lived background async task)
    try:
        from agent.moltbook.config import get_moltbook_config
        _mb_config = get_moltbook_config()
        if _mb_config.enabled:
            import asyncio as _asyncio
            from agent.moltbook.heartbeat import MoltbookHeartbeat

            async def _start_moltbook_heartbeat(context):
                async def _mb_alert(msg):
                    for cid in _get_chat_ids():
                        await _safe_send(context.bot, cid, text=msg)

                global _heartbeat_task
                hb = MoltbookHeartbeat(config=_mb_config, telegram_alert_fn=_mb_alert)
                _heartbeat_task = _asyncio.create_task(hb.run())
                logger.info("Moltbook heartbeat loop started")

            jq.run_once(
                _start_moltbook_heartbeat,
                when=300,  # 5 minutes after startup — gives other systems time to initialize
                name="moltbook_heartbeat_start",
            )
            logger.info("Moltbook heartbeat scheduled to start in 5min")
        else:
            logger.info("Moltbook disabled — heartbeat not registered")
    except Exception as e:
        logger.error(f"Failed to register Moltbook heartbeat: {e}", exc_info=True)

    # Personality snapshot cold-sync (one-shot at T+30s — exports snapshot for Hydrinia bridge)
    try:
        _personality_enabled = os.environ.get("PERSONALITY_ENABLED", "").lower() == "true"
        if _personality_enabled and _bridge_on:
            from agent.personality.engine import get_personality_engine

            async def _export_personality_snapshot(context):
                try:
                    engine = get_personality_engine()
                    engine.export_snapshot()
                    logger.info("Personality snapshot cold-sync export complete")
                except Exception as e:
                    logger.warning(f"Personality snapshot cold-sync failed: {e}")

            jq.run_once(
                _export_personality_snapshot,
                when=30,
                name="personality_snapshot_cold_sync",
            )
            logger.info("Personality snapshot cold-sync scheduled (T+30s)")
        elif not _personality_enabled:
            logger.info("Personality disabled — snapshot cold-sync not registered")
        else:
            _skipped_hydrinia_jobs.append("personality_snapshot_cold_sync")
    except Exception as e:
        logger.error(f"Failed to register personality snapshot cold-sync: {e}", exc_info=True)

    # Pre-market alignment check (9:00 AM ET weekdays — Feature 4)
    try:
        from agent.integrations.alignment_checker import run_alignment_check as _real_alignment_check

        async def _guarded_alignment_check(context):
            if _daily_task_already_ran("alignment_check"):
                logger.debug("Alignment check already ran today, skipping")
                return
            try:
                await _real_alignment_check(context)
                await asyncio.get_running_loop().run_in_executor(None, _mark_daily_task_ran, "alignment_check")
            except Exception as e:
                logger.error(f"Guarded alignment check failed: {e}", exc_info=True)

        _ny = ZoneInfo("America/New_York")
        _now = datetime.now(_ny)
        _target = _now.replace(hour=9, minute=0, second=0, microsecond=0)
        if _now >= _target:
            _target += timedelta(days=1)
        # Skip weekends
        while _target.weekday() >= 5:
            _target += timedelta(days=1)
        _first_delay = (_target - _now).total_seconds()

        jq.run_repeating(
            _guarded_alignment_check,
            interval=86400,  # Daily
            first=_first_delay,
            name="alignment_check",
        )
        logger.info(f"Pre-market alignment check registered (first run in {_first_delay / 3600:.1f}h)")
        # Catch-up: if bot starts after 9:00 AM but before 10:30 AM on weekday, run soon
        if _now.weekday() < 5:
            _align_target = _now.replace(hour=9, minute=0, second=0, microsecond=0)
            _align_catchup_end = _now.replace(hour=10, minute=30, second=0, microsecond=0)
            if _align_target <= _now < _align_catchup_end:
                jq.run_once(_guarded_alignment_check, when=210)
                logger.info("Alignment check catch-up scheduled in 210s")
    except Exception as e:
        logger.error(f"Failed to register alignment check: {e}", exc_info=True)

    # Weekly knowledge synthesis (Friday 4:30 PM ET — Feature 5)
    try:
        from agent.integrations.weekly_synthesis import run_weekly_synthesis

        _ny = ZoneInfo("America/New_York")
        _now = datetime.now(_ny)
        _days_until_friday = (4 - _now.weekday()) % 7
        # S118 W-028: If it's Friday and already past 4:30 PM ET, push to next Friday.
        # Boundary: hour>16 catches 17:00+; hour==16 && minute>=30 catches 16:30-16:59.
        if _days_until_friday == 0 and (_now.hour > 16 or (_now.hour == 16 and _now.minute >= 30)):
            _days_until_friday = 7
        _synth_target = (_now + timedelta(days=_days_until_friday)).replace(
            hour=16, minute=30, second=0, microsecond=0
        )
        _synth_first = (_synth_target - _now).total_seconds()

        jq.run_repeating(
            run_weekly_synthesis,
            interval=604800,  # Weekly (7 days)
            first=_synth_first,
            name="weekly_synthesis",
        )
        logger.info(f"Weekly synthesis registered (first run in {_synth_first / 3600:.1f}h)")
    except Exception as e:
        logger.error(f"Failed to register weekly synthesis: {e}", exc_info=True)

    # Contradiction scan (Friday 4:35 PM ET, after synthesis — Feature 6)
    try:
        from agent.integrations.contradiction_detector import run_full_contradiction_scan

        _ny = ZoneInfo("America/New_York")
        _now = datetime.now(_ny)
        _days_until_friday = (4 - _now.weekday()) % 7
        # S118 W-029: If it's Friday and already past 4:35 PM ET, push to next Friday.
        # Boundary: hour>16 catches 17:00+; hour==16 && minute>=35 catches 16:35-16:59.
        if _days_until_friday == 0 and (_now.hour > 16 or (_now.hour == 16 and _now.minute >= 35)):
            _days_until_friday = 7
        _contra_target = (_now + timedelta(days=_days_until_friday)).replace(
            hour=16, minute=35, second=0, microsecond=0
        )
        _contra_first = (_contra_target - _now).total_seconds()

        jq.run_repeating(
            run_full_contradiction_scan,
            interval=604800,  # Weekly (7 days)
            first=_contra_first,
            name="contradiction_scan",
        )
        logger.info(f"Contradiction scan registered (first run in {_contra_first / 3600:.1f}h)")
    except Exception as e:
        logger.error(f"Failed to register contradiction scan: {e}", exc_info=True)

    # Directive feedback polling (300s — operator directive compliance notifications)
    if _bridge_on:
        try:
            from agent.integrations.directive_feedback import poll_directive_feedback
            jq.run_repeating(
                poll_directive_feedback,
                interval=300,
                first=300,  # 300s after startup
                name="directive_feedback",
            )
            logger.info("Directive feedback polling registered (300s interval)")
        except Exception as e:
            logger.error(f"Failed to register directive feedback: {e}", exc_info=True)
    else:
        _skipped_hydrinia_jobs.append("directive_feedback")

    # Self-check (every 10 min — directives, memory, bridge files, tasks)
    try:
        jq.run_repeating(
            run_self_check,
            interval=SELF_CHECK_INTERVAL,
            first=SELF_CHECK_FIRST_RUN,
            name="self_check",
        )
        logger.info(f"Self-check registered ({SELF_CHECK_INTERVAL}s interval)")
    except Exception as e:
        logger.error(f"Failed to register self-check: {e}", exc_info=True)

    # System health report (hourly — comprehensive status to Telegram)
    if has_chat_ids:
        try:
            jq.run_repeating(
                run_system_health_report,
                interval=HEALTH_REPORT_INTERVAL,
                first=HEALTH_REPORT_FIRST_RUN,
                name="system_health_report",
            )
            logger.info(f"System health report registered ({HEALTH_REPORT_INTERVAL}s interval)")
        except Exception as e:
            logger.error(f"Failed to register system health report: {e}", exc_info=True)

    # Pre-market scan (9:00 AM ET weekdays — comprehensive morning brief)
    try:
        from agent.integrations.premarket_scan import run_premarket_scan as _real_premarket_scan

        async def _guarded_premarket_scan(context):
            if _daily_task_already_ran("premarket_scan"):
                logger.info("premarket_scan already ran today, skipping")
                return
            await _real_premarket_scan(context)
            await asyncio.get_running_loop().run_in_executor(None, _mark_daily_task_ran, "premarket_scan")

        _ny = ZoneInfo("America/New_York")
        _now = datetime.now(_ny)
        _target_pm = _now.replace(hour=9, minute=0, second=0, microsecond=0)
        if _now >= _target_pm:
            _target_pm += timedelta(days=1)
        while _target_pm.weekday() >= 5:
            _target_pm += timedelta(days=1)
        _first_pm = (_target_pm - _now).total_seconds()

        # Catch-up: if we're past 9 AM but within a 90-min window, run soon
        _catchup_start = _now.replace(hour=9, minute=0, second=0, microsecond=0)
        _catchup_end = _now.replace(hour=10, minute=30, second=0, microsecond=0)
        if _now.weekday() < 5 and _now >= _catchup_start and _now < _catchup_end:
            _first_pm = 60  # Run in 60 seconds as catch-up

        jq.run_repeating(
            _with_timeout(_guarded_premarket_scan, 180, "premarket_scan"),  # S233
            interval=86400,
            first=_first_pm,
            name="premarket_scan",
        )
        logger.info(f"Pre-market scan registered (first run in {_first_pm / 3600:.1f}h)")
    except Exception as e:
        logger.error(f"Failed to register pre-market scan: {e}", exc_info=True)

    # Hourly trade debrief (9:30 AM–4:30 PM ET weekdays, every hour)
    try:
        from agent.integrations.hourly_debrief import run_hourly_debrief

        _ny = ZoneInfo("America/New_York")
        _now = datetime.now(_ny)
        # Calculate first run to next :30 mark
        _next_30 = _now.replace(minute=30, second=0, microsecond=0)
        if _now.minute >= 30:
            _next_30 += timedelta(hours=1)
        _first_hd = (_next_30 - _now).total_seconds()
        if _first_hd <= 0:
            _first_hd = 3600

        jq.run_repeating(
            _with_timeout(run_hourly_debrief, 180, "hourly_debrief"),  # S233
            interval=3600,
            first=_first_hd,
            name="hourly_debrief",
        )
        logger.info(f"Hourly debrief registered (first run in {_first_hd / 60:.0f}min)")
    except Exception as e:
        logger.error(f"Failed to register hourly debrief: {e}", exc_info=True)

    # CC-MB-UW-5: UW Intelligence → Knowledge entries (hourly, aligned with debrief)
    try:
        from agent.tools.knowledge_tools import generate_uw_intelligence_entries

        async def _run_uw_intel(context) -> None:
            """Wrapper for UW intelligence generation (fire-and-forget)."""
            try:
                count = await generate_uw_intelligence_entries()
                if count > 0:
                    logger.info(f"UW intelligence: {count} entries generated")
            except Exception as e:
                logger.debug(f"UW intelligence generation failed: {e}")

        # Calculate first run independently (don't depend on _first_hd from prior try block)
        _ny_uw = ZoneInfo("America/New_York")
        _now_uw = datetime.now(_ny_uw)
        _next_35 = _now_uw.replace(minute=35, second=0, microsecond=0)
        if _now_uw.minute >= 35:
            _next_35 += timedelta(hours=1)
        _first_uw = max((_next_35 - _now_uw).total_seconds(), 300)

        jq.run_repeating(
            _run_uw_intel,
            interval=3600,
            first=_first_uw,
            name="uw_intelligence",
        )
        logger.info(f"UW intelligence generation registered (first run in {_first_uw / 60:.0f}min)")
    except Exception as e:
        logger.error(f"Failed to register UW intelligence: {e}", exc_info=True)

    # CC-CLIO-UW-5 v2: Extended UW Intelligence → Trident generators 4-8 (hourly, staggered +5min)
    try:
        from agent.tools.knowledge_tools import generate_uw_intelligence_entries_v2

        async def _run_uw_intel_v2(context) -> None:
            """Wrapper for extended UW intelligence generation (Trident generators 4-8)."""
            try:
                count = await generate_uw_intelligence_entries_v2()
                if count > 0:
                    logger.info(f"UW intelligence v2: {count} entries generated")
            except Exception as e:
                logger.debug(f"UW intelligence v2 generation failed: {e}")

        # Stagger 5 minutes after the v1 job (v1 runs at :35, v2 at :40 past the hour)
        _ny_uw2 = ZoneInfo("America/New_York")
        _now_uw2 = datetime.now(_ny_uw2)
        _next_40 = _now_uw2.replace(minute=40, second=0, microsecond=0)
        if _now_uw2.minute >= 40:
            _next_40 += timedelta(hours=1)
        _first_uw2 = max((_next_40 - _now_uw2).total_seconds(), 300)

        jq.run_repeating(
            _run_uw_intel_v2,
            interval=3600,
            first=_first_uw2,
            name="uw_intelligence_v2",
        )
        logger.info(f"UW intelligence v2 registered (first run in {_first_uw2 / 60:.0f}min)")
    except Exception as e:
        logger.error(f"Failed to register UW intelligence v2: {e}", exc_info=True)

    # Post-market debrief (5:00 PM ET weekdays — comprehensive EOD)
    try:
        from agent.integrations.postmarket_debrief import run_postmarket_debrief as _real_postmarket

        async def _guarded_postmarket(context):
            if _daily_task_already_ran("postmarket_debrief"):
                logger.info("postmarket_debrief already ran today, skipping")
                return
            await _real_postmarket(context)
            await asyncio.get_running_loop().run_in_executor(None, _mark_daily_task_ran, "postmarket_debrief")

        _ny = ZoneInfo("America/New_York")
        _now = datetime.now(_ny)
        _target_eod = _now.replace(hour=17, minute=0, second=0, microsecond=0)
        if _now >= _target_eod:
            _target_eod += timedelta(days=1)
        while _target_eod.weekday() >= 5:
            _target_eod += timedelta(days=1)
        _first_eod = (_target_eod - _now).total_seconds()

        # Catch-up: if we're past 5 PM but within a 2h window, run soon
        _catchup_eod_start = _now.replace(hour=17, minute=0, second=0, microsecond=0)
        _catchup_eod_end = _now.replace(hour=19, minute=0, second=0, microsecond=0)
        if _now.weekday() < 5 and _now >= _catchup_eod_start and _now < _catchup_eod_end:
            _first_eod = 60  # Run in 60 seconds as catch-up

        jq.run_repeating(
            _with_timeout(_guarded_postmarket, 180, "postmarket_debrief"),  # S233
            interval=86400,
            first=_first_eod,
            name="postmarket_debrief",
        )
        logger.info(f"Post-market debrief registered (first run in {_first_eod / 3600:.1f}h)")
    except Exception as e:
        logger.error(f"Failed to register post-market debrief: {e}", exc_info=True)

    # CLIO DPOF EOD digest (16:10 ET weekdays -- the automatic answer to "why
    # didn't CLIO trade today"; Task ORCH-7Q4K-DPOF, 2026-09-25). Self-gates on
    # CLIO_DPOF_EOD_DIGEST (config.eod_digest_enabled, default True) and skips
    # market holidays via the same MARKET_HOLIDAYS set the DPOF engine itself
    # uses (agent.integrations.market_events) -- weekday alone isn't enough.
    try:
        from agent.dpof.config import load_dpof_config
        from agent.dpof.readmodel import eod_digest_text, split_for_telegram

        async def _guarded_dpof_eod_digest(context):
            if not load_dpof_config().eod_digest_enabled:
                return
            _now_et = _dpof_eod_now()
            # Weekdays after the 16:00 close only (the schedule already says
            # so; this keeps a catch-up/manual run from sending mid-session).
            if _now_et.weekday() >= 5 or _now_et.hour < 16:
                logger.info("DPOF EOD digest: not a weekday after the close, skipping")
                return
            try:
                from agent.integrations.market_events import MARKET_HOLIDAYS
                _today = _now_et.date()
                if _today in MARKET_HOLIDAYS:
                    logger.info("DPOF EOD digest: market holiday, skipping")
                    return
            except Exception as e:
                logger.debug(f"DPOF EOD digest: holiday check unavailable ({e}); proceeding")
            if _daily_task_already_ran("dpof_eod_digest"):
                logger.info("dpof_eod_digest already ran today, skipping")
                return
            text = await asyncio.get_running_loop().run_in_executor(None, eod_digest_text)
            for cid in _get_chat_ids():
                for chunk in split_for_telegram(text):
                    await _safe_send(context.bot, cid, text=chunk)
            await asyncio.get_running_loop().run_in_executor(None, _mark_daily_task_ran, "dpof_eod_digest")

        # run_daily with a tz-aware time: the cron trigger follows ET across
        # DST changes (an 86400 s repeat would drift to 15:10 after Nov 1).
        # PTB v20+ numbers days 0 = Sunday ... 6 = Saturday.
        from datetime import time as _dtime

        _ny_dpof = ZoneInfo("America/New_York")
        jq.run_daily(
            _with_timeout(_guarded_dpof_eod_digest, 60, "dpof_eod_digest"),
            time=_dtime(16, 10, tzinfo=_ny_dpof),
            days=(1, 2, 3, 4, 5),
            name="dpof_eod_digest",
        )
        # Catch-up: a restart between 16:10 and 18:10 on a weekday runs it soon
        # (_daily_task_already_ran prevents a second send).
        _now_dpof = _dpof_eod_now()
        if _now_dpof.weekday() < 5 and _dtime(16, 10) <= _now_dpof.time() < _dtime(18, 10):
            jq.run_once(
                _with_timeout(_guarded_dpof_eod_digest, 60, "dpof_eod_digest"),
                when=90,
                name="dpof_eod_digest_catchup",
            )
        logger.info("DPOF EOD digest registered (weekdays 16:10 ET)")
    except Exception as e:
        logger.error(f"Failed to register DPOF EOD digest: {e}", exc_info=True)

    # News update (every 30 min, 8:15 AM–8:15 PM ET, every day)
    try:
        from agent.integrations.news_update import run_news_update

        _ny = ZoneInfo("America/New_York")
        _now = datetime.now(_ny)
        # Calculate first run to next :15 or :45 mark
        if _now.minute < 15:
            _next_news = _now.replace(minute=15, second=0, microsecond=0)
        elif _now.minute < 45:
            _next_news = _now.replace(minute=45, second=0, microsecond=0)
        else:
            _next_news = (_now + timedelta(hours=1)).replace(minute=15, second=0, microsecond=0)
        _first_news = (_next_news - _now).total_seconds()
        if _first_news <= 0:
            _first_news = 1800
        # Skip catch-up outside operational hours (8:15 AM - 8:15 PM ET)
        if _now.hour < 8 or (_now.hour == 8 and _now.minute < 15) or _now.hour >= 20 or (_now.hour == 20 and _now.minute >= 15):
            _first_news = max(_first_news, 1800)

        jq.run_repeating(
            _with_timeout(run_news_update, 120, "news_update"),  # S233
            interval=1800,
            first=_first_news,
            name="news_update",
        )
        logger.info(f"News update registered (first run in {_first_news / 60:.0f}min)")
    except Exception as e:
        logger.error(f"Failed to register news update: {e}", exc_info=True)

    # Economic calendar refresh (daily 6:00 AM ET)
    try:
        from agent.integrations.econ_calendar import run_calendar_refresh as _real_calendar

        # S185 W2-3: wrap with dedup + mtime gate for staleness visibility
        async def _guarded_calendar(context):
            if _daily_task_already_ran("calendar_refresh"):
                logger.info("calendar_refresh already ran today, skipping")
                return
            _baseline = _capture_mtime(_STATE_FILE_FOR_TASK["calendar_refresh"])
            await _real_calendar(context)
            _mark_daily_task_if_state_progressed(
                "calendar_refresh", _STATE_FILE_FOR_TASK["calendar_refresh"], _baseline,
                bot=getattr(context, "bot", None),
            )

        _ny = ZoneInfo("America/New_York")
        _now = datetime.now(_ny)
        _target_cal = _now.replace(hour=6, minute=0, second=0, microsecond=0)
        if _now >= _target_cal:
            _target_cal += timedelta(days=1)
        _first_cal = (_target_cal - _now).total_seconds()

        # Catch-up: if past 6 AM but before 9 AM, run in 60s
        _catchup_cal_start = _now.replace(hour=6, minute=0, second=0, microsecond=0)
        _catchup_cal_end = _now.replace(hour=9, minute=0, second=0, microsecond=0)
        if _now.weekday() < 5 and _now >= _catchup_cal_start and _now < _catchup_cal_end:
            _first_cal = 60

        jq.run_repeating(
            _guarded_calendar,
            interval=86400,
            first=_first_cal,
            name="calendar_refresh",
        )
        logger.info(f"Economic calendar refresh registered (first run in {_first_cal / 3600:.1f}h)")
    except Exception as e:
        logger.error(f"Failed to register calendar refresh: {e}", exc_info=True)

    # News intelligence — event-triggered search (every 5 min)
    try:
        from agent.integrations.news_intel import run_event_search

        jq.run_repeating(
            run_event_search,
            interval=300,
            first=120,  # Start 2 min after boot
            name="intel_event_search",
        )
        logger.info("News intel event search registered (300s interval)")
    except Exception as e:
        logger.error(f"Failed to register intel event search: {e}", exc_info=True)

    # News intelligence — topic monitor (every 30 min, 8AM-8PM ET)
    try:
        from agent.integrations.news_intel import run_topic_monitor

        jq.run_repeating(
            run_topic_monitor,
            interval=1800,
            first=300,  # Start 5 min after boot
            name="intel_topic_monitor",
        )
        logger.info("News intel topic monitor registered (1800s interval)")
    except Exception as e:
        logger.error(f"Failed to register intel topic monitor: {e}", exc_info=True)

    # Macro research pipeline (daily 7:00 AM ET weekdays)
    try:
        from agent.integrations.macro_research import run_macro_research as _real_macro

        async def _guarded_macro(context):
            if _daily_task_already_ran("macro_research"):
                logger.info("macro_research already ran today, skipping")
                return
            _baseline = _capture_mtime(_STATE_FILE_FOR_TASK["macro_research"])
            await _real_macro(context)
            _mark_daily_task_if_state_progressed(
                "macro_research", _STATE_FILE_FOR_TASK["macro_research"], _baseline,
                bot=getattr(context, "bot", None),
            )

        _ny = ZoneInfo("America/New_York")
        _now = datetime.now(_ny)
        _target_mr = _now.replace(hour=7, minute=0, second=0, microsecond=0)
        if _now >= _target_mr:
            _target_mr += timedelta(days=1)
        while _target_mr.weekday() >= 5:
            _target_mr += timedelta(days=1)
        _first_mr = (_target_mr - _now).total_seconds()

        # Catch-up: if past 7 AM but before 10 AM, run in 60s
        _catchup_mr_start = _now.replace(hour=7, minute=0, second=0, microsecond=0)
        _catchup_mr_end = _now.replace(hour=10, minute=0, second=0, microsecond=0)
        if _now.weekday() < 5 and _now >= _catchup_mr_start and _now < _catchup_mr_end:
            _first_mr = 60

        jq.run_repeating(
            _guarded_macro,
            interval=86400,
            first=_first_mr,
            name="macro_research",
        )
        logger.info(f"Macro research pipeline registered (first run in {_first_mr / 3600:.1f}h)")
    except Exception as e:
        logger.error(f"Failed to register macro research: {e}", exc_info=True)

    # Fed communication analysis (daily 7:30 AM ET weekdays)
    try:
        from agent.integrations.fed_analysis import run_fed_analysis as _real_fed

        async def _guarded_fed(context):
            if _daily_task_already_ran("fed_analysis"):
                logger.info("fed_analysis already ran today, skipping")
                return
            _baseline = _capture_mtime(_STATE_FILE_FOR_TASK["fed_analysis"])
            await _real_fed(context)
            _mark_daily_task_if_state_progressed(
                "fed_analysis", _STATE_FILE_FOR_TASK["fed_analysis"], _baseline,
                bot=getattr(context, "bot", None),
            )

        _ny = ZoneInfo("America/New_York")
        _now = datetime.now(_ny)
        _target_fa = _now.replace(hour=7, minute=30, second=0, microsecond=0)
        if _now >= _target_fa:
            _target_fa += timedelta(days=1)
        while _target_fa.weekday() >= 5:
            _target_fa += timedelta(days=1)
        _first_fa = (_target_fa - _now).total_seconds()

        # Catch-up: past 7:30 AM but before 10 AM on weekday
        _catchup_fa_start = _now.replace(hour=7, minute=30, second=0, microsecond=0)
        _catchup_fa_end = _now.replace(hour=10, minute=0, second=0, microsecond=0)
        if _now.weekday() < 5 and _now >= _catchup_fa_start and _now < _catchup_fa_end:
            _first_fa = 90  # 90s — after macro_research catch-up (60s)

        jq.run_repeating(
            _guarded_fed,
            interval=86400,
            first=_first_fa,
            name="fed_analysis",
        )
        logger.info(f"Fed analysis registered (first run in {_first_fa / 3600:.1f}h)")
    except Exception as e:
        logger.error(f"Failed to register fed analysis: {e}", exc_info=True)

    # Earnings season intelligence (daily 7:15 AM ET weekdays)
    try:
        from agent.integrations.earnings_intel import run_earnings_intel as _real_earnings

        async def _guarded_earnings(context):
            if _daily_task_already_ran("earnings_intel"):
                logger.info("earnings_intel already ran today, skipping")
                return
            _baseline = _capture_mtime(_STATE_FILE_FOR_TASK["earnings_intel"])
            await _real_earnings(context)
            _mark_daily_task_if_state_progressed(
                "earnings_intel", _STATE_FILE_FOR_TASK["earnings_intel"], _baseline,
                bot=getattr(context, "bot", None),
            )

        _ny = ZoneInfo("America/New_York")
        _now = datetime.now(_ny)
        _target_ei = _now.replace(hour=7, minute=15, second=0, microsecond=0)
        if _now >= _target_ei:
            _target_ei += timedelta(days=1)
        while _target_ei.weekday() >= 5:
            _target_ei += timedelta(days=1)
        _first_ei = (_target_ei - _now).total_seconds()

        # Catch-up: past 7:15 AM but before 10 AM on weekday
        _catchup_ei_start = _now.replace(hour=7, minute=15, second=0, microsecond=0)
        _catchup_ei_end = _now.replace(hour=10, minute=0, second=0, microsecond=0)
        if _now.weekday() < 5 and _now >= _catchup_ei_start and _now < _catchup_ei_end:
            _first_ei = 120  # 120s — after fed_analysis catch-up (90s)

        jq.run_repeating(
            _guarded_earnings,
            interval=86400,
            first=_first_ei,
            name="earnings_intel",
        )
        logger.info(f"Earnings intelligence registered (first run in {_first_ei / 3600:.1f}h)")
    except Exception as e:
        logger.error(f"Failed to register earnings intelligence: {e}", exc_info=True)

    # Commodity supply chain review (daily 6:30 AM ET, weekdays — S122 4D)
    try:
        from agent.integrations.commodity_research import run_weekly_commodity_review as _real_commodity

        async def _guarded_commodity(context):
            await _run_commodity_review_guarded(context, _real_commodity)

        _ny = ZoneInfo("America/New_York")
        _now = datetime.now(_ny)
        # S122 4D: Target next weekday 6:30 AM (was Monday-only)
        _target_cr = _now.replace(hour=6, minute=30, second=0, microsecond=0)
        if _now >= _target_cr:
            _target_cr += timedelta(days=1)
        # Skip weekends
        while _target_cr.weekday() >= 5:
            _target_cr += timedelta(days=1)
        _first_cr = (_target_cr - _now).total_seconds()

        # Catch-up: if it's a weekday, past 6:30 AM but before 10 AM
        if _now.weekday() < 5 and (_now.hour > 6 or (_now.hour == 6 and _now.minute >= 30)) and _now.hour < 10:
            _first_cr = 150  # 2.5 min

        jq.run_repeating(
            _guarded_commodity,
            interval=86400,  # S122 4D: Daily (was weekly) — _already_analyzed dedup prevents re-work
            first=_first_cr,
            name="commodity_review",
        )
        logger.info(f"Commodity review registered (first run in {_first_cr / 3600:.1f}h)")
    except Exception as e:
        logger.error(f"Failed to register commodity review: {e}", exc_info=True)

    # Bond auction intelligence (daily 8:00 AM ET, Tue-Thu)
    try:
        from agent.integrations.bond_intel import run_bond_intel as _real_bond

        async def _guarded_bond(context):
            if _daily_task_already_ran("bond_intel"):
                logger.info("bond_intel already ran today, skipping")
                return
            _baseline = _capture_mtime(_STATE_FILE_FOR_TASK["bond_intel"])
            await _real_bond(context)
            _mark_daily_task_if_state_progressed(
                "bond_intel", _STATE_FILE_FOR_TASK["bond_intel"], _baseline,
                bot=getattr(context, "bot", None),
            )

        _ny = ZoneInfo("America/New_York")
        _now = datetime.now(_ny)
        _target_bi = _now.replace(hour=8, minute=0, second=0, microsecond=0)
        if _now >= _target_bi:
            _target_bi += timedelta(days=1)
        # Skip to next Tue-Thu
        while _target_bi.weekday() not in (1, 2, 3):  # Tue=1, Wed=2, Thu=3
            _target_bi += timedelta(days=1)
        _first_bi = (_target_bi - _now).total_seconds()

        # Catch-up: if it's Tue-Thu, past 8 AM but before 10 AM
        if _now.weekday() in (1, 2, 3) and _now.hour >= 8 and _now.hour < 10:
            _first_bi = 180  # 3 min

        jq.run_repeating(
            _guarded_bond,
            interval=86400,
            first=_first_bi,
            name="bond_intel",
        )
        logger.info(f"Bond auction intelligence registered (first run in {_first_bi / 3600:.1f}h)")
    except Exception as e:
        logger.error(f"Failed to register bond intel: {e}", exc_info=True)

    # Brier score prediction resolution (4:30 PM ET weekdays — Feature 7). Operates
    # on Hydrinia's state/prediction_log.json, which CLIO never writes.
    if _bridge_on:
        try:
            from agent.integrations.brier_resolver import run_brier_resolution

            _ny = ZoneInfo("America/New_York")
            _now = datetime.now(_ny)
            _target_b = _now.replace(hour=16, minute=30, second=0, microsecond=0)
            if _now >= _target_b:
                _target_b += timedelta(days=1)
            # Skip weekends
            while _target_b.weekday() >= 5:
                _target_b += timedelta(days=1)
            _first_delay_b = (_target_b - _now).total_seconds()

            jq.run_repeating(
                run_brier_resolution,
                interval=86400,  # Daily
                first=_first_delay_b,
                name="brier_resolution",
            )
            logger.info(f"Brier resolution check registered (first run in {_first_delay_b / 3600:.1f}h)")
        except Exception as e:
            logger.error(f"Failed to register Brier resolution: {e}", exc_info=True)
    else:
        _skipped_hydrinia_jobs.append("brier_resolution")

    # Bridge file cleanup (daily — removes files > 7 days from moltbot_knowledge subdirs).
    # Stays registered even with the bridge off: _cleanup_bridge_files() also prunes
    # CLIO-native trade idea / SEAL impl files and runs the daily WAL checkpoint, which
    # must keep running. It internally gates the moltbot_knowledge/-specific cleanup on
    # bridge_enabled().
    try:
        jq.run_repeating(
            _cleanup_bridge_files,
            interval=86400,
            first=3600,  # 1 hour after startup
            name="bridge_cleanup",
        )
        logger.info("Bridge file cleanup registered (daily, first run in 1h)")
    except Exception as e:
        logger.error(f"Failed to register bridge file cleanup: {e}", exc_info=True)

    # Trade idea feedback loop (every 4h — polls Hydrinia debriefs, feeds outcomes to memory)
    if _bridge_on:
        try:
            from agent.integrations.trade_idea_feedback import process_trade_idea_feedback
            jq.run_repeating(
                process_trade_idea_feedback,
                interval=14400,  # 4 hours
                first=600,  # 10 min after startup
                name="trade_idea_feedback",
            )
            logger.info("Trade idea feedback registered (4h interval, first run in 10min)")
        except Exception as e:
            logger.error(f"Failed to register trade idea feedback: {e}", exc_info=True)
    else:
        _skipped_hydrinia_jobs.append("trade_idea_feedback")

    # Hot context refresh — keeps Hydrinia mirror fresh even without user interaction (Fix S103-0E)
    try:
        async def _hot_context_refresh(context):
            try:
                from agent.memory.memory_engine import get_memory_engine
                me = get_memory_engine()
                # S233-V2: export_hot_context() flushes 3 memory tiers with flock —
                # must not run on event loop thread (S163-B rule).
                _loop = asyncio.get_running_loop()
                await _loop.run_in_executor(None, me.export_hot_context)
                # SEAL Memory Integration (Phase 5) — sync reward signals into memory tiers
                try:
                    _seal_stats = await _loop.run_in_executor(None, me.integrate_seal_signals)
                    if any(_seal_stats.values()):
                        logger.info(f"SEAL memory integration: {_seal_stats}")
                except Exception as e:
                    logger.debug(f"SEAL memory integration skipped: {e}")
                logger.debug("Hot context refresh completed")
            except Exception as e:
                logger.debug(f"Hot context refresh failed: {e}")
        jq.run_repeating(
            _hot_context_refresh,
            interval=7200,  # every 2 hours
            first=600,  # 10 min after startup
            name="hot_context_refresh",
        )
        logger.info("Hot context refresh registered (2h interval)")
    except Exception as e:
        logger.error(f"Failed to register hot context refresh: {e}", exc_info=True)

    # Daily trait reflection — 2:00 PM ET every day (personality self-authorship)
    try:
        from agent.integrations.daily_trait_reflection import run_daily_trait_reflection as _real_trait_reflection

        async def _guarded_trait_reflection(context):
            if _daily_task_already_ran("trait_reflection"):
                logger.debug("Trait reflection already ran today, skipping")
                return
            try:
                await _real_trait_reflection(context)
                await asyncio.get_running_loop().run_in_executor(None, _mark_daily_task_ran, "trait_reflection")
            except Exception as e:
                logger.error(f"Guarded trait reflection failed: {e}", exc_info=True)

        _ny = ZoneInfo("America/New_York")
        _now = datetime.now(_ny)
        _target = _now.replace(hour=14, minute=0, second=0, microsecond=0)
        if _now >= _target:
            _target += timedelta(days=1)
        _first_delay = (_target - _now).total_seconds()

        jq.run_repeating(
            _guarded_trait_reflection,
            interval=86400,  # Daily
            first=_first_delay,
            name="daily_trait_reflection",
        )
        logger.info(f"Daily trait reflection registered (first run in {_first_delay / 3600:.1f}h)")
        # Catch-up: if bot starts between 2:00-3:00 PM ET, run in 240s
        _trait_target = _now.replace(hour=14, minute=0, second=0, microsecond=0)
        _trait_catchup_end = _now.replace(hour=15, minute=0, second=0, microsecond=0)
        if _trait_target <= _now < _trait_catchup_end:
            jq.run_once(_guarded_trait_reflection, when=240)
            logger.info("Daily trait reflection catch-up scheduled in 240s")
    except Exception as e:
        logger.error(f"Failed to register daily trait reflection: {e}", exc_info=True)

    # Weekly knowledge archival — Sunday 3:00 AM ET
    try:
        async def _run_knowledge_archival(context):
            try:
                from agent.tools.knowledge_tools import run_knowledge_archival
                stats = await asyncio.get_running_loop().run_in_executor(None, run_knowledge_archival)
                if stats.get("archived", 0) > 0:
                    logger.info(f"Knowledge archival: {stats['archived']} entries archived, {stats['active_remaining']} remain")
                    for _cid in _get_chat_ids():
                        await _safe_send(context.bot, _cid, text=f"Knowledge archival: {stats['archived']} entries archived, {stats['active_remaining']} remain active")
            except Exception as e:
                logger.debug(f"Knowledge archival failed: {e}")
            # P0-5: SEAL-impl + code-sentinel pruning does synchronous open()+flock(LOCK_EX),
            # full json.load, per-entry os.path.exists() and a full index rewrite. Offload it
            # off the event loop (weekly cadence). Both prune blocks share _proj_root/_gl_seal/
            # _cutoff and run sequentially with no intervening await, so they stay together in
            # the helper to preserve the original scoping and control flow.
            def _prune_seal_impls():
                # S151-B W1-3: Prune old SEAL impl files (>14 days) + clean orphaned index entries
                try:
                    import glob as _gl_seal
                    import fcntl as _fcntl_seal
                    _proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    _cutoff = time.time() - (14 * 86400)
                    _removed = 0
                    # S160-6: Prune from all CLIO trading subdirs + Hydrinia mirror
                    _seal_dirs = [
                        os.path.join(_proj_root, "data", "knowledge", "trading", "market_analysis"),
                        os.path.join(_proj_root, "data", "knowledge", "trading", "strategies"),
                        "/home/agent/tradingbot/moltbot_knowledge/market_analysis",
                        "/home/agent/tradingbot/moltbot_knowledge/strategies",
                    ]
                    for _sd in _seal_dirs:
                        if not os.path.isdir(_sd):
                            continue
                        for _sp in _gl_seal.glob(os.path.join(_sd, "seal_impl_*.json")):
                            try:
                                if os.path.getmtime(_sp) < _cutoff:
                                    os.remove(_sp)
                                    _removed += 1
                            except OSError:
                                pass
                    if _removed:
                        logger.info(f"Pruned {_removed} SEAL impl files older than 14 days")
                    # Also clean orphaned SEAL impl entries from knowledge_index.json
                    _idx_path = os.path.join(_proj_root, "data", "knowledge", "knowledge_index.json")
                    _lock_fd = None
                    try:
                        _lock_fd = open(_idx_path + ".lock", "a")
                        _fcntl_seal.flock(_lock_fd, _fcntl_seal.LOCK_EX)
                        with open(_idx_path, "r") as _if:
                            _idx_data = json.load(_if)
                        _before = len(_idx_data.get("entries", []))
                        _idx_data["entries"] = [
                            e for e in _idx_data.get("entries", [])
                            if "seal_impl" not in e.get("file_path", "")
                            or os.path.exists(e.get("file_path", ""))
                        ]
                        _cleaned = _before - len(_idx_data["entries"])
                        if _cleaned > 0:
                            _tmp_idx = _idx_path + ".tmp"
                            with open(_tmp_idx, "w") as _of:
                                json.dump(_idx_data, _of)
                            os.replace(_tmp_idx, _idx_path)
                            logger.info(f"Cleaned {_cleaned} orphaned SEAL impl entries from knowledge index")
                    finally:
                        if _lock_fd:
                            try:
                                _fcntl_seal.flock(_lock_fd, _fcntl_seal.LOCK_UN)
                                _lock_fd.close()
                            except Exception as e:
                                logger.debug(f"Lock cleanup: {e}")
                except Exception as _se:
                    logger.debug(f"SEAL impl pruning failed: {_se}")

                # S162: Prune old code sentinel scan files (>14 days, keep latest 10)
                try:
                    _sentinel_dir = os.path.join(_proj_root, "data", "knowledge", "system_lessons")
                    if os.path.isdir(_sentinel_dir):
                        _sentinel_files = []
                        for _sf in _gl_seal.glob(os.path.join(_sentinel_dir, "*.json")):
                            _bn = os.path.basename(_sf)
                            if _bn == "code_sentinel_state.json":
                                continue  # Never prune the state file
                            try:
                                _sentinel_files.append((os.path.getmtime(_sf), _sf))
                            except OSError:
                                pass
                        # Sort newest first, keep latest 10, delete rest if older than 14 days
                        _sentinel_files.sort(reverse=True)
                        _sentinel_removed = 0
                        for _si, (_smt, _sfp) in enumerate(_sentinel_files):
                            if _si >= 10 and _smt < _cutoff:
                                try:
                                    os.remove(_sfp)
                                    _sentinel_removed += 1
                                except OSError:
                                    pass
                        if _sentinel_removed:
                            logger.info(f"Pruned {_sentinel_removed} old code sentinel scan files")
                except Exception as _se2:
                    logger.debug(f"Code sentinel pruning failed: {_se2}")

            await asyncio.get_running_loop().run_in_executor(None, _prune_seal_impls)

        _ny = ZoneInfo("America/New_York")
        _now = datetime.now(_ny)
        # Next Sunday 3 AM ET
        _days_until_sunday = (6 - _now.weekday()) % 7
        if _days_until_sunday == 0 and _now.hour >= 3:
            _days_until_sunday = 7
        _arch_target = (_now + timedelta(days=_days_until_sunday)).replace(hour=3, minute=0, second=0, microsecond=0)
        _arch_first = (_arch_target - _now).total_seconds()

        jq.run_repeating(
            _run_knowledge_archival,
            interval=604800,  # Weekly
            first=_arch_first,
            name="knowledge_archival",
        )
        logger.info(f"Knowledge archival registered (weekly Sunday 3 AM, first in {_arch_first / 3600:.1f}h)")

        # Archival catch-up: if last archival was >7 days ago, schedule immediate run
        try:
            _arch_state_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "knowledge", "knowledge_index_archive.json")
            if os.path.exists(_arch_state_path):
                import json as _json_arch
                with open(_arch_state_path, 'r') as _af:
                    _arch_data = _json_arch.load(_af)
                _last_arch = _arch_data.get("last_archival", "")
                if _last_arch:
                    _last_dt = datetime.fromisoformat(_last_arch)
                    if _last_dt.tzinfo is None:
                        _last_dt = _last_dt.replace(tzinfo=_ny)
                    if _now - _last_dt > timedelta(days=7):
                        jq.run_once(_run_knowledge_archival, when=300, name="knowledge_archival_catchup")
                        logger.info("Knowledge archival overdue (>7 days), catch-up scheduled in 300s")
        except Exception as _arch_e:
            logger.debug(f"Archival catch-up check failed: {_arch_e}")
    except Exception as e:
        logger.error(f"Failed to register knowledge archival: {e}", exc_info=True)

    # SEAL Phase 6: Knowledge retention testing (daily QA verification)
    try:
        from agent.integrations.seal_retention_tester import (
            run_retention_test,
        )
        if os.environ.get("MOLTBOT_SEAL_RETENTION", "1") == "1":
            jq.run_repeating(
                run_retention_test,
                interval=RETENTION_INTERVAL,
                first=RETENTION_FIRST_RUN,
                name="seal_retention_test",
            )
            logger.info(f"SEAL retention tester registered ({RETENTION_INTERVAL}s interval)")
        else:
            logger.info("SEAL retention tester disabled via MOLTBOT_SEAL_RETENTION")
    except Exception as e:
        logger.error(f"Failed to register SEAL retention tester: {e}", exc_info=True)

    # Unusual Whales flow intelligence scan — every 15 min during market hours
    try:
        from agent.integrations.uw_flow_intel import run_uw_scan
        jq.run_repeating(
            run_uw_scan,
            interval=900,  # 15 minutes
            first=150,     # 2.5 min after startup (staggered from intel_event_search at T+120)
            name="uw_flow_scan",
        )
        logger.info("UW flow scan registered (15min interval)")
    except Exception as e:
        logger.error(f"Failed to register UW flow scan: {e}", exc_info=True)

    # S162 + S163-B: Event loop heartbeat — proves asyncio loop is alive.
    # Updates main.record_event_loop_beat() so the watchdog thread knows
    # the loop is healthy. If frozen, watchdog stops pinging systemd → auto-restart.
    try:
        async def _event_loop_heartbeat(context):
            try:
                # S233-V2: Fix __main__ module identity split. When main.py runs
                # directly, it's sys.modules['__main__'], NOT sys.modules['main'].
                # 'from main import ...' creates a SEPARATE module with its own
                # _last_event_loop_beat, so the watchdog reads a stale copy.
                import sys
                sys.modules['__main__'].record_event_loop_beat()
            except Exception as e:
                # Surface (not silent): a failed record means the watchdog beat wasn't
                # refreshed. Benign in test contexts (no __main__ record fn); in production
                # this should never throw, so a warning here is real signal.
                logger.warning("Event loop heartbeat: record_event_loop_beat() failed (continuing): %s", e)
            logger.info("Event loop heartbeat: alive")

        jq.run_repeating(
            _event_loop_heartbeat,
            interval=120,  # S163-B: 2 minutes (must be < WatchdogSec/2 = 150s)
            first=30,
            name="event_loop_heartbeat",
        )
    except Exception as e:
        logger.error(f"Failed to register event loop heartbeat: {e}", exc_info=True)

    # Real-time price streaming (Alpaca WebSocket — stocks + options)
    try:
        if config.alpaca_api_key:
            async def _start_price_stream(context):
                try:
                    from agent.integrations.price_stream import get_stream_manager
                    mgr = get_stream_manager()
                    await mgr.start()
                    logger.info("Price stream manager started")
                except Exception as e:
                    logger.error(f"Price stream startup failed: {e}")

            jq.run_once(
                _start_price_stream,
                when=30,  # 30s after startup — wait for Alpaca client init
                name="price_stream_start",
            )
            logger.info("Price stream scheduled to start in 30s")
        else:
            logger.info("No Alpaca API key — price stream not registered")
    except Exception as e:
        logger.error(f"Failed to register price stream: {e}", exc_info=True)

    # ============================================================
    # Code Sentinel — automated code health monitoring (every 4 hours)
    # Operation FORGE Phase 1
    # ============================================================
    try:
        from agent.integrations.code_sentinel import run_code_health_check, SENTINEL_ENABLED
        if SENTINEL_ENABLED():
            CODE_HEALTH_INTERVAL = 14400  # 4 hours
            CODE_HEALTH_FIRST_RUN = 600   # 10 min after startup
            jq.run_repeating(
                run_code_health_check,
                interval=CODE_HEALTH_INTERVAL,
                first=CODE_HEALTH_FIRST_RUN,
                name="code_health_check",
            )
            logger.info(f"Code Sentinel registered ({CODE_HEALTH_INTERVAL}s interval)")
        else:
            logger.info("Code Sentinel disabled via CLIO_CODE_SENTINEL_ENABLED")
    except Exception as e:
        logger.error(f"Failed to register Code Sentinel: {e}", exc_info=True)

    # ============================================================
    # CC-CLIO-FD-5: Daily FD Screener — discover value+quality stocks
    # ============================================================
    try:
        if os.environ.get("CLIO_FD_DAILY_SCREENER", "0") == "1":
            FD_SCREENER_INTERVAL = 86400       # 24 hours
            FD_SCREENER_FIRST_RUN = 1800       # 30 min after startup

            async def _fd_daily_screener(context):
                """Run daily fundamental screener to discover trade candidates."""
                if os.environ.get("CLIO_FD_DAILY_SCREENER", "0") != "1":
                    return
                try:
                    from agent.integrations.fd_client import get_fd_client
                    client = get_fd_client()

                    # Value + Quality screen (screener uses SHORT field names per BC#112)
                    filters = [
                        {"field": "market_cap", "operator": "gte", "value": 2_000_000_000},
                        {"field": "pe_ratio", "operator": "lte", "value": 25},
                        {"field": "pe_ratio", "operator": "gt", "value": 0},
                        {"field": "revenue_growth", "operator": "gte", "value": 0.05},
                        {"field": "return_on_equity", "operator": "gte", "value": 0.10},
                    ]

                    # screen_stocks is sync — run in executor to avoid blocking event loop
                    loop = asyncio.get_running_loop()
                    results = await loop.run_in_executor(
                        None, lambda: client.screen_stocks(filters, limit=15)
                    )
                    if not results or (isinstance(results, dict) and "_error" in results):
                        logger.info("CC-CLIO-FD-5: Daily screener returned no results")
                        return

                    result_list = results if isinstance(results, list) else []
                    tickers = [r.get("ticker", "") for r in result_list if r.get("ticker")]

                    logger.info("CC-CLIO-FD-5: Daily screener found %d candidates: %s",
                                len(tickers), ", ".join(tickers[:10]))

                    # Store results for learning engine / trade idea pipeline
                    screener_file = os.path.join(config.tradingbot_dir, "moltbot_knowledge",
                                                  "fd_screener_results.json")
                    _ny = ZoneInfo("America/New_York")
                    data = {
                        "timestamp": datetime.now(_ny).isoformat(),
                        "filters": filters,
                        "count": len(result_list),
                        "tickers": tickers,
                        "results": result_list[:15],
                    }
                    os.makedirs(os.path.dirname(screener_file), exist_ok=True)
                    tmp = screener_file + ".tmp"
                    with open(tmp, "w") as f:
                        json.dump(data, f, indent=2, default=str)
                    os.replace(tmp, screener_file)

                except Exception as e:
                    logger.warning("CC-CLIO-FD-5: Daily screener failed: %s", e)

            jq.run_repeating(_fd_daily_screener, interval=FD_SCREENER_INTERVAL,
                             first=FD_SCREENER_FIRST_RUN, name="fd_daily_screener")
            logger.info(f"FD Daily Screener registered ({FD_SCREENER_INTERVAL}s interval)")
        else:
            logger.info("FD Daily Screener disabled via CLIO_FD_DAILY_SCREENER")
    except Exception as e:
        logger.error(f"Failed to register FD Daily Screener: {e}", exc_info=True)

    # Daily analytics fun facts refresh — 8:05 PM ET every market day
    try:
        from agent.integrations.analytics_fun_facts import refresh_analytics_fun_facts

        async def _guarded_fun_facts(context):
            if _daily_task_already_ran("analytics_fun_facts"):
                logger.debug("Analytics fun facts already refreshed today, skipping")
                return
            # Only run on market days (Mon-Fri)
            _ny_now = datetime.now(ZoneInfo("America/New_York"))
            if _ny_now.weekday() >= 5:
                return
            try:
                _baseline = _capture_mtime(_STATE_FILE_FOR_TASK["analytics_fun_facts"])
                await refresh_analytics_fun_facts(context)
                _mark_daily_task_if_state_progressed(
                    "analytics_fun_facts",
                    _STATE_FILE_FOR_TASK["analytics_fun_facts"],
                    _baseline,
                    bot=getattr(context, "bot", None),
                )
            except Exception as e:
                logger.error(f"Analytics fun facts refresh failed: {e}", exc_info=True)

        _ny = ZoneInfo("America/New_York")
        _now = datetime.now(_ny)
        _target = _now.replace(hour=20, minute=5, second=0, microsecond=0)
        if _now >= _target:
            _target += timedelta(days=1)
        _first_delay = (_target - _now).total_seconds()

        jq.run_repeating(
            _guarded_fun_facts,
            interval=86400,
            first=_first_delay,
            name="analytics_fun_facts_refresh",
        )
        logger.info(f"Analytics fun facts refresh registered (first run in {_first_delay / 3600:.1f}h)")
    except Exception as e:
        logger.error(f"Failed to register analytics fun facts refresh: {e}", exc_info=True)

    # Athena bridge health check (Phase 1) — the vault is written by Hydrinia, which
    # is permanently off, so this only makes sense when the bridge is re-enabled.
    if not _bridge_on:
        _skipped_hydrinia_jobs.append("athena_health_check")
    else:
      try:
        async def _check_athena_health(context):
            if os.environ.get("CLIO_ATHENA_ENABLED", "0") != "1":
                return

            try:
                from agent.integrations.athena_bridge import get_athena_bridge
                bridge = get_athena_bridge()
                status = bridge.get_status()

                if not status.get("available"):
                    issues = ", ".join(status.get("issues", ["unknown"]))
                    msg = (
                        f"ATHENA ALERT: Bridge unavailable.\n"
                        f"Issues: {issues}\n"
                        f"Vault entries: {status.get('vault_entries', 0)}\n"
                        f"Vault age: {status.get('vault_age_hours', 'N/A')}h"
                    )
                    for cid in _get_chat_ids():
                        await _safe_send(context.bot, cid, text=msg)
                    logger.warning("Athena health check: %s", msg)
                else:
                    # Vault is available — check for STALENESS (S179 deferred → S184 followup).
                    # Existing get_status() reports staleness in 'issues' but only triggers the
                    # alert above when 'available'=False (which depends on vault_exists, not age).
                    # This branch fires when vault EXISTS but stopped being updated.
                    vault_age = status.get("vault_age_hours")
                    try:
                        stale_threshold = float(os.environ.get("CLIO_ATHENA_STALE_HOURS", "6"))
                    except (TypeError, ValueError):
                        stale_threshold = 6.0

                    is_stale = (
                        isinstance(vault_age, (int, float))
                        and vault_age > stale_threshold
                    )
                    if is_stale:
                        try:
                            now_epoch = time.time()
                            last_alert = _athena_stale_alert_dedup.get("vault_stale", 0)
                            # Dedup: skip if we alerted within the past stale_threshold hours
                            if (now_epoch - last_alert) >= (stale_threshold * 3600):
                                msg = (
                                    f"ATHENA STALE: Vault hasn't updated in {vault_age:.1f}h "
                                    f"(threshold {stale_threshold:.1f}h).\n"
                                    f"Vault entries: {status.get('vault_entries', 0)}\n"
                                    f"Likely cause: Hydrinia writer stopped or hung."
                                )
                                for cid in _get_chat_ids():
                                    await _safe_send(context.bot, cid, text=msg)
                                logger.warning("Athena stale alert: %s", msg)
                                _athena_stale_alert_dedup["vault_stale"] = now_epoch
                            else:
                                logger.debug(
                                    "Athena stale (vault_age=%.1fh) but suppressed by dedup "
                                    "(last alert %.1fh ago)",
                                    vault_age,
                                    (now_epoch - last_alert) / 3600,
                                )
                        except Exception as e:
                            logger.debug("Athena stale-alert fire-and-forget failed: %s", e)
                    else:
                        # Reset dedup if vault recovered (age dropped below threshold)
                        if "vault_stale" in _athena_stale_alert_dedup:
                            _athena_stale_alert_dedup.pop("vault_stale", None)
                            logger.info("Athena vault recovered (age=%s)", vault_age)
                        # Only log "OK" when actually fresh — avoids contradictory message
                        # right after a stale alert.
                        logger.info(
                            "Athena health OK. Cached: %d, Vault entries: %d, Age: %sh",
                            status.get("cached_forecasts", 0),
                            status.get("vault_entries", 0),
                            vault_age if vault_age is not None else "N/A",
                        )

            except Exception as e:
                logger.error("Athena health check failed: %s", e)

        jq.run_repeating(
            _check_athena_health,
            interval=ATHENA_HEALTH_INTERVAL,
            first=ATHENA_HEALTH_FIRST_RUN,
            name="athena_health_check",
        )
        logger.info("Athena health check registered (first run in %.1fmin)", ATHENA_HEALTH_FIRST_RUN / 60)
      except Exception as e:
        logger.error("Failed to register Athena health check: %s", e, exc_info=True)

    # ============================================================
    # E.1 social_intel incremental enrichment — post-ingest hook (11:15 UTC)
    # + 30-min sweep backstop (Atlas Team GO 2026-06-11; bundle commit)
    # ============================================================
    # Default ON: the directive's deliverable IS unattended daily operation.
    # Kill-switch: CLIO_ENRICH_INCREMENTAL_ENABLED=0 (+ restart).
    try:
        if os.environ.get("CLIO_ENRICH_INCREMENTAL_ENABLED", "1") == "1":
            from agent.integrations.social_intel_enrich import run_incremental as _enrich_run

            _ENRICH_DROP_ALERT_PCT = float(os.environ.get("CLIO_ENRICH_DROP_ALERT_PCT", "10"))
            _ENRICH_DROP_ALERT_MIN_N = 20      # drop-rate needs a denominator to mean anything
            _ENRICH_ALERT_WINDOW_S = 6 * 3600  # per-kind dedup window

            async def _enrich_alert(context, kind: str, msg: str) -> None:
                """Plain-text Telegram alert with per-kind 6h dedup. Pass-2 D6: the dedup stamp
                lands only AFTER >=1 successful delivery (a failed send must not mute the kind
                for 6h), per-chat failures are isolated, and failures log at WARNING — DEBUG is
                invisible at production log level (anti-pattern #27)."""
                try:
                    now_epoch = time.time()
                    if (now_epoch - _enrich_alert_dedup.get(kind, 0)) < _ENRICH_ALERT_WINDOW_S:
                        logger.debug("enrich alert '%s' suppressed by dedup", kind)
                        return
                    sent_any = False
                    for cid in _get_chat_ids():
                        try:
                            if await _safe_send(context.bot, cid, text=msg):
                                sent_any = True
                        except Exception as send_e:
                            logger.warning("enrich alert '%s' send to %s failed: %s",
                                           kind, cid, send_e)
                    if sent_any:
                        _enrich_alert_dedup[kind] = now_epoch
                    else:
                        logger.warning("enrich alert '%s' undelivered to any chat: %s",
                                       kind, msg[:120])
                except Exception as e:
                    logger.warning("enrich alert '%s' failed: %s", kind, e)

            async def _run_enrich_monitored(context, run_type: str) -> dict:
                """Run + the bundle's DROP-RATE MONITOR (plus credit/cap/budget/error alerts)."""
                stats = await _enrich_run(run_type=run_type)
                if not isinstance(stats, dict) or stats.get("status") in (
                        "skipped_concurrent", "skipped_inflight"):
                    return stats if isinstance(stats, dict) else {}
                processed = (stats.get("written_ok") or 0) + (stats.get("dropped") or 0)
                if processed >= _ENRICH_DROP_ALERT_MIN_N:
                    rate = 100.0 * (stats.get("dropped") or 0) / processed
                    if rate > _ENRICH_DROP_ALERT_PCT:
                        await _enrich_alert(
                            context, "drop_rate",
                            f"ENRICH DROP-RATE ALERT ({run_type}): {rate:.0f}% of "
                            f"{processed} processed tweets dropped "
                            f"(threshold {_ENRICH_DROP_ALERT_PCT:.0f}%). "
                            f"run_id={stats.get('run_id')} — check enrich_runs + "
                            f"tweet_signals status breakdown.")
                if stats.get("aborted") == "credit_exhausted":
                    await _enrich_alert(
                        context, "credit",
                        f"ENRICH ABORTED ({run_type}): Anthropic credit exhausted. "
                        f"Rows self-heal once credits are replenished. "
                        f"run_id={stats.get('run_id')}")
                if stats.get("capped"):
                    await _enrich_alert(
                        context, "cap",
                        f"ENRICH ({run_type}): daily spend cap reached "
                        f"(today ${stats.get('spent_today_usd', 0):.2f} + "
                        f"run ${stats.get('spend_usd', 0):.2f}); "
                        f"{stats.get('todo', 0)} window tweets deferred. "
                        f"run_id={stats.get('run_id')}")
                if stats.get("budget_exhausted"):
                    await _enrich_alert(
                        context, "budget",
                        f"ENRICH ({run_type}): 'social_enrich' call budget exhausted "
                        f"(BUDGET_SOCIAL_ENRICH). Remaining tweets deferred. "
                        f"run_id={stats.get('run_id')}")
                if stats.get("error"):
                    await _enrich_alert(
                        context, "error",
                        f"ENRICH ERROR ({run_type}): {str(stats.get('error'))[:300]} "
                        f"run_id={stats.get('run_id')}")
                return stats

            async def _guarded_enrich_hook(context):
                if _daily_task_already_ran("social_enrich_incremental"):
                    logger.info("social_enrich_incremental already ran today, skipping")
                    return
                stats = await _run_enrich_monitored(context, "incremental")
                if (isinstance(stats, dict)
                        and stats.get("status") not in ("skipped_concurrent", "skipped_inflight")
                        and not stats.get("error")):
                    await asyncio.get_running_loop().run_in_executor(None, _mark_daily_task_ran, "social_enrich_incremental")

            async def _enrich_sweep(context):
                # Pass-2 D4/D5: the sweep doubles as the SAME-DAY RETRY vehicle for the daily
                # incremental heartbeat. If today's 11:15 UTC hook never landed (lock collision
                # with an in-flight sweep, run error, or a restart at ANY later time — no
                # separate catch-up branch needed), the next sweep runs the guarded hook instead;
                # the daily marker + run locks make this idempotent and double-spend-safe.
                _now_u = datetime.now(timezone.utc)
                if (_now_u >= _now_u.replace(hour=11, minute=15, second=0, microsecond=0)
                        and not _daily_task_already_ran("social_enrich_incremental")):
                    await _guarded_enrich_hook(context)
                    return
                await _run_enrich_monitored(context, "sweep")

            # 11:15 UTC daily — pinned to UTC (NOT ET) because the upstream Apify ingest
            # pg_cron fires at 11:03 UTC year-round; an ET-pinned hook would drift an hour
            # across DST and fire before the ingest in winter.
            _now_utc = datetime.now(timezone.utc)
            _target_en = _now_utc.replace(hour=11, minute=15, second=0, microsecond=0)
            if _now_utc >= _target_en:
                _target_en += timedelta(days=1)
            _first_en = (_target_en - _now_utc).total_seconds()
            # No catch-up branch here (pass-2 D5): overwriting `first` would permanently
            # re-anchor run_repeating's daily fire to restart-time+90s (PTB anchors the interval
            # at the first fire). Missed-hook catch-up is the sweep's job (see _enrich_sweep),
            # which also covers restarts AFTER any fixed window.

            jq.run_repeating(
                _with_timeout(_guarded_enrich_hook, 2100, "social_enrich_incremental"),
                interval=86400,
                first=_first_en,
                name="social_enrich_incremental",
            )
            # Sweep backstop: catches infra-skipped rows (attempt cap permitting), late
            # ingests, a hook that died mid-run, and the missed daily heartbeat (delegation
            # above). Cheap no-op when the window is clean. Timeout matches the hook because
            # a delegated sweep can carry the full day-scale run.
            jq.run_repeating(
                _with_timeout(_enrich_sweep, 2100, "social_enrich_sweep"),
                interval=1800,
                first=1200,  # 20 min after startup — staggered away from the start-up burst
                name="social_enrich_sweep",
            )
            # E.2: daily story-clustering tick (marker-gated; no fixed-anchor catch-up —
            # the S202 pass-2 D5 lesson). Runs once per day, first tick past 11:45 UTC
            # AFTER the day's enrichment marker landed; later ticks no-op via marker.
            # Depends on enrichment, so it lives inside (and is killed by) the same gate;
            # its own additional gate: CLIO_STORIES_DAILY_ENABLED=0.
            if os.environ.get("CLIO_STORIES_DAILY_ENABLED", "1") == "1":
                from agent.integrations.social_intel_stories import run_stories as _stories_run

                async def _stories_tick(context):
                    _now_u = datetime.now(timezone.utc)
                    if _now_u < _now_u.replace(hour=11, minute=45, second=0, microsecond=0):
                        return
                    if _daily_task_already_ran("social_stories_daily"):
                        return
                    if not _daily_task_already_ran("social_enrich_incremental"):
                        logger.debug("stories tick: today's enrichment not yet landed; waiting")
                        return
                    stats = await _stories_run()
                    if not isinstance(stats, dict):
                        return
                    if stats.get("status") in ("skipped_concurrent", "skipped_inflight"):
                        return  # retried by the next 30-min tick
                    if stats.get("aborted") == "credit_exhausted":
                        await _enrich_alert(
                            context, "credit",
                            f"STORIES ABORTED: Anthropic credit exhausted. "
                            f"run_id={stats.get('run_id')}")
                        return  # no marker — retried after credits return
                    if stats.get("error"):
                        await _enrich_alert(
                            context, "stories_error",
                            f"STORIES ERROR: {str(stats.get('error'))[:300]} "
                            f"run_id={stats.get('run_id')}")
                        return  # no marker — retried next tick (alert deduped 6h)
                    # Pass-2 F6: cap/budget parity with the enrich lane — alert (distinct
                    # dedup kinds, so the enrich lane's stamps can't suppress these) but DO
                    # stamp the marker: same-day retry is futile against daily pools, so
                    # deferral-to-tomorrow is the correct semantics.
                    if stats.get("capped"):
                        await _enrich_alert(
                            context, "stories_cap",
                            f"STORIES: daily spend cap reached "
                            f"(today ${stats.get('spent_today_usd', 0):.2f} + "
                            f"run ${stats.get('spend_usd', 0):.2f}); "
                            f"{stats.get('label_deferred', 0)} label candidates deferred. "
                            f"run_id={stats.get('run_id')}")
                    if stats.get("budget_exhausted"):
                        await _enrich_alert(
                            context, "stories_budget",
                            f"STORIES: 'social_enrich' call budget exhausted "
                            f"(BUDGET_SOCIAL_ENRICH); {stats.get('label_deferred', 0)} label "
                            f"candidates deferred. run_id={stats.get('run_id')}")
                    await asyncio.get_running_loop().run_in_executor(None, _mark_daily_task_ran, "social_stories_daily")

                jq.run_repeating(
                    _with_timeout(_stories_tick, 2100, "social_stories_daily"),
                    interval=1800,
                    first=1500,  # permanent 300s stagger behind the enrich sweep (first=1200)
                    name="social_stories_daily",
                )
                logger.info("social_intel stories tick registered (daily past 11:45 UTC, "
                            "30-min marker-gated ticks)")
            else:
                logger.info("social_intel stories disabled via CLIO_STORIES_DAILY_ENABLED")

            logger.info(
                "social_intel enrichment registered (hook in %.1fh @ 11:15 UTC; sweep every "
                "30min, first in 20min)", _first_en / 3600)
        else:
            logger.info("social_intel enrichment disabled via CLIO_ENRICH_INCREMENTAL_ENABLED")
    except Exception as e:
        logger.error("Failed to register social_intel enrichment: %s", e, exc_info=True)

    # §6.D — Narrative-vs-Statistical Divergence Detector (THE CONSUMER; ADVISORY ONLY).
    # Flag-gated OFF by default: the operator flips CLIO_DIVERGENCE_ENABLED=1 + restart to go live
    # (sequencing: after the on-write-dedup restart and the Fri COT run lands). Hourly detector tick
    # (the persistence gate needs k consecutive holds before a verdict fires — anti-flap) + a daily
    # forward-repricing grader. The detector NEVER writes risk authority (advisory only).
    try:
        if os.environ.get("CLIO_DIVERGENCE_ENABLED", "0") == "1":
            from agent.integrations.social_intel_divergence import (
                run_divergence as _div_run, run_divergence_grader as _div_grade)

            async def _divergence_tick(context):
                res = await _div_run(context=context)
                if isinstance(res, dict) and res.get("emitted"):
                    logger.info("divergence advisory EMITTED: %s (conf %.2f)",
                                res.get("divergence_class"), res.get("confidence", 0))

            async def _divergence_grader_tick(context):
                if _daily_task_already_ran("clio_divergence_grader"):
                    return
                await _div_grade()
                await asyncio.get_running_loop().run_in_executor(None, _mark_daily_task_ran, "clio_divergence_grader")

            jq.run_repeating(
                _with_timeout(_divergence_tick, 180, "clio_divergence"),
                interval=3600, first=900, name="clio_divergence")
            jq.run_repeating(
                _with_timeout(_divergence_grader_tick, 300, "clio_divergence_grader"),
                interval=3600, first=2400, name="clio_divergence_grader")
            logger.info("§6.D divergence detector registered (hourly tick + daily grader; ADVISORY)")
        else:
            logger.info("§6.D divergence detector disabled via CLIO_DIVERGENCE_ENABLED")
    except Exception as e:
        logger.error("Failed to register §6.D divergence detector: %s", e, exc_info=True)

    # Hydrinia bridge summary — one INFO line naming every job skipped because
    # Hydrinia is permanently off (CLIO_HYDRINIA_BRIDGE_ENABLED unset/0).
    if _skipped_hydrinia_jobs:
        logger.info(
            "Hydrinia bridge disabled (CLIO_HYDRINIA_BRIDGE_ENABLED=0) — skipped %d "
            "Hydrinia-only job(s): %s",
            len(_skipped_hydrinia_jobs), ", ".join(_skipped_hydrinia_jobs),
        )
