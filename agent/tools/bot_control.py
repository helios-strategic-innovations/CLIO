"""
OpenClaw/Moltbot — Bot Control Tool Handlers

Start, stop, restart Hydrinia via process signals and file-based controls.

Hydrinia is permanently off (operator direction, 2026-09-24); handle_control_hydrinia
is gated on agent.hydrinia_bridge.bridge_enabled() (default False) so it can never
Popen a stale hydrinia.py back to life or SIGTERM a recycled PID from a stale lock
file. This is a second, independent gate on top of agent/tools/executor.py's
dispatch-level gate.
"""

import asyncio
import json
import logging
import os
import signal
import subprocess
import time
from pathlib import Path

from agent.security import log_security_event, sanitize_ingested_content

logger = logging.getLogger("openclaw.bot_control")

TRADINGBOT_DIR = "/home/agent/tradingbot"
STATE_DIR = os.path.join(TRADINGBOT_DIR, "state")
LOCK_FILE = os.path.join(STATE_DIR, "hydrinia.lock")
EMERGENCY_FILE = os.path.join(STATE_DIR, "EMERGENCY_STOP")
PYTHON_BIN = os.path.join(TRADINGBOT_DIR, "venv", "bin", "python3")
STDOUT_LOG = os.path.join(TRADINGBOT_DIR, "logs", "hydrinia_stdout.log")


def _get_pid() -> int | None:
    """Read PID from lock file. Returns None if not found or stale."""
    if not os.path.isfile(LOCK_FILE):
        return None
    try:
        with open(LOCK_FILE, "r") as f:
            data = json.loads(f.read())
        return data.get("pid")
    except Exception as e:
        logger.debug(f"Lock file read failed: {e}")
        return None


def _is_running(pid: int) -> bool:
    """Check if a process is alive."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


async def handle_control_hydrinia(p: dict) -> str:
    """Start, stop, or restart Hydrinia."""
    try:
        from agent.hydrinia_bridge import bridge_enabled
        _enabled = bridge_enabled()
    except Exception as e:
        logger.debug(f"Hydrinia bridge check failed, defaulting to disabled: {e}")
        _enabled = False
    if not _enabled:
        return (
            "RETIRED: Hydrinia is permanently off — control_hydrinia is unavailable. "
            "CLIO trades its own SPY/QQQ book via the DPOF engine now; use get_dpof_status instead."
        )
    action = p.get("action", "status")

    if action == "stop":
        return await _stop_hydrinia()
    elif action == "start":
        return await _start_hydrinia()
    elif action == "restart":
        stop_result = await _stop_hydrinia()
        if "not running" not in stop_result.lower() and "stopped" not in stop_result.lower():
            return f"Stop failed: {stop_result}"
        await asyncio.sleep(2)
        start_result = await _start_hydrinia()
        return f"{stop_result}\n{start_result}"
    elif action == "emergency_stop":
        return await _emergency_stop()
    elif action == "status":
        from agent.tools.hydrinia_tools import handle_hydrinia_status
        return await handle_hydrinia_status({})
    else:
        return f"Unknown action: {action}. Use: start, stop, restart, emergency_stop, status"


async def _stop_hydrinia() -> str:
    """Gracefully stop Hydrinia via SIGTERM."""
    pid = _get_pid()
    if pid is None:
        return "Hydrinia is not running (no lock file)."

    if not _is_running(pid):
        # Clean up stale lock
        try:
            os.unlink(LOCK_FILE)
        except OSError:
            pass
        return f"Hydrinia was not running (stale lock for PID {pid}, cleaned up)."

    log_security_event("hydrinia_stop", pid=pid, method="SIGTERM")
    logger.info(f"Sending SIGTERM to Hydrinia PID {pid}")

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return f"Process {pid} already exited."

    # Wait up to 30 seconds for graceful shutdown
    for i in range(30):
        await asyncio.sleep(1)
        if not _is_running(pid):
            logger.info(f"Hydrinia PID {pid} stopped after {i + 1}s")
            return f"Hydrinia stopped (PID {pid}, took {i + 1}s)."

    return f"SIGTERM sent to PID {pid} but process still running after 30s. Consider emergency_stop."


async def _start_hydrinia() -> str:
    """Start Hydrinia as a background process."""
    pid = _get_pid()
    if pid and _is_running(pid):
        return f"Hydrinia is already running (PID {pid})."

    # Clean stale lock
    if os.path.isfile(LOCK_FILE):
        try:
            os.unlink(LOCK_FILE)
        except OSError:
            pass

    # Remove emergency stop if present
    if os.path.isfile(EMERGENCY_FILE):
        try:
            os.unlink(EMERGENCY_FILE)
            logger.info("Removed EMERGENCY_STOP file before starting")
        except OSError:
            pass

    if not os.path.isfile(PYTHON_BIN):
        return f"ERROR: Python not found at {PYTHON_BIN}"

    hydrinia_path = os.path.join(TRADINGBOT_DIR, "hydrinia.py")
    if not os.path.isfile(hydrinia_path):
        return f"ERROR: hydrinia.py not found at {hydrinia_path}"

    log_security_event("hydrinia_start", method="subprocess")
    logger.info("Starting Hydrinia...")

    try:
        os.makedirs(os.path.dirname(STDOUT_LOG), exist_ok=True)
        with open(STDOUT_LOG, "a") as stdout_f:
            proc = subprocess.Popen(
                [PYTHON_BIN, "hydrinia.py"],
                cwd=TRADINGBOT_DIR,
                stdout=stdout_f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        # stdout_f closed by context manager; child keeps its own fd copy

        # Wait briefly to check startup
        await asyncio.sleep(3)

        if proc.poll() is None:
            logger.info(f"Hydrinia started with PID {proc.pid}")
            return f"Hydrinia started (PID {proc.pid})."
        else:
            return f"Hydrinia failed to start (exit code {proc.returncode}). Check {STDOUT_LOG}"
    except Exception as e:
        return f"ERROR starting Hydrinia: {e}"


async def _emergency_stop() -> str:
    """Create EMERGENCY_STOP file to halt Hydrinia on next check."""
    log_security_event("hydrinia_emergency_stop", method="file")
    logger.warning("Creating EMERGENCY_STOP file")

    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        tmp_file = EMERGENCY_FILE + ".tmp"
        with open(tmp_file, "w") as f:
            json.dump({
                "timestamp": time.time(),
                "reason": "Remote emergency stop via Telegram",
                "created": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            }, f)
        os.replace(tmp_file, EMERGENCY_FILE)
        return "EMERGENCY_STOP file created. Hydrinia will halt on next check cycle."
    except Exception as e:
        return f"ERROR creating emergency stop: {e}"


async def handle_get_bot_processes(p: dict) -> str:
    """List running bot-related processes."""
    keywords = ["openclaw", "moltbot", "streamlit"]
    try:
        from agent.hydrinia_bridge import bridge_enabled
        if bridge_enabled():
            keywords.append("hydrinia")
    except Exception as e:
        logger.debug(f"Hydrinia bridge check failed, excluding hydrinia from process filter: {e}")
    try:
        proc = await asyncio.create_subprocess_exec(
            "ps", "aux",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        output = stdout_bytes.decode(errors="replace") if stdout_bytes else ""
        lines = []
        for line in output.splitlines():
            if any(kw in line for kw in keywords):
                lines.append(line)

        if not lines:
            return "No bot processes found."

        raw = f"Bot Processes ({len(lines)}):\n" + "\n".join(lines)
        return sanitize_ingested_content(raw, source="ps_output")
    except asyncio.TimeoutError:
        return "ERROR: Process listing timed out after 10s"
    except Exception as e:
        return f"ERROR: {e}"
