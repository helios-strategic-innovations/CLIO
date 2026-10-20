"""
OpenClaw/Moltbot — Main Entry Point
"""

import atexit
import faulthandler
import fcntl
import logging
import os
import signal
import socket
import sys
import threading
import time as _time

# S163-B: Enable faulthandler — SIGUSR1 dumps all thread tracebacks to stderr (journalctl)
faulthandler.enable()
faulthandler.register(signal.SIGUSR1, all_threads=True)

from dotenv import load_dotenv
load_dotenv()

from agent.logging_config import setup_logging
setup_logging()

logger = logging.getLogger("openclaw.main")

PID_FILE = "/home/agent/openclaw/moltbot.pid"
_pid_fd = None


def _acquire_pid_lock() -> None:
    """Acquire an exclusive PID file lock. Exits if another instance is running."""
    global _pid_fd
    try:
        _pid_fd = open(PID_FILE, "w")
        fcntl.flock(_pid_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _pid_fd.write(str(os.getpid()))
        _pid_fd.flush()
        atexit.register(_release_pid_lock)
        logger.info(f"PID lock acquired: {os.getpid()}")
    except BlockingIOError:
        # Lock held by another process — check if it's still alive
        try:
            with open(PID_FILE, "r") as f:
                existing_pid = int(f.read().strip())
            os.kill(existing_pid, 0)  # Check if process exists
            logger.error(
                f"Another Moltbot instance is running (PID {existing_pid}). Exiting."
            )
            sys.exit(1)
        except (ValueError, ProcessLookupError, FileNotFoundError):
            # Stale PID file — previous process is dead, take over
            logger.warning("Stale PID file found, taking over lock")
            _pid_fd.close()
            _pid_fd = open(PID_FILE, "w")
            fcntl.flock(_pid_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _pid_fd.write(str(os.getpid()))
            _pid_fd.flush()
            atexit.register(_release_pid_lock)


def _release_pid_lock() -> None:
    """Release PID file lock and remove the file."""
    global _pid_fd
    try:
        if _pid_fd and not _pid_fd.closed:
            fcntl.flock(_pid_fd, fcntl.LOCK_UN)
            _pid_fd.close()
        if os.path.isfile(PID_FILE):
            os.unlink(PID_FILE)
    except OSError:
        pass


def _sd_notify(msg: str) -> None:
    """Send a sd_notify message to systemd (no-op if NOTIFY_SOCKET not set)."""
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        if addr.startswith("@"):
            sock.connect("\0" + addr[1:])
        else:
            sock.connect(addr)
        sock.sendall(msg.encode())
        sock.close()
    except Exception:
        pass  # S233 ME-3: Best-effort — sd_notify may fail outside systemd (container, dev)


# S162/S163-B: Event loop liveness gate for watchdog.
# Updated by scheduled_tasks._event_loop_heartbeat (runs on asyncio loop).
# Watchdog thread only pings systemd if this is fresh — a frozen event loop
# means no heartbeat update, so watchdog stops pinging, systemd kills us.
_last_event_loop_beat: float = _time.time()  # Init to now so startup grace period works
_beat_lock = threading.Lock()


def record_event_loop_beat() -> None:
    """Called from asyncio event loop heartbeat to prove the loop is alive."""
    global _last_event_loop_beat
    with _beat_lock:
        _last_event_loop_beat = _time.time()


def _start_watchdog_thread() -> None:
    """Send WATCHDOG=1 every 120s, but ONLY if event loop is alive.

    WatchdogSec=300 in service file. We ping every 120s with a 300s staleness
    check on the event loop heartbeat. If the event loop freezes:
    - _last_event_loop_beat stops updating (callback can't fire)
    - After 300s stale, watchdog stops pinging systemd
    - systemd kills us after WatchdogSec=300 with no ping
    """
    def _heartbeat():
        while True:
            _time.sleep(120)
            with _beat_lock:
                age = _time.time() - _last_event_loop_beat
                is_fresh = age < 300  # S233 ME-2: Check inside lock to avoid stale read
            if is_fresh:
                _sd_notify("WATCHDOG=1")
            else:
                logger.warning(f"Watchdog: event loop stale ({age:.0f}s) — NOT pinging systemd")
    t = threading.Thread(target=_heartbeat, daemon=True, name="sd_watchdog")
    t.start()


def _log_shutdown_threads() -> None:
    """Record live threads at process exit so stop-path hangs have named evidence."""
    try:
        threads = threading.enumerate()
        logger.info("Shutdown thread dump: %d thread(s) alive at process exit", len(threads))
        for thread in threads:
            logger.info(
                "Shutdown thread: name=%r ident=%s daemon=%s alive=%s",
                thread.name,
                thread.ident,
                thread.daemon,
                thread.is_alive(),
            )
    except Exception as exc:
        logger.debug("Shutdown thread dump failed: %s", exc)


def _blocking_shutdown_threads() -> list[threading.Thread]:
    """Return non-daemon survivors that can keep interpreter shutdown alive."""
    current = threading.current_thread()
    return [
        thread for thread in threading.enumerate()
        if thread is not current and thread.is_alive() and not thread.daemon
    ]


def _final_shutdown_guard() -> None:
    """Bound process exit after post-shutdown cleanup has logged named survivors."""
    try:
        from agent.integrations.telegram_bot import shutdown_cleanup_completed

        if not shutdown_cleanup_completed():
            logger.info("Shutdown final guard skipped: cleanup-completed flag is false")
            return

        blockers = _blocking_shutdown_threads()
        if not blockers:
            logger.info("Shutdown final guard: no blocking non-daemon threads remain")
            return

        logger.error(
            "Shutdown final guard forcing clean process exit after cleanup; "
            "blocking non-daemon survivors=%s",
            [
                {
                    "name": thread.name,
                    "ident": thread.ident,
                    "alive": thread.is_alive(),
                }
                for thread in blockers
            ],
        )
        _release_pid_lock()
        os._exit(0)
    except Exception as exc:
        logger.warning("Shutdown final guard failed: %s", exc)


def main():
    _acquire_pid_lock()

    logger.info("=" * 50)
    logger.info("  OpenClaw/Moltbot Starting...")
    logger.info("=" * 50)

    from agent.config import config

    if not config.anthropic_api_key:
        logger.error("No Anthropic credential configured (ANTHROPIC_API_KEY or ONEPROVIDER_API_KEY). Exiting.")
        sys.exit(1)
    _gw = bool(getattr(config, "anthropic_gateway", False))
    _base = str(getattr(config, "anthropic_base_url", "") or "")
    logger.info("Anthropic route: %s", f"gateway host={_base.removeprefix('https://')}" if _gw else "direct")
    if not config.telegram_bot_token:
        logger.error("TELEGRAM_BOT_TOKEN not set. Exiting.")
        sys.exit(1)
    if not config.telegram_authorized_users:
        logger.warning("TELEGRAM_AUTHORIZED_USERS not set!")

    logger.info(f"Default model: {config.default_model}")
    logger.info(f"Authorized users: {config.telegram_authorized_users}")
    logger.info(f"Paper mode: {config.alpaca_paper}")

    os.makedirs(config.log_dir, exist_ok=True)

    from agent.integrations.telegram_bot import create_bot
    bot = create_bot()

    # PTB owns SIGTERM here; its shutdown path runs _post_shutdown before process exit.

    # S162: Notify systemd we're ready + start watchdog heartbeat
    _sd_notify("READY=1")
    _start_watchdog_thread()

    logger.info("Moltbot is live. Waiting for messages...")
    try:
        bot.run_polling(drop_pending_updates=False)  # S163-B: preserve messages across restarts
    except SystemExit:
        logger.info("Moltbot shutting down cleanly.")
    finally:
        logger.info("Moltbot process exiting.")
        _log_shutdown_threads()
        _final_shutdown_guard()


if __name__ == "__main__":
    main()
