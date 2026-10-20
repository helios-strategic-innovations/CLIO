"""System API — shell execution, process management, logs, metrics."""

import asyncio
import logging
import os
import shlex
import time
from typing import Optional

import psutil
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from mission_control.backend.auth import require_auth
from mission_control.backend.config import mc_config
from mission_control.backend.deps import get_trading_client, run_in_executor

logger = logging.getLogger("mission_control.system")
router = APIRouter(prefix="/api/system", tags=["system"])


# ---------------------------------------------------------------------------
# Shell execution
# ---------------------------------------------------------------------------

# Allowed command prefixes for safety
# NOTE: No cat/env/curl — cat reads .env secrets, env dumps all secrets,
# curl enables SSRF/exfiltration. Use /api/files/read and /api/system/env instead.
_ALLOWED_PREFIXES = (
    "ls ",
    "wc ",
    "git log", "git status", "git diff", "git show", "git branch",
    "python3 -m py_compile",
    "pip list", "pip show",
    "systemctl status ", "systemctl is-active ",
    "journalctl ",
    "df ", "du ",
    "free", "uptime", "uname",
    "ps aux", "ps -ef",
    "date", "whoami", "hostname",
)

# Dangerous patterns to reject
_BLOCKED_PATTERNS = (
    "rm ", "mv ", "cp ",  # No file manipulation
    "mkfs", "dd ",
    "shutdown", "reboot", "halt",
    "passwd", "useradd", "userdel",
    "iptables", "ufw",
    "chmod", "chown",
    "curl ", "wget ",  # No network access (SSRF)
    "cat ", "head ", "tail ",  # No arbitrary file reads (use /api/files/read)
    "env",  # No env dump (use /api/system/env)
    ".env",  # Block any reference to .env files
    "tee ", "truncate",
)


class ShellRequest(BaseModel):
    command: str
    timeout: int = 30  # seconds
    cwd: Optional[str] = None


@router.post("/shell")
async def execute_shell(
    req: ShellRequest,
    auth=Depends(require_auth),
):
    """Execute a shell command. Sandboxed to safe commands."""
    cmd = req.command.strip()

    # Block dangerous patterns
    for blocked in _BLOCKED_PATTERNS:
        if blocked in cmd:
            raise HTTPException(status_code=403, detail=f"Blocked pattern: {blocked}")

    # Check against allowed prefixes
    allowed = any(cmd.startswith(prefix) for prefix in _ALLOWED_PREFIXES)
    # Also allow service management for our services only
    if cmd.startswith("systemctl "):
        parts = cmd.split()
        if len(parts) >= 3 and parts[2] in ("hydrinia", "moltbot", "moltbot-voice", "moltbot-control"):
            allowed = True

    if not allowed:
        raise HTTPException(
            status_code=403,
            detail=f"Command not in allowlist. Allowed prefixes: {', '.join(_ALLOWED_PREFIXES[:10])}...",
        )

    cwd = req.cwd or "/home/agent/openclaw"
    timeout = min(req.timeout, 120)  # Cap at 2 minutes

    try:
        proc = await asyncio.create_subprocess_exec(
            *shlex.split(cmd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return {"command": cmd, "exit_code": -1, "stdout": "", "stderr": "Command timed out", "timed_out": True}
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail=f"Command not found: {shlex.split(cmd)[0]}")

    return {
        "command": cmd,
        "exit_code": proc.returncode,
        "stdout": stdout.decode("utf-8", errors="replace")[-50000:],  # Cap output
        "stderr": stderr.decode("utf-8", errors="replace")[-10000:],
        "timed_out": False,
    }


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------

_MANAGED_SERVICES = ("moltbot", "moltbot-voice", "moltbot-control", "hydrinia")

# Hydrinia is permanently retired (operator direction, 2026-09-24); its unit is
# only offered for management when the bridge is explicitly re-enabled.
_ALWAYS_MANAGED_SERVICES = ("moltbot", "moltbot-voice", "moltbot-control")


def _hydrinia_bridge_enabled() -> bool:
    try:
        from agent.hydrinia_bridge import bridge_enabled

        return bridge_enabled()
    except Exception as e:
        logger.debug(f"system: bridge_enabled() unavailable ({type(e).__name__}: {e})")
        return False


def _managed_services() -> tuple:
    if _hydrinia_bridge_enabled():
        return _MANAGED_SERVICES
    return _ALWAYS_MANAGED_SERVICES


async def _probe_service(svc: str) -> str:
    """Return `systemctl is-active` status for one service (never raises)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "systemctl", "is-active", svc,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        return stdout.decode().strip()
    except Exception:
        return "unknown"


@router.get("/services")
async def list_services(_=Depends(require_auth)):
    """Get status of all managed systemd services."""
    allowed = _managed_services()
    statuses = await asyncio.gather(*[_probe_service(svc) for svc in allowed])
    results = dict(zip(allowed, statuses))
    return {"services": results}


class ServiceAction(BaseModel):
    service: str
    action: str  # start, stop, restart, status


@router.post("/service")
async def manage_service(
    req: ServiceAction,
    auth=Depends(require_auth),
):
    """Start/stop/restart a managed service."""
    allowed = _managed_services()
    if req.service not in allowed:
        raise HTTPException(status_code=400, detail=f"Unknown service. Allowed: {allowed}")
    if req.action not in ("start", "stop", "restart", "status"):
        raise HTTPException(status_code=400, detail="Action must be: start, stop, restart, status")

    try:
        proc = await asyncio.create_subprocess_exec(
            "sudo", "systemctl", req.action, req.service,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
    except asyncio.TimeoutError:
        return {"service": req.service, "action": req.action, "status": "timeout"}

    return {
        "service": req.service,
        "action": req.action,
        "exit_code": proc.returncode,
        "stdout": stdout.decode("utf-8", errors="replace"),
        "stderr": stderr.decode("utf-8", errors="replace"),
    }


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------

@router.get("/logs")
async def get_logs(
    service: str = Query("moltbot", description="Service name"),
    lines: int = Query(100, ge=1, le=1000),
    since: Optional[str] = Query(None, description="Time filter, e.g. '1h ago', '2026-03-11'"),
    grep: Optional[str] = Query(None, description="Filter pattern"),
    _=Depends(require_auth),
):
    """Fetch recent journalctl logs for a service."""
    if service not in _managed_services() and service != "caddy":
        raise HTTPException(status_code=400, detail=f"Unknown service")

    cmd = ["journalctl", "-u", service, "--no-pager", "-n", str(lines)]
    if since:
        cmd.extend(["--since", since])
    if grep:
        import re as _re
        # S162 C-2: Strip regex metacharacters from whitelist to prevent ReDoS
        if not _re.match(r'^[a-zA-Z0-9_\-\.\s]+$', grep) or len(grep) > 100:
            raise HTTPException(status_code=400, detail="Invalid grep pattern")
        cmd.extend(["-g", grep])

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
    except asyncio.TimeoutError:
        return {"lines": [], "error": "Timeout reading logs"}

    log_lines = stdout.decode("utf-8", errors="replace").strip().split("\n")
    return {"service": service, "lines": log_lines, "count": len(log_lines)}


# ---------------------------------------------------------------------------
# System metrics
# ---------------------------------------------------------------------------

@router.get("/metrics")
async def system_metrics(_=Depends(require_auth)):
    """System resource usage — CPU, memory, disk."""
    def _gather():
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.5),
            "memory": {
                "total_gb": round(mem.total / (1024**3), 1),
                "used_gb": round(mem.used / (1024**3), 1),
                "percent": mem.percent,
            },
            "disk": {
                "total_gb": round(disk.total / (1024**3), 1),
                "used_gb": round(disk.used / (1024**3), 1),
                "percent": disk.percent,
            },
            "uptime_hours": round((time.time() - psutil.boot_time()) / 3600, 1),
            "load_avg": list(os.getloadavg()),
        }

    return await run_in_executor(_gather)


# ---------------------------------------------------------------------------
# Bot-specific status
# ---------------------------------------------------------------------------

@router.get("/bot-status")
async def bot_status(_=Depends(require_auth)):
    """Comprehensive bot health check — processes, recent errors, budgets."""
    status = {}

    # Service states (probed concurrently)
    allowed = _managed_services()
    svc_states = await asyncio.gather(*[_probe_service(svc) for svc in allowed])
    status["services"] = dict(zip(allowed, svc_states))

    # File-derived health (batched off the event loop)
    def _gather_files() -> dict:
        import json
        out: dict = {}

        # Rate limits / budgets
        try:
            rl_path = "/home/agent/openclaw/data/rate_limits.json"
            if os.path.exists(rl_path):
                with open(rl_path) as f:
                    rl = json.load(f)
                # Show top budgets by usage
                budgets = {}
                for name, info in rl.items():
                    if isinstance(info, dict) and "max_requests" in info:
                        budgets[name] = {
                            "used": info.get("used", 0),
                            "max": info.get("max_requests", 0),
                            "pct": round(100 * info.get("used", 0) / max(info.get("max_requests", 1), 1), 1),
                        }
                out["budgets"] = dict(sorted(budgets.items(), key=lambda x: x[1]["pct"], reverse=True))
        except Exception as e:
            logger.error(f"Budget read failed: {e}", exc_info=True)
            out["budgets_error"] = "Internal server error"

        # Working memory count
        try:
            wm_path = "/home/agent/openclaw/data/memory/working_memory.json"
            if os.path.exists(wm_path):
                with open(wm_path) as f:
                    wm = json.load(f)
                out["working_memory_entries"] = len(wm.get("entries", []))
        except Exception:
            pass

        # DPOF halt state (CLIO-native — Hydrinia's EMERGENCY_STOP is retired, S/O 2026-09-24)
        try:
            from agent.dpof import readmodel

            risk = readmodel.load_risk()
            halt_file = readmodel.halt_file_present()
            out["emergency_stop"] = bool(halt_file or risk.hard_halted)
            out["dpof_halt_file"] = halt_file
            out["dpof_hard_halted"] = bool(risk.hard_halted)
        except Exception as e:
            logger.debug(f"DPOF halt state read failed: {e}")
            out["emergency_stop"] = False

        # Position count — DPOF's own book (CLIO-native)
        try:
            from agent.dpof import readmodel

            positions = readmodel.load_positions()
            open_positions = readmodel.open_positions()
            out["open_positions"] = len(open_positions)
            out["total_positions"] = len(positions)
        except Exception as e:
            logger.debug(f"DPOF position count read failed: {e}")

        # Knowledge entries
        try:
            ki_path = "/home/agent/openclaw/data/knowledge/knowledge_index.json"
            if os.path.exists(ki_path):
                with open(ki_path) as f:
                    ki = json.load(f)
                out["knowledge_entries"] = len(ki.get("entries", []))
        except Exception:
            pass

        return out

    status.update(await run_in_executor(_gather_files))
    return status


# ---------------------------------------------------------------------------
# Environment (read-only, redacted)
# ---------------------------------------------------------------------------

@router.get("/env")
async def get_environment(_=Depends(require_auth)):
    """List environment variable NAMES (values redacted for secrets)."""
    # Show which vars are set, redact sensitive values
    safe_prefixes = ("BUDGET_", "MOLTBOT_SEAL_", "MOLTBOT_ADAPTIVE", "MOLTBOT_SERVER",
                     "MOLTBOT_CROSS_CATEGORY", "MOLTBOT_COST_AWARE", "NEWS_INTEL",
                     "PERSONALITY_ENABLED", "MOLTBOOK_", "OVERRIDE_DIRECTIVES")
    sensitive_keywords = ("KEY", "SECRET", "PASSWORD", "HASH", "TOKEN", "PIN", "CREDENTIALS")

    env_vars = {}
    for key in sorted(os.environ.keys()):
        if any(key.startswith(p) for p in safe_prefixes):
            env_vars[key] = os.environ[key]
        elif any(s in key.upper() for s in sensitive_keywords):
            env_vars[key] = "[REDACTED]"
        elif key.startswith("ALPACA_") or key.startswith("MISSION_CONTROL_"):
            env_vars[key] = "[REDACTED]"
        # Skip system env vars
    return {"env": env_vars}


# ---------------------------------------------------------------------------
# Data-source health / freshness
# ---------------------------------------------------------------------------

_HEALTH_TTL = 12.0
_ACCOUNT_CACHE = {"ts": -1e9, "data": None}

_NEWS_STATE_PATH = os.path.join(
    mc_config.openclaw_root,
    "data",
    "intelligence",
    "news_intel_state.json",
)
_NEWS_WARN, _NEWS_DOWN = 3600, 21600
_RUNTIME_FILES = [
    os.path.join(mc_config.openclaw_root, "data", "knowledge", "knowledge_index.json"),
    os.path.join(mc_config.openclaw_root, "data", "memory", "working_memory.json"),
    os.path.join(mc_config.openclaw_root, "data", "intelligence", "news_intel_state.json"),
    os.path.join(mc_config.openclaw_root, "data", "intelligence", "macro_research_state.json"),
]
_RUNTIME_WARN, _RUNTIME_DOWN = 1800, 7200


def _to_float_or_none(value):
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _file_freshness(paths, warn: int, down: int) -> dict:
    """Report freshness from the newest existing file in a path set."""
    newest_mtime = None
    newest_path = None
    candidates = paths if isinstance(paths, (list, tuple)) else [paths]
    for path in candidates:
        try:
            modified = os.path.getmtime(path)
        except OSError:
            continue
        if newest_mtime is None or modified > newest_mtime:
            newest_mtime = modified
            newest_path = path
    if newest_mtime is None:
        return {
            "status": "down",
            "age_seconds": None,
            "updated_at": None,
            "detail": "no state files found",
        }

    from datetime import datetime, timezone

    age = max(0.0, time.time() - newest_mtime)
    status = "ok" if age < warn else "warn" if age < down else "down"
    return {
        "status": status,
        "age_seconds": round(age),
        "updated_at": datetime.fromtimestamp(newest_mtime, tz=timezone.utc).isoformat(),
        "detail": f"newest: {os.path.basename(newest_path)} ({round(age / 60)}m ago)",
    }


async def _account_health() -> dict:
    """Return the legacy account identity badge from a bounded read-only cache."""
    now = time.monotonic()
    cached = _ACCOUNT_CACHE.get("data")
    if cached is not None and (now - _ACCOUNT_CACHE["ts"]) < _HEALTH_TTL:
        return cached

    def _fetch() -> dict:
        account = get_trading_client().get_account()
        raw_status = getattr(account, "status", None)
        status = (
            getattr(raw_status, "value", None)
            or (str(raw_status).split(".")[-1] if raw_status is not None else None)
        )
        return {
            "reachable": True,
            "account_number": getattr(account, "account_number", None),
            "equity": _to_float_or_none(getattr(account, "equity", None)),
            "status": status,
        }

    try:
        data = await run_in_executor(_fetch)
    except Exception:
        # Do not return or log raw broker errors from this trust-summary route.
        logger.warning("health: Alpaca account fetch failed")
        data = {
            "reachable": False,
            "account_number": None,
            "equity": None,
            "status": None,
        }

    _ACCOUNT_CACHE["data"] = data
    _ACCOUNT_CACHE["ts"] = now
    return data


@router.get("/health")
async def system_health(_=Depends(require_auth)):
    """Legacy account identity and data-source trust summary."""
    from datetime import datetime, timezone

    account = await _account_health()

    # Config presence only: this route never pings Polygon or Unusual Whales.
    polygon_ok = bool(os.environ.get("POLYGON_API_KEY", "").strip())
    uw_ok = bool(os.environ.get("UW_API_KEY", "").strip())

    news = await run_in_executor(
        _file_freshness,
        _NEWS_STATE_PATH,
        _NEWS_WARN,
        _NEWS_DOWN,
    )
    runtime = await run_in_executor(
        _file_freshness,
        _RUNTIME_FILES,
        _RUNTIME_WARN,
        _RUNTIME_DOWN,
    )

    sources = [
        {
            "key": "alpaca",
            "label": "Alpaca",
            "check": "live",
            "status": "ok" if account["reachable"] else "down",
            "detail": (
                f"account {account['account_number']} - {account['status']}"
                if account["reachable"]
                else "unreachable"
            ),
        },
        {
            "key": "polygon",
            "label": "Polygon",
            "check": "config",
            "status": "ok" if polygon_ok else "down",
            "detail": "API key configured" if polygon_ok else "no API key",
        },
        {
            "key": "uw",
            "label": "Unusual Whales",
            "check": "config",
            "status": "ok" if uw_ok else "down",
            "detail": "API key configured" if uw_ok else "no API key",
        },
        {"key": "news", "label": "News Intel", "check": "file", **news},
        {"key": "files", "label": "Runtime State", "check": "file", **runtime},
    ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "account": account,
        "sources": sources,
    }
