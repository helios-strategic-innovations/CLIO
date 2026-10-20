"""Health check routes."""

import subprocess
import logging

import psutil
from fastapi import APIRouter, Depends

from mission_control.backend.auth import require_auth
from mission_control.backend.config import mc_config, effective_allowed_services
from mission_control.backend.deps import run_in_executor

router = APIRouter(prefix="/api", tags=["health"])
logger = logging.getLogger("mission_control.health")


def _service_status(name: str) -> dict:
    try:
        result = subprocess.run(
            ["systemctl", "is-active", name],
            capture_output=True, text=True, timeout=5,
        )
        active = result.stdout.strip()
        return {"name": name, "status": active}
    except Exception:
        return {"name": name, "status": "unknown"}


def _collect_health() -> dict:
    """Blocking health collection — runs in thread executor."""
    services = [_service_status(s) for s in effective_allowed_services()]
    services.append(_service_status("moltbot-control"))

    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    return {
        "services": services,
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=0.5),
            "memory_total_gb": round(mem.total / (1024**3), 1),
            "memory_used_gb": round(mem.used / (1024**3), 1),
            "memory_percent": mem.percent,
            "disk_total_gb": round(disk.total / (1024**3), 1),
            "disk_used_gb": round(disk.used / (1024**3), 1),
            "disk_percent": disk.percent,
        },
    }


@router.get("/health")
async def health():
    """Public health check — no auth required."""
    return {"status": "ok", "service": "mission-control"}


@router.get("/health/detailed")
async def health_detailed(_=Depends(require_auth)):
    return await run_in_executor(_collect_health)
