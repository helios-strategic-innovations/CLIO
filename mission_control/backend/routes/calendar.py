"""Calendar routes — earnings, economic, geopolitical."""

import json
import logging
import os
import time

from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel
from typing import Optional

from mission_control.backend.auth import require_auth
from mission_control.backend.deps import read_json, load_symbols_universe, run_in_executor
from mission_control.backend.config import mc_config

router = APIRouter(prefix="/api/calendar", tags=["calendar"])
logger = logging.getLogger("mission_control.calendar")

_INTEL_DIR = os.path.join(mc_config.openclaw_root, "data", "intelligence")
_GEO_CALENDAR_PATH = os.path.join(_INTEL_DIR, "geopolitical_calendar.json")


class GeoEvent(BaseModel):
    date: str
    event: str
    location: str = ""
    type: str = ""  # summit, visit, election, military, sanctions, other
    impact: str = "MEDIUM"  # HIGH, MEDIUM, LOW
    notes: str = ""
    linked_symbols: list[str] = []


@router.get("/earnings")
async def get_earnings_calendar(days: int = Query(7, ge=1, le=30), _=Depends(require_auth)):
    """Upcoming earnings — Polygon TMX + Benzinga, cross-referenced with symbols.txt."""
    try:
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo
        from agent.config import config

        _NY = ZoneInfo("America/New_York")
        now = datetime.now(_NY)
        start = now.strftime("%Y-%m-%d")
        end = (now + timedelta(days=days)).strftime("%Y-%m-%d")

        universe = load_symbols_universe()

        # Read cached earnings calendar from earnings_intel module
        events = []
        earnings_cache = read_json(os.path.join(_INTEL_DIR, "earnings_calendar.json"))
        if earnings_cache and isinstance(earnings_cache, dict):
            for ev in earnings_cache.get("earnings", earnings_cache.get("results", [])):
                symbol = ev.get("ticker", ev.get("symbol", ""))
                ev_date = ev.get("report_date", ev.get("date", ""))
                if start <= ev_date <= end:
                    events.append({
                        "symbol": symbol,
                        "date": ev_date,
                        "time": ev.get("time", ""),
                        "name": ev.get("name", symbol),
                        "in_universe": symbol in universe,
                    })

        events.sort(key=lambda e: e.get("date", ""))
        return {"events": events, "start": start, "end": end, "count": len(events)}
    except Exception as e:
        logger.error(f"Earnings calendar failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/economic")
async def get_economic_calendar(_=Depends(require_auth)):
    """Economic data calendar — from cached TE data or FRED."""
    # Try cached TE calendar first
    cached = read_json(os.path.join(_INTEL_DIR, "econ_calendar.json"))
    if cached and isinstance(cached, list):
        return {"events": cached[:50], "source": "trading_economics_cache"}

    # Fallback: return FOMC schedule (S162 H-7: year-gated to prevent stale data)
    from datetime import datetime as _dt_cls
    _now_year = _dt_cls.now().year
    if _now_year == 2026:
        fomc_dates = [
            "2026-01-28", "2026-01-29", "2026-03-18", "2026-03-19",
            "2026-05-06", "2026-05-07", "2026-06-17", "2026-06-18",
            "2026-07-29", "2026-07-30", "2026-09-16", "2026-09-17",
            "2026-11-04", "2026-11-05", "2026-12-16", "2026-12-17",
        ]
    else:
        logger.warning(f"S162 H-7: FOMC dates not configured for {_now_year} — returning empty")
        fomc_dates = []
    events = [{"date": d, "event": "FOMC Meeting", "importance": 3, "source": "hardcoded_2026"} for d in fomc_dates]
    return {"events": events, "source": "fomc_hardcoded"}


@router.get("/geopolitical")
async def get_geopolitical_calendar(_=Depends(require_auth)):
    """Geopolitical events calendar."""
    events = read_json(_GEO_CALENDAR_PATH) or []
    return {"events": events}


@router.post("/geopolitical")
async def add_geopolitical_event(event: GeoEvent, _=Depends(require_auth)):
    """Add or update a geopolitical event."""
    events = read_json(_GEO_CALENDAR_PATH) or []
    entry = event.model_dump()
    entry["created"] = time.time()
    entry["source"] = "manual"
    events.append(entry)

    # Sort by date
    events.sort(key=lambda e: e.get("date", ""))

    # Atomic write
    os.makedirs(os.path.dirname(_GEO_CALENDAR_PATH), exist_ok=True)
    tmp = _GEO_CALENDAR_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(events, f, indent=2)
    os.replace(tmp, _GEO_CALENDAR_PATH)

    return {"status": "added", "total_events": len(events)}
