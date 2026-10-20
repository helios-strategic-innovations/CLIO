"""News routes — recent news items, categorical breaking news feed."""

import json
import logging
import os
import re
import time
from datetime import datetime

from fastapi import APIRouter, Depends, Query
from typing import Optional

from mission_control.backend.auth import require_auth
from mission_control.backend.deps import read_json, run_in_executor
from mission_control.backend.config import mc_config

router = APIRouter(prefix="/api", tags=["news"])
logger = logging.getLogger("mission_control.news")

_INTEL_STATE = os.path.join(mc_config.openclaw_root, "data", "intelligence", "news_intel_state.json")
_NEWS_STATE = os.path.join(mc_config.openclaw_root, "data", "knowledge", "news_update_state.json")
_ANALYSIS_DIR = os.path.join(mc_config.openclaw_root, "data", "knowledge", "trading", "market_analysis")

# Category classification keywords — checked against title + summary
_CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("Politics", ["trump", "biden", "congress", "senate", "election", "elections", "gop", "democrat", "republican", "white house", "executive order", "legislation", "midterm"]),
    ("Tariffs & Trade", ["tariff", "tariffs", "trade war", "import duty", "export ban", "sanctions", "trade deal", "trade policy", "wto", "nafta", "usmca"]),
    ("Geopolitics", ["war", "conflict", "invasion", "military", "nato", "geopolitical", "missile", "nuclear", "escalation", "ceasefire", "peace deal", "troops"]),
    ("Defense", ["defense", "pentagon", "lockheed", "raytheon", "northrop", "general dynamics", "bae", "military contract", "arms deal"]),
    ("Oil & Gas", ["brent", "wti", "crude", "opec", "oil price", "oil output", "petroleum", "natural gas", "lng", "pipeline", "strait of hormuz", "oil supply"]),
    ("Commodities", ["gold", "silver", "copper", "wheat", "corn", "soybean", "iron ore", "lithium", "uranium", "commodity", "commodities", "mining"]),
    ("Precious Metals", ["gold", "silver", "platinum", "palladium", "bullion", "precious metal"]),
    ("Tech", ["tech sector", "nasdaq", "semiconductor", "chip", "ai ", "artificial intelligence", "nvidia", "apple", "google", "microsoft", "meta", "amazon", "tsmc"]),
    ("Fed & Rates", ["federal reserve", "fomc", "interest rate", "rate cut", "rate hike", "powell", "inflation", "cpi", "pce", "monetary policy", "quantitative"]),
    ("Bonds", ["treasury", "bond", "yield", "10-year", "2-year", "auction", "duration", "credit spread", "fixed income"]),
    ("Equities", ["correction", "rally", "selloff", "sell-off", "bear market", "bull market", "s&p 500", "spy", "dow jones", "russell", "earnings", "ipo"]),
    ("Crypto", ["bitcoin", "crypto", "ethereum", "blockchain", "stablecoin", "defi"]),
]


def _classify_item(title: str, summary: str) -> str:
    """Classify a news item into a category based on keyword matching."""
    text = (title + " " + summary).lower()
    best_cat = ""
    best_count = 0
    for cat, keywords in _CATEGORY_RULES:
        count = sum(1 for kw in keywords if kw in text)
        if count > best_count:
            best_count = count
            best_cat = cat
    return best_cat or "General"


def _parse_epoch(ts) -> float:
    """Parse timestamp to epoch. Returns 0 on failure."""
    try:
        if isinstance(ts, (int, float)):
            return float(ts)
        if isinstance(ts, str) and ts:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return dt.timestamp()
    except Exception:
        pass
    return 0


@router.get("/news")
async def get_news(
    limit: int = Query(50, ge=1, le=200),
    category: Optional[str] = None,
    _=Depends(require_auth),
):
    """Recent news items from all sources."""

    def _gather() -> list:
        items = []

        # Read news_intel_state for intelligence items
        intel = read_json(_INTEL_STATE)
        if intel and isinstance(intel, dict):
            for item in intel.get("items", [])[-limit:]:
                title = item.get("title", item.get("headline", ""))
                summary = item.get("summary", "")
                items.append({
                    "source": "intelligence",
                    "topic": item.get("topic", ""),
                    "category": _classify_item(title, summary),
                    "title": title,
                    "summary": summary,
                    "timestamp": item.get("sent_at", item.get("timestamp", "")),
                    "symbols": item.get("symbols", []),
                })

        # Read recent market analysis files
        if os.path.isdir(_ANALYSIS_DIR):
            files = sorted(os.listdir(_ANALYSIS_DIR), reverse=True)[:limit]
            for fname in files:
                if not fname.endswith(".json"):
                    continue
                data = read_json(os.path.join(_ANALYSIS_DIR, fname))
                if data:
                    title = data.get("title", fname)
                    summary = data.get("content", "")[:500]
                    items.append({
                        "source": "analysis",
                        "topic": data.get("category", ""),
                        "category": _classify_item(title, summary),
                        "title": title,
                        "summary": summary,
                        "timestamp": data.get("created", ""),
                        "symbols": data.get("tags", []),
                    })
        return items

    items = await run_in_executor(_gather)

    # Sort by timestamp descending
    items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    if category:
        items = [i for i in items if i.get("category") == category or i.get("topic") == category]

    return {"items": items[:limit], "count": len(items[:limit])}


@router.get("/news/breaking")
async def get_breaking_news(
    hours: int = Query(12, ge=1, le=48),
    _=Depends(require_auth),
):
    """Categorical news feed — recent items classified by topic, chronological."""
    cutoff = time.time() - hours * 3600

    def _gather() -> list:
        recent = []

        # Intel items (news_intel_state)
        intel = read_json(_INTEL_STATE) or {}
        for item in intel.get("items", []):
            ts = item.get("sent_at", item.get("timestamp", ""))
            epoch = _parse_epoch(ts)
            if epoch and epoch >= cutoff:
                title = item.get("title", item.get("headline", ""))
                summary = item.get("summary", "")
                recent.append({
                    "source": "intelligence",
                    "topic": item.get("topic", ""),
                    "category": _classify_item(title, summary),
                    "title": title,
                    "summary": summary,
                    "timestamp": ts,
                    "symbols": item.get("symbols", []),
                })

        # Recent market analysis files
        if os.path.isdir(_ANALYSIS_DIR):
            files = sorted(os.listdir(_ANALYSIS_DIR), reverse=True)[:30]
            for fname in files:
                if not fname.endswith(".json"):
                    continue
                data = read_json(os.path.join(_ANALYSIS_DIR, fname))
                if not data:
                    continue
                created = data.get("created", "")
                epoch = _parse_epoch(created)
                if epoch and epoch >= cutoff:
                    title = data.get("title", fname)
                    summary = data.get("content", "")[:500]
                    recent.append({
                        "source": "analysis",
                        "topic": data.get("category", ""),
                        "category": _classify_item(title, summary),
                        "title": title,
                        "summary": summary,
                        "timestamp": created,
                        "symbols": data.get("tags", []),
                    })
        return recent

    recent = await run_in_executor(_gather)

    recent.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    # Collect available categories for filter UI
    categories = sorted({r["category"] for r in recent})

    return {"breaking": recent[:30], "count": len(recent), "categories": categories}
