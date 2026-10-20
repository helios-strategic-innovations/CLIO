"""
OpenClaw/Moltbot — Commodity Supply Chain Analysis (Session 92, Phase 2)

Tracks supply/demand fundamentals across energy, metals, and agriculture,
validates data sources, and produces supply balance scenario analysis via
the A/B/C/D macro research framework:
  A) Multi-source research — EIA, OPEC, USDA, tanker tracking, rig counts
  B) Memory recall — prior supply balance history from knowledge base
  C) Data validity — EIA vs OPEC, China opacity, weather reliability, USDA bias
  D) Scenario analysis — 5-level supply balance (GLUT / OVERSUPPLIED / BALANCED /
     TIGHT / CRISIS)

Two triggers:
1. Escalation from news_intel.py when commodities topic produces supply keywords
2. Scheduled weekly callback — Monday 6:30 AM ET comprehensive supply review
"""

import asyncio
import fcntl
import json
import logging
import os
import random
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import anthropic

from agent.config import config
from agent.prompts.system_prompt import cached_system_block
from agent.rate_limiter import budget_tracker, BudgetExhaustedError
from agent.security import sanitize_ingested_content

logger = logging.getLogger("openclaw.commodity_research")

_NY = ZoneInfo("America/New_York")

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "intelligence",
)
_STATE_PATH = os.path.join(_DATA_DIR, "commodity_research_state.json")

BRIDGE_DIR = os.path.join(config.tradingbot_dir, "moltbot_knowledge", "intelligence")

_cached_client: anthropic.Anthropic | None = None
_client_lock = __import__("threading").Lock()


# ============================================================
# CC-CLIO-FD-4: Fundamental Snapshot for Commodity-Linked Equities
# ============================================================

# Representative tickers by commodity type for fundamental enrichment
_COMMODITY_EQUITY_TICKERS: dict[str, list[str]] = {
    "crude_oil": ["XOM", "CVX", "COP"],
    "natural_gas": ["EQT", "AR", "SWN"],
    "metals": ["FCX", "NEM", "SCCO"],
    "agriculture": ["ADM", "BG", "MOS"],
}


def _fd_commodity_equity_snapshot(ticker: str) -> str:
    """Add fundamental context for commodity-linked stocks (XOM, CVX, FCX, etc.).

    BC#112: Snapshot fields use full names — price_to_earnings_ratio (not pe_ratio),
    free_cash_flow_yield (not free_cash_flow_per_share).
    """
    if os.environ.get("CLIO_FD_COMMODITY_FUNDAMENTALS", "0") != "1":
        return ""
    try:
        from agent.integrations.fd_client import get_fd_client
        snap = get_fd_client().get_financial_metrics_snapshot(ticker)
        if not snap or (isinstance(snap, dict) and "_error" in snap):
            return ""

        parts = []
        # BC#112: Use actual FD API field names (long form for snapshot)
        for field, label in [
            ("price_to_earnings_ratio", "PE"),
            ("enterprise_value_to_ebitda_ratio", "EV/EBITDA"),
            ("net_margin", "Margin"),
            ("revenue_growth", "RevG"),
            ("free_cash_flow_yield", "FCFy"),
        ]:
            val = snap.get(field)
            if val is not None:
                try:
                    fval = float(val)
                    if field in ("net_margin", "revenue_growth", "free_cash_flow_yield"):
                        parts.append(f"{label}={fval:.1%}")
                    else:
                        parts.append(f"{label}={fval:.1f}")
                except (ValueError, TypeError):
                    parts.append(f"{label}={val}")

        return f"[FD {ticker}: {' | '.join(parts)}]" if parts else ""
    except Exception as e:
        logger.debug("CC-CLIO-FD-4: Commodity equity snapshot failed for %s: %s", ticker, e)
        return ""


def _fd_commodity_fundamentals(commodity_type: str) -> str:
    """Get fundamental snapshots for representative commodity-linked equities."""
    if os.environ.get("CLIO_FD_COMMODITY_FUNDAMENTALS", "0") != "1":
        return ""
    tickers = _COMMODITY_EQUITY_TICKERS.get(commodity_type, [])
    if not tickers:
        return ""
    snippets = []
    for ticker in tickers[:2]:  # Cap at 2 to conserve API calls
        snippet = _fd_commodity_equity_snapshot(ticker)
        if snippet:
            snippets.append(snippet)
    if not snippets:
        return ""
    result = "Commodity-Linked Equity Fundamentals (FD):\n" + "\n".join(snippets)
    logger.debug("CC-CLIO-FD-4: Commodity fundamentals injected (%d tickers)", len(snippets))
    return result


def _get_client() -> anthropic.Anthropic:
    global _cached_client
    if _cached_client is None:
        with _client_lock:
            if _cached_client is None:
                _cached_client = anthropic.Anthropic(**config.anthropic_client_kwargs(timeout=120))
    return _cached_client


# ============================================================
# Commodity Type Taxonomy + Research Query Maps
# ============================================================

COMMODITY_RESEARCH_MAP: dict[str, dict] = {
    "crude_oil": {
        "keywords": ["crude oil", "WTI", "brent", "OPEC", "petroleum", "oil inventory"],
        "obvious_queries": [
            "EIA weekly petroleum status report latest",
            "OPEC monthly oil market report",
            "Cushing Oklahoma inventory level",
            "WTI Brent spread current",
        ],
        "niche_queries": [
            "OPEC+ compliance by country tanker tracking AIS",
            "Russia actual oil production vs OPEC quota",
            "strategic petroleum reserve SPR level replenishment",
            "refinery utilization by PADD district",
            "crack spread 3:2:1 gasoline heating oil",
            "Baker Hughes rig count US oil",
            "China crude oil imports customs data",
            "jet fuel demand air travel recovery",
            "floating storage crude oil tankers",
            "Iran oil exports sanctions evasion",
        ],
        "validity_queries": [
            "EIA vs OPEC production estimate discrepancy",
            "tanker tracking vs reported production satellite",
            "seasonal adjustment gasoline demand summer winter",
        ],
    },
    "natural_gas": {
        "keywords": ["natural gas", "henry hub", "LNG", "gas storage", "gas inventory"],
        "obvious_queries": [
            "EIA natural gas storage report weekly",
            "Henry Hub natural gas price forecast",
            "LNG terminal utilization US export capacity",
        ],
        "niche_queries": [
            "natural gas storage vs 5 year average",
            "LNG cargo rerouting Europe Asia spot",
            "pipeline capacity constraint Appalachian Permian",
            "power generation natural gas demand",
            "residential heating demand degree days",
            "Freeport LNG Sabine Pass export volumes",
        ],
        "validity_queries": [
            "EIA revision history natural gas storage",
            "weather forecast accuracy 7 day vs 14 day reliability",
        ],
    },
    "metals": {
        "keywords": ["gold", "silver", "copper", "platinum", "aluminum", "steel",
                      "lithium", "rare earth"],
        "obvious_queries": [
            "gold price central bank buying WGC report",
            "copper LME inventory warehouse stocks",
            "silver industrial demand solar panels electronics",
        ],
        "niche_queries": [
            "central bank gold purchases WGC quarterly",
            "gold ETF flows GLD IAU",
            "COMEX gold silver futures positioning COT",
            "copper mine disruptions Chile Peru Escondida",
            "lithium spot price battery demand EV",
            "rare earth export controls China gallium germanium",
            "aluminum LME premium Midwest",
            "steel production capacity utilization US",
        ],
        "validity_queries": [
            "LME warehouse manipulation off warrant inventory",
            "China commodity data accuracy opacity lag",
            "COMEX delivery vs paper contract ratio",
        ],
    },
    "agriculture": {
        "keywords": ["wheat", "corn", "soybeans", "cotton", "USDA", "crop report", "WASDE"],
        "obvious_queries": [
            "USDA crop production report latest",
            "WASDE world supply demand estimates",
            "planting progress report USDA weekly",
        ],
        "niche_queries": [
            "fertilizer prices urea DAP potash",
            "drought monitor US Palmer index",
            "Brazil soybean harvest progress Mato Grosso",
            "Ukraine grain corridor Black Sea exports",
            "El Nino La Nina ENSO forecast crop impact",
            "Mississippi River barge traffic low water",
            "Iowa crop conditions good excellent percentage",
            "China soybean import demand",
        ],
        "validity_queries": [
            "USDA political pressure election year estimates",
            "satellite crop monitoring vs USDA ground truth",
            "WASDE revision history systematic bias",
        ],
    },
}


# ============================================================
# Data Validity Rules
# ============================================================

COMMODITY_DATA_VALIDITY: dict[str, str] = {
    "eia_vs_opec": (
        "EIA and OPEC production estimates can diverge 0.5-1.0 mbpd. "
        "Use tanker tracking (AIS data) as tiebreaker when available."
    ),
    "china_opacity": (
        "Chinese commodity data arrives with 4-6 week delay and +/-10% uncertainty. "
        "Satellite proxies (steel mill activity, port congestion) provide earlier signals."
    ),
    "weather_reliability": (
        "HARD RULE: Only 7-day weather forecasts are useful for demand adjustments. "
        "14-day forecasts are noise -- do NOT use them."
    ),
    "usda_pressure": (
        "In election years, USDA production estimates may be optimistically biased. "
        "Cross-reference with private estimates (Informa, AgriCensus)."
    ),
    "lme_manipulation": (
        "LME on-warrant vs off-warrant inventory can be manipulated. "
        "Flag any single-day inventory moves >10% as suspicious."
    ),
}

# Which validity rules apply to which commodity types
_VALIDITY_MAP: dict[str, list[str]] = {
    "crude_oil": ["eia_vs_opec", "china_opacity"],
    "natural_gas": ["weather_reliability"],
    "metals": ["china_opacity", "lme_manipulation"],
    "agriculture": ["usda_pressure", "weather_reliability"],
}


# ============================================================
# Commodity Type Classification
# ============================================================

def classify_commodity(text: str) -> str | None:
    """Classify text into a commodity type by keyword count.

    Returns best matching commodity type or None.
    """
    text_lower = text.lower()
    best_type = None
    best_count = 0
    for ctype, cfg in COMMODITY_RESEARCH_MAP.items():
        count = sum(1 for kw in cfg["keywords"] if re.search(r'\b' + re.escape(kw.lower()) + r'\b', text_lower))
        if count > best_count:
            best_count = count
            best_type = ctype
    return best_type if best_count > 0 else None


# ============================================================
# Memory Recall — Supply Balance History
# ============================================================

def _recall_supply_balance_history(commodity_type: str) -> str:
    """Search knowledge base for prior commodity research entries."""
    try:
        from agent.tools.knowledge_tools import load_all_entries
        all_entries = load_all_entries()
    except Exception as e:
        logger.debug(f"Failed to load entries for supply balance history: {e}")
        return "No prior supply balance history available."

    cutoff = datetime.now(_NY) - timedelta(days=120)
    relevant = []

    keywords = COMMODITY_RESEARCH_MAP.get(commodity_type, {}).get("keywords", [])

    for entry in all_entries:
        tags = entry.get("tags", [])
        tags_str = " ".join(str(t) for t in tags) if isinstance(tags, list) else ""
        title = entry.get("title", "").lower()
        combined = f"{tags_str} {title}".lower()

        has_tag = "commodity_research" in combined
        has_keyword = any(re.search(r'\b' + re.escape(kw.lower()) + r'\b', combined) for kw in keywords)

        if not (has_tag or has_keyword):
            continue

        created = entry.get("created", "")
        if created:
            try:
                dt = datetime.fromisoformat(created)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=_NY)
                if dt < cutoff:
                    continue
            except (ValueError, TypeError):
                pass
        relevant.append(entry)

    if not relevant:
        return "No prior supply balance history found in knowledge base."

    def _sort_key(e):
        try:
            dt = datetime.fromisoformat(e.get("created", "2020-01-01"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_NY)
            return dt
        except (ValueError, TypeError):
            return datetime(2020, 1, 1, tzinfo=_NY)

    relevant.sort(key=_sort_key, reverse=True)

    lines = []
    for entry in relevant[:5]:
        title = entry.get("title", "Unknown")
        created = entry.get("created", "")[:10]
        content = ""
        file_path_rel = entry.get("file_path", "")
        if file_path_rel:
            try:
                full_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "data", "knowledge", file_path_rel,
                )
                if os.path.isfile(full_path):
                    with open(full_path, "r") as f:
                        data = json.load(f)
                    content = data.get("content", "")[:300]
            except Exception as e:
                logger.debug(f"Failed to read knowledge entry file: {e}")
        lines.append(f"[{created}] {title}\n{content}")

    return "\n---\n".join(lines)


# ============================================================
# Data Source Validation
# ============================================================

def _validate_data_sources(commodity_type: str, results: list[dict]) -> str:
    """Cross-reference validity rules against commodity type and sources."""
    applicable_rules = _VALIDITY_MAP.get(commodity_type, [])
    if not applicable_rules:
        return "No specific data validity concerns for this commodity type."

    parts = []
    for rule_key in applicable_rules:
        rule_text = COMMODITY_DATA_VALIDITY.get(rule_key, "")
        if rule_text:
            parts.append(f"- {rule_key.upper()}: {rule_text}")

    # Note data vintage from results
    source_count = len(results)
    parts.append(f"\nData vintage: {source_count} sources gathered. "
                 "Cross-reference dates in source material for staleness.")

    return "\n".join(parts)


# ============================================================
# Research Gathering
# ============================================================

async def _gather_commodity_research(commodity_type: str) -> list[dict]:
    """Execute research queries for a commodity type.

    Runs 2 obvious + 4 niche queries. Uses report_web_search budget.
    """
    commodity_config = COMMODITY_RESEARCH_MAP.get(commodity_type, {})
    obvious = commodity_config.get("obvious_queries", [])
    niche = commodity_config.get("niche_queries", [])

    queries = []
    for q in obvious[:2]:
        queries.append((q, "obvious"))
    for q in niche[:4]:
        queries.append((q, "niche"))

    now = datetime.now(_NY)
    date_suffix = now.strftime("%B %d %Y")
    results_all = []

    for query, query_type in queries:
        try:
            allowed, msg = await budget_tracker.check_budget("report_web_search")
            if not allowed:
                break
            from agent.integrations.web_search import search_web
            full_query = f"{query} {date_suffix}"
            results = await search_web(full_query, max_results=3)
            await budget_tracker.record_usage("report_web_search")  # S118 W-010: charge AFTER search

            for r in (results or []):
                title = r.get("title", "")
                snippet = r.get("snippet", r.get("body", ""))
                url = r.get("url", r.get("href", ""))
                if title:
                    results_all.append({
                        "title": title,
                        "snippet": snippet,
                        "url": url,
                        "query_type": query_type,
                    })

        except BudgetExhaustedError:
            logger.debug("Budget exhausted during commodity research")
            break
        except Exception as e:
            logger.warning(f"Commodity research query failed: {query}: {e}")

        await asyncio.sleep(0.5)

    # S100: Apply staleness filter to freshly gathered results
    try:
        from agent.integrations.news_update import _url_is_stale, _snippet_is_stale
        results_all = [
            r for r in results_all
            if not _url_is_stale(r.get("url", ""))
            and not _snippet_is_stale(r.get("title", ""), r.get("snippet", ""))
        ]
    except ImportError:
        pass

    return results_all


# ============================================================
# Supply Balance Analysis (Opus-powered)
# ============================================================

_SUPPLY_BALANCE_PROMPT = """You are a commodity supply chain analyst producing supply balance scenario analysis for a trading system.

Analyze the provided commodity data and produce a structured intelligence report.

FORMAT:

COMMODITY: {name, current price level if available}
CURRENT TREND: {direction + magnitude over 30/90/180 days}

SUPPLY CHAIN STATUS:
- Production: {output levels, capacity utilization, disruption risk}
- Inventory: {current level vs 5-year average, trajectory, operational minimums}
- Capacity: {spare capacity, new projects, decommissioning}
- Transportation: {pipeline/shipping constraints, seasonal factors}

DATA CROSS-CHECK:
{Source discrepancies from validity rules}
{Data vintage — how stale is each source?}

PRIOR CYCLE COMPARISON:
{What has changed since last review? Trend direction.}

SUPPLY BALANCE SCENARIOS (probabilities must sum to ~100%):
GLUT (X%): {conditions} — {price target}, downstream impact
OVERSUPPLIED (X%): {conditions} — {price target}, downstream impact
BALANCED (X%): {conditions} — {price target}, downstream impact
TIGHT (X%): {conditions} — {price target}, downstream impact
CRISIS (X%): {conditions} — {price target}, downstream impact

CURVE SHAPE: {backwardation/contango — what it implies for physical market}

DOWNSTREAM IMPACT:
{Airlines, chemicals, utilities, consumer — specific tickers + price impacts}

KEY SIGNAL: {The single most important data point}
CONFIDENCE: {LOW/MEDIUM/HIGH} — {brief justification}

RULES:
- Every scenario MUST include a specific price target or range
- NEVER use: "monitor closely", "watch for", "keep an eye on", "worth noting", "remains to be seen"
- State impacts directly: "WTI $72-75/bbl", "Henry Hub $3.20/MMBtu", "copper $4.15/lb"
- HARD RULE: Only use 7-day weather forecasts for demand adjustments. 14-day forecasts are noise.
- If quarter-end, note potential for inventory window dressing
- Write in plain text only, no markdown formatting"""


async def _generate_supply_balance_analysis(
    commodity_type: str,
    research_results: list[dict],
    supply_history: str,
    validity_assessment: str,
) -> str | None:
    """Generate supply balance analysis using Claude Opus."""
    try:
        allowed, msg = await budget_tracker.check_budget("commodity_research")
        if not allowed:
            logger.debug("commodity_research budget exhausted")
            return None
    except BudgetExhaustedError:
        return None

    # Format research results
    research_text = ""
    for i, r in enumerate(research_results[:15], 1):
        qt = r.get("query_type", "")
        research_text += (
            f"\n[{i}] ({qt}) {r['title']}\n"
            f"{r.get('snippet', '')}\n"
            f"URL: {r.get('url', '')}\n"
        )

    commodity_label = commodity_type.replace("_", " ").title()

    # CC-CLIO-FD-4: Inject commodity equity fundamentals (run_in_executor)
    fd_fundamentals_section = ""
    try:
        _fd_ok, _ = await budget_tracker.check_budget("financial_datasets")
        if _fd_ok:
            _loop = asyncio.get_running_loop()
            fd_fundamentals = await _loop.run_in_executor(None, lambda: _fd_commodity_fundamentals(commodity_type))
            if fd_fundamentals:
                await budget_tracker.record_usage("financial_datasets")
        else:
            fd_fundamentals = ""
    except Exception as e:
        logger.debug("CC-CLIO-FD-4: Commodity fundamentals fetch failed: %s", e)
        fd_fundamentals = ""
    if fd_fundamentals:
        fd_fundamentals_section = f"\n{fd_fundamentals}\n\n"

    user_message = (
        f"COMMODITY TYPE: {commodity_label}\n\n"
        + (fd_fundamentals_section if fd_fundamentals_section else "")
        + f"DATA VALIDITY ASSESSMENT:\n{validity_assessment}\n\n"
        f"PRIOR SUPPLY BALANCE HISTORY:\n{supply_history}\n\n"
        f"NEW RESEARCH FINDINGS:\n{sanitize_ingested_content(research_text, source='ddg_commodity')}\n\n"
        f"Generate a 5-scenario supply balance analysis (GLUT / OVERSUPPLIED / "
        f"BALANCED / TIGHT / CRISIS) with specific price targets, curve shape "
        f"assessment, and downstream impact on equities."
    )

    try:
        client = _get_client()
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, lambda: client.messages.create(
            model=config.default_model,
            max_tokens=2500,
            system=cached_system_block(_SUPPLY_BALANCE_PROMPT),
            messages=[{"role": "user", "content": user_message}],
        ))

        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

        result = text.strip()
        if result:
            await budget_tracker.record_usage("commodity_research")
            return result
        return None

    except Exception as e:
        logger.error(f"Supply balance analysis generation failed: {e}", exc_info=True)
        return None


# ============================================================
# State Management (6-day dedup per commodity type)
# ============================================================

def _load_state() -> dict:
    default = {"analyses": []}
    if not os.path.isfile(_STATE_PATH):
        return default
    lock_fd = None
    try:
        lock_fd = open(_STATE_PATH + ".lock", "a")
        fcntl.flock(lock_fd, fcntl.LOCK_SH)
        with open(_STATE_PATH, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return default
        return data
    except (json.JSONDecodeError, OSError):
        return default
    finally:
        if lock_fd:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
            except Exception:
                pass


def _save_state(state: dict) -> None:
    lock_fd = None
    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
        lock_fd = open(_STATE_PATH + ".lock", "a")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        tmp = _STATE_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp, _STATE_PATH)
    except Exception as e:
        logger.error(f"Failed to save commodity research state: {e}")
    finally:
        if lock_fd:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
            except Exception:
                pass


def _already_analyzed(commodity_type: str, state: dict, days: int = 6) -> bool:
    """Check if we've already analyzed this commodity type within `days` days."""
    cutoff = datetime.now(_NY) - timedelta(days=days)
    for entry in state.get("analyses", []):
        if entry.get("commodity_type") == commodity_type:
            ts = entry.get("analyzed_at", "")
            if ts:
                try:
                    dt = datetime.fromisoformat(ts)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=_NY)
                    if dt > cutoff:
                        return True
                except (ValueError, TypeError):
                    pass
    return False


def get_stale_commodities(days: int = 7) -> list[str]:
    """Return commodity types not analyzed within `days` days.

    S122 4D: Used by daily staleness check and monitoring dashboards.
    """
    state = _load_state()
    return [
        ctype for ctype in COMMODITY_RESEARCH_MAP
        if not _already_analyzed(ctype, state, days=days)
    ]


def _record_analysis(commodity_type: str, findings_count: int, state: dict) -> None:
    """Record that we analyzed a commodity type."""
    state.setdefault("analyses", [])
    state["analyses"].append({
        "commodity_type": commodity_type,
        "analyzed_at": datetime.now(_NY).isoformat(),
        "findings_count": findings_count,
    })
    # Prune: keep last 100, remove entries >30 days
    cutoff = datetime.now(_NY) - timedelta(days=30)
    pruned = []
    for entry in state["analyses"]:
        ts = entry.get("analyzed_at", "")
        if ts:
            try:
                dt = datetime.fromisoformat(ts)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=_NY)
                if dt < cutoff:
                    continue
            except (ValueError, TypeError):
                continue
        pruned.append(entry)
    if len(pruned) > 100:
        pruned = pruned[-100:]
    state["analyses"] = pruned


# ============================================================
# Delivery
# ============================================================

async def _deliver_commodity_intel(
    context,
    commodity_type: str,
    analysis: str,
    urls: list[str],
) -> None:
    """Deliver: Telegram + knowledge commit + bridge write."""
    now = datetime.now(_NY)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    commodity_label = commodity_type.replace("_", " ").title()

    # 1. Telegram
    try:
        from agent.integrations.scheduled_tasks import _get_chat_ids, _safe_send
        chat_ids = _get_chat_ids()
        if chat_ids:
            msg = (
                f"COMMODITY RESEARCH: {commodity_label}\n"
                f"{now.strftime('%I:%M %p ET')}\n\n"
                f"{analysis}"
            )
            if len(msg) > 4000:
                msg = msg[:3997] + "..."
            for cid in chat_ids:
                await _safe_send(context.bot, cid, text=msg)
    except Exception as e:
        logger.warning(f"Failed to send commodity intel to Telegram: {e}")

    # 2. Knowledge base
    try:
        from agent.tools.knowledge_tools import handle_save_knowledge
        title = (
            f"Commodity: {commodity_label} Supply Balance "
            f"— {now.strftime('%b %d %Y')}"
        )
        source_urls = ", ".join(urls[:3]) if urls else "DuckDuckGo research"
        await handle_save_knowledge({
            "title": title,
            "content": analysis,
            "category": "trading_market_analysis",
            "tags": ["commodity_research", commodity_type, now.strftime("%Y-%m-%d")],
            "source": source_urls,
        })
    except Exception as e:
        logger.warning(f"Failed to save commodity intel to knowledge: {e}")

    # 3. Bridge write
    try:
        os.makedirs(BRIDGE_DIR, exist_ok=True)
        filename = f"{timestamp}_commodity_{commodity_type}.json"
        entry = {
            "id": f"commodity_{timestamp}_{commodity_type}",
            "type": "commodity_research",
            "commodity_type": commodity_type,
            "created": now.isoformat(),
            "content": analysis,
            "urls": urls[:5],
        }
        file_path = os.path.join(BRIDGE_DIR, filename)
        tmp = file_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(entry, f, indent=2, default=str)
        os.replace(tmp, file_path)
    except Exception as e:
        logger.warning(f"Failed to write commodity intel to bridge: {e}")


# ============================================================
# Public Entry Points
# ============================================================

async def escalate(context, analysis_text: str, results: list[dict]) -> None:
    """Escalation entry point — called from news_intel.py topic monitor.

    Args:
        context: Telegram bot context (for sending messages)
        analysis_text: Sonnet headline analysis from topic monitor
        results: Raw DuckDuckGo search results list
    """
    try:
        logger.info("Commodity research escalation triggered")

        # 1. Classify commodity type
        commodity_type = classify_commodity(analysis_text)
        if not commodity_type:
            logger.debug("Could not classify commodity type from analysis text")
            return

        logger.info(f"Commodity: {commodity_type}")

        # 2. Dedup check
        state = _load_state()
        if _already_analyzed(commodity_type, state):
            logger.debug(f"Already analyzed {commodity_type} recently, skipping")
            return

        # 3. Gather research (A — multi-source)
        research_results = await _gather_commodity_research(commodity_type)

        # Merge original results from topic monitor (pre-filtered by news_intel)
        # S100: defense-in-depth staleness check on incoming results
        seen_urls = set(r.get("url", "") for r in research_results if r.get("url"))
        try:
            from agent.integrations.news_update import _url_is_stale, _snippet_is_stale
            for r in results:
                url = r.get("url", r.get("href", ""))
                title = r.get("title", "")
                snippet = r.get("snippet", r.get("body", ""))
                if url and url not in seen_urls and not _url_is_stale(url) and not _snippet_is_stale(title, snippet):
                    research_results.append({
                        "title": title,
                        "snippet": snippet,
                        "url": url,
                        "query_type": "trigger",
                    })
                    seen_urls.add(url)
        except ImportError:
            for r in results:
                url = r.get("url", r.get("href", ""))
                if url and url not in seen_urls:
                    research_results.append({
                        "title": r.get("title", ""),
                        "snippet": r.get("snippet", r.get("body", "")),
                        "url": url,
                        "query_type": "trigger",
                    })
                    seen_urls.add(url)

        if not research_results:
            logger.debug("No research results for commodity escalation")
            return

        # 4. Recall supply balance history (B — memory recall)
        supply_history = _recall_supply_balance_history(commodity_type)

        # 5. Validate data sources (C)
        validity_assessment = _validate_data_sources(commodity_type, research_results)

        # 6. Generate supply balance analysis (D — scenario analysis)
        analysis = await _generate_supply_balance_analysis(
            commodity_type=commodity_type,
            research_results=research_results,
            supply_history=supply_history,
            validity_assessment=validity_assessment,
        )

        if not analysis:
            logger.debug("No supply balance analysis generated")
            return

        # 7. Deliver
        urls = [r.get("url", "") for r in research_results if r.get("url")]
        await _deliver_commodity_intel(context, commodity_type, analysis, urls)

        # 8. Record in state
        _record_analysis(commodity_type, len(research_results), state)
        _save_state(state)

        logger.info(
            f"Commodity escalation complete: {commodity_type}, "
            f"{len(research_results)} findings"
        )

    except Exception as e:
        logger.error(f"Commodity escalation failed: {e}", exc_info=True)


async def run_weekly_commodity_review(context) -> dict | None:
    """Scheduled callback — Monday 6:30 AM ET, comprehensive supply review.

    Iterates through all 4 commodity types. Max 4 per run (one per type).
    Time guard: Mon-Fri 5-10 AM ET (Monday primary, other days for catch-up).
    """
    if not config.news_intel_enabled:
        return

    now = datetime.now(_NY)

    # Time guard: weekdays, 5-10 AM ET
    if now.weekday() >= 5:
        return
    if now.hour < 5 or now.hour >= 10:
        return

    logger.info("Starting weekly commodity supply review...")

    try:
        state = _load_state()
        reviewed_count = 0
        # CM-1 (S211): counts let the scheduler tell "nothing was due" from a silent failure.
        skipped_recent = 0
        failed = 0

        _commodity_order = list(COMMODITY_RESEARCH_MAP.keys())
        random.shuffle(_commodity_order)
        for commodity_type in _commodity_order:
            if _already_analyzed(commodity_type, state):
                logger.debug(f"Already analyzed {commodity_type} recently, skipping")
                skipped_recent += 1
                continue

            commodity_label = commodity_type.replace("_", " ").title()
            logger.info(f"Commodity review: {commodity_label}")

            # A) Gather research
            research_results = await _gather_commodity_research(commodity_type)
            if not research_results:
                # S186 W1: visibility on upstream-empty (S185-B G17)
                logger.info(f"Commodity review skipping {commodity_type}: 0 research results (web_search budget? upstream issue?)")
                failed += 1
                continue

            # B) Recall supply balance history
            supply_history = _recall_supply_balance_history(commodity_type)

            # C) Validate data sources
            validity_assessment = _validate_data_sources(
                commodity_type, research_results
            )

            # D) Generate supply balance analysis
            analysis = await _generate_supply_balance_analysis(
                commodity_type=commodity_type,
                research_results=research_results,
                supply_history=supply_history,
                validity_assessment=validity_assessment,
            )

            if not analysis:
                # S186 W1: surface Anthropic failures
                logger.warning(f"Commodity review: no supply-balance analysis generated for {commodity_type} (Anthropic failure?)")
                failed += 1
                continue

            urls = [r.get("url", "") for r in research_results if r.get("url")]
            await _deliver_commodity_intel(context, commodity_type, analysis, urls)

            _record_analysis(commodity_type, len(research_results), state)
            _save_state(state)

            reviewed_count += 1
            logger.info(
                f"Commodity review complete for {commodity_label}: "
                f"{len(research_results)} findings"
            )

            await asyncio.sleep(2)

        if reviewed_count == 0:
            logger.debug("No commodity types needed review this cycle")
        else:
            logger.info(f"Weekly commodity review done: {reviewed_count} types reviewed")
        return {"reviewed": reviewed_count, "skipped_recent": skipped_recent, "failed": failed}

    except Exception as e:
        logger.error(f"Weekly commodity review failed: {e}", exc_info=True)
        return {"reviewed": 0, "skipped_recent": 0, "failed": 1, "error": True}
