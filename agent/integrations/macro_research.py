"""
OpenClaw/Moltbot — Macro Research Pipeline (Session 90)

Develops predictive views on upcoming economic data releases via:
  A) Multi-source research (obvious + niche data gathering)
  B) Memory recall (compare new data with prior cycle findings)
  C) Data validity assessment (gov shutdown, collection issues, political bias)
  D) Scenario analysis (worst / bad / base / good / best)

Fires 3-5 days before high-importance economic events, gathers intelligence,
commits to knowledge base, and produces scenario analysis before positioning.
"""

import asyncio
import fcntl
import json
import logging
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import anthropic

from agent.config import config
from agent.prompts.system_prompt import cached_system_block
from agent.rate_limiter import budget_tracker, BudgetExhaustedError
from agent.security import sanitize_ingested_content

logger = logging.getLogger("openclaw.macro_research")

_NY = ZoneInfo("America/New_York")

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "intelligence",
)
RESEARCH_STATE_PATH = os.path.join(_DATA_DIR, "macro_research_state.json")

BRIDGE_DIR = os.path.join(config.tradingbot_dir, "moltbot_knowledge", "intelligence")

_cached_client: anthropic.Anthropic | None = None
_client_lock = __import__("threading").Lock()


# ============================================================
# CC-CLIO-FD-4: Rate Regime Classification
# ============================================================

def _fd_rate_regime() -> dict:
    """Classify current rate environment: TIGHTENING / EASING / PAUSED / UNKNOWN.

    Uses FD API historical interest rates to determine trajectory.
    """
    if os.environ.get("CLIO_FD_MACRO_RATES", "0") != "1":
        return {"regime": "UNKNOWN", "enabled": False}
    try:
        from agent.integrations.fd_client import get_fd_client
        client = get_fd_client()

        end = datetime.now(_NY).strftime("%Y-%m-%d")
        start = (datetime.now(_NY) - timedelta(days=730)).strftime("%Y-%m-%d")

        history = client.get_interest_rates("FED", start_date=start, end_date=end)
        if not history or (isinstance(history, dict) and "_error" in history):
            return {"regime": "UNKNOWN", "error": "No rate data"}

        rates = history if isinstance(history, list) else []
        if len(rates) < 2:
            return {"regime": "UNKNOWN", "error": "Insufficient data"}

        # BC#113: ASCENDING sort (oldest first) — use [-1] for most recent
        current_rate = float(rates[-1].get("rate", 0))

        # H-3 fix: Find last rate that DIFFERS from current (not just adjacent entry)
        # This correctly detects PAUSED even when many entries share the same rate
        prev_rate = current_rate  # default: same as current = PAUSED
        for entry in reversed(rates[:-1]):
            r = float(entry.get("rate", 0))
            if r != current_rate:
                prev_rate = r
                break

        if current_rate > prev_rate:
            regime = "TIGHTENING"
        elif current_rate < prev_rate:
            regime = "EASING"
        else:
            regime = "PAUSED"

        logger.debug("CC-CLIO-FD-4: Rate regime=%s (%.2f%% -> %.2f%%)", regime, prev_rate, current_rate)
        return {
            "regime": regime,
            "current_rate": current_rate,
            "prev_rate": prev_rate,
            "direction": "UP" if current_rate > prev_rate else "DOWN" if current_rate < prev_rate else "FLAT",
            "enabled": True,
        }
    except Exception as e:
        logger.warning("CC-CLIO-FD-4: Rate regime failed: %s", e)
        return {"regime": "UNKNOWN", "error": str(e)}


def _get_client() -> anthropic.Anthropic:
    global _cached_client
    if _cached_client is None:
        with _client_lock:
            if _cached_client is None:
                _cached_client = anthropic.Anthropic(**config.anthropic_client_kwargs(timeout=120))
    return _cached_client


# ============================================================
# Event Type Taxonomy + Research Query Maps
# ============================================================

# Each event type maps to:
#   - keywords: TE calendar event name patterns that match this type
#   - related_types: interconnected data types (jobs ↔ spending ↔ inflation)
#   - obvious_queries: straightforward data searches
#   - niche_queries: nuanced, non-obvious searches that reveal leading indicators
#   - validity_queries: data collection quality checks

EVENT_RESEARCH_MAP: dict[str, dict] = {
    "employment": {
        "keywords": ["nonfarm payroll", "employment", "jobs", "unemployment", "jobless claims",
                      "adp employment", "jolts", "labor"],
        "related_types": ["consumer_spending", "inflation", "manufacturing"],
        "obvious_queries": [
            "USA nonfarm payrolls forecast consensus",
            "USA unemployment rate current trend",
            "USA weekly jobless claims trend",
            "ADP private employment report latest",
        ],
        "niche_queries": [
            "USA major company layoffs announced last 90 days",
            "USA small business hiring intentions survey",
            "USA gig economy workforce participation trend",
            "USA homelessness rate increase by state",
            "USA state unemployment offices wait times increasing",
            "USA temp staffing industry revenue trend",
            "USA WARN Act layoff notices filed last 60 days",
            "Indeed job postings index USA trend",
        ],
        "validity_queries": [
            "BLS data collection methodology changes",
            "USA government shutdown affecting data collection",
            "BLS seasonal adjustment controversy",
        ],
    },
    "inflation": {
        "keywords": ["cpi", "pce", "inflation", "consumer price", "producer price", "ppi"],
        "related_types": ["employment", "consumer_spending", "housing"],
        "obvious_queries": [
            "USA CPI forecast consensus next release",
            "USA core CPI trend last 6 months",
            "USA PCE price index latest",
            "USA PPI trend wholesale prices",
        ],
        "niche_queries": [
            "USA grocery prices trend last 90 days",
            "USA rent prices trend major cities",
            "USA used car prices Manheim index latest",
            "USA health insurance premiums increase",
            "USA egg milk bread price tracking",
            "USA shipping container rates trend",
            "USA restaurant menu prices increase",
            "gasoline prices average USA trend",
            "USA owners equivalent rent methodology debate",
        ],
        "validity_queries": [
            "CPI methodology criticism basket composition",
            "USA government shutdown BLS data delay",
            "inflation data political manipulation concerns",
        ],
    },
    "gdp": {
        "keywords": ["gdp", "gross domestic product", "economic growth"],
        "related_types": ["consumer_spending", "manufacturing", "employment"],
        "obvious_queries": [
            "USA GDP growth forecast consensus",
            "USA GDP nowcast Atlanta Fed GDPNow",
            "USA GDP components breakdown latest",
        ],
        "niche_queries": [
            "Atlanta Fed GDPNow vs actual track record",
            "USA inventory buildup concern GDP",
            "USA net exports contribution GDP trend",
            "USA government spending contribution GDP",
            "USA residential investment GDP component",
            "USA intellectual property investment trend GDP",
        ],
        "validity_queries": [
            "GDP revision history patterns USA",
            "BEA data collection issues",
        ],
    },
    "consumer_spending": {
        "keywords": ["retail sales", "consumer spending", "consumer confidence",
                      "consumer sentiment", "personal spending", "personal income"],
        "related_types": ["employment", "inflation", "housing"],
        "obvious_queries": [
            "USA retail sales forecast consensus",
            "USA consumer confidence index trend",
            "Michigan consumer sentiment latest",
        ],
        "niche_queries": [
            "USA credit card delinquency rate trend",
            "USA buy now pay later default rates",
            "USA savings rate personal trend",
            "Walmart Target earnings consumer spending signals",
            "USA auto loan delinquency rate increasing",
            "USA consumer discretionary vs staples spending ratio",
            "USA restaurant spending vs grocery spending trend",
        ],
        "validity_queries": [
            "retail sales seasonal adjustment issues",
            "consumer confidence survey methodology changes",
        ],
    },
    "manufacturing": {
        "keywords": ["ism manufacturing", "industrial production", "manufacturing",
                      "pmi", "factory orders", "durable goods"],
        "related_types": ["employment", "gdp", "trade"],
        "obvious_queries": [
            "USA ISM manufacturing PMI forecast consensus",
            "USA industrial production trend",
            "USA durable goods orders forecast",
        ],
        "niche_queries": [
            "USA reshoring manufacturing announcements",
            "USA semiconductor fab construction progress",
            "USA auto production inventory levels",
            "USA steel production capacity utilization",
            "regional Fed manufacturing surveys latest",
            "USA new manufacturing plant openings closings",
        ],
        "validity_queries": [
            "ISM survey response rate declining",
            "PMI diffusion index interpretation limitations",
        ],
    },
    "housing": {
        "keywords": ["housing starts", "building permits", "existing home sales",
                      "new home sales", "home price", "case-shiller", "pending home"],
        "related_types": ["inflation", "consumer_spending"],
        "obvious_queries": [
            "USA housing starts forecast consensus",
            "USA existing home sales trend",
            "USA mortgage rates 30 year current",
        ],
        "niche_queries": [
            "USA housing inventory months supply trend",
            "USA mortgage application volume trend",
            "USA homebuilder sentiment NAHB index",
            "USA rent vs buy affordability ratio",
            "USA institutional investor home purchases trend",
            "USA housing permits by region Sun Belt vs Northeast",
        ],
        "validity_queries": [
            "housing data seasonal adjustment winter",
            "NAR methodology changes existing home sales",
        ],
    },
    "fed_policy": {
        "keywords": ["fomc", "federal reserve", "interest rate", "fed funds",
                      "fed minutes", "fed beige book"],
        "related_types": ["inflation", "employment", "gdp"],
        "obvious_queries": [
            "FOMC meeting rate decision expectations",
            "CME FedWatch tool rate probabilities",
            "Federal Reserve dot plot latest",
        ],
        "niche_queries": [
            "Fed officials speeches this week hawkish dovish",
            "Fed balance sheet runoff QT pace",
            "reverse repo facility usage trend",
            "bank reserve levels adequate scarce",
            "Fed regional presidents dissent voting pattern",
            "Treasury general account balance trend",
        ],
        "validity_queries": [
            "Fed communication blackout period dates",
            "FOMC minutes release delay issues",
        ],
    },
    "trade": {
        "keywords": ["trade balance", "trade deficit", "imports", "exports"],
        "related_types": ["manufacturing", "gdp"],
        "obvious_queries": [
            "USA trade deficit forecast consensus",
            "USA tariff impact trade data latest",
            "USA China trade volume trend",
        ],
        "niche_queries": [
            "USA port container traffic volume trend",
            "USA agricultural exports to China latest",
            "USA semiconductor equipment export controls impact",
            "USA LNG exports to Europe volume",
            "USA services trade surplus trend",
        ],
        "validity_queries": [
            "trade data revision patterns Census Bureau",
        ],
    },
}

# Map from normalized TE event name patterns to our taxonomy
def classify_event(event_name: str) -> str | None:
    """Classify a TE calendar event into our research taxonomy.

    Returns event type key (e.g., "employment") or None if unrecognized.
    """
    name_lower = event_name.lower()
    for event_type, config_map in EVENT_RESEARCH_MAP.items():
        for keyword in config_map["keywords"]:
            if keyword in name_lower:
                return event_type
    return None


# ============================================================
# Research State Management
# ============================================================

def _load_research_state() -> dict:
    """Load macro research state."""
    default = {
        "researched_events": [],  # [{event_name, event_type, researched_at, findings_summary}]
        "scenario_analyses": [],  # [{event_name, event_type, created_at, scenarios}]
    }
    if not os.path.isfile(RESEARCH_STATE_PATH):
        return default
    lock_fd = None
    try:
        lock_fd = open(RESEARCH_STATE_PATH + ".lock", "a")
        fcntl.flock(lock_fd, fcntl.LOCK_SH)
        with open(RESEARCH_STATE_PATH, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return default
        for key in default:
            if key not in data:
                data[key] = default[key]
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load research state: {e}")
        return default
    finally:
        if lock_fd:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
            except Exception:
                pass


def _save_research_state(state: dict) -> None:
    """Atomic write research state."""
    lock_fd = None
    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
        lock_fd = open(RESEARCH_STATE_PATH + ".lock", "a")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        tmp = RESEARCH_STATE_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp, RESEARCH_STATE_PATH)
    except Exception as e:
        logger.error(f"Failed to save research state: {e}")
    finally:
        if lock_fd:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
            except Exception:
                pass


def _prune_research_state(state: dict) -> dict:
    """Remove entries older than 30 days, cap at 100."""
    cutoff = datetime.now(_NY) - timedelta(days=30)
    for key in ("researched_events", "scenario_analyses"):
        entries = state.get(key, [])
        pruned = []
        for entry in entries:
            ts = entry.get("researched_at", entry.get("created_at", ""))
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
        state[key] = pruned
    return state


def _already_researched(event_name: str, event_type: str, state: dict) -> bool:
    """Check if we've already researched this event in the current cycle (last 5 days)."""
    cutoff = datetime.now(_NY) - timedelta(days=5)
    for entry in state.get("researched_events", []):
        if entry.get("event_type") == event_type:
            ts = entry.get("researched_at", "")
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


# ============================================================
# Memory Recall — Prior Findings
# ============================================================

def _recall_prior_research(event_type: str) -> list[dict]:
    """Search knowledge base for prior research on this event type.

    Returns recent entries (last 60 days) tagged with this event type,
    sorted by date descending. Used for trend comparison.
    """
    try:
        from agent.tools.knowledge_tools import load_all_entries
        all_entries = load_all_entries()
    except Exception as e:
        logger.debug(f"Failed to load entries for prior research: {e}")
        return []

    cutoff = datetime.now(_NY) - timedelta(days=60)
    relevant = []

    # Search for entries with matching tags or title patterns
    event_config = EVENT_RESEARCH_MAP.get(event_type, {})
    keywords = event_config.get("keywords", [])

    for entry in all_entries:
        # Check tags — but still filter by date
        tags = entry.get("tags", [])
        if isinstance(tags, list) and any(
            "macro_research" in str(t) or event_type in str(t) for t in tags
        ):
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
            continue

        # Check title for event keywords
        title = entry.get("title", "").lower()
        if any(kw in title for kw in keywords):
            # Filter by date
            created = entry.get("created", "")
            if created:
                try:
                    dt = datetime.fromisoformat(created)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=_NY)
                    if dt >= cutoff:
                        relevant.append(entry)
                except (ValueError, TypeError):
                    pass

    # Sort by created date, newest first
    def _sort_key(e):
        try:
            dt = datetime.fromisoformat(e.get("created", "2020-01-01"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_NY)
            return dt
        except (ValueError, TypeError):
            return datetime(2020, 1, 1, tzinfo=_NY)

    relevant.sort(key=_sort_key, reverse=True)
    return relevant[:10]  # Cap at 10 most recent


def _format_prior_research(entries: list[dict]) -> str:
    """Format prior research entries for prompt injection."""
    if not entries:
        return "No prior research found for this event type."

    lines = []
    for entry in entries[:5]:
        title = entry.get("title", "")
        created = entry.get("created", "")[:10]  # Just date
        content = entry.get("content", "")[:300] if "content" in entry else ""
        # Try to read actual file content
        file_path_rel = entry.get("file_path", "")
        if file_path_rel and not content:
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
# Data Validity Assessment
# ============================================================

async def _check_data_validity(event_type: str) -> str | None:
    """Search for data collection issues that could affect reliability.

    Checks for: government shutdowns, methodology changes, political bias indicators.
    Returns brief validity assessment or None.
    """
    event_config = EVENT_RESEARCH_MAP.get(event_type, {})
    validity_queries = event_config.get("validity_queries", [])
    if not validity_queries:
        return None

    # Use first validity query only (budget conservation)
    query = validity_queries[0]
    now = datetime.now(_NY)
    query = f"{query} {now.strftime('%B %d %Y')}"

    try:
        allowed, msg = await budget_tracker.check_budget("report_web_search")
        if not allowed:
            return None

        from agent.integrations.web_search import search_web
        results = await search_web(query, max_results=2)
        await budget_tracker.record_usage("report_web_search")

        if not results:
            return None

        snippets = []
        for r in (results or []):
            title = r.get("title", "")
            snippet = r.get("snippet", r.get("body", ""))
            if title:
                snippets.append(f"{title}: {snippet}")

        if not snippets:
            return None

        return sanitize_ingested_content("\n".join(snippets[:2]), source="ddg_search")

    except Exception as e:
        logger.debug(f"Validity check failed for {event_type}: {e}")
        return None


# ============================================================
# Research Pipeline
# ============================================================

async def _gather_research(event_type: str, max_queries: int = 6) -> list[dict]:
    """Execute research queries for an event type.

    Searches both obvious and niche queries. Returns list of {title, snippet, url, query_type}.
    """
    event_config = EVENT_RESEARCH_MAP.get(event_type, {})
    obvious = event_config.get("obvious_queries", [])
    niche = event_config.get("niche_queries", [])

    # Mix: 2 obvious + 4 niche (niche has more alpha)
    queries_to_run = []
    for q in obvious[:2]:
        queries_to_run.append((q, "obvious"))
    for q in niche[:max_queries - 2]:
        queries_to_run.append((q, "niche"))

    now = datetime.now(_NY)
    date_suffix = now.strftime("%B %d %Y")
    results_all = []

    for query, query_type in queries_to_run:
        try:
            allowed, msg = await budget_tracker.check_budget("report_web_search")
            if not allowed:
                break

            from agent.integrations.web_search import search_web
            full_query = f"{query} {date_suffix}"
            results = await search_web(full_query, max_results=3)
            await budget_tracker.record_usage("report_web_search")

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
            logger.debug(f"Budget exhausted during macro research for {event_type}")
            break
        except Exception as e:
            logger.warning(f"Research query failed: {query}: {e}")

        await asyncio.sleep(0.5)

    # Filter stale results (defense-in-depth — same pattern as all other deep-analysis modules)
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
# Scenario Analysis (Opus-powered)
# ============================================================

_SCENARIO_PROMPT = """You are a macro-economic analyst developing pre-positioning intelligence for an upcoming data release.

Your job: analyze the gathered research data and produce a SCENARIO ANALYSIS with 5 scenarios.

FORMAT:
EVENT: {event name} — Expected: {date}
DATA VALIDITY: {any collection issues, gov shutdown effects, or methodology concerns — or "No issues detected"}

PRIOR CYCLE COMPARISON:
{What has changed since the last time this data was released? Identify trends.}

SCENARIOS:
WORST: {specific number or range} — {what this means, market reaction, positioning implication}
BAD: {specific number or range} — {what this means, market reaction, positioning implication}
BASE: {specific number or range} — {what this means, market reaction, positioning implication}
GOOD: {specific number or range} — {what this means, market reaction, positioning implication}
BEST: {specific number or range} — {what this means, market reaction, positioning implication}

INTERCONNECTIONS:
{How does this data relate to other upcoming releases? e.g., strong jobs → higher inflation → hawkish Fed}

KEY SIGNAL: {The single most important data point from the research that informs our base case}

CONFIDENCE: {LOW/MEDIUM/HIGH} — {brief justification}

RULES:
- Every scenario MUST include a specific number or range
- NEVER use vague language: "monitor", "watch for", "keep an eye on", "remains to be seen"
- State market implications directly: "SPY likely -1.5%", "10Y yield +15bp", "DXY +0.5%"
- If political bias could affect the data, state it explicitly with discount factor
- Base case should reflect consensus + your niche research adjustment
- WORST and BEST should be tail scenarios (5-10% probability each)
- Write in plain text only, no markdown"""


async def _generate_scenario_analysis(
    event_name: str,
    event_type: str,
    event_date: str,
    research_results: list[dict],
    prior_research: str,
    validity_note: str | None,
) -> str | None:
    """Generate scenario analysis using Claude Opus.

    Uses macro_research budget (separate from news_intel).
    """
    try:
        allowed, msg = await budget_tracker.check_budget("macro_research")
        if not allowed:
            logger.debug("macro_research budget exhausted")
            return None
    except BudgetExhaustedError:
        return None

    # Format research results
    research_text = ""
    for i, r in enumerate(research_results[:12], 1):
        qt = r.get("query_type", "")
        research_text += (
            f"\n[{i}] ({qt}) {r['title']}\n"
            f"{r.get('snippet', '')}\n"
            f"URL: {r.get('url', '')}\n"
        )
    research_text = sanitize_ingested_content(research_text, source="ddg_search")

    # Format event config for related types
    event_config = EVENT_RESEARCH_MAP.get(event_type, {})
    related = event_config.get("related_types", [])
    related_str = ", ".join(related) if related else "None identified"

    validity = validity_note or "No issues detected"

    # CC-CLIO-FD-4: Inject rate regime context (run_in_executor to avoid blocking)
    rate_regime_section = ""
    try:
        _fd_ok, _ = await budget_tracker.check_budget("financial_datasets")
        if _fd_ok:
            loop = asyncio.get_running_loop()
            rate_data = await loop.run_in_executor(None, _fd_rate_regime)
            if rate_data.get("enabled"):
                await budget_tracker.record_usage("financial_datasets")
        else:
            rate_data = {"regime": "UNKNOWN"}
    except Exception as e:
        logger.debug("CC-CLIO-FD-4: Rate regime fetch failed: %s", e)
        rate_data = {"regime": "UNKNOWN"}
    if rate_data.get("enabled"):
        regime = rate_data.get("regime", "UNKNOWN")
        cur = rate_data.get("current_rate", "?")
        prev = rate_data.get("prev_rate", "?")
        rate_regime_section = (
            f"\nRATE REGIME (FD API): {regime} — Current Fed Funds: {cur}% (prev: {prev}%)\n"
        )

    user_message = (
        f"EVENT: {event_name}\n"
        f"EXPECTED DATE: {event_date}\n"
        f"EVENT TYPE: {event_type}\n"
        f"RELATED DATA TYPES: {related_str}\n\n"
        + (f"{rate_regime_section}\n" if rate_regime_section else "")
        + f"DATA VALIDITY ASSESSMENT:\n{validity}\n\n"
        f"PRIOR RESEARCH (from previous cycles):\n{prior_research}\n\n"
        f"NEW RESEARCH FINDINGS:\n{research_text}\n\n"
        f"Generate a scenario analysis with 5 scenarios (WORST/BAD/BASE/GOOD/BEST) "
        f"including specific numbers, market reactions, and interconnections."
    )

    try:
        client = _get_client()
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, lambda: client.messages.create(
            model=config.default_model,  # Opus for deep analysis
            max_tokens=2500,
            system=cached_system_block(_SCENARIO_PROMPT),
            messages=[{"role": "user", "content": user_message}],
        ))

        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

        result = text.strip() if text.strip() else None
        if result:
            await budget_tracker.record_usage("macro_research")
        return result

    except Exception as e:
        logger.error(f"Scenario analysis generation failed: {e}", exc_info=True)
        return None


# ============================================================
# Delivery
# ============================================================

async def _deliver_research(
    context,
    event_name: str,
    event_type: str,
    analysis: str,
    urls: list[str],
) -> None:
    """Deliver research: Telegram + knowledge commit + bridge write."""
    now = datetime.now(_NY)
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    # 1. Telegram notification
    try:
        from agent.integrations.scheduled_tasks import _get_chat_ids, _safe_send
        chat_ids = _get_chat_ids()
        if chat_ids:
            msg = f"MACRO RESEARCH: {event_name}\n{now.strftime('%I:%M %p ET')}\n\n{analysis}"
            if len(msg) > 4000:
                msg = msg[:3997] + "..."
            for cid in chat_ids:
                await _safe_send(context.bot, cid, text=msg)
    except Exception as e:
        logger.warning(f"Failed to send macro research to Telegram: {e}")

    # 2. Knowledge base commit
    try:
        from agent.tools.knowledge_tools import handle_save_knowledge
        title = f"Macro Research: {event_name} — {now.strftime('%b %d %Y')}"
        source_urls = ", ".join(urls[:3]) if urls else "DuckDuckGo research"
        await handle_save_knowledge({
            "title": title,
            "content": analysis,
            "category": "trading_market_analysis",
            "tags": ["macro_research", event_type, now.strftime("%Y-%m-%d")],
            "source": source_urls,
        })
    except Exception as e:
        logger.warning(f"Failed to save macro research to knowledge: {e}")

    # 3. Bridge write
    try:
        os.makedirs(BRIDGE_DIR, exist_ok=True)
        filename = f"{timestamp}_macro_{event_type}.json"
        entry = {
            "id": f"macro_{timestamp}_{event_type}",
            "type": "macro_research",
            "event_name": event_name,
            "event_type": event_type,
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
        logger.warning(f"Failed to write macro research to bridge: {e}")


# ============================================================
# Scheduled Callback
# ============================================================

async def run_macro_research(context) -> None:
    """Scheduled callback — runs daily at 7:00 AM ET.

    Scans economic calendar for high-importance events in the next 5 days.
    For each qualifying event not yet researched this cycle:
    1. Gather research (obvious + niche queries)
    2. Recall prior findings from knowledge base
    3. Check data validity
    4. Generate scenario analysis (Opus)
    5. Deliver to Telegram + knowledge + bridge
    """
    if not config.news_intel_enabled:
        return

    now = datetime.now(_NY)

    # Time guard: weekdays, 6 AM - 10 AM ET
    if now.weekday() >= 5:
        return
    if now.hour < 6 or now.hour >= 10:
        return

    logger.info("Starting macro research pipeline...")

    try:
        from agent.integrations.econ_calendar import get_upcoming_events

        # Events in next 5 days, importance >= 3 (high only)
        upcoming = get_upcoming_events(hours=120)
        if not upcoming:
            # S186 W1: promote DEBUG→INFO so operator sees WHY this exits without state advance
            # (typically: econ_calendar stale because TE budget exhausted)
            logger.info("Macro research exiting: no upcoming events in next 5d (likely stale econ_calendar)")
            return

        high_importance = [e for e in upcoming if e.get("importance", 0) >= 3]
        if not high_importance:
            logger.info("Macro research exiting: no high-importance events in next 5d")
            return

        state = _load_research_state()
        state = _prune_research_state(state)

        researched_count = 0
        max_per_run = 2  # Cap at 2 events per run (budget conservation)

        for event in high_importance:
            if researched_count >= max_per_run:
                break

            event_name = event.get("name", "")
            event_type = classify_event(event_name)
            if not event_type:
                continue

            if _already_researched(event_name, event_type, state):
                logger.debug(f"Already researched {event_type} this cycle, skipping")
                continue

            event_date = event.get("datetime_et", "")
            logger.info(f"Macro research: {event_name} ({event_type}) — {event_date[:10]}")

            # A) Gather research (obvious + niche)
            research_results = await _gather_research(event_type, max_queries=6)
            if not research_results:
                # S186 W1: visibility on upstream-empty path
                logger.info(f"Macro research skipping {event_type}: 0 research results (web_search budget? upstream provider issue?)")
                continue

            # Defense-in-depth: filter stale at caller level too
            try:
                from agent.integrations.news_update import _url_is_stale, _snippet_is_stale
                research_results = [
                    r for r in research_results
                    if not _url_is_stale(r.get("url", ""))
                    and not _snippet_is_stale(r.get("title", ""), r.get("snippet", ""))
                ]
            except ImportError:
                pass

            if not research_results:
                logger.debug(f"All results stale for {event_type}")
                continue

            # B) Recall prior findings
            prior_entries = _recall_prior_research(event_type)
            prior_text = _format_prior_research(prior_entries)

            # C) Data validity check
            validity_note = await _check_data_validity(event_type)

            # D) Scenario analysis
            analysis = await _generate_scenario_analysis(
                event_name=event_name,
                event_type=event_type,
                event_date=event_date,
                research_results=research_results,
                prior_research=prior_text,
                validity_note=validity_note,
            )

            if not analysis:
                # S186 W1: surface Anthropic-call failures (the original silent-persistence root cause for many cycles)
                logger.warning(f"Macro research: no scenario analysis generated for {event_type} (Anthropic failure?)")
                continue

            # Extract URLs
            urls = [r.get("url", "") for r in research_results if r.get("url")]

            # Deliver
            await _deliver_research(context, event_name, event_type, analysis, urls)

            # Record in state
            state["researched_events"].append({
                "event_name": event_name,
                "event_type": event_type,
                "researched_at": now.isoformat(),
                "findings_count": len(research_results),
            })
            state["scenario_analyses"].append({
                "event_name": event_name,
                "event_type": event_type,
                "created_at": now.isoformat(),
                "scenarios_preview": analysis[:200],
            })
            _save_research_state(state)

            researched_count += 1
            logger.info(f"Macro research complete for {event_name}: {len(research_results)} findings")

            await asyncio.sleep(2)  # Space out events

        if researched_count == 0:
            logger.debug("No new events to research this cycle")

    except Exception as e:
        logger.error(f"Macro research pipeline failed: {e}", exc_info=True)
