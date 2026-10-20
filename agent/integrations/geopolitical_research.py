"""
OpenClaw/Moltbot — Geopolitical Risk Scoring (Session 91, Phase 2)

Deep-analysis module triggered by news_intel.py escalation when geopolitics
or international topics contain military/conflict/sanctions keywords.

Applies the A/B/C/D research framework:
  A) Multi-source research — obvious (Reuters, UN) + niche (shipping insurance,
     CDS spreads, diplomatic recall, tanker tracking)
  B) Memory recall — search knowledge base for historical analogs
  C) Source credibility — 4-tier system with discount factors
  D) Scenario analysis — 5-level escalation ladder (DE-ESCALATION / FROZEN /
     ESCALATION / EXPANSION / CONFRONTATION)
"""

import asyncio
import fcntl
import json
import logging
import os
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import anthropic

from agent.config import config
from agent.prompts.system_prompt import cached_system_block
from agent.rate_limiter import budget_tracker, BudgetExhaustedError
from agent.security import sanitize_ingested_content

logger = logging.getLogger("openclaw.geopolitical_research")

_NY = ZoneInfo("America/New_York")

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "intelligence",
)
RESEARCH_STATE_PATH = os.path.join(_DATA_DIR, "geopolitical_research_state.json")

BRIDGE_DIR = os.path.join(config.tradingbot_dir, "moltbot_knowledge", "intelligence")

_cached_client: anthropic.Anthropic | None = None
_client_lock = __import__("threading").Lock()


def _get_client() -> anthropic.Anthropic:
    global _cached_client
    if _cached_client is None:
        with _client_lock:
            if _cached_client is None:
                _cached_client = anthropic.Anthropic(**config.anthropic_client_kwargs(timeout=120))
    return _cached_client


# ============================================================
# Crisis Type Taxonomy + Research Query Maps
# ============================================================

GEOPOLITICAL_RESEARCH_MAP: dict[str, dict] = {
    "conflict_escalation": {
        "keywords": ["military", "strike", "attack", "troops", "invasion", "missile",
                      "blockade", "naval", "deployed", "airstrike", "bombardment",
                      "shelling", "offensive", "warship", "fighter jet"],
        "obvious_queries": [
            "{region} military conflict latest",
            "UN Security Council {region}",
            "NATO response {region}",
        ],
        "niche_queries": [
            "{region} shipping insurance rates war risk premium",
            "AIS ship tracking {chokepoint} vessel traffic",
            "{country} sovereign debt CDS spread",
            "{country} currency black market premium",
            "{country} ambassador recalled expelled",
            "{country} foreign nationals evacuation advisory",
        ],
        "validity_queries": [
            "{country} state media propaganda assessment",
            "{region} fog of war initial reports accuracy",
        ],
    },
    "sanctions_regime": {
        "keywords": ["sanctions", "embargo", "OFAC", "SDN", "asset freeze",
                      "export controls", "SWIFT", "blacklist", "designation"],
        "obvious_queries": [
            "OFAC sanctions latest designations",
            "{country} sanctions impact",
        ],
        "niche_queries": [
            "{country} sanctions oil exports tanker tracking",
            "{country} sanctions cryptocurrency evasion",
            "SWIFT disconnection {country}",
            "{country} gold reserves sanctions hedge",
        ],
        "validity_queries": [
            "sanctions enforcement effectiveness {country}",
        ],
    },
    "regime_change": {
        "keywords": ["coup", "election crisis", "protest", "revolution", "martial law",
                      "junta", "uprising", "overthrow", "deposed"],
        "obvious_queries": [
            "{country} political crisis latest",
            "{country} government stability",
        ],
        "niche_queries": [
            "{country} military loyalty",
            "{country} capital flight outflows",
            "{country} key commodity production disruption risk",
        ],
        "validity_queries": [],
    },
}


# ============================================================
# Source Credibility Tiers
# ============================================================

SOURCE_CREDIBILITY = {
    "tier_1_primary": {
        "domains": ["state.gov", "mod.gov", "nato.int", "un.org", "gov.uk",
                     "defense.gov", "europa.eu"],
        "discount": 0.0,
        "label": "Primary Government/Institutional",
    },
    "tier_2_wire": {
        "domains": ["reuters.com", "apnews.com", "afp.com", "bloomberg.com"],
        "discount": 0.1,
        "label": "Wire Services",
    },
    "tier_3_quality": {
        "domains": ["bbc.com", "bbc.co.uk", "aljazeera.com", "ft.com", "nytimes.com",
                     "washingtonpost.com", "economist.com", "foreignaffairs.com"],
        "discount": 0.2,
        "label": "Quality Reporting",
    },
    "tier_4_state_media": {
        "domains": ["rt.com", "presstv.ir", "globaltimes.cn", "tass.com",
                     "sputniknews.com", "xinhua.net"],
        "discount": 0.7,
        "label": "State Media",
    },
}


# ============================================================
# Country + Region Entity Lists
# ============================================================

_COUNTRIES = [
    "United States", "China", "Russia", "Iran", "Israel", "Saudi Arabia",
    "Turkey", "India", "Pakistan", "North Korea", "South Korea", "Japan",
    "Taiwan", "Ukraine", "Syria", "Iraq", "Afghanistan", "Yemen", "Libya",
    "Egypt", "Lebanon", "Jordan", "Qatar", "UAE", "Kuwait", "Bahrain",
    "Oman", "Venezuela", "Cuba", "Brazil", "Mexico", "Colombia",
    "Nigeria", "South Africa", "Ethiopia", "Somalia", "Sudan", "Kenya",
    "Myanmar", "Thailand", "Philippines", "Indonesia", "Vietnam",
    "Poland", "Germany", "France", "UK", "Italy", "Spain",
]

_REGIONS = [
    "Middle East", "Eastern Europe", "South China Sea", "Taiwan Strait",
    "Korean Peninsula", "Persian Gulf", "Gulf of Aden", "Red Sea",
    "Horn of Africa", "Sahel", "Balkans", "Central Asia", "South Asia",
    "Southeast Asia", "Baltic States", "Black Sea", "Arctic",
    "Strait of Hormuz", "Suez Canal", "Strait of Malacca",
]

_CHOKEPOINTS = {
    "Middle East": "Strait of Hormuz",
    "Red Sea": "Bab el-Mandeb",
    "Gulf of Aden": "Bab el-Mandeb",
    "Persian Gulf": "Strait of Hormuz",
    "South China Sea": "Strait of Malacca",
    "Taiwan Strait": "Taiwan Strait",
    "Suez Canal": "Suez Canal",
    "Southeast Asia": "Strait of Malacca",
}


# ============================================================
# Crisis Classification
# ============================================================

def classify_crisis(text: str) -> tuple[str | None, list[str], list[str]]:
    """Classify text into crisis type and extract entities.

    Returns (crisis_type, regions_mentioned, countries_mentioned).
    """
    text_lower = text.lower()

    # Find best matching crisis type by keyword count
    best_type = None
    best_count = 0
    for crisis_type, cfg in GEOPOLITICAL_RESEARCH_MAP.items():
        count = sum(1 for kw in cfg["keywords"] if re.search(r'\b' + re.escape(kw.lower()) + r'\b', text_lower))
        if count > best_count:
            best_count = count
            best_type = crisis_type

    if best_count == 0:
        return None, [], []

    # Extract countries
    countries = [c for c in _COUNTRIES if re.search(r'\b' + re.escape(c.lower()) + r'\b', text_lower)]

    # Extract regions
    regions = [r for r in _REGIONS if re.search(r'\b' + re.escape(r.lower()) + r'\b', text_lower)]

    # If no regions found but countries found, infer region
    if not regions and countries:
        _country_to_region = {
            "Ukraine": "Eastern Europe", "Russia": "Eastern Europe",
            "Poland": "Eastern Europe",
            "Israel": "Middle East", "Iran": "Middle East",
            "Saudi Arabia": "Middle East", "Yemen": "Middle East",
            "Iraq": "Middle East", "Syria": "Middle East",
            "Lebanon": "Middle East", "Jordan": "Middle East",
            "Qatar": "Middle East", "UAE": "Middle East",
            "Kuwait": "Middle East", "Bahrain": "Middle East",
            "Oman": "Middle East",
            "Taiwan": "Taiwan Strait", "China": "South China Sea",
            "North Korea": "Korean Peninsula", "South Korea": "Korean Peninsula",
            "Japan": "East Asia", "India": "South Asia",
            "Pakistan": "South Asia", "Afghanistan": "Central Asia",
            "Somalia": "Horn of Africa", "Ethiopia": "Horn of Africa",
            "Kenya": "East Africa", "Sudan": "Horn of Africa",
            "Nigeria": "West Africa", "South Africa": "Southern Africa",
            "Egypt": "Middle East", "Libya": "Middle East",
            "Myanmar": "Southeast Asia", "Philippines": "South China Sea",
            "Thailand": "Southeast Asia", "Indonesia": "Southeast Asia",
            "Vietnam": "Southeast Asia",
            "Germany": "Western Europe", "France": "Western Europe",
            "UK": "Western Europe", "Italy": "Western Europe",
            "Spain": "Western Europe",
            # S159 L-5: Americas
            "Venezuela": "South America", "Cuba": "Central America",
            "Brazil": "South America", "Mexico": "Central America",
            "Colombia": "South America",
            "United States": "North America",
            "Turkey": "Middle East",
        }
        for c in countries:
            r = _country_to_region.get(c)
            if r and r not in regions:
                regions.append(r)

    return best_type, regions, countries


# ============================================================
# Source Credibility Assessment
# ============================================================

def _assess_source_credibility(results: list[dict]) -> str:
    """Classify results by source credibility tier."""
    tier_counts: dict[str, list[str]] = {}

    for r in results:
        url = r.get("url", r.get("href", ""))
        if not url:
            continue
        domain = url.split("//")[-1].split("/")[0].lower().replace("www.", "")

        classified = False
        for tier_name, tier_info in SOURCE_CREDIBILITY.items():
            for d in tier_info["domains"]:
                if domain.endswith(d):
                    if tier_name not in tier_counts:
                        tier_counts[tier_name] = []
                    tier_counts[tier_name].append(d)
                    classified = True
                    break
            if classified:
                break

    if not tier_counts:
        return "Sources: Unclassified (no recognized domains). Fog of war risk: HIGH."

    parts = []
    has_state_media = False
    for tier_name, domains in sorted(tier_counts.items()):
        tier_info = SOURCE_CREDIBILITY[tier_name]
        discount = tier_info["discount"]
        unique_domains = list(set(domains))
        domain_str = ", ".join(unique_domains[:3])
        suffix = f" -- {int(discount * 100)}% discount" if discount > 0 else ""
        parts.append(f"{len(domains)} {tier_info['label']} ({domain_str}{suffix})")
        if tier_name == "tier_4_state_media":
            has_state_media = True

    fog_risk = "HIGH" if has_state_media and "tier_1_primary" not in tier_counts else (
        "MODERATE" if has_state_media or "tier_1_primary" not in tier_counts else "LOW"
    )

    return f"Sources: {'; '.join(parts)}. Fog of war risk: {fog_risk}."


# ============================================================
# Memory Recall — Historical Analogs
# ============================================================

def _recall_historical_analogs(regions: list[str], countries: list[str], crisis_type: str) -> str:
    """Search knowledge base for prior geopolitical crisis entries."""
    try:
        from agent.tools.knowledge_tools import load_all_entries
        all_entries = load_all_entries()
    except Exception as e:
        logger.debug(f"Failed to load entries for historical analogs: {e}")
        return "No historical analogs found (knowledge base unavailable)."

    cutoff = datetime.now(_NY) - timedelta(days=180)
    relevant = []

    search_terms = [r.lower() for r in regions] + [c.lower() for c in countries]
    crisis_keywords = GEOPOLITICAL_RESEARCH_MAP.get(crisis_type, {}).get("keywords", [])

    for entry in all_entries:
        # Check tags for geopolitical markers
        tags = entry.get("tags", [])
        tags_str = " ".join(str(t) for t in tags) if isinstance(tags, list) else ""
        title = entry.get("title", "").lower()
        combined = f"{tags_str} {title}".lower()

        # Match by region/country name in title or tags (word-boundary to avoid "us" in "discuss")
        has_entity = any(re.search(r'\b' + re.escape(term) + r'\b', combined) for term in search_terms)
        # Match by crisis keywords
        has_crisis = any(re.search(r'\b' + re.escape(kw.lower()) + r'\b', combined) for kw in crisis_keywords)

        if has_entity or has_crisis:
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
        return "No historical analogs found in knowledge base."

    # Sort newest first
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
        # Read file content for details
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
# Research Gathering
# ============================================================

async def _gather_geopolitical_research(
    crisis_type: str,
    regions: list[str],
    countries: list[str],
) -> list[dict]:
    """Execute research queries with entity substitution.

    Runs 2 obvious + 4 niche queries. Uses report_web_search budget.
    """
    crisis_config = GEOPOLITICAL_RESEARCH_MAP.get(crisis_type, {})
    obvious = crisis_config.get("obvious_queries", [])
    niche = crisis_config.get("niche_queries", [])

    # Build substitution values
    region = regions[0] if regions else "global conflict zone"
    country = countries[0] if countries else "affected nation"
    chokepoint = ""
    for r in regions:
        if r in _CHOKEPOINTS:
            chokepoint = _CHOKEPOINTS[r]
            break
    if not chokepoint:
        chokepoint = "major shipping lane"

    subs = {
        "region": region,
        "country": country,
        "chokepoint": chokepoint,
        "country1": countries[0] if len(countries) >= 1 else country,
        "country2": countries[1] if len(countries) >= 2 else "",
    }

    # Mix: 2 obvious + 4 niche
    queries_to_run = []
    for q in obvious[:2]:
        queries_to_run.append((q, "obvious"))
    for q in niche[:4]:
        queries_to_run.append((q, "niche"))

    now = datetime.now(_NY)
    date_suffix = now.strftime("%B %d %Y")
    results_all = []

    for query_template, query_type in queries_to_run:
        try:
            # Substitute placeholders
            query = query_template
            for key, val in subs.items():
                query = query.replace("{" + key + "}", val)

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
            logger.debug("Budget exhausted during geopolitical research")
            break
        except Exception as e:
            logger.warning(f"Geopolitical research query failed: {e}")

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
# Escalation Analysis (Opus-powered)
# ============================================================

_ESCALATION_PROMPT = """You are a geopolitical intelligence analyst producing escalation-ladder scenario analysis for a trading system.

Your job: analyze a developing crisis using gathered research, historical analogs, and source credibility assessment.

FORMAT:

CRISIS: {who, what, where, when — 2-3 sentences max}

HISTORICAL ANALOG: {closest prior crisis with specific market data — price moves, durations, asset impacts}

SOURCE ASSESSMENT: {credibility tier analysis from the provided assessment}

ESCALATION SCENARIOS:

DE-ESCALATION (X%): {specific conditions} — Duration: {days/weeks}
  Assets: {specific tickers + price impacts, e.g., "USO -3% as war premium unwinds", "LMT/RTX -5%"}

FROZEN (X%): {specific conditions, status quo} — Duration: {weeks/months}
  Assets: {tickers + ranges}

ESCALATION (X%): {specific conditions, next step up} — Duration: {days/weeks}
  Assets: {tickers + impacts}

EXPANSION (X%): {conditions, widening conflict/sanctions} — Duration: {weeks/months}
  Assets: {tickers + impacts}

CONFRONTATION (X%): {tail risk — direct great power involvement} — Duration: {months}
  Assets: {tickers + impacts}

(Probabilities must sum to approximately 100%)

CHOKEPOINT RISK: {shipping lanes, pipelines affected — specific volumes in barrels/day or TEU}

COMMODITY IMPACT: {oil, gas, metals — specific $ moves per scenario}

SAFE HAVEN FLOWS: {GLD, UUP, FXY, FXF, TLT — direction + magnitude}

KEY INTELLIGENCE GAP: {the single most important unknown that would change the analysis}

CONFIDENCE: {LOW/MEDIUM/HIGH} — {brief justification}

RULES:
- Every scenario MUST include specific tickers and price impacts
- NEVER use: "monitor closely", "fluid situation", "remains to be seen", "watch for", "keep an eye on"
- State impacts directly: "Brent +$8/bbl", "defense stocks +5-8%", "GLD +2.5%"
- If only state media is reporting, flag this explicitly with credibility discount
- If event is < 6 hours old, note high fog-of-war risk
- Write in plain text only, no markdown formatting"""


async def _generate_escalation_analysis(
    crisis_type: str,
    regions: list[str],
    countries: list[str],
    research_results: list[dict],
    historical_analogs: str,
    source_assessment: str,
) -> str | None:
    """Generate escalation analysis using Claude Opus."""
    try:
        allowed, msg = await budget_tracker.check_budget("geopolitical_research")
        if not allowed:
            logger.debug("geopolitical_research budget exhausted")
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
    research_text = sanitize_ingested_content(research_text, source="ddg_search")

    region_str = ", ".join(regions) if regions else "Unknown region"
    country_str = ", ".join(countries) if countries else "Unknown countries"

    user_message = (
        f"CRISIS TYPE: {crisis_type}\n"
        f"REGION(S): {region_str}\n"
        f"COUNTRIES: {country_str}\n\n"
        f"SOURCE CREDIBILITY ASSESSMENT:\n{source_assessment}\n\n"
        f"HISTORICAL ANALOGS FROM KNOWLEDGE BASE:\n{historical_analogs}\n\n"
        f"GATHERED INTELLIGENCE:\n{research_text}\n\n"
        f"Generate a 5-scenario escalation ladder analysis with specific assets, "
        f"price impacts, chokepoint risks, and commodity implications."
    )

    try:
        client = _get_client()
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, lambda: client.messages.create(
            model=config.default_model,
            max_tokens=2500,
            system=cached_system_block(_ESCALATION_PROMPT),
            messages=[{"role": "user", "content": user_message}],
        ))

        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

        result = text.strip() if text.strip() else None
        if result:
            await budget_tracker.record_usage("geopolitical_research")
        return result

    except Exception as e:
        logger.error(f"Escalation analysis generation failed: {e}", exc_info=True)
        return None


# ============================================================
# Research State Management
# ============================================================

def _load_research_state() -> dict:
    """Load geopolitical research state."""
    default = {
        "researched_crises": [],  # [{crisis_type, regions, researched_at}]
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
        logger.warning(f"Failed to load geopolitical research state: {e}")
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
        logger.error(f"Failed to save geopolitical research state: {e}")
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
    entries = state.get("researched_crises", [])
    pruned = []
    for entry in entries:
        ts = entry.get("researched_at", "")
        if ts:
            try:
                dt = datetime.fromisoformat(ts)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=_NY)
                if dt < cutoff:
                    continue
            except (ValueError, TypeError):
                continue  # Drop entries with unparseable dates
        pruned.append(entry)
    if len(pruned) > 100:
        pruned = pruned[-100:]
    state["researched_crises"] = pruned
    return state


def _already_researched(crisis_type: str, regions: list[str], state: dict) -> bool:
    """Check if this crisis type + region was researched in the last 5 days."""
    cutoff = datetime.now(_NY) - timedelta(days=5)
    region_set = set(r.lower() for r in regions)

    for entry in state.get("researched_crises", []):
        if entry.get("crisis_type") != crisis_type:
            continue
        # Check region overlap
        entry_regions = set(r.lower() for r in entry.get("regions", []))
        if region_set & entry_regions:
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
# Delivery
# ============================================================

async def _deliver_geopolitical_intel(
    context,
    crisis_type: str,
    analysis: str,
    urls: list[str],
) -> None:
    """Deliver: Telegram + knowledge commit + bridge write."""
    now = datetime.now(_NY)
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    # 1. Telegram
    try:
        from agent.integrations.scheduled_tasks import _get_chat_ids, _safe_send
        chat_ids = _get_chat_ids()
        if chat_ids:
            crisis_label = crisis_type.replace("_", " ").title()
            msg = (
                f"GEOPOLITICAL INTELLIGENCE: {crisis_label}\n"
                f"{now.strftime('%I:%M %p ET')}\n\n"
                f"{analysis}"
            )
            if len(msg) > 4000:
                msg = msg[:3997] + "..."
            for cid in chat_ids:
                await _safe_send(context.bot, cid, text=msg)
    except Exception as e:
        logger.warning(f"Failed to send geopolitical intel to Telegram: {e}")

    # 2. Knowledge base
    try:
        from agent.tools.knowledge_tools import handle_save_knowledge
        title = f"Geopolitical: {crisis_type.replace('_', ' ').title()} — {now.strftime('%b %d %Y')}"
        source_urls = ", ".join(urls[:3]) if urls else "DuckDuckGo research"
        await handle_save_knowledge({
            "title": title,
            "content": analysis,
            "category": "trading_market_analysis",
            "tags": ["geopolitical_research", crisis_type, now.strftime("%Y-%m-%d")],
            "source": source_urls,
        })
    except Exception as e:
        logger.warning(f"Failed to save geopolitical intel to knowledge: {e}")

    # 3. Bridge write
    try:
        os.makedirs(BRIDGE_DIR, exist_ok=True)
        filename = f"{timestamp}_geopolitical_{crisis_type}.json"
        entry = {
            "id": f"geopolitical_{timestamp}_{crisis_type}",
            "type": "geopolitical_research",
            "crisis_type": crisis_type,
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
        logger.warning(f"Failed to write geopolitical intel to bridge: {e}")


# ============================================================
# Public Entry Point
# ============================================================

async def escalate(context, analysis_text: str, results: list[dict]) -> None:
    """Escalation entry point — called from news_intel.py topic monitor.

    Args:
        context: Telegram bot context (for sending messages)
        analysis_text: Sonnet headline analysis from topic monitor
        results: Raw DuckDuckGo search results list
    """
    try:
        logger.info("Geopolitical escalation triggered")

        # 1. Classify the crisis
        crisis_type, regions, countries = classify_crisis(analysis_text)
        if not crisis_type:
            # S186 W1: visibility on classification miss (S185-B G17)
            logger.info("Geopolitical escalation exiting: classifier could not categorize crisis from analysis text")
            return

        logger.info(f"Crisis: {crisis_type}, regions={regions}, countries={countries}")

        # 2. Check dedup
        state = _load_research_state()
        state = _prune_research_state(state)

        if _already_researched(crisis_type, regions, state):
            logger.debug(f"Already researched {crisis_type}/{regions} recently, skipping")
            return

        # 3. Gather additional research (A — multi-source)
        research_results = await _gather_geopolitical_research(crisis_type, regions, countries)

        # Merge with original results from topic monitor (pre-filtered by news_intel)
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
                        "title": title, "snippet": snippet,
                        "url": url, "query_type": "trigger",
                    })
                    seen_urls.add(url)
        except ImportError:
            for r in results:
                url = r.get("url", r.get("href", ""))
                if url and url not in seen_urls:
                    research_results.append({
                        "title": r.get("title", ""), "snippet": r.get("snippet", r.get("body", "")),
                        "url": url, "query_type": "trigger",
                    })
                    seen_urls.add(url)

        if not research_results:
            # S186 W1: visibility on upstream-empty
            logger.info(f"Geopolitical escalation exiting: 0 research results for {crisis_type}/{regions}")
            return

        # 4. Recall historical analogs (B — memory recall)
        historical_analogs = _recall_historical_analogs(regions, countries, crisis_type)

        # 5. Assess source credibility (C)
        source_assessment = _assess_source_credibility(research_results)

        # 6. Generate escalation analysis (D — scenario analysis)
        analysis = await _generate_escalation_analysis(
            crisis_type=crisis_type,
            regions=regions,
            countries=countries,
            research_results=research_results,
            historical_analogs=historical_analogs,
            source_assessment=source_assessment,
        )

        if not analysis:
            # S186 W1: surface Anthropic failures (original silent-persistence root cause)
            logger.warning(f"Geopolitical escalation: no analysis generated for {crisis_type}/{regions} (Anthropic failure?)")
            return

        # 7. Deliver
        urls = [r.get("url", "") for r in research_results if r.get("url")]
        await _deliver_geopolitical_intel(context, crisis_type, analysis, urls)

        # 8. Record in state
        state["researched_crises"].append({
            "crisis_type": crisis_type,
            "regions": regions,
            "countries": countries,
            "researched_at": datetime.now(_NY).isoformat(),
            "findings_count": len(research_results),
        })
        _save_research_state(state)

        logger.info(f"Geopolitical escalation complete: {crisis_type}, {len(research_results)} findings")

    except Exception as e:
        logger.error(f"Geopolitical escalation failed: {e}", exc_info=True)
