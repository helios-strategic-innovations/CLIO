"""
OpenClaw/Moltbot — News Intelligence Monitor (Session 90)

Two modes:
  A) Event-triggered: fires DuckDuckGo search when economic calendar event is due
  B) Topic monitor: cycles through 10 news categories every 30 min

Delivers: Telegram + knowledge commit + Hydrinia bridge write.
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

logger = logging.getLogger("openclaw.news_intel")

_NY = ZoneInfo("America/New_York")

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "intelligence",
)
INTEL_STATE_PATH = os.path.join(_DATA_DIR, "news_intel_state.json")

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
# Topic Configuration
# ============================================================

_TOPIC_SITES: dict[str, list[str]] = {
    "us_politics": [
        'site:reuters.com OR site:apnews.com executive order legislation regulation',
        'site:politico.com OR site:thehill.com policy economy trade',
    ],
    "geopolitics": [
        'site:reuters.com OR site:bbc.com conflict sanctions trade war',
        'site:aljazeera.com geopolitics international',
    ],
    "fomc_fed": [
        'site:reuters.com "federal reserve" OR "FOMC" OR "rate decision"',
        'site:cnbc.com fed speech powell',
    ],
    "fx_markets": [
        'site:reuters.com USD EUR JPY GBP forex',
        'site:forexlive.com OR site:dailyfx.com currency',
    ],
    "bonds_treasuries": [
        'site:reuters.com treasury yield "bond auction"',
        'site:cnbc.com "yield curve" OR "10-year" OR "2-year"',
    ],
    "commodities": [
        'site:reuters.com oil gold silver copper natural gas commodity',
        'site:investing.com commodity prices today',
    ],
    "us_sectors": [
        'site:reuters.com OR site:cnbc.com sector rotation earnings',
        'site:marketwatch.com sector performance today',
    ],
    "international": [
        'site:reuters.com emerging markets europe asia economy',
        'site:ft.com global economy',
    ],
    "defense_gov": [
        'site:defensenews.com contract award defense procurement',
        'site:reuters.com defense spending military budget',
    ],
    "biotech_pharma": [
        'site:statnews.com FDA approval trial results',
        'site:fiercepharma.com OR site:biopharmadive.com pharma biotech drug',
    ],
    "japan_boj": [
        'site:reuters.com OR site:bbc.com "Bank of Japan" OR "BOJ" OR "yen" OR "Ueda"',
        'site:nikkei.com OR site:japantimes.co.jp economy monetary policy "yield curve"',
    ],
}

_TOPIC_LABELS: dict[str, str] = {
    "us_politics": "US POLITICS",
    "geopolitics": "GEOPOLITICS",
    "fomc_fed": "FOMC/FED",
    "fx_markets": "FX MARKETS",
    "bonds_treasuries": "BONDS",
    "commodities": "COMMODITIES",
    "us_sectors": "US SECTORS",
    "international": "INTERNATIONAL",
    "defense_gov": "DEFENSE/GOV",
    "biotech_pharma": "BIOTECH/PHARMA",
    "japan_boj": "JAPAN/BOJ",
}

_TOPIC_ROTATION = list(_TOPIC_SITES.keys())
_TOPICS_PER_CYCLE = 2

# Escalation throttle: {module_name: (count_today, last_reset_date)}
_escalation_counts: dict[str, tuple[int, str]] = {}
_MAX_ESCALATIONS_PER_DAY = 2


# ============================================================
# Analysis System Prompt
# ============================================================

_ANALYSIS_PROMPT = """You are a precision intelligence analyst. Produce structured intelligence items.

FORMAT (one per search result that has actionable content):
HEADLINE: {specific, factual headline}
{1-3 sentences with specific numbers, names, dates, or levels. State the market implication explicitly.}
SOURCE: {url}

RULES:
- NEVER use: "monitor closely", "watch for", "keep an eye on", "worth noting", "remains to be seen"
- EVERY sentence must contain at least one specific fact (number, name, date, price level)
- If there is nothing specific or actionable in the search results, respond with exactly "SKIP"
- Do NOT speculate beyond what the search results explicitly state
- Prefer Reuters/Bloomberg/AP sourced facts over opinion pieces
- For ECONOMIC EVENT searches: if a search result discusses a PRIOR release period (e.g., "January data" when the current release covers February), respond with SKIP — do not report stale data as current
- SOURCE RECENCY: Only cite events, quotes, and data from the LAST 7 DAYS unless explicitly noting the historical reference. If only stale data is available, respond with SKIP. Include the date of each cited source.
- Maximum 3 items per response
- Write in plain text only, no markdown formatting"""


# ============================================================
# Dedup State Management
# ============================================================

_INTEL_STATE_LOCK = INTEL_STATE_PATH + ".lock"


def _load_intel_state() -> dict:
    """Load intelligence dedup state from disk."""
    default = {
        "items": [],
        "last_topics": [],
        "event_searches": [],
    }
    if not os.path.isfile(INTEL_STATE_PATH):
        return default
    lock_fd = None
    try:
        lock_fd = open(_INTEL_STATE_LOCK, "a")
        fcntl.flock(lock_fd, fcntl.LOCK_SH)
        with open(INTEL_STATE_PATH, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return default
        for key in default:
            if key not in data:
                data[key] = default[key]
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load intel state: {e}")
        return default
    finally:
        if lock_fd:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
            except Exception:
                pass


def _save_intel_state(state: dict) -> None:
    """Save intel state atomically with file locking."""
    lock_fd = None
    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
        lock_fd = open(_INTEL_STATE_LOCK, "a")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        tmp = INTEL_STATE_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp, INTEL_STATE_PATH)
    except Exception as e:
        logger.error(f"Failed to save intel state: {e}")
    finally:
        if lock_fd:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
            except Exception:
                pass


def _prune_intel_state(state: dict) -> dict:
    """Remove items older than 48h, cap at 200."""
    cutoff = datetime.now(_NY) - timedelta(hours=48)

    for key in ("items", "event_searches"):
        entries = state.get(key, [])
        pruned = []
        for entry in entries:
            sent_at = entry.get("sent_at", "")
            if sent_at:
                try:
                    dt = datetime.fromisoformat(sent_at)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=_NY)
                    if dt < cutoff:
                        continue
                except (ValueError, TypeError):
                    continue  # Drop unparseable entries
            else:
                continue  # Drop entries without sent_at timestamp
            pruned.append(entry)
        if len(pruned) > 200:
            pruned = pruned[-200:]
        state[key] = pruned

    # Keep last 10 topics
    state["last_topics"] = state.get("last_topics", [])[-10:]

    return state


def _is_headline_duplicate(headline: str, state: dict) -> bool:
    """Check if a headline (case-insensitive) already exists in recent items."""
    hl = headline.lower().strip()
    for item in state.get("items", []):
        if item.get("headline", "").lower().strip() == hl:
            return True
    return False


def _is_headline_similar(headline: str, state: dict, threshold: float = 0.7) -> bool:
    """Check if headline has high word overlap with recent items (S142-B W3-B)."""
    hl_words = set(headline.lower().split())
    if len(hl_words) < 4:
        return False  # Too short for meaningful comparison
    for item in state.get("items", [])[-50:]:  # Check last 50
        item_words = set(item.get("headline", "").lower().split())
        if len(item_words) < 4:
            continue
        overlap = len(hl_words & item_words) / max(len(hl_words), len(item_words))
        if overlap >= threshold:
            return True
    return False


def _is_event_searched(event_name: str, state: dict) -> bool:
    """Check if we already searched for this event today."""
    today = datetime.now(_NY).date().isoformat()
    for es in state.get("event_searches", []):
        if es.get("event_name") == event_name and es.get("date") == today:
            return True
    return False


# S186 W3.2: date-coherence guard for news intel content
# Catch items that reference ONLY off-years (e.g. "April 29 2025" when current year is 2026).
# Either stale-URL pulls or LLM date hallucinations. Flag so downstream consumers can skip.
_YEAR_RE = re.compile(r"\b(20\d{2})\b")


def _detect_off_year_only(text: str) -> tuple[bool, set[str]]:
    """Return (is_off_year_only, years_mentioned).

    True iff at least one year is mentioned AND the current year is not among them.
    Returns False (coherent) if no years mentioned, or current year is mentioned.
    Defensive: returns (False, set()) for non-string / falsy / unparseable inputs
    (S186 second-pass adversarial finding — TypeError if text is int/list etc.).
    """
    if not text or not isinstance(text, str):
        return (False, set())
    years = set(_YEAR_RE.findall(text))
    if not years:
        return (False, years)
    current_year = str(datetime.now(_NY).year)
    if current_year in years:
        return (False, years)
    # All years referenced are off-year. Likely stale-URL or hallucination.
    return (True, years)


# ============================================================
# Topic Rotation
# ============================================================

def _is_trade_policy_stale(days: int = 7) -> bool:
    """Check if trade_policy_state.json is older than `days` days."""
    try:
        tp_path = os.path.join(_DATA_DIR, "trade_policy_state.json")
        if not os.path.isfile(tp_path):
            return True
        mtime = os.path.getmtime(tp_path)
        age_s = datetime.now(_NY).timestamp() - mtime
        return age_s > days * 86400
    except OSError:
        # S190 L4: defensive default. If stat() fails (perms, racing delete) we
        # don't actually know the file age. The pessimistic-merge posture is to
        # assume stale → force a re-check rather than mask the problem. Earlier
        # "don't force" default could leave trade_policy quietly un-refreshed
        # for weeks if a permission glitch made stat fail repeatedly.
        return True


async def _maybe_check_trade_policy(context) -> None:
    """TP-1 (S211): run the forced trade-policy check when trade_policy_state.json is >7 days old.

    Weekly, matching the sentinel SLA. The escalation counter is charged first so a failing check retries
    at most _MAX_ESCALATIONS_PER_DAY times a day. Never raises.
    """
    try:
        if not (_is_trade_policy_stale(7) and _can_escalate("trade_policy")):
            return
        import importlib
        mod = importlib.import_module("agent.integrations.trade_policy")
        logger.info("TP-1: trade_policy_state.json >7d stale -- running the weekly forced policy check")
        _record_escalation("trade_policy")
        await mod.escalate(context, "", [], forced=True)
    except Exception as e:
        logger.warning("TP-1: weekly trade-policy check failed: %s", e)


def _get_next_topics(state: dict) -> list[str]:
    """Pick next topics from rotation, prioritizing stale topics (>24h since last update)."""
    recent = set(state.get("last_topics", [])[-4:])  # Avoid last 4
    candidates = [t for t in _TOPIC_ROTATION if t not in recent]
    if not candidates:
        candidates = list(_TOPIC_ROTATION)

    # S166: Force us_politics to front when trade_policy_state.json is >7 days stale.
    # trade_policy.py has no scheduler — only triggers via us_politics escalation.
    if _is_trade_policy_stale(7) and "us_politics" in candidates:
        candidates.remove("us_politics")
        candidates.insert(0, "us_politics")
        logger.info("trade_policy_state.json >7d stale — prioritizing us_politics topic")

    # Partition candidates into stale (>24h) vs fresh
    now = datetime.now(_NY)
    items = state.get("items", [])
    stale = []
    fresh = []
    for t in candidates:
        # Find most recent item for this topic
        topic_items = [i for i in items if i.get("topic") == t]
        if not topic_items:
            stale.append(t)
            continue
        try:
            last_ts = max((i.get("sent_at", "") for i in topic_items), default="")
            if last_ts:
                last_dt = datetime.fromisoformat(last_ts)
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=_NY)
                if (now - last_dt).total_seconds() > 86400:
                    stale.append(t)
                else:
                    fresh.append(t)
            else:
                stale.append(t)
        except (ValueError, TypeError):
            fresh.append(t)

    # Stale topics get priority, then fresh in rotation order
    selected = (stale + fresh)[:_TOPICS_PER_CYCLE]

    return selected


def _classify_event_topic(event: dict) -> str:
    """Map an economic calendar event to the closest news topic."""
    name = (event.get("name", "") + " " + event.get("category", "")).lower()
    def _wb(kws):
        return any(re.search(r'\b' + re.escape(kw) + r'\b', name) for kw in kws)
    if _wb(("fed", "fomc", "rate decision", "federal reserve")):
        return "fomc_fed"
    if _wb(("gdp", "cpi", "ppi", "pce", "retail sales", "employment", "payroll", "jobs")):
        return "us_sectors"
    if _wb(("oil", "gas", "gold", "copper", "commodity")):
        return "commodities"
    if _wb(("treasury", "auction", "bond", "yield")):
        return "bonds_treasuries"
    if _wb(("trade", "tariff", "import", "export")):
        return "us_politics"
    return "us_sectors"  # default for macro data releases


# ============================================================
# Escalation Framework
# ============================================================

def _can_escalate(module_name: str) -> bool:
    """Check if we're within the daily escalation limit for a module."""
    today = datetime.now(_NY).date().isoformat()
    count, last_date = _escalation_counts.get(module_name, (0, ""))
    if last_date != today:
        return True  # New day, reset
    return count < _MAX_ESCALATIONS_PER_DAY


def _record_escalation(module_name: str) -> None:
    """Record that an escalation was triggered."""
    today = datetime.now(_NY).date().isoformat()
    count, last_date = _escalation_counts.get(module_name, (0, ""))
    if last_date != today:
        _escalation_counts[module_name] = (1, today)
    else:
        _escalation_counts[module_name] = (count + 1, today)


# Topic → (module_name, import_path, keyword_set)
_ESCALATION_MAP: dict[str, tuple[str, str, set[str]]] = {
    "geopolitics": (
        "geopolitical_research",
        "agent.integrations.geopolitical_research",
        {"military", "strike", "attack", "troops", "invasion", "missile", "blockade",
         "sanctions", "embargo", "ceasefire", "naval", "deployed"},
    ),
    "international": (
        "geopolitical_research",
        "agent.integrations.geopolitical_research",
        {"military", "strike", "attack", "troops", "invasion", "missile", "blockade",
         "sanctions", "embargo", "ceasefire", "naval", "deployed"},
    ),
    "fomc_fed": (
        "fed_analysis",
        "agent.integrations.fed_analysis",
        {"speech", "remarks", "testimony", "rate cut", "rate hike", "dot plot",
         "minutes", "powell", "waller", "bowman", "jefferson", "cook", "miran"},
    ),
    "us_politics": (
        "trade_policy",
        "agent.integrations.trade_policy",
        {"tariff", "tariffs", "import duty", "trade war", "sanctions",
         "export control", "section 301", "section 232", "trade deal"},
    ),
    "commodities": (
        "commodity_research",
        "agent.integrations.commodity_research",
        {"surge", "crash", "disruption", "outage", "opec", "spr release",
         "pipeline", "refinery", "embargo", "supply shock"},
    ),
    "japan_boj": (
        "fed_analysis",
        "agent.integrations.fed_analysis",
        {"rate", "yield curve", "intervention", "ueda", "boj", "yen", "negative rate",
         "monetary policy", "tightening", "easing", "jgb"},
    ),
    "bonds_treasuries": (
        "bond_intel",
        "agent.integrations.bond_intel",
        {"tailed", "weak auction", "bid-to-cover", "primary dealer",
         "indirect bidder", "auction results", "treasury auction", "failed auction"},
    ),
}


_NEGATION_WORDS = frozenset({
    "not", "no", "without", "hardly", "barely", "neither", "rarely",
    "never", "isn't", "aren't", "doesn't", "don't", "won't", "lack",
    "avoid", "unlikely", "denied",
})
# Multi-word negation phrases checked separately via substring match
_NEGATION_PHRASES = ("ruled out", "no sign of", "failed to")


def _non_negated_match(keyword: str, text: str) -> bool:
    """Check if keyword appears in text without a preceding negation within 40 chars."""
    for m in re.finditer(r'\b' + re.escape(keyword) + r'\b', text):
        prefix = text[max(0, m.start() - 40):m.start()].lower()
        # Check single-word negations via split
        prefix_words = prefix.split()
        if any(neg in prefix_words for neg in _NEGATION_WORDS):
            continue  # Negated — skip this occurrence
        # Check multi-word negation phrases via substring
        if any(phrase in prefix for phrase in _NEGATION_PHRASES):
            continue
        return True  # Non-negated match found
    return False  # No non-negated match


def _should_escalate(topic: str, analysis: str) -> tuple[bool, str, str]:
    """Check if analysis text warrants deep-analysis escalation.

    Returns (should_escalate, module_name, import_path).
    Requires 2+ non-negated keyword matches to avoid false positives.
    """
    if topic not in _ESCALATION_MAP:
        return False, "", ""

    module_name, import_path, keywords = _ESCALATION_MAP[topic]

    if not _can_escalate(module_name):
        return False, "", ""

    analysis_lower = analysis.lower()
    match_count = sum(1 for kw in keywords if _non_negated_match(kw, analysis_lower))

    if match_count >= 2:
        return True, module_name, import_path

    return False, "", ""


# ============================================================
# Search Functions
# ============================================================

async def _search_topic(topic: str) -> list[dict]:
    """Search DuckDuckGo with site-targeted queries for a topic.

    Uses report_web_search budget. Returns list of {title, url, snippet}.
    """
    queries = _TOPIC_SITES.get(topic, [])
    if not queries:
        return []

    now = datetime.now(_NY)
    today_str = now.strftime("%B %d %Y")
    results_all = []

    for query_template in queries:
        try:
            allowed, msg = await budget_tracker.check_budget("report_web_search")
            if not allowed:
                break

            from agent.integrations.web_search import search_web
            query = f"{query_template} {today_str}"
            results = await search_web(query, max_results=3)
            await budget_tracker.record_usage("report_web_search")

            # S159-B: If site-scoped query returned nothing, retry without
            # the site: prefix. DDG has poor site-scoped coverage for
            # same-day content on many financial news sites.
            if not results and "site:" in query_template:
                bare_query = re.sub(r'site:\S+\s*(OR\s+site:\S+\s*)*', '', query_template).strip()
                if bare_query:
                    allowed2, _ = await budget_tracker.check_budget("report_web_search")
                    if allowed2:
                        logger.debug("Topic '%s': site-scoped query empty, retrying bare: '%s'", topic, bare_query[:60])
                        results = await search_web(f"{bare_query} {today_str}", max_results=3)
                        await budget_tracker.record_usage("report_web_search")

            for r in (results or []):
                title = r.get("title", "")
                snippet = r.get("snippet", r.get("body", ""))
                url = r.get("url", r.get("href", ""))
                if not title:
                    continue
                results_all.append({
                    "title": title,
                    "snippet": snippet,
                    "url": url,
                })

        except BudgetExhaustedError:
            logger.debug(f"report_web_search budget exhausted during topic search: {topic}")
            break
        except Exception as e:
            logger.warning(f"Topic search failed for '{topic}': {e}")

        await asyncio.sleep(0.5)  # Gentle rate limiting

    return results_all


async def _search_event(event: dict) -> list[dict]:
    """Event-triggered search for an economic event release.

    Builds query from event name + date, searches targeted sites.
    """
    name = event.get("name", "")
    if not name:
        return []

    now = datetime.now(_NY)
    today_str = now.strftime("%B %d %Y")

    queries = [
        f'"{name}" {today_str} actual site:reuters.com OR site:bloomberg.com',
        f'"{name}" {today_str} results site:cnbc.com OR site:marketwatch.com',
    ]

    results_all = []
    for query in queries:
        try:
            allowed, msg = await budget_tracker.check_budget("report_web_search")
            if not allowed:
                break

            from agent.integrations.web_search import search_web
            results = await search_web(query, max_results=3)
            await budget_tracker.record_usage("report_web_search")

            for r in (results or []):
                title = r.get("title", "")
                snippet = r.get("snippet", r.get("body", ""))
                url = r.get("url", r.get("href", ""))
                if not title:
                    continue
                results_all.append({
                    "title": title,
                    "snippet": snippet,
                    "url": url,
                })

        except BudgetExhaustedError:
            break
        except Exception as e:
            logger.warning(f"Event search failed for '{name}': {e}")

        await asyncio.sleep(0.5)

    return results_all


# ============================================================
# Analysis
# ============================================================

async def _analyze_results(topic: str, results: list[dict], event: dict | None = None) -> str | None:
    """Send search results to Claude Sonnet for structured analysis.

    Returns formatted analysis string or None if SKIP/empty.
    Uses news_intel budget.
    """
    if not results:
        return None

    try:
        allowed, msg = await budget_tracker.check_budget("news_intel")
        if not allowed:
            logger.debug(f"news_intel budget exhausted, skipping analysis for {topic}")
            return None
    except BudgetExhaustedError:
        return None

    # Apply staleness filters from news_update module
    try:
        from agent.integrations.news_update import _url_is_stale, _snippet_is_stale
        filtered = []
        for r in results:
            url = r.get("url", "")
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            if _url_is_stale(url):
                logger.debug(f"Intel: stale URL filtered: {url[:60]}")
                continue
            if _snippet_is_stale(title, snippet):
                logger.debug(f"Intel: stale snippet filtered: {title[:60]}")
                continue
            filtered.append(r)
        results = filtered
    except ImportError:
        pass  # If import fails, proceed without staleness filter

    if not results:
        return None

    now_str = datetime.now(_NY).strftime("%I:%M %p ET, %A %B %d %Y")
    label = _TOPIC_LABELS.get(topic, topic.upper())

    # Build user message with search results
    results_text = ""
    for i, r in enumerate(results[:5], 1):
        results_text += f"\n[{i}] {r['title']}\n{r.get('snippet', '')}\nURL: {r.get('url', '')}\n"
    results_text = sanitize_ingested_content(results_text, source="ddg_search")

    event_context = ""
    if event:
        event_context = (
            f"\nECONOMIC EVENT: {event.get('name', '')}\n"
            f"Release date: {(event.get('datetime_et') or 'N/A')[:10]}\n"
            f"Expected: {event.get('forecast', 'N/A')} | Previous: {event.get('previous', 'N/A')}\n"
        )

    user_message = (
        f"Current time: {now_str}\n"
        f"Topic: {label}\n"
        f"{event_context}\n"
        f"SEARCH RESULTS:\n{results_text}\n\n"
        f"Produce intelligence items from these results. If nothing is actionable, respond with SKIP."
    )

    try:
        client = _get_client()
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, lambda: client.messages.create(
            model=config.fast_model,
            max_tokens=1000,
            system=cached_system_block(_ANALYSIS_PROMPT),
            messages=[{"role": "user", "content": user_message}],
            output_config={"effort": "medium"},
        ))

        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

        text = text.strip()
        await budget_tracker.record_usage("news_intel")  # Track ALL API calls, including SKIPs
        if not text or text.upper() == "SKIP":
            return None
        return text

    except Exception as e:
        # S190 M11: drop exc_info=True — full SDK traces can leak secrets via dependent frames (S118 W-075).
        logger.error("Intel analysis failed for %s: %s", topic, str(e)[:300])
        return None


# ============================================================
# Delivery
# ============================================================

async def _deliver_intel(topic: str, analysis: str, urls: list[str], event: dict | None = None) -> None:
    """Deliver intelligence: Telegram + knowledge commit + bridge write."""
    # S189 C2: defense-in-depth null-intel rejection at the bridge writer.
    # The caller (_run_topic_monitor) already drops None at line 991; this is a
    # second-line guard for any future caller-side regression that bypasses the
    # check. Refuses SKIP/empty content so Hydrinia's intel stream stays clean.
    # S189 UL-5: tightened from startswith("SKIP") to a word-boundary regex
    # so "SKIPPING analysis until next FOMC..." (legitimate content) is NOT
    # rejected, while "SKIP\n\nExplanation" (the audit's documented null-
    # intel pattern) IS rejected.
    if not analysis or re.match(r"^\s*SKIP(\s|$)", analysis, re.IGNORECASE):
        logger.info(
            f"S189 C2: null intel rejected at bridge writer (topic={topic}, "
            f"analysis={'empty' if not analysis else 'SKIP'})"
        )
        return
    now = datetime.now(_NY)
    label = _TOPIC_LABELS.get(topic, topic.upper())
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    # 1. Knowledge base commit (fire-and-forget)
    try:
        from agent.tools.knowledge_tools import handle_save_knowledge
        title = f"Intel: {label} — {now.strftime('%b %d %Y %I:%M %p ET')}"
        source_urls = ", ".join(urls[:3]) if urls else "DuckDuckGo search"
        # S122 2B: Heuristic quality scoring for Intel bulletins (enables SEAL + archival)
        _src_count = len(urls) if isinstance(urls, list) else 1
        _content_len = len(analysis)
        _intel_quality = 5
        if _src_count >= 3:
            _intel_quality += 1
        if _content_len > 1500:
            _intel_quality += 1
        if _content_len < 300:
            _intel_quality -= 2
        await handle_save_knowledge({
            "title": title,
            "content": analysis,
            "category": "trading_market_analysis",
            "tags": ["intelligence", topic, now.strftime("%Y-%m-%d")],
            "source": source_urls,
            "quality_score": max(1, min(10, _intel_quality)),
        })
    except Exception as e:
        logger.warning(f"Failed to save intel to knowledge base: {e}")

    # 3. Bridge write (fire-and-forget). Hydrinia is permanently off (operator
    # direction, 2026-09-24) and nothing reads this directory anymore -- gated,
    # default OFF.
    try:
        from agent.hydrinia_bridge import bridge_enabled
        _bridge_write_on = bridge_enabled()
    except Exception as e:
        logger.debug(f"Intel bridge: hydrinia_bridge check failed: {e}")
        _bridge_write_on = False
    if _bridge_write_on:
        try:
            os.makedirs(BRIDGE_DIR, exist_ok=True)
            filename = f"{timestamp}_{topic}.json"
            entry = {
                "id": f"intel_{timestamp}_{topic}",
                "type": "news_intelligence",
                "topic": topic,
                "label": label,
                "created": now.isoformat(),
                "content": analysis,
                "urls": urls[:5],
                "event": event,
            }
            file_path = os.path.join(BRIDGE_DIR, filename)
            tmp = file_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(entry, f, indent=2, default=str)
            os.replace(tmp, file_path)
            logger.debug(f"Intel bridge write: {filename}")
        except Exception as e:
            logger.warning(f"Failed to write intel to bridge: {e}")


async def _deliver_and_notify(context, topic: str, analysis: str, urls: list[str],
                               event: dict | None = None) -> None:
    """Full delivery with Telegram notification using scheduler context."""
    # S189 C2 + UL-5: defense-in-depth null-intel rejection. Same logic as
    # _deliver_intel. Regex `^SKIP(\s|$)` matches bare "SKIP", "SKIP\n",
    # "SKIP "; excludes "SKIPPING", "Skip-list", etc.
    if not analysis or re.match(r"^\s*SKIP(\s|$)", analysis, re.IGNORECASE):
        logger.info(
            f"S189 C2: null intel rejected at notify-deliver writer "
            f"(topic={topic}, analysis={'empty' if not analysis else 'SKIP'})"
        )
        return
    now = datetime.now(_NY)
    label = _TOPIC_LABELS.get(topic, topic.upper())

    # Telegram notification
    try:
        from agent.integrations.scheduled_tasks import _get_chat_ids, _safe_send
        chat_ids = _get_chat_ids()
        if chat_ids:
            header = f"INTEL: {label}"
            if event:
                header = f"MACRO RELEASE: {event.get('name', '')}"
            msg = f"{header}\n{now.strftime('%I:%M %p ET')}\n\n{analysis}"
            # Strip markdown formatting for plain text Telegram
            msg = msg.replace("**", "").replace("*", "").replace("`", "").replace("##", "").replace("# ", "")
            # Truncate if too long
            if len(msg) > 4000:
                msg = msg[:3997] + "..."
            for cid in chat_ids:
                await _safe_send(context.bot, cid, text=msg)
    except Exception as e:
        logger.warning(f"Failed to send intel to Telegram: {e}")

    # Knowledge + bridge
    await _deliver_intel(topic, analysis, urls, event)


# ============================================================
# Worldbrief Command
# ============================================================

def get_worldbrief() -> str:
    """Get last 5 intelligence items formatted for /worldbrief command.

    Sync function — reads from state file directly.
    """
    state = _load_intel_state()
    items = state.get("items", [])

    if not items:
        return "No intelligence data available yet. The system will begin gathering intelligence shortly."

    now = datetime.now(_NY)
    header = f"WORLD BRIEF — {now.strftime('%b %d, %Y %I:%M %p ET')}\n"

    # Take last 5 items (most recent)
    recent = items[-5:]
    lines = []
    for item in reversed(recent):  # Newest first
        label = item.get("label", item.get("topic", "").upper())
        headline = item.get("headline", "")
        if headline:
            lines.append(f"{label}: {headline}")

    if not lines:
        return f"{header}\nNo actionable intelligence in the last 48 hours."

    return header + "\n" + "\n\n".join(lines)


# ============================================================
# Scheduled Callbacks
# ============================================================

async def run_event_search(context) -> None:
    """Scheduled callback — runs every 5 minutes.

    Checks economic calendar for events in the next 30 min.
    Fires targeted search for high-importance events at release time.
    """
    if not config.news_intel_enabled:
        return

    now = datetime.now(_NY)

    # Time guard: weekdays 8 AM - 5 PM ET only
    if now.weekday() >= 5:
        return
    if now.hour < 8 or now.hour >= 17:
        return

    try:
        from agent.integrations.econ_calendar import get_upcoming_events

        # Events in next 30 minutes, importance >= 2
        upcoming = get_upcoming_events(hours=0.5)
        if not upcoming:
            return

        state = _load_intel_state()
        state = _prune_intel_state(state)

        for event in upcoming:
            importance = event.get("importance", 1)
            if importance < 2:
                continue

            event_name = event.get("name", "")
            if _is_event_searched(event_name, state):
                continue

            logger.info(f"Event trigger: searching for '{event_name}' (importance {importance})")

            # Mark as searched before searching (prevent double-fire)
            state["event_searches"].append({
                "event_name": event_name,
                "date": now.date().isoformat(),
                "sent_at": now.isoformat(),
            })
            _save_intel_state(state)

            # Search and analyze
            results = await _search_event(event)
            if not results:
                continue

            # Determine topic from event category (default to macro_event)
            event_topic = _classify_event_topic(event)
            analysis = await _analyze_results(event_topic, results, event=event)
            if not analysis:
                continue

            # Extract URLs from filtered results (avoid stale URL leakage)
            _filtered_for_urls = results
            try:
                from agent.integrations.news_update import _url_is_stale, _snippet_is_stale
                _filtered_for_urls = [
                    r for r in results
                    if not _url_is_stale(r.get("url", ""))
                    and not _snippet_is_stale(r.get("title", ""), r.get("snippet", r.get("body", "")))
                ]
            except ImportError:
                pass
            urls = [r.get("url", "") for r in _filtered_for_urls if r.get("url")]

            # Deliver
            await _deliver_and_notify(context, event_topic, analysis, urls, event=event)

            # Record in state
            headline = ""
            for _hl_line in analysis.split("\n"):
                if _hl_line.strip().startswith("HEADLINE:"):
                    headline = _hl_line.replace("HEADLINE:", "").strip()[:200]
                    break
            if not headline:
                headline = analysis.split("\n")[0].strip()[:200]
            # S186 W3.2: date-coherence guard — flag items referencing only off-years
            _off_year, _years = _detect_off_year_only(analysis)
            _item = {
                "headline": headline,
                "topic": event_topic,
                "label": event_name,
                "urls": urls[:3],
                "sent_at": now.isoformat(),
                "analysis": analysis[:500],
            }
            if _off_year:
                _item["_quarantined"] = True
                _item["_quarantine_reason"] = f"off_year_only:{','.join(sorted(_years))}"
                logger.warning(
                    "news_intel: off-year-only item quarantined (years=%s, headline=%r)",
                    sorted(_years), headline[:80],
                )
            state["items"].append(_item)
            _save_intel_state(state)

            await asyncio.sleep(1)  # Space out multiple events

    except Exception as e:
        # S190 M11: drop exc_info=True (S118 W-075).
        logger.error("Event search failed: %s", str(e)[:300])


async def run_topic_monitor(context) -> None:
    """Scheduled callback — runs every 30 minutes.

    Cycles through topic categories, searching targeted sites.
    Active window: 8 AM - 8 PM ET, every day (including weekends for geopolitics).
    """
    if not config.news_intel_enabled:
        return

    now = datetime.now(_NY)

    # Time guard: 8 AM - 8 PM ET
    if now.hour < 8 or now.hour >= 20:
        return

    logger.info("Starting topic monitor cycle...")

    try:
        state = _load_intel_state()
        state = _prune_intel_state(state)

        topics = _get_next_topics(state)
        topics_covered = []

        # TP-1 (S211): weekly trade-policy check, independent of what the topics deliver this cycle
        await _maybe_check_trade_policy(context)

        for topic in topics:
            logger.info(f"Topic monitor: searching '{topic}'")

            results = await _search_topic(topic)
            if not results:
                logger.debug(f"No results for topic '{topic}'")
                topics_covered.append(topic)
                continue

            # Cross-category URL dedup (S142-B W3-A)
            try:
                from agent.integrations.learning_engine import is_url_seen, record_url_seen
                _fresh = [r for r in results if not is_url_seen(r.get("url", r.get("href", "")))]
                if not _fresh:
                    logger.info(f"All URLs already processed for '{topic}', skipping")
                    topics_covered.append(topic)
                    continue
                for r in _fresh:
                    _u = r.get("url", r.get("href", ""))
                    if _u:
                        record_url_seen(_u, f"news_intel_{topic[:20]}")
                results = _fresh
            except Exception as _e:
                logger.debug(f"Cross-category URL dedup skipped: {_e}")

            analysis = await _analyze_results(topic, results)
            if not analysis:
                logger.debug(f"Analysis returned SKIP for topic '{topic}'")
                topics_covered.append(topic)
                continue

            # Extract URLs from filtered results (avoid stale URL leakage)
            _filtered_for_urls = results
            try:
                from agent.integrations.news_update import _url_is_stale, _snippet_is_stale
                _filtered_for_urls = [
                    r for r in results
                    if not _url_is_stale(r.get("url", ""))
                    and not _snippet_is_stale(r.get("title", ""), r.get("snippet", r.get("body", "")))
                ]
            except ImportError:
                pass
            urls = [r.get("url", "") for r in _filtered_for_urls if r.get("url")]

            # Extract headline from analysis
            headline = ""
            for line in analysis.split("\n"):
                if line.startswith("HEADLINE:"):
                    headline = line.replace("HEADLINE:", "").strip()[:200]
                    break
            if not headline:
                headline = analysis.split("\n")[0][:200]

            # Check dedup (exact + fuzzy similarity)
            if _is_headline_duplicate(headline, state) or _is_headline_similar(headline, state):
                logger.debug(f"Duplicate/similar headline, skipping: {headline[:60]}")
                topics_covered.append(topic)
                continue

            # Deliver
            await _deliver_and_notify(context, topic, analysis, urls)

            # Escalation: trigger deep analysis if high-relevance keywords detected
            # S166: Force trade_policy escalation when state is >14 days stale,
            # even if keyword gate didn't fire (escalation-only module has no scheduler)
            try:
                # (S166's forced trade_policy escalation lived here; it only ran when a topic had new content and
                # handed over general text that the specificity gate always rejected -- replaced by
                # _maybe_check_trade_policy above, TP-1 S211.)
                should_esc, esc_module, esc_import = _should_escalate(topic, analysis)
                if should_esc:
                    import importlib
                    mod = importlib.import_module(esc_import)
                    if hasattr(mod, "escalate"):
                        logger.info(f"Escalating {topic} to {esc_module}")
                        # Filter stale results before passing to deep-analysis
                        # (S100 fix: was passing raw unfiltered results)
                        esc_results = results
                        try:
                            from agent.integrations.news_update import _url_is_stale, _snippet_is_stale
                            esc_results = [
                                r for r in results
                                if not _url_is_stale(r.get("url", ""))
                                and not _snippet_is_stale(r.get("title", ""), r.get("snippet", ""))
                            ]
                        except ImportError:
                            pass
                        await mod.escalate(context, analysis, esc_results)
                        _record_escalation(esc_module)
            except Exception as e:
                logger.debug(f"Escalation skipped for {topic}: {e}")

            # Record in state
            label = _TOPIC_LABELS.get(topic, topic.upper())
            # S186 W3.2: date-coherence guard
            _off_year, _years = _detect_off_year_only(analysis)
            _topic_item = {
                "headline": headline,
                "topic": topic,
                "label": label,
                "urls": urls[:3],
                "sent_at": now.isoformat(),
                "analysis": analysis[:500],
            }
            if _off_year:
                _topic_item["_quarantined"] = True
                _topic_item["_quarantine_reason"] = f"off_year_only:{','.join(sorted(_years))}"
                logger.warning(
                    "news_intel topic_monitor: off-year-only item quarantined (topic=%s, years=%s, headline=%r)",
                    topic, sorted(_years), headline[:80],
                )
            state["items"].append(_topic_item)

            topics_covered.append(topic)
            _save_intel_state(state)  # S118 W-024: Save per-topic to avoid data loss on crash
            await asyncio.sleep(1)

        # Update rotation state
        state["last_topics"].extend(topics_covered)
        _save_intel_state(state)

        logger.info(f"Topic monitor complete: {len(topics_covered)} topics covered ({', '.join(topics_covered)})")

    except Exception as e:
        # S190 M11: drop exc_info=True (S118 W-075).
        logger.error("Topic monitor failed: %s", str(e)[:300])
