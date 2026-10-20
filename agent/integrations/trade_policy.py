"""
OpenClaw/Moltbot — Trade Policy Impact Chains (Session 92, Phase 2)

Deep-analysis module triggered by news_intel.py escalation when us_politics
topic contains tariff/trade policy keywords.

Applies the A/B/C/D research framework:
  A) Multi-source research — USTR announcements, HTS codes, company exposure,
     historical analogs (2018 steel/China tariffs)
  B) Memory recall — prior trade policy episodes + hardcoded historical analogs
  C) Policy validity — EO vs legislation, election year discount, waiver likelihood
  D) Scenario analysis — FULL_IMPLEMENTATION / PARTIAL / DELAYED / REDUCED / REVERSED
     with company-level margin impact
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

logger = logging.getLogger("openclaw.trade_policy")

_NY = ZoneInfo("America/New_York")

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "intelligence",
)
RESEARCH_STATE_PATH = os.path.join(_DATA_DIR, "trade_policy_state.json")

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
# Policy Type Taxonomy + Research Query Maps
# ============================================================

TRADE_POLICY_RESEARCH_MAP: dict[str, dict] = {
    "tariff_announcement": {
        "keywords": ["tariff", "tariffs", "import duty", "section 301", "section 232",
                      "anti-dumping", "countervailing duty", "trade war"],
        "obvious_queries": [
            "USTR {country} tariff details {goods}",
            "{country} tariff {goods} industry impact",
            "{country} retaliatory tariff list USA exports",
        ],
        "niche_queries": [
            "HTS code {goods} import volume USA from {country}",
            "company COGS sourcing {country} {goods} exposure percentage",
            "pricing power pass-through tariff {sector} companies",
            "2018 tariff actual impact {goods} prices companies",
            "tariff front-loading import surge before effective date {country}",
            "domestic production capacity {goods} USA substitute",
            "substitute source countries {goods} Vietnam Mexico India",
            "{country} trade retaliation vulnerable USA exports agriculture",
        ],
        "validity_queries": [
            "{country} tariff exemption history waiver rate",
            "executive order tariff vs legislation durability",
            "election year trade policy performative announcement history",
        ],
    },
    "export_controls": {
        "keywords": ["export control", "entity list", "BIS", "CHIPS Act",
                      "semiconductor controls", "technology transfer"],
        "obvious_queries": [
            "BIS entity list latest designations {country}",
            "{country} semiconductor export controls impact",
        ],
        "niche_queries": [
            "semiconductor equipment export {country} revenue exposure companies",
            "CHIPS Act subsidy progress recipients timeline",
            "allied coordination export controls {country} Netherlands Japan",
            "{country} semiconductor self-sufficiency progress",
        ],
        "validity_queries": [
            "export control enforcement effectiveness {country}",
        ],
    },
    "trade_agreement": {
        "keywords": ["trade agreement", "trade deal", "USMCA", "bilateral",
                      "free trade", "market access", "trade pact"],
        "obvious_queries": [
            "{country} trade agreement deal provisions",
            "{country} market access changes USA",
        ],
        "niche_queries": [
            "{country} trade deal rules of origin provisions",
            "{country} trade agreement IP provisions enforcement",
            "{country} trade deal currency provisions manipulation",
            "{country} trade agreement ratification timeline congress",
        ],
        "validity_queries": [
            "trade agreement commitment vs actual purchases history",
        ],
    },
}

# Country name variations for detection
_COUNTRY_VARIANTS = {
    "china": "China", "chinese": "China", "beijing": "China", "prc": "China",
    "japan": "Japan", "japanese": "Japan", "tokyo": "Japan",
    "mexico": "Mexico", "mexican": "Mexico",
    "canada": "Canada", "canadian": "Canada",
    "eu": "European Union", "europe": "European Union", "european union": "European Union",
    "india": "India", "indian": "India", "new delhi": "India",
    "south korea": "South Korea", "korean": "South Korea", "seoul": "South Korea",
    "taiwan": "Taiwan", "taiwanese": "Taiwan",
    "vietnam": "Vietnam", "vietnamese": "Vietnam",
    "brazil": "Brazil", "brazilian": "Brazil",
    "uk": "United Kingdom", "britain": "United Kingdom", "british": "United Kingdom",
    "germany": "Germany", "german": "Germany",
    "russia": "Russia", "russian": "Russia",
    "iran": "Iran", "iranian": "Iran",
}


# ============================================================
# Historical Analogs (hardcoded)
# ============================================================

HISTORICAL_ANALOGS: dict[str, dict] = {
    "2018_steel_232": {
        "label": "2018 Steel/Aluminum Section 232 Tariffs",
        "policy": "25% steel, 10% aluminum tariffs",
        "announced": "March 1, 2018",
        "effective": "March 23, 2018",
        "market_impact": "X (steel ETF SLX) +8% on announcement, then -15% over 6 months as retaliation fears grew. SPY -2.5% week of implementation.",
        "exemptions": "~40% of affected imports received country or product exemptions within 12 months (Canada, Mexico, EU partial, Australia, South Korea, Brazil, Argentina).",
        "front_loading": "Steel imports surged 15-20% in Mar-Jun 2018 as buyers front-loaded before exemption expiry. Reversed Q3 2018.",
        "key_lesson": "Announced rate ≠ effective rate. Exemptions carved out 40% of volume. Domestic steel prices spiked but downstream manufacturers (autos, appliances) absorbed cost.",
    },
    "2018_china_301": {
        "label": "2018-2019 China Section 301 Tariffs",
        "policy": "$250B in 3 tranches (List 1: $34B Jul, List 2: $16B Aug, List 3: $200B Sep at 10% raised to 25% May 2019)",
        "announced": "March 22, 2018 (initial memo)",
        "effective": "July 6, 2018 (List 1)",
        "market_impact": "SPY -4% Jul-Oct 2018 drawdown. KWEB -25% peak-to-trough. Soybeans -18% on China retaliation. CNY weakened 8% (offset tariff cost). Tech supply chain stocks (AAPL, MU, AVGO) volatile.",
        "front_loading": "Massive front-loading Mar-Jun 2018 — container traffic at LA/Long Beach port up 8% YoY. Reversed Q4 2018 creating inventory hangover.",
        "key_lesson": "Multi-tranche rollout created prolonged uncertainty. Currency adjustment partially offset tariff cost. Agricultural retaliation was targeted at politically sensitive states.",
    },
    "2019_china_phase1": {
        "label": "2019 Phase One Trade Deal",
        "policy": "Partial de-escalation. China committed to $200B additional purchases over 2 years. US delayed Dec 15 tariff tranche, cut Sep tranche from 15% to 7.5%.",
        "announced": "December 13, 2019",
        "effective": "February 14, 2020",
        "market_impact": "SPY +4% in December 2019 on announcement. Risk-on rally across EM, commodities. Soybean +5%.",
        "key_lesson": "Commitments ≠ purchases. China met only ~58% of purchase commitments (COVID + structural issues). Market priced in full deal value upfront — subsequent disappointment was gradual, not sharp.",
    },
}


# ============================================================
# Policy Validity Checks
# ============================================================

POLICY_VALIDITY_CHECKS = {
    "eo_vs_legislation": {
        "description": "Executive orders are reversible by next administration. Discount 20-30% for EO-based tariffs.",
        "discount": 0.25,
    },
    "election_year": {
        "description": "Within 6 months of election, trade actions 40% more likely performative (announcements without full implementation).",
        "discount": 0.40,
    },
    "waiver_history": {
        "description": "Historically 30-50% of tariff-affected imports receive exemptions within 12 months.",
        "discount": 0.40,
    },
    "implementation_gap": {
        "description": "EO tariffs: 45-90 day implementation gap. Legislation: 120-360 days. Market often overreacts to announcement.",
    },
}


# ============================================================
# Policy Classification + Entity Extraction
# ============================================================

def classify_policy(text: str) -> str:
    """Classify text into policy type by keyword count."""
    text_lower = text.lower()
    best_type = "tariff_announcement"  # default
    best_count = 0
    for policy_type, cfg in TRADE_POLICY_RESEARCH_MAP.items():
        count = sum(1 for kw in cfg["keywords"] if re.search(r'\b' + re.escape(kw.lower()) + r'\b', text_lower))
        if count > best_count:
            best_count = count
            best_type = policy_type
    return best_type


def _extract_policy_details(text: str) -> dict:
    """Extract country, goods category, rate from analysis text."""
    text_lower = text.lower()

    # Country detection (word-boundary matching to avoid "uk" in "ukraine", "indian" in "indiana")
    country = None
    for variant, canonical in _COUNTRY_VARIANTS.items():
        if re.search(r'\b' + re.escape(variant) + r'\b', text_lower):
            country = canonical
            break

    # Goods category extraction (common trade goods)
    goods = None
    goods_patterns = [
        "steel", "aluminum", "semiconductors", "auto", "automobiles",
        "agriculture", "soybeans", "electronics", "solar panels",
        "lumber", "dairy", "pharmaceuticals", "rare earth", "lithium",
        "electric vehicles", "batteries", "chemicals", "textiles",
        "oil", "natural gas", "copper", "machinery",
    ]
    for g in goods_patterns:
        if re.search(r'\b' + re.escape(g) + r'\b', text_lower):
            goods = g
            break

    # Rate extraction (e.g., "25% tariff", "10 percent")
    rate = None
    rate_match = re.search(r'(\d{1,3})\s*(%|percent)', text_lower)
    if rate_match:
        rate = f"{rate_match.group(1)}%"

    # Authority extraction
    authority = None
    if "section 301" in text_lower:
        authority = "Section 301 (unfair trade practices)"
    elif "section 232" in text_lower:
        authority = "Section 232 (national security)"
    elif "executive order" in text_lower or " eo " in text_lower:
        authority = "Executive Order"
    elif "legislation" in text_lower or "congress" in text_lower or "bill" in text_lower:
        authority = "Legislation"

    return {
        "country": country or "unspecified",
        "goods": goods or "unspecified goods",
        "rate": rate or "unspecified rate",
        "authority": authority or "unspecified authority",
    }


def _is_specific_policy(text: str) -> bool:
    """Validate that the analysis is about a SPECIFIC policy action, not general trade discussion.

    Requires at least one country + one action verb or specific policy reference.
    """
    text_lower = text.lower()
    has_country = any(re.search(r'\b' + re.escape(v) + r'\b', text_lower) for v in _COUNTRY_VARIANTS)
    has_action = any(re.search(r'\b' + re.escape(kw) + r'\b', text_lower) for kw in [
        "impose", "announced", "signed", "effective", "implement",
        "tariff", "section 301", "section 232", "executive order",
        "ban", "restrict", "sanction", "duty", "levy",
    ])
    return has_country and has_action


# ============================================================
# Research Gathering
# ============================================================

async def _gather_impact_chain_research(
    policy_type: str,
    country: str,
    goods: str,
) -> list[dict]:
    """Execute research queries with entity substitution.

    Runs 2 obvious + 4 niche queries. Uses report_web_search budget.
    """
    policy_config = TRADE_POLICY_RESEARCH_MAP.get(policy_type, {})
    obvious = policy_config.get("obvious_queries", [])
    niche = policy_config.get("niche_queries", [])

    # Infer sector from goods
    _goods_to_sector = {
        "steel": "industrials", "aluminum": "materials", "semiconductors": "technology",
        "auto": "consumer discretionary", "automobiles": "consumer discretionary",
        "agriculture": "staples", "soybeans": "agriculture",
        "electronics": "technology", "solar panels": "clean energy",
        "pharmaceuticals": "healthcare", "rare earth": "materials",
        "electric vehicles": "automotive", "batteries": "technology",
    }
    sector = _goods_to_sector.get(goods, "affected industry")

    subs = {
        "country": country,
        "goods": goods,
        "sector": sector,
    }

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
            logger.debug("Budget exhausted during trade policy research")
            break
        except Exception as e:
            logger.warning(f"Trade policy research query failed: {e}")

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
# Memory Recall — Historical Analogs
# ============================================================

def _recall_historical_analogs(country: str, policy_type: str) -> str:
    """Combine hardcoded analogs with knowledge base search."""
    parts = []

    # 1. Check hardcoded analogs
    country_lower = country.lower()
    for key, analog in HISTORICAL_ANALOGS.items():
        # Match by country name in key or policy text
        if country_lower in key.lower() or country_lower in analog.get("policy", "").lower():
            parts.append(
                f"HISTORICAL ANALOG: {analog['label']}\n"
                f"Policy: {analog['policy']}\n"
                f"Announced: {analog['announced']}, Effective: {analog['effective']}\n"
                f"Market Impact: {analog['market_impact']}\n"
                f"Exemptions: {analog.get('exemptions', 'N/A')}\n"
                f"Front-Loading: {analog.get('front_loading', 'N/A')}\n"
                f"Key Lesson: {analog['key_lesson']}"
            )

    # If no country-specific match, include the most relevant one
    if not parts:
        # Default to 2018 China (most comprehensive analog)
        analog = HISTORICAL_ANALOGS["2018_china_301"]
        parts.append(
            f"CLOSEST ANALOG (different country): {analog['label']}\n"
            f"Market Impact: {analog['market_impact']}\n"
            f"Key Lesson: {analog['key_lesson']}"
        )

    # 2. Search knowledge base
    try:
        from agent.tools.knowledge_tools import load_all_entries
        all_entries = load_all_entries()

        cutoff = datetime.now(_NY) - timedelta(days=180)
        trade_keywords = ["tariff", "trade policy", "trade war", "import duty", "export control"]

        relevant = []
        for entry in all_entries:
            title = entry.get("title", "").lower()
            tags = entry.get("tags", [])
            tags_str = " ".join(str(t) for t in tags) if isinstance(tags, list) else ""
            combined = f"{tags_str} {title}".lower()

            has_trade = any(re.search(r'\b' + re.escape(kw) + r'\b', combined, re.IGNORECASE) for kw in trade_keywords)
            if has_trade:
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

        # Sort newest first, take top 3
        def _sort_key(e):
            try:
                dt = datetime.fromisoformat(e.get("created", "2020-01-01"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=_NY)
                return dt
            except (ValueError, TypeError):
                return datetime(2020, 1, 1, tzinfo=_NY)

        relevant.sort(key=_sort_key, reverse=True)

        for entry in relevant[:3]:
            title = entry.get("title", "")
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
                    logger.debug(f"Fire-and-forget knowledge entry read failed: {e}")
            parts.append(f"[{created}] {title}\n{content}")

    except Exception as e:
        logger.debug(f"Fire-and-forget load_all_entries failed: {e}")

    return "\n---\n".join(parts) if parts else "No historical analogs found."


# ============================================================
# Policy Validity Assessment
# ============================================================

def _assess_policy_validity(policy_type: str, details: dict) -> str:
    """Assess validity/durability of the policy action."""
    parts = []
    now = datetime.now(_NY)

    # Election year check — US elections in November of even years (midterms/presidential)
    months_to_election = (11 - now.month) if now.year % 2 == 0 else 12  # S118 W-074: allow negative for abs()
    if abs(months_to_election) <= 6:
        parts.append(
            f"ELECTION YEAR DISCOUNT: {now.year} midterms in November. "
            f"{months_to_election} months away. "
            f"Trade actions within 6 months of election are ~40% more likely performative. "
            f"Apply performative discount."
        )

    # EO vs legislation
    authority = details.get("authority", "")
    if "executive order" in authority.lower() or "eo" in authority.lower():
        parts.append(
            "DURABILITY: Executive Order — reversible by next administration. "
            "Discount 20-30% for EO-based tariffs vs legislative action."
        )
    elif "legislation" in authority.lower():
        parts.append(
            "DURABILITY: Legislative action — more durable, requires congressional repeal. "
            "Implementation timeline: 120-360 days."
        )
    elif "section 301" in authority.lower():
        parts.append(
            "DURABILITY: Section 301 (presidential authority) — semi-durable. "
            "Requires USTR investigation + annual review. Can be modified/revoked by president."
        )
    elif "section 232" in authority.lower():
        parts.append(
            "DURABILITY: Section 232 (national security) — executive authority with broad discretion. "
            "Historically subject to extensive exemptions (40% carved out in 2018 steel tariffs)."
        )

    # Waiver/exemption likelihood
    parts.append(
        f"EXEMPTION LIKELIHOOD: Historically 30-50% of tariff-affected imports "
        f"receive exemptions within 12 months. Higher for allied nations and "
        f"inputs with no domestic substitute."
    )

    # Country relationship
    country = details.get("country", "")
    allies = {"canada", "japan", "south korea", "united kingdom", "european union",
              "australia", "germany", "france", "india"}
    if country.lower() in allies:
        parts.append(
            f"ALLY STATUS: {country} is a US ally/partner — higher waiver probability. "
            f"Tariffs on allies often serve as negotiating leverage, not permanent policy."
        )

    # Implementation gap
    parts.append(
        "IMPLEMENTATION GAP: EO tariffs typically have 45-90 day effective date. "
        "Market often overreacts to announcement — pricing in full implementation "
        "when actual rate will be lower after exemptions."
    )

    return "\n".join(parts) if parts else "No validity concerns identified."


# ============================================================
# Impact Chain Analysis (Opus-powered)
# ============================================================

_IMPACT_CHAIN_PROMPT = """You are a trade policy analyst producing impact chain analysis for a trading system.

Your job: trace a trade policy action through its full economic impact chain, from announcement to specific company-level margin impacts.

FORMAT:

POLICY: {authority, rate, goods, countries — 2-3 sentences}

IMPACT CHAIN:
Level 1 — GOODS: {what products, HTS categories, dollar volume}
Level 2 — COMPANIES: {specific tickers with import exposure %, COGS sourcing}
Level 3 — MARGINS: {pricing power assessment — can they pass through? by how much?}
Level 4 — CONSUMER: {end price impact, demand elasticity response}
Level 5 — GDP: {aggregate macro impact — investment, employment, trade balance}

HISTORICAL ANALOG: {closest prior episode with SPECIFIC market data — price moves, durations, exemption rates}

FRONT-LOADING RISK: {will importers rush to bring goods in before effective date? volume estimate, timing, reversal risk}

RETALIATION RISK: {vulnerable US exports to target country — specific products, dollar values, politically sensitive states}

5 SCENARIOS:

FULL_IMPLEMENTATION (X%): {rate applied as announced, no exemptions}
  Market: {SPY, sector ETFs, specific tickers — direction + magnitude}
  Duration: {how long until priced in}

PARTIAL (X%): {reduced rate or limited scope after negotiations}
  Market: {impacts}

DELAYED (X%): {implementation pushed back 90-180 days}
  Market: {impacts}

REDUCED (X%): {significant carve-outs, effective rate much lower than announced}
  Market: {impacts}

REVERSED (X%): {policy withdrawn or superseded by agreement}
  Market: {impacts}

(Probabilities must sum to approximately 100%)

POSITIONING: {specific actionable trades — long/short, tickers, rationale}

RULES:
- Every scenario MUST include specific tickers and price impacts
- The impact chain MUST go from goods to companies to margins — no skipping levels
- NEVER use: "monitor closely", "watch for", "keep an eye on", "remains to be seen"
- State impacts directly: "NUE +8% on domestic pricing power", "AAPL COGS +2% on component tariff"
- If election year, EXPLICITLY apply 40% performative discount to base probability
- If EO-based, note 20-30% durability discount
- Include front-loading timing: "importers will surge Q1, creating inventory hangover Q2"
- Write in plain text only, no markdown formatting"""


async def _generate_impact_chain_analysis(
    policy_type: str,
    details: dict,
    research_results: list[dict],
    historical_analogs: str,
    validity_assessment: str,
) -> str | None:
    """Generate impact chain analysis using Claude Opus."""
    try:
        allowed, msg = await budget_tracker.check_budget("trade_policy")
        if not allowed:
            logger.debug("trade_policy budget exhausted")
            return None
    except BudgetExhaustedError:
        return None

    research_text = ""
    for i, r in enumerate(research_results[:15], 1):
        qt = r.get("query_type", "")
        research_text += (
            f"\n[{i}] ({qt}) {r['title']}\n"
            f"{r.get('snippet', '')}\n"
            f"URL: {r.get('url', '')}\n"
        )
    research_text = sanitize_ingested_content(research_text, source="ddg_search")

    user_message = (
        f"POLICY TYPE: {policy_type}\n"
        f"COUNTRY: {details.get('country', 'unspecified')}\n"
        f"GOODS: {details.get('goods', 'unspecified')}\n"
        f"RATE: {details.get('rate', 'unspecified')}\n"
        f"AUTHORITY: {details.get('authority', 'unspecified')}\n\n"
        f"VALIDITY ASSESSMENT:\n{validity_assessment}\n\n"
        f"HISTORICAL ANALOGS:\n{historical_analogs}\n\n"
        f"GATHERED RESEARCH:\n{research_text}\n\n"
        f"Generate a full impact chain analysis tracing this policy from goods "
        f"through companies through margins, with 5 scenarios and positioning recommendations."
    )

    try:
        client = _get_client()
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, lambda: client.messages.create(
            model=config.default_model,
            max_tokens=2500,
            system=cached_system_block(_IMPACT_CHAIN_PROMPT),
            messages=[{"role": "user", "content": user_message}],
        ))

        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

        result = text.strip() if text.strip() else None
        if result:
            await budget_tracker.record_usage("trade_policy")
        return result

    except Exception as e:
        logger.error(f"Impact chain analysis generation failed: {e}", exc_info=True)
        return None


# ============================================================
# Research State Management
# ============================================================

def _load_research_state() -> dict:
    """Load trade policy research state."""
    default = {"researched_policies": []}
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
        logger.warning(f"Failed to load trade policy state: {e}")
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
        logger.error(f"Failed to save trade policy state: {e}")
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
    entries = state.get("researched_policies", [])
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
    state["researched_policies"] = pruned
    return state


def _already_researched(policy_type: str, country: str, state: dict) -> bool:
    """Check if this policy type + country was researched in the last 5 days."""
    cutoff = datetime.now(_NY) - timedelta(days=5)
    country_lower = country.lower()
    for entry in state.get("researched_policies", []):
        if entry.get("policy_type") != policy_type:
            continue
        if entry.get("country", "").lower() != country_lower:
            continue
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


# Goods values that mean "no goods category": _extract_policy_details writes 'unspecified goods' when none matched.
# The concrete-story gate is OR-logic, so 'unspecified goods' still passes when a rate is present; it is denied here.
_GOODS_DENY_TOKENS = frozenset({"unspecified", "unknown", "n/a", "none"})


def _stored_candidate_is_concrete(entry: dict) -> bool:
    """Re-applies _is_concrete_policy to what a stored researched_policies entry recorded (country, policy type,
    goods, and -- for entries written since R-17 -- rate and authority). A denied goods token fails regardless."""
    goods = str(entry.get("goods") or "").strip().lower()
    if not goods or goods in _GOODS_DENY_TOKENS or any(w in _GOODS_DENY_TOKENS for w in goods.split()):
        return False
    country = str(entry.get("country") or "")
    variant = next((k for k, v in _COUNTRY_VARIANTS.items() if v == country), None)
    if variant is None:
        return False
    parts = [variant, str(entry.get("policy_type") or "").replace("_", " "), "announced", goods]
    rate = str(entry.get("rate") or "")
    if rate and not rate.startswith("unspecified"):
        parts.append(rate)
    authority = str(entry.get("authority") or "")
    if authority and not authority.startswith("unspecified"):
        parts.append(authority)
    return _is_concrete_policy(" ".join(parts))


def _usable_research_state(state: dict) -> dict:
    """Read-time view of researched_policies for the escalation dedup. Nothing is removed from the file: a stored
    candidate that fails the concrete-story gate (or carries a denied goods token) is left out of THIS view only,
    and one INFO line is logged per call that drops any. Entries without a goods field (pre-goods shape) pass."""
    kept: list = []
    dropped: list[str] = []
    for entry in state.get("researched_policies", []):
        if not isinstance(entry, dict) or "goods" not in entry or _stored_candidate_is_concrete(entry):
            kept.append(entry)
        else:
            dropped.append(f"{entry.get('policy_type')}/{entry.get('country')}/{entry.get('goods')}")
    if dropped:
        logger.info(
            "Trade policy: dropped %d stored candidate(s) at read time (failed concrete-story gate or denied goods): %s",
            len(dropped), "; ".join(dropped),
        )
    return {**state, "researched_policies": kept}


# ============================================================
# TP-1 (S211): weekly forced check + heartbeat
# ============================================================
# The module is escalation-only. S166 forced an escalation when the state file was stale, but it handed over
# the topic's GENERAL analysis, which the specificity gate below always rejects -- so the state file never
# appeared and "no file" (MISSING) was read as "never checked". Two questions were conflated: "did anyone check
# recently?" and "was there something specific to analyse?". A forced check now runs its own targeted search and
# either escalates a genuinely specific story through the unchanged gate, or records a heartbeat
# ("checked, nothing specific") -- no model call, no Telegram message. The heartbeat writes a top-level
# last_updated (the sentinel's first-probed field) and refreshes the file mtime (news_intel's staleness check).

_FORCED_QUERIES = (
    "tariff announced OR executive order OR section 232 OR section 301 trade policy",
    "sanctions OR export controls OR trade restrictions announced",
)


def _record_check_sync(result: str, forced: bool = True) -> None:
    """Heartbeat: a check ran and found `result`. Blocking flock I/O -- call via run_in_executor."""
    state = _load_research_state()
    now = datetime.now(_NY).isoformat()
    state["last_checked"] = {"at": now, "result": result, "forced": bool(forced)}
    state["last_updated"] = now
    _save_research_state(_prune_research_state(state))


# The forced check must be STRICTER than the escalation gate. `_is_specific_policy` needs only a country plus one
# keyword, and "tariff" is in the search query itself, so almost any result that mentions a country passes. The
# first live check (2026-09-29) picked such a hit and produced a speculative "EU tariff, rate/goods/authority all
# unspecified" analysis that went to Telegram. A candidate now needs a country, a policy term, an event verb AND at
# least one concrete parameter (a rate, a recognised goods category, or a named legal instrument). The gate is
# self-contained rather than layered on `_is_specific_policy`, whose exact-form keywords ("tariff", not "tariffs";
# "announced", not "announces") reject most real headlines; escalate() skips that older gate for forced candidates.
_EVENT_VERBS = (
    "impose", "imposed", "imposes", "imposing", "announced", "announces", "signed", "signs",
    "effective", "implement", "implemented", "ban", "banned", "sanction", "sanctions", "sanctioned",
    "raise", "raised", "raises", "hike", "hiked", "hikes", "levy", "levied", "levies", "slap", "slapped", "slaps",
)
_POLICY_TERM = re.compile(
    r"\b(?:tariffs?|duty|duties|levy|levies|quotas?|embargo(?:es)?|sanctions?|export controls?|"
    r"(?:import|export) bans?|section\s+(?:122|201|232|301)|ieepa|executive order|proclamation)\b",
    re.IGNORECASE,
)
_NAMED_INSTRUMENT = re.compile(
    r"\bsection\s+(?:122|201|232|301)\b|\bieepa\b|\bexecutive order\b|\bproclamation\b", re.IGNORECASE
)


def _is_concrete_policy(text: str) -> bool:
    """A specific policy story with something to analyse (see the note above). Forced check only."""
    text_lower = text.lower()
    if not any(re.search(r"\b" + re.escape(v) + r"\b", text_lower) for v in _COUNTRY_VARIANTS):
        return False
    if _POLICY_TERM.search(text) is None:
        return False
    if not any(re.search(r"\b" + verb + r"\b", text_lower) for verb in _EVENT_VERBS):
        return False
    details = _extract_policy_details(text)
    return (
        details["rate"] != "unspecified rate"
        or details["goods"] != "unspecified goods"
        or _NAMED_INSTRUMENT.search(text) is not None
    )


async def _forced_policy_pick() -> tuple[str, list[dict]] | None:
    """Targeted search for a concrete policy action. Returns (text, results) for the first hit that passes
    _is_concrete_policy (result first), or None when nothing concrete turned up."""
    date_suffix = datetime.now(_NY).strftime("%B %d %Y")
    gathered: list[dict] = []
    for query in _FORCED_QUERIES:
        try:
            allowed, _msg = await budget_tracker.check_budget("report_web_search")
            if not allowed:
                break
            from agent.integrations.web_search import search_web
            hits = await search_web(f"{query} {date_suffix}", max_results=5)
            await budget_tracker.record_usage("report_web_search")
        except BudgetExhaustedError:
            break
        except Exception as e:
            logger.warning(f"Trade policy forced check query failed: {e}")
            continue
        for r in (hits or []):
            title = r.get("title", "")
            if title:
                gathered.append({
                    "title": title,
                    "snippet": r.get("snippet", r.get("body", "")),
                    "url": r.get("url", r.get("href", "")),
                    "query_type": "forced_check",
                })
        await asyncio.sleep(0.5)
    try:
        from agent.integrations.news_update import _url_is_stale, _snippet_is_stale
        gathered = [
            r for r in gathered
            if not _url_is_stale(r.get("url", ""))
            and not _snippet_is_stale(r.get("title", ""), r.get("snippet", ""))
        ]
    except ImportError:
        pass
    for r in gathered:
        text = f"{r['title']}. {r['snippet']}"
        if _is_concrete_policy(text):
            return text, [r] + [g for g in gathered if g is not r]
    return None


# ============================================================
# Delivery
# ============================================================

async def _deliver_trade_policy_intel(
    context,
    policy_type: str,
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
            policy_label = policy_type.replace("_", " ").title()
            msg = (
                f"TRADE POLICY INTELLIGENCE: {policy_label}\n"
                f"{now.strftime('%I:%M %p ET')}\n\n"
                f"{analysis}"
            )
            if len(msg) > 4000:
                msg = msg[:3997] + "..."
            for cid in chat_ids:
                await _safe_send(context.bot, cid, text=msg)
    except Exception as e:
        logger.warning(f"Failed to send trade policy intel to Telegram: {e}")

    # 2. Knowledge base
    try:
        from agent.tools.knowledge_tools import handle_save_knowledge
        title = f"Trade Policy: {policy_type.replace('_', ' ').title()} — {now.strftime('%b %d %Y')}"
        source_urls = ", ".join(urls[:3]) if urls else "DuckDuckGo research"
        await handle_save_knowledge({
            "title": title,
            "content": analysis,
            "category": "trading_market_analysis",
            "tags": ["trade_policy", policy_type, now.strftime("%Y-%m-%d")],
            "source": source_urls,
        })
    except Exception as e:
        logger.warning(f"Failed to save trade policy intel to knowledge: {e}")

    # 3. Bridge write
    try:
        os.makedirs(BRIDGE_DIR, exist_ok=True)
        filename = f"{timestamp}_trade_policy_{policy_type}.json"
        entry = {
            "id": f"trade_policy_{timestamp}_{policy_type}",
            "type": "trade_policy",
            "policy_type": policy_type,
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
        logger.warning(f"Failed to write trade policy intel to bridge: {e}")


# ============================================================
# Public Entry Point
# ============================================================

async def escalate(context, analysis_text: str, results: list[dict], forced: bool = False) -> None:
    """Escalation entry point — called from news_intel.py topic monitor.

    Args:
        context: Telegram bot context (for sending messages)
        analysis_text: Sonnet headline analysis from topic monitor
        results: Raw DuckDuckGo search results list
        forced: TP-1 weekly check (news_intel). analysis_text/results are ignored: a targeted search runs
            instead, and a heartbeat is recorded so "checked" is distinguishable from "never checked".
    """
    try:
        logger.info("Trade policy escalation triggered")

        if forced:
            loop = asyncio.get_running_loop()
            picked = await _forced_policy_pick()
            if picked is None:
                await loop.run_in_executor(None, _record_check_sync, "none_specific", True)
                logger.info("Trade policy weekly check: no specific policy action found -- heartbeat recorded")
                return
            # heartbeat first: if the full pipeline below exits early (already researched, no results,
            # model failure) the check still counts as done; a success then appends to this same state.
            await loop.run_in_executor(None, _record_check_sync, "candidate_found", True)
            analysis_text, results = picked
            logger.info("Trade policy weekly check: specific policy candidate found -- running the full pipeline")

        # Validate this is about a SPECIFIC policy action (a forced candidate already passed the stricter
        # _is_concrete_policy in _forced_policy_pick)
        if not forced and not _is_specific_policy(analysis_text):
            # TP-1 (S211): ADR-002 -- an upstream-classification miss is INFO, not DEBUG. The S166
            # forced escalation (state >14 d stale) fires only when news_intel's keyword gate FAILED,
            # then lands here (country + action verb required), so trade_policy_state.json was never
            # produced and nothing said why.
            logger.info(
                "Trade policy escalation skipped: analysis is general trade discussion, "
                "not a specific policy action (country + action verb required)"
            )
            return

        # 1. Classify policy type
        policy_type = classify_policy(analysis_text)
        logger.info(f"Policy type: {policy_type}")

        # 2. Extract details
        details = _extract_policy_details(analysis_text)
        logger.info(f"Policy details: {details}")

        # 3. Check dedup
        state = _load_research_state()
        state = _prune_research_state(state)

        if _already_researched(policy_type, details["country"], _usable_research_state(state)):
            logger.debug(f"Already researched {policy_type}/{details['country']} recently, skipping")
            return

        # 4. Gather impact chain research (A)
        research_results = await _gather_impact_chain_research(
            policy_type, details["country"], details["goods"]
        )

        # Merge original results (pre-filtered by news_intel)
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
            # S186 W1: visibility on upstream-empty (S185-B G17)
            logger.info(f"Trade policy escalation exiting: 0 research results for {policy_type}/{details.get('country')}")
            return

        # 5. Recall historical analogs (B)
        historical_analogs = _recall_historical_analogs(details["country"], policy_type)

        # 6. Assess policy validity (C)
        validity_assessment = _assess_policy_validity(policy_type, details)

        # 7. Generate impact chain analysis (D)
        analysis = await _generate_impact_chain_analysis(
            policy_type=policy_type,
            details=details,
            research_results=research_results,
            historical_analogs=historical_analogs,
            validity_assessment=validity_assessment,
        )

        if not analysis:
            # S186 W1: surface Anthropic failures
            logger.warning(f"Trade policy escalation: no impact chain analysis generated for {policy_type} (Anthropic failure?)")
            return

        # 8. Deliver
        urls = [r.get("url", "") for r in research_results if r.get("url")]
        await _deliver_trade_policy_intel(context, policy_type, analysis, urls)

        # 9. Record in state
        state["researched_policies"].append({
            "policy_type": policy_type,
            "country": details["country"],
            "goods": details["goods"],
            "rate": details["rate"],
            "authority": details["authority"],
            "researched_at": datetime.now(_NY).isoformat(),
            "findings_count": len(research_results),
        })
        _save_research_state(state)

        logger.info(f"Trade policy escalation complete: {policy_type}, {len(research_results)} findings")

    except Exception as e:
        logger.error(f"Trade policy escalation failed: {e}", exc_info=True)
