"""
OpenClaw/Moltbot — Treasury Auction Intelligence (Session 92, Phase 2)

Tracks auction dynamics, foreign demand shifts, primary dealer positioning,
and produces auction demand scenario analysis via the A/B/C/D framework:
  A) Multi-source research — auction results, bid-to-cover, indirect/direct
     bidder %, TIC data, primary dealer positioning, repo rates, term premium
  B) Memory recall — auction trend tracking (last 3 auctions per maturity)
  C) Auction validity — quarter-end distortion, year-end window dressing,
     Fed QT supply impact, primary dealer obligation, basis trade unwind risk
  D) Scenario analysis — 5 demand scenarios (STRONG_DEMAND / ADEQUATE / SOFT /
     WEAK / FAILED) with bid-to-cover ranges, tail sizes, yield impacts

Two triggers:
1. Scheduled callback — Tue/Wed/Thu 8:00 AM ET (Treasury auction days)
2. Escalation from news_intel.py when bonds_treasuries topic produces auction keywords
"""

import asyncio
import calendar
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

logger = logging.getLogger("openclaw.bond_intel")

_NY = ZoneInfo("America/New_York")

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "intelligence",
)
_STATE_PATH = os.path.join(_DATA_DIR, "bond_intel_state.json")
_CALENDAR_PATH = os.path.join(_DATA_DIR, "auction_calendar.json")

BRIDGE_DIR = os.path.join(config.tradingbot_dir, "moltbot_knowledge", "intelligence")

_cached_client: anthropic.Anthropic | None = None
_client_lock = __import__("threading").Lock()


# ============================================================
# CC-CLIO-FD-4: Financial Datasets Rate Context for Bond Analysis
# ============================================================

def _fd_rate_context() -> str:
    """Fetch current Fed rate + recent history for bond auction analysis context."""
    if os.environ.get("CLIO_FD_RATE_CONTEXT", "0") != "1":
        return ""
    try:
        from agent.integrations.fd_client import get_fd_client
        client = get_fd_client()

        snap = client.get_interest_rates_snapshot("FED")
        if not snap or (isinstance(snap, dict) and "_error" in snap):
            return ""

        rate = snap.get("rate", "N/A")
        # BC#114: Snapshot uses "date" field (not "effective_date" — that's historical)
        eff_date = snap.get("date", snap.get("effective_date", "N/A"))

        # BC#113: ASCENDING sort — use [-N:] for most recent
        end = datetime.now(_NY).strftime("%Y-%m-%d")
        start = (datetime.now(_NY) - timedelta(days=365)).strftime("%Y-%m-%d")
        history = client.get_interest_rates("FED", start_date=start, end_date=end)

        hist_lines = []
        if history and isinstance(history, list):
            for h in history[-5:]:
                if isinstance(h, dict):
                    hist_lines.append(f"  {h.get('effective_date', '?')}: {h.get('rate', '?')}%")

        result = f"Current Fed Funds Rate: {rate}% (effective {eff_date})"
        if hist_lines:
            result += "\nRecent Changes:\n" + "\n".join(hist_lines)

        logger.debug("CC-CLIO-FD-4: Bond rate context injected (%s%%)", rate)
        return result
    except Exception as e:
        logger.warning("CC-CLIO-FD-4: Bond rate context failed: %s", e)
        return ""


def _get_client() -> anthropic.Anthropic:
    global _cached_client
    if _cached_client is None:
        with _client_lock:
            if _cached_client is None:
                _cached_client = anthropic.Anthropic(**config.anthropic_client_kwargs(timeout=120))
    return _cached_client


# ============================================================
# Research Query Maps
# ============================================================

BOND_RESEARCH_MAP: dict[str, dict] = {
    "auction_analysis": {
        "keywords": ["treasury auction", "bond auction", "bid to cover", "tailed",
                      "indirect bidder", "primary dealer"],
        "obvious_queries": [
            "treasury auction results latest",
            "treasury auction schedule this week",
            "bid-to-cover ratio trend treasury",
        ],
        "niche_queries": [
            "indirect bidder percentage trend foreign demand treasury",
            "primary dealer forced take-down treasury auction",
            "when-issued trading yield treasury",
            "treasury futures positioning CFTC commitments of traders",
            "TIC data foreign holdings US treasury Japan China",
            "quarterly refunding announcement size treasury",
            "repo rate SOFR overnight funding",
            "reverse repo RRP facility balance",
            "money market fund AUM treasury bills",
            "dealer balance sheet capacity treasury",
            "term premium ACM model treasury",
            "TIPS breakeven inflation expectations",
        ],
        "validity_queries": [
            "quarter-end regulatory treasury demand distortion",
            "year-end window dressing treasury",
            "Fed QT reinvestment cap treasury supply",
            "primary dealer obligation bid treasury auction",
            "basis trade hedge fund treasury futures leverage",
            "settlement date effects treasury demand",
        ],
    },
    "yield_curve": {
        "keywords": ["yield curve", "2-10 spread", "inversion", "steepening", "flattening"],
        "obvious_queries": [
            "yield curve shape latest US treasury",
            "2s10s spread current treasury",
            "yield curve inversion recession indicator history",
        ],
        "niche_queries": [
            "yield curve un-inversion steepening analysis",
            "real rate decomposition TIPS nominal",
            "3-month 10-year recession indicator probability",
            "term premium expectations component risk premium",
        ],
        "validity_queries": [
            "QE QT yield curve distortion signal reliability",
            "post-2020 yield curve recession signal accuracy",
        ],
    },
}


# ============================================================
# Auction Validity Checks
# ============================================================

AUCTION_VALIDITY_CHECKS: dict[str, dict] = {
    "quarter_end": {
        "description": "Last 2 weeks of Mar/Jun/Sep/Dec",
        "impact": "Demand overstated 5-10% due to GSIB capital requirements",
        "discount": "5-10%",
    },
    "year_end": {
        "description": "December auctions",
        "impact": "Demand 10-15% above sustainable due to window dressing",
        "discount": "10-15%",
    },
    "fed_qt": {
        "description": "$25B/month treasuries not rolled by Fed",
        "impact": "Adds private sector absorption need, structural supply pressure",
        "discount": "Ongoing supply headwind",
    },
    "primary_dealer_obligation": {
        "description": "Primary dealer take-down exceeds 25%",
        "impact": "Weak real demand — dealers forced to absorb, not voluntary",
        "discount": "Demand quality reduced",
    },
    "basis_trade": {
        "description": "Leveraged fund positioning at extremes in cash-futures basis",
        "impact": "Synthetic demand vulnerable to unwind (March 2020 precedent)",
        "discount": "Reversal risk elevated",
    },
    "settlement_timing": {
        "description": "Month-end T-bill demand",
        "impact": "Cash management demand, not structural — reverses 5-10 business days later",
        "discount": "Temporary demand boost",
    },
}


# ============================================================
# Quarter-End Detection
# ============================================================

def _is_quarter_end_window() -> bool:
    """Returns True if within last 14 days of a quarter (Mar/Jun/Sep/Dec)."""
    today = datetime.now(_NY).date()
    month = today.month
    day = today.day

    # Quarter-end months
    if month not in (3, 6, 9, 12):
        return False

    # Days in each quarter-end month
    days_in_month = {3: 31, 6: 30, 9: 30, 12: 31}
    last_day = days_in_month[month]

    return day > (last_day - 14)


# ============================================================
# Auction Calendar
# ============================================================

def _write_auction_cache(cache_data: dict) -> None:
    """Sync: atomically write auction calendar cache to disk under flock. Must not be called on event loop thread."""
    os.makedirs(_DATA_DIR, exist_ok=True)
    lock_fd = None
    try:
        lock_fd = open(_CALENDAR_PATH + ".lock", "a")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        tmp = _CALENDAR_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cache_data, f, indent=2, default=str)
        os.replace(tmp, _CALENDAR_PATH)
    finally:
        if lock_fd:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
            except Exception:
                pass


async def _refresh_auction_calendar() -> list[dict]:
    """Search DDG for Treasury auction schedule. Cache weekly (refresh Monday).

    Returns list of {maturity, date, size_B, time_et} (best-effort parsing).
    """
    now = datetime.now(_NY)

    # Check cache: refresh only on Monday or if stale (> 7 days)
    if os.path.isfile(_CALENDAR_PATH):
        try:
            with open(_CALENDAR_PATH, "r") as f:
                cached = json.load(f)
            cached_at = cached.get("refreshed_at", "")
            if cached_at:
                dt = datetime.fromisoformat(cached_at)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=_NY)
                age_days = (now - dt).total_seconds() / 86400
                # If less than 5 days old AND today is not Monday, use cache
                if age_days < 5 and now.weekday() != 0:
                    return cached.get("auctions", [])
        # TypeError included: fromisoformat raises TypeError (not ValueError) on a
        # non-string on-disk value (e.g. refreshed_at:12345). This is a skip-if-fresh
        # gate for an advisory refresh → must fail OPEN (fall through to fetch fresh),
        # never abort and skip the day's analysis.
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            pass

    # Fetch fresh calendar
    try:
        allowed, msg = await budget_tracker.check_budget("report_web_search")
        if not allowed:
            return _load_cached_calendar()
        from agent.integrations.web_search import search_web
        date_str = now.strftime("%B %d %Y")
        results = await search_web(
            f"US Treasury auction schedule this week {date_str}",
            max_results=5,
        )
        await budget_tracker.record_usage("report_web_search")  # S118 W-010: charge AFTER search

        # We can't reliably parse structured auction data from search snippets,
        # but we store the raw results for the Opus prompt to interpret
        auctions = []
        for r in (results or []):
            snippet = r.get("snippet", r.get("body", ""))
            title = r.get("title", "")
            url = r.get("url", r.get("href", ""))
            if title:
                auctions.append({
                    "title": title,
                    "snippet": snippet,
                    "url": url,
                })

        # Cache
        cache_data = {
            "refreshed_at": now.isoformat(),
            "auctions": auctions,
        }
        await asyncio.get_running_loop().run_in_executor(None, _write_auction_cache, cache_data)

        return auctions

    except BudgetExhaustedError:
        return _load_cached_calendar()
    except Exception as e:
        logger.warning(f"Failed to refresh auction calendar: {e}")
        return _load_cached_calendar()


def _load_cached_calendar() -> list[dict]:
    """Load cached auction calendar if available."""
    if not os.path.isfile(_CALENDAR_PATH):
        return []
    lock_fd = None
    try:
        lock_fd = open(_CALENDAR_PATH + ".lock", "a")
        fcntl.flock(lock_fd, fcntl.LOCK_SH)
        with open(_CALENDAR_PATH, "r") as f:
            data = json.load(f)
        return data.get("auctions", [])
    except (json.JSONDecodeError, OSError):
        return []
    finally:
        if lock_fd:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
            except Exception:
                pass


# ============================================================
# Memory Recall — Auction History
# ============================================================

def _recall_auction_history(maturity: str) -> str:
    """Search knowledge base for prior auction entries tagged bond_intel.

    Returns formatted trend string, e.g.:
    "10Y B/C: 2.6x (Dec) -> 2.5x (Jan) -> 2.3x (Feb) -- declining demand"
    """
    try:
        from agent.tools.knowledge_tools import load_all_entries
        all_entries = load_all_entries()
    except Exception as e:
        logger.debug(f"Failed to load entries for auction history: {e}")
        return "No auction history available (knowledge base unavailable)."

    cutoff = datetime.now(_NY) - timedelta(days=120)
    relevant = []

    maturity_lower = maturity.lower()
    search_terms = ["bond_intel", "auction", maturity_lower]

    for entry in all_entries:
        tags = entry.get("tags", [])
        title = entry.get("title", "").lower()
        combined = " ".join(str(t) for t in tags).lower() + " " + title if isinstance(tags, list) else title

        has_bond = any(term in combined for term in search_terms)
        if not has_bond:
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
        return f"No prior auction history for {maturity} in knowledge base."

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
                logger.debug(f"Failed to read knowledge entry file: {e}")
        lines.append(f"[{created}] {title}\n{content}")

    return "\n---\n".join(lines)


# ============================================================
# Demand Distortion Detection
# ============================================================

def _detect_demand_distortion() -> str:
    """Check all validity factors and return formatted distortion assessment."""
    now = datetime.now(_NY)
    distortions = []

    # Quarter-end
    if _is_quarter_end_window():
        check = AUCTION_VALIDITY_CHECKS["quarter_end"]
        distortions.append(
            f"QUARTER-END ACTIVE: {check['description']}. {check['impact']}. "
            f"Discount demand by {check['discount']}."
        )

    # Year-end
    if now.month == 12:
        check = AUCTION_VALIDITY_CHECKS["year_end"]
        distortions.append(
            f"YEAR-END ACTIVE: {check['description']}. {check['impact']}. "
            f"Discount demand by {check['discount']}."
        )

    # Fed QT (ongoing)
    qt_check = AUCTION_VALIDITY_CHECKS["fed_qt"]
    distortions.append(f"FED QT: {qt_check['description']}. {qt_check['impact']}.")

    # Settlement timing (month-end)
    day = now.day
    if now.month in (1, 3, 5, 7, 8, 10, 12):
        last_day = 31
    elif now.month in (4, 6, 9, 11):
        last_day = 30
    else:
        last_day = 29 if calendar.isleap(now.year) else 28
    if day > (last_day - 5):
        check = AUCTION_VALIDITY_CHECKS["settlement_timing"]
        distortions.append(f"MONTH-END: {check['description']}. {check['impact']}.")

    if not distortions:
        return "No major demand distortion factors detected. Fed QT remains ongoing supply headwind."

    return "\n".join(distortions)


# ============================================================
# Auction Analysis Prompt (Opus)
# ============================================================

_AUCTION_PROMPT = """You are a Treasury auction intelligence analyst producing demand scenario analysis for a trading system.

Analyze the provided auction data, historical trends, and distortion factors.

FORMAT:

AUCTION: {maturity, estimated size, date, time if known}

PRE-AUCTION POSITIONING:
- When-issued trading yield vs prior auction
- CFTC treasury futures net positioning
- Primary dealer inventory levels

DEMAND COMPOSITION TREND (last 3 auctions if available):
| Auction | B/C Ratio | Indirect % | Direct % | Tail (bp) |
(Format as plain text table)

FOREIGN DEMAND ASSESSMENT:
- TIC data trends (Japan, China holdings — note 6-8 week lag)
- Indirect bidder trajectory as proxy for real-time foreign demand

SUPPLY CONTEXT:
- Quarterly refunding changes
- Fed QT impact ($25B/month treasury runoff)
- Net new issuance vs rolling maturities

DEMAND DISTORTION ALERT:
{List any active distortion factors from the provided assessment}

5 DEMAND SCENARIOS (probabilities must sum to ~100%):

STRONG_DEMAND (X%): B/C > 2.7x, tail < 0.5bp, indirect > 70%
  Yield impact: {specific bp move}
  Equity impact: {SPY/QQQ reaction}
  Duration: {how long does the bid persist?}

ADEQUATE (X%): B/C 2.3-2.7x, tail 0-1bp, indirect 60-70%
  Yield impact: {bp move}
  Equity impact: {SPY/QQQ}

SOFT (X%): B/C 2.0-2.3x, tail 1-2bp, indirect 55-60%
  Yield impact: {bp move}
  Equity impact: {SPY/QQQ}

WEAK (X%): B/C 1.5-2.0x, tail 2-4bp, indirect < 55%
  Yield impact: {bp move}
  Equity impact: {SPY/QQQ}

FAILED (X%): B/C < 1.5x, tail > 4bp, massive primary dealer take-down
  Yield impact: {bp move}
  Equity impact: {SPY/QQQ}
  Contagion: {other assets affected}

CURVE IMPACT: {steepen/flatten, which part of curve moves most}

FOLLOW-THROUGH: {does weak/strong demand persist 24-48 hours? Historical pattern}

KEY SIGNAL: {single most important indicator to watch}
CONFIDENCE: {LOW/MEDIUM/HIGH} — {brief justification}

RULES:
- Every sentence must contain a specific fact: a number, a name, a date, or a price level
- NEVER use: "monitor closely", "watch for", "keep an eye on", "worth noting", "remains to be seen"
- If quarter-end, explicitly discount demand by 5-10%
- Primary dealer take-down > 25% = weak demand signal — flag this
- Bid-to-cover > 2.5x is healthy for 10Y. Below 2.0x is concerning. Below 1.5x is crisis.
- A tail of 2bp+ is notable. Negative tail (through) = very strong demand.
- Write in plain text only, no markdown formatting"""


# ============================================================
# Research Pipeline
# ============================================================

async def _gather_bond_research(research_type: str = "auction_analysis") -> list[dict]:
    """Execute research queries for bond/auction analysis.

    Uses report_web_search budget. Returns list of {title, snippet, url, query_type}.
    """
    research_config = BOND_RESEARCH_MAP.get(research_type, BOND_RESEARCH_MAP["auction_analysis"])
    obvious = research_config.get("obvious_queries", [])
    niche = research_config.get("niche_queries", [])

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
            logger.debug("Budget exhausted during bond research")
            break
        except Exception as e:
            logger.warning(f"Bond research query failed: {query}: {e}")

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
# Opus Analysis Generation
# ============================================================

async def _generate_auction_analysis(
    research_results: list[dict],
    auction_history: str,
    distortion_assessment: str,
    calendar_info: list[dict],
) -> str | None:
    """Generate auction demand scenario analysis using Claude Opus. Budget: bond_intel."""
    try:
        allowed, msg = await budget_tracker.check_budget("bond_intel")
        if not allowed:
            logger.debug("bond_intel budget exhausted")
            return None
    except BudgetExhaustedError:
        return None

    # Format research
    research_text = ""
    for i, r in enumerate(research_results[:15], 1):
        qt = r.get("query_type", "")
        research_text += (
            f"\n[{i}] ({qt}) {r['title']}\n"
            f"{r.get('snippet', '')}\n"
            f"URL: {r.get('url', '')}\n"
        )

    # Format calendar info
    calendar_text = ""
    if calendar_info:
        for c in calendar_info[:5]:
            calendar_text += f"- {c.get('title', '')}: {c.get('snippet', '')[:200]}\n"
    else:
        calendar_text = "No auction calendar data available."

    now = datetime.now(_NY)

    # CC-CLIO-FD-4: Inject current rate context for bond auction analysis (run_in_executor)
    fd_rate_section = ""
    try:
        _fd_ok, _ = await budget_tracker.check_budget("financial_datasets")
        if _fd_ok:
            _loop = asyncio.get_running_loop()
            fd_rate_ctx = await _loop.run_in_executor(None, _fd_rate_context)
            if fd_rate_ctx:
                await budget_tracker.record_usage("financial_datasets")
        else:
            fd_rate_ctx = ""
    except Exception as e:
        logger.debug("CC-CLIO-FD-4: Bond rate context fetch failed: %s", e)
        fd_rate_ctx = ""
    if fd_rate_ctx:
        fd_rate_section = f"\nCURRENT RATE DATA (Financial Datasets API):\n{fd_rate_ctx}\n\n"

    user_message = (
        f"DATE: {now.strftime('%A, %B %d, %Y %I:%M %p ET')}\n\n"
        + (fd_rate_section if fd_rate_section else "")
        + f"AUCTION CALENDAR INFO:\n{calendar_text}\n\n"
        f"PRIOR AUCTION HISTORY:\n{auction_history}\n\n"
        f"DEMAND DISTORTION ASSESSMENT:\n{distortion_assessment}\n\n"
        f"GATHERED RESEARCH:\n{sanitize_ingested_content(research_text, source='ddg_bond')}\n\n"
        f"Generate a 5-scenario demand analysis (STRONG_DEMAND / ADEQUATE / SOFT / WEAK / FAILED) "
        f"with specific bid-to-cover ranges, tail sizes, yield impacts, and equity reactions."
    )

    try:
        client = _get_client()
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, lambda: client.messages.create(
            model=config.default_model,
            max_tokens=2500,
            system=cached_system_block(_AUCTION_PROMPT),
            messages=[{"role": "user", "content": user_message}],
        ))

        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

        result = text.strip()
        if result:
            await budget_tracker.record_usage("bond_intel")
            return result
        return None

    except Exception as e:
        logger.error(f"Auction analysis generation failed: {e}", exc_info=True)
        return None


# ============================================================
# Delivery
# ============================================================

async def _deliver_bond_intel(
    context,
    analysis: str,
    urls: list[str],
) -> None:
    """Deliver bond intel: Telegram + knowledge commit + bridge write."""
    now = datetime.now(_NY)
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    # 1. Telegram
    try:
        from agent.integrations.scheduled_tasks import _get_chat_ids, _safe_send
        chat_ids = _get_chat_ids()
        if chat_ids:
            msg = (
                f"BOND AUCTION INTELLIGENCE\n"
                f"{now.strftime('%I:%M %p ET')}\n\n"
                f"{analysis}"
            )
            if len(msg) > 4000:
                msg = msg[:3997] + "..."
            for cid in chat_ids:
                await _safe_send(context.bot, cid, text=msg)
    except Exception as e:
        logger.warning(f"Failed to send bond intel to Telegram: {e}")

    # 2. Knowledge base
    try:
        from agent.tools.knowledge_tools import handle_save_knowledge
        title = f"Bond Auction Intel — {now.strftime('%b %d %Y')}"
        source_urls = ", ".join(urls[:3]) if urls else "DuckDuckGo research"
        await handle_save_knowledge({
            "title": title,
            "content": analysis,
            "category": "trading_market_analysis",
            "tags": ["bond_intel", "auction", now.strftime("%Y-%m-%d")],
            "source": source_urls,
        })
    except Exception as e:
        logger.warning(f"Failed to save bond intel to knowledge: {e}")

    # 3. Bridge write
    try:
        os.makedirs(BRIDGE_DIR, exist_ok=True)
        filename = f"{timestamp}_bond_auction.json"
        entry = {
            "id": f"bond_{timestamp}",
            "type": "bond_intel",
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
        logger.warning(f"Failed to write bond intel to bridge: {e}")


# ============================================================
# State Management (1-day dedup per maturity/date)
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
        logger.error(f"Failed to save bond intel state: {e}")
    finally:
        if lock_fd:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
            except Exception:
                pass


def _already_analyzed_today(state: dict) -> bool:
    """Check if we've already run bond analysis today."""
    today_str = datetime.now(_NY).strftime("%Y-%m-%d")
    for entry in state.get("analyses", []):
        ts = entry.get("analyzed_at", "")
        if ts and ts[:10] == today_str:
            return True
    return False


def _record_analysis(state: dict) -> None:
    """Record that we ran bond analysis."""
    state.setdefault("analyses", [])
    state["analyses"].append({
        "analyzed_at": datetime.now(_NY).isoformat(),
    })
    # Prune old (keep last 50)
    if len(state["analyses"]) > 50:
        state["analyses"] = state["analyses"][-50:]


# ============================================================
# Public Entry Points
# ============================================================

async def escalate(context, analysis_text: str, results: list[dict]) -> None:
    """Escalation entry point from news_intel.py.

    Called when bonds_treasuries topic produces auction keywords.
    """
    logger.info("Bond intel escalation triggered")

    # Dedup
    state = _load_state()
    if _already_analyzed_today(state):
        logger.debug("Already ran bond analysis today, skipping escalation")
        return

    # Determine research type
    text_lower = analysis_text.lower()
    if any(kw in text_lower for kw in ("yield curve", "2-10 spread", "inversion",
                                        "steepening", "flattening")):
        research_type = "yield_curve"
    else:
        research_type = "auction_analysis"

    # A) Gather research
    research_results = await _gather_bond_research(research_type)

    # Include original escalation results
    # S100: defense-in-depth staleness check on incoming results
    try:
        from agent.integrations.news_update import _url_is_stale, _snippet_is_stale
        for r in results[:3]:
            url = r.get("url", r.get("href", ""))
            title = r.get("title", "")
            snippet = r.get("snippet", r.get("body", ""))
            if not _url_is_stale(url) and not _snippet_is_stale(title, snippet):
                research_results.append({
                    "title": title, "snippet": snippet,
                    "url": url, "query_type": "escalation_source",
                })
    except ImportError:
        for r in results[:3]:
            research_results.append({
                "title": r.get("title", ""),
                "snippet": r.get("snippet", r.get("body", "")),
                "url": r.get("url", r.get("href", "")),
                "query_type": "escalation_source",
            })

    if not research_results:
        logger.debug("No research results for bond escalation")
        return

    # B) Recall auction history
    auction_history = _recall_auction_history("10Y")  # Default to 10Y for escalation

    # C) Detect demand distortion
    distortion = _detect_demand_distortion()

    # Get calendar info
    calendar_info = await _refresh_auction_calendar()

    # D) Generate analysis
    analysis = await _generate_auction_analysis(
        research_results=research_results,
        auction_history=auction_history,
        distortion_assessment=distortion,
        calendar_info=calendar_info,
    )

    if not analysis:
        logger.debug("No analysis generated for bond escalation")
        return

    # Deliver
    urls = [r.get("url", "") for r in research_results if r.get("url")]
    await _deliver_bond_intel(context, analysis, urls)

    # Record
    _record_analysis(state)
    _save_state(state)

    logger.info("Bond intel escalation complete")


_GATE_SKIP_LOGGED: set[tuple[str, str]] = set()


def _log_gate_skip_once(now: datetime, msg: str) -> None:
    """INFO-log a silent run_bond_intel gate skip at most once per ET calendar day.

    The job fires daily, so an unconditional log would repeat every day on the skipped
    weekdays. Entries from earlier days are pruned so the set stays bounded.
    """
    today = now.date().isoformat()
    for key in [k for k in _GATE_SKIP_LOGGED if k[0] != today]:
        _GATE_SKIP_LOGGED.discard(key)
    key = (today, msg)
    if key in _GATE_SKIP_LOGGED:
        return
    _GATE_SKIP_LOGGED.add(key)
    logger.info(msg)


async def run_bond_intel(context) -> None:
    """Scheduled callback — daily 8:00 AM ET, Tue-Thu only.

    Checks auction calendar for auctions today and runs analysis.
    """
    if not config.news_intel_enabled:
        return

    now = datetime.now(_NY)

    # Time guard: Tue(1), Wed(2), Thu(3) only, 6-10 AM ET
    if now.weekday() not in (1, 2, 3):
        _log_gate_skip_once(now, "bond_intel skipped: weekday gate (Tue-Thu only)")
        # Monday: refresh calendar only
        if now.weekday() == 0 and 6 <= now.hour < 10:
            try:
                await _refresh_auction_calendar()
                logger.debug("Monday auction calendar refresh complete")
            except Exception as e:
                logger.warning(f"Monday calendar refresh failed: {e}")
        return

    if now.hour < 6 or now.hour >= 10:
        _log_gate_skip_once(now, "bond_intel skipped: outside 06:00-10:00 ET window")
        return

    logger.info("Starting scheduled bond auction intelligence...")

    try:
        # Dedup
        state = _load_state()
        if _already_analyzed_today(state):
            logger.debug("Already ran bond analysis today, skipping")
            return

        # Get auction calendar
        calendar_info = await _refresh_auction_calendar()

        # A) Gather research
        research_results = await _gather_bond_research("auction_analysis")
        if not research_results:
            # S186 W1: visibility on upstream-empty (S185-B G17)
            logger.info("Bond intel exiting: 0 research results for scheduled auction analysis")
            return

        # B) Recall auction history (search for common maturities)
        auction_history_parts = []
        for maturity in ("10Y", "30Y", "2Y", "5Y"):
            history = _recall_auction_history(maturity)
            if "No prior" not in history and "unavailable" not in history:
                auction_history_parts.append(f"{maturity}:\n{history}")

        auction_history = "\n\n".join(auction_history_parts) if auction_history_parts else (
            "No prior auction history in knowledge base."
        )

        # C) Demand distortion
        distortion = _detect_demand_distortion()

        # D) Generate analysis
        analysis = await _generate_auction_analysis(
            research_results=research_results,
            auction_history=auction_history,
            distortion_assessment=distortion,
            calendar_info=calendar_info,
        )

        if not analysis:
            # S186 W1: surface Anthropic failures
            logger.warning("Bond intel: no auction analysis generated (Anthropic failure?)")
            return

        urls = [r.get("url", "") for r in research_results if r.get("url")]
        await _deliver_bond_intel(context, analysis, urls)

        _record_analysis(state)
        _save_state(state)

        logger.info("Scheduled bond auction intelligence complete")

    except Exception as e:
        logger.error(f"Scheduled bond intel failed: {e}", exc_info=True)
