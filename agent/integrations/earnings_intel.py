"""
OpenClaw/Moltbot — Earnings Season Intelligence (Session 91 Phase 2)

Tracks bellwether companies, uses supply chain leading indicators,
validates accounting comparability, and produces beat/miss scenario analysis.

A/B/C/D framework applied to earnings:
  A) Multi-source research — consensus estimates + supply chain leading indicators + channel checks
  B) Memory recall — prior guidance revisions, management credibility, sector margin trends
  C) Data validity — accounting standard changes, one-time items, SBC, channel stuffing, FY mismatches
  D) Scenario analysis — MISS_BIG / MISS_SLIGHT / IN_LINE / BEAT_SLIGHT / BEAT_BIG

Runs daily at 7:15 AM ET weekdays. NOT triggered by news_intel.py escalation.
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

logger = logging.getLogger("openclaw.earnings_intel")

_NY = ZoneInfo("America/New_York")

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "intelligence",
)
CALENDAR_PATH = os.path.join(_DATA_DIR, "earnings_calendar.json")
DEDUP_PATH = os.path.join(_DATA_DIR, "earnings_dedup.json")

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
# Supply Chain Map + Accounting Red Flags
# ============================================================

EARNINGS_SUPPLY_CHAIN: dict[str, dict] = {
    "NVDA": {
        "leading_indicators": ["TSMC earnings", "SK Hynix HBM shipments", "ASML orders"],
        "channel_checks": ["NVIDIA GPU cloud pricing", "data center backlog reports"],
        "sector": "semiconductors",
        "contagion": ["AMD", "MRVL", "AVGO", "MU"],
        "fiscal_note": "FY ends Jan. TSMC Q4 informs NVDA Q1 (one-quarter lag).",
    },
    "AAPL": {
        "leading_indicators": ["TSMC revenue", "Foxconn revenue", "Qualcomm modem shipments"],
        "channel_checks": ["iPhone sales China market share", "App Store revenue growth"],
        "sector": "consumer_tech",
        "contagion": ["QCOM", "SWKS", "CRUS", "LRCX"],
        "fiscal_note": "FY ends Sep. Holiday quarter is Q1 (Oct-Dec).",
    },
    "AMZN": {
        "leading_indicators": ["FedEx volume trend", "UPS earnings", "Shopify merchant growth"],
        "channel_checks": ["AWS growth rate", "Prime membership churn"],
        "sector": "e_commerce_cloud",
        "contagion": ["MSFT", "GOOG", "SHOP", "FDX"],
        "fiscal_note": "Calendar FY. AWS is ~60% of operating income.",
    },
    "JPM": {
        "leading_indicators": ["credit card delinquency rate", "FDIC bank failure list"],
        "channel_checks": ["CRE default rates", "IB deal pipeline", "NIM trends"],
        "sector": "financials",
        "contagion": ["BAC", "GS", "MS", "C"],
        "fiscal_note": "Calendar FY. Reports first among big banks (sets tone).",
    },
    "WMT": {
        "leading_indicators": ["PG earnings", "KO volume", "SNAP benefits data"],
        "channel_checks": ["grocery market share", "consumer spending downgrade signals"],
        "sector": "consumer_staples",
        "contagion": ["TGT", "COST", "DG", "KR"],
        "fiscal_note": "FY ends Jan 31. Q4 = Nov-Jan (holiday quarter).",
    },
    "XOM": {
        "leading_indicators": ["WTI crude price trend", "refinery crack spreads"],
        "channel_checks": ["Permian Basin production volume", "LNG export volume"],
        "sector": "energy",
        "contagion": ["CVX", "COP", "SLB", "HAL"],
        "fiscal_note": "Calendar FY. Upstream earnings dominate when oil >$70.",
    },
}

ACCOUNTING_RED_FLAGS: dict[str, dict] = {
    "revenue_recognition_change": {
        "keywords": ["ASC 606", "restatement", "revenue recognition change"],
        "description": "Accounting standard change affecting revenue comparability",
    },
    "one_time_items": {
        "keywords": ["restructuring charge", "write-down", "gain on sale", "impairment"],
        "description": "Non-recurring items distorting core earnings",
    },
    "stock_based_compensation": {
        "keywords": ["stock-based compensation", "SBC as percentage of revenue"],
        "description": "High SBC inflating non-GAAP vs GAAP gap",
    },
    "channel_stuffing": {
        "keywords": ["rising inventory", "inventory days increasing", "declining cash flow"],
        "description": "Revenue pulled forward — watch receivables and inventory growth vs revenue",
    },
    "fiscal_year_mismatch": {
        "keywords": ["fiscal year end", "quarter mismatch"],
        "description": "Non-calendar FY complicates YoY and peer comparisons",
    },
}


# ============================================================
# Earnings Calendar Management
# ============================================================

def _load_calendar() -> dict:
    """Load cached earnings calendar."""
    default = {"entries": [], "refreshed_at": ""}
    if not os.path.isfile(CALENDAR_PATH):
        return default
    lock_fd = None
    try:
        lock_fd = open(CALENDAR_PATH + ".lock", "a")
        fcntl.flock(lock_fd, fcntl.LOCK_SH)
        with open(CALENDAR_PATH, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else default
    except (json.JSONDecodeError, OSError):
        return default
    finally:
        if lock_fd:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
            except Exception:
                pass


def _save_calendar(data: dict) -> None:
    """Atomic write earnings calendar."""
    lock_fd = None
    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
        lock_fd = open(CALENDAR_PATH + ".lock", "a")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        tmp = CALENDAR_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, CALENDAR_PATH)
    except Exception as e:
        logger.error(f"Failed to save earnings calendar: {e}")
    finally:
        if lock_fd:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
            except Exception:
                pass


def _calendar_is_stale(data: dict) -> bool:
    """Calendar is stale if older than 3 days."""
    refreshed = data.get("refreshed_at", "")
    if not refreshed:
        return True
    try:
        dt = datetime.fromisoformat(refreshed)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_NY)
        return (datetime.now(_NY) - dt) > timedelta(days=3)
    except (ValueError, TypeError):
        return True


async def _refresh_earnings_calendar() -> list[dict]:
    """Search DDG for upcoming bellwether earnings dates.

    Cross-references with EARNINGS_SUPPLY_CHAIN keys.
    Caches results, refreshes weekly.
    """
    now = datetime.now(_NY)
    calendar = _load_calendar()

    if not _calendar_is_stale(calendar):
        return calendar.get("entries", [])

    logger.info("Refreshing earnings calendar...")

    try:
        allowed, msg = await budget_tracker.check_budget("report_web_search")
        if not allowed:
            return calendar.get("entries", [])

        from agent.integrations.web_search import search_web
        date_str = now.strftime("%B %d %Y")
        results = await search_web(
            f"earnings calendar this week major companies {date_str}",
            max_results=10,
        )
        await budget_tracker.record_usage("report_web_search")

        entries = []
        bellwethers = set(EARNINGS_SUPPLY_CHAIN.keys())

        for r in (results or []):
            title = r.get("title", "")
            snippet = r.get("snippet", r.get("body", ""))
            text = f"{title} {snippet}".upper()

            for ticker in bellwethers:
                if ticker in text:
                    # Try to extract date from snippet
                    report_date = _extract_date_from_text(snippet, now)
                    entries.append({
                        "ticker": ticker,
                        "date": report_date or "",
                        "source": title[:80],
                    })

        # Deduplicate by ticker (keep first occurrence)
        seen = set()
        unique = []
        for e in entries:
            if e["ticker"] not in seen:
                seen.add(e["ticker"])
                unique.append(e)

        calendar = {"entries": unique, "refreshed_at": now.isoformat()}
        _save_calendar(calendar)
        logger.info(f"Earnings calendar refreshed: {len(unique)} bellwethers found")
        return unique

    except BudgetExhaustedError:
        return calendar.get("entries", [])
    except Exception as e:
        logger.warning(f"Failed to refresh earnings calendar: {e}")
        return calendar.get("entries", [])


def _extract_date_from_text(text: str, reference: datetime) -> str | None:
    """Best-effort date extraction from snippet text.

    Looks for patterns like 'March 5', 'Mar 5, 2026', 'March 5th'.
    Returns ISO date string or None.
    """
    import re

    months = {
        "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
        "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6,
        "jul": 7, "july": 7, "aug": 8, "august": 8, "sep": 9, "september": 9,
        "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
    }

    pattern = r'(?:' + '|'.join(months.keys()) + r')\w*\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(\d{4}))?'
    match = re.search(pattern, text.lower())
    if not match:
        return None

    # Find which month matched
    month_match = re.search(r'(' + '|'.join(months.keys()) + r')\w*', text.lower())
    if not month_match:
        return None

    month_key = month_match.group(1).lower()[:3]
    month = months.get(month_key)
    if not month:
        return None

    day = int(match.group(1))
    year = int(match.group(2)) if match.group(2) else reference.year

    try:
        dt = datetime(year, month, day, tzinfo=_NY)
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return None


def _bellwethers_reporting_soon(calendar_entries: list[dict], days: int = 3) -> list[dict]:
    """Filter calendar entries to bellwethers reporting in the next N days."""
    now = datetime.now(_NY)
    cutoff = now + timedelta(days=days)
    reporting = []

    for entry in calendar_entries:
        date_str = entry.get("date", "")
        if not date_str:
            continue  # Skip entries with unknown dates (Fix S104-I)
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=_NY)
            if now.date() <= dt.date() <= cutoff.date():
                reporting.append(entry)
        except (ValueError, TypeError):
            continue  # Skip on parse failure (Fix S104-I)

    return reporting


# ============================================================
# Dedup Management
# ============================================================

def _load_dedup() -> dict:
    """Load dedup state: {ticker: last_analyzed_iso}."""
    if not os.path.isfile(DEDUP_PATH):
        return {}
    lock_fd = None
    try:
        lock_fd = open(DEDUP_PATH + ".lock", "a")
        fcntl.flock(lock_fd, fcntl.LOCK_SH)
        with open(DEDUP_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    finally:
        if lock_fd:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
            except Exception:
                pass


def _save_dedup(data: dict) -> None:
    """Atomic write dedup state."""
    lock_fd = None
    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
        lock_fd = open(DEDUP_PATH + ".lock", "a")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        tmp = DEDUP_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, DEDUP_PATH)
    except Exception as e:
        logger.error(f"Failed to save earnings dedup: {e}")
    finally:
        if lock_fd:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
            except Exception:
                pass


def _already_analyzed(ticker: str, dedup: dict) -> bool:
    """Check if ticker was analyzed in the last 5 days."""
    ts = dedup.get(ticker, "")
    if not ts:
        return False
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_NY)
        return (datetime.now(_NY) - dt) < timedelta(days=5)
    except (ValueError, TypeError):
        return False


# ============================================================
# A) Multi-Source Research
# ============================================================

async def _gather_supply_chain_research(ticker: str) -> list[dict]:
    """Execute consensus + supply chain + channel check queries for a bellwether.

    2 consensus queries + up to 4 supply chain / channel check queries.
    Budget: report_web_search.
    """
    chain = EARNINGS_SUPPLY_CHAIN.get(ticker)
    if not chain:
        return []

    now = datetime.now(_NY)
    date_suffix = now.strftime("%B %d %Y")

    queries = [
        (f"{ticker} earnings consensus estimate EPS revenue {date_suffix}", "consensus"),
        (f"{ticker} earnings preview analyst expectations {date_suffix}", "consensus"),
    ]

    # Supply chain leading indicators
    for indicator in chain.get("leading_indicators", [])[:2]:
        queries.append((f"{indicator} latest results {date_suffix}", "supply_chain"))

    # Channel checks
    for check in chain.get("channel_checks", [])[:2]:
        queries.append((f"{check} {date_suffix}", "channel_check"))

    results_all = []
    for query, query_type in queries:
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
                if title:
                    results_all.append({
                        "title": title,
                        "snippet": snippet,
                        "url": url,
                        "query_type": query_type,
                    })

        except BudgetExhaustedError:
            break
        except Exception as e:
            logger.warning(f"Supply chain query failed: {query}: {e}")

        await asyncio.sleep(0.5)

    # S101: Apply staleness filter to freshly gathered results
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
# B) Memory Recall — Prior Guidance History
# ============================================================

def _recall_guidance_history(ticker: str) -> str:
    """Search knowledge base for prior entries tagged with ticker or titled with earnings.

    Returns formatted string of prior guidance/earnings entries.
    """
    try:
        from agent.tools.knowledge_tools import load_all_entries
        all_entries = load_all_entries()
    except Exception as e:
        logger.debug(f"Failed to load entries for guidance history: {e}")
        return "No prior guidance history available."

    cutoff = datetime.now(_NY) - timedelta(days=180)
    relevant = []
    ticker_lower = ticker.lower()

    for entry in all_entries:
        title = entry.get("title", "").lower()
        tags = entry.get("tags", [])
        content = ""

        # Match by ticker in title, tags, or content
        tag_match = isinstance(tags, list) and any(ticker_lower in str(t).lower() for t in tags)
        title_match = ticker_lower in title

        if not (tag_match or title_match):
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

        # Try to read file content
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
                    content = data.get("content", "")[:200]
            except Exception as e:
                logger.debug(f"Failed to read knowledge entry file: {e}")

        relevant.append({
            "title": entry.get("title", ""),
            "created": created[:10],
            "content": content,
        })

    if not relevant:
        return "No prior guidance history found."

    # Sort newest first (parsed datetime, timezone-safe)
    def _sort_key(entry: dict) -> datetime:
        try:
            dt = datetime.fromisoformat(entry.get("created", "2000-01-01"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_NY)
            return dt
        except (ValueError, TypeError):
            return datetime(2000, 1, 1, tzinfo=_NY)

    relevant.sort(key=_sort_key, reverse=True)

    lines = []
    for e in relevant[:5]:
        lines.append(f"[{e['created']}] {e['title']}\n{e['content']}")

    return "\n---\n".join(lines)


# ============================================================
# C) Data Validity — Accounting Red Flags
# ============================================================

async def _check_accounting_validity(ticker: str) -> str | None:
    """Search for accounting standard changes, restatements, SBC trends.

    Budget: report_web_search (1 query).
    Returns brief validity note or None.
    """
    try:
        allowed, msg = await budget_tracker.check_budget("report_web_search")
        if not allowed:
            return None

        from agent.integrations.web_search import search_web
        now = datetime.now(_NY)
        query = f"{ticker} accounting changes restatement SBC stock compensation {now.strftime('%B %d %Y')}"
        results = await search_web(query, max_results=3)
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

        # Check for red flag keywords
        combined = " ".join(snippets).lower()
        flags_found = []
        for flag_name, flag_info in ACCOUNTING_RED_FLAGS.items():
            for kw in flag_info["keywords"]:
                if kw.lower() in combined:
                    flags_found.append(flag_info["description"])
                    break

        if flags_found:
            evidence = sanitize_ingested_content("\n".join(snippets[:2]), source="ddg_search")
            return "ACCOUNTING FLAGS: " + "; ".join(flags_found) + "\n\nEvidence:\n" + evidence

        return None

    except BudgetExhaustedError:
        return None
    except Exception as e:
        logger.debug(f"Accounting validity check failed for {ticker}: {e}")
        return None


# ============================================================
# CC-CLIO-FD-2: Financial Datasets Enrichment
# ============================================================

def _fd_analyst_consensus(ticker: str) -> str:
    """Pre-earnings: Fetch analyst consensus EPS/revenue estimates.

    Sync — callers use run_in_executor(). Budget check is caller's responsibility.
    BC#112: Fields are "revenue" and "earnings_per_share" (NOT estimated_*).
    No fiscal_year field — only fiscal_period.
    """
    if os.environ.get("CLIO_FD_EARNINGS_CONSENSUS", "0") != "1":
        return ""
    try:
        from agent.integrations.fd_client import get_fd_client
        client = get_fd_client()
        estimates = client.get_analyst_estimates(ticker, period="quarterly", limit=4)
        if not estimates or (isinstance(estimates, dict) and "_error" in estimates):
            return ""
        est_list = estimates if isinstance(estimates, list) else []
        if not est_list:
            return ""

        lines = ["Analyst Consensus (FD):"]
        for e in est_list[:2]:
            if not isinstance(e, dict):
                continue
            fp = e.get("fiscal_period", "")
            # BC#112: "revenue" and "earnings_per_share" (NOT estimated_revenue/eps)
            rev = e.get("revenue")
            eps = e.get("earnings_per_share")
            if rev is not None and rev > 1e9:
                rev_str = f"${rev / 1e9:.2f}B"
            elif rev is not None:
                rev_str = f"${rev / 1e6:.0f}M"
            else:
                rev_str = "N/A"
            eps_str = f"${eps:.2f}" if eps is not None else "N/A"
            lines.append(f"  {fp}: Rev={rev_str}, EPS={eps_str}")

        return "\n".join(lines) if len(lines) > 1 else ""
    except Exception as e:
        logger.warning("CC-CLIO-FD-2: Analyst consensus failed for %s: %s", ticker, e)
        return ""


def _fd_risk_factors(ticker: str) -> str:
    """Pre-earnings: Surface latest 10-K Risk Factors (Item 1A).

    Sync — callers use run_in_executor(). Budget check is caller's responsibility.
    BC#112: Filing field is "report_date" (NOT "report_period").
    Item filter uses hyphen: "Item-1A".
    """
    if os.environ.get("CLIO_FD_EARNINGS_FILING_CONTEXT", "0") != "1":
        return ""
    try:
        from agent.integrations.fd_client import get_fd_client
        client = get_fd_client()

        filings = client.get_sec_filings(ticker, filing_type=["10-K"], limit=1)
        if not filings or (isinstance(filings, dict) and "_error" in filings):
            return ""
        filing_list = filings if isinstance(filings, list) else []
        if not filing_list:
            return ""

        # BC#112: "report_date" (NOT "report_period")
        year_str = filing_list[0].get("report_date", "")[:4]
        if not year_str:
            return ""

        items = client.get_sec_filing_items(ticker, "10-K", int(year_str), item="Item-1A")
        if not items or (isinstance(items, dict) and "_error" in items):
            return ""
        item_list = items if isinstance(items, list) else []
        if not item_list:
            return ""

        text = item_list[0].get("text", "")
        if len(text) > 2000:
            text = text[:2000] + "..."

        # NOTE: Sanitization handled by combined fd_context sanitization in pipeline wiring
        # Do NOT sanitize here to avoid double-wrapping in <ingested_data> tags
        return f"Risk Factors ({year_str} 10-K, Item 1A):\n{text}"
    except Exception as e:
        logger.warning("CC-CLIO-FD-2: Risk factors failed for %s: %s", ticker, e)
        return ""


def _fd_earnings_surprise(ticker: str) -> dict:
    """Post-earnings: Detailed surprise analysis.

    Sync — callers use run_in_executor(). Budget check is caller's responsibility.
    BC#112: earnings_per_share (NOT eps_diluted), estimated_earnings_per_share (NOT estimated_eps).
    revenue_surprise and eps_surprise are STRINGS ("BEAT"/"MISS"), NOT numeric percentages.
    """
    if os.environ.get("CLIO_FD_EARNINGS_SURPRISE", "0") != "1":
        return {}
    try:
        from agent.integrations.fd_client import get_fd_client
        client = get_fd_client()
        context = client.get_earnings_context(ticker)

        latest = context.get("latest_earnings")
        if not latest or (isinstance(latest, dict) and "_error" in latest):
            return {}

        # Earnings returns {"quarterly": <list|dict>, "annual": <list|dict>}
        # BC#112: quarterly can be a list of earnings entries or a dict with nested data
        quarterly = latest.get("quarterly") if isinstance(latest, dict) else None
        if not quarterly:
            return {}
        # Handle both list and dict structures defensively
        if isinstance(quarterly, list):
            entries = quarterly
        elif isinstance(quarterly, dict):
            # If quarterly is a dict, it may contain a list or be a single entry
            entries = [quarterly]
        else:
            return {}
        if not entries:
            return {}
        e = entries[0]
        if not isinstance(e, dict):
            return {}

        return {
            "revenue_actual": e.get("revenue"),
            "revenue_estimated": e.get("estimated_revenue"),
            "revenue_surprise": e.get("revenue_surprise"),  # STRING: "BEAT"/"MISS"
            "eps_actual": e.get("earnings_per_share"),
            "eps_estimated": e.get("estimated_earnings_per_share"),
            "eps_surprise": e.get("eps_surprise"),  # STRING: "BEAT"/"MISS"
            "_source": "financial_datasets",
        }
    except Exception as e:
        logger.warning("CC-CLIO-FD-2: Earnings surprise failed for %s: %s", ticker, e)
        return {}


# ============================================================
# D) Scenario Analysis (Opus-powered)
# ============================================================

_EARNINGS_PROMPT = """You are a sell-side equity analyst producing a pre-earnings scenario analysis for a bellwether company.

Your job: synthesize the research data and produce 5 scenarios with specific numbers, market reactions, and sector contagion effects.

FORMAT:
TICKER: {ticker} — Earnings Expected: {date}
SECTOR: {sector}

SUPPLY CHAIN SIGNAL:
{What do leading indicators (TSMC, FedEx, etc.) suggest about this quarter? Be specific with numbers.}

ACCOUNTING CAVEAT:
{Any comparability issues — ASC changes, one-time items, SBC, FY mismatch — or "No comparability issues detected"}

PRIOR GUIDANCE CONTEXT:
{Historical pattern — does management sandbag or over-promise? What did they guide last quarter?}

SCENARIOS:
MISS_BIG (5-10% probability): EPS ${low_range}, Rev ${low_rev} — {specific market reaction, e.g. "stock -8-12%, sector peer reactions"}
MISS_SLIGHT (15-20%): EPS ${range}, Rev ${range} — {reaction}
IN_LINE (30-40%): EPS ${range}, Rev ${range} — {reaction, guidance matters more than beat}
BEAT_SLIGHT (20-25%): EPS ${range}, Rev ${range} — {reaction}
BEAT_BIG (5-10%): EPS ${range}, Rev ${range} — {reaction, sector contagion}

For each scenario, specify:
- Exact EPS and revenue ranges
- Stock price reaction (percentage)
- Sector contagion: name specific tickers and expected moves
- Forward guidance implication

KEY METRIC: {The single number that matters most on the call — not always EPS. Could be AWS growth, iPhone units, NIM, same-store sales, etc.}

SUPPLY CHAIN LAG NOTE: {If applicable, note fiscal calendar mismatches between the company and its supply chain indicators}

RULES:
- Every scenario MUST have specific dollar amounts for EPS and revenue
- MISS_BIG and BEAT_BIG are tail scenarios (5-10% probability each)
- Name specific contagion tickers with expected percentage moves
- NEVER say "monitor", "watch for", "keep an eye on", "remains to be seen", "worth noting"
- If you lack data for a specific number, say "insufficient data for estimate" — do NOT make up numbers
- Write in plain text only, no markdown formatting
- State the single most important call metric and why"""


async def _generate_earnings_scenario(
    ticker: str,
    sector: str,
    fiscal_note: str,
    research_results: list[dict],
    guidance_history: str,
    accounting_note: str | None,
    fd_context: str = "",
) -> str | None:
    """Generate 5-scenario earnings analysis using Claude Opus.

    Budget: earnings_intel.
    """
    try:
        allowed, msg = await budget_tracker.check_budget("earnings_intel")
        if not allowed:
            logger.debug("earnings_intel budget exhausted")
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
    research_text = sanitize_ingested_content(research_text, source="ddg_search")

    accounting = accounting_note or "No accounting issues detected"

    chain = EARNINGS_SUPPLY_CHAIN.get(ticker, {})
    contagion = chain.get("contagion", [])
    contagion_str = ", ".join(contagion) if contagion else "None mapped"

    # CC-CLIO-FD-2: Include FD fundamental data if available
    fd_section = ""
    if fd_context:
        fd_section = f"FUNDAMENTAL DATA (Financial Datasets):\n{fd_context}\n\n"

    user_message = (
        f"TICKER: {ticker}\n"
        f"SECTOR: {sector}\n"
        f"FISCAL NOTE: {fiscal_note}\n"
        f"CONTAGION PEERS: {contagion_str}\n\n"
        f"ACCOUNTING ASSESSMENT:\n{accounting}\n\n"
        f"PRIOR GUIDANCE HISTORY:\n{guidance_history}\n\n"
        f"{fd_section}"
        f"RESEARCH FINDINGS:\n{research_text}\n\n"
        f"Generate a 5-scenario earnings analysis (MISS_BIG / MISS_SLIGHT / IN_LINE / BEAT_SLIGHT / BEAT_BIG) "
        f"with specific EPS/revenue ranges, stock reactions, and sector contagion."
    )

    try:
        client = _get_client()
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, lambda: client.messages.create(
            model=config.default_model,
            max_tokens=3000,
            system=cached_system_block(_EARNINGS_PROMPT),
            messages=[{"role": "user", "content": user_message}],
        ))

        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

        result = text.strip() if text.strip() else None
        if result:
            await budget_tracker.record_usage("earnings_intel")
        return result

    except Exception as e:
        logger.error(f"Earnings scenario generation failed for {ticker}: {e}", exc_info=True)
        return None


# ============================================================
# Delivery
# ============================================================

async def _deliver_earnings_intel(
    context,
    ticker: str,
    analysis: str,
    urls: list[str],
) -> None:
    """Deliver earnings intelligence: Telegram + knowledge commit + bridge write."""
    now = datetime.now(_NY)
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    # 1. Telegram notification
    try:
        from agent.integrations.scheduled_tasks import _get_chat_ids, _safe_send
        chat_ids = _get_chat_ids()
        if chat_ids:
            msg = f"EARNINGS INTEL: {ticker}\n{now.strftime('%I:%M %p ET')}\n\n{analysis}"
            if len(msg) > 4000:
                msg = msg[:3997] + "..."
            for cid in chat_ids:
                await _safe_send(context.bot, cid, text=msg)
    except Exception as e:
        logger.warning(f"Failed to send earnings intel to Telegram: {e}")

    # 2. Knowledge base commit
    try:
        from agent.tools.knowledge_tools import handle_save_knowledge
        title = f"Earnings Intel: {ticker} — {now.strftime('%b %d %Y')}"
        source_urls = ", ".join(urls[:3]) if urls else "DuckDuckGo research"
        await handle_save_knowledge({
            "title": title,
            "content": analysis,
            "category": "trading_market_analysis",
            "tags": ["earnings_intel", ticker, now.strftime("%Y-%m-%d")],
            "source": source_urls,
        })
    except Exception as e:
        logger.warning(f"Failed to save earnings intel to knowledge: {e}")

    # 3. Bridge write
    try:
        os.makedirs(BRIDGE_DIR, exist_ok=True)
        filename = f"{timestamp}_earnings_{ticker}.json"
        entry = {
            "id": f"earnings_{timestamp}_{ticker}",
            "type": "earnings_intel",
            "ticker": ticker,
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
        logger.warning(f"Failed to write earnings intel to bridge: {e}")


# ============================================================
# Scheduled Callback
# ============================================================

async def run_earnings_intel(context) -> None:
    """Scheduled callback — daily 7:15 AM ET weekdays.

    1. Refresh earnings calendar if stale (>3 days)
    2. For each bellwether reporting in next 3 days:
       research -> recall -> validity -> scenario -> deliver
    3. Max 2 per run (budget conservation)
    4. 5-day dedup per ticker
    """
    if not config.news_intel_enabled:
        return

    now = datetime.now(_NY)

    # Time guard: weekdays, 6-10 AM ET
    if now.weekday() >= 5:
        return
    if now.hour < 6 or now.hour >= 10:
        return

    logger.info("Starting earnings intelligence pipeline...")

    try:
        # Refresh calendar
        calendar_entries = await _refresh_earnings_calendar()
        if not calendar_entries:
            # S186 W1: visibility — earnings calendar empty (could be UW/FD failure or genuinely no bellwethers)
            logger.info("Earnings intel exiting: 0 bellwethers in earnings calendar (DDG search returned none -- off-season or blocked?)")
            return

        # Filter to reporting in next 3 days
        reporting_soon = _bellwethers_reporting_soon(calendar_entries, days=3)
        if not reporting_soon:
            # S186 W1: visibility — calendar fresh but no bellwethers in 3d window
            logger.info("Earnings intel exiting: 0 bellwethers reporting in next 3d")
            return

        dedup = _load_dedup()
        analyzed_count = 0

        for entry in reporting_soon:
            if analyzed_count >= 2:
                break

            ticker = entry.get("ticker", "")
            if not ticker or ticker not in EARNINGS_SUPPLY_CHAIN:
                continue

            if _already_analyzed(ticker, dedup):
                logger.debug(f"Already analyzed {ticker} in last 5 days, skipping")
                continue

            chain = EARNINGS_SUPPLY_CHAIN[ticker]
            sector = chain.get("sector", "unknown")
            fiscal_note = chain.get("fiscal_note", "")

            logger.info(f"Earnings intel: {ticker} ({sector})")

            # A) Multi-source research
            research_results = await _gather_supply_chain_research(ticker)

            # S101: Filter stale results after gather
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
                # S186 W1: visibility on per-ticker upstream-empty
                logger.info(f"Earnings intel skipping {ticker}: 0 research results (web_search budget? upstream issue?)")
                continue

            # B) Memory recall
            guidance_history = _recall_guidance_history(ticker)

            # C) Accounting validity
            accounting_note = await _check_accounting_validity(ticker)

            # CC-CLIO-FD-2: Financial Datasets enrichment (sync → run_in_executor)
            fd_parts = []
            try:
                fd_allowed, _ = await budget_tracker.check_budget("financial_datasets")
                if not fd_allowed:
                    logger.debug("CC-CLIO-FD-2: financial_datasets budget exhausted")
                    raise BudgetExhaustedError("financial_datasets")
                loop = asyncio.get_running_loop()
                fd_consensus = await loop.run_in_executor(None, _fd_analyst_consensus, ticker)
                if fd_consensus:
                    fd_parts.append(fd_consensus)
                fd_risks = await loop.run_in_executor(None, _fd_risk_factors, ticker)
                if fd_risks:
                    fd_parts.append(fd_risks)
                fd_surprise = await loop.run_in_executor(None, _fd_earnings_surprise, ticker)
                if fd_surprise:
                    surprise_lines = ["Recent Earnings Surprise (FD):"]
                    for k, v in fd_surprise.items():
                        if k != "_source" and v is not None:
                            surprise_lines.append(f"  {k}: {v}")
                    if len(surprise_lines) > 1:
                        fd_parts.append("\n".join(surprise_lines))
                await budget_tracker.record_usage("financial_datasets")
            except BudgetExhaustedError:
                pass  # Silently skip FD enrichment when budget exhausted
            except Exception as e:
                logger.debug("CC-CLIO-FD-2: FD enrichment failed for %s: %s", ticker, e)
            fd_context = "\n\n".join(fd_parts)
            # Sanitize combined FD output (risk factors already sanitized individually,
            # but this catches analyst consensus + surprise data too)
            if fd_context:
                fd_context = sanitize_ingested_content(fd_context, source="financial_datasets")

            # D) Scenario analysis
            analysis = await _generate_earnings_scenario(
                ticker=ticker,
                sector=sector,
                fiscal_note=fiscal_note,
                research_results=research_results,
                guidance_history=guidance_history,
                accounting_note=accounting_note,
                fd_context=fd_context,
            )

            if not analysis:
                logger.debug(f"No scenario analysis generated for {ticker}")
                continue

            urls = [r.get("url", "") for r in research_results if r.get("url")]

            await _deliver_earnings_intel(context, ticker, analysis, urls)

            # Record in dedup
            dedup[ticker] = now.isoformat()
            _save_dedup(dedup)

            analyzed_count += 1
            logger.info(f"Earnings intel complete for {ticker}: {len(research_results)} findings")

            await asyncio.sleep(2)

        if analyzed_count == 0:
            logger.debug("No new bellwethers to analyze this cycle")

    except Exception as e:
        logger.error(f"Earnings intelligence pipeline failed: {e}", exc_info=True)
