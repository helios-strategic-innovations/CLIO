"""
OpenClaw/Moltbot — Federal Reserve Communication Analysis (Session 91)

Monitors FOMC member positioning, detects linguistic shifts, and produces
rate path scenario analysis via the A/B/C/D macro research framework:
  A) Multi-source research — Fed speeches, CME FedWatch, balance sheet, trial balloons
  B) Memory recall — individual member hawk/dove positioning from knowledge base
  C) Communication validity — blackout period, voting weight, trial balloon detection
  D) Scenario analysis — 5 rate path scenarios with market impact estimates

Two triggers:
1. Escalation from news_intel.py when fomc_fed topic produces speech/policy keywords
2. Scheduled daily callback at 7:30 AM ET checking TE calendar for Fed events
"""

import asyncio
import fcntl
import json
import logging
import os
import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import anthropic

from agent.config import config
from agent.prompts.system_prompt import cached_system_block
from agent.rate_limiter import budget_tracker, BudgetExhaustedError
from agent.security import sanitize_ingested_content

logger = logging.getLogger("openclaw.fed_analysis")

_NY = ZoneInfo("America/New_York")

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "intelligence",
)
_STATE_PATH = os.path.join(_DATA_DIR, "fed_analysis_state.json")

BRIDGE_DIR = os.path.join(config.tradingbot_dir, "moltbot_knowledge", "intelligence")

_cached_client: anthropic.Anthropic | None = None
_client_lock = __import__("threading").Lock()


# ============================================================
# CC-CLIO-FD-4: Financial Datasets Rate Context
# ============================================================

def _fd_rate_context() -> str:
    """Fetch current Fed rate + recent history for FOMC analysis context."""
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

        # Also get recent history for trajectory
        # BC#113: returns ASCENDING (oldest first) — use [-N:]
        end = datetime.now(_NY).strftime("%Y-%m-%d")
        start = (datetime.now(_NY) - timedelta(days=365)).strftime("%Y-%m-%d")
        history = client.get_interest_rates("FED", start_date=start, end_date=end)

        hist_lines = []
        if history and isinstance(history, list):
            # Last 5 entries = most recent changes
            for h in history[-5:]:
                if isinstance(h, dict):
                    hist_lines.append(f"  {h.get('effective_date', '?')}: {h.get('rate', '?')}%")

        result = f"Current Fed Funds Rate: {rate}% (effective {eff_date})"
        if hist_lines:
            result += "\nRecent Changes:\n" + "\n".join(hist_lines)

        logger.debug("CC-CLIO-FD-4: Rate context injected (%s%%)", rate)
        return result
    except Exception as e:
        logger.warning("CC-CLIO-FD-4: Rate context failed: %s", e)
        return ""


def _fd_global_rates() -> str:
    """Compare major central bank rates for divergence analysis."""
    if os.environ.get("CLIO_FD_GLOBAL_RATES", "0") != "1":
        return ""
    try:
        from agent.integrations.fd_client import get_fd_client
        client = get_fd_client()

        banks = ["FED", "ECB", "BOJ", "BOE", "BOC", "RBA"]
        rates = {}
        for bank in banks:
            snap = client.get_interest_rates_snapshot(bank)
            if snap and isinstance(snap, dict) and "_error" not in snap:
                r = snap.get("rate")
                if r is not None:
                    rates[bank] = r

        if len(rates) < 2:
            return ""

        lines = ["Global Central Bank Rates (FD):"]
        for bank, r in sorted(rates.items()):
            lines.append(f"  {bank}: {r}%")

        rate_vals = []
        for r in rates.values():
            try:
                rate_vals.append(float(r))
            except (ValueError, TypeError):
                pass
        if len(rate_vals) >= 2:
            spread = max(rate_vals) - min(rate_vals)
            lines.append(f"  Spread: {spread:.2f}% ({'DIVERGENT' if spread > 3.0 else 'converging'})")

        logger.debug("CC-CLIO-FD-4: Global rates injected (%d banks)", len(rates))
        return "\n".join(lines)
    except Exception as e:
        logger.warning("CC-CLIO-FD-4: Global rates failed: %s", e)
        return ""


def _get_client() -> anthropic.Anthropic:
    global _cached_client
    if _cached_client is None:
        with _client_lock:
            if _cached_client is None:
                _cached_client = anthropic.Anthropic(**config.anthropic_client_kwargs(timeout=120))
    return _cached_client


# ============================================================
# FOMC Composition (2026)
# Updated 2026-03-12 per Federal Reserve FOMC roster. Must be reviewed annually.
# ============================================================

FOMC_MEMBERS: dict[str, dict] = {
    # --- Permanent voters: Board of Governors + NY Fed ---
    "powell": {"role": "Chair", "voting": True, "default_lean": "neutral"},
    "jefferson": {"role": "Vice Chair", "voting": True, "default_lean": "dove"},
    "williams": {"role": "NY Fed", "voting": True, "default_lean": "neutral"},
    "waller": {"role": "Governor", "voting": True, "default_lean": "hawk"},
    "bowman": {"role": "Governor", "voting": True, "default_lean": "hawk"},
    "cook": {"role": "Governor", "voting": True, "default_lean": "dove"},
    # S138: Barr resigned from the Board of Governors (was Vice Chair for Supervision).
    # Miran appointed as Governor (pro-Trump loyalist, ultra-dove). Replaces Barr's seat.
    "miran": {"role": "Governor", "voting": True, "default_lean": "dove"},
    # --- 2026 rotating voters ---
    "hammack": {"role": "Cleveland", "voting": True, "default_lean": "hawk"},
    "paulson": {"role": "Philadelphia", "voting": True, "default_lean": "neutral"},
    "logan": {"role": "Dallas", "voting": True, "default_lean": "hawk"},
    "kashkari": {"role": "Minneapolis", "voting": True, "default_lean": "dove"},
    # --- Non-voting 2026 (still tracked for speech analysis) ---
    "bostic": {"role": "Atlanta", "voting": False, "default_lean": "neutral"},
    "collins": {"role": "Boston", "voting": False, "default_lean": "neutral"},
    "musalem": {"role": "St. Louis", "voting": False, "default_lean": "hawk"},
    "goolsbee": {"role": "Chicago", "voting": False, "default_lean": "dove"},
    "daly": {"role": "San Francisco", "voting": False, "default_lean": "dove"},
}

# ============================================================
# Research Query Maps
# ============================================================

FED_RESEARCH_MAP: dict[str, dict] = {
    "speech_analysis": {
        "keywords": ["speech", "remarks", "testimony", "press conference", "comment"],
        "obvious_queries": [
            "Federal Reserve speech today FOMC member comments",
            "Fed official remarks monetary policy latest",
        ],
        "niche_queries": [
            "Fed speech word frequency shifts patient gradual data-dependent",
            "Federal Reserve balance sheet QT quantitative tightening pace",
            "reverse repo facility RRP usage trend daily",
            "Treasury general account TGA balance trend",
            "regional Fed president dissent patterns voting history",
            "Nick Timiraos WSJ Fed exclusive sources",
        ],
    },
    "rate_decision": {
        "keywords": ["rate decision", "fomc meeting", "dot plot", "rate cut", "rate hike"],
        "obvious_queries": [
            "CME FedWatch tool rate probabilities next FOMC meeting",
            "FOMC dot plot rate forecast consensus",
        ],
        "niche_queries": [
            "SOFR forward curve implied rate path",
            "OIS overnight index swap Fed rate expectations",
            "IORB corridor interest on reserve balances spread",
            "Fed standing repo facility usage banks",
            "Fed funds futures implied probability term structure",
            "federal reserve swap lines international dollar liquidity",
        ],
    },
}

# ============================================================
# 2026 FOMC Meeting Dates (hardcoded, update annually)
# WARNING: 2026 dates only. Must update for 2027+ by Jan 2027.
# Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
# After 2026-12-17, _next_fomc_meeting() returns None and triggers repeated warnings.
# ============================================================

_FOMC_MEETING_DATES: list[tuple[date, date]] = [
    (date(2026, 1, 28), date(2026, 1, 29)),
    (date(2026, 3, 18), date(2026, 3, 19)),
    (date(2026, 5, 6), date(2026, 5, 7)),
    (date(2026, 6, 17), date(2026, 6, 18)),
    (date(2026, 7, 29), date(2026, 7, 30)),
    (date(2026, 9, 16), date(2026, 9, 17)),
    (date(2026, 11, 4), date(2026, 11, 5)),
    (date(2026, 12, 16), date(2026, 12, 17)),
]


# ============================================================
# Speaker Detection
# ============================================================

_SPEAKER_RE = re.compile(
    r"\b(?:fed(?:'s|s)?\.?\s+)?("
    + "|".join(re.escape(name) for name in FOMC_MEMBERS)
    + r")\b",
    re.IGNORECASE,
)


def _detect_speaker(text: str) -> str | None:
    """Extract first FOMC member name from text. Case-insensitive."""
    m = _SPEAKER_RE.search(text)
    if m:
        return m.group(1).lower()
    return None


# ============================================================
# Blackout Period Detection
# ============================================================

def _get_blackout_windows() -> list[tuple[date, date]]:
    """Compute blackout windows for all 2026 FOMC meetings.

    Blackout starts Saturday before meeting week, ends Thursday after meeting day.
    """
    windows = []
    for day1, day2 in _FOMC_MEETING_DATES:
        # Saturday before meeting week: meeting is Tue-Wed, so meeting week
        # starts Monday. Saturday before = day1 - (day1.weekday() + 2) days
        # day1 is typically Tuesday (weekday=1), so Monday = day1 - 1, Saturday = day1 - 3
        meeting_monday = day1 - timedelta(days=day1.weekday())
        blackout_start = meeting_monday - timedelta(days=2)  # Saturday before

        # Thursday after meeting = day2 + offset to Thursday
        # day2 is typically Wednesday (weekday=2), Thursday = day2 + 1
        days_to_thursday = (3 - day2.weekday()) % 7
        if days_to_thursday == 0:
            days_to_thursday = 1  # If meeting ends Thursday, blackout ends Friday
        blackout_end = day2 + timedelta(days=days_to_thursday)

        windows.append((blackout_start, blackout_end))
    return windows


_BLACKOUT_WINDOWS = _get_blackout_windows()


def _is_blackout_period() -> bool:
    """Check if today falls within an FOMC blackout window."""
    today = datetime.now(_NY).date()
    for start, end in _BLACKOUT_WINDOWS:
        if start <= today <= end:
            return True
    return False


_fomc_stale_warned = False

def _next_fomc_meeting() -> date | None:
    """Return the start date of the next FOMC meeting, or None."""
    global _fomc_stale_warned
    today = datetime.now(_NY).date()
    for day1, _day2 in _FOMC_MEETING_DATES:
        if day1 >= today:
            return day1
    # All dates in the past — dates need updating for the current year
    if not _fomc_stale_warned:
        logger.warning("FOMC meeting dates are stale — update _FOMC_MEETING_DATES for the current year")
        _fomc_stale_warned = True
    return None


# ============================================================
# Trial Balloon Detection
# ============================================================

def _detect_trial_balloon(results: list[dict]) -> str | None:
    """Check search results for possible Fed trial balloon signals.

    Looks for Timiraos (WSJ) or Schneider (Reuters) exclusives near FOMC meetings.
    Returns brief note or None.
    """
    next_meeting = _next_fomc_meeting()
    if next_meeting is None:
        return None

    today = datetime.now(_NY).date()
    days_to_meeting = (next_meeting - today).days
    if days_to_meeting > 14:
        return None

    for r in results:
        title = (r.get("title", "") + " " + r.get("snippet", "")).lower()
        url = r.get("url", "").lower()

        # WSJ / Timiraos
        if "timiraos" in title or "timiraos" in url:
            if any(kw in title for kw in ("exclusive", "sources say", "sources familiar")):
                return f"Possible trial balloon: Timiraos/WSJ piece detected {days_to_meeting} days before FOMC"

        # Reuters / Schneider
        if ("schneider" in title or "schneider" in url) and "reuters" in url:
            if any(kw in title for kw in ("exclusive", "sources say", "sources familiar")):
                return f"Possible trial balloon: Schneider/Reuters piece detected {days_to_meeting} days before FOMC"

    return None


# ============================================================
# Memory Recall — Prior Member History
# ============================================================

def _recall_member_history(speaker: str) -> str:
    """Search knowledge base for prior entries about a specific FOMC member."""
    try:
        from agent.tools.knowledge_tools import load_all_entries
        all_entries = load_all_entries()
    except Exception as e:
        logger.debug(f"Failed to load entries for member history: {e}")
        return "No prior history available."

    cutoff = datetime.now(_NY) - timedelta(days=90)
    relevant = []

    for entry in all_entries:
        tags = entry.get("tags", [])
        title = entry.get("title", "").lower()

        tag_match = isinstance(tags, list) and any(
            "fed_analysis" in str(t) or speaker in str(t) for t in tags
        )
        title_match = speaker in title

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

        relevant.append(entry)

    if not relevant:
        member = FOMC_MEMBERS.get(speaker, {})
        return f"No prior history. Default lean: {member.get('default_lean', 'unknown')}"

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
# Member Positioning Map
# ============================================================

def _get_positioning_map() -> str:
    """Format current FOMC member hawk/dove positioning for prompt injection."""
    hawks = []
    doves = []
    neutrals = []
    for name, info in FOMC_MEMBERS.items():
        if not info.get("voting"):
            continue
        label = f"{name.capitalize()} ({info['role']})"
        lean = info.get("default_lean", "neutral")
        if lean == "hawk":
            hawks.append(label)
        elif lean == "dove":
            doves.append(label)
        else:
            neutrals.append(label)

    return (
        f"HAWKS ({len(hawks)}): {', '.join(hawks)}\n"
        f"DOVES ({len(doves)}): {', '.join(doves)}\n"
        f"NEUTRAL ({len(neutrals)}): {', '.join(neutrals)}"
    )


# ============================================================
# Rate Path Analysis Prompt (Opus)
# ============================================================

_RATE_PATH_PROMPT = """You are a Federal Reserve communication analyst producing rate path scenario analysis.

Analyze the provided Fed communication data and produce a structured intelligence report.

FORMAT:

SPEAKER: {name, role, voting status, default lean}
COMMUNICATION CONTEXT: {blackout status, trial balloon indicator, prepared remarks vs Q&A}

LANGUAGE ANALYSIS:
- Key phrases used (exact quotes if available)
- Comparison to prior speeches by this member
- Shift assessment: MORE HAWKISH / UNCHANGED / MORE DOVISH vs last appearance

MEMBER POSITIONING MAP:
{Current hawk/dove breakdown for voting members}

RATE PATH SCENARIOS (probabilities must sum to ~100%):
EMERGENCY CUT (-50bp+): X% — 10Y yield move, SPY move, DXY move, GLD move
NORMAL CUT (-25bp): X% — 10Y yield move, SPY move, DXY move, GLD move
HOLD: X% — 10Y yield move, SPY move, DXY move, GLD move
HIKE (+25bp): X% — 10Y yield move, SPY move, DXY move, GLD move
EMERGENCY HIKE (+50bp+): X% — 10Y yield move, SPY move, DXY move, GLD move

CME FEDWATCH COMPARISON:
{Where our scenario probabilities disagree with CME pricing, and why}

BALANCE SHEET SIGNAL:
{QT pace assessment, RRP facility usage, reserve adequacy}

KEY SIGNAL: {The single most important data point}
CONFIDENCE: {LOW/MEDIUM/HIGH} — {brief justification}

RULES:
- Every sentence must contain a specific fact: a number, a name, a date, or a price level
- NEVER use: "monitor closely", "watch for", "keep an eye on", "worth noting", "remains to be seen"
- State market implications with specific numbers: "SPY likely -1.2%", "10Y yield +8bp"
- If in BLACKOUT PERIOD, explicitly note that speech has no policy signal value
- Write in plain text only, no markdown formatting"""


# ============================================================
# Research Pipeline
# ============================================================

async def _gather_fed_research(research_type: str = "speech_analysis") -> list[dict]:
    """Execute research queries for Fed analysis.

    Uses report_web_search budget. Returns list of {title, snippet, url, query_type}.
    """
    research_config = FED_RESEARCH_MAP.get(research_type, FED_RESEARCH_MAP["speech_analysis"])
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
            logger.debug("Budget exhausted during Fed research")
            break
        except Exception as e:
            logger.warning(f"Fed research query failed: {query}: {e}")

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

async def _generate_rate_path_analysis(
    speaker: str | None,
    research_results: list[dict],
    member_history: str,
    blackout: bool,
    trial_balloon: str | None,
) -> str | None:
    """Generate rate path analysis using Claude Opus. Budget: fed_analysis."""
    try:
        allowed, msg = await budget_tracker.check_budget("fed_analysis")
        if not allowed:
            logger.debug("fed_analysis budget exhausted")
            return None
    except BudgetExhaustedError:
        return None

    # Format speaker info
    if speaker and speaker in FOMC_MEMBERS:
        info = FOMC_MEMBERS[speaker]
        speaker_text = (
            f"{speaker.capitalize()} — {info['role']}, "
            f"Voting: {'Yes' if info['voting'] else 'No'}, "
            f"Default lean: {info['default_lean']}"
        )
    else:
        speaker_text = "No specific speaker identified — general Fed communication analysis"

    # Format research
    research_text = ""
    for i, r in enumerate(research_results[:12], 1):
        qt = r.get("query_type", "")
        research_text += (
            f"\n[{i}] ({qt}) {r['title']}\n"
            f"{r.get('snippet', '')}\n"
            f"URL: {r.get('url', '')}\n"
        )
    research_text = sanitize_ingested_content(research_text, source="ddg_search")

    blackout_text = "YES — BLACKOUT PERIOD ACTIVE. Speech has no policy signal value." if blackout else "No"
    balloon_text = trial_balloon or "None detected"
    positioning = _get_positioning_map()

    # CC-CLIO-FD-4: Inject FD rate context into FOMC analysis (run_in_executor to avoid blocking)
    fd_rate_section = ""
    try:
        _fd_ok, _ = await budget_tracker.check_budget("financial_datasets")
        if _fd_ok:
            _fd_loop = asyncio.get_running_loop()
            fd_rates = await _fd_loop.run_in_executor(None, _fd_rate_context)
            if fd_rates:
                fd_rate_section += f"\n\n{fd_rates}"
            fd_global = await _fd_loop.run_in_executor(None, _fd_global_rates)
            if fd_global:
                fd_rate_section += f"\n\n{fd_global}"
            if fd_rate_section:
                await budget_tracker.record_usage("financial_datasets")
    except Exception as e:
        logger.debug("CC-CLIO-FD-4: FD rate context fetch failed: %s", e)

    user_message = (
        f"SPEAKER: {speaker_text}\n\n"
        f"BLACKOUT PERIOD: {blackout_text}\n"
        f"TRIAL BALLOON: {balloon_text}\n\n"
        f"MEMBER POSITIONING MAP:\n{positioning}\n\n"
        f"PRIOR HISTORY FOR THIS SPEAKER:\n{member_history}\n\n"
        + (f"RATE DATA (Financial Datasets API):{fd_rate_section}\n\n" if fd_rate_section else "")
        + f"NEW RESEARCH FINDINGS:\n{research_text}\n\n"
        f"Generate a rate path scenario analysis with 5 scenarios "
        f"(EMERGENCY CUT / NORMAL CUT / HOLD / HIKE / EMERGENCY HIKE) "
        f"including specific probabilities, market impact numbers, and CME FedWatch comparison."
    )

    try:
        client = _get_client()
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, lambda: client.messages.create(
            model=config.default_model,
            max_tokens=2500,
            system=cached_system_block(_RATE_PATH_PROMPT),
            messages=[{"role": "user", "content": user_message}],
        ))

        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

        result = text.strip() if text.strip() else None
        if result:
            await budget_tracker.record_usage("fed_analysis")
        return result

    except Exception as e:
        logger.error(f"Rate path analysis generation failed: {e}", exc_info=True)
        return None


# ============================================================
# Delivery
# ============================================================

async def _deliver_fed_intel(
    context,
    speaker: str | None,
    analysis: str,
    urls: list[str],
) -> None:
    """Deliver Fed intel: Telegram + knowledge commit + bridge write."""
    now = datetime.now(_NY)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    speaker_label = speaker.capitalize() if speaker else "General"

    # 1. Telegram
    try:
        from agent.integrations.scheduled_tasks import _get_chat_ids, _safe_send
        chat_ids = _get_chat_ids()
        if chat_ids:
            msg = f"FED ANALYSIS: {speaker_label}\n{now.strftime('%I:%M %p ET')}\n\n{analysis}"
            if len(msg) > 4000:
                msg = msg[:3997] + "..."
            for cid in chat_ids:
                await _safe_send(context.bot, cid, text=msg)
    except Exception as e:
        logger.warning(f"Failed to send Fed analysis to Telegram: {e}")

    # 2. Knowledge base commit
    try:
        from agent.tools.knowledge_tools import handle_save_knowledge
        tags = ["fed_analysis", now.strftime("%Y-%m-%d")]
        if speaker:
            tags.append(speaker)
        title = f"Fed Analysis: {speaker_label} — {now.strftime('%b %d %Y')}"
        source_urls = ", ".join(urls[:3]) if urls else "DuckDuckGo research"
        await handle_save_knowledge({
            "title": title,
            "content": analysis,
            "category": "trading_market_analysis",
            "tags": tags,
            "source": source_urls,
        })
    except Exception as e:
        logger.warning(f"Failed to save Fed analysis to knowledge: {e}")

    # 3. Bridge write
    try:
        os.makedirs(BRIDGE_DIR, exist_ok=True)
        filename = f"{timestamp}_fed_{speaker or 'general'}.json"
        entry = {
            "id": f"fed_{timestamp}_{speaker or 'general'}",
            "type": "fed_analysis",
            "speaker": speaker,
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
        logger.warning(f"Failed to write Fed analysis to bridge: {e}")


# ============================================================
# State Management (3-day dedup per speaker)
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
        logger.error(f"Failed to save Fed analysis state: {e}")
    finally:
        if lock_fd:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
            except Exception:
                pass


def _already_analyzed(speaker: str | None, state: dict) -> bool:
    """Check if we've already analyzed this speaker in the last 3 days."""
    key = speaker or "general"
    cutoff = datetime.now(_NY) - timedelta(days=3)
    for entry in state.get("analyses", []):
        if entry.get("speaker") == key:
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


def _record_analysis(speaker: str | None, state: dict) -> None:
    """Record that we analyzed a speaker."""
    state.setdefault("analyses", [])
    state["analyses"].append({
        "speaker": speaker or "general",
        "analyzed_at": datetime.now(_NY).isoformat(),
    })
    # Prune old entries (keep last 50)
    if len(state["analyses"]) > 50:
        state["analyses"] = state["analyses"][-50:]


# ============================================================
# Public Entry Points
# ============================================================

async def escalate(context, analysis_text: str, results: list[dict]) -> None:
    """Escalation entry point from news_intel.py.

    Called when fomc_fed topic produces speech/policy keywords.
    """
    logger.info("Fed analysis escalation triggered")

    # Detect speaker from the analysis text
    speaker = _detect_speaker(analysis_text)

    # Dedup check
    state = _load_state()
    if _already_analyzed(speaker, state):
        logger.info(f"Fed escalation: already analyzed {speaker or 'general'} recently, skipping")
        return

    # A) Gather research
    research_type = "rate_decision" if any(
        kw in analysis_text.lower()
        for kw in ("rate cut", "rate hike", "dot plot", "rate decision")
    ) else "speech_analysis"
    research_results = await _gather_fed_research(research_type)

    # Also include the original results from news_intel
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
        logger.info("Fed escalation: no research results found, skipping")
        return

    # B) Recall history
    member_history = _recall_member_history(speaker) if speaker else "No specific speaker identified"

    # C) Communication validity
    blackout = _is_blackout_period()
    trial_balloon = _detect_trial_balloon(research_results)

    # D) Generate analysis
    analysis = await _generate_rate_path_analysis(
        speaker=speaker,
        research_results=research_results,
        member_history=member_history,
        blackout=blackout,
        trial_balloon=trial_balloon,
    )

    if not analysis:
        logger.info("Fed escalation: analysis generation returned empty, skipping")
        return

    # Deliver
    urls = [r.get("url", "") for r in research_results if r.get("url")]
    await _deliver_fed_intel(context, speaker, analysis, urls)

    # Record
    _record_analysis(speaker, state)
    _save_state(state)

    logger.info(f"Fed analysis escalation complete: speaker={speaker or 'general'}")


async def run_fed_analysis(context) -> None:
    """Scheduled callback — daily 7:30 AM ET weekdays.

    Checks TE calendar for Fed events in next 3 days and runs analysis.
    """
    if not config.news_intel_enabled:
        return

    now = datetime.now(_NY)

    # Time guard: weekdays, 6 AM - 10 AM ET
    if now.weekday() >= 5:
        return
    if now.hour < 6 or now.hour >= 10:
        return

    logger.info("Starting scheduled Fed analysis...")

    try:
        from agent.integrations.econ_calendar import get_upcoming_events

        # Events in next 3 days
        upcoming = get_upcoming_events(hours=72)
        if not upcoming:
            # S186 W1: promote DEBUG→INFO (S185-B G17 — DEBUG invisible at INFO root)
            logger.info("Fed analysis exiting: no upcoming events in next 3d (likely stale econ_calendar)")
            return

        # Filter for Fed events
        fed_keywords = ("federal reserve", "fomc", "fed ", "fed chair", "fed governor",
                        "fed interest rate", "fed funds", "fed minutes", "fed beige book")
        fed_events = [
            ev for ev in upcoming
            if any(kw in ev.get("name", "").lower() for kw in fed_keywords)
        ]

        if not fed_events:
            # S186 W1: visibility — calendar may be fresh but Fed events may have already passed
            logger.info("Fed analysis exiting: 0 Fed events in next 3d window")
            return

        state = _load_state()

        for event in fed_events[:2]:  # Cap at 2 per run
            event_name = event.get("name", "")
            speaker = _detect_speaker(event_name)

            if _already_analyzed(speaker, state):
                logger.debug(f"Already analyzed {speaker or 'general'} recently, skipping")
                continue

            logger.info(f"Fed analysis: {event_name} (speaker: {speaker or 'general'})")

            # Determine research type
            name_lower = event_name.lower()
            if any(kw in name_lower for kw in ("rate decision", "rate", "fomc meeting")):
                research_type = "rate_decision"
            else:
                research_type = "speech_analysis"

            # A) Research
            research_results = await _gather_fed_research(research_type)
            if not research_results:
                # S186 W1: visibility on per-event upstream-empty
                logger.info(f"Fed analysis skipping {speaker or 'general'}: 0 research results (web_search budget? upstream issue?)")
                continue

            # B) Memory recall
            member_history = _recall_member_history(speaker) if speaker else "No specific speaker"

            # C) Communication validity
            blackout = _is_blackout_period()
            trial_balloon = _detect_trial_balloon(research_results)

            # D) Rate path analysis
            analysis = await _generate_rate_path_analysis(
                speaker=speaker,
                research_results=research_results,
                member_history=member_history,
                blackout=blackout,
                trial_balloon=trial_balloon,
            )

            if not analysis:
                # S186 W1: surface Anthropic failures (original silent-persistence root cause)
                logger.warning(f"Fed analysis: no rate-path analysis generated for {speaker or 'general'} (Anthropic failure?)")
                continue

            urls = [r.get("url", "") for r in research_results if r.get("url")]
            await _deliver_fed_intel(context, speaker, analysis, urls)

            _record_analysis(speaker, state)
            _save_state(state)

            logger.info(f"Fed analysis complete for {event_name}")
            await asyncio.sleep(2)

    except Exception as e:
        logger.error(f"Scheduled Fed analysis failed: {e}", exc_info=True)
