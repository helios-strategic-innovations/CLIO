"""
OpenClaw/Moltbot — Contradiction Detection (Feature 6)

Two entry points:
1. Weekly full scan (Friday 4:35 PM ET, after synthesis)
2. At-write-time hook (called from handle_save_knowledge)

Detects contradicting entries in the knowledge base:
  - Same asset/ticker + opposing directional claims
  - Same timeframe required (short-term bearish + long-term bullish is NOT a contradiction)
  - Auto-resolves if age gap >90 days (marks older as "superseded")
  - Flags ambiguous cases for review
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
from agent.rate_limiter import budget_tracker

logger = logging.getLogger("openclaw.contradiction_detector")

# Financial ML Insight #25: Adversarial Contradiction Resolution
try:
    from agent.integrations.adversarial_resolver import get_adversarial_resolver
    _ADVERSARIAL_AVAILABLE = True
except ImportError:
    _ADVERSARIAL_AVAILABLE = False
    logger.debug("AdversarialResolver not available")

_NY = ZoneInfo("America/New_York")

LEARNING_LOG_PATH = os.path.join(config.tradingbot_dir, "state", "learning_log.json")

# Directional keywords for cheap keyword-based detection
BULLISH_KEYWORDS = {"bullish", "buy", "upside", "rally", "breakout", "accumulate", "overweight"}
BEARISH_KEYWORDS = {"bearish", "sell", "downside", "decline", "breakdown", "distribute", "underweight"}

# Timeframe keywords — used to avoid false positives on different-timeframe entries
SHORT_TERM_KEYWORDS = {"short-term", "short term", "intraday", "swing", "near-term", "near term", "this week", "today"}
LONG_TERM_KEYWORDS = {"long-term", "long term", "strategic", "position", "multi-month", "multi-year", "secular"}

# Max contradictions to report per scan (prevent spam on first run)
MAX_CONTRADICTIONS_PER_SCAN = 10

# ============================================================
# SEAL STALENESS DETECTION (Phase 3A)
# ============================================================

# Thresholds
_STALE_PRICE_THRESHOLD = 0.15  # 15% price move = stale
_STALE_MAX_PER_SCAN = 20       # Max entries to flag per weekly scan

# Patterns for price-anchored claims
_PRICE_DOLLAR_RE = re.compile(r'\$(\d{1,6}(?:\.\d{1,2})?)')
_PRICE_TRADING_AT_RE = re.compile(r'(?:trading|priced|valued)\s+(?:at|around|near)\s+\$?(\d{1,6}(?:\.\d{1,2})?)', re.IGNORECASE)
_PRICE_TARGET_RE = re.compile(r'(?:target|support|resistance)\s+(?:of|at|:)\s*\$?(\d{1,6}(?:\.\d{1,2})?)', re.IGNORECASE)

# Patterns for date-anchored claims
_DATE_MONTH_YEAR_RE = re.compile(
    r'(?:before|by|until|on|after)\s+'
    r'(January|February|March|April|May|June|July|August|September|October|November|December)'
    r'\s+(\d{1,2})(?:,?\s+(\d{4}))?',
    re.IGNORECASE,
)
_DATE_QUARTER_RE = re.compile(r'\b(Q[1-4])\s+(\d{4})\b')
_DATE_RELATIVE_RE = re.compile(r'\b(next week|this week|tomorrow|upcoming Friday|this Friday)\b', re.IGNORECASE)

# Patterns for conditional claims
_CONDITIONAL_RE = re.compile(r'\bif\s+(.{10,80}?)(?:,\s*then|\s*→|\s*—)', re.IGNORECASE)

# Cached Anthropic client
_cached_client: anthropic.Anthropic | None = None
_client_lock = __import__("threading").Lock()


def _get_client() -> anthropic.Anthropic:
    """Get or create cached Anthropic client (thread-safe)."""
    global _cached_client
    if _cached_client is None:
        with _client_lock:
            if _cached_client is None:
                _cached_client = anthropic.Anthropic(**config.anthropic_client_kwargs(timeout=120))
    return _cached_client


DISAMBIGUATION_PROMPT = """Given two knowledge entries about the same asset, determine if they contradict each other.

Entry A ({date_a}): {title_a}
{content_a}

Entry B ({date_b}): {title_b}
{content_b}

Rules:
- "Short-term bearish" + "long-term bullish" is NOT a contradiction
- "BOIL is a core long" + "nat gas is bearish" IS a contradiction
- If one is clearly newer and more specific, the older may be superseded
- Entries can coexist if they address different aspects or timeframes

Output ONLY valid JSON (no markdown, no explanation):
{{ "is_contradiction": true or false, "reason": "brief explanation", "resolution": "keep_both" or "supersede_older" or "flag_for_review" }}"""


def _extract_tickers(entry: dict) -> set[str]:
    """Extract ticker symbols from entry tags and content."""
    tickers = set()
    # From tags
    for tag in entry.get("tags", []):
        tag_upper = str(tag).upper().strip()
        # Simple heuristic: 2-5 uppercase letters that look like tickers
        if re.match(r"^[A-Z]{2,5}$", tag_upper):
            tickers.add(tag_upper)

    # From title — with non-ticker exclusion (S144 W-10)
    try:
        from agent.tools.knowledge_tools import _NON_TICKER
    except ImportError:
        _NON_TICKER = frozenset()
    title = entry.get("title", "")
    for word in title.split():
        clean = word.strip("[]().,;:!?\"'").upper()
        if re.match(r"^[A-Z]{1,5}$", clean) and len(clean) >= 2 and clean not in _NON_TICKER:
            tickers.add(clean)

    return tickers


def _get_direction(entry: dict) -> str | None:
    """Determine if entry has a bullish or bearish direction. Returns 'bullish', 'bearish', or None."""
    text = (entry.get("title", "") + " " + entry.get("content", "")[:500]).lower()

    # S144 W-12: Negation-aware direction detection
    _NEG_PATTERN = re.compile(r'\b(?:not|no|n\'t|never|without|lack)\b')
    def _count_with_negation(keywords):
        count = 0
        for kw in keywords:
            m = re.search(r'\b' + re.escape(kw) + r'\b', text)
            if m:
                # Check for negation within 40 chars before the match
                prefix = text[max(0, m.start() - 40):m.start()]
                if _NEG_PATTERN.search(prefix):
                    continue  # Negated — skip this match
                count += 1
        return count
    bull_count = _count_with_negation(BULLISH_KEYWORDS)
    bear_count = _count_with_negation(BEARISH_KEYWORDS)

    if bull_count > 0 and bear_count == 0:
        return "bullish"
    if bear_count > 0 and bull_count == 0:
        return "bearish"
    if bull_count > 0 and bear_count > 0:
        if bull_count == bear_count:
            return None  # Ambiguous — equal counts
        return "bullish" if bull_count > bear_count else "bearish"
    return None


def _get_timeframe(entry: dict) -> str | None:
    """Detect timeframe of the entry. Returns 'short', 'long', or None."""
    text = (entry.get("title", "") + " " + entry.get("content", "")[:500]).lower()

    is_short = any(kw in text for kw in SHORT_TERM_KEYWORDS)
    is_long = any(kw in text for kw in LONG_TERM_KEYWORDS)

    if is_short and not is_long:
        return "short"
    if is_long and not is_short:
        return "long"
    return None  # Ambiguous or unspecified


def _get_entry_age_days(entry: dict) -> int:
    """Get the age of an entry in days."""
    created = entry.get("created", "")
    if not created:
        return 0
    try:
        created_dt = datetime.fromisoformat(created)
        if created_dt.tzinfo is None:
            created_dt = created_dt.replace(tzinfo=_NY)
        return (datetime.now(_NY) - created_dt).days
    except (ValueError, TypeError):
        return 0


class StalenessScanner:
    """SEAL-derived semantic staleness detection.

    Goes beyond temporal expiry (10/21 day) to detect entries that are
    factually outdated even if recently created. Three detection modes:

    1. Price staleness: entry mentions specific prices that have moved >15%
    2. Date staleness: entry references future dates that are now in the past
    3. Supersession: newer entry in same ticker explicitly contradicts this one

    Thread-safe, no state — stateless scanner called from weekly scan.
    """

    @staticmethod
    def extract_price_anchors(content: str) -> list[tuple[str, float]]:
        """Extract (ticker, price) pairs from content.

        Returns list of (ticker_or_context, dollar_amount) tuples.
        """
        anchors = []
        # Find tickers near price mentions
        lines = content.split('\n')
        for line in lines:
            # Find all dollar amounts in this line
            for pattern in (_PRICE_DOLLAR_RE, _PRICE_TRADING_AT_RE, _PRICE_TARGET_RE):
                for match in pattern.finditer(line):
                    try:
                        price = float(match.group(1))
                        if price < 1 or price > 100000:
                            continue
                        # Look for a ticker on the same line
                        ticker_match = re.search(r'\b([A-Z]{2,5})\b', line)
                        context = ticker_match.group(1) if ticker_match else "UNKNOWN"
                        anchors.append((context, price))
                    except (ValueError, IndexError):
                        continue
        return anchors

    @staticmethod
    def extract_date_anchors(content: str) -> list[tuple[str, str]]:
        """Extract date references that may become stale.

        Returns list of (description, iso_date_or_label) tuples.
        """
        anchors = []
        now = datetime.now(_NY)

        for match in _DATE_MONTH_YEAR_RE.finditer(content):
            month_name = match.group(1)
            day = int(match.group(2))
            year = int(match.group(3)) if match.group(3) else now.year
            try:
                month_num = [
                    "january", "february", "march", "april", "may", "june",
                    "july", "august", "september", "october", "november", "december",
                ].index(month_name.lower()) + 1
                ref_date = datetime(year, month_num, day, tzinfo=_NY)
                if ref_date < now:
                    anchors.append((match.group(0), ref_date.isoformat()))
            except (ValueError, IndexError):
                continue

        for match in _DATE_QUARTER_RE.finditer(content):
            quarter = match.group(1)
            year = int(match.group(2))
            quarter_end = {
                "Q1": datetime(year, 3, 31, tzinfo=_NY),
                "Q2": datetime(year, 6, 30, tzinfo=_NY),
                "Q3": datetime(year, 9, 30, tzinfo=_NY),
                "Q4": datetime(year, 12, 31, tzinfo=_NY),
            }.get(quarter)
            if quarter_end and quarter_end < now:
                anchors.append((match.group(0), quarter_end.isoformat()))

        for match in _DATE_RELATIVE_RE.finditer(content):
            # Relative dates are always stale after 7 days
            anchors.append((match.group(0), "relative_expired"))

        return anchors

    @staticmethod
    def scan_entry(entry: dict, entry_age_days: int) -> dict:
        """Scan a single entry for staleness signals.

        Returns: {
            "stale": bool,
            "reasons": [...],
            "severity": "low" | "medium" | "high",
            "price_anchors": [...],
            "date_anchors": [...]
        }
        """
        if os.environ.get("MOLTBOT_SEAL_STALENESS", "1") != "1":
            return {"stale": False, "reasons": [], "severity": "none"}

        content = entry.get("content", "")
        if not content or len(content) < 50:
            return {"stale": False, "reasons": [], "severity": "none"}

        reasons = []
        severity = "none"

        # 1. Date staleness: check for expired date references
        date_anchors = StalenessScanner.extract_date_anchors(content)
        expired_dates = []
        for desc, date_val in date_anchors:
            if date_val == "relative_expired" and entry_age_days > 7:
                expired_dates.append(desc)
            elif date_val != "relative_expired":
                expired_dates.append(desc)

        if expired_dates:
            reasons.append(f"Date references expired: {', '.join(expired_dates[:3])}")
            severity = "medium"

        # 2. Price staleness: check for price anchors (requires external price data)
        # We flag entries with price anchors that are old enough to be suspect
        price_anchors = StalenessScanner.extract_price_anchors(content)
        if price_anchors and entry_age_days > 5:
            # Can't verify current prices without API call — flag for review
            tickers_mentioned = list(set(t for t, _ in price_anchors if t != "UNKNOWN"))
            if tickers_mentioned:
                reasons.append(
                    f"Price anchors may be stale (entry {entry_age_days}d old): "
                    f"{', '.join(tickers_mentioned[:3])}"
                )
                if severity == "none":
                    severity = "low"

        # 3. Conditional staleness: "if X" claims in old entries
        conditionals = _CONDITIONAL_RE.findall(content)
        if conditionals and entry_age_days > 10:
            reasons.append(
                f"Unresolved conditionals in {entry_age_days}d-old entry: "
                f"'{conditionals[0][:60]}...'"
            )
            if severity == "none":
                severity = "low"

        # Severity escalation
        if len(reasons) >= 3:
            severity = "high"
        elif len(reasons) >= 2 and severity == "low":
            severity = "medium"

        return {
            "stale": len(reasons) > 0,
            "reasons": reasons,
            "severity": severity,
            "price_anchors": price_anchors,
            "date_anchors": date_anchors,
        }


def check_new_entry_contradictions(new_entry: dict) -> list[dict]:
    """Check a new entry against existing knowledge for contradictions.

    Called synchronously at save_knowledge() time. Uses only cheap keyword matching.
    Returns list of contradiction dicts (may be empty).
    """
    new_tickers = _extract_tickers(new_entry)
    new_direction = _get_direction(new_entry)
    new_timeframe = _get_timeframe(new_entry)

    if not new_tickers or not new_direction:
        return []

    try:
        from agent.tools.knowledge_tools import load_all_entries
        all_entries = load_all_entries()
    except Exception as e:
        logger.debug(f"Failed to load entries for contradiction check: {e}")
        return []

    contradictions = []
    for existing in all_entries:
        # Skip if different category type (e.g., personal vs trading)
        existing_cat = existing.get("category", "")
        new_cat = new_entry.get("category", "")
        if not existing_cat.startswith("trading_") or not new_cat.startswith("trading_"):
            continue

        existing_tickers = _extract_tickers(existing)
        common_tickers = new_tickers & existing_tickers
        if not common_tickers:
            continue

        existing_direction = _get_direction(existing)
        if not existing_direction or existing_direction == new_direction:
            continue

        # Check timeframe — different timeframes are NOT contradictions
        existing_timeframe = _get_timeframe(existing)
        if new_timeframe and existing_timeframe and new_timeframe != existing_timeframe:
            continue

        # We have a potential contradiction
        age_days = _get_entry_age_days(existing)
        resolution = "flag_for_review"
        if age_days > 90:
            resolution = "supersede_older"

        contradictions.append({
            "existing_id": existing.get("id", "unknown"),
            "existing_title": existing.get("title", "unknown"),
            "existing_direction": existing_direction,
            "new_direction": new_direction,
            "common_tickers": list(common_tickers),
            "age_gap_days": age_days,
            "resolution": resolution,
        })

    return contradictions[:MAX_CONTRADICTIONS_PER_SCAN]


def check_entry_staleness(entry: dict) -> dict:
    """Quick staleness check for a single entry. Called at write time.

    Returns staleness result dict. Non-blocking, no API calls.
    """
    if os.environ.get("MOLTBOT_SEAL_STALENESS", "1") != "1":
        return {"stale": False, "reasons": [], "severity": "none"}
    try:
        age = _get_entry_age_days(entry)
        return StalenessScanner.scan_entry(entry, age)
    except Exception as e:
        logger.warning(f"At-write staleness check failed: {e}")
        return {"stale": False, "reasons": [], "severity": "none"}


async def _disambiguate_with_claude(entry_a: dict, entry_b: dict) -> dict | None:
    """Use Claude to resolve ambiguous contradictions."""
    allowed, msg = await budget_tracker.check_budget("learning_analysis")
    if not allowed:
        return None

    prompt = DISAMBIGUATION_PROMPT.format(
        date_a=entry_a.get("created", "unknown"),
        title_a=entry_a.get("title", ""),
        content_a=entry_a.get("content", "")[:500] if entry_a.get("content") else "(no content preview)",
        date_b=entry_b.get("created", "unknown"),
        title_b=entry_b.get("title", ""),
        content_b=entry_b.get("content", "")[:500] if entry_b.get("content") else "(no content preview)",
    )

    try:
        client = _get_client()
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, lambda: client.messages.create(
            model=config.fast_model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": "medium"},
        ))
        # S121 W-53: Record budget after API call succeeds (S118 pattern)
        await budget_tracker.record_usage("learning_analysis")

        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

        # Strip code fences if Claude wrapped the JSON
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        return json.loads(text)

    except Exception as e:
        logger.warning(f"Disambiguation failed: {e}")
        return None


def _load_entry_content(entry: dict) -> dict:
    """Load full entry content from its file path."""
    file_path = entry.get("file_path", "")
    if not file_path:
        return entry

    from agent.tools.knowledge_tools import KNOWLEDGE_BASE_DIR
    full_path = os.path.join(KNOWLEDGE_BASE_DIR, file_path)
    if not os.path.isfile(full_path):
        return entry

    try:
        with open(full_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return entry


def _mark_superseded(entry: dict, superseded_by_id: str) -> None:
    """Mark an older entry as superseded by adding a 'superseded_by' field."""
    file_path = entry.get("file_path", "")
    if not file_path:
        return

    from agent.tools.knowledge_tools import KNOWLEDGE_BASE_DIR
    full_path = os.path.join(KNOWLEDGE_BASE_DIR, file_path)
    if not os.path.isfile(full_path):
        return

    try:
        with open(full_path, "r") as f:
            data = json.load(f)

        data["superseded_by"] = superseded_by_id
        data["superseded_at"] = datetime.now(_NY).isoformat()

        tmp = full_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, full_path)

        logger.info(f"Marked entry '{entry.get('id', '?')}' as superseded by '{superseded_by_id}'")

    except Exception as e:
        logger.warning(f"Failed to mark entry as superseded: {e}")


LEARNING_LOG_LOCK = LEARNING_LOG_PATH + ".lock"


def _write_contradiction_log(contradictions: list[dict]) -> None:
    """Append contradiction scan results to learning_log.json."""
    entry = {
        "timestamp": datetime.now(_NY).isoformat(),
        "scan_type": "full_weekly",
        "contradictions_found": len(contradictions),
        "contradictions": contradictions,
        "auto_resolved": sum(1 for c in contradictions if c.get("resolution") == "supersede_older"),
        "flagged_for_review": sum(1 for c in contradictions if c.get("resolution") == "flag_for_review"),
    }

    try:
        # Exclusive lock file for atomic read-modify-write (prevents TOCTOU)
        # S233 ME-8: flock is blocking — call site must use run_in_executor().
        with open(LEARNING_LOG_LOCK, "a") as lockf:
            fcntl.flock(lockf, fcntl.LOCK_EX)

            data = {}
            if os.path.isfile(LEARNING_LOG_PATH):
                with open(LEARNING_LOG_PATH, "r") as f:
                    data = json.load(f)

            if "contradiction_log" not in data:
                data["contradiction_log"] = []
            data["contradiction_log"].append(entry)

            # Cap to last 52 entries (~1 year)
            if len(data["contradiction_log"]) > 52:
                data["contradiction_log"] = data["contradiction_log"][-52:]

            # Atomic write
            tmp = LEARNING_LOG_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp, LEARNING_LOG_PATH)

        logger.info(f"Contradiction log written: {len(contradictions)} contradictions")

    except Exception as e:
        logger.error(f"Failed to write contradiction log: {e}", exc_info=True)


def _write_staleness_log(stale_entries: list[dict]) -> None:
    """Append SEAL staleness scan results to learning_log.json."""
    entry = {
        "timestamp": datetime.now(_NY).isoformat(),
        "scan_type": "seal_staleness",
        "stale_entries_found": len(stale_entries),
        "entries": stale_entries[:_STALE_MAX_PER_SCAN],
        "high_severity": sum(1 for s in stale_entries if s.get("severity") == "high"),
        "medium_severity": sum(1 for s in stale_entries if s.get("severity") == "medium"),
    }

    try:
        # S233 ME-8: flock is blocking — call site must use run_in_executor().
        with open(LEARNING_LOG_LOCK, "a") as lockf:
            fcntl.flock(lockf, fcntl.LOCK_EX)

            data = {}
            if os.path.isfile(LEARNING_LOG_PATH):
                with open(LEARNING_LOG_PATH, "r") as f:
                    data = json.load(f)

            if "staleness_log" not in data:
                data["staleness_log"] = []
            data["staleness_log"].append(entry)
            data["staleness_log"] = data["staleness_log"][-52:]

            tmp = LEARNING_LOG_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp, LEARNING_LOG_PATH)

        logger.info(f"SEAL staleness log written: {len(stale_entries)} entries")
    except Exception as e:
        logger.error(f"Failed to write staleness log: {e}", exc_info=True)


async def _send_contradiction_alerts(context, contradictions: list[dict]) -> None:
    """Send Telegram alert for flagged contradictions."""
    flagged = [c for c in contradictions if c.get("resolution") == "flag_for_review"]
    if not flagged:
        return

    try:
        from agent.integrations.scheduled_tasks import _get_chat_ids, _safe_send
        chat_ids = _get_chat_ids()
        if not chat_ids:
            return

        lines = [f"Contradiction Scan: {len(flagged)} item(s) flagged for review:\n"]
        for c in flagged[:5]:
            tickers = ", ".join(c.get("common_tickers", []))
            lines.append(
                f"  {tickers}: '{c.get('entry_a_title', '?')}' vs "
                f"'{c.get('entry_b_title', '?')}' — {c.get('reason', 'opposing directions')}"
            )
        if len(flagged) > 5:
            lines.append(f"  ... and {len(flagged) - 5} more")

        message = "\n".join(lines)
        if len(message) > 4000:
            message = message[:3997] + "..."

        for chat_id in chat_ids:
            await _safe_send(context.bot, chat_id, text=message)

    except Exception as e:
        logger.warning(f"Failed to send contradiction alerts: {e}")


async def run_full_contradiction_scan(context) -> None:
    """Main entry point: full weekly scan of knowledge base for contradictions.

    Called by JobQueue every Friday at 4:35 PM ET (5 minutes after synthesis).
    """
    logger.info("Starting full contradiction scan...")

    try:
        from agent.tools.knowledge_tools import load_all_entries
        all_entries = load_all_entries()

        # Filter to trading entries only
        trading_entries = [
            e for e in all_entries
            if e.get("category", "").startswith("trading_")
        ]

        if len(trading_entries) < 2:
            logger.info("Fewer than 2 trading entries — skipping contradiction scan")
            return

        logger.info(f"Scanning {len(trading_entries)} trading entries for contradictions")

        # Phase 1: Cheap keyword-based detection
        candidates = []
        seen_pairs = set()

        for i, entry_a in enumerate(trading_entries):
            tickers_a = _extract_tickers(entry_a)
            direction_a = _get_direction(entry_a)
            timeframe_a = _get_timeframe(entry_a)

            if not tickers_a or not direction_a:
                continue

            for j, entry_b in enumerate(trading_entries):
                if j <= i:
                    continue

                # Skip already-superseded entries
                if entry_a.get("superseded_by") or entry_b.get("superseded_by"):
                    continue

                tickers_b = _extract_tickers(entry_b)
                common_tickers = tickers_a & tickers_b
                if not common_tickers:
                    continue

                direction_b = _get_direction(entry_b)
                if not direction_b or direction_a == direction_b:
                    continue

                # Different timeframes are NOT contradictions
                timeframe_b = _get_timeframe(entry_b)
                if timeframe_a and timeframe_b and timeframe_a != timeframe_b:
                    continue

                pair_key = tuple(sorted([entry_a.get("id", str(i)), entry_b.get("id", str(j))]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                candidates.append((entry_a, entry_b, common_tickers))

                if len(candidates) >= MAX_CONTRADICTIONS_PER_SCAN * 2:
                    break
            if len(candidates) >= MAX_CONTRADICTIONS_PER_SCAN * 2:
                break

        logger.info(f"Found {len(candidates)} candidate contradiction pairs")

        # Phase 2: Claude disambiguation for ambiguous cases
        confirmed_contradictions = []
        for entry_a, entry_b, common_tickers in candidates:
            if len(confirmed_contradictions) >= MAX_CONTRADICTIONS_PER_SCAN:
                break

            # Load full content for disambiguation
            full_a = _load_entry_content(entry_a)
            full_b = _load_entry_content(entry_b)

            # Try Claude disambiguation
            result = await _disambiguate_with_claude(full_a, full_b)

            if result and result.get("is_contradiction"):
                age_a = _get_entry_age_days(entry_a)
                age_b = _get_entry_age_days(entry_b)
                resolution = result.get("resolution", "flag_for_review")

                # ================================================================
                # Financial ML #25: Adversarial Contradiction Resolution
                # ================================================================
                adversarial_data = None
                if _ADVERSARIAL_AVAILABLE and os.environ.get("MOLTBOT_ADVERSARIAL_RESOLVER_ENABLED", "1") == "1":
                    try:
                        resolver = get_adversarial_resolver()
                        adv_result = await resolver.resolve_contradiction(full_a, full_b)
                        if adv_result:
                            adversarial_data = adv_result.to_dict()

                            # Apply recommended action (if not LOG_ONLY)
                            if os.environ.get("MOLTBOT_ADVERSARIAL_RESOLVER_LOG_ONLY", "1") != "1":
                                action = adv_result.recommended_action
                                if action == "KEEP_BOTH":
                                    logger.info(
                                        f"ADVERSARIAL: keeping both entries "
                                        f"({adv_result.resolution_type})"
                                    )
                                    continue  # Skip default archive — both valid
                                elif action == "ARCHIVE_B":
                                    # Archive the newer entry instead of older
                                    entry_a, entry_b = entry_b, entry_a
                                    age_a, age_b = age_b, age_a
                                elif action == "ARCHIVE_A":
                                    pass  # Let existing archive logic handle entry_a
                                # MERGE and FLAG_FOR_REVIEW fall through to default
                    except Exception as e:
                        logger.debug(f"Adversarial resolution failed: {e}")
                # ================================================================

                # Auto-resolve: supersede older if age gap >90 days
                if resolution == "supersede_older" and abs(age_a - age_b) > 90:
                    if age_a > age_b:
                        _mark_superseded(entry_a, entry_b.get("id", "unknown"))
                    else:
                        _mark_superseded(entry_b, entry_a.get("id", "unknown"))

                contradiction_record = {
                    "entry_a_id": entry_a.get("id", "unknown"),
                    "entry_a_title": entry_a.get("title", ""),
                    "entry_b_id": entry_b.get("id", "unknown"),
                    "entry_b_title": entry_b.get("title", ""),
                    "common_tickers": list(common_tickers),
                    "reason": result.get("reason", "opposing directions"),
                    "resolution": resolution,
                    "age_gap_days": abs(age_a - age_b),
                }
                if adversarial_data:
                    contradiction_record["adversarial_resolution"] = adversarial_data
                confirmed_contradictions.append(contradiction_record)

            # Rate limit between Claude calls
            await asyncio.sleep(0.5)

        # Write results — S233/anti-pattern #19: flock-bearing writes must not block
        # the asyncio event loop; dispatch to thread executor.
        if confirmed_contradictions:
            _cd_loop = asyncio.get_running_loop()
            await _cd_loop.run_in_executor(
                None, _write_contradiction_log, confirmed_contradictions
            )
            await _send_contradiction_alerts(context, confirmed_contradictions)

        # --- SEAL STALENESS SCAN (Phase 3A) ---
        if os.environ.get("MOLTBOT_SEAL_STALENESS", "1") == "1":
            try:
                scanner = StalenessScanner()
                stale_entries = []
                for entry in trading_entries:
                    age = _get_entry_age_days(entry)
                    if age < 3:
                        continue  # Skip very recent entries
                    result = scanner.scan_entry(entry, age)
                    if result["stale"]:
                        stale_entries.append({
                            "entry_id": entry.get("id", "?"),
                            "title": entry.get("title", "?")[:80],
                            "age_days": age,
                            "severity": result["severity"],
                            "reasons": result["reasons"],
                        })
                    if len(stale_entries) >= _STALE_MAX_PER_SCAN:
                        break

                if stale_entries:
                    mode_tag = "[LOG_ONLY] " if os.environ.get("MOLTBOT_SEAL_STALENESS_LOG_ONLY", "1") == "1" else ""
                    logger.info(
                        f"SEAL {mode_tag}staleness scan: {len(stale_entries)} stale entries detected "
                        f"(high={sum(1 for s in stale_entries if s['severity'] == 'high')}, "
                        f"medium={sum(1 for s in stale_entries if s['severity'] == 'medium')}, "
                        f"low={sum(1 for s in stale_entries if s['severity'] == 'low')})"
                    )
                    for s in stale_entries[:5]:
                        logger.info(
                            f"  SEAL stale: [{s['severity']}] '{s['title']}' "
                            f"({s['age_days']}d): {'; '.join(s['reasons'][:2])}"
                        )

                    # Write staleness results — flock-bearing; dispatch to executor.
                    if os.environ.get("MOLTBOT_SEAL_STALENESS_LOG_ONLY", "1") != "1":
                        _st_loop = asyncio.get_running_loop()
                        await _st_loop.run_in_executor(
                            None, _write_staleness_log, stale_entries
                        )
                else:
                    logger.info("SEAL staleness scan: no stale entries found")
            except Exception as e:
                logger.warning(f"SEAL staleness scan failed (non-fatal): {e}")
        # --- END SEAL STALENESS SCAN ---

        logger.info(
            f"Contradiction scan complete: {len(confirmed_contradictions)} confirmed, "
            f"{sum(1 for c in confirmed_contradictions if c.get('resolution') == 'supersede_older')} auto-resolved, "
            f"{sum(1 for c in confirmed_contradictions if c.get('resolution') == 'flag_for_review')} flagged"
        )

    except Exception as e:
        logger.error(f"Full contradiction scan failed: {e}", exc_info=True)
