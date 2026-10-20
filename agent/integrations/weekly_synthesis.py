"""
OpenClaw/Moltbot — Weekly Knowledge Synthesis (Feature 5)

Runs every Friday at 4:30 PM ET. Aggregates the week's learning:
  - Daily reflections + morning briefings + alignment checks
  - Identifies patterns appearing 3+ times → elevates to permanent knowledge
  - Checks for contradictions before saving elevated patterns
  - Writes one-paragraph market narrative
  - Appends to learning_log.json under weekly_syntheses[]
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
from agent.rate_limiter import budget_tracker

logger = logging.getLogger("openclaw.weekly_synthesis")

_NY = ZoneInfo("America/New_York")

_SYNTHESIS_SYSTEM = (
    "You are a trading intelligence analyst synthesizing a week of market learning. "
    "Identify recurring patterns (3+ occurrences), contradictions, and elevated insights. "
    "Be concise, quantitative, and actionable."
)

LEARNING_LOG_PATH = os.path.join(config.tradingbot_dir, "state", "learning_log.json")

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


SYNTHESIS_PROMPT = """You are synthesizing one week of trading intelligence into permanent knowledge.

DAILY REFLECTIONS (Mon-Fri):
{reflections}

MORNING BRIEFINGS (Mon-Fri):
{briefings}

ALIGNMENT CHECKS (Mon-Fri):
{alignment_checks}

INSTRUCTIONS:
1. Identify themes/insights appearing 3+ times across all entries
2. For each elevated pattern: provide title, category, content, and tags for permanent storage
3. Identify single-occurrence anomalies — flag for monitoring
4. Write a one-paragraph market narrative for the week
5. Note any contradictions between entries
6. If fewer than 3 trading days of data exist (holiday week), note the reduced sample

Valid categories for elevated patterns: trading_strategies, trading_market_analysis

Output ONLY valid JSON (no markdown, no explanation):
{{
  "market_narrative": "one paragraph summary of the week",
  "elevated_patterns": [
    {{
      "title": "pattern title",
      "category": "trading_strategies or trading_market_analysis",
      "content": "detailed description of the pattern",
      "tags": ["tag1", "tag2"]
    }}
  ],
  "anomalies": ["single occurrence items worth monitoring"],
  "contradictions": ["any contradictions found between entries"],
  "week_summary_stats": {{
    "reflections_analyzed": 0,
    "briefings_analyzed": 0,
    "alignment_checks_analyzed": 0,
    "patterns_elevated": 0,
    "trading_days": 0
  }}
}}"""


def _load_learning_log() -> dict:
    """Load learning_log.json with shared lock."""
    if not os.path.isfile(LEARNING_LOG_PATH):
        return {}
    try:
        lock_path = LEARNING_LOG_PATH + ".lock"
        with open(lock_path, "a+") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_SH)
            try:
                with open(LEARNING_LOG_PATH, "r") as f:
                    return json.load(f)
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to read learning_log.json: {e}")
        return {}


def _get_week_entries(data: dict) -> dict:
    """Extract this week's entries (Monday through Friday) from learning_log.json."""
    now = datetime.now(_NY)
    # Find Monday of this week
    monday = now - timedelta(days=now.weekday())
    monday = monday.replace(hour=0, minute=0, second=0, microsecond=0)
    monday_iso = monday.isoformat()

    def _is_this_week(timestamp_str: str) -> bool:
        """Check if a timestamp falls within this week (Mon-Fri)."""
        if not timestamp_str:
            return False
        try:
            return timestamp_str >= monday_iso
        except (ValueError, TypeError):
            return False

    reflections = []
    for entry in data.get("daily_reflections", []):
        ts = entry.get("timestamp", entry.get("date", ""))
        if _is_this_week(ts):
            reflections.append(entry)

    briefings = []
    for entry in data.get("morning_briefings", []):
        ts = entry.get("timestamp", entry.get("date", ""))
        if _is_this_week(ts):
            briefings.append(entry)

    alignment_checks = []
    for entry in data.get("alignment_checks", []):
        ts = entry.get("timestamp", "")
        if _is_this_week(ts):
            alignment_checks.append(entry)

    return {
        "reflections": reflections,
        "briefings": briefings,
        "alignment_checks": alignment_checks,
    }


def _format_entries_for_prompt(entries: list[dict], max_chars: int = 3000) -> str:
    """Format a list of entries into a readable string for the synthesis prompt."""
    if not entries:
        return "(none available this week)"
    lines = []
    total_chars = 0
    for entry in entries:
        text = json.dumps(entry, indent=2, default=str)
        if total_chars + len(text) > max_chars:
            lines.append(f"... and {len(entries) - len(lines)} more entries (truncated)")
            break
        lines.append(text)
        total_chars += len(text)
    return "\n".join(lines)


async def _synthesize_with_claude(week_data: dict) -> dict | None:
    """Use Claude Opus to synthesize the week's learning."""
    allowed, msg = await budget_tracker.check_budget("learning_synthesis")
    if not allowed:
        logger.warning(f"Learning synthesis budget exhausted: {msg}")
        return None

    prompt = SYNTHESIS_PROMPT.format(
        reflections=_format_entries_for_prompt(week_data["reflections"]),
        briefings=_format_entries_for_prompt(week_data["briefings"]),
        alignment_checks=_format_entries_for_prompt(week_data["alignment_checks"]),
    )

    try:
        client = _get_client()
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, lambda: client.messages.create(
            model=config.judgment_model,  # S201: Fable 5 judgment tier (weekly consolidation)
            max_tokens=4096,
            system=cached_system_block(_SYNTHESIS_SYSTEM),
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": "medium"},
        ))
        # S121 W-53: Record budget after API call succeeds (S118 pattern)
        await budget_tracker.record_usage("learning_synthesis")

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

    except json.JSONDecodeError as e:
        logger.error(f"Synthesis output was not valid JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Synthesis Claude call failed: {e}", exc_info=True)
        return None


async def _elevate_patterns(patterns: list[dict]) -> int:
    """Save elevated patterns as permanent knowledge entries. Returns count saved."""
    if not patterns:
        return 0

    saved = 0
    for pattern in patterns:
        title = pattern.get("title", "").strip()
        category = pattern.get("category", "trading_strategies")
        content = pattern.get("content", "").strip()
        tags = pattern.get("tags", [])

        if not title or not content:
            continue

        # Validate category
        valid_categories = {"trading_strategies", "trading_market_analysis"}
        if category not in valid_categories:
            category = "trading_strategies"

        # Check for existing contradictions before saving
        try:
            from agent.tools.knowledge_tools import handle_recall_knowledge
            existing = await handle_recall_knowledge({
                "keywords": title.split()[0] if title.split() else "",
                "category": category,
                "limit": 5,
            })
            if "contradicts" in existing.lower() or "opposing" in existing.lower():
                logger.warning(
                    f"Skipping elevated pattern '{title}' — potential contradiction with existing knowledge"
                )
                continue
        except Exception as e:
            logger.debug(f"Recall check failed for pattern '{title}': {e}")

        try:
            from agent.tools.knowledge_tools import handle_save_knowledge
            result = await handle_save_knowledge({
                "title": f"[Weekly Pattern] {title}",
                "category": category,
                "content": content,
                "tags": tags + ["weekly_synthesis", "elevated_pattern"],
                "source": "weekly_synthesis",
            })
            if result and not result.startswith("ERROR"):
                saved += 1
                logger.info(f"Elevated pattern saved: '{title}'")
            else:
                logger.warning(f"Failed to save elevated pattern '{title}': {result}")
        except Exception as e:
            logger.warning(f"Failed to save elevated pattern '{title}': {e}")

    return saved


LEARNING_LOG_LOCK = LEARNING_LOG_PATH + ".lock"


def _write_synthesis_result(synthesis: dict, elevated_count: int) -> None:
    """Append synthesis result to learning_log.json under weekly_syntheses[]."""
    entry = {
        "timestamp": datetime.now(_NY).isoformat(),
        "market_narrative": synthesis.get("market_narrative", ""),
        "elevated_patterns": synthesis.get("elevated_patterns", []),
        "anomalies": synthesis.get("anomalies", []),
        "contradictions": synthesis.get("contradictions", []),
        "stats": synthesis.get("week_summary_stats", {}),
        "patterns_saved": elevated_count,
    }

    try:
        # Exclusive lock file for atomic read-modify-write (prevents TOCTOU)
        with open(LEARNING_LOG_LOCK, "a") as lockf:
            fcntl.flock(lockf, fcntl.LOCK_EX)

            data = {}
            if os.path.isfile(LEARNING_LOG_PATH):
                try:
                    with open(LEARNING_LOG_PATH, "r") as f:
                        data = json.load(f)
                except json.JSONDecodeError:
                    # S118 W-064: Corrupted file — start fresh rather than abort
                    logger.warning("learning_log.json corrupted — starting with empty data")
                    data = {}

            if "weekly_syntheses" not in data:
                data["weekly_syntheses"] = []
            data["weekly_syntheses"].append(entry)

            # Cap to last 52 entries (~1 year)
            if len(data["weekly_syntheses"]) > 52:
                data["weekly_syntheses"] = data["weekly_syntheses"][-52:]

            # Atomic write
            tmp = LEARNING_LOG_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp, LEARNING_LOG_PATH)

        logger.info(f"Weekly synthesis written: {elevated_count} patterns elevated")

    except Exception as e:
        logger.error(f"Failed to write weekly synthesis: {e}", exc_info=True)


async def _send_synthesis_summary(context, synthesis: dict, elevated_count: int) -> None:
    """Send synthesis summary to Telegram."""
    try:
        from agent.integrations.scheduled_tasks import _get_chat_ids, _safe_send
        chat_ids = _get_chat_ids()
        if not chat_ids:
            return

        narrative = synthesis.get("market_narrative", "No narrative generated")
        stats = synthesis.get("week_summary_stats", {})
        patterns = synthesis.get("elevated_patterns", [])
        contradictions = synthesis.get("contradictions", [])

        lines = [
            "Weekly Knowledge Synthesis\n",
            f"Market Narrative: {narrative[:500]}",
            f"\nStats: {stats.get('reflections_analyzed', 0)} reflections, "
            f"{stats.get('briefings_analyzed', 0)} briefings, "
            f"{stats.get('alignment_checks_analyzed', 0)} alignment checks",
            f"Patterns elevated: {elevated_count}",
        ]

        if patterns:
            lines.append("\nElevated patterns:")
            for p in patterns[:5]:
                lines.append(f"  - {p.get('title', '?')}")

        if contradictions:
            lines.append(f"\nContradictions found: {len(contradictions)}")

        message = "\n".join(lines)
        if len(message) > 4000:
            message = message[:3997] + "..."

        for chat_id in chat_ids:
            await _safe_send(context.bot, chat_id, text=message)

    except Exception as e:
        logger.warning(f"Failed to send synthesis summary: {e}")


async def run_weekly_synthesis(context) -> None:
    """Main entry point: run weekly knowledge synthesis.

    Called by JobQueue every Friday at 4:30 PM ET.
    """
    # Guard against catch-up firing on non-Friday
    if datetime.now(_NY).weekday() != 4:  # 4 = Friday
        logger.info("Weekly synthesis skipped — not Friday")
        return

    logger.info("Starting weekly knowledge synthesis...")

    try:
        # Load this week's data
        data = _load_learning_log()
        week_data = _get_week_entries(data)

        total_entries = (
            len(week_data["reflections"])
            + len(week_data["briefings"])
            + len(week_data["alignment_checks"])
        )

        if total_entries == 0:
            logger.info("No data available for weekly synthesis — skipping")
            return

        logger.info(
            f"Synthesis input: {len(week_data['reflections'])} reflections, "
            f"{len(week_data['briefings'])} briefings, "
            f"{len(week_data['alignment_checks'])} alignment checks"
        )

        # Run synthesis with Claude Opus
        synthesis = await _synthesize_with_claude(week_data)
        if synthesis is None:
            logger.warning("Weekly synthesis failed — no output from Claude")
            return

        # Elevate patterns to permanent knowledge
        elevated_count = await _elevate_patterns(synthesis.get("elevated_patterns", []))

        # Write synthesis result
        _write_synthesis_result(synthesis, elevated_count)

        # Send summary
        await _send_synthesis_summary(context, synthesis, elevated_count)

        logger.info(
            f"Weekly synthesis complete: {elevated_count} patterns elevated, "
            f"{len(synthesis.get('contradictions', []))} contradictions found"
        )

    except Exception as e:
        logger.error(f"Weekly synthesis failed: {e}", exc_info=True)
