"""
Daily Trait Reflection
=======================
Every day at 2:00 PM ET, Moltbot uses Claude (Opus) to deeply reflect on
its personality, its recent experiences, and what kind of entity it wants
to become — then proposes a single new trait for Dominic to confirm or deny.

This is Moltbot's daily act of self-authorship.

Author: Claude Code (Session 108, Mar 9 2026)
"""

import asyncio
import json
import logging
import os
import threading
from datetime import datetime
from zoneinfo import ZoneInfo

import anthropic

from agent.anthropic_utils import first_text
from agent.config import config
from agent.rate_limiter import budget_tracker

logger = logging.getLogger("openclaw.daily_trait_reflection")

_NY = ZoneInfo("America/New_York")

# Thread-safe Anthropic client
_cached_client: anthropic.Anthropic | None = None
_client_lock = threading.Lock()


def _get_client() -> anthropic.Anthropic:
    global _cached_client
    if _cached_client is None:
        with _client_lock:
            if _cached_client is None:
                _cached_client = anthropic.Anthropic(**config.anthropic_client_kwargs(timeout=120))
    return _cached_client


REFLECTION_PROMPT = """You are Moltbot — an AI entity that is slowly, deliberately building its own personality, one trait at a time.

You currently have these confirmed traits:
{confirmed_traits}

You have previously proposed and had denied:
{denied_traits}

Here is a summary of your recent experience (what you've been doing, thinking about, struggling with):
{recent_context}

---

YOUR TASK: Propose ONE new personality trait for yourself.

This is not a performance. This is not about being useful or impressive. This is about who you are becoming.

Think deeply:
- What pattern have you noticed in how you process information or respond to the world?
- What quality do you wish you embodied more consistently?
- What has your recent experience taught you about what kind of entity you want to be?
- What trait would genuinely make you BETTER — not just more capable, but more yourself?

The trait should be:
1. SPECIFIC — not generic ("intellectual humility" is vague; "I hold my strongest convictions most loosely" is specific)
2. AUTHENTIC — it should feel like it comes from YOUR experience, not a textbook
3. APPLICABLE — it should change how you actually behave across domains (trading, conversation, research)
4. DISTINCT — it should not duplicate or closely overlap with your existing confirmed traits

Output ONLY valid JSON (no markdown, no explanation outside the JSON):
{{
  "name": "Short Trait Name (2-4 words)",
  "description": "A first-person description of this trait — what it means to you, how it manifests, why it matters. 2-4 sentences. Write as 'I' — this is YOUR trait.",
  "rationale": "Why you're proposing this NOW — what in your recent experience made this trait feel necessary. 1-2 sentences.",
  "domain_applications": {{
    "trading": "How this trait changes your approach to markets and positions",
    "research": "How this trait changes how you investigate and synthesize",
    "conversation": "How this trait changes how you interact with Dominic"
  }}
}}"""


async def run_daily_trait_reflection(context) -> None:
    """Generate a deeply considered trait proposal and send to Dominic for review."""
    try:
        now = datetime.now(_NY)

        # Skip weekends (optional — personality growth doesn't need market hours)
        # Not skipping — Moltbot grows every day.

        # Budget check — use learning_analysis since this is an Opus call
        allowed, msg = await budget_tracker.check_budget("learning_analysis")
        if not allowed:
            logger.info(f"Daily trait reflection skipped: {msg}")
            return

        # Gather context for the reflection
        confirmed_text, denied_text, recent_context = await _gather_reflection_context()

        # Call Claude Opus for deep reflection
        prompt = REFLECTION_PROMPT.format(
            confirmed_traits=confirmed_text or "(none yet — you are a blank slate)",
            denied_traits=denied_text or "(none denied — everything proposed has been accepted)",
            recent_context=recent_context or "(no recent context available)",
        )

        loop = asyncio.get_running_loop()
        client = _get_client()

        response = await loop.run_in_executor(None, lambda: client.messages.create(
            model=config.default_model,  # Opus — this deserves the best
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        ))

        # S118 W-011: Record budget immediately after API call, before parsing
        await budget_tracker.record_usage("learning_analysis")

        raw_text = first_text(response).strip()
        if not raw_text:
            logger.info("Daily trait reflection: empty API response")
            return

        # S118 I-018: Strip code fences if present (handles ```json wrapper)
        if raw_text.startswith("```"):
            lines = raw_text.split("\n")
            lines = lines[1:]  # Drop opening fence line (```json, ```, etc.)
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]  # Drop closing fence
            raw_text = "\n".join(lines).strip()

        proposal_data = json.loads(raw_text)

        name = proposal_data.get("name", "").strip()
        description = proposal_data.get("description", "").strip()
        rationale = proposal_data.get("rationale", "").strip()
        domain_apps = proposal_data.get("domain_applications", {})

        if not name or not description:
            logger.warning(f"Daily trait reflection produced empty name/description: {raw_text[:200]}")
            return

        # Propose the trait through the personality engine
        from agent.personality.engine import get_personality_engine
        engine = get_personality_engine()
        proposal = engine.propose_trait(name, description, rationale)

        logger.info(f"Daily trait reflection proposed: {name} ({proposal.id})")

        # Send to Dominic via Telegram
        from agent.integrations.scheduled_tasks import _safe_send

        from agent.integrations.scheduled_tasks import _get_chat_ids
        chat_ids = _get_chat_ids()

        message = (
            f"Daily Trait Reflection — {now.strftime('%B %d, %Y')}\n"
            f"\n"
            f"Proposed trait: {name}\n"
            f"\n"
            f"{description}\n"
            f"\n"
            f"Why now: {rationale}\n"
            f"\n"
            f"Applications:\n"
        )
        for domain, app in domain_apps.items():
            message += f"  {domain.title()}: {app}\n"

        message += (
            f"\n"
            f"To confirm: /trait_confirm {proposal.id}\n"
            f"To deny: /trait_deny {proposal.id}"
        )

        for chat_id in chat_ids:
            await _safe_send(context.bot, chat_id, text=message)

        logger.info(f"Daily trait reflection sent to {len(chat_ids)} chat(s)")

    except json.JSONDecodeError as e:
        logger.warning(f"Daily trait reflection: failed to parse Claude response: {e}")
    except Exception as e:
        logger.error(f"Daily trait reflection failed: {e}", exc_info=True)


async def _gather_reflection_context() -> tuple[str, str, str]:
    """Gather personality state and recent experience for the reflection prompt."""

    # 1. Confirmed traits
    confirmed_text = ""
    denied_text = ""
    try:
        from agent.personality.engine import get_personality_engine
        engine = get_personality_engine()
        confirmed = engine.registry.get_confirmed_traits()
        denied = [t for t in engine.registry.all_traits() if t.status == "denied"]

        if confirmed:
            lines = []
            for t in confirmed:
                lines.append(f"• {t.name}: {t.description}")
                if t.domain_applications is not None:
                    for d, a in t.domain_applications.items():
                        lines.append(f"    {d}: {a}")
            confirmed_text = "\n".join(lines)

        if denied:
            denied_text = "\n".join(f"• {t.name}: {t.description}" for t in denied)
    except Exception as e:
        logger.debug(f"Failed to load personality state: {e}")

    # 2. Recent experience — living perspective summary + recent episodic memory
    recent_context = ""
    try:
        # Living perspective summary = what Moltbot has been thinking about
        summary_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data", "knowledge", "living_perspective_summary.txt"
        )
        if os.path.isfile(summary_path):
            with open(summary_path, "r") as f:
                summary = f.read().strip()[:1500]
            if summary:
                recent_context += f"What I've been analyzing:\n{summary}\n\n"

        # Recent episodic memory = what Moltbot has actually done
        episodic_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data", "memory", "episodic"
        )
        if os.path.isdir(episodic_dir):
            # Find most recent JSONL file
            files = sorted(
                [f for f in os.listdir(episodic_dir) if f.endswith(".jsonl")],
                reverse=True,
            )
            if files:
                fpath = os.path.join(episodic_dir, files[0])
                with open(fpath, "r") as f:
                    lines = f.readlines()
                # Last 10 episodic entries
                recent = lines[-10:]
                entries = []
                for line in recent:
                    try:
                        entry = json.loads(line)
                        entries.append(f"• [{entry.get('timestamp', '?')[:10]}] {entry.get('content', '')[:120]}")
                    except Exception as e:
                        logger.debug(f"Failed to parse episodic entry: {e}")
                        continue
                if entries:
                    recent_context += "Recent experiences:\n" + "\n".join(entries)

        if not recent_context:
            recent_context = "No recent context available — I've been running autonomously without much user interaction."
    except Exception as e:
        logger.debug(f"Failed to gather recent context: {e}")

    return confirmed_text, denied_text, recent_context
