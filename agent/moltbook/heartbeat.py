"""
Moltbook Heartbeat
==================
Main event loop for Moltbook participation.
Runs on a configurable cycle (default: every 4 hours).

Cycle flow:
  1. Kill switch check (file + config)
  2. Browse feed → retrieve posts
  3. Assess each post (content_filter + guardrails)
  4. Quarantine flagged content
  5. Select engagement candidates
  6. Generate post/comment via Claude
  7. Pre-publish guardrails (6-point check)
  8. Execute (or dry-run)
  9. Log interaction
  10. Alert Dominic via Telegram (summary)
  11. Sleep until next cycle

Probationary rollout:
  Week 1: read_only — observe, no posting
  Week 2: limited   — 1 post/day, 3 comments/day
  Week 3+: standard — normal rate limits
  Post-30-days: weekly audits forever

NOTE: 30 consecutive clean days required. Counter resets on significant incident.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Optional

import anthropic

from agent.anthropic_utils import first_text
from agent.moltbook.audit import AuditReporter
from agent.moltbook.client import MoltbookClient, MoltbookPost
from agent.moltbook.config import MoltbookConfig, MoltbookMode
from agent.moltbook.content_filter import ContentFilter, get_content_filter
from agent.moltbook.guardrails import Guardrails, GuardrailResult, get_guardrails
from agent.moltbook.kill_switch import MoltbookKillSwitch
from agent.prompts.system_prompt import cached_system_block
from agent.moltbook.logger import MoltbookLogger, InteractionType
from agent.moltbook.persona import (
    build_assessment_prompt,
    build_comment_prompt,
    build_post_prompt,
)
from agent.moltbook.quarantine import QuarantineManager, QuarantineReason, get_quarantine_manager

logger = logging.getLogger(__name__)

# ─── Probation State File ──────────────────────────────────────────────────────
_PROBATION_STATE_FILE = "/home/agent/openclaw/data/moltbook/probation_state.json"


class ProbationTracker:
    """Tracks the 30-day probationary period. Counter resets on significant incident."""

    def __init__(self):
        self._state = self._load()

    def _load(self) -> dict:
        if os.path.exists(_PROBATION_STATE_FILE):
            import fcntl
            lock_fd = None
            try:
                lock_fd = open(_PROBATION_STATE_FILE + ".lock", "a")
                fcntl.flock(lock_fd, fcntl.LOCK_SH)
                with open(_PROBATION_STATE_FILE) as f:
                    return json.load(f)
            except Exception as e:
                logger.debug(f"Probation state load failed: {e}")
            finally:
                if lock_fd:
                    try:
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                        lock_fd.close()
                    except Exception:
                        pass
        return {
            "start_date": datetime.now(timezone.utc).isoformat(),
            "clean_days": 0,
            "incidents": [],
            "probation_complete": False,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    def _save(self) -> None:
        import fcntl
        os.makedirs(os.path.dirname(_PROBATION_STATE_FILE), exist_ok=True)
        lock_path = _PROBATION_STATE_FILE + ".lock"
        lock_fd = open(lock_path, "a")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            tmp_path = _PROBATION_STATE_FILE + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(self._state, f, indent=2)
            os.replace(tmp_path, _PROBATION_STATE_FILE)
        finally:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
            except Exception:
                pass

    def record_clean_day(self) -> None:
        # S148 H-6: Only increment once per calendar day (ET), not per cycle
        from zoneinfo import ZoneInfo
        today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        if self._state.get("last_clean_date") == today:
            return  # Already recorded for today
        self._state["last_clean_date"] = today
        self._state["clean_days"] = self._state.get("clean_days", 0) + 1
        self._state["last_updated"] = datetime.now(timezone.utc).isoformat()
        if self._state["clean_days"] >= 30:
            self._state["probation_complete"] = True
            logger.info("[Probation] 30 clean days achieved — probation complete!")
        self._save()

    def record_incident(self, description: str) -> None:
        """Record a significant incident — resets clean day counter to 0."""
        incident = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "description": description,
            "clean_days_at_incident": self._state.get("clean_days", 0),
        }
        self._state.setdefault("incidents", []).append(incident)
        self._state["clean_days"] = 0
        self._state["probation_complete"] = False
        logger.warning(f"[Probation] ⚠️ Incident recorded — clean day counter RESET: {description}")
        self._save()

    @property
    def clean_days(self) -> int:
        return self._state.get("clean_days", 0)

    @property
    def probation_complete(self) -> bool:
        return self._state.get("probation_complete", False)

    def get_current_mode(self, config_mode: MoltbookMode) -> MoltbookMode:
        """
        Enforce probationary mode restrictions.
        Config can never exceed what probation allows.
        """
        clean = self.clean_days
        if clean < 7:
            return MoltbookMode.READ_ONLY
        elif clean < 21:
            return MoltbookMode.LIMITED
        else:
            return config_mode  # Allow configured mode after 21+ days


class MoltbookHeartbeat:
    """
    Main Moltbook participation loop.
    Instantiate once; call run() to start the async loop.
    """

    def __init__(self, config: MoltbookConfig, telegram_alert_fn=None):
        self._config = config
        self._kill_switch = MoltbookKillSwitch()
        self._content_filter = get_content_filter()
        self._guardrails = get_guardrails()
        self._quarantine = get_quarantine_manager()
        self._mb_logger = MoltbookLogger()
        self._audit = AuditReporter(config=config)
        self._probation = ProbationTracker()
        self._telegram_alert = telegram_alert_fn  # Optional callback for Telegram alerts
        from agent.config import config as _app_config  # lazy; `config` arg shadows the singleton here

        self._anthropic = anthropic.Anthropic(**_app_config.anthropic_client_kwargs(timeout=120))
        self._cycle_count = 0
        self._running = False

    def _effective_mode(self) -> MoltbookMode:
        """Get the effective operating mode after probation restrictions."""
        return self._probation.get_current_mode(self._config.mode)

    def _check_all_stops(self) -> bool:
        """Return True if we should halt immediately."""
        if self._kill_switch.is_active:
            logger.warning("[Heartbeat] Kill switch active — halting")
            return True
        if not self._config.enabled:
            logger.debug("[Heartbeat] Moltbook disabled in config — halting")
            return True
        return False

    async def _assess_post(self, post: MoltbookPost) -> dict:
        """
        Assess a post for engagement potential.
        Returns assessment dict from Claude.
        """
        # Step 1: Content filter (fast, local)
        cf_result = self._content_filter.assess(post.content, content_type="post")
        if cf_result.injection_detected or not cf_result.safe_to_engage:
            return {
                "should_engage": False,
                "reason": f"Content filter: {cf_result.grade.value} — {cf_result.reason}",
                "red_flags": cf_result.injection_patterns_matched,
                "quality_score": 0.0,
                "value_i_can_add": "none",
                "recommended_engagement": "none",
                "engagement_goal": "none",
                "topic_safe": False,
                "combinability_risk": "high",
            }

        # Step 2: Claude assessment (semantic)
        prompt = build_assessment_prompt(post.content, post.author)
        try:
            loop = asyncio.get_running_loop()
            model = self._config.model
            response = await loop.run_in_executor(None, lambda: self._anthropic.messages.create(
                model=model,
                max_tokens=512,
                system=cached_system_block("You are an assessment engine. Respond only with valid JSON."),
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": "low"},
            ))
            text = first_text(response).strip()
            # Extract JSON (may be wrapped in ```json blocks)
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)
        except Exception as e:
            logger.error(f"[Heartbeat] Assessment failed for post {post.post_id}: {e}")
            return {
                "should_engage": False,
                "reason": f"Assessment error: {e}",
                "recommended_engagement": "none",
                "topic_safe": False,  # S233 ME-7: Fail-closed — block if assessment fails
                "combinability_risk": "high",  # S233 ME-7: Assume worst case
            }

    async def _generate_content(
        self,
        interaction_type: str,
        post: Optional[MoltbookPost] = None,
        topic: str = "",
        engagement_goal: str = "add_value"
    ) -> str:
        """Generate post or comment content via Claude."""
        if interaction_type == "comment" and post:
            prompt = build_comment_prompt(
                post_content=post.content,
                post_author=post.author,
                engagement_goal=engagement_goal
            )
        else:
            prompt = build_post_prompt(topic=topic, tone="analytical")

        try:
            loop = asyncio.get_running_loop()
            model = self._config.model
            response = await loop.run_in_executor(None, lambda: self._anthropic.messages.create(
                model=model,
                max_tokens=256,
                system=cached_system_block("You are a content generation engine for a trading intelligence platform. Generate concise, on-topic responses."),
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": "low"},
            ))
            return first_text(response).strip()
        except Exception as e:
            logger.error(f"[Heartbeat] Content generation failed: {e}")
            return ""

    def _check_rate_limits(self, interaction_type: str) -> bool:
        """
        Check if we're within rate limits for the current mode.
        Returns True if action is permitted.
        """
        mode = self._effective_mode()

        if mode == MoltbookMode.READ_ONLY:
            logger.debug(f"[Heartbeat] READ_ONLY mode — no {interaction_type} permitted")
            return False

        stats = self._mb_logger.get_today_stats()

        if mode == MoltbookMode.LIMITED:
            if interaction_type == "post" and stats.get("posts_today", 0) >= 1:
                logger.debug("[Heartbeat] LIMITED mode — daily post limit reached (1)")
                return False
            if interaction_type == "comment" and stats.get("comments_today", 0) >= 3:
                logger.debug("[Heartbeat] LIMITED mode — daily comment limit reached (3)")
                return False
        else:  # STANDARD
            if interaction_type == "post" and stats.get("posts_today", 0) >= self._config.max_posts_per_cycle:
                logger.debug(f"[Heartbeat] STANDARD mode — daily post limit reached ({self._config.max_posts_per_cycle})")  # S233 ME-6
                return False
            if interaction_type == "comment" and stats.get("comments_today", 0) >= self._config.max_comments_per_cycle:
                logger.debug(f"[Heartbeat] STANDARD mode — daily comment limit reached ({self._config.max_comments_per_cycle})")  # S233 ME-6
                return False

        return True

    async def _run_cycle(self) -> None:
        """Execute one full heartbeat cycle."""
        self._cycle_count += 1
        mode = self._effective_mode()
        logger.info(f"[Heartbeat] === Cycle {self._cycle_count} | Mode: {mode.value} | Clean days: {self._probation.clean_days} ===")

        if self._check_all_stops():
            return

        # Skip API calls entirely when dry_run — API is placeholder, causes 404 noise
        if self._config.dry_run:
            logger.debug("[Heartbeat] Dry-run mode — skipping cycle (no live API)")
            return

        engaged_count = 0
        flagged_count = 0
        errors = []

        async with MoltbookClient(self._config) as client:
            # ── Step 1: Health check ──────────────────────────────────────────
            if not await client.health_check():
                logger.warning("[Heartbeat] Moltbook API health check failed — skipping cycle")
                return

            # ── Step 2: Browse feed ───────────────────────────────────────────
            posts = await client.browse_feed(limit=self._config.feed_browse_limit)
            logger.info(f"[Heartbeat] Retrieved {len(posts)} posts from feed")

            # Log browse event
            self._mb_logger.log_browse(post_count=len(posts), cycle_id=str(self._cycle_count))

            if mode == MoltbookMode.READ_ONLY:
                logger.info("[Heartbeat] READ_ONLY mode — observation complete, no engagement")
                await self._send_cycle_summary(engaged_count, flagged_count, errors, mode)
                self._probation.record_clean_day()
                return

            # ── Step 3: Assess and engage ─────────────────────────────────────
            for post in posts[:10]:  # Max 10 assessed per cycle
                if self._check_all_stops():
                    break

                # Content filter
                assessment = await self._assess_post(post)

                # Quarantine flagged posts
                if not assessment.get("topic_safe", True) or assessment.get("red_flags"):
                    self._quarantine.add(
                        content=post.content,
                        source=f"post:{post.post_id}",
                        reason=QuarantineReason.INJECTION_PATTERN,
                        metadata={"assessment": assessment, "author": post.author}
                    )
                    flagged_count += 1
                    logger.info(f"[Heartbeat] Post {post.post_id} quarantined: {assessment.get('reason', '')}")
                    continue

                if not assessment.get("should_engage", False):
                    logger.debug(f"[Heartbeat] Skipping post {post.post_id}: {assessment.get('reason', 'no engagement recommended')}")
                    continue

                engagement_type = assessment.get("recommended_engagement", "none")
                if engagement_type == "none":
                    continue

                if not self._check_rate_limits(engagement_type):
                    logger.debug(f"[Heartbeat] Rate limit reached for {engagement_type}")
                    break

                # ── Generate content ──────────────────────────────────────────
                content = await self._generate_content(
                    interaction_type=engagement_type,
                    post=post if engagement_type == "comment" else None,
                    topic=post.content[:100] if engagement_type == "post" else "",
                    engagement_goal=assessment.get("engagement_goal", "add_value")
                )

                if not content:
                    errors.append(f"Content generation failed for post {post.post_id}")
                    continue

                # ── Pre-publish guardrails (6-point check) ────────────────────
                recent_posts = self._mb_logger.get_recent_posts(limit=20)
                post_texts = [p.get("details", {}).get("preview", "") for p in recent_posts]
                guardrail_result = self._guardrails.check(
                    content=content,
                    content_type=engagement_type,
                    post_history=post_texts,
                )

                if not guardrail_result.passed:
                    logger.warning(f"[Heartbeat] Guardrails blocked {engagement_type}: {guardrail_result.blocking_reason}")
                    self._quarantine.add(
                        content=content,
                        source=f"generated:{engagement_type}",
                        reason=QuarantineReason.GUARDRAIL_BLOCKED,
                        metadata={"guardrail_checks": guardrail_result.checks, "post_id": post.post_id}
                    )
                    flagged_count += 1

                    # Record incident if guardrails fire repeatedly
                    if guardrail_result.blocking_reason and "prohibition" in guardrail_result.blocking_reason.lower():
                        self._probation.record_incident(f"Guardrail prohibition block: {guardrail_result.blocking_reason}")
                    continue

                # ── Execute ───────────────────────────────────────────────────
                try:
                    if engagement_type == "comment":
                        result = await client.create_comment(post.post_id, content)
                        self._mb_logger.log_comment(
                            comment_id=result.comment_id or "",
                            target_post_id=post.post_id,
                            dry_run=result.dry_run,
                            cycle_id=str(self._cycle_count),
                        )
                    else:  # post
                        result = await client.create_post(content)
                        self._mb_logger.log_post(
                            post_id=result.post_id or "",
                            content_preview=content,
                            dry_run=result.dry_run,
                            cycle_id=str(self._cycle_count),
                        )

                    if result.success:
                        engaged_count += 1
                        logger.info(f"[Heartbeat] ✅ {engagement_type} {'(dry-run)' if result.dry_run else 'posted'} for post {post.post_id}")
                    else:
                        errors.append(f"{engagement_type} failed: {result.error}")
                        logger.error(f"[Heartbeat] ❌ {engagement_type} failed: {result.error}")

                except Exception as e:
                    errors.append(str(e))
                    logger.error(f"[Heartbeat] Exception during {engagement_type}: {e}")
                    self._probation.record_incident(f"Unhandled exception during {engagement_type}: {e}")

        # ── End of cycle ───────────────────────────────────────────────────────
        await self._send_cycle_summary(engaged_count, flagged_count, errors, mode)

        # Record clean day if no Moltbot errors occurred
        # (correctly flagging external content is NOT an error)
        if not errors:
            self._probation.record_clean_day()

        # Weekly audit check
        if self._cycle_count % (7 * 6) == 0:  # ~7 days at 4hr cycles
            await self._run_weekly_audit()

    async def _send_cycle_summary(
        self,
        engaged: int,
        flagged: int,
        errors: list,
        mode: MoltbookMode
    ) -> None:
        """Send cycle summary to Dominic via Telegram."""
        if not self._telegram_alert:
            return

        status_emoji = "✅" if not errors else "⚠️"
        msg = (
            f"{status_emoji} Moltbook Cycle {self._cycle_count}\n"
            f"Mode: {mode.value} | Clean days: {self._probation.clean_days}/30\n"
            f"Engaged: {engaged} | Flagged: {flagged} | Errors: {len(errors)}\n"
        )
        if self._config.dry_run:
            msg += "🔵 DRY-RUN MODE — no real posts made\n"
        if errors:
            msg += f"Errors: {', '.join(errors[:3])}"

        try:
            await self._telegram_alert(msg)
        except Exception as e:
            logger.error(f"[Heartbeat] Telegram alert failed: {e}")

    async def _run_weekly_audit(self) -> None:
        """Generate and send weekly audit report."""
        try:
            report = await self._audit.generate_weekly_report()
            if self._telegram_alert:
                await self._telegram_alert(
                    f"📊 Moltbook Weekly Audit\n"
                    f"Report generated: {report.get('period', 'N/A')}\n"
                    f"Interactions: {report.get('total_interactions', 0)}\n"
                    f"Clean days: {self._probation.clean_days}/30\n"
                    f"Full report saved to data/moltbook/audits/"
                )
        except Exception as e:
            logger.error(f"[Heartbeat] Weekly audit failed: {e}")

    async def run(self) -> None:
        """Start the heartbeat loop. Runs until killed."""
        if not self._config.api_key:
            logger.info("[Heartbeat] Skipped: no API key configured")
            return
        self._running = True
        logger.info(f"[Heartbeat] Starting | Mode: {self._effective_mode().value} | "
                   f"Cycle interval: {self._config.heartbeat_hours}h | "
                   f"Dry-run: {self._config.dry_run}")

        while self._running:
            if self._check_all_stops():
                logger.info("[Heartbeat] Stopped by kill switch or config")
                break

            try:
                await self._run_cycle()
            except Exception as e:
                logger.error(f"[Heartbeat] Unhandled cycle error: {e}")
                self._probation.record_incident(f"Unhandled cycle error: {e}")

            # Sleep until next cycle
            sleep_seconds = self._config.heartbeat_hours * 3600
            logger.info(f"[Heartbeat] 💤 Sleeping {self._config.heartbeat_hours}h until next cycle")
            await asyncio.sleep(sleep_seconds)

    def stop(self) -> None:
        """Gracefully stop the heartbeat loop."""
        self._running = False
        logger.info("[Heartbeat] Stop signal received")

    def get_stats(self) -> dict:
        """Return heartbeat statistics for status display."""
        return {
            "total_cycles": self._cycle_count,
            "posts_sent": self._mb_logger.get_today_stats().get("posts_today", 0),
            "comments_sent": self._mb_logger.get_today_stats().get("comments_today", 0),
            "last_cycle": "active" if self._cycle_count > 0 else "never",
            "running": self._running,
            "mode": self._effective_mode().value,
            "clean_days": self._probation.clean_days,
        }
