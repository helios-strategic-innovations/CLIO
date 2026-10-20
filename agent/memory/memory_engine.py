"""
Memory Engine — Unified Interface (Tier Coordinator)
Coordinates all 5 memory tiers into a single coherent system.

Tier 1: Working Memory   — hot, sliding window (working_memory.py)
Tier 2: Episodic Memory   — warm, append-only events (episodic.py)
Tier 3: Semantic Memory    — cold, structured knowledge (existing knowledge_tools.py)
Tier 4: Procedural Memory  — persistent, learned patterns (procedural.py)
Tier 5: Meta-Memory        — persistent, self-awareness (meta.py)

Provides:
- Unified context injection for system prompt
- Cross-tier search
- Learning from interaction outcomes
- Memory statistics dashboard
"""

import json
import logging
import os
import time
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .working_memory import WorkingMemory
from .episodic import EpisodicMemory
from .procedural import ProceduralMemory
from .meta import MetaMemory

logger = logging.getLogger("moltbot.memory.engine")


def _word_match(word: str, text: str) -> bool:
    """Word-boundary match — prevents 'wrong' matching 'wrongdoing'."""
    import re
    return bool(re.search(r'\b' + re.escape(word) + r'\b', text, re.IGNORECASE))

# Maximum characters for combined memory context injection
MAX_CONTEXT_CHARS = 3000

# S209 Phase 3/4 — cross-tier liveness thresholds (env-tunable; soak may retune).
_LIVENESS_EPISODIC_STALE_S = float(os.getenv("MEMORY_LIVENESS_EPISODIC_STALE_S", "86400"))      # 24h
_LIVENESS_EPISODIC_SIZE_ALARM_KB = float(os.getenv("MEMORY_EPISODIC_SIZE_ALARM_KB", "51200"))   # 50MB growth alarm

# Hydrinia is permanently off (operator direction, 2026-09-24) — the hot-context
# mirror write in export_hot_context() below is gated on this and logs its
# skip only once per process (called every response; avoid log spam).
_hydrinia_mirror_skip_logged = False

# Module-level constant (not an inline literal in export_hot_context() below) so
# tests can monkeypatch it to a tmp_path instead of writing into the real
# /home/agent/tradingbot/ tree when bridge-on behavior is under test — that
# directory is a real, live path on the production VPS.
HYDRINIA_HOT_CONTEXT_MIRROR_DIR = "/home/agent/tradingbot/moltbot_knowledge/working_memory"


class MemoryEngine:
    """
    Unified memory interface. Coordinates all tiers.
    Injected into system prompt via get_full_context().
    """

    def __init__(self):
        self.working = WorkingMemory()
        self.episodic = EpisodicMemory()
        self.procedural = ProceduralMemory()
        self.meta = MetaMemory()
        self._state_lock = threading.Lock()  # S136: protects _last_content, _tool_success_counts, _tool_success_date
        self._last_content: str = ""
        self._tool_success_counts: dict = {}  # (action, domain) -> count for auto-learning
        self._tool_success_date: str = ""  # Reset daily
        self._last_seal_milestone: int = self._load_seal_milestone()  # Cross-restart dedup
        self._last_working_probe: Optional[dict] = None  # S209: last liveness-probe result

        # Wire promote-before-purge: when working memory entries expire,
        # qualifying ones get recorded to episodic before deletion
        self.working._on_expire_callback = self._on_working_entry_expired
        self.working.purge_after_init()  # Now safe — callback wired before purge

        logger.info("Memory engine initialized — all 5 tiers active")

    @staticmethod
    def _load_seal_milestone() -> int:
        """Load last SEAL milestone from reward state to prevent duplicate entries on restart."""
        try:
            import json
            state_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "data", "knowledge", "seal_reward_state.json"
            )
            if os.path.isfile(state_path):
                with open(state_path) as f:
                    data = json.load(f)
                total = data.get("total_evaluated", 0)
                return (total // 10) * 10  # Last milestone boundary
        except Exception as e:
            logger.debug(f"SEAL milestone load failed: {e}")
        return 0

    def _on_working_entry_expired(self, entry) -> None:
        """Called when a qualifying working memory entry expires.
        Records it to episodic memory before it's deleted — the 'last will' pattern.
        Called OUTSIDE _state_lock. Thread-safe via fcntl in episodic.record().
        """
        self.episodic.record(
            event_type="promoted_on_expire",
            content=entry.content,
            category=entry.domain,
            domain=entry.domain,  # S153: populate domain field
            importance=entry.priority,
            metadata={
                "source": "working_memory_expiry",
                "access_count": entry.access_count,
                "original_id": entry.id,
            },
        )
        logger.debug(f"Promoted expiring entry {entry.id[:12]} to episodic "
                     f"(accesses={entry.access_count}, priority={entry.priority})")

    # === CONTEXT INJECTION ===

    def get_full_context(self, domain: Optional[str] = None, task_context: Optional[str] = None) -> str:
        """
        Generate unified memory context for system prompt injection.
        Budget allocation:
          - Working memory:   ~800 chars  (hot, most relevant)
          - Procedural hints: ~500 chars  (behavioral patterns)
          - Meta-awareness:   ~500 chars  (gaps, calibration, corrections)
          - Episodic summary: ~500 chars  (conversation memory first, then recent events)
          Total:             ~2300 chars
        
        Semantic memory (Tier 3) is injected separately by existing get_knowledge_context().
        """
        sections = []

        # Tier 1: Working memory — current session context
        working_ctx = self.working.get_context(max_chars=800)
        if working_ctx:
            sections.append(f"Active context:\n{working_ctx}")

        # Tier 4: Procedural hints — relevant patterns (content-matched if available)
        proc_ctx = self.procedural.get_hints(domain=domain or "general", max_chars=500)
        with self._state_lock:
            last_content = self._last_content
        if last_content:
            try:
                triggered = self.procedural.get_triggered_patterns(
                    content=last_content, domain=domain or "general"
                )
                if triggered:
                    proc_ctx = triggered
            except Exception as e:
                logger.debug("get_triggered_patterns failed: %s", e)
        if proc_ctx:
            sections.append(f"Learned patterns:\n{proc_ctx}")

        # Tier 5: Meta-awareness — brief gap/calibration note + corrections
        meta_ctx = self._get_meta_context(max_chars=500)
        if meta_ctx:
            sections.append(f"Self-awareness:\n{meta_ctx}")

        # Tier 2: Episodic — conversation memory first, then brief recent activity
        episodic_ctx = self._get_episodic_context(max_chars=500)
        if episodic_ctx:
            sections.append(f"Recent events:\n{episodic_ctx}")

        full = "\n\n".join(sections) if sections else ""
        # Hard cap (tier content only; the liveness marker is prepended AFTER so a
        # degradation alarm is never truncated away).
        if len(full) > MAX_CONTEXT_CHARS:
            full = full[:MAX_CONTEXT_CHARS - 3] + "..."

        # S209 Phase 2a — consumer-side liveness assertion. A dead/degraded tier
        # must NEVER render as a quiet omission (the S164 silent death). Prepend a
        # VISIBLE marker + log LOUD when a tier's write path is down. Cheap (no IO).
        degraded = self._memory_liveness_marker()
        if degraded:
            full = (degraded + "\n\n" + full) if full else degraded

        return full

    def _memory_liveness_marker(self) -> str:
        """Build a visible degradation marker (and log LOUD) for any memory tier
        whose write path is broken. Returns "" when healthy. Reads persist-health
        only (no IO, no synthetic write) — the deep timeout-probe runs periodically
        from export_hot_context(). Reflects the REAL write path, not a file-age
        proxy (Phase-4 'file-age != liveness')."""
        markers = []
        try:
            ph = self.working.persist_health()
            if ph.get("broken"):
                reason = ph.get("reason") or "unknown"
                logger.error("MEMORY LIVENESS: working tier persistence BROKEN — %s", reason)
                markers.append(f"- working memory: persistence UNAVAILABLE ({reason})")
        except Exception as e:
            logger.error("MEMORY LIVENESS: working persist_health raised: %s", e)
            markers.append("- working memory: liveness check FAILED")
        if not markers:
            return ""
        return (
            "⚠️ MEMORY DEGRADED — a tier's write path is down; recall below may be "
            "incomplete (operator: investigate):\n" + "\n".join(markers)
        )

    def _get_meta_context(self, max_chars: int = 500) -> str:
        """Generate meta-memory context snippet including recent corrections."""
        assessment = self.meta.get_self_assessment()
        lines = []

        cal = assessment.get("calibration", {})
        if cal.get("status") == "overconfident":
            lines.append("⚠️ Recent calibration: overconfident — lower stated confidence levels")
        elif cal.get("status") == "well_calibrated":
            lines.append("✅ Calibration: well-calibrated")

        gaps = assessment.get("unresolved_gaps", 0)
        if gaps > 0:
            top_domains = assessment.get("top_gap_domains", [])
            lines.append(f"Knowledge gaps: {gaps} unresolved (domains: {', '.join(top_domains[:3])})")

        # Surface last 3 corrections with actionable detail (not just topic)
        recent = assessment.get("recent_corrections", [])
        if recent:
            # Filter to corrections within last 14 days
            from datetime import datetime, timezone
            cutoff = datetime.now(timezone.utc).timestamp() - (14 * 86400)
            fresh = []
            for c in recent:
                try:
                    _ts_dt = datetime.fromisoformat(c.get("timestamp", ""))
                    if _ts_dt.tzinfo is None:
                        _ts_dt = _ts_dt.replace(tzinfo=timezone.utc)
                    ts = _ts_dt.timestamp()
                    if ts >= cutoff:
                        fresh.append(c)
                except Exception as e:
                    logger.debug("Correction timestamp parse failed: %s", e)
                    fresh.append(c)  # Keep if unparseable
            for c in fresh[-3:]:
                what_said = c.get("what_i_said", "")[:80]
                what_correct = c.get("what_was_correct", "")[:80]
                topic = c.get("topic", "?")
                if what_said and what_correct:
                    lines.append(f"⚠️ Correction ({topic}): I said '{what_said}' — correct: '{what_correct}'")
                else:
                    lines.append(f"⚠️ Correction: {topic} ({c.get('domain', '?')})")

        result = "\n".join(lines)
        return result[:max_chars] if result else ""

    def _get_episodic_context(self, max_chars: int = 500) -> str:
        """Generate episodic memory context snippet with conversation memory first."""
        memory_first = self.episodic.get_recent(
            limit=4,
            months_back=6,
            event_types=["memory", "promoted", "promoted_on_expire"],
        )
        recent_events = self.episodic.get_recent(limit=4)

        ordered = []
        seen = set()
        for entry in memory_first + recent_events:
            key = (
                entry.get("timestamp", ""),
                entry.get("event_type", ""),
                entry.get("content", ""),
            )
            if key in seen:
                continue
            seen.add(key)
            ordered.append(entry)
            if len(ordered) >= 6:
                break

        if not ordered:
            return ""

        lines = []
        for e in ordered:
            line = f"- {e.get('event_type', '?')}: {e.get('content', '?')[:120]}"
            lines.append(line)

        kept = []
        current_len = 0
        for line in lines:
            next_len = len(line) if not kept else current_len + 1 + len(line)
            if next_len > max_chars:
                break
            kept.append(line)
            current_len = next_len
        return "\n".join(kept)

    # === CROSS-TIER OPERATIONS ===

    def remember(self, key: str, content: Any, category: str = "general",
                 priority: float = 0.5, domain: str = "general",
                 source: str = "unknown", outcome: Optional[str] = None) -> None:
        """
        Primary entry point: remember something in working memory.
        High-priority items are also recorded episodically.

        Args:
            source: Where this memory came from (telegram_direct, twilio_voice, etc.)
            outcome: Optional outcome hint (success, failure, correction, None)
        """
        with self._state_lock:
            self._last_content = str(content)[:500]
        self.working.add(content=f"{key}: {str(content)[:250]}", domain=domain, priority=priority)

        # Detect corrections even at low priority (Fix S103-0C)
        _content_lower = str(content)[:500].lower()
        _is_correction = any(_word_match(w, _content_lower) for w in ("wrong", "incorrect", "that's not", "you're wrong", "actually no"))
        if _is_correction and outcome is None:
            outcome = "correction"

        # Increment access_count on BEST matching same-domain entry (Fix 3: working memory promotion)
        # Only the highest-priority unpromoted entry gets incremented to avoid trivially
        # reaching the promotion threshold by bulk-incrementing all entries.
        try:
            candidates = [e for e in self.working.get_by_domain(domain, limit=50) if not e.promoted]
            if candidates:
                best = max(candidates, key=lambda e: e.priority)
                with self.working._lock:
                    best.access()
                    self.working._dirty = True
        except Exception as e:
            logger.debug(f"Fire-and-forget access_count increment failed: {e}")

        # Record most conversations episodically (Fix S103-0B: lowered from 0.6 to 0.3)
        # Only noise (0.0) is excluded. Short acks (0.3) are borderline-included.
        if priority >= 0.3:
            # Derive outcome from content signals if not provided
            # NOTE: correction detection already handled above (lines 213-216), only check success here
            _content_str = str(content)[:500].lower()
            if outcome is None:
                if any(_word_match(w, _content_str) for w in ("thank", "great", "perfect", "excellent")):
                    outcome = "success"

            _metadata = {"source": source, "topic": domain}
            self.episodic.record(
                event_type="memory",
                content=f"{key}: {str(content)[:200]}",
                category=category,
                domain=category,  # S153: populate domain field
                importance=priority,
                outcome=outcome,
                metadata=_metadata,
            )

            # Correction: record in meta memory + resolve calibration (Fix S103-0C)
            if outcome == "correction":
                try:
                    self.meta.record_correction(
                        topic=domain,
                        domain=domain,
                        what_i_said=key,
                        what_was_correct=str(content)[:200],
                    )
                except Exception as e:
                    logger.debug("Correction meta recording failed: %s", e)
                try:
                    unresolved = self.meta.get_unresolved_calibration(domain)
                    if unresolved:
                        self.meta.record_outcome(unresolved[-1]["prediction_id"], False)
                except Exception as e:
                    logger.debug("Correction calibration update failed: %s", e)

        # Track domain query — pass content so hedging detection works (Fix E, Session 89)
        # Don't use priority as confident proxy — it's based on content type, not actual certainty.
        # Instead, let record_domain_query() detect hedging from the content itself.
        # Default to confident=True; hedging patterns in content will override to uncertain.
        self.meta.record_domain_query(
            domain=domain, confident=True,
            content=str(content)[:500],
        )

        # Check for working memory entries ready for promotion (Fix A, Session 89)
        self._check_promotions()

    def _check_promotions(self) -> None:
        """Promote working memory entries that have high access + priority to episodic.
        Runs after every remember() call. Only promotes entries not already promoted.
        (Fix A, Session 89)

        S209 Phase 3 — cascade seam: this promotion READS the working tier. When
        that SOURCE's write path is dead, a stale/empty read must DROP-AND-FLAG
        (loud), never silently yield zero promotions — the exact seam that silently
        staled procedural memory for ~12 weeks behind the S164 deadlock. Discipline:
        an INFRA error on the health read SKIPS (retryable); a DEAD upstream
        DROPS-AND-FLAGS.
        """
        try:
            ph = self.working.persist_health()
        except Exception as e:
            logger.warning("Promotion: working-source health check failed — skip pass (retryable): %s", e)
            return
        if ph.get("broken"):
            logger.error(
                "PROMOTION drop-and-flag: working-memory SOURCE unavailable (%s) — "
                "working→episodic promotion SUSPENDED this pass (NOT silently empty)",
                ph.get("reason"),
            )
            return
        try:
            promotable = self.working.get_promotable(min_access=5, min_priority=0.5)
            for entry in promotable:
                self.episodic.record(
                    event_type="promoted",
                    content=entry.content,
                    category=entry.domain,
                    domain=entry.domain,  # S153: populate domain field
                    importance=entry.priority,
                    metadata={"source": "working_memory_promotion", "access_count": entry.access_count},
                )
                self.working.promote(entry.id)
                logger.debug(f"Promoted working memory entry {entry.id[:12]} to episodic "
                             f"(accesses={entry.access_count}, priority={entry.priority})")
        except Exception as e:
            logger.warning("Fire-and-forget working memory promotion check failed: %s: %s", type(e).__name__, e, exc_info=True)

    def learn_from_outcome(self, action: str, outcome: str, domain: str = "general",
                           was_correct: bool = True, details: str = "") -> None:
        """
        Learn from an interaction outcome. Updates multiple tiers:
        - Episodic: records the event
        - Procedural: may create/update patterns
        - Meta: updates calibration
        """
        # Episodic record
        self.episodic.record(
            event_type="outcome",
            content=f"{action}: {details[:150]}",
            category=domain,
            domain=domain,  # S153: populate domain field
            outcome="success" if was_correct else "failure",
            importance=0.6 if was_correct else 0.8,  # Failures are more important to remember
        )

        # Meta calibration
        if not was_correct:
            self.meta.record_correction(
                topic=action,
                domain=domain,
                what_i_said=details[:200],
                what_was_correct=outcome,
            )

        # Track failure as uncertain domain query (Fix E, Session 89)
        # Tool failures = genuine uncertainty in that domain
        self.meta.record_domain_query(
            domain=domain, confident=was_correct,
            content=details[:500],
        )

        # Auto-learn procedural patterns from repeated tool success (Fix S103-0D)
        if was_correct:
            try:
                from datetime import date
                today = date.today().isoformat()
                with self._state_lock:
                    if self._tool_success_date != today:
                        self._tool_success_counts = {}
                        self._tool_success_date = today
                    key = (action, domain)
                    self._tool_success_counts[key] = self._tool_success_counts.get(key, 0) + 1
                    count = self._tool_success_counts[key]
                if count == 3 and self.procedural.stats().get("total_patterns", 0) < 50:
                    self.learn_pattern(
                        trigger=f"User asks about {domain} topics",
                        action=f"Tool '{action}' works well for {domain} tasks — use it proactively",
                        domain=domain,
                        confidence=0.6,
                    )
                    logger.debug(f"Auto-learned procedural pattern: {action} for {domain}")
            except Exception as e:
                logger.debug(f"Auto-learn pattern failed: {e}")

        # Link outcome to most recent unresolved calibration prediction in same domain
        try:
            unresolved = self.meta.get_unresolved_calibration(domain)
            if unresolved:
                self.meta.record_outcome(unresolved[-1]["prediction_id"], was_correct)
        except Exception as e:
            logger.debug("learn_from_outcome calibration failed: %s", e)

        logger.debug(f"Learned from outcome: {action} — {'success' if was_correct else 'failure'}")

    def learn_pattern(self, trigger: str, action: str, domain: str = "general",
                      confidence: float = 0.5) -> str:
        """Explicitly learn a procedural pattern."""
        return self.procedural.add_pattern(
            pattern_type="learned",
            trigger=trigger,
            procedure=action,
            domain=domain,
            conviction=confidence,
        )

    def record_gap(self, topic: str, domain: str, context: str = "") -> None:
        """Record a knowledge gap."""
        self.meta.record_gap(topic=topic, domain=domain, context=context)

    def clear_session_state(self) -> None:
        """Clear per-session state. Call at start of new conversation (Fix S103-2A).
        Prevents _last_content from leaking across conversation boundaries,
        which would trigger procedural patterns based on stale content.
        """
        with self._state_lock:
            self._last_content = ""

    # === SEAL SIGNAL INTEGRATION ===

    def integrate_seal_signals(self) -> dict:
        """Pull SEAL reward data into appropriate memory tiers.

        Called periodically (e.g., from hot context refresh) to sync SEAL's
        binary reward signals into Moltbot's 5-tier memory system.

        Returns stats dict with counts of what was integrated.
        """
        stats = {"meta_updated": False, "procedural_patterns": 0, "episodic_events": 0}

        try:
            from agent.integrations.seal_reward_engine import (
                get_reward_engine,
            )
            if os.environ.get("MOLTBOT_SEAL_REWARDS", "1") != "1":
                return stats

            engine = get_reward_engine()
            summary = engine.get_reward_summary()
            total_evaluated = summary.get("total_evaluated", 0)

            if total_evaluated == 0:
                return stats

            # --- TIER 5: META-MEMORY — Self-Awareness of Knowledge Quality ---
            win_rate = summary.get("win_rate", 0.5)
            try:
                # S151-B: Skip calibration for small samples (noisy signal)
                if total_evaluated < 5:
                    logger.debug(f"SEAL calibration skipped: only {total_evaluated} evaluations (min 5)")
                else:
                    # Stable ID: the write is upserted (replace_resolved=True below), so one
                    # row is kept. A per-timestamp hash here made every 6h write a new row
                    # (497 identical rows crowded the 500-row calibration cap).
                    pred_id = "seal_ti_quality"
                    # Throttle: max 1 calibration entry per 6 hours (check any seal_t* prefix)
                    _should_update = True
                    _seal_cal = self.meta.get_calibration_by_prefix("seal_t")
                    if _seal_cal:
                        last = _seal_cal[-1]
                        try:
                            last_ts = datetime.fromisoformat(last.get("timestamp", ""))
                            if last_ts.tzinfo is None:
                                last_ts = last_ts.replace(tzinfo=timezone.utc)
                            hours_since = (datetime.now(timezone.utc) - last_ts).total_seconds() / 3600
                            if hours_since < 6:
                                _should_update = False
                        except (ValueError, TypeError):
                            pass

                    if _should_update:
                        # S151-B: stated_confidence = reliability of estimate, not the rate itself
                        sample_confidence = min(total_evaluated / 50.0, 0.9)
                        self.meta.record_prediction(
                            prediction_id=pred_id,
                            domain="trade_ideas",
                            stated_confidence=round(sample_confidence, 2),
                            description=(
                                f"SEAL trade idea win rate: {win_rate:.0%} "
                                f"({summary.get('total_wins', 0)}/{total_evaluated})"
                            ),
                            replace_resolved=True,
                        )
                        self.meta.record_outcome(pred_id, win_rate >= 0.3)
                        stats["meta_updated"] = True
                        logger.info(
                            f"SEAL meta-memory updated: trade idea win rate {win_rate:.0%} "
                            f"({total_evaluated} evaluated)"
                        )
            except Exception as e:
                logger.debug(f"SEAL meta-memory integration failed: {e}")

            # --- TIER 4: PROCEDURAL MEMORY — Extract Learned Patterns ---
            # Only after sufficient sample size (20+ trades)
            if total_evaluated >= 20:
                try:
                    adjustments = summary.get("category_adjustments", {})
                    if adjustments:
                        # Build set of category names already in seal_learned patterns
                        # Extract from trigger: "...from 'catname' learning..."
                        import re as _re
                        _existing_seal_cats = set()
                        _patterns_snap = self.procedural.get_all_patterns()
                        for p in _patterns_snap:
                            if p.pattern_type == "seal_learned":
                                _m = _re.search(r"from '([^']+)'", p.trigger)
                                if _m:
                                    _existing_seal_cats.add(_m.group(1))

                        best_cat = max(adjustments, key=adjustments.get)
                        worst_cat = min(adjustments, key=adjustments.get)

                        if adjustments[best_cat] > 0.1 and best_cat not in _existing_seal_cats:
                            self.procedural.add_pattern(
                                pattern_type="seal_learned",
                                trigger=f"Generating trade ideas from '{best_cat}' learning category",
                                procedure=(
                                    f"'{best_cat}' category produces above-average trade ideas "
                                    f"(reward score: {adjustments[best_cat]:+.2f}). "
                                    f"Consider this domain in research."
                                ),
                                domain="trade_ideas",
                                conviction=min(0.85, 0.5 + adjustments[best_cat]),
                            )
                            _existing_seal_cats.add(best_cat)
                            stats["procedural_patterns"] += 1
                            logger.info(
                                f"SEAL procedural pattern: '{best_cat}' "
                                f"is best category (score {adjustments[best_cat]:+.2f})"
                            )
                            # S121 CR-4: Use distinct domain prefix to avoid add_pattern() ID collision
                            # (replaced time.sleep(0.002) which blocked the event loop)

                        if (adjustments[worst_cat] < -0.05
                                and worst_cat not in _existing_seal_cats):
                            self.procedural.add_pattern(
                                pattern_type="seal_learned",
                                trigger=f"Generating trade ideas from '{worst_cat}' learning category",
                                procedure=(
                                    f"'{worst_cat}' category has poor trade idea outcomes "
                                    f"(reward score: {adjustments[worst_cat]:+.2f}). "
                                    f"Apply extra scrutiny to ideas from this domain."
                                ),
                                domain="trade_ideas",
                                conviction=0.6,
                            )
                            stats["procedural_patterns"] += 1

                except Exception as e:
                    logger.debug(f"SEAL procedural pattern extraction failed: {e}")

            # --- TIER 2: EPISODIC MEMORY — Milestone Events ---
            try:
                recent_rewards = summary.get("recent_rewards", [])
                # Fire at every 10th evaluation, but only once per boundary
                _milestone_target = (total_evaluated // 10) * 10
                if (recent_rewards and _milestone_target >= 10
                        and _milestone_target > self._last_seal_milestone):
                    self._last_seal_milestone = _milestone_target
                    adjustments = summary.get("category_adjustments", {})
                    self.episodic.record(
                        event_type="seal_milestone",
                        content=(
                            f"SEAL evaluation milestone: {total_evaluated} trades evaluated, "
                            f"win rate {win_rate:.0%} "
                            f"({summary.get('total_wins', 0)}W/{summary.get('total_losses', 0)}L)"
                        ),
                        category="trade_ideas",
                        domain="trade_ideas",  # S153: populate domain field
                        importance=0.7,
                        metadata={
                            "total_evaluated": total_evaluated,
                            "win_rate": win_rate,
                            "category_adjustments": adjustments if total_evaluated >= 20 else {},
                        },
                    )
                    stats["episodic_events"] += 1
            except Exception as e:
                logger.warning("SEAL episodic milestone recording failed: %s: %s", type(e).__name__, e, exc_info=True)

        except ImportError:
            logger.debug("SEAL reward engine not available — skipping memory integration")
        except Exception as e:
            logger.warning(f"SEAL memory integration failed (non-fatal): {e}")

        return stats

    def _collect_liveness(self) -> dict:
        """S209 Phase 3/4 — structured cross-tier liveness/freshness snapshot for
        the Hydrinia mirror payload + LOUD logging. Cross-tier activity-conditioned:
        a tier stale while it SHOULD be writing is dead, not idle. Best-effort —
        never raises (this is observability, not a new failure surface)."""
        from agent.memory.liveness import age_seconds
        report = {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "working_probe": self._last_working_probe,
            "degraded": False,
            "alarms": [],
        }
        # Working tier — the AUTHORITATIVE death signal is persist-health (the real
        # write path), NOT activity-staleness: working writes are conversation-gated,
        # so 'idle' != 'dead' (file-age != liveness, Phase 4).
        try:
            ph = self.working.persist_health()
            report["working_persist"] = ph
            if ph.get("broken"):
                report["degraded"] = True
                report["alarms"].append(f"working:{ph.get('reason')}")
                logger.error("MEMORY LIVENESS: working tier BROKEN — %s", ph.get("reason"))
        except Exception as e:
            logger.warning("liveness: working persist_health failed: %s", e)
        # Episodic — autonomous cadence (15-min learning cycle + trade feedback), so
        # staleness IS meaningful: stale-while-running ≈ dead.
        try:
            recent = self.episodic.get_recent(limit=1)
            ep_age = age_seconds(recent[0].get("timestamp")) if recent else None
            report["episodic_age_s"] = ep_age
            if ep_age is not None and ep_age > _LIVENESS_EPISODIC_STALE_S:
                report["alarms"].append(f"episodic:stale {ep_age / 3600:.1f}h")
                logger.warning("MEMORY LIVENESS: episodic tier stale (%.1fh since last event)", ep_age / 3600)
            # Growth alarm — observability only. Pruning is operator-gated (episodic
            # is append-only AND Hydrinia-read; deleting it is a separate decision).
            est = self.episodic.stats()
            size_kb = est.get("total_size_kb", 0) or 0
            report["episodic_size_kb"] = size_kb
            if size_kb > _LIVENESS_EPISODIC_SIZE_ALARM_KB:
                report["alarms"].append(f"episodic:size {size_kb:.0f}KB")
                logger.warning(
                    "MEMORY GROWTH: episodic store %.0fKB > %.0fKB alarm threshold "
                    "(pruning policy is operator-gated)", size_kb, _LIVENESS_EPISODIC_SIZE_ALARM_KB,
                )
        except Exception as e:
            logger.warning("liveness: episodic age/size check failed: %s", e)
        return report

    # === SHUTDOWN / HYDRINIA BRIDGE EXPORTS ===

    def flush(self) -> None:
        """Flush dirty memory tiers and refresh the Hydrinia hot-context mirror."""
        self.export_hot_context()

    def export_hot_context(self) -> None:
        """Flush dirty memory tiers and export a hot-context snapshot to the
        (retired) Hydrinia mirror.

        The tier flush (working/meta/procedural) and liveness probe below are
        CLIO-internal and always run. Only the final write to
        /home/agent/tradingbot/moltbot_knowledge/working_memory/hot_context.json
        is gated on agent.hydrinia_bridge.bridge_enabled() (default False,
        operator direction 2026-09-24) — Hydrinia's files are frozen/stale and
        must not keep being written to.
        Entire method is fail-safe — errors are logged but never raised.
        """
        # Purge expired working memory entries before export (Fix S103-0E)
        try:
            self.working._purge_expired()
        except Exception as e:
            logger.debug("Pre-export purge failed: %s", e)

        # Flush deferred writes from dirty-flagged memory tiers
        for tier in (self.working, self.meta, self.procedural):
            tier_name = type(tier).__name__
            try:
                if hasattr(tier, "flush"):
                    tier.flush()
            except Exception as e:
                # S190 H5: WARNING (was DEBUG, invisible at default log level).
                # A flush failure means the tier may stay dirty across cycles,
                # eventually returning stale state. Operator needs to see this.
                logger.warning("Memory tier %s flush failed: %s", tier_name, str(e)[:200])
            # S190 H5: post-condition invariant check. If flush succeeded but the
            # dirty flag is still set, the tier's own _save logic has a bug we
            # need to investigate before the next export cycle.
            if hasattr(tier, "_dirty") and tier._dirty:
                logger.error("Memory tier %s remained dirty after flush attempt", tier_name)

        # S209 Phase 2b — periodic TIMEOUT-BOUNDED liveness probe of the working
        # tier's real write→read path. Logs LOUD on TIMEOUT/BROKEN. This is the
        # guard that would have caught the S164 deadlock the day it shipped.
        try:
            from agent.memory.liveness import probe_working_memory
            self._last_working_probe = probe_working_memory(self.working, timeout=2.0)
            if self._last_working_probe.get("status") != "LIVE":
                logger.error(
                    "MEMORY LIVENESS: working-memory periodic probe = %s (%s)",
                    self._last_working_probe.get("status"),
                    self._last_working_probe.get("detail"),
                )
        except Exception as e:
            logger.warning("Working-memory liveness probe failed to run: %s", e)

        global _hydrinia_mirror_skip_logged
        try:
            from agent.hydrinia_bridge import bridge_enabled
            _mirror_on = bridge_enabled()
        except Exception as e:
            logger.debug(f"Hydrinia bridge check failed, skipping mirror export: {e}")
            _mirror_on = False
        if not _mirror_on:
            if not _hydrinia_mirror_skip_logged:
                logger.info("Hot context mirror export skipped — Hydrinia bridge disabled")
                _hydrinia_mirror_skip_logged = True
            return

        try:
            out_dir = HYDRINIA_HOT_CONTEXT_MIRROR_DIR
            os.makedirs(out_dir, exist_ok=True)

            snapshot = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": "1.1",  # S209: +liveness block (additive; Hydrinia reads only full_context)
                "full_context": self.get_full_context(),  # already carries the Phase-2a degraded marker
                "working_memory_stats": self.working.stats() if self.working else {},
                # S209 Phase 3 — freshness marker for consumers. The MCP getter also
                # stamps _file_age_seconds on read (see hydrinia_mcp_server.safe_load_json),
                # so consumers get both structured liveness AND file-age freshness.
                "liveness": self._collect_liveness(),
            }

            out_path = os.path.join(out_dir, "hot_context.json")
            tmp_path = out_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(snapshot, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, out_path)
        except Exception as e:
            logger.warning(f"Hot context export failed: {e}")

    # === SEARCH ===

    def search_all(self, query: str, domain: Optional[str] = None, limit: int = 10) -> Dict:
        """Search across all memory tiers."""
        working_results = self.working.get_by_domain(domain, limit=limit) if domain else []
        results = {
            "working": [{"content": e.content, "domain": e.domain, "priority": e.priority} for e in working_results],
            "episodic": self.episodic.search(keyword=query, category=domain, limit=limit),
            "procedural": self.procedural.get_hints(domain=domain or "general"),
        }
        return results

    # === DASHBOARD ===

    def stats(self) -> Dict:
        """Full memory system statistics."""
        return {
            "working": self.working.stats(),
            "episodic": self.episodic.stats(),
            "procedural": self.procedural.stats(),
            "meta": self.meta.stats(),
        }

    def health_check(self) -> Dict:
        """Detailed health check across all tiers (Fix S103-2C).
        Returns per-tier health with 'healthy' flag for monitoring integration.
        """
        try:
            now = datetime.now(timezone.utc)
            working_stats = self.working.stats()
            working_total = working_stats.get("total_entries", 0)

            # Episodic freshness — check if any recent events exist
            episodic_stale = True
            try:
                recent = self.episodic.get_recent(limit=1)
                if recent:
                    last_ts = datetime.fromisoformat(recent[0].get("timestamp", ""))
                    if last_ts.tzinfo is None:
                        last_ts = last_ts.replace(tzinfo=timezone.utc)
                    episodic_stale = (now - last_ts).total_seconds() > 86400  # >24h
            except Exception as e:
                logger.debug(f"Health check episodic freshness failed: {e}")

            # Procedural trigger health (use locked snapshot)
            _proc_stats = self.procedural.stats()
            any_triggered = any(p.trigger_count > 0 for p in self.procedural.get_all_patterns())

            # Meta corrections
            _counts = self.meta.get_counts()
            corrections_count = _counts["corrections"]
            gaps_count = _counts["gaps"]

            return {
                "working": {"total": working_total, "healthy": True},
                "episodic": {"stale": episodic_stale, "healthy": not episodic_stale},
                "procedural": {"patterns": _proc_stats.get("total_patterns", 0), "any_triggered": any_triggered, "healthy": True},
                "meta": {"corrections": corrections_count, "gaps": gaps_count, "healthy": True},
                "overall_healthy": not episodic_stale,
            }
        except Exception as e:
            logger.error(f"Memory health check failed: {e}")
            return {"status": "error", "error": str(e), "overall_healthy": False}


# Singleton instance
_engine: Optional[MemoryEngine] = None
_engine_lock = threading.Lock()


def get_memory_engine() -> MemoryEngine:
    """Get or create the singleton MemoryEngine instance (thread-safe)."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = MemoryEngine()
    return _engine
