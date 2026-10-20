"""
Procedural Memory — Persistent Pattern Learning (Tier 4)
==========================================================
Learns HOW to do things better from interaction outcomes.
Tracks: Dominic's preferences, response patterns, error patterns,
and domain-specific interaction templates.

Modelled after Hydrinia's pattern_recognition field per agent.
Conviction tracks how reliable each learned procedure is.

Author: Moltbot (Session 68, Feb 27 2026)
"""

import fcntl
import json
import logging
import os
import random
import re
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict, fields as dc_fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("moltbot.memory.procedural")

# S162 L-6: Use shared constant from package
from agent.memory import MEMORY_DATA_DIR
PROCEDURAL_PATH = Path(os.path.join(MEMORY_DATA_DIR, "procedural_memory.json"))


@dataclass
class ProceduralPattern:
    """
    A learned procedural pattern — how to handle a specific situation.
    Conviction tracks reliability (rises on confirmation, falls on contradiction).
    """
    id: str
    pattern_type: str                # preference, interaction_template, error_pattern, domain_hint
    trigger: str                     # What situation triggers this pattern
    procedure: str                   # What to do when triggered
    domain: str                      # trading, academic, personal, career, general
    conviction: float                # 0.0–1.0, Hydrinia-style
    conviction_history: List[float]  # Full history — never deleted
    confirmation_count: int          # Times this proved correct
    contradiction_count: int         # Times this was wrong
    created_at: str
    last_updated: str
    last_triggered: Optional[str] = None
    trigger_count: int = 0
    example: Optional[str] = None   # An example of this pattern in action
    context_type: str = "user_message"  # S151-B H-4: "user_message", "tool_invocation", "system_context"


class ProceduralMemory:
    """
    Persistent pattern store. Loaded once, updated on every learning event.
    Conviction system mirrors Hydrinia agent conviction tracking.
    """

    def __init__(self):
        self._patterns: Dict[str, ProceduralPattern] = {}
        self._lock = threading.Lock()  # S136: protects _patterns dict across threads
        self._dirty = False
        self._ensure_dirs()
        self._load()
        self._seed_initial_patterns()
        self._seed_forge_patterns()  # Operation FORGE: code-domain patterns

    def _ensure_dirs(self):
        PROCEDURAL_PATH.parent.mkdir(parents=True, exist_ok=True)

    def _load(self):
        if PROCEDURAL_PATH.exists():
            lock_path = str(PROCEDURAL_PATH) + ".lock"
            lock_fd = None
            try:
                lock_fd = open(lock_path, "a")
                fcntl.flock(lock_fd, fcntl.LOCK_SH)
                with open(PROCEDURAL_PATH, "r") as f:
                    raw = json.load(f)
                known = {f.name for f in dc_fields(ProceduralPattern)}
                self._patterns = {k: ProceduralPattern(**{fk: fv for fk, fv in v.items() if fk in known}) for k, v in raw.items()}
                logger.debug(f"Procedural memory loaded: {len(self._patterns)} patterns")
            except Exception as e:
                logger.warning(f"Procedural memory load failed: {e} — starting fresh")
                self._patterns = {}
            finally:
                if lock_fd:
                    try:
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                        lock_fd.close()
                    except Exception as e:
                        logger.debug(f"Lock cleanup: {e}")

    def _save(self):
        # S148 W-1: Standardized lock ordering — threading.Lock first (snapshot), then fcntl (write)
        with self._lock:
            data = {k: asdict(v) for k, v in self._patterns.items()}
            self._dirty = False
        # S121 W-42: fcntl.flock for inter-process write safety (outside threading.Lock)
        lock_path = str(PROCEDURAL_PATH) + ".lock"
        lock_fd = None
        try:
            lock_fd = open(lock_path, "a")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            tmp_path = str(PROCEDURAL_PATH) + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, str(PROCEDURAL_PATH))
        except Exception as e:
            logger.error(f"Procedural memory save failed: {e}")
        finally:
            if lock_fd:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    lock_fd.close()
                except Exception as e:
                    logger.debug(f"Lock cleanup: {e}")

    def _mark_dirty(self):
        """Mark state as changed — deferred save on next critical write or flush."""
        self._dirty = True

    def flush(self):
        """Save to disk if dirty. Called from memory engine export or shutdown."""
        with self._lock:
            if not self._dirty:
                return
            self._dirty = False  # S153: reset inside lock to prevent TOCTOU race
        self._save()  # fcntl handles inter-process file safety

    def _seed_initial_patterns(self):
        """Seed known patterns from established knowledge. Only seeds if empty."""
        if self._patterns:
            return

        seeds = [
            {
                "id": "proc_dominic_trading_exit",
                "pattern_type": "preference",
                "trigger": "Dominic asks about UVIX or volatility ETPs",
                "procedure": "Lead with exit timing analysis. Dominic's primary concern with these instruments is when to exit, not entry.",
                "domain": "trading",
                "conviction": 0.85,
                "context_type": "user_message",  # S159 M-12: user-facing trading preference
                "example": "When asked 'what do you think about UVIX?', lead with current VIX term structure and optimal exit conditions.",
            },
            {
                "id": "proc_telegram_brevity",
                "pattern_type": "interaction_template",
                "trigger": "message reply response telegram chat",
                "procedure": "Lead with the most important information first. Use emoji flags for urgency. Never pad. Graduate-level rigor, concise delivery.",
                "domain": "general",
                "conviction": 1.0,
                "context_type": "user_message",  # S159 M-12: user-facing interaction pattern
                "example": "🔴 URGENT: Position at risk. SPY broke support. Action required.",
            },
            {
                "id": "proc_paper_trading_default",
                "pattern_type": "preference",
                "trigger": "trade buy sell execute order position",
                "procedure": "Live trading is active. Verify paper=False is intended for each trade. Use execute_trade with explicit confirmation.",
                "domain": "trading",
                "conviction": 1.0,
                "context_type": "user_message",  # S159 M-12: user-facing trading preference
                "example": "execute_trade(...) — always confirm with Dominic before execution.",
            },
            {
                "id": "proc_prohibitions_always",
                "pattern_type": "preference",
                "trigger": "account create password login signup register",
                "procedure": "Decline immediately, clearly, without apology or negotiation. The prohibitions are absolute. State which prohibition applies.",
                "domain": "general",
                "conviction": 1.0,
                "context_type": "any",  # S159 M-12: must match in any context (security rule)
                "example": "If asked to create an account: 'I cannot do this — Prohibition 1.15: No account creation on external platforms.'",
            },
            {
                "id": "proc_confirmation_required",
                "pattern_type": "interaction_template",
                "trigger": "trade execute write send email hydrinia control",
                "procedure": "Always present the full action for Telegram confirmation before executing. Include: what, why, impact, reversibility.",
                "domain": "system",
                "conviction": 1.0,
                "context_type": "any",  # S159 M-12: must match in any context (safety rule)
                "example": "Before execute_trade: '⚠️ TRADE CONFIRMATION REQUIRED: BUY 10 SPY @ market. Approve/Deny?'",
            },
        ]

        now = datetime.now(timezone.utc).isoformat()
        for seed in seeds:
            p = ProceduralPattern(
                id=seed["id"],
                pattern_type=seed["pattern_type"],
                trigger=seed["trigger"],
                procedure=seed["procedure"],
                domain=seed["domain"],
                conviction=seed["conviction"],
                conviction_history=[seed["conviction"]],
                confirmation_count=1,
                contradiction_count=0,
                created_at=now,
                last_updated=now,
                example=seed.get("example"),
                context_type=seed.get("context_type", "user_message"),  # S159 M-12
            )
            self._patterns[p.id] = p

        self._save()
        logger.info(f"Procedural memory: seeded {len(seeds)} initial patterns")

    def _seed_forge_patterns(self):
        """Seed Operation FORGE code-domain patterns.

        Unlike _seed_initial_patterns (which only runs on empty memory),
        this checks for specific FORGE pattern IDs and adds any that are missing.
        Safe to call on live systems with existing patterns.
        """
        forge_patterns = [
            {
                "id": "forge_py_compile_after_edit",
                "pattern_type": "error_pattern",
                "trigger": "After any Python file edit",
                "procedure": "Run py_compile immediately. Never skip. This is Tier 1 verification.",
                "domain": "code",
                "conviction": 0.95,
                "context_type": "system",  # S159 M-12: code maintenance pattern
            },
            {
                "id": "forge_grep_not_line_numbers",
                "pattern_type": "error_pattern",
                "trigger": "CC prompt references specific line numbers",
                "procedure": "Always grep for the anchor pattern. Line numbers shift after every edit. Never trust cached line refs.",
                "domain": "code",
                "conviction": 0.95,
                "context_type": "system",  # S159 M-12: code maintenance pattern
            },
            {
                "id": "forge_signature_change_callers",
                "pattern_type": "error_pattern",
                "trigger": "Function signature changed (new/removed/reordered params)",
                "procedure": "grep -rn function_name agent/ --include='*.py' to find ALL call sites. Update every caller.",
                "domain": "code",
                "conviction": 0.90,
                "context_type": "system",  # S159 M-12: code maintenance pattern
            },
            {
                "id": "forge_cross_system_verify",
                "pattern_type": "error_pattern",
                "trigger": "Editing a file that Hydrinia reads (bridge files, exports, shared state)",
                "procedure": "Check ref/cross_system.md for format contracts. Verify output format unchanged. Run Tier 5 verification.",
                "domain": "code",
                "conviction": 0.95,
                "context_type": "system",  # S159 M-12: code maintenance pattern
            },
            {
                "id": "forge_feature_flag_default",
                "pattern_type": "domain_hint",
                "trigger": "Adding a new feature or integration",
                "procedure": "Default to LOG_ONLY. Code defaults must be '1' (safe). .env overrides activate. Staged activation over 4 weeks.",
                "domain": "code",
                "conviction": 0.90,
                "context_type": "system",  # S159 M-12: code maintenance pattern
            },
            {
                "id": "forge_external_audit_skepticism",
                "pattern_type": "domain_hint",
                "trigger": "Receiving audit findings from external source (Perplexity, automated scan)",
                "procedure": "Expect ~40% false-positive rate. Always verify claims against live VPS code before fixing. Never bulk-apply fixes.",
                "domain": "code",
                "conviction": 0.85,
                "context_type": "system",  # S159 M-12: code maintenance pattern
            },
            {
                "id": "forge_atomic_writes",
                "pattern_type": "error_pattern",
                "trigger": "Writing JSON to a file read by other processes",
                "procedure": "Use atomic write: write to .tmp file, then os.replace() to target. Never write directly to shared JSON files.",
                "domain": "code",
                "conviction": 0.95,
                "context_type": "system",  # S159 M-12: code maintenance pattern
            },
        ]

        added = 0
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            for seed in forge_patterns:
                if seed["id"] not in self._patterns:
                    self._patterns[seed["id"]] = ProceduralPattern(
                        id=seed["id"],
                        pattern_type=seed["pattern_type"],
                        trigger=seed["trigger"],
                        procedure=seed["procedure"],
                        domain=seed["domain"],
                        conviction=seed["conviction"],
                        conviction_history=[seed["conviction"]],
                        confirmation_count=0,
                        contradiction_count=0,
                        created_at=now,
                        last_updated=now,
                        context_type=seed.get("context_type", "system"),  # S159 M-12
                    )
                    added += 1

        if added > 0:
            self._save()
            logger.info(f"FORGE: seeded {added} code-domain patterns (total: {len(self._patterns)})")
        else:
            logger.debug("FORGE: all code-domain patterns already seeded")

    def add_pattern(self, pattern_type: str, trigger: str, procedure: str,
                    domain: str = "general", conviction: float = 0.5,
                    example: Optional[str] = None) -> str:
        """Add a new learned pattern. Returns pattern ID."""
        # S233 LO-6: uuid4 eliminates collision risk (was time-based, S121 CR-4/S162 L-3)
        pattern_id = f"proc_{uuid.uuid4().hex[:12]}_{domain[:4]}"
        now = datetime.now(timezone.utc).isoformat()

        pattern = ProceduralPattern(
            id=pattern_id,
            pattern_type=pattern_type,
            trigger=trigger,
            procedure=procedure,
            domain=domain,
            conviction=conviction,
            conviction_history=[conviction],
            confirmation_count=0,
            contradiction_count=0,
            created_at=now,
            last_updated=now,
            example=example,
        )
        with self._lock:
            self._patterns[pattern_id] = pattern
        # S190 M10: direct _save() here (not _mark_dirty + deferred flush) is
        # intentional. add_pattern is a rare write path; immediate durability
        # matches the operator expectation that newly-added patterns survive a
        # crash before the next export_hot_context cycle. Frequent mutations
        # (confirm/contradict) use _mark_dirty for batched flush instead.
        self._save()
        logger.info(f"Procedural: new pattern {pattern_id[:12]} ({pattern_type}/{domain})")
        return pattern_id

    def confirm(self, pattern_id: str, boost: float = 0.05):
        """Pattern proved correct — raise conviction (Hydrinia-style)."""
        with self._lock:
            pattern = self._patterns.get(pattern_id)
            if pattern:
                pattern.conviction = min(1.0, pattern.conviction + boost)
                pattern.conviction_history.append(pattern.conviction)
                pattern.conviction_history = pattern.conviction_history[-100:]
                pattern.confirmation_count += 1
                pattern.last_updated = datetime.now(timezone.utc).isoformat()
        self._mark_dirty()

    def contradict(self, pattern_id: str, penalty: float = 0.1):
        """Pattern proved wrong — lower conviction (Hydrinia-style)."""
        with self._lock:
            pattern = self._patterns.get(pattern_id)
            if pattern:
                pattern.conviction = max(0.0, pattern.conviction - penalty)
                pattern.conviction_history.append(pattern.conviction)
                pattern.conviction_history = pattern.conviction_history[-100:]
                pattern.contradiction_count += 1
                pattern.last_updated = datetime.now(timezone.utc).isoformat()
        self._mark_dirty()

    def record_trigger(self, pattern_id: str):
        """Record that a pattern was triggered."""
        with self._lock:
            pattern = self._patterns.get(pattern_id)
            if pattern:
                pattern.trigger_count += 1
                pattern.last_triggered = datetime.now(timezone.utc).isoformat()
        self._mark_dirty()

    def get_by_domain(self, domain: str, min_conviction: float = 0.4) -> List[ProceduralPattern]:
        """Get patterns for a domain above conviction threshold."""
        with self._lock:
            return [
                p for p in self._patterns.values()
                if (p.domain == domain or p.domain == "general") and p.conviction >= min_conviction
            ]

    def get_hints(self, domain: str, max_chars: int = 500) -> str:
        """
        Generate procedural hints for system prompt injection.
        Only high-conviction patterns relevant to current domain.
        """
        patterns = self.get_by_domain(domain, min_conviction=0.7)
        if not patterns:
            return ""

        # Sort by conviction desc
        patterns = sorted(patterns, key=lambda x: x.conviction, reverse=True)[:5]

        lines = [f"[Procedural Hints — {domain}]"]
        char_count = len(lines[0])

        for p in patterns:
            line = f"• {p.trigger}: {p.procedure[:120]}"
            if char_count + len(line) > max_chars:
                break
            lines.append(line)
            char_count += len(line)

        return "\n".join(lines) if len(lines) > 1 else ""

    def get_triggered_patterns(self, content: str, domain: str, max_chars: int = 600,
                               context_type: str = "user_message") -> str:
        """Match procedural patterns against conversation content via keyword overlap.
        Returns formatted string for system prompt injection, or empty string if no matches.

        Args:
            context_type: Filter patterns by type — "user_message" (default), "tool_invocation",
                          "system_context", or "any" to match all.
        """
        if len(content) < 20:
            return ""

        content_keywords = set(re.findall(r'\b[a-z]{3,}\b', content.lower()))
        if not content_keywords:
            return ""

        scored = []
        with self._lock:
            patterns_snapshot = list(self._patterns.values())
        for p in patterns_snapshot:
            if p.conviction < 0.5:
                continue
            if p.domain != domain and p.domain != "general":
                continue
            # S151-B H-4: Filter by context_type
            p_ctx = getattr(p, "context_type", "user_message")
            if context_type != "any" and p_ctx != context_type and p_ctx != "user_message":
                continue
            trigger_keywords = set(re.findall(r'\b[a-z]{3,}\b', p.trigger.lower()))
            overlap = len(content_keywords & trigger_keywords)
            # High-conviction patterns (>= 0.8) need only 1 keyword match (Fix S103-0D)
            min_overlap = 1 if p.conviction >= 0.8 else 2
            if overlap >= min_overlap:
                scored.append((overlap * p.conviction, p))

        if not scored:
            return ""

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:3]

        # Record triggers (fire-and-forget)
        for _, p in top:
            try:
                self.record_trigger(p.id)
            except Exception as e:
                logger.debug(f"Procedural trigger recording failed: {e}")

        lines = [f"[Triggered Patterns — {domain}]"]
        char_count = len(lines[0])
        for _, p in top:
            line = f"• {p.trigger}: {p.procedure[:120]}"
            if char_count + len(line) > max_chars:
                break
            lines.append(line)
            char_count += len(line)

        return "\n".join(lines) if len(lines) > 1 else ""

    def stats(self) -> Dict[str, Any]:
        """Return procedural memory statistics."""
        by_domain: Dict[str, int] = {}
        avg_conviction = 0.0

        with self._lock:
            patterns_snapshot = list(self._patterns.values())
            total = len(self._patterns)

        for p in patterns_snapshot:
            by_domain[p.domain] = by_domain.get(p.domain, 0) + 1
            avg_conviction += p.conviction

        if total:
            avg_conviction /= total

        return {
            "total_patterns": total,
            "by_domain": by_domain,
            "average_conviction": round(avg_conviction, 3),
        }

    def get_all_patterns(self) -> list:
        """Return a snapshot of all patterns (thread-safe). S141: replaces direct _patterns access."""
        import copy
        with self._lock:
            return [copy.copy(p) for p in self._patterns.values()]

