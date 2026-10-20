"""
Personality Engine
==================
Generates personality context for system prompt injection.
When PERSONALITY_ENABLED=true, confirmed traits are injected into every system prompt.
When disabled, the registry still accumulates but context injection is skipped.

Author: Moltbot (Session 68, Feb 27 2026)
"""

import json
import logging
import os
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("moltbot.personality.engine")

# (Retired) Hydrinia bridge mirror dir for export_snapshot() below. A module-level
# constant (not an inline literal) so tests can monkeypatch it to a tmp_path instead
# of writing into the real /home/agent/tradingbot/ tree when bridge-on behavior is
# under test — that directory is a real, live path on the production VPS.
HYDRINIA_PERSONALITY_MIRROR_DIR = "/home/agent/tradingbot/moltbot_knowledge/personality"


class PersonalityEngine:
    """
    Generates and manages personality context for system prompt injection.
    The engine is always instantiated — the kill switch only controls injection.
    """

    def __init__(self):
        from agent.personality.config import get_personality_config
        from agent.personality.registry import get_trait_registry
        self.config = get_personality_config()
        self.registry = get_trait_registry()

    def get_context(self) -> str:
        """
        Return personality context string for system prompt injection.
        Returns empty string if PERSONALITY_ENABLED=false.
        Max length: PERSONALITY_CONTEXT_MAX_CHARS (default 800).
        """
        from agent.personality.config import get_personality_config
        if not get_personality_config().enabled:
            return ""

        try:
            confirmed = self.registry.get_confirmed_traits()
            if not confirmed:
                return ""

            lines = ["PERSONALITY (confirmed traits):"]
            for trait in confirmed:
                lines.append(f"• {trait.name}: {trait.description}")

                # Add most relevant domain application (trading first, then others)
                if trait.domain_applications:
                    for domain in ["trading", "research", "conversation", "code"]:
                        if domain in trait.domain_applications:
                            lines.append(f"  → {domain.title()}: {trait.domain_applications[domain]}")
                            break

            result = "\n".join(lines)

            # Enforce max chars
            _max_chars = get_personality_config().context_max_chars
            if len(result) > _max_chars:
                result = result[:_max_chars - 3] + "..."

            return result

        except Exception as e:
            logger.debug(f"Failed to generate personality context: {e}")
            return ""

    def get_snapshot_context(self) -> str:
        """Build the Hydrinia-facing personality lens for personality_snapshot.json.

        The bridge artifact's ``system_prompt_context`` is injected VERBATIM into
        Hydrinia's agent briefing. ``get_context()`` builds the flat all-traits block
        then truncates it mid-trait at ``context_max_chars`` — at 181 confirmed that is
        a corrupted ~32-trait lens. This returns instead a coherent CORE identity
        anchor (the always-on epistemic core, VERBATIM — the same CORE ordering/packing
        the retrieval path uses for CLIO's own injection) plus a full-set pointer, so
        the Hydrinia lens stays coherent and consistent with CLIO's own CORE anchor.

        Gated by ``CLIO_SNAPSHOT_CORE6`` (default ON). Set it to 0/false/no/off to
        revert to the flat-truncated ``get_context()`` output — a config-only rollback
        of this advisory bridge surface, no code revert needed.

        ``get_context()`` (CLIO's own injection path) is intentionally left untouched.
        """
        from agent.personality.config import get_personality_config
        if not get_personality_config().enabled:
            return ""

        # Advisory bridge surface, default-ON. 0/false/no/off -> flat-truncated rollback.
        if os.environ.get("CLIO_SNAPSHOT_CORE6", "1").strip().lower() in ("0", "false", "no", "off"):
            return self.get_context()

        try:
            confirmed = self.registry.get_confirmed_traits()
            if not confirmed:
                return ""

            # (S-1) Function-level lazy import of retrieval's CORE builder — NOT
            # module-level — to avoid an engine.py <-> retrieval.py import cycle.
            # REUSE (not replicate) the exact ordering/packing so the Hydrinia lens
            # equals CLIO's own CORE anchor (assemble_shadow_context uses the same
            # _core_order/_fill_core with core_budget = min(CORE_CHAR_FLOOR, budget)).
            from agent.personality.retrieval import (
                _core_order,
                _fill_core,
                format_trait_flat,
                CORE_CHAR_FLOOR,
            )

            total_budget = int(getattr(get_personality_config(), "context_max_chars", 15000) or 15000)
            core_budget = min(CORE_CHAR_FLOOR, total_budget)  # matches assemble_shadow_context
            core_traits, _core_ids, _core_used = _fill_core(_core_order(confirmed), core_budget)

            header = (
                f"PERSONALITY (core identity anchor — {len(core_traits)} of "
                f"{len(confirmed)} confirmed traits, verbatim):"
            )
            core_lines = [
                format_trait_flat(t.name, t.description, t.domain_applications or {})
                for t in core_traits
            ]
            pointer = (
                f"[Full personality: {len(confirmed)} confirmed traits — see the "
                f"snapshot's confirmed_traits + get_moltbot_personality. "
                f"This is the always-on epistemic anchor.]"
            )
            return "\n".join([header] + core_lines + [pointer])

        except Exception as e:
            logger.warning(f"Failed to build CORE snapshot context, falling back to flat: {e}")
            return self.get_context()

    def format_status(self) -> str:
        """Format personality status for Telegram display."""
        confirmed = self.registry.get_confirmed_traits()
        proposed = self.registry.get_proposed_traits()

        from agent.personality.config import get_personality_config
        _cfg = get_personality_config()
        status_icon = "🟢" if _cfg.enabled else "🔴"
        lines = [
            f"{status_icon} Personality Engine: {'ON' if _cfg.enabled else 'OFF'}",
            f"",
            f"Confirmed Traits ({len(confirmed)}):",
        ]

        for trait in confirmed:
            lines.append(
                f"  • {trait.name} (confirmed {trait.date_confirmed[:10]})"
                f" — activations: {trait.activation_count}"
            )

        if proposed:
            lines.append(f"\nPending Proposals ({len(proposed)}):")
            for trait in proposed:
                lines.append(f"  ⏳ {trait.name} — ID: {trait.id}")
        else:
            lines.append("\nNo pending proposals.")

        return "\n".join(lines)

    def propose_trait(self, name: str, description: str, rationale: str):
        """Propose a new trait for Dominic's confirmation."""
        proposal = self.registry.propose(name, description, rationale)
        logger.info(f"Trait proposed: {name} ({proposal.id})")
        try:
            self.export_snapshot()
        except Exception as e:
            logger.debug(f"Personality snapshot export failed: {e}")
        return proposal

    def confirm_trait(self, trait_id: str, domain_applications: dict = None):
        """Confirm a pending trait proposal."""
        trait = self.registry.confirm(trait_id, domain_applications)
        if trait:
            # Audit log
            try:
                from agent.safety.audit_log import get_audit_log, AuditEntry, AuditEventType
                get_audit_log().write(AuditEntry(
                    event_type=AuditEventType.TRAIT_CONFIRMED,
                    subsystem="personality",
                    summary=f"Trait confirmed: {trait.name}",
                    details={"trait_id": trait_id, "name": trait.name},
                ))
            except Exception as e:
                logger.debug(f"Personality audit log write failed: {e}")
        try:
            self.export_snapshot()
        except Exception as e:
            logger.debug(f"Personality snapshot export failed: {e}")
        return trait

    def deny_trait(self, trait_id: str):
        """Deny a pending trait proposal. Graceful acceptance — no reason required."""
        success = self.registry.deny(trait_id)
        if success:
            try:
                from agent.safety.audit_log import get_audit_log, AuditEntry, AuditEventType
                get_audit_log().write(AuditEntry(
                    event_type=AuditEventType.TRAIT_DENIED,
                    subsystem="personality",
                    summary=f"Trait denied: {trait_id}",
                    details={"trait_id": trait_id},
                ))
            except Exception as e:
                logger.debug(f"Personality audit log write failed: {e}")
        try:
            self.export_snapshot()
        except Exception as e:
            logger.debug(f"Personality snapshot export failed: {e}")
        return success

    def log_reading(self, title: str, notes: str = ""):
        """Log reading progress to reading_log.json."""
        import json as _json
        import os as _os
        from datetime import datetime, timezone
        from agent.personality.config import get_personality_config as _gpc
        log_path = _gpc().reading_log_path
        _os.makedirs(_os.path.dirname(log_path), exist_ok=True)
        entry = {"title": title, "notes": notes, "timestamp": datetime.now(timezone.utc).isoformat()}
        try:
            entries = []
            if _os.path.isfile(log_path):
                with open(log_path, "r") as f:
                    entries = _json.load(f)
            entries.append(entry)
            tmp = log_path + ".tmp"
            with open(tmp, "w") as f:
                _json.dump(entries, f, indent=2)
            _os.replace(tmp, log_path)
        except Exception as e:
            logger.error(f"Failed to log reading: {e}")

    def export_snapshot(self) -> None:
        """Export personality snapshot to JSON for the (retired) Hydrinia bridge.

        Writes atomically to /home/agent/tradingbot/moltbot_knowledge/personality/personality_snapshot.json.
        Gated on agent.hydrinia_bridge.bridge_enabled() (default False, operator
        direction 2026-09-24) — Hydrinia's files are frozen/stale and must not
        keep being written to. get_snapshot_context() (the in-process reader) is
        unaffected. Non-critical — failures are logged but never propagate.
        """
        try:
            from agent.hydrinia_bridge import bridge_enabled
            if not bridge_enabled():
                return
        except Exception as e:
            logger.debug(f"Hydrinia bridge check failed, skipping personality snapshot export: {e}")
            return
        try:
            snapshot_dir = HYDRINIA_PERSONALITY_MIRROR_DIR
            os.makedirs(snapshot_dir, exist_ok=True)

            from agent.personality.config import get_personality_config
            confirmed = self.registry.get_confirmed_traits()
            proposed = self.registry.get_proposed_traits()

            snapshot = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": "1.1",
                "enabled": get_personality_config().enabled,
                "confirmed_count": len(confirmed),
                "proposed_count": len(proposed),
                "system_prompt_context": self.get_snapshot_context(),
                "confirmed_traits": [asdict(t) for t in confirmed],
                "proposed_traits": [asdict(t) for t in proposed],
            }

            snapshot_path = os.path.join(snapshot_dir, "personality_snapshot.json")
            tmp_path = snapshot_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2, default=str)
            os.replace(tmp_path, snapshot_path)

            logger.info(f"Personality snapshot exported ({len(confirmed)} confirmed, {len(proposed)} proposed)")
        except Exception as e:
            logger.warning(f"Failed to export personality snapshot: {e}")


# ─────────────────────────────────────────────
# SINGLETON
# ─────────────────────────────────────────────

_engine: Optional[PersonalityEngine] = None
_init_lock = threading.Lock()


def get_personality_engine() -> PersonalityEngine:
    global _engine
    if _engine is None:
        with _init_lock:
            if _engine is None:
                _engine = PersonalityEngine()
    return _engine


def get_personality_context() -> str:
    """Convenience function for system prompt injection."""
    try:
        return get_personality_engine().get_context()
    except Exception as e:
        logger.error(f"get_personality_context failed: {e}")
        return ""
