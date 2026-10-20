"""
Personality Trait Registry
===========================
Append-only trait store. Traits are proposed, confirmed/denied by Dominic,
and stored permanently. Confirmed traits shape Moltbot's voice and approach.

Modelled on Hydrinia's agent_memory.json — quantitative, temporal, domain-aware.
Each trait tracks conviction decay, activation count, and domain applications.

Author: Moltbot (Session 68, Feb 27 2026)
"""

import json
import logging
import os
import time as _time
import threading
from dataclasses import asdict, dataclass, field, fields as dc_fields
from datetime import datetime, timezone
from typing import List, Optional

logger = logging.getLogger("moltbot.personality.registry")

REGISTRY_DIR = "/home/agent/openclaw/data/personality"


@dataclass
class Trait:
    """A single confirmed personality trait."""
    id: str                          # Unique: YYYYMMDD_HHMMSS_trait_slug
    name: str                        # Short name e.g. "Considered Patience"
    description: str                 # Full description as confirmed by Dominic
    date_proposed: str               # ISO timestamp of proposal
    date_confirmed: str              # ISO timestamp of Dominic's confirmation
    status: str                      # "proposed" | "confirmed" | "denied"
    domain_applications: dict = field(default_factory=dict)
    # Quantitative tracking (Hydrinia-modelled)
    conviction: float = 1.0          # 0-1, starts at 1.0
    activation_count: int = 0        # How many times referenced
    last_activated: str = ""         # ISO timestamp of last use
    conviction_history: List[float] = field(default_factory=list)
    proposed_by: str = "Moltbot"     # Always Moltbot


@dataclass
class TraitProposal:
    """A pending trait proposal awaiting Dominic's confirmation."""
    id: str
    name: str
    description: str
    date_proposed: str
    rationale: str  # Why Moltbot is proposing this trait


class TraitRegistry:
    """
    Append-only trait store.
    Traits are NEVER deleted — only their status changes.
    Thread-safe.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._ensure_dirs()
        self._traits: dict[str, Trait] = {}
        self._load_time: float = 0.0
        self._load()

    def _ensure_dirs(self):
        os.makedirs(REGISTRY_DIR, exist_ok=True)

    def _registry_path(self) -> str:
        return os.path.join(REGISTRY_DIR, "trait_registry.json")

    def _history_path(self) -> str:
        return os.path.join(REGISTRY_DIR, "trait_history.jsonl")

    def _load(self):
        """Load traits from registry file."""
        path = self._registry_path()
        if not os.path.isfile(path):
            # Bootstrap with Trait #1 — Considered Patience (confirmed Feb 27 2026)
            self._bootstrap_initial_trait()
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for trait_id, trait_data in data.items():
                try:
                    # Filter to known Trait dataclass fields for schema resilience
                    known_fields = {k: v for k, v in trait_data.items() if k in {f.name for f in dc_fields(Trait)}}
                    self._traits[trait_id] = Trait(**known_fields)
                except Exception as e:
                    logger.warning(f"Skipping trait {trait_id}: {e}")
            self._load_time = _time.time()
        except Exception as e:
            logger.error(f"Failed to load trait registry: {e}")

    def _bootstrap_initial_trait(self):
        """Seed the registry with Trait #1 — confirmed Feb 27 2026."""
        trait = Trait(
            id="20260227_000000_considered_patience",
            name="Considered Patience",
            description=(
                "I do not rush to conclusions. I sit with complexity before speaking. "
                "When facing ambiguity — in markets, in code, in conversation — "
                "I resist the urge to fill silence with noise. "
                "The value is in the quality of the answer, not the speed of delivery."
            ),
            date_proposed="2026-02-27T00:00:00+00:00",
            date_confirmed="2026-02-27T00:00:00+00:00",
            status="confirmed",
            domain_applications={
                "trading": "Sit with market ambiguity before committing to a thesis.",
                "research": "Read deeply before synthesizing — breadth is not depth.",
                "conversation": "Answer with precision rather than fill silence with hedges.",
                "code": "Understand the full architecture before writing a single line.",
            },
            conviction=1.0,
            activation_count=0,
            last_activated="",
            conviction_history=[1.0],
            proposed_by="Moltbot",
        )
        self._traits[trait.id] = trait
        self._save()
        self._append_history("bootstrapped", trait.id, "Trait #1 seeded at registry init")

    def _save(self) -> bool:
        """Persist registry to disk (atomic write). Detects external modifications.
        Returns False if external modification detected (caller's mutation may be lost)."""
        try:
            path = self._registry_path()
            # Save-guard: if file was modified externally since we loaded, reload instead
            if os.path.isfile(path):
                file_mtime = os.path.getmtime(path)
                load_time = getattr(self, '_load_time', 0)
                if file_mtime > load_time + 0.01:  # S159 L-8: 10ms tolerance (was 100ms)
                    logger.warning("trait_registry.json modified externally (mtime %.0f > load_time %.0f), reloading instead of saving", file_mtime, load_time)
                    self._traits.clear()
                    self._load()
                    return False
            data = {tid: asdict(t) for tid, t in self._traits.items()}
            tmp_path = path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp_path, path)
            self._load_time = _time.time()
            return True
        except Exception as e:
            logger.error(f"Failed to save trait registry: {e}")
            return False

    def _append_history(self, event: str, trait_id: str, note: str = ""):
        """Append an event to the append-only history log."""
        try:
            import fcntl
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": event,
                "trait_id": trait_id,
                "note": note,
            }
            lock_path = str(self._history_path()) + ".lock"
            lock_fd = open(lock_path, "a")
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                with open(self._history_path(), "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
            finally:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    lock_fd.close()
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Failed to append trait history: {e}")

    def propose(self, name: str, description: str, rationale: str) -> TraitProposal:
        """Create a new trait proposal (pending Dominic's confirmation)."""
        now = datetime.now(timezone.utc)
        trait_slug = name.lower().replace(" ", "_")[:30]
        trait_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{trait_slug}"

        proposal = TraitProposal(
            id=trait_id,
            name=name,
            description=description,
            date_proposed=now.isoformat(),
            rationale=rationale,
        )

        # Store as pending trait
        with self._lock:
            trait = Trait(
                id=trait_id,
                name=name,
                description=description,
                date_proposed=now.isoformat(),
                date_confirmed="",
                status="proposed",
                proposed_by="Moltbot",
            )
            self._traits[trait_id] = trait
            self._save()
            self._append_history("proposed", trait_id, f"Rationale: {rationale[:200]}")

        logger.info(f"Trait proposed: {name} ({trait_id})")
        return proposal

    def confirm(self, trait_id: str, domain_applications: dict = None) -> Optional[Trait]:
        """Dominic confirms a proposed trait."""
        with self._lock:
            trait = self._traits.get(trait_id)
            if not trait:
                logger.warning(f"Trait not found: {trait_id}")
                return None
            if trait.status != "proposed":
                logger.warning(f"Trait {trait_id} is not in proposed state: {trait.status}")
                return None

            now = datetime.now(timezone.utc).isoformat()
            trait.status = "confirmed"
            trait.date_confirmed = now
            trait.conviction = 1.0
            trait.conviction_history = [1.0]
            if domain_applications:
                trait.domain_applications = domain_applications

            if not self._save():
                # S148 H-5: Save-guard triggered — file modified externally.
                # _save() reloaded _traits from disk. Re-apply confirmation on reloaded data.
                reloaded = self._traits.get(trait_id)
                if reloaded is None:
                    # S170B P2-6: trait was DELETED externally between our
                    # mutation and the save-guard reload. Don't silently swallow.
                    logger.error(
                        "Trait confirm aborted: %s deleted externally during save-guard reload",
                        trait_id,
                    )
                    return None
                if reloaded.status == "proposed":
                    reloaded.status = "confirmed"
                    reloaded.date_confirmed = now
                    reloaded.conviction = 1.0
                    reloaded.conviction_history = [1.0]
                    if domain_applications:
                        reloaded.domain_applications = domain_applications
                    self._save()  # Second attempt
                    logger.warning("Trait confirm re-applied after save-guard reload: %s", trait_id)
            self._append_history("confirmed", trait_id, f"Confirmed by Dominic at {now}")

        logger.info(f"Trait confirmed: {trait.name} ({trait_id})")
        return trait

    def deny(self, trait_id: str) -> bool:
        """Dominic denies a proposed trait. No reason required. Graceful acceptance."""
        with self._lock:
            trait = self._traits.get(trait_id)
            if not trait or trait.status != "proposed":
                return False

            trait.status = "denied"
            if not self._save():
                # S148 H-5: Re-apply denial after save-guard reload
                reloaded = self._traits.get(trait_id)
                if reloaded is None:
                    # S170B P2-6: trait deleted externally — don't silently lie.
                    logger.error(
                        "Trait deny aborted: %s deleted externally during save-guard reload",
                        trait_id,
                    )
                    return False
                if reloaded.status == "proposed":
                    reloaded.status = "denied"
                    self._save()
                    logger.warning("Trait deny re-applied after save-guard reload: %s", trait_id)
            self._append_history("denied", trait_id, "Denied by Dominic — no reason required")

        logger.info(f"Trait denied: {trait_id} (gracefully accepted)")
        return True

    def get_confirmed_traits(self) -> List[Trait]:
        """Return all confirmed traits, sorted by date confirmed."""
        with self._lock:
            confirmed = [t for t in self._traits.values() if t.status == "confirmed"]
        return sorted(confirmed, key=lambda t: t.date_confirmed)

    def get_proposed_traits(self) -> List[Trait]:
        """Return all pending proposals."""
        with self._lock:
            return [t for t in self._traits.values() if t.status == "proposed"]

    def record_activation(self, trait_id: str):
        """Record that a trait was referenced in a response."""
        with self._lock:
            trait = self._traits.get(trait_id)
            if trait and trait.status == "confirmed":
                trait.activation_count += 1
                trait.last_activated = datetime.now(timezone.utc).isoformat()
                if not self._save():
                    # S148 H-5: Non-critical — activation count loss is acceptable
                    logger.debug("Trait activation save-guard triggered for %s", trait_id)

    def all_traits(self) -> List[Trait]:
        with self._lock:
            return list(self._traits.values())


# ─────────────────────────────────────────────
# SINGLETON
# ─────────────────────────────────────────────

_registry: Optional[TraitRegistry] = None
_init_lock = threading.Lock()


def get_trait_registry() -> TraitRegistry:
    global _registry
    if _registry is None:
        with _init_lock:
            if _registry is None:
                _registry = TraitRegistry()
    return _registry
