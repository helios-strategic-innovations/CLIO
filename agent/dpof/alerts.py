"""
CLIO DPOF — plain-text Telegram alerts (I/O).

Every alert is plain text (CLAUDE.md: no parse_mode with dynamic content) and
starts with a "[MODE]" tag so an operator scanning Telegram can tell at a
glance whether a message describes a LOG_ONLY simulation, a PAPER fill, or
real money. Sending is dedupe-gated per `dedupe_key` so a condition that
holds across many cycles (config invalid, gates failing, stale data) pages
once, not every cycle_seconds.

Lazy-imports agent.integrations.scheduled_tasks (the default sender) only
inside the function that needs it — this module must be importable with zero
CLIO integrations loaded (test isolation, startup ordering).
"""

from __future__ import annotations

import logging
import time
from typing import Awaitable, Callable, Optional

from agent.dpof.models import JevVerdict, PositionRecord, Signal

logger = logging.getLogger("clio.dpof.alerts")

_DEFAULT_DEDUPE_SECONDS = 86400
_TELEGRAM_CHUNK_CHARS = 3500
# Opportunistic bound on the in-memory dedupe map so a long-lived process
# (weeks of 30s cycles) never accumulates an unbounded number of keys —
# e.g. one "dpof_reconcile_reconciled_absent:<uuid>" key per historical
# position. Pruned lazily inside send(), never on a timer.
_DEDUPE_MAP_PRUNE_AT = 5000
_DEDUPE_MAP_PRUNE_AGE_SECONDS = 7 * 86400


async def _default_sender(bot, text: str) -> None:
    """Fan out to every configured chat id via CLIO's existing Telegram plumbing."""
    from agent.integrations import scheduled_tasks  # lazy — avoids importing telegram/CLIO at module load

    for chat_id in scheduled_tasks._get_chat_ids():
        await scheduled_tasks._safe_send(bot, chat_id, text=text[:_TELEGRAM_CHUNK_CHARS])


class DPOFAlerts:
    """Sends plain-text DPOF alerts with per-key dedupe.

    `sender(bot, text)` is injectable for tests. `clock` is injectable so
    dedupe windows are testable without real sleeps.
    """

    def __init__(
        self,
        sender: Optional[Callable[[object, str], Awaitable[None]]] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._sender = sender or _default_sender
        self._clock = clock
        self._last_sent: dict[str, float] = {}

    def _prune_if_large(self, now: float) -> None:
        if len(self._last_sent) <= _DEDUPE_MAP_PRUNE_AT:
            return
        cutoff = now - _DEDUPE_MAP_PRUNE_AGE_SECONDS
        self._last_sent = {k: v for k, v in self._last_sent.items() if v >= cutoff}

    async def send(
        self,
        bot,
        text: str,
        dedupe_key: Optional[str] = None,
        dedupe_seconds: int = _DEFAULT_DEDUPE_SECONDS,
    ) -> bool:
        """Send `text` via `bot`, suppressing repeats within `dedupe_seconds` of the
        same `dedupe_key`. `bot` None logs the alert at INFO (so it is never
        silently lost) and returns False without touching the dedupe map — a
        cycle running with no bot (e.g. a direct engine test) should not burn
        the dedupe window for whenever a real bot does show up.
        """
        now = self._clock()

        if dedupe_key is not None:
            last = self._last_sent.get(dedupe_key)
            if last is not None and (now - last) < dedupe_seconds:
                logger.debug(
                    "DPOF alert suppressed (dedupe_key=%s, %.0fs < %ds): %s",
                    dedupe_key, now - last, dedupe_seconds, text.splitlines()[0] if text else "",
                )
                return False

        if bot is None:
            logger.info("DPOF alert (no bot configured): %s", text)
            return False

        try:
            await self._sender(bot, text)
        except Exception as e:
            logger.error(f"DPOF alert send failed: {type(e).__name__}: {e}")
            return False

        if dedupe_key is not None:
            self._last_sent[dedupe_key] = now
            self._prune_if_large(now)
        return True


# ─────────────────────────────────────────────────────────────────────────
# Message formatting (pure)
# ─────────────────────────────────────────────────────────────────────────

def _mode_tag(mode: str) -> str:
    return f"[{(mode or 'unknown').upper()}]"


def fmt_entry(pos: PositionRecord, signal: Signal, verdict: JevVerdict, mode: str) -> str:
    """Fired right when an order is accepted (pending/open) or, in LOG_ONLY, when
    it would have been submitted (`pos.exit_reason == "log_only"`)."""
    tag = _mode_tag(mode)
    legs = ", ".join(str(leg.get("symbol", "?")) for leg in pos.legs) or "n/a"
    composite = f"{signal.composite:+.2f}" if signal.composite is not None else "n/a"

    jev_note = ""
    if verdict.called:
        jev_note = f" | jev={verdict.mode}"
        if verdict.veto_reasons:
            jev_note += f" notes=({','.join(verdict.veto_reasons)})"

    header = f"{tag} DPOF would enter" if pos.exit_reason == "log_only" else f"{tag} DPOF ENTRY SUBMITTED"

    return (
        f"{header}\n"
        f"{pos.underlying} {pos.direction} {pos.structure} x{pos.qty_original}\n"
        f"legs: {legs}\n"
        f"signal: composite={composite} strength={signal.strength} "
        f"conviction={signal.conviction:.2f}{jev_note}"
    )


def fmt_exit(pos: PositionRecord, reason: str, pnl: Optional[float]) -> str:
    """Fired on a full close (work_orders() 'exit_filled' with closed=True)."""
    tag = _mode_tag(pos.mode)
    pnl_str = f"${pnl:+.2f}" if pnl is not None else "n/a"
    return (
        f"{tag} DPOF EXIT ({reason or 'unknown'})\n"
        f"{pos.underlying} {pos.direction} {pos.structure} qty={pos.qty_original}\n"
        f"realized P&L: {pnl_str}"
    )


def fmt_gates(reasons: list[str]) -> str:
    return "[LOG_ONLY] DPOF gates failed — running LOG_ONLY: " + (", ".join(reasons) or "unknown")


def fmt_stale(minutes: float, issues: list[str]) -> str:
    return f"[DATA] DPOF data unhealthy for {minutes:.0f} min: " + (", ".join(issues) or "unknown")


def fmt_model_down(n_open: int, flatten_et: str) -> str:
    """Fired when both the primary and fallback decision providers are
    unavailable (agent.dpof.decider.DecisionUnavailable, or the decider
    itself hasn't landed yet). Open positions are HELD, never auto-closed —
    the one exception is the EOD flatten, which still fires regardless."""
    return (
        "DPOF: decision model unavailable (Opus 5.5 and GPT-6 Sol) — holding "
        f"{n_open} open position(s); no new entries. Positions will still be "
        f"flattened at {flatten_et} ET."
    )


def fmt_model_recovered() -> str:
    return "DPOF: decision model back online — resuming normal entries/reviews."
