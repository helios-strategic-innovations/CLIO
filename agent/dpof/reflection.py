"""
CLIO DPOF — learning loop F1: post-trade lessons (reflection).

Task ORCH-7Q4K-DPOF builder B2, 2026-09-26. Design frame:
docs/design/clio_dpof_learning_loop.md §4/§6/§7 (also see CLAUDE.md's
"DPOF / Repo Gotchas" + "Intel v3 Gotchas" sections).

DPOF records outcomes (positions.json, decisions.jsonl) but the decider never
sees WHY a past trade worked or failed -- only a five-item {label, pnl,
exit_reason, held_minutes} recent-closed list. This module is the missing
feedback edge: after every closed trade, the decision model writes a short
lesson from CODE-COMPUTED outcome metrics (never the model's own guess at
P&L/timing); recent lessons plus a code-computed tally then enter future
entry/review prompts as CONTEXT — never a rule, a gate, or a vote. Every new
LLM call here fails open (feature silently absent, logged at INFO); no sync
file I/O or flock on the event loop; every piece of external/model-written
text reaching a prompt goes through ONE `sanitize_ingested_content(...,
source=...)` envelope.

Contents:
  - `build_trade_record` / `render_trade_record` — pure position -> dict ->
    sanitized text (§4.2).
  - `is_eligible` — pure position-only eligibility gate (§4.3). Attempt-count/
    backoff/has-lesson checks live in `LessonStore` state and are applied by
    `ReflectionService.run_pending` (they need the store, not just the
    position).
  - `REFLECTION_SYSTEM_PROMPT` / `REFLECTION_SCHEMA` / `validate_reflection` —
    the reflection call's static system prompt, JSON schema (§4.4), and
    lenient clamp/enum-fallback validator (mirrors `agent.dpof.decider`'s
    `_clamp01`/`_validate_entry` style, but never marks the whole response a
    failure over one bad field — this call is advisory-only, unlike the
    decider's own fail-closed-on-malformed-JSON contract).
  - `LessonStore` — owns `lessons.jsonl` + `reflection_state.json` under
    `data_dir`, mirroring `agent.dpof.signal_log.SignalLog`'s locking/atomic-
    write/line-cap conventions (sidecar `.dpof_lessons.lock`; JSONL append +
    cap-and-trim; `.tmp` + fsync + `os.replace` for the state file). Loaded
    once into an in-memory cache; every new lesson updates the cache in
    place, so `ReflectionService.lessons_ctx` never touches disk.
  - `ReflectionService` — the async orchestrator: `run_pending` (status-driven
    discovery + single-flight + the actual reflection calls) and
    `lessons_ctx` (pure over the cache — same/cross/tally for one underlying).
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import re
import time
from datetime import date, datetime, time as dt_time, timezone
from typing import Any, Awaitable, Callable, Optional
from zoneinfo import ZoneInfo

from agent.dpof.models import Direction, PositionRecord

logger = logging.getLogger("clio.dpof.reflection")

ET = ZoneInfo("America/New_York")

_LESSONS_FILE = "lessons.jsonl"
_STATE_FILE = "reflection_state.json"
_LOCK_FILE = ".dpof_lessons.lock"

DEFAULT_MAX_LINES = 5000
_TRIM_KEEP_FRACTION = 0.9   # trim to 90% of the cap, matching signal_log.py

_REVIEW_TRAIL_CAP = 6
_REVIEW_TEXT_CAP = 200
_LESSON_TEXT_CAP = 600
_ELLIPSIS = "…"
_MIN_SENTENCE_CUT_FRACTION = 0.4   # a sentence cut must keep >= 40% of the cap


def _cut_at_sentence(text: str, cap: int) -> str:
    """Return `text` unchanged when it fits `cap`; otherwise cut it to at
    most `cap` characters without ending mid-word. Prefers the last sentence
    boundary ('. ' or '; ') in the window (the sentence ends cleanly, no
    ellipsis) -- but only when that boundary keeps at least
    _MIN_SENTENCE_CUT_FRACTION of the cap, so a lone early sentence cannot
    throw the rest of the lesson away; else the last space, with an ellipsis
    appended; else a hard cut with an ellipsis (a single unbroken token).
    Never raises."""
    if len(text) <= cap:
        return text
    window = text[: cap - len(_ELLIPSIS)]
    sentence_end = max(window.rfind(". "), window.rfind("; "))
    if sentence_end >= int(cap * _MIN_SENTENCE_CUT_FRACTION):
        return window[: sentence_end + 1]
    space = window.rfind(" ")
    if space > 0:
        return window[:space].rstrip() + _ELLIPSIS
    return window + _ELLIPSIS

_BACKOFF_SECONDS_PER_ATTEMPT = 600.0   # 10 minutes x attempt number

# Excluded exit reasons (§4.3): a position that never really traded, or whose
# broker-side status is itself unreliable, has nothing to teach.
_EXCLUDED_EXIT_REASONS = ("log_only", "submit_unconfirmed", "reconciled_absent")
_EXCLUDED_EXIT_REASON_PREFIXES = ("submit_failed:",)

KEY_FACTORS: tuple[str, ...] = (
    "flow_signal", "price_confirmation", "timing", "theta", "event_risk",
    "vol_pricing", "sizing", "exit_discipline", "regime", "other",
)
_ENTRY_QUALITIES = ("good", "mixed", "poor")
_EXIT_QUALITIES = ("good", "mixed", "poor", "forced")


# ─────────────────────────────────────────────────────────────────────────
# Small pure helpers (self-contained on purpose — module-boundary discipline
# matches agent.dpof.exits/decider each keeping their own tiny copies rather
# than cross-importing private helpers)
# ─────────────────────────────────────────────────────────────────────────

def _compact_json(val: Any) -> str:
    try:
        return json.dumps(val, default=str, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(val)


def _clamp01(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if v != v or v in (float("inf"), float("-inf")):   # NaN/inf guard w/o math import
        return 0.0
    return max(0.0, min(1.0, v))


_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _strip_control_chars(text: str) -> str:
    return _CONTROL_CHAR_RE.sub("", text)


def _default_now_et_fn(epoch: float) -> datetime:
    return datetime.fromtimestamp(epoch, tz=ET)


def _flatten_hhmm_for_date(d: date) -> str:
    """Best-effort flatten time (HH:MM ET) for the session `d` falls on —
    agent.dpof.config's flatten_et, clamped to close-10min on an
    agent.integrations.market_events early-close date, else the hardcoded
    "15:50" fallback (config helpers unavailable is fine; this is
    observability, never a gate)."""
    flatten_et = "15:50"
    try:
        from agent.dpof.config import load_dpof_config  # lazy

        flatten_et = load_dpof_config().flatten_et or flatten_et
    except Exception as e:
        logger.debug(f"DPOF reflection: load_dpof_config unavailable for flatten_et: {type(e).__name__}: {e}")

    try:
        from agent.integrations.market_events import MARKET_EARLY_CLOSE  # lazy

        if d in MARKET_EARLY_CLOSE:
            close_min = 13 * 60 - 10
            hh, mm = divmod(close_min, 60)
            return f"{hh:02d}:{mm:02d}"
    except Exception as e:
        logger.debug(f"DPOF reflection: MARKET_EARLY_CLOSE unavailable: {type(e).__name__}: {e}")

    return flatten_et


def _parse_hhmm(s: str) -> Optional[dt_time]:
    try:
        hh, mm = s.split(":")
        return dt_time(hour=int(hh), minute=int(mm))
    except (ValueError, AttributeError, TypeError):
        return None


def minutes_to_flatten_at_entry(
    entry_time: Optional[float], now_et_fn: Callable[[float], datetime] = _default_now_et_fn,
) -> Optional[float]:
    """Minutes from `entry_time` to that session's flatten time. None if
    `entry_time` is missing or the flatten time can't be parsed. Never raises."""
    if entry_time is None:
        return None
    try:
        entry_dt = now_et_fn(entry_time)
    except Exception as e:
        logger.debug(f"DPOF reflection: now_et_fn failed: {type(e).__name__}: {e}")
        return None
    t = _parse_hhmm(_flatten_hhmm_for_date(entry_dt.date()))
    if t is None:
        return None
    flatten_dt = entry_dt.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
    return max(0.0, (flatten_dt - entry_dt).total_seconds() / 60.0)


def _exit_category(exit_reason: str) -> str:
    if exit_reason in ("decider_exit", "decider_trim", "eod_flatten"):
        return exit_reason
    return "other"


def _cap_reviews(reviews: Optional[list[dict]]) -> list[dict]:
    out: list[dict] = []
    for r in (reviews or [])[-_REVIEW_TRAIL_CAP:]:
        if not isinstance(r, dict):
            continue
        out.append({
            "action": r.get("action"),
            "reason": str(r.get("reason") or "")[:_REVIEW_TEXT_CAP],
            "thesis_status": r.get("thesis_status"),
        })
    return out


# ─────────────────────────────────────────────────────────────────────────
# §4.2 — trade record (pure)
# ─────────────────────────────────────────────────────────────────────────

def build_trade_record(
    pos: PositionRecord,
    reviews: list[dict],
    *,
    now_et_fn: Callable[[float], datetime] = _default_now_et_fn,
) -> dict:
    """Pure. Turns one CLOSED position + its review trail into a compact dict
    of CODE-COMPUTED outcome metrics for the reflection call to interpret —
    the model never computes P&L/timing itself. `now_et_fn` converts an epoch
    (seconds) into an ET datetime; injectable for tests, default
    `datetime.fromtimestamp(x, tz=ET)`. `reviews` is whatever the caller
    already read (e.g. decisions.jsonl kind="position_review" rows for this
    position_id) — each element is treated defensively (`.get(...)` with
    fallbacks), since the exact set of keys on a decision_log row may grow
    over time.
    """
    entry_time = pos.entry_time
    closed_time = pos.closed_time

    entry_time_et = None
    closed_at = None
    held_minutes = None
    try:
        if entry_time is not None:
            entry_time_et = now_et_fn(entry_time).isoformat()
        if closed_time is not None:
            closed_at = now_et_fn(closed_time).isoformat()
        if entry_time is not None and closed_time is not None:
            held_minutes = max(0.0, (closed_time - entry_time) / 60.0)
    except Exception as e:
        logger.debug(f"DPOF reflection: time formatting failed for {pos.id}: {type(e).__name__}: {e}")

    pnl_usd = pos.realized_pnl
    pnl_pct = None
    premium_paid = None
    if pos.entry_price is not None and pos.entry_price > 0 and pos.qty_original > 0:
        premium_paid = pos.entry_price * 100.0 * pos.qty_original
        if premium_paid > 0:
            pnl_pct = pnl_usd / premium_paid

    underlying_move_bps = None
    entry_u = pos.entry_underlying_price
    exit_u = pos.exit_underlying_price
    if entry_u is not None and entry_u > 0 and exit_u is not None:
        # Levels are passed through so the reader can check the sign of
        # underlying_move_bps against the actual direction of the move.
        try:
            sign = Direction(pos.direction).sign
        except ValueError:
            sign = 0
        if sign != 0:
            underlying_move_bps = sign * (exit_u - entry_u) / entry_u * 10_000.0

    entry_signal = pos.entry_signal if isinstance(pos.entry_signal, dict) else {}
    entry_decision = pos.entry_decision if isinstance(pos.entry_decision, dict) else {}

    return {
        "position_id": pos.id,
        "underlying": pos.underlying,
        "direction": pos.direction,
        "structure": pos.structure,
        "qty": pos.qty_original,
        "entry_time_et": entry_time_et,
        "closed_at": closed_at,
        "held_minutes": held_minutes,
        "minutes_to_flatten_at_entry": minutes_to_flatten_at_entry(entry_time, now_et_fn),
        "pnl_usd": pnl_usd,
        "pnl_pct": pnl_pct,
        "underlying_move_bps": underlying_move_bps,
        "underlying_entry": entry_u if entry_u is not None and entry_u > 0 else None,
        "underlying_exit": exit_u,
        "obs_max_pnl_pct": pos.obs_max_pnl_pct,
        "obs_min_pnl_pct": pos.obs_min_pnl_pct,
        "exit_reason": pos.exit_reason,
        "exit_category": _exit_category(pos.exit_reason or ""),
        "entry_composite": entry_signal.get("composite"),
        "entry_strength": entry_signal.get("strength"),
        "entry_regime": entry_signal.get("regime"),
        "entry_pillars": entry_signal.get("pillars_used") or {},
        "entry_conviction": entry_decision.get("conviction"),
        "entry_qty_decided": entry_decision.get("qty"),
        "entry_provider": entry_decision.get("provider", ""),
        "thesis": pos.thesis or "",
        "invalidation": pos.invalidation or "",
        "review_count": pos.review_count,
        "reviews": _cap_reviews(reviews),
    }


_TRADE_RECORD_KEY_ORDER = (
    "position_id", "underlying", "direction", "structure", "qty",
    "entry_time_et", "closed_at", "held_minutes", "minutes_to_flatten_at_entry",
    "pnl_usd", "pnl_pct", "underlying_move_bps", "underlying_entry", "underlying_exit",
    "obs_max_pnl_pct", "obs_min_pnl_pct",
    "exit_reason", "exit_category", "entry_composite", "entry_strength", "entry_regime",
    "entry_pillars", "entry_conviction", "entry_qty_decided", "entry_provider",
    "thesis", "invalidation", "review_count", "reviews",
)


def render_trade_record(record: dict) -> str:
    """Compact label:value text for one trade record, wrapped in ONE
    `sanitize_ingested_content(..., source="dpof_trade_record")` envelope —
    the thesis/invalidation/review reasons are model-written text that can
    echo headlines (CLAUDE.md: every external/model-written field reaching a
    prompt is sanitized). Manual `<ingested_data>` fallback if the import
    fails (CLAUDE.md anti-pattern #16 — sanitization fallback must not
    bypass sanitization)."""
    lines = []
    for key in _TRADE_RECORD_KEY_ORDER:
        if key in record:
            lines.append(f"{key}: {_compact_json(record[key])}")
    text = "\n".join(lines)
    try:
        from agent.security import sanitize_ingested_content  # lazy import

        return sanitize_ingested_content(text, source="dpof_trade_record")
    except Exception as e:
        logger.debug(f"sanitize_ingested_content unavailable; wrapping manually: {type(e).__name__}: {e}")
        safe = text.replace("<", "&lt;").replace(">", "&gt;")
        return f'<ingested_data source="dpof_trade_record">\n{safe}\n</ingested_data>'


# ─────────────────────────────────────────────────────────────────────────
# §4.3 — eligibility (pure, position-only)
# ─────────────────────────────────────────────────────────────────────────

def is_eligible(
    pos: PositionRecord,
    now: float,
    *,
    lookback_days: float,
    excluded_exit_reasons: tuple[str, ...] = _EXCLUDED_EXIT_REASONS,
) -> bool:
    """Pure, POSITION-ONLY eligibility (§4.3): closed + actually filled +
    within the lookback window + not one of the excluded exit reasons.

    Deliberately does NOT check has_lesson / attempts / backoff — those live
    in `LessonStore` state (per position_id), not on the position itself, and
    are applied by `ReflectionService.run_pending` (which has the store).
    """
    if pos.status != "closed":
        return False
    if not pos.qty_original or pos.qty_original <= 0:
        return False
    if pos.entry_price is None:
        return False
    if pos.closed_time is None:
        return False
    if (now - pos.closed_time) > max(0.0, lookback_days) * 86400.0:
        return False
    reason = pos.exit_reason or ""
    if reason in excluded_exit_reasons:
        return False
    if any(reason.startswith(p) for p in _EXCLUDED_EXIT_REASON_PREFIXES):
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────
# §4.4 — the reflection call: system prompt, schema, validation
# ─────────────────────────────────────────────────────────────────────────

REFLECTION_SYSTEM_PROMPT = """You are Buddy, CLIO's DPOF decision layer, reviewing one of your OWN closed paper trades in SPY/QQQ long options for Dominic Newman -- never someone else's trade, never a hypothetical.

The trade record below is entirely CODE-COMPUTED: the flow signal and regime at entry, why the trade was entered, how it was reviewed while open (if at all), the underlying's actual move, how long it was held versus the session's flatten time, and how it closed (decider exit/trim, or the 15:50 ET forced flatten). obs_max_pnl_pct/obs_min_pnl_pct are the position's own observed profit/loss extremes while it was open (best-effort -- they may reset if the process restarted mid-position, so treat them as a rough sense of how much the position moved against or in favor of you, not an exact peak).

underlying_move_bps is signed in the position's favour: positive means the underlying moved in the direction that helps the position (for a long put, positive means the underlying fell). Use the given underlying_entry and underlying_exit levels for direction.

State what the evidence from THIS ONE TRADE actually says -- never a generic rule ("always cut losers fast", "never hold into the close") and never advice phrased as a command. This is ONE closed trade: a small sample on its own. Phrase the lesson as an observation about this specific case, framed so a future reader can weigh it against however many similar trades have gone before drawing any real conclusion from it.

Respond with ONLY the JSON object matching the schema you were given for this call. No prose, no markdown, nothing outside the JSON object."""


REFLECTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "lesson": {
            "type": "string",
            "description": "What the evidence from THIS trade says -- an observation, not a rule or a command. At most 600 characters.",
        },
        "entry_quality": {"type": "string", "enum": list(_ENTRY_QUALITIES)},
        "exit_quality": {"type": "string", "enum": list(_EXIT_QUALITIES)},
        "key_factor": {
            "type": "string",
            "enum": list(KEY_FACTORS),
            "description": "The single factor that most explains this trade's outcome.",
        },
        "confidence": {
            "type": "number",
            "description": "0.0 to 1.0 -- how confident you are in this specific lesson, given this is one trade.",
        },
    },
    "required": ["lesson", "entry_quality", "exit_quality", "key_factor", "confidence"],
    "additionalProperties": False,
}


def validate_reflection(parsed: Any) -> dict:
    """Lenient clamp/enum-fallback validator (never fails the whole response
    over one bad field -- this call is advisory-only context, unlike the
    decider's own fail-closed-on-malformed-JSON contract in
    agent.dpof.decider). Never raises."""
    if not isinstance(parsed, dict):
        parsed = {}

    lesson = _cut_at_sentence(_strip_control_chars(str(parsed.get("lesson") or "")).strip(), _LESSON_TEXT_CAP)

    entry_quality = parsed.get("entry_quality")
    if entry_quality not in _ENTRY_QUALITIES:
        entry_quality = "mixed"

    exit_quality = parsed.get("exit_quality")
    if exit_quality not in _EXIT_QUALITIES:
        exit_quality = "mixed"

    key_factor = parsed.get("key_factor")
    if key_factor not in KEY_FACTORS:
        key_factor = "other"

    confidence = _clamp01(parsed.get("confidence"))

    return {
        "lesson": lesson,
        "entry_quality": entry_quality,
        "exit_quality": exit_quality,
        "key_factor": key_factor,
        "confidence": confidence,
    }


# ─────────────────────────────────────────────────────────────────────────
# §4.5 — LessonStore (I/O)
# ─────────────────────────────────────────────────────────────────────────

class LessonStore:
    """Owns `lessons.jsonl` + `reflection_state.json` under `data_dir`.

    Mirrors `agent.dpof.signal_log.SignalLog`'s locking/atomic-write/line-cap
    conventions: ONE sidecar lock (`.dpof_lessons.lock`) guards both files;
    every write happens inside a single `run_in_executor` call holding an
    exclusive `fcntl.flock` for its whole duration (CLAUDE.md #19/#22 — never
    flock + os.replace on the SAME locked file; the lock file is its own
    sidecar, never the file being replaced).

    Single writer per data dir (one DPOF engine per process; readers such as
    readmodel / Mission Control never write): the in-memory line counter that
    drives the cap/trim assumes no other process appends to lessons.jsonl.

    `load()` reads the full lessons.jsonl (capped at `max_lines`, so this is
    bounded) into an in-memory cache once, off-loop. Every subsequent
    `add_lesson()` updates that cache in the same call, so
    `ReflectionService.lessons_ctx` (and the Telegram/EOD digest readers in
    readmodel.py) never need to touch disk.
    """

    def __init__(self, data_dir: str, *, max_lines: int = DEFAULT_MAX_LINES) -> None:
        self.data_dir = data_dir
        self.max_lines = max(1, int(max_lines))
        self._cache: list[dict] = []
        self._lesson_position_ids: set = set()
        self._state: dict = {}
        self._loaded = False
        # Line count of lessons.jsonl -- counted once (on load/first write) and
        # kept in memory after that (mirrors agent.dpof.signal_log.SignalLog's
        # `_line_count`), so a write doesn't re-read an up-to-cap file every call.
        self._line_count: Optional[int] = None

    # ------------------------------------------------------------------
    # Locking / executor plumbing
    # ------------------------------------------------------------------

    def _lock_path(self) -> str:
        return os.path.join(self.data_dir, _LOCK_FILE)

    def _with_lock(self, fn: Callable[[], Any]) -> Any:
        os.makedirs(self.data_dir, exist_ok=True)
        with open(self._lock_path(), "a") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                return fn()
            finally:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

    async def _run_locked(self, fn: Callable[[], Any]) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self._with_lock(fn))

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    async def load(self) -> None:
        """(Re)loads the in-memory cache + state from disk. Safe to call
        again after a restart (a fresh instance reproduces the prior
        instance's cache exactly, given the same on-disk files)."""

        def _do() -> tuple[list[dict], dict]:
            return self._read_lessons(), self._read_state()

        lessons, state = await self._run_locked(_do)
        self._cache = lessons
        self._lesson_position_ids = {r.get("position_id") for r in lessons if r.get("position_id")}
        self._state = state if isinstance(state, dict) else {}
        self._line_count = self._count_lines(self._lessons_path())
        self._loaded = True

    @staticmethod
    def _count_lines(path: str) -> int:
        try:
            with open(path, "rb") as f:
                return sum(1 for _ in f)
        except FileNotFoundError:
            return 0
        except OSError as e:
            logger.error(f"DPOF reflection: failed to count lines in {path}: {type(e).__name__}: {e}")
            return 0

    def _lessons_path(self) -> str:
        return os.path.join(self.data_dir, _LESSONS_FILE)

    def _state_path(self) -> str:
        return os.path.join(self.data_dir, _STATE_FILE)

    def _read_lessons(self) -> list[dict]:
        path = self._lessons_path()
        if not os.path.exists(path):
            return []
        out: list[dict] = []
        try:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(rec, dict):
                        out.append(rec)
        except OSError as e:
            logger.error(f"DPOF reflection: failed to read {path}: {type(e).__name__}: {e}")
            return []
        return out

    def _read_state(self) -> dict:
        path = self._state_path()
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r") as f:
                raw = f.read()
            if not raw.strip():
                return {}
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError) as e:
            logger.error(f"DPOF reflection: corrupt {path}: {type(e).__name__}: {e}; starting fresh")
            return {}

    # ------------------------------------------------------------------
    # Reads (cache-only — no I/O)
    # ------------------------------------------------------------------

    def has_lesson(self, position_id: str) -> bool:
        return position_id in self._lesson_position_ids

    def get_state(self, position_id: str) -> dict:
        st = self._state.get(position_id)
        if not isinstance(st, dict):
            return {"attempts": 0, "next_attempt_at": None}
        return {"attempts": int(st.get("attempts", 0) or 0), "next_attempt_at": st.get("next_attempt_at")}

    def recent(self, n: int) -> list[dict]:
        """Newest-first, up to `n` lesson records from the in-memory cache."""
        if n <= 0:
            return []
        return list(reversed(self._cache))[:n]

    @property
    def cache(self) -> list[dict]:
        return self._cache

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    async def add_lesson(self, record: dict) -> None:
        """Appends `record` to lessons.jsonl (cap + trim-to-90%), updates the
        in-memory cache, and clears any reflection_state.json entry for its
        position_id (a stored lesson means no more retries are needed)."""
        position_id = record.get("position_id")

        def _do() -> None:
            self._append_lesson_line(record)
            if position_id and position_id in self._state:
                new_state = dict(self._state)
                new_state.pop(position_id, None)
                self._atomic_write_json(self._state_path(), new_state)
                self._state = new_state

        # Index BEFORE awaiting the write: a CancelledError landing on this
        # await (shutdown / mode OFF) does not stop the executor write, so the
        # in-memory index must already say "done" or the next scan re-reflects
        # the same trade (duplicate lesson). A write that genuinely fails is
        # rolled back and re-raised.
        self._cache.append(record)
        if position_id:
            self._lesson_position_ids.add(position_id)
        try:
            await self._run_locked(_do)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._cache = [r for r in self._cache if r is not record]
            if position_id and not any(r.get("position_id") == position_id for r in self._cache):
                self._lesson_position_ids.discard(position_id)
            raise

    def _append_lesson_line(self, record: dict) -> None:
        """Appends one line and keeps `self._line_count` in sync (mirrors
        agent.dpof.signal_log.SignalLog.write_cycle's in-memory counter
        pattern) -- an append never re-reads the whole file just to check the
        cap; only a trim (rare, once every ~10% of `max_lines` writes) does."""
        path = self._lessons_path()
        os.makedirs(self.data_dir, exist_ok=True)
        line = json.dumps(record, default=str)
        with open(path, "a") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        if self._line_count is None:
            self._line_count = self._count_lines(path)
        else:
            self._line_count += 1
        if self._line_count > self.max_lines:
            kept = self._trim_if_needed(path)
            if kept is not None:
                self._line_count = kept

    def _trim_if_needed(self, path: str) -> Optional[int]:
        """Keeps the newest ~90% of `max_lines` once the file crosses the cap.
        Returns the resulting line count, or None if the file couldn't be
        read (in which case `self._line_count` is left as the caller's
        best-effort in-memory value, same as SignalLog does)."""
        try:
            with open(path, "r") as f:
                lines = f.readlines()
        except OSError as e:
            logger.error(f"DPOF reflection: failed to read {path} for trim check: {e}")
            return None
        if len(lines) <= self.max_lines:
            return len(lines)
        keep = max(1, int(self.max_lines * _TRIM_KEEP_FRACTION))
        kept = lines[-keep:]
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w") as f:
            f.writelines(kept)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        logger.info(f"DPOF reflection: trimmed {path} from {len(lines)} to {len(kept)} lines")
        return len(kept)

    async def record_attempt(self, position_id: str, error: str, now: float) -> None:
        """Bumps the attempt counter for `position_id` and sets its next
        retry time to `now + 10min * (new attempt count)` (§4.3's backoff).
        Persisted atomically (.tmp + fsync + os.replace)."""
        if not position_id:
            return

        def _do() -> dict:
            state = self._read_state()
            cur = state.get(position_id) if isinstance(state.get(position_id), dict) else {}
            attempts = int(cur.get("attempts", 0) or 0) + 1
            new_entry = {
                "attempts": attempts,
                "next_attempt_at": now + _BACKOFF_SECONDS_PER_ATTEMPT * attempts,
                "last_error": str(error or "")[:300],
                "last_attempt_at": now,
            }
            new_state = dict(state)
            new_state[position_id] = new_entry
            self._atomic_write_json(self._state_path(), new_state)
            return new_state

        new_state = await self._run_locked(_do)
        self._state = new_state

    @staticmethod
    def _atomic_write_json(path: str, data: dict) -> None:
        text = json.dumps(data, default=str, indent=2)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)


# ─────────────────────────────────────────────────────────────────────────
# §4.3/§4.6 — ReflectionService (async orchestrator)
# ─────────────────────────────────────────────────────────────────────────

def _win_loss_bucket(pnl_usd: Any) -> Optional[bool]:
    """True = win, False = loss, None = unusable (never a fake 0)."""
    if isinstance(pnl_usd, bool) or not isinstance(pnl_usd, (int, float)):
        return None
    return pnl_usd > 0


def _format_lesson_alert(record: dict) -> str:
    """§4.7 plain-text format: 'DPOF lesson — SPY bullish −$120 (−38%, 42 min): <lesson>'."""
    underlying = record.get("underlying") or "?"
    direction = record.get("direction") or "?"
    pnl_usd = record.get("pnl_usd")
    pnl_pct = record.get("pnl_pct")
    held = record.get("held_minutes")

    if isinstance(pnl_usd, (int, float)) and not isinstance(pnl_usd, bool):
        sign = "+" if pnl_usd >= 0 else "-"
        pnl_str = f"{sign}${abs(pnl_usd):,.0f}"
    else:
        pnl_str = "n/a"
    pct_str = f"{pnl_pct:+.0%}" if isinstance(pnl_pct, (int, float)) and not isinstance(pnl_pct, bool) else "n/a"
    held_str = f"{held:.0f} min" if isinstance(held, (int, float)) and not isinstance(held, bool) else "n/a"
    lesson = str(record.get("lesson") or "").strip()

    return f"DPOF lesson — {underlying} {direction} {pnl_str} ({pct_str}, {held_str}): {lesson}"


class ReflectionService:
    """Orchestrates the post-trade reflection loop.

    `cfg_getter()` returns the current `DPOFConfig` (called fresh each
    `run_pending`, matching `load_dpof_config()`'s "re-read every call"
    convention elsewhere in this package). `decisions_reader(position_id)` is
    an injected async callable that returns that position's review-decision
    rows (e.g. decisions.jsonl kind="position_review" records, read off-loop
    by the caller — B3 supplies the real one; tests supply a fake).
    `alert_fn(text)` is optional (sync or async; plain text, §4.7 format).
    `clock` defaults to `time.time` and is injectable for tests.
    """

    def __init__(
        self,
        store: LessonStore,
        llm: Any,
        *,
        cfg_getter: Callable[[], Any],
        decisions_reader: Callable[[str], Awaitable[list[dict]]],
        alert_fn: Optional[Callable[[str], Any]] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._store = store
        self._llm = llm
        self._cfg_getter = cfg_getter
        self._decisions_reader = decisions_reader
        self._alert_fn = alert_fn
        self._clock = clock
        self._running = False

    # ------------------------------------------------------------------
    # §4.3 — discovery + single-flight
    # ------------------------------------------------------------------

    async def run_pending(self, positions: list[PositionRecord], now: float) -> dict:
        """Status-driven scan of `positions` (whatever the caller currently
        holds — typically every closed position in the ledger) for ones
        eligible for a first-ever reflection, up to `cfg.reflection_max_per_run`,
        newest-closed first. Single-flight: a call that arrives while a prior
        one is still running is a no-op (never queues, never raises)."""
        if self._running:
            return {"skipped": "already_running"}
        self._running = True
        try:
            try:
                cfg = self._cfg_getter()
            except Exception as e:
                logger.error(f"DPOF reflection: cfg_getter failed: {type(e).__name__}: {e}")
                return {"skipped": "cfg_unavailable"}

            if not getattr(cfg, "reflection_enabled", True):
                return {"skipped": "disabled"}

            lookback_days = getattr(cfg, "reflection_lookback_days", 7.0)
            max_attempts = getattr(cfg, "reflection_max_attempts", 3)
            max_per_run = max(0, getattr(cfg, "reflection_max_per_run", 2))

            candidates: list[PositionRecord] = []
            for pos in positions or []:
                if not pos.id or self._store.has_lesson(pos.id):
                    continue
                if not is_eligible(pos, now, lookback_days=lookback_days):
                    continue
                state = self._store.get_state(pos.id)
                if state["attempts"] >= max_attempts:
                    continue
                next_at = state["next_attempt_at"]
                if next_at is not None and now < next_at:
                    continue
                candidates.append(pos)

            candidates.sort(key=lambda p: p.closed_time or 0.0, reverse=True)
            candidates = candidates[:max_per_run]

            result = {"attempted": 0, "stored": 0, "failed": 0, "positions": []}
            for pos in candidates:
                result["attempted"] += 1
                result["positions"].append(pos.id)
                try:
                    ok = await self._reflect_one(pos, cfg, now)
                except Exception as e:
                    logger.error(f"DPOF reflection: unexpected failure for {pos.id}: {type(e).__name__}: {e}", exc_info=True)
                    ok = False
                if ok:
                    result["stored"] += 1
                else:
                    result["failed"] += 1
            return result
        finally:
            self._running = False

    async def _reflect_one(self, pos: PositionRecord, cfg: Any, now: float) -> bool:
        try:
            reviews = await self._decisions_reader(pos.id)
        except Exception as e:
            logger.error(f"DPOF reflection: decisions_reader failed for {pos.id}: {type(e).__name__}: {e}")
            reviews = []
        if not isinstance(reviews, list):
            reviews = []

        try:
            record = build_trade_record(pos, reviews)
        except Exception as e:
            logger.error(f"DPOF reflection: build_trade_record failed for {pos.id}: {type(e).__name__}: {e}", exc_info=True)
            await self._safe_record_attempt(pos.id, f"build_trade_record:{type(e).__name__}", now)
            return False

        user_text = render_trade_record(record)
        primary_model = getattr(cfg, "reflection_model", "") or getattr(cfg, "decision_model", "")
        fallback_model = getattr(cfg, "reflection_fallback_model", "") or getattr(cfg, "fallback_model", "")
        timeout_seconds = getattr(cfg, "reflection_timeout_seconds", 60.0)

        try:
            outcome = await self._llm.complete_json(
                purpose="reflection",
                system=REFLECTION_SYSTEM_PROMPT,
                user=user_text,
                schema=REFLECTION_SCHEMA,
                schema_name="dpof_reflection",
                primary_model=primary_model,
                fallback_model=fallback_model,
                timeout_seconds=timeout_seconds,
                budget="dpof_reflection",
                effort="medium",  # a lesson is judgment; the debate stays on "low"
            )
        except Exception as e:
            logger.info(f"DPOF reflection: LLM call raised for {pos.id}: {type(e).__name__}: {e}")
            await self._safe_record_attempt(pos.id, f"llm_exception:{type(e).__name__}", now)
            return False

        if not getattr(outcome, "ok", False) or not isinstance(getattr(outcome, "parsed", None), dict):
            err = getattr(outcome, "error", "") or "unavailable"
            logger.info(f"DPOF reflection: unavailable for {pos.id}: {err}")
            await self._safe_record_attempt(pos.id, str(err), now)
            return False

        validated = validate_reflection(outcome.parsed)
        lesson_record = {
            "position_id": pos.id,
            "underlying": record.get("underlying"),
            "direction": record.get("direction"),
            "closed_at": record.get("closed_at"),
            "pnl_usd": record.get("pnl_usd"),
            "pnl_pct": record.get("pnl_pct"),
            "underlying_move_bps": record.get("underlying_move_bps"),
            "held_minutes": record.get("held_minutes"),
            "exit_category": record.get("exit_category"),
            "lesson": validated["lesson"],
            "entry_quality": validated["entry_quality"],
            "exit_quality": validated["exit_quality"],
            "key_factor": validated["key_factor"],
            "confidence": validated["confidence"],
            "provider": getattr(outcome, "provider", "") or "",
            "model": getattr(outcome, "model", "") or "",
            "mode": pos.mode or "",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            await self._store.add_lesson(lesson_record)
        except Exception as e:
            logger.error(f"DPOF reflection: LessonStore.add_lesson failed for {pos.id}: {type(e).__name__}: {e}", exc_info=True)
            return False

        if getattr(cfg, "lesson_alerts", True) and self._alert_fn is not None:
            try:
                text = _format_lesson_alert(lesson_record)
                maybe = self._alert_fn(text)
                if asyncio.iscoroutine(maybe):
                    await maybe
            except Exception as e:
                logger.debug(f"DPOF reflection: alert_fn failed for {pos.id}: {type(e).__name__}: {e}")

        logger.info(
            f"DPOF reflection: lesson stored for {pos.id} ({record.get('underlying')} {record.get('direction')}, "
            f"pnl={record.get('pnl_usd')})"
        )
        return True

    async def _safe_record_attempt(self, position_id: str, error: str, now: float) -> None:
        try:
            await self._store.record_attempt(position_id, error, now)
        except Exception as e:
            logger.error(f"DPOF reflection: record_attempt failed for {position_id}: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # §4.6 — lessons_ctx (pure over the cache — no I/O)
    # ------------------------------------------------------------------

    def lessons_ctx(self, underlying: str, *, same_n: int = 5, cross_n: int = 3) -> dict:
        """Pure over `self._store`'s in-memory cache. Returns
        {"same": [...up to same_n, newest first, for `underlying`],
         "cross": [...up to cross_n, newest first, for other underlyings],
         "tally": {underlying_symbol: {n, wins, losses, net_pnl_usd, avg_pnl_pct}, ...}}
        computed over the last 20 lessons overall (not just `underlying`'s
        own subset — the tally gives the model a sense of the whole recent
        track record, same and cross alike)."""
        recent20 = self._store.recent(20)

        same: list[dict] = []
        cross: list[dict] = []
        all_newest_first = self._store.recent(len(self._store.cache))
        for rec in all_newest_first:
            sym = rec.get("underlying")
            if sym == underlying and len(same) < same_n:
                same.append(rec)
            elif sym != underlying and len(cross) < cross_n:
                cross.append(rec)
            if len(same) >= same_n and len(cross) >= cross_n:
                break

        tally: dict[str, dict] = {}
        for rec in recent20:
            sym = rec.get("underlying")
            if not sym:
                continue
            bucket = tally.setdefault(sym, {"n": 0, "wins": 0, "losses": 0, "net_pnl_usd": 0.0, "_pnl_pcts": []})
            bucket["n"] += 1
            win = _win_loss_bucket(rec.get("pnl_usd"))
            if win is True:
                bucket["wins"] += 1
            elif win is False:
                bucket["losses"] += 1
            pnl_usd = rec.get("pnl_usd")
            if isinstance(pnl_usd, (int, float)) and not isinstance(pnl_usd, bool):
                bucket["net_pnl_usd"] += pnl_usd
            pnl_pct = rec.get("pnl_pct")
            if isinstance(pnl_pct, (int, float)) and not isinstance(pnl_pct, bool):
                bucket["_pnl_pcts"].append(pnl_pct)

        out_tally: dict[str, dict] = {}
        for sym, bucket in tally.items():
            pcts = bucket.pop("_pnl_pcts")
            bucket["avg_pnl_pct"] = (sum(pcts) / len(pcts)) if pcts else None
            out_tally[sym] = bucket

        return {"same": same, "cross": cross, "tally": out_tally}
