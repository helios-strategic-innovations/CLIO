"""CLIO producer side of the shared decision record.

`canonical_json`/`state_hash` are a byte-identical copy of the consumer's own
two functions; `tests/decisions/test_records.py` pins the same golden vector
the consumer pins. Every other function here is CLIO-side only: pure builders that turn a
DPOF `/v1/systemone` request (state + questions), one validated answer, or one
closed `PositionRecord` into the row shapes `decisions.questions` /
`decisions.answers` / `decisions.outcomes` expect (migration
020_decisions_schema.sql). No I/O, no imports of agent.dpof.* — this module
never runs anything, it only shapes dicts.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from typing import Any, Mapping, Optional

NOUL_KEYS = ("false", "true")

# Fixed namespace for uuid5 position-outcome ids.
# Generated once (uuid4) and frozen here as a constant — never regenerate this
# value, or every previously-written outcome id becomes unreachable by retry.
_POSITION_OUTCOME_NAMESPACE = uuid.UUID("7d1a3c9e-5b2f-4b41-9a2e-6f0d7c8b5a31")


class RecordError(ValueError):
    """A malformed request/answer/position handed to a builder in this module."""


# ─────────────────────────────────────────────────────────────────────────
# Canonical JSON / state hashing — byte-identical to the consumer's reference records module
# ─────────────────────────────────────────────────────────────────────────

def canonical_json(obj: Any) -> str:
    """The one serialisation used for hashing: sorted keys, no whitespace, UTF-8 kept as is."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def state_hash(state: Mapping[str, Any]) -> str:
    """sha256 hex digest of the canonical JSON of `state`."""
    return hashlib.sha256(canonical_json(state).encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────
# Canonical probability shapes
# ─────────────────────────────────────────────────────────────────────────

def _prob(value: Any) -> Optional[float]:
    """isinstance-safe numeric coercion in [0, 1]; rejects bool (bool is an int subclass)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        return None
    return value


def canonical_keys(qtype: str, criteria: Optional[Mapping[str, str]]) -> tuple[str, ...]:
    """Canonical (sorted) option keys for one question; noul is always ("false", "true")."""
    if qtype == "noul":
        return NOUL_KEYS
    return tuple(sorted((criteria or {}).keys()))


def noul_probabilities(p: Any) -> dict:
    """A noul probability p stored canonically as {"false": 1-p, "true": p}."""
    val = _prob(p)
    if val is None:
        raise RecordError("noul probability must be a finite number in [0, 1]")
    return {"false": 1.0 - val, "true": val}


def canonical_probabilities(qtype: str, criteria: Optional[Mapping[str, str]], answer: Mapping[str, Any]) -> dict:
    """Canonical-key probabilities for one validated `/v1/systemone` answer
    (`agent.dpof.jev.validate_answers`'s clean shape: `{"type": "noul", "noul": v}`
    or `{"type": "choice", "choice": k, "probabilities": {...}, "confidence": c}`).
    Choice probabilities are renormalised to sum to exactly 1 over the
    question's own canonical keys — mirrors the consumer's own
    `canonical_probabilities` so both sides agree on the same corpus."""
    if not isinstance(answer, Mapping) or answer.get("type") != qtype:
        raise RecordError("answer type does not match the question")
    if qtype == "noul":
        return noul_probabilities(answer.get("noul"))
    if qtype == "choice":
        probs = answer.get("probabilities")
        keys = canonical_keys(qtype, criteria)
        if not isinstance(probs, Mapping) or set(probs.keys()) != set(keys):
            raise RecordError("probabilities must cover exactly the question's keys")
        vals = {k: _prob(v) for k, v in probs.items()}
        if any(v is None for v in vals.values()):
            raise RecordError("every probability must be a finite number in [0, 1]")
        total = sum(vals.values())
        if total <= 0:
            raise RecordError("probabilities sum to zero")
        return {k: vals[k] / total for k in keys}
    raise RecordError(f"qtype {qtype!r} is not stored (score answers unsupported until the rounding fix lands)")


def argmax_key(probabilities: Mapping[str, float]) -> str:
    """The canonical key with the highest probability. Ties broken by
    alphabetical key order — deterministic regardless of dict insertion order."""
    return max(sorted(probabilities.keys()), key=lambda k: probabilities[k])


# ─────────────────────────────────────────────────────────────────────────
# decisions.questions row builders
# ─────────────────────────────────────────────────────────────────────────

def build_question_row(
    *,
    producer: str,
    producer_ref: Mapping[str, Any],
    call_id: str,
    question_key: str,
    qtype: str,
    instructions: str,
    criteria: Optional[Mapping[str, str]],
    state: Mapping[str, Any],
    position_id: Optional[str] = None,
    rotation_group: Optional[str] = None,
    rotation_index: int = 0,
) -> dict:
    """One `decisions.questions` row. `rotation_group` defaults to a fresh
    uuid4 (CLIO never rotates option order itself in production — one
    canonical order per question, rotation_index=0; the downstream labeller is
    the one that asks a rotated copy on replay)."""
    keys = canonical_keys(qtype, criteria)
    return {
        "id": str(uuid.uuid4()),
        "producer": producer,
        "producer_ref": dict(producer_ref or {}),
        "call_id": call_id,
        "rotation_group": rotation_group or str(uuid.uuid4()),
        "rotation_index": rotation_index,
        "question_key": question_key,
        "qtype": qtype,
        "instructions": instructions,
        "criteria": dict(criteria) if criteria is not None else None,
        "option_order": list(keys),
        "state": dict(state),
        "state_hash": state_hash(state),
        "position_id": position_id,
        "split": None,
    }


def build_question_rows(
    *,
    producer: str,
    producer_ref: Mapping[str, Any],
    call_id: str,
    state: Mapping[str, Any],
    questions: Mapping[str, Mapping[str, Any]],
    position_id: Optional[str] = None,
) -> list[dict]:
    """One row per question in a `/v1/systemone`-shaped `questions` dict
    (question_key -> {"type", "instructions", "criteria"}), same `state` on
    every row (the full state sent for that call)."""
    rows: list[dict] = []
    for qid, spec in (questions or {}).items():
        if not isinstance(spec, Mapping):
            continue
        rows.append(
            build_question_row(
                producer=producer,
                producer_ref=producer_ref,
                call_id=call_id,
                question_key=qid,
                qtype=spec.get("type"),
                instructions=spec.get("instructions", ""),
                criteria=spec.get("criteria"),
                state=state,
                position_id=position_id,
            )
        )
    return rows


# ─────────────────────────────────────────────────────────────────────────
# decisions.answers row builders
# ─────────────────────────────────────────────────────────────────────────

def build_answer_row(
    question_row: Mapping[str, Any],
    *,
    model: str,
    role: str,
    endpoint: str = "",
    answer: Optional[Mapping[str, Any]] = None,
    error: str = "",
    confidence: Optional[float] = None,
    latency_ms: Optional[float] = None,
) -> dict:
    """One `decisions.answers` row for one `decisions.questions` row (as
    built above). `answer` is one validated `/v1/systemone` answer for this
    question_key, or None on failure — the DB CHECK requires exactly one of
    (probabilities, error) to be set, which this always satisfies."""
    qtype = question_row["qtype"]
    criteria = question_row.get("criteria")
    probabilities: Optional[dict] = None
    choice: Optional[str] = None
    conf = confidence
    row_error = error or ""

    if answer is not None and not row_error:
        try:
            probabilities = canonical_probabilities(qtype, criteria, answer)
        except RecordError as e:
            row_error = str(e)
        else:
            choice = argmax_key(probabilities)
            if conf is None and isinstance(answer, Mapping):
                conf = _prob(answer.get("confidence"))

    if probabilities is None and not row_error:
        row_error = "missing_answer"

    lat: Optional[int] = None
    if isinstance(latency_ms, (int, float)) and not isinstance(latency_ms, bool) and math.isfinite(latency_ms):
        lat = max(0, int(round(latency_ms)))

    return {
        "question_id": question_row["id"],
        "model": model or "",
        "endpoint": endpoint or "",
        "role": role,
        "probabilities": probabilities,
        "choice": choice,
        "confidence": conf,
        "latency_ms": lat,
        "error": row_error or None,
    }


def build_answer_rows(
    question_rows: list[dict],
    *,
    model: str,
    role: str,
    endpoint: str = "",
    answers: Optional[Mapping[str, Mapping[str, Any]]] = None,
    error: str = "",
    latency_ms: Optional[float] = None,
) -> list[dict]:
    """One `decisions.answers` row per question row. When `answers` is
    falsy (the call failed outright), every row is an error row using
    `error`; a question_key present in `question_rows` but missing from
    `answers` is ALSO an error row (never silently dropped — its question
    row was already recorded)."""
    out: list[dict] = []
    for qrow in question_rows:
        qid = qrow["question_key"]
        ans = (answers or {}).get(qid) if answers else None
        row_error = "" if ans is not None else (error or "missing_answer")
        out.append(
            build_answer_row(
                qrow, model=model, role=role, endpoint=endpoint, answer=ans,
                error=row_error, latency_ms=latency_ms,
            )
        )
    return out


# ─────────────────────────────────────────────────────────────────────────
# decisions.outcomes row builder — closed positions
# ─────────────────────────────────────────────────────────────────────────

def position_outcome_id(position_id: str, outcome_source: str) -> str:
    """uuid5(position_id|outcome_source) under a fixed namespace — a retry
    for the same position/source is always the same id (idempotent write)."""
    return str(uuid.uuid5(_POSITION_OUTCOME_NAMESPACE, f"{position_id}|{outcome_source}"))


def _outcome_bucket(realized_pnl: Optional[float]) -> str:
    if realized_pnl is None:
        return "unknown"
    if realized_pnl > 0:
        return "worked"
    if realized_pnl < 0:
        return "failed"
    return "mixed"


def _numeric_or_none(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def build_position_outcome_row(
    position: Mapping[str, Any],
    *,
    outcome_source: str = "live",
    call_id: Optional[str] = None,
) -> Optional[dict]:
    """One `decisions.outcomes` row for one CLOSED position. `position` is a
    `PositionRecord.to_dict()`-shaped mapping. Returns None when the position
    carries no id (nothing to key the row on). `call_id` is the position's
    recorded entry-veto `decision_call_id` when present — the row is still
    written without one (call_id NULL, position_id still set) so a position
    that entered before this feature existed, or while Jev was OFF/decisions
    disabled at entry time, still gets an outcome row for a later join.

    `detail` only ever carries fields that exist on `position` — a missing/non-numeric field is simply omitted, never
    a fabricated 0 (CLAUDE.md's own "no fake zeros" rule)."""
    position_id = position.get("id") if isinstance(position, Mapping) else None
    if not position_id:
        return None

    realized_pnl = _numeric_or_none(position.get("realized_pnl"))

    pnl_pct: Optional[float] = None
    entry_price = _numeric_or_none(position.get("entry_price"))
    qty_original = _numeric_or_none(position.get("qty_original"))
    if entry_price is not None and entry_price > 0 and qty_original is not None and qty_original > 0 and realized_pnl is not None:
        premium_paid = entry_price * 100.0 * qty_original
        if premium_paid > 0:
            pnl_pct = realized_pnl / premium_paid

    hold_minutes: Optional[float] = None
    entry_time = _numeric_or_none(position.get("entry_time"))
    closed_time = _numeric_or_none(position.get("closed_time"))
    if entry_time is not None and closed_time is not None:
        hold_minutes = max(0.0, (closed_time - entry_time) / 60.0)

    detail: dict = {}
    exit_reason = position.get("exit_reason")
    if exit_reason:
        detail["exit_reason"] = exit_reason
    if pnl_pct is not None:
        detail["pnl_pct"] = pnl_pct
    obs_max = _numeric_or_none(position.get("obs_max_pnl_pct"))
    if obs_max is not None:
        detail["obs_max_pnl_pct"] = obs_max
    obs_min = _numeric_or_none(position.get("obs_min_pnl_pct"))
    if obs_min is not None:
        detail["obs_min_pnl_pct"] = obs_min
    if hold_minutes is not None:
        detail["hold_minutes"] = hold_minutes

    return {
        "id": position_outcome_id(str(position_id), outcome_source),
        "question_id": None,
        "rotation_group": None,
        "call_id": call_id,
        "position_id": str(position_id),
        "outcome": _outcome_bucket(realized_pnl),
        "outcome_source": outcome_source,
        "state_complete": "unknown",
        "pnl_usd": realized_pnl,
        "detail": detail,
    }
