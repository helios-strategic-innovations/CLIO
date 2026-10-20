"""Tests for agent.decisions.records — canonical JSON/state hashing, the
canonical probability shapes, and the row builders for decisions.questions /
decisions.answers / decisions.outcomes (migration 020_decisions_schema.sql).

The golden state_hash vector is the one pinned in the consumer's own
tests/test_records.py -- both sides must hash a state
byte-identically.
"""

from __future__ import annotations

import uuid

import pytest

from agent.decisions.records import (
    RecordError,
    argmax_key,
    build_answer_row,
    build_answer_rows,
    build_position_outcome_row,
    build_question_row,
    build_question_rows,
    canonical_json,
    canonical_keys,
    canonical_probabilities,
    noul_probabilities,
    position_outcome_id,
    state_hash,
)

_GOLDEN_STATE = {
    "underlying": "SPY",
    "signals": {"options_flow": "strong bullish", "vix": "normal (15-20)"},
    "headlines": ["Fed holds rates — «steady»"],
    "n": 3,
}
_GOLDEN_HASH = "3df70c3d9e795bb32ecf338c059bcf924bbbb4e8d61b39d4524f376076098eae"


# ─────────────────────────────────────────────────────────────────────────
# Golden vector — must match the consumer's reference records module
# ─────────────────────────────────────────────────────────────────────────

def test_golden_state_hash_vector():
    assert state_hash(_GOLDEN_STATE) == _GOLDEN_HASH


def test_canonical_json_sorted_keys_no_whitespace():
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_state_hash_is_sha256_hex():
    h = state_hash({"a": 1})
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_state_hash_order_independent():
    assert state_hash({"a": 1, "b": 2}) == state_hash({"b": 2, "a": 1})


def test_state_hash_changes_with_content():
    assert state_hash({"a": 1}) != state_hash({"a": 2})


# ─────────────────────────────────────────────────────────────────────────
# canonical_keys / noul_probabilities / canonical_probabilities / argmax_key
# ─────────────────────────────────────────────────────────────────────────

def test_canonical_keys_noul_is_fixed():
    assert canonical_keys("noul", None) == ("false", "true")
    assert canonical_keys("noul", {"true": "x", "false": "y"}) == ("false", "true")


def test_canonical_keys_choice_sorted():
    assert canonical_keys("choice", {"zeta": "z", "alpha": "a", "mid": "m"}) == ("alpha", "mid", "zeta")


def test_canonical_keys_choice_missing_criteria_empty():
    assert canonical_keys("choice", None) == ()


def test_noul_probabilities_shape():
    assert noul_probabilities(0.3) == {"false": pytest.approx(0.7), "true": pytest.approx(0.3)}
    assert noul_probabilities(0.0) == {"false": 1.0, "true": 0.0}
    assert noul_probabilities(1.0) == {"false": 0.0, "true": 1.0}


def test_noul_probabilities_rejects_out_of_range_and_non_numeric():
    with pytest.raises(RecordError):
        noul_probabilities(1.5)
    with pytest.raises(RecordError):
        noul_probabilities(-0.1)
    with pytest.raises(RecordError):
        noul_probabilities(True)  # bool must not pass as numeric
    with pytest.raises(RecordError):
        noul_probabilities("nope")
    with pytest.raises(RecordError):
        noul_probabilities(float("nan"))


def test_canonical_probabilities_noul():
    answer = {"type": "noul", "noul": 0.25}
    assert canonical_probabilities("noul", None, answer) == {"false": 0.75, "true": 0.25}


def test_canonical_probabilities_choice_renormalizes():
    criteria = {"aligned": "x", "mixed": "y", "late_or_exhausted": "z"}
    answer = {
        "type": "choice", "choice": "aligned",
        "probabilities": {"aligned": 0.5, "mixed": 0.3, "late_or_exhausted": 0.19},  # sums to 0.99
        "confidence": 0.9,
    }
    probs = canonical_probabilities("choice", criteria, answer)
    assert set(probs.keys()) == set(criteria.keys())
    assert probs["aligned"] == pytest.approx(0.5 / 0.99)
    assert sum(probs.values()) == pytest.approx(1.0)


def test_canonical_probabilities_rejects_type_mismatch():
    with pytest.raises(RecordError):
        canonical_probabilities("noul", None, {"type": "choice", "choice": "a"})


def test_canonical_probabilities_rejects_missing_keys():
    criteria = {"a": "x", "b": "y"}
    with pytest.raises(RecordError):
        canonical_probabilities("choice", criteria, {"type": "choice", "choice": "a", "probabilities": {"a": 1.0}})


def test_canonical_probabilities_rejects_non_finite_values():
    criteria = {"a": "x", "b": "y"}
    with pytest.raises(RecordError):
        canonical_probabilities(
            "choice", criteria, {"type": "choice", "choice": "a", "probabilities": {"a": float("nan"), "b": 0.5}},
        )


def test_argmax_key_simple():
    assert argmax_key({"a": 0.1, "b": 0.9}) == "b"


def test_argmax_key_ties_break_alphabetically():
    assert argmax_key({"z": 0.5, "a": 0.5}) == "a"
    assert argmax_key({"true": 0.5, "false": 0.5}) == "false"


# ─────────────────────────────────────────────────────────────────────────
# build_question_row / build_question_rows
# ─────────────────────────────────────────────────────────────────────────

def _questions_wire():
    return {
        "setup": {
            "type": "choice", "instructions": "judge the setup",
            "criteria": {"aligned": "x", "mixed": "y", "late_or_exhausted": "z"},
        },
        "headline_conflict": {
            "type": "noul", "instructions": "do headlines conflict",
            "criteria": {"true": "yes", "false": "no"},
        },
    }


def test_build_question_row_shape():
    state = {"underlying": "SPY"}
    row = build_question_row(
        producer="dpof_entry_veto", producer_ref={"symbol": "SPY"}, call_id=str(uuid.uuid4()),
        question_key="setup", qtype="choice", instructions="judge",
        criteria={"aligned": "x", "mixed": "y"}, state=state,
    )
    assert uuid.UUID(row["id"])  # valid uuid4
    assert uuid.UUID(row["rotation_group"])
    assert row["rotation_index"] == 0
    assert row["option_order"] == ["aligned", "mixed"]
    assert row["state"] == state
    assert row["state_hash"] == state_hash(state)
    assert row["producer"] == "dpof_entry_veto"
    assert row["position_id"] is None
    assert row["split"] is None


def test_build_question_row_noul_option_order_fixed():
    row = build_question_row(
        producer="dpof_exit_sentinel", producer_ref={}, call_id=str(uuid.uuid4()),
        question_key="thesis_danger", qtype="noul", instructions="judge",
        criteria=None, state={"a": 1}, position_id="P1",
    )
    assert row["option_order"] == ["false", "true"]
    assert row["position_id"] == "P1"


def test_build_question_rows_one_per_question_same_state_different_rotation_groups():
    state = {"underlying": "SPY"}
    call_id = str(uuid.uuid4())
    rows = build_question_rows(
        producer="dpof_entry_veto", producer_ref={}, call_id=call_id, state=state, questions=_questions_wire(),
    )
    assert len(rows) == 2
    assert {r["question_key"] for r in rows} == {"setup", "headline_conflict"}
    assert all(r["call_id"] == call_id for r in rows)
    assert all(r["state"] == state for r in rows)
    # Every question gets its own fresh rotation_group (CLIO never rotates option order itself).
    assert rows[0]["rotation_group"] != rows[1]["rotation_group"]
    assert all(r["id"] != rows[0]["id"] for r in rows[1:])


def test_build_question_rows_position_id_threaded_through():
    rows = build_question_rows(
        producer="dpof_exit_sentinel", producer_ref={}, call_id=str(uuid.uuid4()),
        state={"a": 1}, questions=_questions_wire(), position_id="P42",
    )
    assert all(r["position_id"] == "P42" for r in rows)


# ─────────────────────────────────────────────────────────────────────────
# build_answer_row / build_answer_rows
# ─────────────────────────────────────────────────────────────────────────

def _qrow(qtype="choice", criteria=None):
    return build_question_row(
        producer="dpof_entry_veto", producer_ref={}, call_id=str(uuid.uuid4()),
        question_key="setup", qtype=qtype, instructions="judge",
        criteria=criteria if criteria is not None else ({"aligned": "x", "mixed": "y"} if qtype == "choice" else None),
        state={"a": 1},
    )


def test_build_answer_row_choice_success():
    qrow = _qrow(qtype="choice")
    answer = {"type": "choice", "choice": "aligned", "probabilities": {"aligned": 0.8, "mixed": 0.2}, "confidence": 0.9}
    row = build_answer_row(qrow, model="jev-1.13.0", role="served", endpoint="primary", answer=answer, latency_ms=42.4)
    assert row["question_id"] == qrow["id"]
    assert row["choice"] == "aligned"
    assert row["probabilities"] == {"aligned": 0.8, "mixed": 0.2}
    assert row["confidence"] == pytest.approx(0.9)
    assert row["error"] is None
    assert row["latency_ms"] == 42
    assert row["model"] == "jev-1.13.0"
    assert row["role"] == "served"


def test_build_answer_row_noul_success_choice_is_argmax():
    qrow = _qrow(qtype="noul")
    answer = {"type": "noul", "noul": 0.9}
    row = build_answer_row(qrow, model="jev-1.13.0", role="served", answer=answer)
    assert row["probabilities"] == {"false": pytest.approx(0.1), "true": pytest.approx(0.9)}
    assert row["choice"] == "true"
    assert row["error"] is None


def test_build_answer_row_error_when_answer_is_none():
    qrow = _qrow(qtype="noul")
    row = build_answer_row(qrow, model="jev-1.13.0", role="served", answer=None, error="jev_unavailable:timeout")
    assert row["probabilities"] is None
    assert row["choice"] is None
    assert row["error"] == "jev_unavailable:timeout"


def test_build_answer_row_exactly_one_of_probabilities_or_error():
    qrow = _qrow(qtype="noul")
    ok = build_answer_row(qrow, model="m", role="served", answer={"type": "noul", "noul": 0.5})
    bad = build_answer_row(qrow, model="m", role="served", answer=None, error="boom")
    assert (ok["probabilities"] is not None) != (ok["error"] is not None)
    assert (bad["probabilities"] is not None) != (bad["error"] is not None)


def test_build_answer_row_malformed_answer_becomes_error_row():
    qrow = _qrow(qtype="choice")
    bad_answer = {"type": "choice", "choice": "aligned", "probabilities": {"aligned": 1.5}, "confidence": 0.9}
    row = build_answer_row(qrow, model="jev-1.13.0", role="served", answer=bad_answer)
    assert row["probabilities"] is None
    assert row["error"] is not None


def test_build_answer_rows_missing_question_key_becomes_error_row():
    qrows = build_question_rows(
        producer="dpof_entry_veto", producer_ref={}, call_id=str(uuid.uuid4()),
        state={"a": 1}, questions=_questions_wire(),
    )
    answers = {"setup": {"type": "choice", "choice": "aligned", "probabilities": {"aligned": 0.9, "mixed": 0.05, "late_or_exhausted": 0.05}, "confidence": 0.9}}
    rows = build_answer_rows(qrows, model="jev-1.13.0", role="served", answers=answers)
    by_key = {r["question_id"]: r for r in rows}
    setup_row = next(r for qr, r in zip(qrows, rows) if qr["question_key"] == "setup")
    hc_row = next(r for qr, r in zip(qrows, rows) if qr["question_key"] == "headline_conflict")
    assert setup_row["error"] is None
    assert hc_row["error"] == "missing_answer"
    assert hc_row["probabilities"] is None


def test_build_answer_rows_falsy_answers_dict_all_error():
    qrows = build_question_rows(
        producer="dpof_entry_veto", producer_ref={}, call_id=str(uuid.uuid4()),
        state={"a": 1}, questions=_questions_wire(),
    )
    rows = build_answer_rows(qrows, model="jev-1.13.0", role="served", answers={}, error="jev_unavailable:timeout")
    assert len(rows) == 2
    assert all(r["error"] == "jev_unavailable:timeout" for r in rows)
    assert all(r["probabilities"] is None for r in rows)


def test_build_answer_rows_answers_none_all_error_default_message():
    qrows = build_question_rows(
        producer="dpof_entry_veto", producer_ref={}, call_id=str(uuid.uuid4()),
        state={"a": 1}, questions=_questions_wire(),
    )
    rows = build_answer_rows(qrows, model="jev-1.13.0", role="served", answers=None)
    assert all(r["error"] == "missing_answer" for r in rows)


# ─────────────────────────────────────────────────────────────────────────
# build_position_outcome_row / position_outcome_id
# ─────────────────────────────────────────────────────────────────────────

def _closed_position(**overrides) -> dict:
    base = {
        "id": "P1", "realized_pnl": 120.0, "exit_reason": "decider_exit",
        "entry_price": 2.0, "qty_original": 3, "entry_time": 1000.0, "closed_time": 1900.0,
        "obs_max_pnl_pct": 0.35, "obs_min_pnl_pct": -0.05,
    }
    base.update(overrides)
    return base


def test_position_outcome_id_deterministic_and_stable():
    id1 = position_outcome_id("P1", "live")
    id2 = position_outcome_id("P1", "live")
    assert id1 == id2
    assert uuid.UUID(id1)
    assert id1 != position_outcome_id("P1", "review")
    assert id1 != position_outcome_id("P2", "live")


def test_build_position_outcome_row_worked():
    row = build_position_outcome_row(_closed_position(realized_pnl=120.0), call_id="C1")
    assert row["id"] == position_outcome_id("P1", "live")
    assert row["position_id"] == "P1"
    assert row["call_id"] == "C1"
    assert row["outcome"] == "worked"
    assert row["outcome_source"] == "live"
    assert row["state_complete"] == "unknown"
    assert row["pnl_usd"] == 120.0
    assert row["detail"]["exit_reason"] == "decider_exit"
    assert row["detail"]["pnl_pct"] == pytest.approx(120.0 / (2.0 * 100.0 * 3))
    assert row["detail"]["obs_max_pnl_pct"] == 0.35
    assert row["detail"]["obs_min_pnl_pct"] == -0.05
    assert row["detail"]["hold_minutes"] == pytest.approx((1900.0 - 1000.0) / 60.0)


def test_build_position_outcome_row_failed_and_mixed_and_unknown():
    failed = build_position_outcome_row(_closed_position(realized_pnl=-50.0))
    assert failed["outcome"] == "failed"
    mixed = build_position_outcome_row(_closed_position(realized_pnl=0.0))
    assert mixed["outcome"] == "mixed"
    unknown = build_position_outcome_row(_closed_position(realized_pnl=None))
    assert unknown["outcome"] == "unknown"
    assert unknown["pnl_usd"] is None


def test_build_position_outcome_row_none_call_id_still_written():
    row = build_position_outcome_row(_closed_position(), call_id=None)
    assert row["call_id"] is None
    assert row["position_id"] == "P1"


def test_build_position_outcome_row_no_id_returns_none():
    assert build_position_outcome_row({"realized_pnl": 10.0}) is None
    assert build_position_outcome_row({"id": "", "realized_pnl": 10.0}) is None


def test_build_position_outcome_row_only_fields_that_exist():
    """A position missing entry_price/qty_original/obs_*/hold-time fields
    must never fabricate them -- detail only carries what's actually there."""
    minimal = {"id": "P9", "realized_pnl": 5.0}
    row = build_position_outcome_row(minimal)
    assert "pnl_pct" not in row["detail"]
    assert "obs_max_pnl_pct" not in row["detail"]
    assert "obs_min_pnl_pct" not in row["detail"]
    assert "hold_minutes" not in row["detail"]
    assert "exit_reason" not in row["detail"]
    assert row["pnl_usd"] == 5.0


def test_build_position_outcome_row_idempotent_id_across_calls():
    row1 = build_position_outcome_row(_closed_position(), call_id="C1")
    row2 = build_position_outcome_row(_closed_position(), call_id="C1")
    assert row1["id"] == row2["id"]
