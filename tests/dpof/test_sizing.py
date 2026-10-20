"""Tests for agent/dpof/sizing.py.

Operator directive (2026-09-24, no-hard-caps amendment): sizing is the
decision model's call (EntryDecision.qty), not code's. The ONLY code-side
size bound left is SOLVENCY -- these tests cover max_affordable_qty (the
context ceiling offered to the model) and clamp_to_solvency (the fresh
re-check at actual entry time).
"""

import math

from agent.dpof.sizing import clamp_to_solvency, max_affordable_qty


# ── max_affordable_qty ───────────────────────────────────────────────────────


def test_max_affordable_qty_basic():
    # capital_available=2000, unit_debit=2.00 -> contract_cost=200 -> 10
    assert max_affordable_qty(2000.0, 2.00) == 10


def test_max_affordable_qty_floors_never_rounds_up():
    # 2050 / 200 = 10.25 -> floors to 10, never up to 11
    assert max_affordable_qty(2050.0, 2.00) == 10


def test_max_affordable_qty_exact_boundary():
    assert max_affordable_qty(200.0, 2.00) == 1


def test_max_affordable_qty_just_under_one_contract():
    assert max_affordable_qty(199.99, 2.00) == 0


def test_max_affordable_qty_none_capital():
    assert max_affordable_qty(None, 2.00) == 0


def test_max_affordable_qty_nonpositive_capital():
    assert max_affordable_qty(0.0, 2.00) == 0
    assert max_affordable_qty(-100.0, 2.00) == 0


def test_max_affordable_qty_none_debit():
    assert max_affordable_qty(2000.0, None) == 0


def test_max_affordable_qty_nonpositive_debit():
    assert max_affordable_qty(2000.0, 0.0) == 0
    assert max_affordable_qty(2000.0, -1.0) == 0


def test_max_affordable_qty_nonfinite_guarded():
    assert max_affordable_qty(math.inf, 2.00) == 0
    assert max_affordable_qty(2000.0, math.nan) == 0


# ── clamp_to_solvency ────────────────────────────────────────────────────────


def test_clamp_to_solvency_within_budget_passes_through():
    qty, reasons = clamp_to_solvency(5, 2000.0, 2.00)  # max_affordable=10
    assert qty == 5
    assert reasons == []


def test_clamp_to_solvency_exactly_at_ceiling_passes_through():
    qty, reasons = clamp_to_solvency(10, 2000.0, 2.00)
    assert qty == 10
    assert reasons == []


def test_clamp_to_solvency_over_budget_clamps_and_reports():
    qty, reasons = clamp_to_solvency(50, 2000.0, 2.00)  # decider asked for 50, only 10 affordable
    assert qty == 10
    assert reasons == ["clamped_by_solvency:50->10"]


def test_clamp_to_solvency_over_budget_and_insolvent_after_clamp():
    # capital just under one contract -> clamp lands at 0
    qty, reasons = clamp_to_solvency(5, 199.99, 2.00)
    assert qty == 0
    assert "insolvent" in reasons


def test_clamp_to_solvency_qty_none_skips():
    qty, reasons = clamp_to_solvency(None, 2000.0, 2.00)
    assert qty == 0
    assert reasons == ["qty_below_one_contract"]


def test_clamp_to_solvency_qty_zero_or_negative_skips():
    qty, reasons = clamp_to_solvency(0, 2000.0, 2.00)
    assert qty == 0
    assert reasons == ["qty_below_one_contract"]
    qty, reasons = clamp_to_solvency(-3, 2000.0, 2.00)
    assert qty == 0
    assert reasons == ["qty_below_one_contract"]


def test_clamp_to_solvency_qty_bool_rejected():
    # bool is an int subclass -- must not silently pass as qty=1/0.
    qty, reasons = clamp_to_solvency(True, 2000.0, 2.00)
    assert qty == 0
    assert reasons == ["qty_below_one_contract"]


def test_clamp_to_solvency_never_rounds_up_to_one_contract():
    # Hydrinia lesson: max(1, int(...)) must NOT happen here.
    qty, reasons = clamp_to_solvency(1, 1000.0, 1000.0)  # $100,000/contract, only $1000 available
    assert qty == 0
    assert "insolvent" in reasons


def test_clamp_to_solvency_no_capital_data_is_insolvent():
    qty, reasons = clamp_to_solvency(5, None, 2.00)
    assert qty == 0
    assert "insolvent" in reasons


def test_clamp_to_solvency_float_qty_from_a_lenient_decider_coerced_to_int():
    qty, reasons = clamp_to_solvency(4.0, 2000.0, 2.00)
    assert qty == 4
    assert isinstance(qty, int)
    assert reasons == []
