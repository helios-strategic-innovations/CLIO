"""DPOF-1 (S211): merge_entry_counters carries mid-cycle entry counters into the cycle's RiskState.

The cycle-end save used a RiskState rolled before the entry pipeline ran, so entries_today was
overwritten back to its pre-entry value. The merge must restore ONLY the entry counters.
(The end-to-end regression lives in test_engine.py::test_entry_counters_survive_the_cycle_end_save.)
"""

from agent.dpof.models import RiskState
from agent.dpof.risk import merge_entry_counters

DAY = "2026-09-29"


def _r(**kw):
    kw.setdefault("day", DAY)
    return RiskState(**kw)


def test_stale_cycle_state_picks_up_the_ledgers_entry():
    cycle = _r()  # rolled at step 7, before the entry
    ledger = _r(entries_today=1, entries_by_symbol={"SPY": 1})
    out = merge_entry_counters(cycle, ledger)
    assert out.entries_today == 1
    assert out.entries_by_symbol == {"SPY": 1}


def test_takes_the_max_per_symbol_and_overall():
    cycle = _r(entries_today=3, entries_by_symbol={"SPY": 2, "QQQ": 1})
    ledger = _r(entries_today=4, entries_by_symbol={"SPY": 1, "QQQ": 3, "IWM": 1})
    out = merge_entry_counters(cycle, ledger)
    assert out.entries_today == 4
    assert out.entries_by_symbol == {"SPY": 2, "QQQ": 3, "IWM": 1}


def test_lower_ledger_counters_never_reduce_the_cycles():
    cycle = _r(entries_today=5, entries_by_symbol={"SPY": 5})
    ledger = _r(entries_today=1, entries_by_symbol={"SPY": 1})
    out = merge_entry_counters(cycle, ledger)
    assert out.entries_today == 5 and out.entries_by_symbol == {"SPY": 5}


def test_day_mismatch_keeps_the_cycles_state_untouched():
    """A rollover happened between the two copies: yesterday's ledger counters must not leak in."""
    cycle = _r(day="2026-09-29", entries_today=0)
    ledger = _r(day="2026-09-28", entries_today=7, entries_by_symbol={"SPY": 7})
    assert merge_entry_counters(cycle, ledger) is cycle


def test_only_entry_counters_are_merged():
    """Nothing else in the ledger copy may be resurrected -- halts, P&L, cooldown, streaks."""
    cycle = _r(realized_pnl_today=130.0, consecutive_losses=0, cooldown_until=None,
               daily_halted=False, hard_halted=False, halt_reason="")
    ledger = _r(entries_today=1, entries_by_symbol={"SPY": 1}, realized_pnl_today=999.0,
                consecutive_losses=4, cooldown_until=1790000000.0, daily_halted=True,
                hard_halted=True, halt_reason="operator")
    out = merge_entry_counters(cycle, ledger)
    assert out.entries_today == 1
    assert out.realized_pnl_today == 130.0
    assert out.consecutive_losses == 0
    assert out.cooldown_until is None
    assert out.daily_halted is False and out.hard_halted is False and out.halt_reason == ""


def test_inputs_are_not_mutated_and_result_is_idempotent():
    cycle = _r(entries_today=0, entries_by_symbol={})
    ledger = _r(entries_today=2, entries_by_symbol={"SPY": 2})
    out = merge_entry_counters(cycle, ledger)
    assert cycle.entries_today == 0 and cycle.entries_by_symbol == {}
    assert ledger.entries_by_symbol == {"SPY": 2}
    assert merge_entry_counters(out, ledger) == out
    out.entries_by_symbol["SPY"] = 99  # the merged dict is its own copy
    assert ledger.entries_by_symbol == {"SPY": 2}
