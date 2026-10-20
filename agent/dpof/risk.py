"""
CLIO DPOF — risk envelope (pure).

Operator directive (2026-09-24): NO mechanical breakers. There is no
daily/weekly loss breaker, no max-drawdown breaker, no loss-streak cooldown,
no per-symbol re-entry cooldown — entries AND exits are judged by CLIO's
decision model, and the only rail code enforces is the 15:50 ET forced
flatten (exits.py). This module keeps exactly what the operator kept:

  - P&L ACCOUNTING (realized today/week/total, entries_today,
    entries_by_symbol, start-of-day/week equity) for `/dpof status` and
    reporting — day/week rollover via `roll_state`.
  - The ENTRY GATE (`check_entry_allowed`): operator halt (manual, via
    `/dpof halt` or the `data/dpof/HALT` sentinel file) + the entry time
    window. Operator directive (no-hard-caps amendment, 2026-09-24): there is
    no max_concurrent_positions/max_entries_per_day/max_entries_per_symbol_day
    count cap any more — the ONE code-side size bound left is SOLVENCY
    (sizing.py's `max_affordable_qty`/`clamp_to_solvency`, enforced at actual
    entry-sizing time, not here). One-in-flight-deliberation-per-underlying
    (engine.py) is concurrency control, not a cap, and stays engine-side.
  - OPERATOR halt/resume (`operator_halt` / `operator_resume`) — the only
    way `hard_halted` is ever set or cleared; nothing in this module sets it
    on its own any more.

See docs/design/clio_dpof.md §6.

No I/O, no clock reads: every function takes `now_et` / `now_epoch` as a
parameter. Every function here returns a *new* RiskState — dataclasses.replace()
throughout, and any dict field that changes is rebuilt as a new dict — the
input RiskState (including its nested dicts) is never mutated in place.

ET day/week keys are computed from a tz-aware datetime (ZoneInfo("America/
New_York")) via strftime/isocalendar; cooldown/window comparisons use
`datetime.timestamp()` (an absolute UTC instant), so all of this is DST-safe
by construction — no manual UTC-offset math anywhere in this module.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime, time

from agent.dpof.config import DPOFConfig
from agent.dpof.models import GateResult, RiskState

logger = logging.getLogger("clio.dpof.risk")


def et_day(now_et: datetime) -> str:
    """"YYYY-MM-DD" in the timezone `now_et` already carries."""
    return now_et.strftime("%Y-%m-%d")


def et_week(now_et: datetime) -> str:
    """ISO week, "YYYY-Www" (e.g. "2026-W39"), in the timezone `now_et` carries."""
    iso = now_et.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def parse_hhmm(s: str) -> time:
    """Parse "HH:MM" (as stored in DPOFConfig.entry_start_et etc.) into a naive time."""
    hh, mm = s.split(":")
    return time(hour=int(hh), minute=int(mm))


def roll_state(state: RiskState, now_et: datetime, equity: float | None) -> RiskState:
    """Day/week rollover + high-water-mark update. Never clears hard_halted —
    that only happens via operator_resume().
    """
    new_state = replace(state)

    today = et_day(now_et)
    if today != state.day:
        new_state = replace(
            new_state,
            day=today,
            realized_pnl_today=0.0,
            entries_today=0,
            entries_by_symbol={},
            daily_halted=False,
            start_of_day_equity=equity,
            # Stale-data alarm is per session: yesterday's last-healthy stamp must
            # not make the alarm fire the moment today's window opens.
            last_valid_snapshot=None,
            stale_alarm_sent=False,
            # consecutive_losses intentionally NOT reset here (spec: "stays").
        )

    week = et_week(now_et)
    if week != state.week:
        new_state = replace(
            new_state,
            week=week,
            realized_pnl_week=0.0,
            start_of_week_equity=equity,
        )

    if equity is not None:
        prior_hwm = new_state.hwm_equity
        new_hwm = equity if prior_hwm is None else max(prior_hwm, equity)
        new_state = replace(new_state, hwm_equity=new_hwm)

    return new_state


def check_entry_allowed(
    state: RiskState,
    now_et: datetime,
    symbol: str,
    cfg: DPOFConfig,
    halt_file_present: bool = False,
) -> GateResult:
    """All gates are checked (not short-circuited) so `reasons` lists every
    currently-failing check, not just the first one.

    Operator directive (2026-09-24, no-hard-caps amendment): no mechanical
    breakers AND no entry-count caps. The only halt is the OPERATOR's (manual
    `/dpof halt` -> `hard_halted`, or the `data/dpof/HALT` sentinel file) plus
    the entry time window -- there is no daily/weekly/drawdown breaker,
    loss-streak cooldown, per-symbol re-entry cooldown,
    max_concurrent_positions, max_entries_per_day, or max_entries_per_symbol_day
    here any more. `symbol` is accepted for call-site/logging symmetry (an
    unused parameter is cheaper than churning every caller's signature again
    if a future per-symbol gate returns). The ONE code-side size bound left
    is SOLVENCY, enforced by sizing.py at actual entry-sizing time, not here.

    Entry window: BOTH ends inclusive — `entry_start_et <= now.time() <=
    entry_end_et` (e.g. exactly 09:45:00 and exactly 15:15:00 both pass).
    """
    reasons: list[str] = []

    if state.hard_halted:
        reasons.append(f"hard_halt:{state.halt_reason or 'unknown'}")
    if halt_file_present:
        reasons.append("halt_file")

    entry_start = parse_hhmm(cfg.entry_start_et)
    entry_end = parse_hhmm(cfg.entry_end_et)
    t = now_et.time()
    if not (entry_start <= t <= entry_end):
        reasons.append("outside_entry_window")

    return GateResult(allowed=(len(reasons) == 0), reasons=reasons)


def record_entry(state: RiskState, symbol: str, now_epoch: float) -> RiskState:
    """Bumps entries_today and entries_by_symbol[symbol]. `now_epoch` is accepted
    for signature/caller-logging symmetry with record_close but RiskState has no
    per-entry timestamp field to store it in today.
    """
    new_by_symbol = dict(state.entries_by_symbol)
    new_by_symbol[symbol] = new_by_symbol.get(symbol, 0) + 1
    return replace(state, entries_today=state.entries_today + 1, entries_by_symbol=new_by_symbol)


def merge_entry_counters(cycle_risk: RiskState, ledger_risk: RiskState) -> RiskState:
    """Carry entry counters recorded mid-cycle into the cycle's own RiskState (DPOF-1, S211).

    run_cycle rolls its RiskState at step 7, BEFORE the entry pipeline. _apply_entry_outcome then
    bumps entries_today/entries_by_symbol on the LEDGER's risk and saves it -- but the cycle-end
    save_risk(cycle_risk) wrote the pre-entry counters straight back (entries_today read 0 all day).
    Only the entry counters are merged (max, per symbol too): they only ever increase within a day,
    and nothing else (halt state, P&L, cooldown) may be resurrected from the ledger copy. A day
    mismatch means a rollover happened in between, so the cycle's own state wins untouched.
    """
    if cycle_risk.day != ledger_risk.day:
        return cycle_risk
    by_symbol = dict(cycle_risk.entries_by_symbol)
    for symbol, count in ledger_risk.entries_by_symbol.items():
        by_symbol[symbol] = max(by_symbol.get(symbol, 0), count)
    return replace(
        cycle_risk,
        entries_today=max(cycle_risk.entries_today, ledger_risk.entries_today),
        entries_by_symbol=by_symbol,
    )


def record_close(
    state: RiskState,
    symbol: str,
    realized_pnl: float,
    now_epoch: float,
) -> RiskState:
    """P&L accounting only — realized_pnl_today/_week/_total accumulate the
    close. No loss-streak counter, no cooldown, no per-symbol last-stop
    bookkeeping any more (operator directive: no mechanical breakers/
    cooldowns); `symbol`/`now_epoch` are accepted for call-site/logging
    symmetry but RiskState has no per-symbol field left to write them into.
    """
    return replace(
        state,
        realized_pnl_today=state.realized_pnl_today + realized_pnl,
        realized_pnl_week=state.realized_pnl_week + realized_pnl,
        realized_pnl_total=state.realized_pnl_total + realized_pnl,
    )


def operator_halt(state: RiskState, reason: str) -> RiskState:
    return replace(state, hard_halted=True, halt_reason=reason)


def operator_resume(state: RiskState, equity: float | None = None) -> RiskState:
    """Clears hard_halted, daily_halted, cooldown_until, halt_reason — the
    ONLY halt left is this manual operator one, so resume no longer has to
    out-run a breaker re-tripping on the same stale baseline the instant it
    clears (there is no breaker any more).

    With `equity` known (the DPOF-scoped equity — see engine._dpof_equity(),
    not necessarily the shared account's raw equity), also re-baselines
    start-of-day/week equity, the high-water mark, and realized
    day/week/total P&L -> 0.0 — a re-baseline is an explicit operator
    decision that DPOF's virtual capital now starts fresh from this equity,
    not a continuation of the old cumulative P&L track. Entry counters and
    day/week keys are untouched. Resume is an explicit operator decision to
    accept the current equity as the new baseline.
    """
    new_state = replace(state, hard_halted=False, daily_halted=False, cooldown_until=None, halt_reason="")
    if equity is not None and equity > 0:
        new_state = replace(
            new_state,
            start_of_day_equity=equity,
            start_of_week_equity=equity,
            hwm_equity=equity,
            realized_pnl_today=0.0,
            realized_pnl_week=0.0,
            realized_pnl_total=0.0,
        )
    return new_state
