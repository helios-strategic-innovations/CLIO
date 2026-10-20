"""
CLIO DPOF — position sizing (PURE).

Operator directive (2026-09-24, no-hard-caps amendment): sizing is the
decision model's call, not code's. There is no per-trade premium cap, no
open-premium cap, no max-contracts cap, no risk_pct band. The decision
model (`EntryDecision.qty`) picks the contract count directly; the ONLY
code-side bound left is SOLVENCY — DPOF can never be asked to commit more
premium than it actually has available:

    capital_available = dpof_equity - open_premium_committed
    max_affordable_qty = floor(capital_available / (unit_debit * 100))

`open_premium_committed` already includes PENDING entries (priced at their
working `entry_limit_price` — see engine.py's `_position_premium`), so a
still-unfilled entry counts against solvency exactly like a filled one.
`max_affordable_qty` is used twice: as `EntryCandidate.max_affordable_qty`
(informational context offered to the decision model at deliberation time)
and, again, freshly recomputed, to CLAMP the decider's chosen `qty` at
actual-entry time (equity/open-premium may have moved during the model's
10-60s latency) — `clamp_to_solvency` does that second job and reports why
it clamped. qty is a floor division and a hard clamp: this NEVER rounds up
or lets a decision spend more than DPOF actually has (the Hydrinia
`max(1, int(...))` lesson, generalized).
"""

from __future__ import annotations

import logging
import math
from typing import Optional

logger = logging.getLogger("clio.dpof.sizing")

# Guards against float round-trip noise (e.g. 0.1 + 0.2) nudging a budget
# exactly at a contract-cost boundary down by an epsilon. Negligible next to
# a dollar-denominated premium budget; never large enough to let qty exceed
# what the (uncapped) dollar budget actually supports.
_EPS = 1e-9


def max_affordable_qty(capital_available: Optional[float], unit_debit: Optional[float]) -> int:
    """floor(capital_available / (unit_debit*100)). 0 on any unusable input
    (None/non-finite/<=0) — never guesses, never rounds up."""
    if capital_available is None or not math.isfinite(capital_available) or capital_available <= 0:
        return 0
    if unit_debit is None or not math.isfinite(unit_debit) or unit_debit <= 0:
        return 0
    contract_cost = unit_debit * 100.0
    return int((capital_available + _EPS) // contract_cost)


def clamp_to_solvency(
    qty: Optional[int], capital_available: Optional[float], unit_debit: Optional[float],
) -> tuple[int, list[str]]:
    """Clamps the decision model's chosen `qty` (an int; None/non-finite/<1
    -> 0) to what a FRESH solvency check actually allows. Returns (qty,
    reasons); qty=0 always carries at least one reason. This is the ONLY
    code-side size gate left (operator directive: no hard caps) — it exists
    solely so a decision can never spend premium DPOF doesn't have, not to
    second-guess the model's own risk judgment.
    """
    if qty is None or not isinstance(qty, (int, float)) or isinstance(qty, bool) or qty < 1:
        return 0, ["qty_below_one_contract"]
    qty = int(qty)

    max_qty = max_affordable_qty(capital_available, unit_debit)
    reasons: list[str] = []
    if qty > max_qty:
        reasons.append(f"clamped_by_solvency:{qty}->{max_qty}")
        qty = max_qty

    if qty < 1:
        reasons.append("insolvent")
        return 0, reasons

    return qty, reasons


# ── Advisory size anchor (operator 2026-09-29) ─────────────────────────────
# NOT a cap and NOT a rule: the no-hard-caps directive stands and the decision model still picks its own qty.
# The anchor is a reference scale shown next to each candidate so the model is not sizing in a vacuum. Recent
# entries landed at ~4-5% of the slice regardless of setup; the operator asked for ~10% of dpof_equity in
# premium at conviction 0.50 and ~20% at 0.80. The line is linear through those two points and clamped, and the
# anchor contracts are always bounded by `max_affordable_qty` (solvency), exactly like the model's own qty.
# No track-record shrinkage on purpose: with a handful of closed trades it would cancel the requested scale, and
# the model already sees its own recent closes and lessons.
ANCHOR_CONVICTIONS = (0.40, 0.50, 0.65, 0.80, 0.90)
ANCHOR_SHARE_FLOOR = 0.02
ANCHOR_SHARE_CEILING = 0.30
_ANCHOR_C_LO, _ANCHOR_C_HI = 0.50, 0.80


def _finite(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def anchor_share(conviction: object, share_c50: float, share_c80: float) -> Optional[float]:
    """Premium share of dpof_equity for a full entry at `conviction` (0..1): linear through
    (0.50, share_c50) and (0.80, share_c80), clamped to [ANCHOR_SHARE_FLOOR, ANCHOR_SHARE_CEILING].
    None on an unusable conviction or share (never a guess)."""
    if not (_finite(conviction) and _finite(share_c50) and _finite(share_c80)):
        return None
    if not 0.0 <= conviction <= 1.0:
        return None
    slope = (share_c80 - share_c50) / (_ANCHOR_C_HI - _ANCHOR_C_LO)
    return max(ANCHOR_SHARE_FLOOR, min(ANCHOR_SHARE_CEILING, share_c50 + (conviction - _ANCHOR_C_LO) * slope))


def anchor_qty(share: Optional[float], dpof_equity: object, unit_premium: object, max_qty: object) -> Optional[int]:
    """floor(share * dpof_equity / unit_premium), bounded by max_affordable_qty (>= 0). `unit_premium` is dollars
    per contract/spread (already x100). None on any unusable input."""
    if share is None or not (_finite(dpof_equity) and _finite(unit_premium) and _finite(max_qty)):
        return None
    if dpof_equity <= 0 or unit_premium <= 0 or share <= 0:
        return None
    contracts = int((share * dpof_equity + _EPS) // unit_premium)
    return max(0, min(contracts, int(max_qty)))


# ── Sizing telemetry for decision rows (S212 SZ-1) ─────────────────────────
# Every entry decision row (including Jev-vetoed and `no_size` ones) records what was asked, what was clamped,
# what it cost as a share of capital, what the advisory anchor suggested and how close Jev's `size_reckless`
# answer was to its threshold -- so sizing and CLIO_DPOF_JEV_SIZE_T can be tuned from data. Pure, None for anything
# unknown (never a fake zero), and it never influences a decision: the engine only WRITES it to the row.
def _r(v: object, nd: int) -> Optional[float]:
    return round(float(v), nd) if _finite(v) else None


def _int_or_none(v: object) -> Optional[int]:
    return int(v) if _finite(v) else None


def jev_size_reckless_noul(answers: object) -> Optional[float]:
    """The `size_reckless` noul from a JevVerdict.answers dict; None when absent/malformed (SHADOW, unavailable)."""
    if not isinstance(answers, dict):
        return None
    sr = answers.get("size_reckless")
    if isinstance(sr, dict) and sr.get("type") == "noul" and _finite(sr.get("noul")):
        return float(sr["noul"])
    return None


def size_telemetry(
    *,
    qty_requested: object,
    qty: object,
    unit_premium: object,
    dpof_equity: object,
    capital_available: object,
    conviction: object,
    anchor_shares: Optional[tuple] = None,
    jev_answers: object = None,
    jev_size_veto_threshold: object = None,
) -> dict:
    """The `sizing` block of an entry decision row. `unit_premium` is dollars per contract/spread (already x100);
    `anchor_shares` is (share_c50, share_c80) when the advisory anchor is enabled, else None."""
    q = _int_or_none(qty)
    premium = _r(q * unit_premium, 2) if (q is not None and _finite(unit_premium)) else None
    equity_ok = _finite(dpof_equity) and dpof_equity > 0
    afford = (
        max_affordable_qty(capital_available, unit_premium / 100.0)
        if (_finite(capital_available) and _finite(unit_premium) and unit_premium > 0) else None
    )
    if anchor_shares is None:
        anchor: dict = {"enabled": False}
    else:
        share = anchor_share(conviction, anchor_shares[0], anchor_shares[1])
        anchor = {
            "enabled": True,
            "share_c50": _r(anchor_shares[0], 4),
            "share_c80": _r(anchor_shares[1], 4),
            "share": _r(share, 4),
            "qty": anchor_qty(share, dpof_equity, unit_premium, afford) if afford is not None else None,
        }
    return {
        "qty_requested": _int_or_none(qty_requested),
        "qty": q,
        "unit_premium": _r(unit_premium, 2),
        "premium": premium,
        "premium_pct_of_capital": _r(premium / dpof_equity, 4) if (premium is not None and equity_ok) else None,
        "dpof_equity": _r(dpof_equity, 2),
        "capital_available": _r(capital_available, 2),
        "max_affordable_qty": afford,
        "conviction": _r(conviction, 4),
        "anchor": anchor,
        "jev_size_reckless_noul": jev_size_reckless_noul(jev_answers),
        "jev_size_veto_threshold": _r(jev_size_veto_threshold, 4),
    }
