"""
CLIO DPOF — the one kept exit rail (PURE).

Operator directive (2026-09-24): NO mechanical trading. Stop-loss, take-
profit tiers, breakeven, trailing stop, max-hold timer, thesis-vote auto-
exit, and the standalone expiry-day guard are all GONE — exits (like
entries) are judged by CLIO's decision model (see engine.py's position-
review pipeline / agent.dpof.decider.ReviewDecision). The ONLY exit this
module still enforces in code is the forced flatten of every DPOF position
at `cfg.flatten_et` (15:50 ET by default; half-days clamp via engine.py's
`_session_cfg`, which moves `flatten_et` itself to close-10min before this
function ever sees it) — the "one kept rail" from docs/design/clio_dpof.md.

No model is EVER consulted for this decision: it is wall-clock arithmetic
over the position record and `now_et` (passed in, never read).
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, time
from typing import Optional

from agent.dpof.config import DPOFConfig
from agent.dpof.models import ExitDecision, PositionRecord

logger = logging.getLogger("clio.dpof.exits")

# OCC symbol: ROOT + YYMMDD + C/P + strike*1000 (8 digits), e.g. SPY261002C00575000.
_OCC_RE = re.compile(r"^[A-Z]+(\d{2})(\d{2})(\d{2})[CP]\d{8}$")


def position_pnl_pct(entry_price: Optional[float], mark: Optional[float]) -> Optional[float]:
    """Fractional P&L on the option's premium. None if either input is unusable."""
    if entry_price is None or entry_price <= 0 or mark is None:
        return None
    return (mark - entry_price) / entry_price


def parse_occ_expiry(symbol: str) -> Optional[date]:
    """Extract the OCC yymmdd expiry as a date. None if the symbol doesn't parse.

    2-digit year is treated as 2000+yy — every DPOF contract traded by this
    system is a near-dated 2020s/2030s option, so there is no century
    ambiguity to resolve here (unlike a general-purpose OCC parser).
    """
    m = _OCC_RE.match(symbol)
    if not m:
        return None
    yy, mm, dd = (int(g) for g in m.groups())
    try:
        return date(2000 + yy, mm, dd)
    except ValueError:
        return None


def position_expiry(pos: PositionRecord) -> Optional[date]:
    """Expiry date of the position's (first) leg, if it parses as OCC. Used by
    the engine to show days-to-expiry in review context — informational only,
    never a gate."""
    for leg in pos.legs:
        symbol = leg.get("symbol") if isinstance(leg, dict) else None
        if not symbol:
            continue
        exp = parse_occ_expiry(symbol)
        if exp is not None:
            return exp
    return None


def _parse_hhmm(s: str) -> time:
    hh, mm = s.split(":")
    return time(int(hh), int(mm))


def evaluate_exit(
    pos: PositionRecord,
    mark: Optional[float],
    now_et: datetime,
    cfg: DPOFConfig,
) -> ExitDecision:
    """The ONLY exit rule left: EOD flatten.

        flatten_eod and now_et.time() >= flatten_et  -> close ALL qty_open, urgent
        else                                          -> hold (no mechanical exit)

    `mark` is accepted for signature symmetry with the old exit ladder (and
    because a caller may want it for logging) but is not read here — the
    flatten fires purely on the wall clock, regardless of P&L or whether a
    mark is even available.
    """
    if cfg.flatten_eod and now_et.time() >= _parse_hhmm(cfg.flatten_et):
        return ExitDecision(action="close", qty=pos.qty_open, reason="eod_flatten", urgent=True)
    return ExitDecision(action="hold", qty=0, reason="no_exit_condition_met")
