"""
CLIO DPOF — follow-ups F2: trade-quality statistics (pure).

Task ORCH-7Q4K-DPOF builder B2, 2026-09-26. Design frame:
docs/design/clio_dpof_followups_2026-09-26.md §4 (also see CLAUDE.md's
"DPOF / Repo Gotchas" section).

Pure, stdlib-only textbook formulas over closed PositionRecords: win rate,
profit factor, expectancy, drawdown. No I/O, no third-party imports -- every
number here is a plain sum/mean/ratio a reader can hand-check against the
ledger.

Two exclusions matter, both "never a fake number, never inf" rules
(CLAUDE.md anti-pattern #1, "no fake zeros"):

  - `is_real_trade`: a position that never really traded (log-only run, an
    order sent but never confirmed filled, or a reconciliation that just
    discovered the broker doesn't have it) has nothing to say about trade
    quality. Mirrors `agent.dpof.reflection`'s `_EXCLUDED_EXIT_REASONS` /
    `_EXCLUDED_EXIT_REASON_PREFIXES` exactly (kept as a LOCAL copy, not an
    import -- the same module-boundary discipline reflection.py's own
    docstring describes: each module keeps its own tiny copy rather than
    cross-importing another module's private constants. A parity test in
    tests/dpof/test_trade_stats.py pins the two lists equal, so a future
    edit to one side can't silently drift from the other).
  - `pnl_incomplete`: a position carrying a SYNTHETIC reconcile fill
    (`agent.dpof.models.RECONCILE_FILL_PURPOSES`, from
    docs/design/clio_dpof_followups_2026-09-26.md §3's F1) has an
    admittedly-unknown price on at least one fill, so its `realized_pnl`
    can be silently wrong. Such a position still counts toward `n` (it
    really did trade) but is excluded from every dollar-denominated
    statistic -- counted once in `incomplete` instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from agent.dpof.models import RECONCILE_FILL_PURPOSES

if TYPE_CHECKING:
    from agent.dpof.models import PositionRecord

# Local copy of agent.dpof.reflection's exclusion sets (module-boundary
# discipline -- see module docstring above). tests/dpof/test_trade_stats.py
# asserts these stay equal to reflection.py's own constants.
_EXCLUDED_EXIT_REASONS = ("log_only", "submit_unconfirmed", "reconciled_absent")
_EXCLUDED_EXIT_REASON_PREFIXES = ("submit_failed:",)


def is_real_trade(pos: "PositionRecord") -> bool:
    """A position that actually closed with a known, broker-confirmed fill
    history -- not a log-only run, an unconfirmed/failed submission, or a
    reconciliation that found the broker never had it."""
    if pos.status != "closed":
        return False
    if not (pos.qty_original > 0):
        return False
    if pos.entry_price is None:
        return False
    if pos.closed_time is None:
        return False
    reason = pos.exit_reason or ""
    if reason in _EXCLUDED_EXIT_REASONS:
        return False
    if any(reason.startswith(p) for p in _EXCLUDED_EXIT_REASON_PREFIXES):
        return False
    return True


def pnl_incomplete(pos: "PositionRecord") -> bool:
    """True if ANY order_fills entry on this position is a synthetic
    reconcile fill (avg=None, unknown price) -- realized_pnl may therefore
    be understated/overstated versus the position's true economics."""
    fills = pos.order_fills or {}
    for v in fills.values():
        if isinstance(v, dict) and v.get("purpose") in RECONCILE_FILL_PURPOSES:
            return True
    return False


def compute_trade_stats(positions: "list[PositionRecord]") -> dict:
    """Trade-quality statistics over `positions`' real, P&L-complete closed
    trades. Every dollar-denominated field is None when its inputs are
    empty -- never a fake 0.0 or float('inf') (CLAUDE.md anti-pattern #1).
    Counts (`n`, `wins`, `losses`, `scratches`, `incomplete`, `n_complete`)
    are always plain ints -- 0 is a genuine fact ("zero real trades"), not a
    stand-in for missing data."""
    real = [p for p in positions if is_real_trade(p)]
    n = len(real)
    complete = [p for p in real if not pnl_incomplete(p)]
    n_complete = len(complete)
    incomplete = n - n_complete

    wins = [p for p in complete if p.realized_pnl > 0]
    losses = [p for p in complete if p.realized_pnl < 0]
    scratches = [p for p in complete if p.realized_pnl == 0]
    n_wins = len(wins)
    n_losses = len(losses)
    n_scratches = len(scratches)

    win_rate: Optional[float] = (n_wins / (n_wins + n_losses)) if (n_wins + n_losses) > 0 else None

    net_pnl: Optional[float] = sum(p.realized_pnl for p in complete) if n_complete > 0 else None

    gross_win = sum(p.realized_pnl for p in wins) if n_wins > 0 else 0.0
    gross_loss = sum(p.realized_pnl for p in losses) if n_losses > 0 else 0.0  # strictly < 0 when non-empty

    avg_win: Optional[float] = (gross_win / n_wins) if n_wins > 0 else None
    avg_loss: Optional[float] = (gross_loss / n_losses) if n_losses > 0 else None
    # Never inf: an empty-loss ratio is undefined (None), not infinite.
    profit_factor: Optional[float] = (gross_win / abs(gross_loss)) if n_losses > 0 else None

    expectancy: Optional[float] = (net_pnl / n_complete) if (net_pnl is not None and n_complete > 0) else None

    best = max((p.realized_pnl for p in complete), default=None)
    worst = min((p.realized_pnl for p in complete), default=None)

    if n_complete == 0:
        max_drawdown: Optional[float] = None
    else:
        ordered = sorted(complete, key=lambda p: p.closed_time)
        cum = 0.0
        peak = 0.0
        worst_fall = 0.0
        for p in ordered:
            cum += p.realized_pnl
            if cum > peak:
                peak = cum
            fall = peak - cum
            if fall > worst_fall:
                worst_fall = fall
        max_drawdown = -worst_fall if worst_fall > 0 else 0.0

    return {
        "n": n,
        "wins": n_wins,
        "losses": n_losses,
        "scratches": n_scratches,
        "win_rate": win_rate,
        "net_pnl": net_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "best": best,
        "worst": worst,
        "max_drawdown": max_drawdown,
        "incomplete": incomplete,
        "n_complete": n_complete,
    }


def _dollar(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    if round(v) == 0:
        return "$0"
    sign = "+" if v > 0 else "-"
    return f"{sign}${abs(v):,.0f}"


def _pct(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    return f"{v:.0%}"


def _ratio(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    return f"{v:.1f}"


def render_trade_stats(label: str, stats: dict) -> str:
    """Compact plain-text summary for one window (e.g. `label="today"` or
    `label="since 2026-06-01"`). `n/a` for every None field. Plain text
    only -- no markup (this feeds Telegram, which forbids parse_mode with
    dynamic content, and the EOD digest, which is plain text throughout).

    Example: "Trades today: 4 (3W 1L) | win 75% | net +$212 | PF 3.4 |
    exp +$53/trade | avg W +$96 L -$76 | maxDD -$76", with a second line
    "(1 excluded: P&L incomplete after reconcile)" when `incomplete` > 0.
    """
    n = stats.get("n", 0)
    wins = stats.get("wins", 0)
    losses = stats.get("losses", 0)
    scratches = stats.get("scratches", 0)
    incomplete = stats.get("incomplete", 0)

    counts = f"{wins}W {losses}L" + (f" {scratches}S" if scratches else "")
    bits = [
        f"Trades {label}: {n} ({counts})",
        f"win {_pct(stats.get('win_rate'))}",
        f"net {_dollar(stats.get('net_pnl'))}",
        f"PF {_ratio(stats.get('profit_factor'))}",
        "exp " + (
            f"{_dollar(stats['expectancy'])}/trade" if stats.get("expectancy") is not None else "n/a"
        ),
        f"avg W {_dollar(stats.get('avg_win'))} L {_dollar(stats.get('avg_loss'))}",
        f"maxDD {_dollar(stats.get('max_drawdown'))}",
    ]
    line = " | ".join(bits)
    if incomplete:
        line += f"\n({incomplete} excluded: P&L incomplete after reconcile)"
    return line
