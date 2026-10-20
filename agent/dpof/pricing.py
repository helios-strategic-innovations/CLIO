"""
CLIO DPOF — limit-price helpers (pure).

Entry leans toward the ask for fill probability but is hard-capped by a
slippage ceiling; exit concedes toward the bid but never below it. Prices
are per unit (x100 for a contract's dollar value), rounded to cents.
"""

from __future__ import annotations

from typing import Optional

from agent.dpof.config import DPOFConfig

_MIN_TICK = 0.01


def _round_cents(x: float) -> float:
    return round(round(x / _MIN_TICK) * _MIN_TICK, 2)


def entry_limit_price(bid: Optional[float], ask: Optional[float], cfg: DPOFConfig) -> Optional[float]:
    """mid + lean*spread, capped at mid*(1+max_slippage). None if the quote is unusable."""
    if bid is None or ask is None or ask <= 0 or bid < 0 or ask < bid:
        return None
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return None
    price = mid + cfg.entry_lean * (ask - bid)
    price = min(price, mid * (1.0 + cfg.entry_max_slippage_pct), ask)
    return max(_MIN_TICK, _round_cents(price))


def exit_limit_price(bid: Optional[float], ask: Optional[float], cfg: DPOFConfig) -> Optional[float]:
    """mid - lean*spread, floored at the bid. None if the quote is unusable."""
    if bid is None or ask is None or ask <= 0 or bid < 0 or ask < bid:
        return None
    mid = (bid + ask) / 2.0
    price = max(mid - cfg.exit_lean * (ask - bid), bid)
    return max(_MIN_TICK, _round_cents(price))


def reprice_entry(bid: Optional[float], ask: Optional[float], attempt: int, cfg: DPOFConfig) -> Optional[float]:
    """Walk an unfilled entry toward the ask, one step per reprice, never past the slippage cap."""
    base = entry_limit_price(bid, ask, cfg)
    if base is None:
        return None
    mid = (bid + ask) / 2.0
    cap = min(mid * (1.0 + cfg.entry_max_slippage_pct), ask)
    steps = max(1, cfg.entry_max_reprices)
    frac = min(1.0, max(0, attempt) / steps)
    return max(_MIN_TICK, _round_cents(base + (cap - base) * frac))


def reprice_exit(bid: Optional[float], ask: Optional[float], attempt: int, cfg: DPOFConfig) -> Optional[float]:
    """Walk an unfilled exit toward the bid, reaching the bid on the last reprice."""
    base = exit_limit_price(bid, ask, cfg)
    if base is None:
        return None
    steps = max(1, cfg.exit_max_reprices)
    frac = min(1.0, max(0, attempt) / steps)
    return max(_MIN_TICK, _round_cents(base - (base - bid) * frac))
