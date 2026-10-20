"""
CLIO flowcollector — universe selection (mostly PURE).

Picks the bounded SPY/QQQ 0-2DTE, ±band% option-contract universe this
collector subscribes to. Kept as plain functions over plain dicts (never
alpaca-py SDK objects) so it is testable with zero network/SDK imports;
`collector.py` is the only place that talks to alpaca-py and converts SDK
`OptionContract` objects into the dicts these functions consume.

Sizing rationale (recon doc §1, "Contract-count / volume estimate"): a
±4%/0-2DTE SPY+QQQ band is estimated at roughly 700-800 contract symbols,
comfortably under Algo Trader Plus's documented 1000-quotes-per-connection
cap if that cap is quotes-only, borderline if it is a combined
trades+quotes cap (UNVERIFIED either way per the recon doc). `partition_symbols`
below exists so the caller can split across multiple WebSocket connections
if the live-measured universe size warrants it -- this module never assumes
which case is true.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Optional


def dte(expiration: date, today: date) -> int:
    """PURE. Days to expiry, can be negative for an already-past expiration
    (the caller filters those out; this function itself never clamps)."""
    return (expiration - today).days


def market_holidays() -> frozenset:
    """agent.integrations.market_events.MARKET_HOLIDAYS, imported lazily;
    weekends-only (empty set) if that import fails. Same degrade rule as
    agent.dpof.contracts.target_expiries."""
    try:
        from agent.integrations.market_events import MARKET_HOLIDAYS

        return frozenset(MARKET_HOLIDAYS)
    except Exception:
        return frozenset()


def _is_session(d: date, holidays: frozenset) -> bool:
    return d.weekday() < 5 and d not in holidays


def trading_dte(expiration: date, today: date, holidays: frozenset = frozenset()) -> int:
    """PURE. Expiry distance in TRADING SESSIONS, the way DPOF counts 1DTE
    (agent.dpof.contracts.target_expiries): 0 = expires today, 1 = the next
    session (Friday -> Monday), ...; -1 for an already-past expiration.
    Calendar days would make max_dte=2 on a Friday cover only Fri/Sat/Sun,
    i.e. 0DTE contracts alone -- the 2026-09-25 universe (Alpaca returns no
    greeks for same-day expiries, so it had none at all)."""
    days = (expiration - today).days
    if days < 0:
        return -1
    if days > 60:
        return days  # far expiries: exact session count doesn't matter here
    n = 0
    d = today
    for _ in range(days):
        d += timedelta(days=1)
        if _is_session(d, holidays):
            n += 1
    return n


def session_date_ahead(today: date, sessions: int, holidays: frozenset = frozenset()) -> date:
    """PURE. Calendar date of the `sessions`-th trading session after
    `today` (0 -> today itself). Used for the server-side expiration_date_lte
    so the REST pull covers the same trading-session window select_universe
    keeps."""
    d = today
    n = 0
    while n < max(0, sessions):
        d += timedelta(days=1)
        if _is_session(d, holidays):
            n += 1
    return d


def in_band(strike: float, spot: float, band_pct: float) -> bool:
    """PURE. True if `strike` sits within +/- band_pct of `spot`. A
    non-positive spot/band never matches anything (defensive -- a caller
    passing a missing spot price should get an empty universe, not a
    band computed off a bogus 0)."""
    if spot is None or spot <= 0 or band_pct is None or band_pct < 0:
        return False
    lo = spot * (1.0 - band_pct)
    hi = spot * (1.0 + band_pct)
    return lo <= strike <= hi


def _parse_expiration(raw: Any) -> Optional[date]:
    if isinstance(raw, date):
        return raw
    if isinstance(raw, str) and raw:
        try:
            return date.fromisoformat(raw[:10])
        except ValueError:
            return None
    return None


def _parse_float(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def select_universe(
    contracts: list[dict],
    spot: Optional[float],
    band_pct: float,
    max_dte: int,
    today: date,
    holidays: frozenset = frozenset(),
) -> list[dict]:
    """PURE. Filters `contracts` (each a dict with at least "expiration_date"
    and "strike_price"/"strike") to those expiring within [0, max_dte]
    TRADING SESSIONS of `today` (see trading_dte) and within `band_pct` of
    `spot`. Malformed entries (unparsable
    expiration/strike) are skipped, never raise. Returns [] (not None) for
    "no spot yet" / "nothing matched" -- both are legitimate empty results,
    never confused with a fetch failure (that distinction is the caller's
    job, same MISSING-IS-NEVER-ZERO discipline as agent.dpof.data)."""
    if spot is None or spot <= 0:
        return []
    out: list[dict] = []
    for c in contracts or []:
        if not isinstance(c, dict):
            continue
        exp = _parse_expiration(c.get("expiration_date") or c.get("expiration"))
        if exp is None:
            continue
        d = trading_dte(exp, today, holidays)
        if d < 0 or d > max_dte:
            continue
        strike = _parse_float(c.get("strike_price") if c.get("strike_price") is not None else c.get("strike"))
        if strike is None:
            continue
        if not in_band(strike, spot, band_pct):
            continue
        out.append(c)
    return out


def needs_universe_refresh(
    last_refresh_epoch: Optional[float],
    now_epoch: float,
    refresh_interval_seconds: float,
    last_spot: Optional[float],
    current_spot: Optional[float],
    drift_pct: float,
) -> bool:
    """PURE. True if the universe has never been built, the periodic cadence
    is due, or the underlying's spot has drifted by >= drift_pct since the
    last refresh (the universe is a strike BAND around spot -- a stale band
    silently stops covering the money as price moves). A missing spot never
    forces a refresh by itself (nothing to compare); the cadence check alone
    still applies."""
    if last_refresh_epoch is None or (now_epoch - last_refresh_epoch) >= refresh_interval_seconds:
        return True
    if last_spot is None or current_spot is None or last_spot <= 0:
        return False
    drift = abs(current_spot - last_spot) / last_spot
    return drift >= drift_pct


def partition_symbols(symbols: list[str], cap: int) -> list[list[str]]:
    """PURE. Splits `symbols` into chunks of at most `cap` each, preserving
    order. `cap <= 0` is treated as "no cap" (one chunk) -- defensive, never
    raises or infinite-loops on a misconfigured 0."""
    symbols = list(symbols or [])
    if cap is None or cap <= 0:
        return [symbols] if symbols else []
    return [symbols[i : i + cap] for i in range(0, len(symbols), cap)]


@dataclass
class UniverseState:
    """Plain snapshot of the currently-selected universe, held by
    collector.FlowCollectorService. Kept here (not in collector.py) so
    tests can construct/compare it without importing collector.py's
    network-adjacent code."""

    symbols_by_underlying: dict[str, list[str]]
    contracts_by_symbol: dict[str, dict]
    built_at: float
    spot_by_underlying: dict[str, float]

    @property
    def all_symbols(self) -> list[str]:
        out: list[str] = []
        for syms in self.symbols_by_underlying.values():
            out.extend(syms)
        return out

    @property
    def total_count(self) -> int:
        return len(self.all_symbols)


def build_universe_state(
    per_underlying_contracts: dict[str, list[dict]],
    spot_by_underlying: dict[str, float],
    band_pct: float,
    max_dte: int,
    today: date,
    now_epoch: float,
    holidays: frozenset = frozenset(),
) -> UniverseState:
    """PURE. Applies select_universe() per underlying and assembles a
    UniverseState. `per_underlying_contracts` values are the RAW contract
    lists as fetched (already scoped to the right underlying_symbols/DTE
    range server-side, per collector.py's request -- this is a second,
    client-side belt-and-suspenders filter, not the only one)."""
    symbols_by_underlying: dict[str, list[str]] = {}
    contracts_by_symbol: dict[str, dict] = {}
    for underlying, contracts in (per_underlying_contracts or {}).items():
        spot = spot_by_underlying.get(underlying)
        matched = select_universe(contracts, spot, band_pct, max_dte, today, holidays)
        syms: list[str] = []
        for c in matched:
            sym = c.get("symbol")
            if not sym:
                continue
            syms.append(sym)
            contracts_by_symbol[sym] = c
        symbols_by_underlying[underlying] = syms
    return UniverseState(
        symbols_by_underlying=symbols_by_underlying,
        contracts_by_symbol=contracts_by_symbol,
        built_at=now_epoch,
        spot_by_underlying=dict(spot_by_underlying or {}),
    )
