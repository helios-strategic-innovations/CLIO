"""
CLIO flowcollector — trade-side classification (PURE).

Alpaca's OPRA option trades and SIP equity trades carry no aggressor/side
flag (confirmed in `scratchpad/recon/SELF_COMPUTE_FLOW.md` §1 against the
installed alpaca-py 0.44.0 `Trade`/`Quote` models and docs.alpaca.markets --
there is no such field anywhere in the schema). Every directional pillar this
package computes therefore depends on a DIY classifier: this module is that
classifier, and nothing else.

Method (standard market-microstructure methodology, not Alpaca-specific --
see recon doc §3): the **quote rule** is primary -- a trade printing at or
above the prevailing NBBO ask is buyer-initiated, at or below the bid is
seller-initiated. A trade printing strictly inside the spread (or with no
usable NBBO at all) falls back to the **tick test** (Lee & Ready 1991): an
uptick vs. the previous trade at this symbol is a buy, a downtick is a sell,
and a zero-tick carries forward the previous trade's classification. If
there is truly nothing to compare against (first trade ever seen for a
symbol, no NBBO, no previous trade), the result is "unknown" -- never
guessed, never defaulted to a side (this package's `aggregator.py` treats
"unknown"/"mid" trades as excluded from directional sums, exactly like
`agent.dpof.data`'s MISSING-IS-NEVER-ZERO discipline for pillar values).

Accuracy of this method on options specifically is UNVERIFIED for this exact
feed (recon doc §3: "~70-85% in academic literature for equities; options
literature is thinner") -- this needs empirical backtesting before the
resulting pillars are trusted for anything beyond the advisory shadow
logging this phase is scoped to.

No third-party classifier/microstructure code is copied here (see
CLAUDE.md-in-scope operator directive re: AGPL/unlicensed repos) -- the
quote rule and Lee-Ready tick test are decades-old, textbook, uncopyrightable
methodology, implemented from scratch against Alpaca's own Trade/Quote field
names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# ── Dark-pool tape code (confirmed: alpaca.data.enums.Exchange.D == "FINRA
#    ADF" -- see recon doc §2 and the installed alpaca-py source) ──────────
DARK_POOL_EXCHANGE = "D"

# Trade sides this module ever returns. "mid"/"unknown" are both
# "not directionally usable" but kept distinct for observability:
# "mid" means a real NBBO existed and the print landed exactly at the
# midpoint with no tick-test tiebreaker available; "unknown" means there was
# nothing at all to classify against.
SIDE_BUY = "buy"
SIDE_SELL = "sell"
SIDE_MID = "mid"
SIDE_UNKNOWN = "unknown"

# Default quote-rule tolerance in price units. Real-time NBBO and trade
# prices from the same feed should match exactly at the touch, but a small
# epsilon absorbs sub-cent float noise from msgpack/JSON round-tripping
# without meaningfully widening the "inside the spread" band that falls
# through to the tick test. Chosen, not measured -- UNVERIFIED against real
# tick data; override via FlowCollectorConfig if it proves too generous.
DEFAULT_QUOTE_EPSILON = 0.005

# UNVERIFIED (recon doc §2): only condition code "I" (odd lot) is directly
# confirmed by name in docs.alpaca.markets' FAQ; the full CTS/UTP condition
# table (docs.alpaca.markets/reference/stockmetaconditions-1) returned only
# partial content on the recon pass and was never completed. This default
# set is therefore deliberately MINIMAL (only the one confirmed code) rather
# than a guessed larger exclusion list -- a wrong guess here would silently
# discard good trades. Extend only after fetching and confirming the actual
# table; a caller may always pass its own `exclude` set to
# `is_reliable_trade`.
DEFAULT_UNRELIABLE_CONDITIONS = frozenset({"I"})


@dataclass(frozen=True)
class NBBO:
    """A single best-bid/best-ask snapshot for one symbol at one instant."""

    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    ts: Optional[float] = None  # epoch seconds

    @property
    def usable(self) -> bool:
        return (
            self.bid is not None
            and self.ask is not None
            and self.bid > 0
            and self.ask > 0
            and self.ask >= self.bid
        )

    @property
    def mid(self) -> Optional[float]:
        if not self.usable:
            return None
        return (self.bid + self.ask) / 2.0


class NBBOCache:
    """In-memory, thread-safe per-symbol latest-NBBO cache.

    Updated on every Quote message (from either the options or the stock
    stream -- one cache instance per asset class, kept separate by the
    caller); read by the classifier on every Trade message for the same
    symbol. No I/O, no persistence -- this is deliberately ephemeral,
    process-lifetime-only context, same as `agent.dpof.data.DPOFDataFeed`'s
    in-memory pillar caches.
    """

    def __init__(self) -> None:
        import threading

        self._lock = threading.Lock()
        self._by_symbol: dict[str, NBBO] = {}

    def update(self, symbol: str, nbbo: NBBO) -> None:
        with self._lock:
            self._by_symbol[symbol] = nbbo

    def get(self, symbol: str) -> Optional[NBBO]:
        with self._lock:
            return self._by_symbol.get(symbol)

    def snapshot(self) -> dict[str, NBBO]:
        with self._lock:
            return dict(self._by_symbol)

    def __len__(self) -> int:
        with self._lock:
            return len(self._by_symbol)


class LastPriceTracker:
    """Thread-safe per-symbol (last_price, last_side) tracker for the
    tick-test fallback. Deliberately separate from NBBOCache (a different
    lifecycle: updated on every classified Trade, not on Quotes)."""

    def __init__(self) -> None:
        import threading

        self._lock = threading.Lock()
        self._by_symbol: dict[str, tuple[float, str]] = {}

    def get(self, symbol: str) -> tuple[Optional[float], Optional[str]]:
        with self._lock:
            entry = self._by_symbol.get(symbol)
        return entry if entry is not None else (None, None)

    def update(self, symbol: str, price: float, side: str) -> None:
        with self._lock:
            self._by_symbol[symbol] = (price, side)


def tick_test(
    price: float,
    prev_price: Optional[float],
    prev_side: Optional[str],
    mid: Optional[float] = None,
) -> str:
    """PURE. Lee-Ready tick test: uptick=buy, downtick=sell, zero-tick
    carries the previous classification forward. If there is no previous
    price to compare against, falls back to a plain above/below-midpoint
    split when `mid` is available, else "unknown". Never guesses when
    neither input is available.
    """
    if prev_price is not None:
        if price > prev_price:
            return SIDE_BUY
        if price < prev_price:
            return SIDE_SELL
        if prev_side in (SIDE_BUY, SIDE_SELL):
            return prev_side
        # zero-tick with no known prior side -- fall through to midpoint
    if mid is not None:
        if price > mid:
            return SIDE_BUY
        if price < mid:
            return SIDE_SELL
        return SIDE_MID
    return SIDE_UNKNOWN


def classify_trade_side(
    price: float,
    nbbo: Optional[NBBO],
    prev_price: Optional[float] = None,
    prev_side: Optional[str] = None,
    epsilon: float = DEFAULT_QUOTE_EPSILON,
) -> str:
    """PURE. Quote rule first, Lee-Ready tick test fallback (see module
    docstring). `epsilon` widens "at the ask"/"at the bid" by a small price
    tolerance so sub-cent float noise doesn't spuriously fall through to the
    tick test.
    """
    if nbbo is not None and nbbo.usable:
        if price >= nbbo.ask - epsilon:
            return SIDE_BUY
        if price <= nbbo.bid + epsilon:
            return SIDE_SELL
        return tick_test(price, prev_price, prev_side, mid=nbbo.mid)
    return tick_test(price, prev_price, prev_side, mid=None)


def classify_dark_trade_side(price: float, mid: Optional[float]) -> str:
    """PURE. Dark-pool (FINRA ADF / exchange "D") prints are frequently
    negotiated/crossed and often execute AT or very near the NBBO midpoint,
    which makes the ask/bid quote rule a poor fit for them specifically (it
    would classify most dark prints as "inside the spread" and punt to the
    tick test, discarding the one signal dark prints actually carry: where
    they sit relative to the public midpoint at print time -- recon doc §2).
    This is therefore a SEPARATE, simpler classifier used only for D-tape
    equity trades in `aggregator.compute_darkpool`: strictly above the
    midpoint is a buy, strictly below is a sell, exactly at it is "mid".
    Documented design choice, not Alpaca- or UW-specified.
    """
    if mid is None:
        return SIDE_UNKNOWN
    if price > mid:
        return SIDE_BUY
    if price < mid:
        return SIDE_SELL
    return SIDE_MID


def is_reliable_trade(
    conditions: Optional[list[str]],
    exclude: frozenset[str] = DEFAULT_UNRELIABLE_CONDITIONS,
) -> bool:
    """PURE. True unless `conditions` contains one of `exclude`'s codes. An
    empty/None conditions list is treated as reliable (nothing to exclude
    on) -- see DEFAULT_UNRELIABLE_CONDITIONS's docstring for why this set is
    deliberately minimal/UNVERIFIED rather than a larger guessed table.
    """
    if not conditions:
        return True
    return not any(c in exclude for c in conditions)


@dataclass(frozen=True)
class TradeEvent:
    """A single classified trade, normalized across the options and stock
    streams (see collector.py's handlers for construction). `right` is ""
    for equity trades. `underlying` is the OCC root for options, the symbol
    itself for equities.

    `mid_at_ingest`: the SIP NBBO midpoint captured AT THE MOMENT this trade
    was processed (ingest time -- see collector.py's `_process_stock_trade`),
    not re-derived from the live NBBO cache later at aggregation/snapshot
    time. This matters specifically for `aggregator.compute_darkpool`: a
    dark (D-tape) print can sit anywhere in a 5/15/30-min rolling window, and
    classifying it against "whatever the midpoint happens to be right now"
    (snapshot time) rather than "the midpoint when it actually printed"
    would misclassify trades near the edge of a window during a moving
    market. Capturing it once, cheaply, at ingest (we already fetch the
    live NBBO cache there for the lit-tape quote-rule `side` classification)
    avoids that staleness gap entirely. None if no usable NBBO existed yet
    at ingest time."""

    ts: float
    symbol: str
    underlying: str
    right: str  # "call" | "put" | ""
    strike: Optional[float]
    exchange: str
    price: float
    size: float
    side: str  # SIDE_BUY | SIDE_SELL | SIDE_MID | SIDE_UNKNOWN
    conditions: tuple[str, ...] = ()
    reliable: bool = True
    is_sweep: bool = False
    mid_at_ingest: Optional[float] = None


def find_sweeps(
    events: list[TradeEvent],
    window_seconds: float = 0.5,
    min_exchanges: int = 2,
) -> set[int]:
    """PURE. Sweep heuristic (recon doc §3, not UW's proprietary label):
    same (underlying, strike, right, side) trade printing across
    `min_exchanges`+ distinct exchange codes within `window_seconds` of each
    other. `events` need not be pre-sorted or pre-grouped. Returns the SET
    OF INDICES (into `events`, as given) that participate in at least one
    such cluster -- never mutates `events` (TradeEvent is frozen); the
    caller applies `is_sweep=True` when constructing/storing the final
    record. O(n log n) via a per-group sort + sliding window, not O(n^2).
    """
    groups: dict[tuple, list[int]] = {}
    for i, e in enumerate(events):
        if e.side not in (SIDE_BUY, SIDE_SELL):
            continue  # a sweep is a directional cluster; undirected prints don't qualify
        key = (e.underlying, e.strike, e.right, e.side)
        groups.setdefault(key, []).append(i)

    swept: set[int] = set()
    for idxs in groups.values():
        idxs_sorted = sorted(idxs, key=lambda i: events[i].ts)
        n = len(idxs_sorted)
        left = 0
        for right in range(n):
            while events[idxs_sorted[right]].ts - events[idxs_sorted[left]].ts > window_seconds:
                left += 1
            window = idxs_sorted[left : right + 1]
            if len({events[i].exchange for i in window}) >= min_exchanges:
                swept.update(window)
    return swept
