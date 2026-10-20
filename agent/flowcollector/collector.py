"""
CLIO flowcollector — orchestrator (I/O + threading; this is the module
`python -m agent.flowcollector` runs).

Market-data READ-ONLY. This module never imports `agent.dpof.broker` /
`agent.dpof.orders` / `agent.dpof.engine`, never imports
`alpaca.trading.client`'s order-submission methods, and never calls
`submit_order`/`cancel_order`/`replace_order` anywhere -- the only Alpaca
trading-client call this package ever makes is the READ-ONLY
`get_option_contracts` (contract metadata + open interest), exactly the same
read `agent.dpof.broker.get_option_chain` makes (mirrored structurally, not
imported -- see `AlpacaFlowBroker`'s docstring). Grep-provable:
`grep -rn "submit_order\\|cancel_order\\|replace_order\\|dpof.broker\\|dpof.orders\\|dpof.engine" agent/flowcollector/`
returns nothing.

Threading model: alpaca-py's `DataStream.run()` is a blocking call that
internally does `asyncio.run(self._run_forever())` and already implements
its own reconnect-with-backoff loop against transient websocket errors
(`websockets.WebSocketException`) and connection staleness -- see
`scratchpad/recon/SELF_COMPUTE_FLOW.md`'s source citations of
`alpaca/data/live/websocket.py`. Each stream therefore runs in its OWN
daemon thread via `run_stream_supervisor`/`StreamSupervisor`, which is this
module's OUTER safety net: if a stream's `run()` call itself ever returns or
raises for a reason other than an explicit `stop()` (e.g. alpaca-py's own
"insufficient subscription" unrecoverable-error path, which makes
`_run_forever` return silently rather than loop), the supervisor rebuilds a
fresh stream object and reconnects with exponential backoff, without ever
giving up (unless `max_consecutive_failures` is configured > 0).

All classification/aggregation state (NBBOCache, LastPriceTracker,
Aggregator) is mutated from EXACTLY ONE place: `FlowCollectorService.tick()`,
running on this process's own asyncio loop. The stream handler coroutines
(`_on_option_trade` etc.), which run on a DIFFERENT thread's event loop
(the stream's own, inside `run_stream_supervisor`'s thread), never touch
that state directly -- they only push a normalized message onto a
`BoundedDropOldestQueue` (a cheap, thread-safe, lock-protected deque
operation), which `tick()` drains synchronously on its own thread. This is
the single-writer discipline that lets `Aggregator`/`NBBOCache` skip their
own locking (see those modules' docstrings) and keeps unbounded memory
growth impossible under a slow consumer or a fast producer burst (the queue
drops the OLDEST item and counts the drop, rather than growing without
bound -- CLAUDE.md/INTEL_V3_SPEC "no unbounded memory").
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Callable, Optional

from agent.flowcollector import greeks as greeks_mod
from agent.flowcollector.aggregator import Aggregator, StrikeExposure
from agent.flowcollector.classifier import (
    DARK_POOL_EXCHANGE,
    NBBO,
    LastPriceTracker,
    NBBOCache,
    TradeEvent,
    classify_trade_side,
    is_reliable_trade,
)
from agent.flowcollector.config import FlowCollectorConfig
from agent.flowcollector.state import (
    build_comparison_row,
    append_comparison_row,
    write_state_atomic,
    _read_uw_bridge_snapshot_sync,
)
from agent.flowcollector.universe import (
    UniverseState,
    build_universe_state,
    market_holidays,
    needs_universe_refresh,
    session_date_ahead,
)

logger = logging.getLogger("clio.flowcollector.collector")

# GetOptionContractsRequest.limit maxes out at (undocumented exactly, but
# comfortably >100) contracts per page; a full 0-2DTE SPY+/-4% band is
# estimated at ~400 contracts (recon doc §1), so this cap exists only as a
# defensive backstop against a misconfigured/blown-out band, not a normal
# operating limit -- get_option_universe logs a WARNING if it's ever hit.
_MAX_UNIVERSE_CONTRACTS = 5000

# The server-side strike_price_gte/lte filter (get_option_universe) is
# padded WIDER than the configured client-side band so a slightly-stale
# spot at fetch time can't clip strikes the client-side
# universe.select_universe filter would otherwise have kept. Chosen, not
# measured -- UNVERIFIED, like every other tanh-scale/band constant in this
# package (see aggregator.py's module docstring for the pattern).
_UNIVERSE_FETCH_BAND_PADDING_FACTOR = 1.5

# alpaca-py's OptionSnapshotRequest accepts a symbol list, but a request for
# the FULL 0-2DTE universe (~400-800 symbols) risks an oversized query
# string / URL length limit on some deployments -- chunk defensively. Each
# chunk's failure is independent (a 401/403 downgrades the feed and retries
# that chunk once; any other failure just skips that chunk, never dropping
# the OTHER chunks' data -- see get_option_greeks).
_OPTION_GREEKS_CHUNK_SIZE = 100

# Minimal, local OCC option-symbol parser (deliberately NOT imported from
# agent.dpof.broker._parse_occ or agent.integrations.alpaca_options -- this
# package stays independent of the DPOF/trading import surface entirely;
# see this module's own docstring). Standard OCC format: ROOT + YYMMDD +
# C/P + 8-digit strike*1000.
_OCC_RE = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")


def _parse_date(raw: Any) -> Optional[date]:
    """Best-effort ISO-date parse of a contract's `expiration_date` field
    (stored as a plain string by AlpacaFlowBroker.get_option_universe --
    see that method's `out.append(...)`), for greeks_mod.time_to_expiry_years.
    Deliberately local/minimal (not imported from universe.py's private
    `_parse_expiration`) -- same "kept independent" rationale as this
    module's own `_parse_occ` (see this module's docstring)."""
    if isinstance(raw, date):
        return raw
    if isinstance(raw, str) and raw:
        try:
            return date.fromisoformat(raw[:10])
        except ValueError:
            return None
    return None


def _t_memo(now_dt: datetime):
    """time_to_expiry_years(now_dt, expiry), memoized per expiry for one
    refresh pass. The universe holds ~3 expiries but hundreds of contracts;
    computing T (zoneinfo work) per contract on the event loop cost ~35 ms
    per underlying per recompute (pass-2 review R1 #4)."""
    cache: dict = {}

    def _T(expiry):
        if expiry not in cache:
            cache[expiry] = greeks_mod.time_to_expiry_years(now_dt, expiry)
        return cache[expiry]

    return _T


def _compute_greeks_batch_sync(jobs: list, cfg: FlowCollectorConfig) -> dict:
    """PURE-ish (stdlib math only, via greeks_mod). Runs OFF the event loop
    -- see maybe_refresh_strikes/maybe_recompute_greeks, which each call
    this through `loop.run_in_executor` -- because a batch of a few hundred
    BSM/implied-vol solves is enough real CPU work (item 3's "don't block
    its tick for >50 ms" bound) that it must never run directly on this
    process's own asyncio loop. Touches no collector/aggregator state --
    `jobs` is a plain list of
    (symbol, bid, ask, spot, strike, T, right, iv_hint) tuples the caller
    already assembled synchronously (cheap: dict lookups only, no math), and
    the result is a plain {symbol: {"iv":, "delta":, "gamma":}} dict the
    caller applies back to `UnderlyingFlowState` itself, single-writer, once
    this executor call returns. A job whose `compute_contract_greeks` result
    is None (unusable quote/inputs) is simply absent from the returned dict
    -- never a fabricated entry."""
    out: dict[str, dict] = {}
    for sym, bid, ask, spot, strike, T, right, iv_hint in jobs:
        result = greeks_mod.compute_contract_greeks(
            bid, ask, spot, strike, T, cfg.risk_free_rate, cfg.dividend_yield, right,
            iv_hint=iv_hint, max_spread_pct_for_iv=cfg.max_spread_pct_for_iv,
        )
        if result is not None:
            out[sym] = result
    return out


def _fill_smile_batch_sync(anchors: list, targets: list, spot: float, strike_lo: Optional[float], strike_hi: Optional[float], cfg: FlowCollectorConfig) -> dict:
    """PURE-ish sibling of `_compute_greeks_batch_sync`, same "runs OFF the
    event loop via `run_in_executor`, touches no collector/aggregator state"
    contract (see that function's docstring). `anchors`/`targets` are the
    plain-dict shapes `greeks_mod.fill_iv_from_smile` expects (each target
    additionally carries "T" for the BSM delta/gamma evaluation below, which
    fill_iv_from_smile itself doesn't need). For every target that gets a
    filled IV, evaluates BSM delta/gamma at that IV and runs the SAME
    `sanitize_greeks` bar as any other source -- an interpolated IV that
    happens to land somewhere numerically degenerate (e.g. right at a T
    floor) still can't produce an insane greek pair. Returns
    {symbol: {"iv", "delta", "gamma"}}, the same shape
    `_compute_greeks_batch_sync` returns, so both callers apply results
    identically (only the source LABEL differs: "interpolated" vs
    "computed")."""
    filled_iv = greeks_mod.fill_iv_from_smile(anchors, targets, spot=spot, strike_lo=strike_lo, strike_hi=strike_hi)
    if not filled_iv:
        return {}
    by_symbol = {t["symbol"]: t for t in targets}
    out: dict[str, dict] = {}
    for sym, iv in filled_iv.items():
        t = by_symbol.get(sym)
        if t is None:
            continue
        T = t.get("T")
        strike = t.get("strike")
        right = t.get("right")
        delta = greeks_mod.bsm_delta(spot, strike, T, cfg.risk_free_rate, cfg.dividend_yield, iv, right)
        gamma = greeks_mod.bsm_gamma(spot, strike, T, cfg.risk_free_rate, cfg.dividend_yield, iv)
        sane = greeks_mod.sanitize_greeks(delta, gamma)
        if sane is None:
            continue
        out[sym] = {"iv": iv, "delta": sane["delta"], "gamma": sane["gamma"]}
    return out


def _parse_occ(symbol: str) -> Optional[dict]:
    m = _OCC_RE.match(symbol or "")
    if not m:
        return None
    underlying, date_str, type_char, strike_str = m.groups()
    try:
        expiration = datetime.strptime(date_str, "%y%m%d").date()
    except ValueError:
        return None
    return {
        "underlying": underlying,
        "expiration": expiration,
        "right": "call" if type_char == "C" else "put",
        "strike": int(strike_str) / 1000.0,
    }


def _f(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _ts(val: Any) -> Optional[float]:
    """Best-effort epoch-seconds coercion from whatever alpaca-py hands
    back for a message timestamp (a `datetime`, per this package's Trade/
    Quote handling -- DataStream._cast converts the wire "t" field to a
    datetime before dispatch)."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    ts_method = getattr(val, "timestamp", None)
    if callable(ts_method):
        try:
            return float(ts_method())
        except (TypeError, ValueError, OSError):
            return None
    return None


def _normalize_conditions(raw: Any) -> tuple[str, ...]:
    """A trade's `conditions` field is typed as a list by alpaca-py's model,
    but per the SDK's own typing it may arrive as a bare `str` for a
    single-condition trade -- iterating a bare string with
    `tuple(str(c) for c in raw)` would silently split it into individual
    CHARACTERS instead of keeping it as one condition code. Normalize a bare
    str to a 1-tuple first; anything else falls back to the safe empty
    tuple rather than raising."""
    if raw is None:
        return ()
    if isinstance(raw, str):
        return (raw,)
    try:
        return tuple(str(c) for c in raw)
    except TypeError:
        return ()


# ═══════════════════════════════════════════════════════════════════════════
#  Bounded, thread-safe, drop-oldest ingest buffer
# ═══════════════════════════════════════════════════════════════════════════


class BoundedDropOldestQueue:
    """Thread-safe bounded buffer between a stream handler thread (producer)
    and `tick()`'s single-threaded drain (consumer). Never blocks the
    producer and never grows past `maxsize`: once full, the OLDEST queued
    item is discarded to make room, and the discard is counted (`dropped`)
    rather than silently lost from observability. A plain `collections.deque`
    append/popleft pair under one lock -- not `queue.Queue` (whose
    `put_nowait` raising `Full` would force the producer coroutine to handle
    backpressure itself; drop-oldest is a deliberate choice: a stale queued
    trade is worse than a slightly-gappy one, and this package is advisory
    shadow logging, not an execution-critical feed).
    """

    def __init__(self, maxsize: int) -> None:
        self._maxsize = max(1, int(maxsize))
        self._lock = threading.Lock()
        self._items: deque = deque()
        self._dropped = 0
        self._accepted = 0

    def put(self, item: Any) -> None:
        with self._lock:
            if len(self._items) >= self._maxsize:
                self._items.popleft()
                self._dropped += 1
            self._items.append(item)
            self._accepted += 1

    def drain(self, limit: Optional[int] = None) -> list:
        with self._lock:
            if limit is None or limit >= len(self._items):
                out = list(self._items)
                self._items.clear()
            else:
                out = [self._items.popleft() for _ in range(max(0, limit))]
            return out

    def stats(self) -> dict:
        with self._lock:
            return {"queued": len(self._items), "dropped": self._dropped, "accepted": self._accepted}

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


# ═══════════════════════════════════════════════════════════════════════════
#  Stream supervisor: outer reconnect/backoff safety net (testable, no threads)
# ═══════════════════════════════════════════════════════════════════════════


def backoff_delay(consecutive_failures: int, min_backoff: float, max_backoff: float) -> float:
    """PURE. Exponential backoff capped at max_backoff. consecutive_failures
    <= 0 returns 0.0 (no wait before the very first attempt)."""
    if consecutive_failures <= 0:
        return 0.0
    delay = min_backoff * (2 ** (consecutive_failures - 1))
    return min(delay, max_backoff)


def run_stream_supervisor(
    name: str,
    make_stream: Callable[[], Any],
    stop_event: threading.Event,
    min_backoff: float,
    max_backoff: float,
    max_consecutive_failures: int = 0,
    wait_fn: Optional[Callable[[threading.Event, float], bool]] = None,
    on_stream_created: Optional[Callable[[Any], None]] = None,
) -> dict:
    """Runs a stream's lifecycle (`make_stream()` -> `.run()`, blocking)
    until `stop_event` is set or (if `max_consecutive_failures` > 0) that
    many consecutive unrequested exits happen in a row. `make_stream()` must
    return a FRESH object each call exposing a blocking `.run()` and a
    thread-safe `.stop()` -- exactly alpaca-py's `DataStream` interface, so
    the production caller wires a real `OptionDataStream`/`StockDataStream`
    factory and tests wire a fake. `make_stream()` ITSELF raising (not just
    `.run()`) is treated identically -- logged, backed off, retried -- so a
    construction-time failure (e.g. alpaca-py's `StockDataStream.__init__`
    raising `ValueError` for an unsupported feed) can never kill this
    function/thread permanently. `wait_fn(stop_event, timeout) -> bool`
    (default: `stop_event.wait(timeout=timeout)`) is the only place this
    function "sleeps" -- injectable so tests exercise the backoff/restart
    DECISION logic without real wall-clock delay. `on_stream_created(stream)`
    lets the caller (FlowCollectorService) capture the live stream object
    for dynamic subscribe/unsubscribe while it's connected.

    Returns a status dict on exit: {"restarts": int, "gave_up": bool,
    "last_error": str} -- used by tests and available for
    reporting/observability in production.
    """
    if wait_fn is None:
        def wait_fn(ev: threading.Event, timeout: float) -> bool:
            return ev.wait(timeout=timeout)

    consecutive_failures = 0
    restarts = 0
    last_error = ""
    while not stop_event.is_set():
        try:
            stream = make_stream()
        except Exception as e:
            # Construction itself can fail (bad feed value, transient
            # client-init error, ...) -- this MUST be treated exactly like a
            # run() failure (log + backoff + retry), never let it escape and
            # kill this thread permanently. Confirmed reachable: alpaca-py's
            # StockDataStream.__init__ raises ValueError("only IEX and SIP
            # feeds ar supported") for any other DataFeed, which a
            # misconfigured stock_feed could otherwise hit on every restart.
            last_error = f"{type(e).__name__}: {e}"
            logger.warning(f"flowcollector stream[{name}]: make_stream() raised {last_error}")
            consecutive_failures += 1
            restarts += 1
            if max_consecutive_failures > 0 and consecutive_failures >= max_consecutive_failures:
                logger.error(f"flowcollector stream[{name}]: giving up after {consecutive_failures} consecutive failures")
                return {"restarts": restarts, "gave_up": True, "last_error": last_error}
            delay = backoff_delay(consecutive_failures, min_backoff, max_backoff)
            logger.info(f"flowcollector stream[{name}]: reconnecting in {delay:.1f}s (attempt {consecutive_failures})")
            if wait_fn(stop_event, delay):
                break
            continue

        if on_stream_created is not None:
            try:
                on_stream_created(stream)
            except Exception as e:
                logger.debug(f"flowcollector stream[{name}]: on_stream_created hook failed: {type(e).__name__}: {e}")
        try:
            stream.run()  # blocks until the stream's OWN loop exits
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            logger.warning(f"flowcollector stream[{name}]: run() raised {last_error}")

        if stop_event.is_set():
            break

        consecutive_failures += 1
        restarts += 1
        if max_consecutive_failures > 0 and consecutive_failures >= max_consecutive_failures:
            logger.error(f"flowcollector stream[{name}]: giving up after {consecutive_failures} consecutive failures")
            return {"restarts": restarts, "gave_up": True, "last_error": last_error}

        delay = backoff_delay(consecutive_failures, min_backoff, max_backoff)
        logger.info(f"flowcollector stream[{name}]: reconnecting in {delay:.1f}s (attempt {consecutive_failures})")
        if wait_fn(stop_event, delay):
            break  # stop was requested during the backoff wait

    return {"restarts": restarts, "gave_up": False, "last_error": last_error}


class StreamSupervisor:
    """Thin real-thread wrapper around `run_stream_supervisor` for
    production use. Not itself unit-tested with real threads (trivial glue);
    `run_stream_supervisor`'s decision logic is what carries the test
    coverage (see tests/test_flowcollector_collector.py)."""

    def __init__(
        self,
        name: str,
        make_stream: Callable[[], Any],
        min_backoff: float,
        max_backoff: float,
        max_consecutive_failures: int = 0,
    ) -> None:
        self._name = name
        self._make_stream = make_stream
        self._min_backoff = min_backoff
        self._max_backoff = max_backoff
        self._max_consecutive_failures = max_consecutive_failures
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._current_stream_lock = threading.Lock()
        self._current_stream: Any = None
        self._result: dict = {}

    def _on_stream_created(self, stream: Any) -> None:
        with self._current_stream_lock:
            self._current_stream = stream

    def current_stream(self) -> Any:
        with self._current_stream_lock:
            return self._current_stream

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()

        def _run():
            self._result = run_stream_supervisor(
                self._name, self._make_stream, self._stop_event,
                self._min_backoff, self._max_backoff, self._max_consecutive_failures,
                on_stream_created=self._on_stream_created,
            )

        self._thread = threading.Thread(target=_run, name=f"flowcollector-{self._name}", daemon=True)
        self._thread.start()

    def stop(self, join_timeout: float = 10.0) -> None:
        self._stop_event.set()
        stream = self.current_stream()
        if stream is not None:
            try:
                stream.stop()
            except Exception as e:
                logger.debug(f"flowcollector stream[{self._name}]: stop() during shutdown failed: {type(e).__name__}: {e}")
        if self._thread is not None:
            self._thread.join(timeout=join_timeout)
        with self._current_stream_lock:
            self._current_stream = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


# ═══════════════════════════════════════════════════════════════════════════
#  Market-hours idling
# ═══════════════════════════════════════════════════════════════════════════


def is_collector_active(
    now_et: datetime,
    holidays: frozenset,
    early_close_dates: frozenset,
    pre_open_minutes: float,
    post_close_minutes: float,
) -> bool:
    """PURE. True inside [09:30-pre_open_minutes, close+post_close_minutes]
    ET on a weekday that isn't a full NYSE holiday, where `close` is 13:00 on
    an early-close half-day (from `early_close_dates`) else 16:00. Defaults
    give an active window of 09:25-16:05 ET on a normal day, per
    INTEL_V3_SPEC.md."""
    if now_et.weekday() >= 5:
        return False
    today = now_et.date()
    if today in holidays:
        return False
    close_hour, close_minute = (13, 0) if today in early_close_dates else (16, 0)
    open_dt = now_et.replace(hour=9, minute=30, second=0, microsecond=0) - timedelta(minutes=pre_open_minutes)
    close_dt = now_et.replace(hour=close_hour, minute=close_minute, second=0, microsecond=0) + timedelta(minutes=post_close_minutes)
    return open_dt <= now_et <= close_dt


def _market_calendar() -> tuple[frozenset, frozenset]:
    """Lazy import of agent.integrations.market_events's holiday/early-close
    sets (2026/2027-only per that module's own docstring -- extend annually
    there, not here)."""
    from agent.integrations.market_events import MARKET_EARLY_CLOSE, MARKET_HOLIDAYS

    return MARKET_HOLIDAYS, MARKET_EARLY_CLOSE


def is_active_now(cfg: FlowCollectorConfig, now: Optional[datetime] = None) -> bool:
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    now_et = now.astimezone(et) if now is not None else datetime.now(et)
    holidays, early_close = _market_calendar()
    return is_collector_active(now_et, holidays, early_close, cfg.pre_open_minutes, cfg.post_close_minutes)


# ═══════════════════════════════════════════════════════════════════════════
#  Broker protocol + production Alpaca adapter
# ═══════════════════════════════════════════════════════════════════════════


class FlowBroker:
    """Duck-typed interface `FlowCollectorService` depends on. Tests inject
    a fake implementing the same async methods; `AlpacaFlowBroker` below is
    the production implementation. Kept as a plain (non-abc) class so a
    fake only needs to implement the methods it's exercised through."""

    async def get_spot(self, underlying: str) -> Optional[float]:
        raise NotImplementedError

    async def get_option_universe(
        self, underlying: str, exp_min: date, exp_max: date,
        spot: Optional[float] = None, band_pct: Optional[float] = None,
    ) -> Optional[list[dict]]:
        raise NotImplementedError

    async def get_option_greeks(self, symbols: list[str]) -> Optional[dict[str, dict]]:
        raise NotImplementedError

    def make_option_stream(self, on_trade, on_quote) -> Any:
        raise NotImplementedError

    def make_stock_stream(self, on_trade, on_quote) -> Any:
        raise NotImplementedError


class AlpacaFlowBroker(FlowBroker):
    """Production I/O adapter -- the only place in this package that
    imports/constructs alpaca-py clients. READ-ONLY: `alpaca.trading.client.
    TradingClient` is used here ONLY for `get_option_contracts` (contract
    metadata + open interest, per recon doc §1's "Open interest" section --
    it lives ONLY on the trading API, not the data API). No order-submission
    method of that client is ever called (see this module's docstring).

    Feed downgrade mirrors (does not import) `agent.dpof.broker`'s sticky
    OPRA/SIP -> indicative/IEX-on-403 pattern -- "never go blind": a 401/403
    from a REST call (`get_spot`/`get_option_greeks`) sets
    `_stock_feed_downgraded`/`_options_feed_downgraded` for the rest of this
    process. `_feed_for_options`/`_feed_for_stock` are the single point both
    REST calls AND `make_option_stream`/`make_stock_stream` read the
    effective feed from, so once set, every SUBSEQUENT stream (re)connection
    -- via `run_stream_supervisor` rebuilding a fresh stream object on its
    next reconnect attempt -- also picks up the downgraded feed
    automatically (see `test_flowcollector_collector.py`'s
    `test_stream_construction_honors_sticky_feed_downgrade`). What this does
    NOT do (a known, accepted limitation, not a bug): detect a subscription/
    entitlement rejection that happens ENTIRELY INSIDE a live stream's own
    `run()` call -- alpaca-py's `_run_forever` already catches and silently
    absorbs its own "insufficient subscription" error internally (the call
    just returns rather than raising), so there is no exception for this
    module's `run_stream_supervisor` to inspect and react to in that
    specific case. A REST-level 401/403 (which DOES happen routinely, e.g.
    every `get_option_greeks`/`get_spot` refresh cycle) remains the trigger.
    """

    def __init__(self, cfg: FlowCollectorConfig) -> None:
        self._cfg = cfg
        self._trading_client: Any = None
        self._stock_data_client: Any = None
        self._option_data_client: Any = None
        self._options_feed_downgraded = False
        self._stock_feed_downgraded = False
        self._stock_feed_invalid_warned = False

    def _get_trading_client(self):
        if self._trading_client is None:
            from alpaca.trading.client import TradingClient

            self._trading_client = TradingClient(
                api_key=self._cfg.alpaca_key_id, secret_key=self._cfg.alpaca_secret_key, paper=self._cfg.alpaca_paper,
            )
        return self._trading_client

    def _get_stock_data_client(self):
        if self._stock_data_client is None:
            from alpaca.data.historical.stock import StockHistoricalDataClient

            self._stock_data_client = StockHistoricalDataClient(
                api_key=self._cfg.alpaca_key_id, secret_key=self._cfg.alpaca_secret_key,
            )
        return self._stock_data_client

    def _get_option_data_client(self):
        if self._option_data_client is None:
            from alpaca.data.historical.option import OptionHistoricalDataClient

            self._option_data_client = OptionHistoricalDataClient(
                api_key=self._cfg.alpaca_key_id, secret_key=self._cfg.alpaca_secret_key,
            )
        return self._option_data_client

    def _feed_for_options(self):
        from alpaca.data.enums import OptionsFeed

        if self._options_feed_downgraded:
            return OptionsFeed.INDICATIVE
        return OptionsFeed.OPRA if self._cfg.options_feed == "opra" else OptionsFeed.INDICATIVE

    def _feed_for_stock(self):
        from alpaca.data.enums import DataFeed

        if self._stock_feed_downgraded:
            return DataFeed.IEX
        # Only IEX/SIP -- "delayed_sip" is deliberately absent (see
        # config.py's stock_feed docstring): alpaca-py's live StockDataStream
        # raises ValueError for anything else, so a misconfigured value must
        # never reach it. Falls back to IEX defensively (logged once) rather
        # than relying solely on config.validate(), which is advisory-only
        # and does not block startup.
        mapping = {"iex": DataFeed.IEX, "sip": DataFeed.SIP}
        feed = mapping.get(self._cfg.stock_feed)
        if feed is None:
            if not self._stock_feed_invalid_warned:
                self._stock_feed_invalid_warned = True
                logger.warning(
                    f"flowcollector: stock_feed={self._cfg.stock_feed!r} is not supported by the live "
                    f"stream (only 'iex'/'sip' -- see config.py's stock_feed docstring); using 'iex' "
                    f"for the rest of this process."
                )
            return DataFeed.IEX
        return feed

    @staticmethod
    def _is_feed_subscription_error(e: Exception) -> bool:
        status = getattr(e, "status_code", None)
        if status is None:
            resp = getattr(e, "response", None)
            status = getattr(resp, "status_code", None) if resp is not None else None
        return status in (401, 403)

    async def get_spot(self, underlying: str) -> Optional[float]:
        import asyncio

        loop = asyncio.get_running_loop()
        try:
            from alpaca.data.requests import StockLatestTradeRequest

            client = self._get_stock_data_client()

            def _call():
                req = StockLatestTradeRequest(symbol_or_symbols=underlying, feed=self._feed_for_stock())
                return client.get_stock_latest_trade(req)

            result = await loop.run_in_executor(None, _call)
        except Exception as e:
            if self._is_feed_subscription_error(e) and not self._stock_feed_downgraded:
                self._stock_feed_downgraded = True
                logger.warning(f"flowcollector: stock feed rejected, downgrading to iex: {type(e).__name__}: {e}")
                return await self.get_spot(underlying)
            logger.error(f"flowcollector: get_spot({underlying}) failed: {type(e).__name__}: {e}")
            return None
        try:
            trade = result.get(underlying) if isinstance(result, dict) else None
            return _f(getattr(trade, "price", None)) if trade is not None else None
        except Exception as e:
            logger.debug(f"flowcollector: get_spot({underlying}) parse failed: {e}")
            return None

    async def get_option_universe(
        self, underlying: str, exp_min: date, exp_max: date,
        spot: Optional[float] = None, band_pct: Optional[float] = None,
    ) -> Optional[list[dict]]:
        """`spot`/`band_pct`, when both given and spot > 0, are passed to
        Alpaca as a SERVER-SIDE strike filter (`strike_price_gte/lte` --
        confirmed `Optional[str]` fields on `GetOptionContractsRequest` in
        the installed alpaca-py's `trading/requests.py`, hence the
        `f"{x:.4f}"` string formatting below, not a bare float) so the
        un-banded full chain (which can run into the thousands of contracts
        for SPY/QQQ across even a 0-2DTE window) is never fetched just to be
        filtered client-side afterward -- `universe.select_universe` still
        applies its OWN band client-side too (belt-and-suspenders, and the
        only filter at all when spot/band_pct aren't supplied). The
        server-side band is padded wider than the client-side one
        (`_UNIVERSE_FETCH_BAND_PADDING_FACTOR`) so a stale/slightly-off spot
        at fetch time can't clip strikes the client-side filter would
        otherwise have kept."""
        import asyncio

        loop = asyncio.get_running_loop()
        strike_gte: Optional[str] = None
        strike_lte: Optional[str] = None
        if spot is not None and spot > 0 and band_pct is not None and band_pct > 0:
            padded = band_pct * _UNIVERSE_FETCH_BAND_PADDING_FACTOR
            strike_gte = f"{max(0.0, spot * (1.0 - padded)):.4f}"
            strike_lte = f"{spot * (1.0 + padded):.4f}"

        try:
            from alpaca.trading.enums import AssetStatus
            from alpaca.trading.requests import GetOptionContractsRequest

            client = self._get_trading_client()

            def _call():
                contracts: list = []
                page_token = None
                hit_cap = False
                while True:
                    req = GetOptionContractsRequest(
                        underlying_symbols=[underlying], status=AssetStatus.ACTIVE,
                        expiration_date_gte=exp_min, expiration_date_lte=exp_max,
                        strike_price_gte=strike_gte, strike_price_lte=strike_lte,
                        page_token=page_token,
                    )
                    resp = client.get_option_contracts(req)
                    batch = getattr(resp, "option_contracts", None) or []
                    contracts.extend(batch)
                    page_token = getattr(resp, "next_page_token", None)
                    if len(contracts) > _MAX_UNIVERSE_CONTRACTS:
                        hit_cap = True
                        break
                    if not page_token or not batch:
                        break
                return contracts, hit_cap

            contracts, hit_cap = await loop.run_in_executor(None, _call)
        except Exception as e:
            logger.error(f"flowcollector: get_option_universe({underlying}) failed: {type(e).__name__}: {e}")
            return None

        if hit_cap:
            logger.warning(
                f"flowcollector: get_option_universe({underlying}) hit the {_MAX_UNIVERSE_CONTRACTS}-contract "
                f"cap -- some contracts (possibly near-the-money ones on later pages) were dropped. "
                f"strike_gte={strike_gte!r} strike_lte={strike_lte!r}"
            )

        out: list[dict] = []
        for c in contracts:
            try:
                right = "call" if str(getattr(c, "type", "")).lower().endswith("call") else "put"
                oi_raw = getattr(c, "open_interest", None)
                out.append({
                    "symbol": str(getattr(c, "symbol", "")),
                    "underlying_symbol": underlying,
                    "expiration_date": str(getattr(c, "expiration_date", "")),
                    "strike_price": _f(getattr(c, "strike_price", None)),
                    "right": right,
                    "open_interest": int(float(oi_raw)) if oi_raw not in (None, "") else None,
                    "open_interest_date": str(getattr(c, "open_interest_date", "") or ""),
                })
            except Exception as e:
                logger.debug(f"flowcollector: skipping malformed contract: {e}")
        return out

    async def _get_option_greeks_chunk(self, symbols: list[str]) -> Optional[dict[str, dict]]:
        """One `_OPTION_GREEKS_CHUNK_SIZE`-sized request. None on failure
        (after the one feed-downgrade retry) -- the caller (get_option_greeks)
        treats that as "this chunk's data is unavailable" and continues with
        the other chunks, never letting one bad chunk drop everything.

        Parsing note (2026-09-25 incident): the installed alpaca-py's
        `OptionsSnapshot` (alpaca/data/models/snapshots.py) sets `.greeks`
        to `None` whenever the API response itself omits/nulls the
        `"greeks"` key -- confirmed against that source, this collector's
        OWN `getattr(snap, "greeks", None)` read was always correct; the
        2026-09-25 incident was Alpaca's OWN response containing no greeks
        for any symbol all day, not a parsing bug. The bug THIS fix
        addresses is different: the PRIOR version of this method dropped a
        symbol from its return value entirely whenever `.greeks` was None
        (`if greeks is None: continue`), which threw away that symbol's
        `implied_volatility`/`latest_quote` bid-ask too -- the exact inputs
        a self-computed BSM fallback (greeks.py, maybe_refresh_strikes)
        needs. Every symbol that has a snapshot at all is now returned, with
        `delta`/`gamma` simply None when Alpaca's own greeks are absent."""
        import asyncio

        loop = asyncio.get_running_loop()
        try:
            from alpaca.data.requests import OptionSnapshotRequest

            client = self._get_option_data_client()

            def _call():
                req = OptionSnapshotRequest(symbol_or_symbols=list(symbols), feed=self._feed_for_options())
                return client.get_option_snapshot(req)

            snapshots = await loop.run_in_executor(None, _call)
        except Exception as e:
            if self._is_feed_subscription_error(e) and not self._options_feed_downgraded:
                self._options_feed_downgraded = True
                logger.warning(f"flowcollector: options feed rejected, downgrading to indicative: {type(e).__name__}: {e}")
                return await self._get_option_greeks_chunk(symbols)
            logger.error(f"flowcollector: get_option_greeks chunk of {len(symbols)} failed: {type(e).__name__}: {e}")
            return None

        out: dict[str, dict] = {}
        for sym in symbols:
            snap = snapshots.get(sym) if isinstance(snapshots, dict) else None
            if snap is None:
                continue  # truly nothing for this symbol this cycle
            greeks = getattr(snap, "greeks", None)
            quote = getattr(snap, "latest_quote", None)
            out[sym] = {
                "delta": _f(getattr(greeks, "delta", None)) if greeks is not None else None,
                "gamma": _f(getattr(greeks, "gamma", None)) if greeks is not None else None,
                "iv": _f(getattr(snap, "implied_volatility", None)),
                "bid": _f(getattr(quote, "bid_price", None)) if quote is not None else None,
                "ask": _f(getattr(quote, "ask_price", None)) if quote is not None else None,
            }
        return out

    async def get_option_greeks(self, symbols: list[str]) -> Optional[dict[str, dict]]:
        """Chunks `symbols` into `_OPTION_GREEKS_CHUNK_SIZE`-sized requests
        (avoids an oversized query string for the full ~400-800 symbol
        universe) and merges the results. A single failed chunk logs a
        warning and is skipped -- it does NOT abort/drop the other chunks'
        data (item 8). Returns None only if EVERY chunk failed (mirrors the
        prior single-request "total failure -> None" contract that
        maybe_refresh_strikes relies on to keep the prior snapshot)."""
        if not symbols:
            return {}
        out: dict[str, dict] = {}
        any_success = False
        for i in range(0, len(symbols), _OPTION_GREEKS_CHUNK_SIZE):
            chunk = symbols[i : i + _OPTION_GREEKS_CHUNK_SIZE]
            result = await self._get_option_greeks_chunk(chunk)
            if result is None:
                logger.warning(f"flowcollector: option greeks chunk ({len(chunk)} symbols) failed -- other chunks unaffected")
                continue
            any_success = True
            out.update(result)
        if not any_success:
            return None
        return out

    def make_option_stream(self, on_trade, on_quote):
        from alpaca.data.live.option import OptionDataStream

        return OptionDataStream(
            api_key=self._cfg.alpaca_key_id, secret_key=self._cfg.alpaca_secret_key, feed=self._feed_for_options(),
        )

    def make_stock_stream(self, on_trade, on_quote):
        from alpaca.data.live.stock import StockDataStream

        return StockDataStream(
            api_key=self._cfg.alpaca_key_id, secret_key=self._cfg.alpaca_secret_key, feed=self._feed_for_stock(),
        )


# ═══════════════════════════════════════════════════════════════════════════
#  FlowCollectorService — orchestrator
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class _RefreshClock:
    """Plain timestamps the service tracks for "is X due" cadence checks."""

    universe: Optional[float] = None
    strikes: dict = None
    strikes_spot_retry: dict = None
    greeks_recompute: dict = None
    state_write: Optional[float] = None
    comparison_write: Optional[float] = None

    def __post_init__(self):
        if self.strikes is None:
            self.strikes = {}
        if self.strikes_spot_retry is None:
            self.strikes_spot_retry = {}
        if self.greeks_recompute is None:
            self.greeks_recompute = {}


class FlowCollectorService:
    def __init__(self, cfg: FlowCollectorConfig, broker: Optional[FlowBroker] = None) -> None:
        self._cfg = cfg
        self._broker = broker if broker is not None else AlpacaFlowBroker(cfg)
        self._aggregator = Aggregator(
            list(cfg.underlyings), cfg.nofa_scale_usd, cfg.premium_scale_usd, cfg.darkpool_scale_usd,
            cfg.gex_neutral_band, cfg.sweep_window_seconds, cfg.sweep_min_exchanges,
            gex_min_coverage=cfg.gex_min_coverage, nope_min_coverage=cfg.nope_min_coverage,
        )
        self._option_nbbo = NBBOCache()
        self._stock_nbbo = NBBOCache()
        self._last_price = LastPriceTracker()
        self._option_queue = BoundedDropOldestQueue(cfg.inbound_queue_maxsize)
        self._stock_queue = BoundedDropOldestQueue(cfg.inbound_queue_maxsize)
        self._universe: Optional[UniverseState] = None
        self._clock = _RefreshClock()
        self._option_supervisor: Optional[StreamSupervisor] = None
        self._stock_supervisor: Optional[StreamSupervisor] = None
        # R-19: streams that were stopped on purpose by _ensure_streams_stopped
        # ("stock" / "options"). The next restart of such a stream is the
        # expected market-open restart and is logged at INFO; the restart path
        # consumes its entry, so a later unexpected death logs ERROR again.
        self._stopped_deliberately: set[str] = set()
        # Greeks refresh logging state (item 7: rate-limited INFO summary,
        # once-per-ET-day WARNING when an Alpaca batch returns zero usable
        # greeks -- see _maybe_log_greeks_summary/_maybe_warn_alpaca_greeks_empty).
        self._last_greeks_log_ts: Optional[float] = None
        self._last_alpaca_greeks_empty_warn_date: dict = {}
        # NOTE: _streams_active is a derived @property (below), not an
        # instance attribute -- do not assign to it here.

    # ── Stream handlers (run on the STREAM'S OWN thread/loop -- see module
    #    docstring's single-writer discipline). They only enqueue. ────────

    async def _on_option_trade(self, msg) -> None:
        self._option_queue.put(("trade", msg))

    async def _on_option_quote(self, msg) -> None:
        self._option_queue.put(("quote", msg))

    async def _on_stock_trade(self, msg) -> None:
        self._stock_queue.put(("trade", msg))

    async def _on_stock_quote(self, msg) -> None:
        self._stock_queue.put(("quote", msg))

    # ── Drain + process (single-writer; called only from tick()) ─────────

    def _process_option_quote(self, q) -> None:
        symbol = getattr(q, "symbol", None)
        if not symbol:
            return
        self._option_nbbo.update(symbol, NBBO(
            bid=_f(getattr(q, "bid_price", None)), ask=_f(getattr(q, "ask_price", None)),
            bid_size=_f(getattr(q, "bid_size", None)), ask_size=_f(getattr(q, "ask_size", None)),
            ts=_ts(getattr(q, "timestamp", None)),
        ))

    def _process_stock_quote(self, q) -> None:
        symbol = getattr(q, "symbol", None)
        if not symbol:
            return
        self._stock_nbbo.update(symbol, NBBO(
            bid=_f(getattr(q, "bid_price", None)), ask=_f(getattr(q, "ask_price", None)),
            bid_size=_f(getattr(q, "bid_size", None)), ask_size=_f(getattr(q, "ask_size", None)),
            ts=_ts(getattr(q, "timestamp", None)),
        ))

    def _process_option_trade(self, t, now_epoch: float) -> None:
        symbol = getattr(t, "symbol", None)
        parsed = _parse_occ(symbol) if symbol else None
        if parsed is None or parsed["underlying"] not in self._cfg.underlyings:
            return
        price = _f(getattr(t, "price", None))
        size = _f(getattr(t, "size", None))
        if price is None or size is None:
            return
        conditions = _normalize_conditions(getattr(t, "conditions", None))
        if not is_reliable_trade(list(conditions)):
            return
        nbbo = self._option_nbbo.get(symbol)
        prev_price, prev_side = self._last_price.get(symbol)
        side = classify_trade_side(price, nbbo, prev_price, prev_side, self._cfg.quote_epsilon)
        self._last_price.update(symbol, price, side)
        ts = _ts(getattr(t, "timestamp", None)) or now_epoch
        event = TradeEvent(
            ts=ts, symbol=symbol, underlying=parsed["underlying"], right=parsed["right"], strike=parsed["strike"],
            exchange=str(getattr(t, "exchange", "") or ""), price=price, size=size, side=side,
            conditions=conditions, reliable=True,
        )
        self._aggregator.add_option_trade(parsed["underlying"], event)

    def _process_stock_trade(self, t, now_epoch: float) -> None:
        symbol = getattr(t, "symbol", None)
        if not symbol or symbol not in self._cfg.underlyings:
            return
        price = _f(getattr(t, "price", None))
        size = _f(getattr(t, "size", None))
        if price is None or size is None:
            return
        conditions = _normalize_conditions(getattr(t, "conditions", None))
        reliable = is_reliable_trade(list(conditions))
        exchange = str(getattr(t, "exchange", "") or "")
        nbbo = self._stock_nbbo.get(symbol)
        prev_price, prev_side = self._last_price.get(f"eq:{symbol}")
        side = classify_trade_side(price, nbbo, prev_price, prev_side, self._cfg.quote_epsilon)
        self._last_price.update(f"eq:{symbol}", price, side)
        ts = _ts(getattr(t, "timestamp", None)) or now_epoch
        # Captured once, here, at ingest -- see TradeEvent.mid_at_ingest's
        # docstring for why darkpool classification must not re-read the
        # NBBO cache later at snapshot time (item 9).
        mid_at_ingest = nbbo.mid if nbbo is not None else None
        event = TradeEvent(
            ts=ts, symbol=symbol, underlying=symbol, right="", strike=None, exchange=exchange,
            price=price, size=size, side=side, conditions=conditions, reliable=reliable,
            mid_at_ingest=mid_at_ingest,
        )
        if not reliable and exchange != DARK_POOL_EXCHANGE:
            # An unreliable print is still kept if it's a dark-pool (D-tape)
            # trade -- darkpool bias classifies vs midpoint, not the
            # (equity-condition-code) quote rule, so the reliability filter
            # (aimed at odd lots skewing lit-tape VWAP-ish math) doesn't
            # apply the same way here. Anything else unreliable is dropped.
            return
        self._aggregator.add_equity_trade(symbol, event)

    def drain_and_process(self, now_epoch: float) -> dict:
        opt_msgs = self._option_queue.drain()
        for kind, msg in opt_msgs:
            if kind == "quote":
                self._process_option_quote(msg)
            else:
                self._process_option_trade(msg, now_epoch)
        stock_msgs = self._stock_queue.drain()
        for kind, msg in stock_msgs:
            if kind == "quote":
                self._process_stock_quote(msg)
            else:
                self._process_stock_trade(msg, now_epoch)
        self._aggregator.prune(now_epoch)
        return {"option_drained": len(opt_msgs), "stock_drained": len(stock_msgs)}

    # ── Spot sourcing ──────────────────────────────────────────────────────
    #
    # `_live_spot` is TICK-DATA ONLY (no I/O, no universe fallback) -- used
    # by maybe_refresh_universe's drift check, which must never fall back to
    # the universe's OWN last-refresh spot: that value is exactly what this
    # check exists to compare against, and falling back to it would make the
    # drift check permanently see "no drift" (comparing a stale value to
    # itself) whenever no live tick has arrived yet. `_current_spot` is the
    # broader "best guess right now" used for GEX/max-pain at state-write
    # time, where a slightly-stale universe-refresh spot is an acceptable,
    # documented approximation rather than an outright bug.

    def _live_spot(self, underlying: str) -> Optional[float]:
        price, _side = self._last_price.get(f"eq:{underlying}")
        if price is not None:
            return price
        nbbo = self._stock_nbbo.get(underlying)
        if nbbo is not None and nbbo.usable:
            return nbbo.mid
        return None

    def _current_spot(self, underlying: str) -> Optional[float]:
        live = self._live_spot(underlying)
        if live is not None:
            return live
        if self._universe is not None:
            return self._universe.spot_by_underlying.get(underlying)
        return None

    # ── Universe refresh ──────────────────────────────────────────────────

    async def maybe_refresh_universe(self, now_epoch: float) -> bool:
        from datetime import timezone
        from zoneinfo import ZoneInfo

        et = ZoneInfo("America/New_York")
        today = datetime.fromtimestamp(now_epoch, tz=timezone.utc).astimezone(et).date()

        spots: dict[str, float] = {}
        for u in self._cfg.underlyings:
            spots[u] = self._live_spot(u) or (await self._broker.get_spot(u))

        # An empty universe means the option stream can't even start (item
        # 4 -- see _ensure_option_stream_started) -- retry MUCH sooner than
        # the normal 15-min cadence until it has something, rather than
        # leaving the option stream dark for up to universe_refresh_seconds.
        universe_empty = self._universe is None or self._universe.total_count == 0
        effective_refresh_seconds = (
            min(self._cfg.universe_refresh_seconds, self._cfg.universe_empty_retry_seconds)
            if universe_empty else self._cfg.universe_refresh_seconds
        )

        last_spots = self._universe.spot_by_underlying if self._universe is not None else {}
        due = any(
            needs_universe_refresh(
                self._clock.universe, now_epoch, effective_refresh_seconds,
                last_spots.get(u), spots.get(u), self._cfg.universe_spot_drift_pct,
            )
            for u in self._cfg.underlyings
        )
        if not due:
            return False

        # max_dte counts TRADING SESSIONS (Friday -> Mon/Tue), like DPOF's 1DTE.
        holidays = market_holidays()
        exp_min = today
        exp_max = session_date_ahead(today, self._cfg.max_dte, holidays)
        per_underlying: dict[str, list[dict]] = {}
        for u in self._cfg.underlyings:
            contracts = await self._broker.get_option_universe(u, exp_min, exp_max, spot=spots.get(u), band_pct=self._cfg.strike_band_pct)
            per_underlying[u] = contracts or []

        new_universe = build_universe_state(per_underlying, spots, self._cfg.strike_band_pct, self._cfg.max_dte, today, now_epoch, holidays)
        self._reconcile_subscriptions(self._universe, new_universe)
        self._universe = new_universe
        self._clock.universe = now_epoch
        logger.info(f"flowcollector: universe refreshed, {new_universe.total_count} contract symbols")
        return True

    def _reconcile_subscriptions(self, old: Optional[UniverseState], new: UniverseState) -> None:
        """Diffs old vs. new universe symbols and subscribes/unsubscribes on
        the LIVE option stream (if connected) rather than restarting it --
        alpaca-py's DataStream supports dynamic subscribe/unsubscribe on a
        running connection (recon-confirmed: `_subscribe`/`_unsubscribe`
        send a live message when `self._running`). Never touches the stock
        stream (SPY/QQQ trades+quotes are subscribed once, statically, at
        stream construction -- the universe only affects OPTION contracts).
        A missing/not-yet-connected stream is a no-op here; initial
        subscription happens when the stream is first constructed in
        `_ensure_streams_started`.
        """
        if self._option_supervisor is None:
            return
        stream = self._option_supervisor.current_stream()
        if stream is None:
            return
        old_syms = set(old.all_symbols) if old is not None else set()
        new_syms = set(new.all_symbols)
        added = sorted(new_syms - old_syms)
        removed = sorted(old_syms - new_syms)
        cap = self._cfg.symbol_cap_per_connection
        if len(new_syms) > cap:
            logger.warning(
                f"flowcollector: universe size {len(new_syms)} exceeds the configured single-connection "
                f"cap ({cap}); see partition_symbols/UNVERIFIED cap note in universe.py -- subscribing "
                f"the full set on one connection anyway this phase (shadow-only, best-effort)."
            )
        try:
            if added:
                stream.subscribe_trades(self._on_option_trade, *added)
                stream.subscribe_quotes(self._on_option_quote, *added)
            if removed:
                stream.unsubscribe_trades(*removed)
                stream.unsubscribe_quotes(*removed)
        except Exception as e:
            logger.warning(f"flowcollector: option stream subscription reconcile failed: {type(e).__name__}: {e}")

    # ── Strikes/greeks refresh (feeds GEX/max-pain/NOPE) ──────────────────
    #
    # Two independent cadences (2026-09-25 fix, see agent/flowcollector/
    # greeks.py's module docstring for the incident this whole block fixes):
    #
    #   maybe_refresh_strikes (cfg.strikes_refresh_seconds, ~15 min): the
    #   REST-heavy pull -- open interest AND Alpaca's own snapshot greeks --
    #   still only as fresh as that REST cadence allows (OI is EOD-stale
    #   regardless, per recon doc §1).
    #
    #   maybe_recompute_greeks (cfg.greeks_recompute_seconds, ~60s default):
    #   a network-free recompute of ONLY the self-computed (never
    #   Alpaca-sourced) deltas/gammas, from this collector's OWN live option
    #   NBBO cache, so 0DTE gamma keeps tracking intraday moves between the
    #   slow REST pulls above.
    #
    # Both funnel their actual BSM/implied-vol math through
    # `_compute_greeks_batch_sync` via `loop.run_in_executor` (module-level
    # function, no shared state touched inside the executor thread) so a
    # batch of a few hundred solves never blocks this process's own asyncio
    # loop for the CPU-heavy part (item 3's ">50ms" bound) -- only the cheap
    # dict-lookup bookkeeping around it runs directly on the loop.

    def _nbbo_is_fresh(self, nbbo: Optional[NBBO], now_epoch: float) -> bool:
        """True if `nbbo` is usable AND recent enough to trust for a greeks
        computation right now. "Recent" is tied to cfg.greeks_recompute_seconds
        (the cadence this collector itself intends to refresh self-computed
        greeks at) with a floor of 30s and a 2x margin -- chosen, not
        measured, like every other freshness/scale constant in this
        package."""
        if nbbo is None or not nbbo.usable or nbbo.ts is None:
            return False
        max_age = max(self._cfg.greeks_recompute_seconds * 2.0, 30.0)
        return (now_epoch - nbbo.ts) <= max_age

    def _rebuild_strikes_from_greeks(self, u: str) -> None:
        """Rebuilds `st.strikes`/`st.oi_by_strike`/`st.gamma_oi_coverage`
        for underlying `u` from `self._universe.contracts_by_symbol` (open
        interest/strike/right -- unchanged since the last strikes REST pull)
        combined with `st.gammas_by_symbol` (which EITHER of
        maybe_refresh_strikes/maybe_recompute_greeks may have just updated).
        Network-free, O(#symbols) dict lookups -- cheap enough to call
        directly on the loop (no executor needed here, unlike the BSM solves
        that populate gammas_by_symbol in the first place)."""
        st = self._aggregator.state.get(u)
        if st is None or self._universe is None:
            return
        symbols = self._universe.symbols_by_underlying.get(u, [])
        by_strike: dict[float, StrikeExposure] = {}
        oi_by_strike: dict[float, dict] = {}
        oi_total = 0.0
        oi_known = 0.0
        for sym in symbols:
            contract = self._universe.contracts_by_symbol.get(sym)
            if not contract:
                continue
            strike = _f(contract.get("strike_price"))
            oi = contract.get("open_interest")
            right = contract.get("right")
            if strike is None or oi is None or right not in ("call", "put"):
                continue
            oi_total += oi
            gamma = st.gammas_by_symbol.get(sym)
            if gamma is not None:
                oi_known += oi
            gamma_for_gex = gamma if gamma is not None else 0.0

            bucket = by_strike.setdefault(strike, StrikeExposure(strike=strike))
            oi_bucket = oi_by_strike.setdefault(strike, {"call_oi": 0.0, "put_oi": 0.0})
            # Multiple expiries (0/1/2DTE) commonly share a strike, each
            # with its OWN gamma -- accumulate the PER-CONTRACT weighted
            # sum (Σ gamma_i * OI_i), never overwrite a single "last
            # contract seen" gamma (item 2's original bug). See
            # StrikeExposure's docstring.
            if right == "call":
                bucket.call_oi += oi
                bucket.call_gamma_oi += gamma_for_gex * oi
                oi_bucket["call_oi"] += oi
            else:
                bucket.put_oi += oi
                bucket.put_gamma_oi += gamma_for_gex * oi
                oi_bucket["put_oi"] += oi

        st.strikes = list(by_strike.values())
        st.oi_by_strike = oi_by_strike
        # "No fake zeros" (item 4): this is the OI-weighted coverage
        # aggregator.compute_gex gates on -- a contract with unknown gamma
        # contributes 0 to call_gamma_oi/put_gamma_oi above (so it never
        # silently pulls net_gex toward a wrong number), but it ALSO drags
        # this coverage fraction down, so a mostly-unknown-gamma day reports
        # gex=None ("unavailable") instead of a bogus near-zero "neutral".
        # OI present but summing to 0 is informative (coverage 0.0 -> gated), not
        # "unknown": None would skip compute_gex's gate and report a fake 0.0.
        if oi_total > 0:
            st.gamma_oi_coverage = oi_known / oi_total
        else:
            st.gamma_oi_coverage = 0.0 if st.strikes else None

    def _recount_greeks_sources(self, u: str) -> None:
        st = self._aggregator.state.get(u)
        if st is None:
            return
        counts = {"alpaca": 0, "computed": 0, "interpolated": 0, "missing": 0}
        for src in st.greeks_source_by_symbol.values():
            if src in counts:
                counts[src] += 1
        st.greeks_source_counts = counts

    def _maybe_log_greeks_summary(self, now_epoch: float) -> None:
        """Rate-limited (<=1/5min, item 7) INFO summary of greeks source
        counts + coverage across every configured underlying -- fires from
        BOTH maybe_refresh_strikes and maybe_recompute_greeks (either can
        change the picture), sharing one timestamp so the combined rate
        across both never exceeds the cap."""
        last = self._last_greeks_log_ts
        if last is not None and (now_epoch - last) < 300.0:
            return
        self._last_greeks_log_ts = now_epoch
        parts = []
        for u in self._cfg.underlyings:
            st = self._aggregator.state.get(u)
            if st is None:
                continue
            src = st.greeks_source_counts or {}
            cov = st.gamma_oi_coverage
            cov_str = f"{cov:.2f}" if isinstance(cov, (int, float)) else "n/a"
            parts.append(
                f"{u}: alpaca={src.get('alpaca', 0)} computed={src.get('computed', 0)} "
                f"interpolated={src.get('interpolated', 0)} missing={src.get('missing', 0)} "
                f"gamma_oi_coverage={cov_str}"
            )
        if parts:
            logger.info("flowcollector: greeks refresh -- " + "; ".join(parts))

    def _maybe_warn_alpaca_greeks_empty(self, u: str, symbol_count: int, alpaca_count: int, now_epoch: float) -> None:
        """WARNING at most once per ET calendar day (item 7) when an ENTIRE
        Alpaca snapshot batch for `u` came back with zero usable (sane)
        greeks -- exactly the 2026-09-25 incident's signature."""
        if symbol_count <= 0 or alpaca_count > 0:
            return
        from datetime import timezone as _tz
        from zoneinfo import ZoneInfo as _ZI

        today = datetime.fromtimestamp(now_epoch, tz=_tz.utc).astimezone(_ZI("America/New_York")).date()
        if self._last_alpaca_greeks_empty_warn_date.get(u) == today:
            return
        self._last_alpaca_greeks_empty_warn_date[u] = today
        logger.warning(
            f"flowcollector: Alpaca snapshot greeks empty for {u} ({symbol_count} symbols, 0 usable) -- "
            f"using self-computed Black-Scholes greeks"
        )

    async def maybe_refresh_strikes(self, now_epoch: float) -> None:
        if self._universe is None:
            return
        import asyncio
        from datetime import timezone as _tz

        for u in self._cfg.underlyings:
            last = self._clock.strikes.get(u)
            if last is not None and (now_epoch - last) < self._cfg.strikes_refresh_seconds:
                continue
            symbols = self._universe.symbols_by_underlying.get(u, [])
            if not symbols:
                continue

            spot = self._current_spot(u)
            if spot is None:
                # 2026-09-25 follow-up (item 4): don't burn the full
                # strikes_refresh_seconds cadence -- and don't pay for the
                # REST call at all -- on a cycle that can't self-compute OR
                # interpolate ANYTHING without a spot to square. Structurally
                # this shouldn't happen once `symbols` is non-empty (that
                # requires universe.select_universe to have had a positive
                # spot in the first place, and UniverseState.spot_by_underlying
                # is _current_spot's own fallback -- see that method's
                # docstring), but this guard is kept as defense-in-depth
                # against a future change to that invariant (or a test/fake
                # broker feeding an inconsistent state) rather than relying
                # on it silently. Retries on a short, separate cadence
                # instead of the full 15-min one.
                last_retry = self._clock.strikes_spot_retry.get(u)
                if last_retry is not None and (now_epoch - last_retry) < self._cfg.strikes_spot_retry_seconds:
                    continue
                self._clock.strikes_spot_retry[u] = now_epoch
                logger.debug(
                    f"flowcollector: strikes refresh for {u} skipped -- no spot yet; "
                    f"retrying in {self._cfg.strikes_spot_retry_seconds:.0f}s"
                )
                continue
            self._clock.strikes_spot_retry.pop(u, None)

            greeks = await self._broker.get_option_greeks(symbols)
            self._clock.strikes[u] = now_epoch
            if greeks is None:
                logger.debug(f"flowcollector: strikes/greeks refresh for {u} failed -- keeping prior snapshot")
                continue

            st = self._aggregator.state.get(u)
            now_dt = datetime.fromtimestamp(now_epoch, tz=_tz.utc)
            _T = _t_memo(now_dt)  # T per expiry, not per contract (keeps the loop cheap)
            deltas_by_symbol: dict[str, float] = {}
            gammas_by_symbol: dict[str, float] = {}
            source_by_symbol: dict[str, str] = {}
            iv_by_symbol: dict[str, float] = {}  # TRUSTED (alpaca/computed) anchors only
            strike_by_symbol: dict[str, float] = {}
            right_by_symbol: dict[str, str] = {}
            expiry_by_symbol: dict[str, Any] = {}
            T_by_symbol: dict[str, float] = {}
            jobs: list = []  # (symbol, bid, ask, spot, strike, T, right, iv_hint)

            for sym in symbols:
                contract = self._universe.contracts_by_symbol.get(sym)
                if not contract:
                    continue
                strike = _f(contract.get("strike_price"))
                right = contract.get("right")
                if strike is None or right not in ("call", "put"):
                    continue
                strike_by_symbol[sym] = strike
                right_by_symbol[sym] = right
                expiry = _parse_date(contract.get("expiration_date"))
                T = _T(expiry) if expiry is not None else None
                expiry_by_symbol[sym] = expiry
                if T is not None:
                    T_by_symbol[sym] = T

                g = greeks.get(sym) or {}
                # Source precedence, leg 1: Alpaca's own greeks, only when
                # BOTH delta and gamma are present and pass sanitize_greeks
                # (a partially-sane pair -- e.g. gamma present, delta
                # missing -- falls through to self-compute for BOTH legs
                # rather than mixing an Alpaca leg with a computed one for
                # the same contract; see greeks.sanitize_greeks's docstring).
                sane = greeks_mod.sanitize_greeks(g.get("delta"), g.get("gamma"))
                if sane is not None:
                    deltas_by_symbol[sym] = sane["delta"]
                    gammas_by_symbol[sym] = sane["gamma"]
                    source_by_symbol[sym] = "alpaca"
                    iv = g.get("iv")
                    if iv is not None and greeks_mod.is_finite(iv) and 0.0 < iv <= 5.0:
                        iv_by_symbol[sym] = float(iv)
                    continue

                if T is None:
                    source_by_symbol[sym] = "missing"  # no expiry -> can't BSM or interpolate at all
                    continue

                # Source precedence, leg 2: self-compute, from the FRESHEST
                # quote -- this collector's own live option NBBO cache if
                # usable and recent, else this same REST snapshot's own
                # latest_quote bid/ask (item 3's precedence order).
                nbbo = self._option_nbbo.get(sym)
                if self._nbbo_is_fresh(nbbo, now_epoch):
                    bid, ask = nbbo.bid, nbbo.ask
                else:
                    bid, ask = g.get("bid"), g.get("ask")
                if bid is None or ask is None:
                    source_by_symbol[sym] = "missing"  # tentative -- leg 3 (smile fill) may still resolve it
                    continue
                jobs.append((sym, bid, ask, spot, strike, T, right, g.get("iv")))

            if jobs:
                loop = asyncio.get_running_loop()
                computed = await loop.run_in_executor(None, _compute_greeks_batch_sync, jobs, self._cfg)
            else:
                computed = {}
            for job in jobs:
                sym = job[0]
                result = computed.get(sym)
                if result is not None:
                    deltas_by_symbol[sym] = result["delta"]
                    gammas_by_symbol[sym] = result["gamma"]
                    source_by_symbol[sym] = "computed"
                    iv_by_symbol[sym] = result["iv"]
                else:
                    source_by_symbol[sym] = "missing"

            # Source precedence, leg 3 (2026-09-25 follow-up -- coverage-risk
            # fix): every symbol still "missing" but with a known T gets one
            # more shot -- fill its IV from the same-expiry smile of
            # contracts that DID resolve above (leg 1 or leg 2), then BSM
            # delta/gamma from that filled IV. This is what keeps far-OTM
            # (bid=0 / too-wide-spread) strikes, which often carry large OI,
            # out of "missing" -- their gamma is small, not unknown. See
            # greeks.fill_iv_from_smile's docstring.
            fill_targets = [
                {"symbol": s, "strike": strike_by_symbol[s], "expiry": expiry_by_symbol[s], "right": right_by_symbol[s], "T": T_by_symbol[s]}
                for s, src in source_by_symbol.items()
                if src == "missing" and s in T_by_symbol
            ]
            if fill_targets and iv_by_symbol:
                anchors = [
                    {"symbol": s, "strike": strike_by_symbol[s], "expiry": expiry_by_symbol[s], "right": right_by_symbol[s], "iv": iv}
                    for s, iv in iv_by_symbol.items()
                ]
                band_pct = self._cfg.strike_band_pct
                strike_lo = spot * (1.0 - band_pct) if band_pct else None
                strike_hi = spot * (1.0 + band_pct) if band_pct else None
                loop = asyncio.get_running_loop()
                filled = await loop.run_in_executor(None, _fill_smile_batch_sync, anchors, fill_targets, spot, strike_lo, strike_hi, self._cfg)
                for sym, result in filled.items():
                    deltas_by_symbol[sym] = result["delta"]
                    gammas_by_symbol[sym] = result["gamma"]
                    source_by_symbol[sym] = "interpolated"

            if st is not None:
                st.deltas_by_symbol = deltas_by_symbol
                st.gammas_by_symbol = gammas_by_symbol
                st.greeks_source_by_symbol = source_by_symbol
                st.iv_by_symbol = iv_by_symbol  # anchors only -- never includes "interpolated" IVs
                st.strikes_as_of = now_epoch
                self._rebuild_strikes_from_greeks(u)
                self._recount_greeks_sources(u)
                counts = st.greeks_source_counts or {}
                self._maybe_warn_alpaca_greeks_empty(u, len(symbols), counts.get("alpaca", 0), now_epoch)
        self._maybe_log_greeks_summary(now_epoch)

    async def maybe_recompute_greeks(self, now_epoch: float) -> dict:
        """Lighter/faster cadence (cfg.greeks_recompute_seconds, default
        60s): recomputes ONLY self-computed (never "alpaca"-sourced)
        contract deltas/gammas from this collector's OWN live option NBBO
        cache -- no REST call at all, so it can run far more often than
        maybe_refresh_strikes's OI/Alpaca-greeks REST pull, keeping 0DTE
        gamma tracking intraday moves between those slower refreshes.
        Symbols this cycle's `st.greeks_source_by_symbol` already marks
        "alpaca" are left untouched here -- trusted until the next
        strikes_refresh_seconds pull, same as OI itself.

        2026-09-25 follow-up (item 3): also RE-FILLS every "missing"/
        "interpolated" symbol that did NOT get a fresh live-quote update
        this cycle, via greeks.fill_iv_from_smile, using the CURRENT anchor
        set (`st.iv_by_symbol` -- extended below whenever a live quote
        upgrades a symbol to "computed", so the anchor set can only grow
        richer across recompute cycles within one strikes_refresh_seconds
        window). This re-fill matters even for a symbol whose OWN quote
        didn't move: the smile it's interpolated against can still have
        shifted (nearby anchors' IVs updated, spot moved)."""
        if self._universe is None:
            return {}
        import asyncio
        from datetime import timezone as _tz

        touched: dict[str, int] = {}
        for u in self._cfg.underlyings:
            last = self._clock.greeks_recompute.get(u)
            if last is not None and (now_epoch - last) < self._cfg.greeks_recompute_seconds:
                continue
            self._clock.greeks_recompute[u] = now_epoch
            st = self._aggregator.state.get(u)
            if st is None:
                touched[u] = 0
                continue
            symbols = self._universe.symbols_by_underlying.get(u, [])
            if not symbols:
                touched[u] = 0
                continue
            spot = self._current_spot(u)
            now_dt = datetime.fromtimestamp(now_epoch, tz=_tz.utc)
            _T = _t_memo(now_dt)  # T per expiry, not per contract (keeps the loop cheap)

            jobs: list = []
            for sym in symbols:
                if st.greeks_source_by_symbol.get(sym) == "alpaca":
                    continue
                contract = self._universe.contracts_by_symbol.get(sym)
                if not contract:
                    continue
                nbbo = self._option_nbbo.get(sym)
                if not self._nbbo_is_fresh(nbbo, now_epoch):
                    continue  # nothing fresher than what's already stored -- leave it as-is
                strike = _f(contract.get("strike_price"))
                right = contract.get("right")
                if strike is None or right not in ("call", "put") or spot is None:
                    continue
                expiry = _parse_date(contract.get("expiration_date"))
                if expiry is None:
                    continue
                T = _T(expiry)
                if T is None:
                    continue
                jobs.append((sym, nbbo.bid, nbbo.ask, spot, strike, T, right, None))

            n_updated = 0
            if jobs:
                loop = asyncio.get_running_loop()
                computed = await loop.run_in_executor(None, _compute_greeks_batch_sync, jobs, self._cfg)
                for job in jobs:
                    sym = job[0]
                    result = computed.get(sym)
                    if result is None:
                        continue
                    st.deltas_by_symbol[sym] = result["delta"]
                    st.gammas_by_symbol[sym] = result["gamma"]
                    st.greeks_source_by_symbol[sym] = "computed"
                    st.iv_by_symbol[sym] = result["iv"]  # extends the smile anchor set
                    n_updated += 1

            # Re-fill leftover "missing"/"interpolated" symbols (those that
            # did NOT just get a fresh live-quote update above) from the
            # current anchor set.
            fresh_syms = {job[0] for job in jobs} if jobs else set()
            fill_targets: list = []
            if spot is not None:
                for sym in symbols:
                    src = st.greeks_source_by_symbol.get(sym)
                    if src not in ("missing", "interpolated") or sym in fresh_syms:
                        continue
                    contract = self._universe.contracts_by_symbol.get(sym)
                    if not contract:
                        continue
                    strike = _f(contract.get("strike_price"))
                    right = contract.get("right")
                    if strike is None or right not in ("call", "put"):
                        continue
                    expiry = _parse_date(contract.get("expiration_date"))
                    if expiry is None:
                        continue
                    T = _T(expiry)
                    if T is None:
                        continue
                    fill_targets.append({"symbol": sym, "strike": strike, "expiry": expiry, "right": right, "T": T})

            if fill_targets and st.iv_by_symbol:
                anchors: list = []
                for sym, iv in st.iv_by_symbol.items():
                    contract = self._universe.contracts_by_symbol.get(sym)
                    if not contract:
                        continue
                    strike = _f(contract.get("strike_price"))
                    right = contract.get("right")
                    expiry = _parse_date(contract.get("expiration_date"))
                    if strike is None or right not in ("call", "put") or expiry is None:
                        continue
                    anchors.append({"symbol": sym, "strike": strike, "expiry": expiry, "right": right, "iv": iv})
                if anchors:
                    band_pct = self._cfg.strike_band_pct
                    strike_lo = spot * (1.0 - band_pct) if band_pct else None
                    strike_hi = spot * (1.0 + band_pct) if band_pct else None
                    loop = asyncio.get_running_loop()
                    filled = await loop.run_in_executor(None, _fill_smile_batch_sync, anchors, fill_targets, spot, strike_lo, strike_hi, self._cfg)
                    for sym, result in filled.items():
                        st.deltas_by_symbol[sym] = result["delta"]
                        st.gammas_by_symbol[sym] = result["gamma"]
                        st.greeks_source_by_symbol[sym] = "interpolated"
                        n_updated += 1

            if n_updated:
                self._rebuild_strikes_from_greeks(u)
                self._recount_greeks_sources(u)
            touched[u] = n_updated
        if touched:
            self._maybe_log_greeks_summary(now_epoch)
        return touched

    # ── State + comparison writes ─────────────────────────────────────────
    #
    # (Darkpool classification no longer needs a live NBBO-lookup callback
    # here -- each equity TradeEvent already carries its own mid_at_ingest,
    # captured once in _process_stock_trade at the moment it was processed;
    # see TradeEvent's and Aggregator.snapshot's docstrings, item 9.)

    def _build_state_payload(self, now_epoch: float, extra_status: Optional[dict] = None) -> dict:
        for u in self._cfg.underlyings:
            st = self._aggregator.state.get(u)
            if st is not None:
                st.spot = self._current_spot(u)
        status = {
            "streams_active": self._streams_active,
            "universe_size": self._universe.total_count if self._universe is not None else 0,
            "option_queue": self._option_queue.stats(),
            "stock_queue": self._stock_queue.stats(),
        }
        if extra_status:
            status.update(extra_status)
        return {
            "as_of": now_epoch,
            "collector_status": status,
            "symbols": {
                u: self._aggregator.snapshot(u, now_epoch)
                for u in self._cfg.underlyings
            },
        }

    async def maybe_write_state(self, now_epoch: float) -> bool:
        due = self._clock.state_write is None or (now_epoch - self._clock.state_write) >= self._cfg.aggregation_interval_seconds
        if not due:
            return False
        payload = self._build_state_payload(now_epoch)
        import asyncio

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, write_state_atomic, payload, self._cfg.state_file)
        self._clock.state_write = now_epoch
        return True

    async def _write_final_state(self) -> None:
        """Unconditional (cadence-bypassing) state write on shutdown (item
        7): __main__.py's docstring promises "one final state snapshot"
        and, before this fix, run_forever's finally block never actually
        wrote one (only maybe_write_state did, gated by
        aggregation_interval_seconds, which could easily have JUST fired
        and make this write look "not due"). Explicitly marks
        collector_status.streams_active=False + shutdown=True regardless of
        what the (already-stopped, by the time this runs) supervisors would
        otherwise report. Bounded by a short timeout and never raises --
        a slow/failing final write must never hang graceful shutdown."""
        import asyncio

        now_epoch = time.time()
        payload = self._build_state_payload(now_epoch, extra_status={"streams_active": False, "shutdown": True})
        try:
            loop = asyncio.get_running_loop()
            await asyncio.wait_for(
                loop.run_in_executor(None, write_state_atomic, payload, self._cfg.state_file),
                timeout=5.0,
            )
        except Exception as e:
            logger.error(f"flowcollector: final shutdown state write failed: {type(e).__name__}: {e}")

    async def maybe_write_comparison(self, now_epoch: float) -> bool:
        due = (
            self._clock.comparison_write is None
            or (now_epoch - self._clock.comparison_write) >= self._cfg.comparison_log_interval_seconds
        )
        if not due:
            return False
        self._clock.comparison_write = now_epoch
        import asyncio

        loop = asyncio.get_running_loop()
        uw_snapshot = await loop.run_in_executor(None, _read_uw_bridge_snapshot_sync, self._cfg.tradingbot_dir)
        wrote_any = False
        for u in self._cfg.underlyings:
            entry = self._aggregator.snapshot(u, now_epoch)
            row = build_comparison_row(u, entry, uw_snapshot, now_epoch)
            if row is None:
                continue
            await loop.run_in_executor(None, append_comparison_row, row, self._cfg.compare_file)
            wrote_any = True
        return wrote_any

    # ── Stream lifecycle (market-hours gated) ─────────────────────────────

    @property
    def _streams_active(self) -> bool:
        """Derived, not a manually-toggled flag -- reflects the ACTUAL live
        state of the two supervisor threads (item 1: "streams_active
        reflects reality in the state file"), an OR because the option
        stream can legitimately be absent while the universe is still
        empty (item 4) without that meaning "nothing is running"."""
        stock_up = self._stock_supervisor is not None and self._stock_supervisor.is_running()
        option_up = self._option_supervisor is not None and self._option_supervisor.is_running()
        return stock_up or option_up

    def _ensure_stock_stream_started(self) -> None:
        """The stock stream has no universe dependency (SPY/QQQ come
        straight from cfg.underlyings, never empty) -- always safe to
        (re)start as soon as the market is active. Doubles as the item-1
        liveness re-check: called every tick while active, so a supervisor
        thread that somehow died gets noticed and restarted rather than
        silently leaving streams_active wrong (construction/run failures
        inside run_stream_supervisor itself already retry forever and
        should never actually reach "thread exited" -- this is defense in
        depth, not the primary mechanism)."""
        if self._stock_supervisor is not None and self._stock_supervisor.is_running():
            return
        if self._stock_supervisor is not None:
            if "stock" in self._stopped_deliberately:
                logger.info("flowcollector: stock stream supervisor restarting after deliberate stop (market open)")
            else:
                logger.error("flowcollector: stock stream supervisor thread found dead -- restarting")
        self._stopped_deliberately.discard("stock")

        def _make_stock_stream():
            stream = self._broker.make_stock_stream(self._on_stock_trade, self._on_stock_quote)
            stream.subscribe_trades(self._on_stock_trade, *self._cfg.underlyings)
            stream.subscribe_quotes(self._on_stock_quote, *self._cfg.underlyings)
            return stream

        self._stock_supervisor = StreamSupervisor(
            "stock", _make_stock_stream, self._cfg.reconnect_min_backoff_seconds,
            self._cfg.reconnect_max_backoff_seconds, self._cfg.reconnect_max_consecutive_failures,
        )
        self._stock_supervisor.start()
        logger.info("flowcollector: stock stream started")

    def _ensure_option_stream_started(self) -> None:
        """Item 4 (CRITICAL -- resource exhaustion): alpaca-py's
        `DataStream._run_forever` busy-loops (`await asyncio.sleep(0)`, no
        real wait) until at least one symbol is subscribed -- starting the
        option stream with ZERO handlers would pin a CPU core for as long
        as the universe stays empty (up to universe_refresh_seconds, 15 min
        by default, absent the fix below). NEVER starts with an empty
        universe; `maybe_refresh_universe`'s short
        `universe_empty_retry_seconds` cadence (the other half of item 4)
        keeps an empty universe from staying empty for long, and this
        method is re-checked every tick so the option stream starts within
        one tick of the universe first having >=1 symbol. Also serves as
        the liveness re-check for this stream, same rationale as
        `_ensure_stock_stream_started`."""
        if self._universe is None or self._universe.total_count == 0:
            return
        if self._option_supervisor is not None and self._option_supervisor.is_running():
            return
        if self._option_supervisor is not None:
            if "options" in self._stopped_deliberately:
                logger.info("flowcollector: option stream supervisor restarting after deliberate stop (market open)")
            else:
                logger.error("flowcollector: option stream supervisor thread found dead -- restarting")
        self._stopped_deliberately.discard("options")

        def _make_option_stream():
            stream = self._broker.make_option_stream(self._on_option_trade, self._on_option_quote)
            symbols = self._universe.all_symbols if self._universe is not None else []
            if symbols:
                stream.subscribe_trades(self._on_option_trade, *symbols)
                stream.subscribe_quotes(self._on_option_quote, *symbols)
            return stream

        self._option_supervisor = StreamSupervisor(
            "options", _make_option_stream, self._cfg.reconnect_min_backoff_seconds,
            self._cfg.reconnect_max_backoff_seconds, self._cfg.reconnect_max_consecutive_failures,
        )
        self._option_supervisor.start()
        logger.info("flowcollector: option stream started")

    def _ensure_streams_started(self) -> None:
        self._ensure_stock_stream_started()
        self._ensure_option_stream_started()

    def _ensure_streams_stopped(self) -> None:
        if self._option_supervisor is None and self._stock_supervisor is None:
            return
        was_active = self._streams_active
        if self._option_supervisor is not None:
            self._option_supervisor.stop()
            self._stopped_deliberately.add("options")
        if self._stock_supervisor is not None:
            self._stock_supervisor.stop()
            self._stopped_deliberately.add("stock")
        if was_active:
            logger.info("flowcollector: streams stopped (outside market hours)")

    async def maybe_manage_streams(self, now_epoch: float) -> None:
        from datetime import timezone as _tz

        active = is_active_now(self._cfg, now=datetime.fromtimestamp(now_epoch, tz=_tz.utc))
        if active:
            if self._universe is None or self._universe.total_count == 0:
                await self.maybe_refresh_universe(now_epoch)
            self._ensure_streams_started()
        else:
            self._ensure_streams_stopped()

    # ── Top-level tick + run loop ─────────────────────────────────────────

    async def tick(self, now_epoch: Optional[float] = None) -> dict:
        now_epoch = time.time() if now_epoch is None else now_epoch
        self.drain_and_process(now_epoch)
        await self.maybe_manage_streams(now_epoch)
        if self._streams_active:
            await self.maybe_refresh_universe(now_epoch)
            await self.maybe_refresh_strikes(now_epoch)
            await self.maybe_recompute_greeks(now_epoch)
        wrote_state = await self.maybe_write_state(now_epoch)
        await self.maybe_write_comparison(now_epoch)
        return {"as_of": now_epoch, "streams_active": self._streams_active, "wrote_state": wrote_state}

    async def run_forever(self, stop_event) -> None:
        """`stop_event` is an asyncio.Event set by __main__.py's SIGTERM
        handler (via loop.call_soon_threadsafe). Ticks on cfg.tick_seconds
        cadence; on shutdown, stops any live stream supervisors and writes
        one final state snapshot marking the collector stopped, all off the
        loop."""
        import asyncio

        try:
            while not stop_event.is_set():
                try:
                    await self.tick()
                except Exception as e:
                    logger.error(f"flowcollector: tick failed: {type(e).__name__}: {e}", exc_info=True)
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=self._cfg.tick_seconds)
                except asyncio.TimeoutError:
                    pass
        finally:
            self._ensure_streams_stopped()
            await self._write_final_state()
            logger.info("flowcollector: shutdown complete")
