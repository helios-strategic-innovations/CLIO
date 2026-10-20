"""Tests for agent/flowcollector/collector.py.

Hermetic: no network, no real Alpaca SDK objects, no real websocket threads
for the STREAM SUPERVISOR decision logic (a fake stream + an injected
wait_fn drive it synchronously). A small number of tests DO spin one real
short-lived daemon thread each (StreamSupervisor.start()/stop() with a fake
stream whose .run() returns almost immediately) to exercise the thin
threading glue -- always joined with a bounded timeout so a bug here fails
fast rather than hanging the suite.
"""

import asyncio
import threading
import time
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from agent.flowcollector.collector import (
    AlpacaFlowBroker,
    BoundedDropOldestQueue,
    FlowBroker,
    FlowCollectorService,
    StreamSupervisor,
    _MAX_UNIVERSE_CONTRACTS,
    _OPTION_GREEKS_CHUNK_SIZE,
    _parse_occ,
    backoff_delay,
    is_active_now,
    is_collector_active,
    run_stream_supervisor,
)
from agent.flowcollector.config import FlowCollectorConfig
import agent.flowcollector.collector as collector_mod

ET = ZoneInfo("America/New_York")


def _run(coro):
    return asyncio.run(coro)


# ═══════════════════════════════════════════════════════════════════════════
#  BoundedDropOldestQueue
# ═══════════════════════════════════════════════════════════════════════════


def test_bounded_queue_accepts_up_to_maxsize():
    q = BoundedDropOldestQueue(3)
    q.put("a")
    q.put("b")
    q.put("c")
    assert q.stats() == {"queued": 3, "dropped": 0, "accepted": 3}


def test_bounded_queue_drops_oldest_when_full():
    q = BoundedDropOldestQueue(3)
    for x in ("a", "b", "c", "d", "e"):
        q.put(x)
    stats = q.stats()
    assert stats["queued"] == 3
    assert stats["dropped"] == 2
    assert stats["accepted"] == 5
    assert q.drain() == ["c", "d", "e"]  # oldest (a, b) were dropped


def test_bounded_queue_drain_clears_and_respects_limit():
    q = BoundedDropOldestQueue(10)
    for i in range(5):
        q.put(i)
    first = q.drain(limit=2)
    assert first == [0, 1]
    assert len(q) == 3
    rest = q.drain()
    assert rest == [2, 3, 4]
    assert len(q) == 0


def test_bounded_queue_maxsize_clamped_to_at_least_one():
    q = BoundedDropOldestQueue(0)
    q.put("x")
    q.put("y")
    assert q.stats()["queued"] == 1


# ═══════════════════════════════════════════════════════════════════════════
#  OCC parsing
# ═══════════════════════════════════════════════════════════════════════════


def test_parse_occ_valid():
    parsed = _parse_occ("SPY261002C00580000")
    assert parsed == {"underlying": "SPY", "expiration": date(2026, 10, 2), "right": "call", "strike": 580.0}
    parsed_put = _parse_occ("QQQ261225P00450500")
    assert parsed_put["right"] == "put"
    assert parsed_put["strike"] == 450.5


def test_parse_occ_invalid_returns_none():
    assert _parse_occ("") is None
    assert _parse_occ("not-an-occ-symbol") is None
    assert _parse_occ("SPY261399C00580000") is None  # bad month/day


# ═══════════════════════════════════════════════════════════════════════════
#  backoff_delay
# ═══════════════════════════════════════════════════════════════════════════


def test_backoff_delay_exponential_capped():
    assert backoff_delay(0, 1.0, 30.0) == 0.0
    assert backoff_delay(1, 1.0, 30.0) == 1.0
    assert backoff_delay(2, 1.0, 30.0) == 2.0
    assert backoff_delay(3, 1.0, 30.0) == 4.0
    assert backoff_delay(10, 1.0, 30.0) == 30.0  # capped


# ═══════════════════════════════════════════════════════════════════════════
#  run_stream_supervisor (decision logic, no real threads/sleep)
# ═══════════════════════════════════════════════════════════════════════════


class _FakeStream:
    """A minimal stand-in for alpaca-py's DataStream: .run() and .stop()."""

    def __init__(self, behavior):
        self._behavior = behavior  # callable: () -> None (may raise)

    def run(self):
        self._behavior()

    def stop(self):
        pass


def test_run_stream_supervisor_clean_stop_no_restart():
    stop_event = threading.Event()
    made = []

    def make_stream():
        made.append(1)
        stop_event.set()  # simulate an external stop() landing during run()
        return _FakeStream(lambda: None)

    result = run_stream_supervisor("t", make_stream, stop_event, 0.001, 0.01, wait_fn=lambda ev, t: True)
    assert len(made) == 1
    assert result == {"restarts": 0, "gave_up": False, "last_error": ""}


def test_run_stream_supervisor_restarts_on_unrequested_exit_then_stops():
    stop_event = threading.Event()
    calls = {"n": 0}
    waits = []

    def make_stream():
        calls["n"] += 1

        def behavior():
            if calls["n"] >= 3:
                stop_event.set()  # give up after the 3rd fresh stream
            return None  # run() "exits cleanly" but we never asked for it (until call 3)

        return _FakeStream(behavior)

    def wait_fn(ev, timeout):
        waits.append(timeout)
        return ev.is_set()

    result = run_stream_supervisor("t", make_stream, stop_event, 1.0, 30.0, wait_fn=wait_fn)
    assert calls["n"] == 3
    assert result["restarts"] == 2  # 2 unrequested exits before the 3rd sets stop_event itself
    assert result["gave_up"] is False
    assert waits == [1.0, 2.0]  # exponential backoff recorded via the fake wait_fn


def test_run_stream_supervisor_exception_is_treated_as_failure_and_retried():
    stop_event = threading.Event()
    calls = {"n": 0}

    def make_stream():
        calls["n"] += 1
        n = calls["n"]

        def behavior():
            if n == 1:
                raise RuntimeError("boom")
            stop_event.set()

        return _FakeStream(behavior)

    result = run_stream_supervisor("t", make_stream, stop_event, 0.001, 0.01, wait_fn=lambda ev, t: ev.is_set())
    assert calls["n"] == 2
    assert "boom" in result["last_error"]


def test_run_stream_supervisor_gives_up_after_max_consecutive_failures():
    stop_event = threading.Event()

    def make_stream():
        return _FakeStream(lambda: None)  # always exits "cleanly" but unrequested

    result = run_stream_supervisor(
        "t", make_stream, stop_event, 0.001, 0.01, max_consecutive_failures=3, wait_fn=lambda ev, t: False,
    )
    assert result["gave_up"] is True
    assert result["restarts"] == 3


def test_run_stream_supervisor_stop_set_before_first_iteration_never_builds_a_stream():
    stop_event = threading.Event()
    stop_event.set()
    made = []

    def make_stream():
        made.append(1)
        return _FakeStream(lambda: None)

    result = run_stream_supervisor("t", make_stream, stop_event, 0.001, 0.01)
    assert made == []
    assert result == {"restarts": 0, "gave_up": False, "last_error": ""}


def test_run_stream_supervisor_calls_on_stream_created_hook():
    stop_event = threading.Event()
    created = []

    def make_stream():
        stop_event.set()
        return _FakeStream(lambda: None)

    run_stream_supervisor("t", make_stream, stop_event, 0.001, 0.01, on_stream_created=created.append)
    assert len(created) == 1


def test_run_stream_supervisor_make_stream_raising_once_then_succeeding_retries_and_runs():
    """Item 1 CRITICAL: make_stream() itself raising (not just .run()) must
    be treated exactly like a run() failure -- logged, backed off, retried
    -- never silently killing the supervisor loop/thread forever. Confirmed
    reachable in production: alpaca-py's StockDataStream.__init__ raises
    ValueError("only IEX and SIP feeds ar supported") for
    DataFeed.DELAYED_SIP."""
    stop_event = threading.Event()
    attempts = []
    ran = []

    def make_stream():
        attempts.append(1)
        if len(attempts) == 1:
            raise ValueError("only IEX and SIP feeds ar supported")

        def behavior():
            ran.append(1)
            stop_event.set()

        return _FakeStream(behavior)

    waits = []

    def wait_fn(ev, timeout):
        waits.append(timeout)
        return ev.is_set()

    result = run_stream_supervisor("t", make_stream, stop_event, 1.0, 30.0, wait_fn=wait_fn)
    assert len(attempts) == 2          # first raised, second succeeded
    assert ran == [1]                  # the SECOND stream's run() actually executed
    assert result["restarts"] == 1
    assert result["gave_up"] is False
    assert "only IEX and SIP feeds ar supported" in result["last_error"]
    assert waits == [1.0]              # one backoff wait, for the construction failure


def test_run_stream_supervisor_construction_failure_gives_up_after_max_consecutive_failures():
    stop_event = threading.Event()

    def make_stream():
        raise RuntimeError("always broken")

    result = run_stream_supervisor(
        "t", make_stream, stop_event, 0.001, 0.01, max_consecutive_failures=3, wait_fn=lambda ev, t: False,
    )
    assert result["gave_up"] is True
    assert result["restarts"] == 3
    assert "always broken" in result["last_error"]


# ═══════════════════════════════════════════════════════════════════════════
#  StreamSupervisor (thin real-thread wrapper -- bounded, fast)
# ═══════════════════════════════════════════════════════════════════════════


def test_stream_supervisor_start_stop_real_thread_bounded():
    stop_flag = threading.Event()

    def make_stream():
        class S:
            def run(self):
                stop_flag.wait(timeout=5.0)  # blocks until stop() calls .stop()

            def stop(self):
                stop_flag.set()

        return S()

    sup = StreamSupervisor("test", make_stream, 0.01, 0.1)
    sup.start()
    # give the thread a moment to actually construct the stream and start run()
    deadline = time.time() + 2.0
    while sup.current_stream() is None and time.time() < deadline:
        time.sleep(0.01)
    assert sup.current_stream() is not None
    assert sup.is_running() is True
    sup.stop(join_timeout=3.0)
    assert sup.is_running() is False


# ═══════════════════════════════════════════════════════════════════════════
#  Market-hours idling
# ═══════════════════════════════════════════════════════════════════════════


_NO_HOLIDAYS = frozenset()
_NO_EARLY_CLOSE = frozenset()
_A_HOLIDAY = frozenset({date(2026, 12, 25)})
_AN_EARLY_CLOSE = frozenset({date(2026, 11, 27)})


def _et(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=ET)


def test_is_collector_active_normal_day_window():
    holidays, early = _NO_HOLIDAYS, _NO_EARLY_CLOSE
    assert is_collector_active(_et(2026, 9, 25, 9, 24), holidays, early, 5, 5) is False  # before 09:25
    assert is_collector_active(_et(2026, 9, 25, 9, 25), holidays, early, 5, 5) is True   # at 09:25
    assert is_collector_active(_et(2026, 9, 25, 12, 0), holidays, early, 5, 5) is True
    assert is_collector_active(_et(2026, 9, 25, 16, 5), holidays, early, 5, 5) is True   # at 16:05
    assert is_collector_active(_et(2026, 9, 25, 16, 6), holidays, early, 5, 5) is False  # after 16:05


def test_is_collector_active_weekend_is_never_active():
    saturday = _et(2026, 9, 26, 12, 0)  # 2026-09-26 is a Saturday
    assert is_collector_active(saturday, _NO_HOLIDAYS, _NO_EARLY_CLOSE, 5, 5) is False


def test_is_collector_active_holiday_is_never_active():
    assert is_collector_active(_et(2026, 12, 25, 12, 0), _A_HOLIDAY, _NO_EARLY_CLOSE, 5, 5) is False


def test_is_collector_active_early_close_shrinks_window():
    day = date(2026, 11, 27)
    assert is_collector_active(_et(2026, 11, 27, 13, 4), _NO_HOLIDAYS, _AN_EARLY_CLOSE, 5, 5) is True
    assert is_collector_active(_et(2026, 11, 27, 13, 6), _NO_HOLIDAYS, _AN_EARLY_CLOSE, 5, 5) is False
    assert day in _AN_EARLY_CLOSE


def test_is_active_now_uses_real_market_calendar(monkeypatch):
    cfg = FlowCollectorConfig()
    # A known weekday well within normal hours -- 2026-09-25 is a Friday.
    monkeypatch.setattr(collector_mod, "_market_calendar", lambda: (frozenset(), frozenset()))
    assert is_active_now(cfg, now=_et(2026, 9, 25, 12, 0)) is True
    assert is_active_now(cfg, now=_et(2026, 9, 26, 12, 0)) is False  # Saturday


# ═══════════════════════════════════════════════════════════════════════════
#  FlowCollectorService orchestration (with a fake broker; no network)
# ═══════════════════════════════════════════════════════════════════════════


class FakeBroker(FlowBroker):
    def __init__(self):
        self.spot = {"SPY": 580.0, "QQQ": 480.0}
        self.contracts = {"SPY": [], "QQQ": []}
        self.greeks = {}
        self.universe_calls = 0
        self.greeks_calls = 0
        self.last_universe_call_kwargs: dict = {}

    async def get_spot(self, underlying):
        return self.spot.get(underlying)

    async def get_option_universe(self, underlying, exp_min, exp_max, spot=None, band_pct=None):
        self.universe_calls += 1
        self.last_universe_call_kwargs = {"underlying": underlying, "spot": spot, "band_pct": band_pct}
        return self.contracts.get(underlying, [])

    async def get_option_greeks(self, symbols):
        self.greeks_calls += 1
        return {s: self.greeks.get(s, {"delta": 0.5, "gamma": 0.05}) for s in symbols}

    def make_option_stream(self, on_trade, on_quote):
        raise AssertionError("make_option_stream should not be called in these tests")

    def make_stock_stream(self, on_trade, on_quote):
        raise AssertionError("make_stock_stream should not be called in these tests")


def _cfg(tmp_path, **overrides):
    base = dict(
        state_file=str(tmp_path / "selfflow_state.json"),
        compare_file=str(tmp_path / "selfflow_compare.jsonl"),
        tradingbot_dir=str(tmp_path / "no_such_tradingbot_dir"),
        universe_refresh_seconds=900.0,
        aggregation_interval_seconds=15.0,
        comparison_log_interval_seconds=300.0,
    )
    base.update(overrides)
    return FlowCollectorConfig(**base)


def _et_today() -> date:
    """The ET calendar date, matching exactly what
    FlowCollectorService.maybe_refresh_universe computes internally
    (datetime.fromtimestamp(now_epoch, utc).astimezone(ET).date()) --
    NOT date.today() (system-local), which can differ from the ET date
    for several hours a day (e.g. this sandbox runs UTC; UTC 01:00 is
    still the PREVIOUS ET calendar day). Tests that build multi-day-out
    fixtures (DTE math) must use this, or a contract meant to be "2DTE"
    can silently land 1 day off depending on wall-clock time when the
    suite happens to run."""
    from zoneinfo import ZoneInfo as _ZI
    from datetime import timezone as _tz

    return datetime.now(_tz.utc).astimezone(_ZI("America/New_York")).date()


def _occ(underlying, expiry: date, right: str, strike: float) -> str:
    rc = "C" if right == "call" else "P"
    return f"{underlying}{expiry.strftime('%y%m%d')}{rc}{int(round(strike * 1000)):08d}"


def _trade_obj(symbol, price, size, exchange="A", conditions=None, ts=None):
    return SimpleNamespace(symbol=symbol, price=price, size=size, exchange=exchange, conditions=conditions or [], timestamp=ts)


def _quote_obj(symbol, bid, ask, ts=None):
    return SimpleNamespace(symbol=symbol, bid_price=bid, ask_price=ask, bid_size=10, ask_size=10, timestamp=ts)


def test_drain_and_process_routes_trades_and_quotes(tmp_path):
    cfg = _cfg(tmp_path)
    svc = FlowCollectorService(cfg, broker=FakeBroker())
    now = time.time()
    call_sym = _occ("SPY", date.today(), "call", 580.0)

    asyncio.run(svc._on_option_quote(_quote_obj(call_sym, 1.0, 1.2)))
    asyncio.run(svc._on_option_trade(_trade_obj(call_sym, 1.2, 10)))
    asyncio.run(svc._on_stock_quote(_quote_obj("SPY", 579.9, 580.1)))
    asyncio.run(svc._on_stock_trade(_trade_obj("SPY", 580.1, 100)))

    stats = svc.drain_and_process(now)
    assert stats == {"option_drained": 2, "stock_drained": 2}
    st = svc._aggregator.state["SPY"]
    assert len(st.options) == 1
    assert len(st.equity) == 1
    assert st.options.window(now, 300.0)[0].side == "buy"  # traded at the ask


def test_drain_and_process_ignores_unparsable_or_off_universe_symbols(tmp_path):
    cfg = _cfg(tmp_path)
    svc = FlowCollectorService(cfg, broker=FakeBroker())
    asyncio.run(svc._on_option_trade(_trade_obj("not-an-occ", 1.0, 10)))
    asyncio.run(svc._on_option_trade(_trade_obj(_occ("IWM", date.today(), "call", 200.0), 1.0, 10)))
    asyncio.run(svc._on_stock_trade(_trade_obj("IWM", 200.0, 10)))
    svc.drain_and_process(time.time())
    assert len(svc._aggregator.state["SPY"].options) == 0
    assert len(svc._aggregator.state["QQQ"].options) == 0


def test_process_option_trade_drops_odd_lot_condition(tmp_path):
    cfg = _cfg(tmp_path)
    svc = FlowCollectorService(cfg, broker=FakeBroker())
    call_sym = _occ("SPY", date.today(), "call", 580.0)
    asyncio.run(svc._on_option_trade(_trade_obj(call_sym, 1.2, 10, conditions=["I"])))
    svc.drain_and_process(time.time())
    assert len(svc._aggregator.state["SPY"].options) == 0


def test_process_stock_trade_keeps_unreliable_dark_pool_prints(tmp_path):
    cfg = _cfg(tmp_path)
    svc = FlowCollectorService(cfg, broker=FakeBroker())
    # An "unreliable" (odd-lot) condition on a D-tape (dark pool) trade is
    # still ingested -- see collector.py's _process_stock_trade docstring
    # comment: the reliability filter targets the lit-tape quote rule, not
    # dark-pool midpoint classification.
    asyncio.run(svc._on_stock_trade(_trade_obj("SPY", 580.0, 500, exchange="D", conditions=["I"])))
    svc.drain_and_process(time.time())
    assert len(svc._aggregator.state["SPY"].equity) == 1
    asyncio.run(svc._on_stock_trade(_trade_obj("SPY", 580.0, 500, exchange="P", conditions=["I"])))
    svc.drain_and_process(time.time())
    assert len(svc._aggregator.state["SPY"].equity) == 1  # the lit odd-lot print was dropped


def test_maybe_refresh_universe_due_then_not_due(tmp_path):
    cfg = _cfg(tmp_path, universe_refresh_seconds=900.0)
    broker = FakeBroker()
    today = _et_today()
    broker.contracts["SPY"] = [
        {"symbol": _occ("SPY", today, "call", 580.0), "expiration_date": today.isoformat(), "strike_price": 580.0, "right": "call", "open_interest": 100},
    ]
    svc = FlowCollectorService(cfg, broker=broker)
    now = time.time()
    refreshed = asyncio.run(svc.maybe_refresh_universe(now))
    assert refreshed is True
    assert broker.universe_calls == 2  # SPY + QQQ
    assert svc._universe.total_count == 1
    # immediately again -- not due yet
    refreshed2 = asyncio.run(svc.maybe_refresh_universe(now + 1.0))
    assert refreshed2 is False
    assert broker.universe_calls == 2


def test_maybe_refresh_universe_due_on_spot_drift(tmp_path):
    cfg = _cfg(tmp_path, universe_refresh_seconds=900.0, universe_spot_drift_pct=0.005)
    broker = FakeBroker()
    svc = FlowCollectorService(cfg, broker=broker)
    now = time.time()
    asyncio.run(svc.maybe_refresh_universe(now))
    assert broker.universe_calls == 2
    broker.spot["SPY"] = 580.0 * 1.02  # 2% drift, well past 0.5% trigger
    asyncio.run(svc.maybe_refresh_universe(now + 5.0))
    assert broker.universe_calls == 4


def test_maybe_refresh_strikes_populates_aggregator_state(tmp_path):
    cfg = _cfg(tmp_path, strikes_refresh_seconds=900.0)
    broker = FakeBroker()
    today = _et_today()
    sym = _occ("SPY", today, "call", 580.0)
    broker.contracts["SPY"] = [
        {"symbol": sym, "expiration_date": today.isoformat(), "strike_price": 580.0, "right": "call", "open_interest": 1000},
    ]
    broker.greeks[sym] = {"delta": 0.55, "gamma": 0.04}
    svc = FlowCollectorService(cfg, broker=broker)
    now = time.time()
    asyncio.run(svc.maybe_refresh_universe(now))
    asyncio.run(svc.maybe_refresh_strikes(now))
    st = svc._aggregator.state["SPY"]
    assert len(st.strikes) == 1
    assert st.strikes[0].call_oi == 1000
    assert st.strikes[0].call_gamma_oi == 1000 * 0.04
    assert st.deltas_by_symbol[sym] == 0.55
    assert st.oi_by_strike[580.0]["call_oi"] == 1000
    assert broker.greeks_calls == 1
    # not due again immediately
    asyncio.run(svc.maybe_refresh_strikes(now + 1.0))
    assert broker.greeks_calls == 1


def test_maybe_refresh_strikes_weights_gamma_per_contract_not_last_seen(tmp_path):
    """Item 2 (HIGH): two contracts (0DTE and 2DTE) sharing the SAME strike,
    with 0DTE gamma >> 2DTE gamma. Before the fix, collector.py overwrote a
    single call_gamma with whichever contract was processed LAST, silently
    discarding the other's gamma entirely -- here that would give either
    (2000 oi * 0.01 = 20) or (2000 oi * 0.30 = 600), both wrong. The correct
    pre-weighted sum is (1000*0.30) + (1000*0.01) = 300 + 10 = 310."""
    cfg = _cfg(tmp_path, strikes_refresh_seconds=900.0)
    broker = FakeBroker()
    today = _et_today()
    sym_0dte = _occ("SPY", today, "call", 580.0)
    sym_2dte = _occ("SPY", today + timedelta(days=2), "call", 580.0)
    broker.contracts["SPY"] = [
        {"symbol": sym_0dte, "expiration_date": today.isoformat(), "strike_price": 580.0, "right": "call", "open_interest": 1000},
        {"symbol": sym_2dte, "expiration_date": (today + timedelta(days=2)).isoformat(), "strike_price": 580.0, "right": "call", "open_interest": 1000},
    ]
    broker.greeks[sym_0dte] = {"delta": 0.50, "gamma": 0.30}   # 0DTE: much larger gamma
    broker.greeks[sym_2dte] = {"delta": 0.45, "gamma": 0.01}   # 2DTE: much smaller gamma
    svc = FlowCollectorService(cfg, broker=broker)
    now = time.time()
    asyncio.run(svc.maybe_refresh_universe(now))
    asyncio.run(svc.maybe_refresh_strikes(now))

    st = svc._aggregator.state["SPY"]
    assert len(st.strikes) == 1  # both contracts share one strike bucket
    assert st.strikes[0].call_oi == 2000
    assert st.strikes[0].call_gamma_oi == pytest.approx((1000 * 0.30) + (1000 * 0.01))
    # Neither the "last processed" nor "first processed" WRONG values leak through:
    assert st.strikes[0].call_gamma_oi != 2000 * 0.30
    assert st.strikes[0].call_gamma_oi != 2000 * 0.01


# ═══════════════════════════════════════════════════════════════════════════
#  Self-computed (Black-Scholes-Merton) greeks fallback (2026-09-25 fix) ───
#  See agent/flowcollector/greeks.py's module docstring for the incident:
#  Alpaca's OPRA snapshot returned NO greeks at all, all day, for every
#  symbol -- collector.py must never let that silently zero out GEX/NOPE.
# ═══════════════════════════════════════════════════════════════════════════


def test_maybe_refresh_strikes_computes_greeks_from_quotes_when_alpaca_greeks_missing(tmp_path):
    """Item 1/3: Alpaca's snapshot has NO greeks (delta/gamma both None,
    mirroring the 2026-09-25 incident) but DOES carry a bid/ask quote --
    maybe_refresh_strikes must self-compute delta/gamma via BSM (source
    "computed") instead of the old behavior (silently skip the symbol,
    leaving gamma at an implicit 0.0)."""
    cfg = _cfg(tmp_path, strikes_refresh_seconds=900.0)
    broker = FakeBroker()
    today = _et_today()
    expiry = today + timedelta(days=1)  # comfortably positive T regardless of wall-clock time
    sym = _occ("SPY", expiry, "call", 580.0)
    broker.contracts["SPY"] = [
        {"symbol": sym, "expiration_date": expiry.isoformat(), "strike_price": 580.0, "right": "call", "open_interest": 1000},
    ]
    broker.greeks[sym] = {"delta": None, "gamma": None, "bid": 2.0, "ask": 2.2, "iv": None}
    svc = FlowCollectorService(cfg, broker=broker)
    now = time.time()
    asyncio.run(svc.maybe_refresh_universe(now))
    asyncio.run(svc.maybe_refresh_strikes(now))
    st = svc._aggregator.state["SPY"]
    assert st.greeks_source_by_symbol[sym] == "computed"
    assert st.greeks_source_counts == {"alpaca": 0, "computed": 1, "interpolated": 0, "missing": 0}
    assert st.deltas_by_symbol[sym] > 0.0  # a call's delta is positive
    assert st.gammas_by_symbol[sym] > 0.0
    assert st.strikes[0].call_gamma_oi == pytest.approx(1000 * st.gammas_by_symbol[sym])
    assert st.gamma_oi_coverage == pytest.approx(1.0)


def test_maybe_refresh_strikes_uses_sane_alpaca_greeks_directly(tmp_path):
    """Sane Alpaca delta+gamma are used AS-IS (source "alpaca") -- no BSM
    computation happens for that contract at all."""
    cfg = _cfg(tmp_path, strikes_refresh_seconds=900.0)
    broker = FakeBroker()
    today = _et_today()
    sym = _occ("SPY", today, "call", 580.0)
    broker.contracts["SPY"] = [
        {"symbol": sym, "expiration_date": today.isoformat(), "strike_price": 580.0, "right": "call", "open_interest": 500},
    ]
    broker.greeks[sym] = {"delta": 0.6, "gamma": 0.05}
    svc = FlowCollectorService(cfg, broker=broker)
    now = time.time()
    asyncio.run(svc.maybe_refresh_universe(now))
    asyncio.run(svc.maybe_refresh_strikes(now))
    st = svc._aggregator.state["SPY"]
    assert st.greeks_source_by_symbol[sym] == "alpaca"
    assert st.greeks_source_counts == {"alpaca": 1, "computed": 0, "interpolated": 0, "missing": 0}
    assert st.deltas_by_symbol[sym] == 0.6
    assert st.gammas_by_symbol[sym] == 0.05
    assert st.gamma_oi_coverage == pytest.approx(1.0)


def test_maybe_refresh_strikes_mixed_sources_alpaca_computed_missing(tmp_path):
    """One contract per source: sane Alpaca greeks, self-computed from a
    quote, and no usable quote at all -- exercises the full precedence
    chain and the resulting OI-weighted coverage ("coverage notes
    correct")."""
    cfg = _cfg(tmp_path, strikes_refresh_seconds=900.0)
    broker = FakeBroker()
    today = _et_today()
    expiry = today + timedelta(days=1)
    sym_alpaca = _occ("SPY", expiry, "call", 580.0)
    sym_computed = _occ("SPY", expiry, "put", 580.0)
    sym_missing = _occ("SPY", expiry, "call", 585.0)
    broker.contracts["SPY"] = [
        {"symbol": sym_alpaca, "expiration_date": expiry.isoformat(), "strike_price": 580.0, "right": "call", "open_interest": 1000},
        {"symbol": sym_computed, "expiration_date": expiry.isoformat(), "strike_price": 580.0, "right": "put", "open_interest": 1000},
        {"symbol": sym_missing, "expiration_date": expiry.isoformat(), "strike_price": 585.0, "right": "call", "open_interest": 1000},
    ]
    broker.greeks[sym_alpaca] = {"delta": 0.55, "gamma": 0.04}
    broker.greeks[sym_computed] = {"delta": None, "gamma": None, "bid": 1.8, "ask": 2.0}
    broker.greeks[sym_missing] = {"delta": None, "gamma": None, "bid": None, "ask": None}
    svc = FlowCollectorService(cfg, broker=broker)
    now = time.time()
    asyncio.run(svc.maybe_refresh_universe(now))
    asyncio.run(svc.maybe_refresh_strikes(now))
    st = svc._aggregator.state["SPY"]
    assert st.greeks_source_by_symbol[sym_alpaca] == "alpaca"
    assert st.greeks_source_by_symbol[sym_computed] == "computed"
    assert st.greeks_source_by_symbol[sym_missing] == "missing"
    assert st.greeks_source_counts == {"alpaca": 1, "computed": 1, "interpolated": 0, "missing": 1}
    # 2 of 3 contracts (2000 of 3000 total OI) ended up with a known gamma.
    assert st.gamma_oi_coverage == pytest.approx(2000 / 3000)


def test_maybe_refresh_strikes_computed_greeks_feed_a_non_neutral_gex(tmp_path):
    """Item 1 end-to-end: greeks entirely self-computed (no Alpaca greeks
    at all -- the exact 2026-09-25 incident shape) at a call-OI-dominated
    strike still produces a non-neutral GEX regime, not the old bug's fake
    "neutral" 0.0 (every contract's gamma silently defaulting to 0.0)."""
    cfg = _cfg(tmp_path, strikes_refresh_seconds=900.0)
    broker = FakeBroker()
    today = _et_today()
    expiry = today + timedelta(days=1)
    call_sym = _occ("SPY", expiry, "call", 580.0)
    put_sym = _occ("SPY", expiry, "put", 580.0)
    broker.contracts["SPY"] = [
        {"symbol": call_sym, "expiration_date": expiry.isoformat(), "strike_price": 580.0, "right": "call", "open_interest": 10000},
        {"symbol": put_sym, "expiration_date": expiry.isoformat(), "strike_price": 580.0, "right": "put", "open_interest": 100},
    ]
    broker.greeks[call_sym] = {"delta": None, "gamma": None, "bid": 2.0, "ask": 2.2}
    broker.greeks[put_sym] = {"delta": None, "gamma": None, "bid": 1.9, "ask": 2.1}
    svc = FlowCollectorService(cfg, broker=broker)
    now = time.time()
    asyncio.run(svc.maybe_refresh_universe(now))
    asyncio.run(svc.maybe_refresh_strikes(now))
    st = svc._aggregator.state["SPY"]
    st.spot = 580.0
    assert st.gamma_oi_coverage == pytest.approx(1.0)

    snap = svc._aggregator.snapshot("SPY", now)
    assert snap["quality_notes"]["gamma_oi_coverage"] == pytest.approx(1.0)
    assert snap["quality_notes"]["greeks_source"] == {"alpaca": 0, "computed": 2, "interpolated": 0, "missing": 0}
    assert snap["quality_notes"]["gammas_loaded"] == 2
    assert snap["gex"] is not None
    assert snap["gex_regime"] == "positive"  # call OI (10000) dwarfs put OI (100)


def test_low_coverage_gates_gex_and_nope_to_none(tmp_path):
    """Item 4 ("no fake zeros"): when most of the OI/traded volume has an
    unknown greek, gex/nope must report None (unavailable) rather than a
    value computed mostly off contracts silently treated as zero."""
    cfg = _cfg(tmp_path, strikes_refresh_seconds=900.0, gex_min_coverage=0.5, nope_min_coverage=0.5)
    broker = FakeBroker()
    today = _et_today()
    expiry = today + timedelta(days=1)
    known_sym = _occ("SPY", expiry, "call", 580.0)
    unknown_sym1 = _occ("SPY", expiry, "put", 580.0)
    unknown_sym2 = _occ("SPY", expiry, "call", 585.0)
    broker.contracts["SPY"] = [
        {"symbol": known_sym, "expiration_date": expiry.isoformat(), "strike_price": 580.0, "right": "call", "open_interest": 1000},
        {"symbol": unknown_sym1, "expiration_date": expiry.isoformat(), "strike_price": 580.0, "right": "put", "open_interest": 1000},
        {"symbol": unknown_sym2, "expiration_date": expiry.isoformat(), "strike_price": 585.0, "right": "call", "open_interest": 1000},
    ]
    broker.greeks[known_sym] = {"delta": 0.55, "gamma": 0.04}
    broker.greeks[unknown_sym1] = {"delta": None, "gamma": None, "bid": None, "ask": None}
    broker.greeks[unknown_sym2] = {"delta": None, "gamma": None, "bid": None, "ask": None}
    svc = FlowCollectorService(cfg, broker=broker)
    now = time.time()
    asyncio.run(svc.maybe_refresh_universe(now))
    asyncio.run(svc.maybe_refresh_strikes(now))
    st = svc._aggregator.state["SPY"]
    st.spot = 580.0
    assert st.gamma_oi_coverage == pytest.approx(1000 / 3000)

    # Option trades mostly on the UNKNOWN-delta contract, plus an equity
    # trade so share_volume is positive -- without the coverage gate, NOPE
    # would otherwise compute to a non-None (if unrepresentative) value.
    asyncio.run(svc._on_option_trade(_trade_obj(known_sym, 2.0, 10)))
    asyncio.run(svc._on_option_trade(_trade_obj(unknown_sym1, 1.9, 50)))
    asyncio.run(svc._on_stock_trade(_trade_obj("SPY", 580.05, 1000)))
    svc.drain_and_process(now)

    snap = svc._aggregator.snapshot("SPY", now)
    assert snap["quality_notes"]["delta_volume_coverage"] == pytest.approx(10 / 60)
    assert snap["gex"] is None
    assert snap["gex_regime"] is None
    assert snap["nope"] is None
    assert snap["quality_notes"]["nope_available"] is False


def test_maybe_recompute_greeks_respects_cadence(tmp_path):
    """Item 3: the lighter recompute pass only runs on
    cfg.greeks_recompute_seconds cadence, independent of (and much faster
    than) strikes_refresh_seconds."""
    cfg = _cfg(tmp_path, strikes_refresh_seconds=900.0, greeks_recompute_seconds=60.0)
    broker = FakeBroker()
    today = _et_today()
    expiry = today + timedelta(days=1)
    sym = _occ("SPY", expiry, "call", 580.0)
    broker.contracts["SPY"] = [
        {"symbol": sym, "expiration_date": expiry.isoformat(), "strike_price": 580.0, "right": "call", "open_interest": 1000},
    ]
    broker.greeks[sym] = {"delta": None, "gamma": None, "bid": None, "ask": None}  # "missing" this REST refresh
    svc = FlowCollectorService(cfg, broker=broker)
    now = time.time()
    asyncio.run(svc.maybe_refresh_universe(now))
    asyncio.run(svc.maybe_refresh_strikes(now))
    st = svc._aggregator.state["SPY"]
    assert st.greeks_source_by_symbol[sym] == "missing"

    # a live NBBO quote for this contract arrives between REST refreshes
    asyncio.run(svc._on_option_quote(_quote_obj(sym, 2.0, 2.2, ts=now)))
    svc.drain_and_process(now)

    touched1 = asyncio.run(svc.maybe_recompute_greeks(now))
    assert touched1.get("SPY") == 1
    assert st.greeks_source_by_symbol[sym] == "computed"
    gamma_after_first = st.gammas_by_symbol[sym]

    # not due again immediately -- a second call inside the cadence is a no-op
    touched2 = asyncio.run(svc.maybe_recompute_greeks(now + 1.0))
    assert touched2.get("SPY", 0) == 0
    assert st.gammas_by_symbol[sym] == gamma_after_first

    # due again once the cadence elapses
    touched3 = asyncio.run(svc.maybe_recompute_greeks(now + 61.0))
    assert touched3.get("SPY") == 1


def test_maybe_recompute_greeks_prefers_live_nbbo_and_skips_alpaca_sourced(tmp_path):
    """Item 3's precedence: a live NBBO quote drives the recompute for a
    "computed" symbol (not the stale REST-snapshot quote it was seeded
    from); an "alpaca"-sourced symbol is left untouched even when a fresh,
    very different live quote is available for it."""
    cfg = _cfg(tmp_path, strikes_refresh_seconds=900.0, greeks_recompute_seconds=60.0)
    broker = FakeBroker()
    today = _et_today()
    expiry = today + timedelta(days=1)
    alpaca_sym = _occ("SPY", expiry, "call", 580.0)
    computed_sym = _occ("SPY", expiry, "put", 580.0)
    broker.contracts["SPY"] = [
        {"symbol": alpaca_sym, "expiration_date": expiry.isoformat(), "strike_price": 580.0, "right": "call", "open_interest": 1000},
        {"symbol": computed_sym, "expiration_date": expiry.isoformat(), "strike_price": 580.0, "right": "put", "open_interest": 1000},
    ]
    broker.greeks[alpaca_sym] = {"delta": 0.55, "gamma": 0.04}
    broker.greeks[computed_sym] = {"delta": None, "gamma": None, "bid": 1.8, "ask": 2.0}  # REST-snapshot-time quote
    svc = FlowCollectorService(cfg, broker=broker)
    now = time.time()
    asyncio.run(svc.maybe_refresh_universe(now))
    asyncio.run(svc.maybe_refresh_strikes(now))
    st = svc._aggregator.state["SPY"]
    assert st.greeks_source_by_symbol[alpaca_sym] == "alpaca"
    assert st.greeks_source_by_symbol[computed_sym] == "computed"
    gamma_from_rest_quote = st.gammas_by_symbol[computed_sym]

    # A very different LIVE quote arrives for BOTH symbols before the next
    # recompute.
    asyncio.run(svc._on_option_quote(_quote_obj(alpaca_sym, 50.0, 55.0, ts=now)))
    asyncio.run(svc._on_option_quote(_quote_obj(computed_sym, 5.0, 5.4, ts=now)))
    svc.drain_and_process(now)

    touched = asyncio.run(svc.maybe_recompute_greeks(now))
    assert touched.get("SPY") == 1  # only the non-alpaca symbol was recomputed
    assert st.greeks_source_by_symbol[alpaca_sym] == "alpaca"
    assert st.deltas_by_symbol[alpaca_sym] == 0.55  # untouched by the live quote
    assert st.gammas_by_symbol[alpaca_sym] == 0.04  # untouched by the live quote
    assert st.greeks_source_by_symbol[computed_sym] == "computed"
    assert st.gammas_by_symbol[computed_sym] != pytest.approx(gamma_from_rest_quote)


def test_alpaca_greeks_empty_batch_warns_once_per_day(tmp_path, caplog):
    """Item 7: a WARNING fires when an entire Alpaca snapshot batch for an
    underlying comes back with zero usable greeks (the 2026-09-25
    incident's signature) -- deduped to once per ET calendar day, not once
    per (much more frequent) strikes refresh."""
    cfg = _cfg(tmp_path, strikes_refresh_seconds=1.0)
    broker = FakeBroker()
    today = _et_today()
    expiry = today + timedelta(days=1)
    sym = _occ("SPY", expiry, "call", 580.0)
    broker.contracts["SPY"] = [
        {"symbol": sym, "expiration_date": expiry.isoformat(), "strike_price": 580.0, "right": "call", "open_interest": 1000},
    ]
    broker.greeks[sym] = {"delta": None, "gamma": None, "bid": None, "ask": None}  # Alpaca gives nothing usable
    svc = FlowCollectorService(cfg, broker=broker)
    now = time.time()
    asyncio.run(svc.maybe_refresh_universe(now))
    with caplog.at_level("WARNING"):
        asyncio.run(svc.maybe_refresh_strikes(now))
        asyncio.run(svc.maybe_refresh_strikes(now + 2.0))  # due again (strikes_refresh_seconds=1.0)
    warnings = [r for r in caplog.records if "Alpaca snapshot greeks empty" in r.message]
    assert len(warnings) == 1


# ═══════════════════════════════════════════════════════════════════════════
#  IV smile fill (2026-09-25 coverage-risk follow-up)
# ═══════════════════════════════════════════════════════════════════════════


def _build_realistic_spy_fixture(broker, spot: float, today, bands):
    """SPY-like fixture: $1-wide strikes across +/-4% of `spot`, for each
    (dte, near_band_pct) in `bands` (deliberately 1/2/3 DTE, not 0/1/2, so
    every contract has a comfortably positive T regardless of wall-clock
    time -- see other tests' `today + timedelta(days=1)` comments for the
    same rationale). Strikes on a $5 increment ("round strikes") carry a
    much larger open interest (20000) than off-round strikes (200), mirroring
    real SPY/QQQ OI concentration. Only strikes within `near_band_pct` of
    spot get a usable two-sided quote (a realistic few-dollar-wide market,
    narrowing toward the wings); everything further out (but still inside
    the full +/-4% universe band) quotes bid=0 -- unusable, exactly the real
    SPY/QQQ far-OTM behavior the coverage-risk directive described. Returns
    (near_oi, total_oi): the OI-weighted coverage a run using ONLY the
    near-band (alpaca/computed) contracts as "known" would have gotten --
    i.e. gamma_oi_coverage WITHOUT the smile fill, for the before/after
    comparison (this collector always runs leg 3 now, so there's no runtime
    toggle to compare against directly -- this analytic number is that
    "before" baseline).
    """
    band_pct = 0.04
    lo = int(spot * (1 - band_pct))
    hi = int(spot * (1 + band_pct)) + 1
    contracts = []
    greeks = {}
    near_oi = 0.0
    total_oi = 0.0
    for dte, near_band_pct in bands:
        expiry = today + timedelta(days=dte)
        for strike in range(lo, hi):
            strike_f = float(strike)
            oi = 20000 if strike % 5 == 0 else 200
            near = abs(strike_f - spot) / spot <= near_band_pct
            for right in ("call", "put"):
                sym = _occ("SPY", expiry, right, strike_f)
                contracts.append({
                    "symbol": sym, "expiration_date": expiry.isoformat(),
                    "strike_price": strike_f, "right": right, "open_interest": oi,
                })
                total_oi += oi
                if near:
                    near_oi += oi
                    mid = max(0.05, 3.0 - abs(strike_f - spot) * 0.05)
                    greeks[sym] = {"delta": None, "gamma": None, "bid": round(mid - 0.05, 2), "ask": round(mid + 0.05, 2)}
                else:
                    greeks[sym] = {"delta": None, "gamma": None, "bid": 0.0, "ask": 0.02}
    broker.contracts["SPY"] = contracts
    broker.greeks.update(greeks)
    return near_oi, total_oi


def test_realistic_fixture_smile_fill_closes_the_coverage_gap(tmp_path):
    """The coverage-risk directive's centerpiece: on a realistic SPY-shaped
    +/-4%/multi-DTE, $1-strike fixture where large OI sits on far-OTM round
    strikes with unusable (bid=0) quotes, the near-band-only ("before")
    coverage sits well under the 0.5 gate -- gex would report unavailable
    most of the day, defeating the fix -- but WITH the smile fill ("after"),
    coverage clears 0.9 and gex/regime are available."""
    from agent.flowcollector.aggregator import compute_gex

    spot = 660.0
    today = _et_today()
    bands = ((1, 0.010), (2, 0.020), (3, 0.020))
    cfg = _cfg(tmp_path, strikes_refresh_seconds=900.0, max_dte=3, gex_min_coverage=0.5)
    broker = FakeBroker()
    broker.spot["SPY"] = spot
    near_oi, total_oi = _build_realistic_spy_fixture(broker, spot, today, bands)
    coverage_before = near_oi / total_oi
    assert coverage_before < 0.5  # the scenario this fix targets: near-only coverage breaches the gate

    # "Before" check: at that coverage, compute_gex's own gate reports
    # unavailable, using an otherwise-healthy (non-empty, positive-spot)
    # strikes list -- isolates the coverage gate itself as the cause, not
    # some other empty-input path.
    from agent.flowcollector.aggregator import StrikeExposure

    before_gex = compute_gex([StrikeExposure(strike=spot, call_oi=1000, call_gamma_oi=40.0)], spot, coverage=coverage_before, min_coverage=cfg.gex_min_coverage)
    assert before_gex["empty"] is True
    assert before_gex["regime"] is None

    # "After": run the REAL pipeline (with the smile fill).
    svc = FlowCollectorService(cfg, broker=broker)
    now = time.time()
    asyncio.run(svc.maybe_refresh_universe(now))
    asyncio.run(svc.maybe_refresh_strikes(now))
    st = svc._aggregator.state["SPY"]
    st.spot = spot

    counts = st.greeks_source_counts
    assert counts["missing"] == 0  # every contract resolved to alpaca/computed/interpolated
    assert counts["interpolated"] > 0  # the fill actually did something
    assert st.gamma_oi_coverage >= 0.9

    snap = svc._aggregator.snapshot("SPY", now)
    assert snap["gex"] is not None
    assert snap["gex_regime"] is not None
    assert snap["quality_notes"]["gamma_oi_coverage"] >= 0.9


def test_smile_fill_never_overrides_alpaca_or_computed_source(tmp_path):
    """A contract that resolved via Alpaca or via a direct quote solve keeps
    EXACTLY that value/source -- the fill step only ever touches contracts
    that were still "missing" after legs 1+2."""
    cfg = _cfg(tmp_path, strikes_refresh_seconds=900.0)
    broker = FakeBroker()
    today = _et_today()
    expiry = today + timedelta(days=1)
    alpaca_sym = _occ("SPY", expiry, "call", 578.0)
    computed_sym = _occ("SPY", expiry, "call", 580.0)
    far_sym = _occ("SPY", expiry, "call", 590.0)  # far-OTM, unusable quote -- must get filled
    broker.contracts["SPY"] = [
        {"symbol": alpaca_sym, "expiration_date": expiry.isoformat(), "strike_price": 578.0, "right": "call", "open_interest": 1000},
        {"symbol": computed_sym, "expiration_date": expiry.isoformat(), "strike_price": 580.0, "right": "call", "open_interest": 1000},
        {"symbol": far_sym, "expiration_date": expiry.isoformat(), "strike_price": 590.0, "right": "call", "open_interest": 1000},
    ]
    # Two usable anchors (alpaca's own iv field + a solved-from-quote iv) so
    # far_sym has >= 2 anchor strikes to fill from (fill_iv_from_smile
    # itself requires that -- a single anchor is tested separately in
    # test_flowcollector_greeks.py).
    broker.greeks[alpaca_sym] = {"delta": 0.55, "gamma": 0.04, "iv": 0.20}  # sane Alpaca greeks + iv
    broker.greeks[computed_sym] = {"delta": None, "gamma": None, "bid": 2.0, "ask": 2.2}  # solved from a good quote
    broker.greeks[far_sym] = {"delta": None, "gamma": None, "bid": 0.0, "ask": 0.02}  # unusable -- fill target
    svc = FlowCollectorService(cfg, broker=broker)
    now = time.time()
    asyncio.run(svc.maybe_refresh_universe(now))
    asyncio.run(svc.maybe_refresh_strikes(now))
    st = svc._aggregator.state["SPY"]

    assert st.greeks_source_by_symbol[alpaca_sym] == "alpaca"
    assert st.deltas_by_symbol[alpaca_sym] == 0.55
    assert st.gammas_by_symbol[alpaca_sym] == 0.04

    assert st.greeks_source_by_symbol[computed_sym] == "computed"
    computed_gamma = st.gammas_by_symbol[computed_sym]

    assert st.greeks_source_by_symbol[far_sym] == "interpolated"
    assert st.gammas_by_symbol[far_sym] > 0.0

    # re-run (1s later, so T barely moves) with the SAME quotes -- the
    # ALPACA value is passed through verbatim regardless of T and must be
    # byte-identical; the COMPUTED value legitimately shifts by a tiny,
    # expected amount as T ticks down (BSM greeks are T-dependent) -- that
    # is normal time-decay, not evidence the fill step touched it.
    asyncio.run(svc.maybe_refresh_strikes(now + 1.0))
    assert st.greeks_source_by_symbol[alpaca_sym] == "alpaca"
    assert st.deltas_by_symbol[alpaca_sym] == 0.55
    assert st.gammas_by_symbol[alpaca_sym] == 0.04
    assert st.greeks_source_by_symbol[computed_sym] == "computed"
    assert st.gammas_by_symbol[computed_sym] == pytest.approx(computed_gamma, rel=1e-3)


def test_maybe_recompute_greeks_refreshes_interpolated_symbols(tmp_path):
    """Item 3: maybe_recompute_greeks re-fills a symbol already marked
    "interpolated" (not just "missing") when the anchor set it's
    interpolated against has changed -- here, a fresh live quote for a NEW,
    closer anchor arrives between refreshes and shifts the interpolated
    contract's value."""
    cfg = _cfg(tmp_path, strikes_refresh_seconds=900.0, greeks_recompute_seconds=60.0)
    broker = FakeBroker()
    today = _et_today()
    expiry = today + timedelta(days=1)
    anchor_lo_sym = _occ("SPY", expiry, "put", 575.0)
    anchor_hi_sym = _occ("SPY", expiry, "call", 585.0)
    target_sym = _occ("SPY", expiry, "call", 580.0)  # far enough OTM-quoted to need a fill
    new_anchor_sym = _occ("SPY", expiry, "call", 579.0)  # arrives later, much closer to the target
    broker.contracts["SPY"] = [
        {"symbol": anchor_lo_sym, "expiration_date": expiry.isoformat(), "strike_price": 575.0, "right": "put", "open_interest": 1000},
        {"symbol": anchor_hi_sym, "expiration_date": expiry.isoformat(), "strike_price": 585.0, "right": "call", "open_interest": 1000},
        {"symbol": target_sym, "expiration_date": expiry.isoformat(), "strike_price": 580.0, "right": "call", "open_interest": 1000},
        {"symbol": new_anchor_sym, "expiration_date": expiry.isoformat(), "strike_price": 579.0, "right": "call", "open_interest": 1000},
    ]
    broker.greeks[anchor_lo_sym] = {"delta": None, "gamma": None, "bid": 1.8, "ask": 2.0}
    broker.greeks[anchor_hi_sym] = {"delta": None, "gamma": None, "bid": 2.2, "ask": 2.4}
    broker.greeks[target_sym] = {"delta": None, "gamma": None, "bid": 0.0, "ask": 0.02}  # unusable -- fill target
    broker.greeks[new_anchor_sym] = {"delta": None, "gamma": None, "bid": 0.0, "ask": 0.02}  # unusable AT strikes refresh time
    svc = FlowCollectorService(cfg, broker=broker)
    now = time.time()
    asyncio.run(svc.maybe_refresh_universe(now))
    asyncio.run(svc.maybe_refresh_strikes(now))
    st = svc._aggregator.state["SPY"]
    assert st.greeks_source_by_symbol[target_sym] == "interpolated"
    assert st.greeks_source_by_symbol[new_anchor_sym] == "interpolated"  # also unresolved yet, also filled
    gamma_before = st.gammas_by_symbol[target_sym]

    # A live quote for new_anchor_sym (strike 579, right next to the target)
    # arrives before the next recompute.
    asyncio.run(svc._on_option_quote(_quote_obj(new_anchor_sym, 2.05, 2.15, ts=now)))
    svc.drain_and_process(now)

    touched = asyncio.run(svc.maybe_recompute_greeks(now))
    assert touched.get("SPY", 0) > 0
    assert st.greeks_source_by_symbol[new_anchor_sym] == "computed"  # upgraded via its own live quote
    assert st.greeks_source_by_symbol[target_sym] == "interpolated"  # still interpolated (no quote of its own)
    assert st.gammas_by_symbol[target_sym] != pytest.approx(gamma_before)  # but RE-filled against the new anchor


def test_maybe_refresh_strikes_spot_none_retries_without_expensive_rest_call(tmp_path):
    """Item 4: a strikes refresh that's due but has no spot yet skips the
    REST call entirely and retries on strikes_spot_retry_seconds, never
    burning the full (much longer) strikes_refresh_seconds cadence on a
    cycle that can't self-compute/interpolate anything."""
    from agent.flowcollector.universe import UniverseState

    cfg = _cfg(tmp_path, strikes_refresh_seconds=900.0, strikes_spot_retry_seconds=30.0)
    broker = FakeBroker()
    today = _et_today()
    expiry = today + timedelta(days=1)
    sym = _occ("SPY", expiry, "call", 580.0)
    svc = FlowCollectorService(cfg, broker=broker)
    now = time.time()
    # Directly inject a non-empty universe with NO spot for SPY -- see this
    # test's own docstring / maybe_refresh_strikes's comment for why this
    # combination shouldn't arise via the normal maybe_refresh_universe path
    # (select_universe itself requires spot > 0 to produce any symbols);
    # this simulates it anyway as defense-in-depth.
    svc._universe = UniverseState(
        symbols_by_underlying={"SPY": [sym], "QQQ": []},
        contracts_by_symbol={sym: {"symbol": sym, "expiration_date": expiry.isoformat(), "strike_price": 580.0, "right": "call", "open_interest": 1000}},
        built_at=now,
        spot_by_underlying={"SPY": None, "QQQ": None},
    )

    asyncio.run(svc.maybe_refresh_strikes(now))
    assert broker.greeks_calls == 0  # the expensive REST call never fired
    assert svc._clock.strikes.get("SPY") is None  # full cadence NOT burned
    assert svc._clock.strikes_spot_retry.get("SPY") == now

    # retrying before strikes_spot_retry_seconds elapses is a no-op
    asyncio.run(svc.maybe_refresh_strikes(now + 1.0))
    assert broker.greeks_calls == 0
    assert svc._clock.strikes_spot_retry.get("SPY") == now  # unchanged -- not due yet

    # once a live spot tick arrives AND the retry window elapses, it proceeds
    asyncio.run(svc._on_stock_trade(_trade_obj("SPY", 580.05, 100)))
    svc.drain_and_process(now + 31.0)
    asyncio.run(svc.maybe_refresh_strikes(now + 31.0))
    assert broker.greeks_calls == 1
    assert svc._clock.strikes.get("SPY") == now + 31.0


def test_maybe_write_state_writes_readable_file(tmp_path):
    cfg = _cfg(tmp_path, aggregation_interval_seconds=1.0)
    svc = FlowCollectorService(cfg, broker=FakeBroker())
    now = time.time()
    wrote = asyncio.run(svc.maybe_write_state(now))
    assert wrote is True
    import json

    with open(cfg.state_file) as f:
        payload = json.load(f)
    assert "SPY" in payload["symbols"]
    assert "QQQ" in payload["symbols"]
    assert payload["as_of"] == now
    assert payload["collector_status"]["universe_size"] == 0
    # cadence: not due again immediately
    wrote2 = asyncio.run(svc.maybe_write_state(now + 0.1))
    assert wrote2 is False


def test_maybe_write_comparison_skips_when_uw_snapshot_unavailable(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, comparison_log_interval_seconds=1.0)
    svc = FlowCollectorService(cfg, broker=FakeBroker())
    monkeypatch.setattr(collector_mod, "_read_uw_bridge_snapshot_sync", lambda tradingbot_dir=None: None)
    now = time.time()
    wrote = asyncio.run(svc.maybe_write_comparison(now))
    # self_entry always exists (Aggregator.snapshot never returns falsy for
    # a configured underlying), so a row is still written with uw.available=False.
    assert wrote is True
    assert cfg.compare_file
    with open(cfg.compare_file) as f:
        lines = f.read().splitlines()
    assert len(lines) == 2  # SPY + QQQ
    import json

    row = json.loads(lines[0])
    assert row["uw"]["available"] is False


def test_maybe_manage_streams_stops_when_inactive_never_touches_broker_streams(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    svc = FlowCollectorService(cfg, broker=FakeBroker())
    monkeypatch.setattr(collector_mod, "is_active_now", lambda cfg, now=None: False)
    asyncio.run(svc.maybe_manage_streams(time.time()))
    assert svc._streams_active is False
    # FakeBroker.make_option_stream/make_stock_stream raise if called -- the
    # fact this doesn't raise proves streams were never started.


def test_maybe_refresh_universe_uses_short_retry_cadence_while_empty(tmp_path):
    """Item 4 (other half): while the universe is empty, maybe_refresh_universe
    must retry on cfg.universe_empty_retry_seconds, NOT the normal
    (15-min-default) universe_refresh_seconds cadence."""
    cfg = _cfg(tmp_path, universe_refresh_seconds=900.0, universe_empty_retry_seconds=10.0)
    broker = FakeBroker()  # contracts stay empty for every underlying
    svc = FlowCollectorService(cfg, broker=broker)
    now = time.time()
    asyncio.run(svc.maybe_refresh_universe(now))
    assert broker.universe_calls == 2  # SPY + QQQ
    # 11s later: NOT due under the normal 900s cadence, but IS due under the
    # 10s empty-retry cadence.
    asyncio.run(svc.maybe_refresh_universe(now + 11.0))
    assert broker.universe_calls == 4


class _ThreadFakeStream:
    """A controllable fake stream that actually blocks in .run() (like a
    real DataStream would) until .stop() is called -- for tests that
    exercise real StreamSupervisor threads via FlowCollectorService."""

    def __init__(self):
        self._stop_flag = threading.Event()
        self.started = threading.Event()

    def run(self):
        self.started.set()
        self._stop_flag.wait(timeout=5.0)

    def stop(self):
        self._stop_flag.set()


class StreamCapableFakeBroker(FakeBroker):
    def __init__(self):
        super().__init__()
        self.option_streams_made = 0
        self.stock_streams_made = 0

    def make_option_stream(self, on_trade, on_quote):
        self.option_streams_made += 1
        return _ThreadFakeStream()

    def make_stock_stream(self, on_trade, on_quote):
        self.stock_streams_made += 1
        return _ThreadFakeStream()


def test_option_stream_never_starts_with_empty_universe_stock_stream_does(tmp_path, monkeypatch):
    """Item 4 CRITICAL: starting the option stream with zero subscribed
    symbols would busy-loop alpaca-py's _run_forever (`await
    asyncio.sleep(0)`) pinning a core. The stock stream has no such
    dependency (SPY/QQQ are static) and must start regardless."""
    cfg = _cfg(tmp_path, universe_refresh_seconds=900.0, universe_empty_retry_seconds=60.0)
    broker = StreamCapableFakeBroker()  # broker.contracts stays {} for every underlying
    svc = FlowCollectorService(cfg, broker=broker)
    monkeypatch.setattr(collector_mod, "is_active_now", lambda cfg, now=None: True)
    try:
        asyncio.run(svc.maybe_manage_streams(time.time()))
        assert broker.stock_streams_made == 1
        assert broker.option_streams_made == 0
        assert svc._option_supervisor is None
        assert svc._stock_supervisor is not None
        assert svc._streams_active is True  # stock alone is enough to be "active"
    finally:
        svc._ensure_streams_stopped()


def test_option_stream_starts_once_universe_becomes_non_empty(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, universe_refresh_seconds=900.0, universe_empty_retry_seconds=10.0)
    broker = StreamCapableFakeBroker()
    svc = FlowCollectorService(cfg, broker=broker)
    monkeypatch.setattr(collector_mod, "is_active_now", lambda cfg, now=None: True)
    try:
        now = time.time()
        asyncio.run(svc.maybe_manage_streams(now))
        assert broker.option_streams_made == 0

        today = _et_today()
        broker.contracts["SPY"] = [
            {"symbol": _occ("SPY", today, "call", 580.0), "expiration_date": today.isoformat(), "strike_price": 580.0, "right": "call", "open_interest": 100},
        ]
        # Past the empty-retry cadence -- maybe_refresh_universe is due again.
        asyncio.run(svc.maybe_manage_streams(now + 11.0))
        assert svc._universe is not None and svc._universe.total_count == 1
        assert broker.option_streams_made == 1
        assert svc._option_supervisor is not None
    finally:
        svc._ensure_streams_stopped()


def test_liveness_recheck_restarts_a_dead_stock_supervisor_thread(tmp_path, monkeypatch):
    """Item 1's "or at least streams_active reflects reality" clause, taken
    further: _ensure_stock_stream_started also actively RESTARTS a
    supervisor whose thread has genuinely died, rather than leaving
    streams_active silently wrong forever."""
    cfg = _cfg(tmp_path)
    broker = StreamCapableFakeBroker()
    svc = FlowCollectorService(cfg, broker=broker)
    monkeypatch.setattr(collector_mod, "is_active_now", lambda cfg, now=None: True)
    try:
        asyncio.run(svc.maybe_manage_streams(time.time()))
        assert broker.stock_streams_made == 1
        first_supervisor = svc._stock_supervisor
        # Simulate the thread having died (not via a normal stop()).
        first_supervisor._thread.join(timeout=0.01)  # no-op if still alive
        first_supervisor._stop_event.set()
        current = first_supervisor.current_stream()
        if current is not None:
            current.stop()
        first_supervisor._thread.join(timeout=3.0)
        assert first_supervisor.is_running() is False

        asyncio.run(svc.maybe_manage_streams(time.time()))
        assert broker.stock_streams_made == 2  # a fresh supervisor was built
        assert svc._stock_supervisor is not first_supervisor
        assert svc._streams_active is True
    finally:
        svc._ensure_streams_stopped()


def _kill_supervisor_thread_unexpectedly(sup):
    """Make a supervisor thread exit WITHOUT a stop() call from the service
    (the 'unexpected death' path the liveness re-check exists for)."""
    sup._stop_event.set()
    current = sup.current_stream()
    if current is not None:
        current.stop()
    sup._thread.join(timeout=3.0)
    assert sup.is_running() is False


def _dead_msgs(caplog, level):
    return [r for r in caplog.records if r.levelname == level and "found dead" in r.getMessage()]


def test_deliberate_stop_then_restart_logs_info_not_error(tmp_path, monkeypatch, caplog):
    """R-19: the 09:25 restart after the overnight deliberate stop is
    expected -- INFO, never the 'found dead' ERROR."""
    cfg = _cfg(tmp_path)
    broker = StreamCapableFakeBroker()
    svc = FlowCollectorService(cfg, broker=broker)
    active = {"on": True}
    monkeypatch.setattr(collector_mod, "is_active_now", lambda cfg, now=None: active["on"])
    try:
        asyncio.run(svc.maybe_manage_streams(time.time()))
        assert broker.stock_streams_made == 1
        active["on"] = False
        asyncio.run(svc.maybe_manage_streams(time.time()))  # overnight: deliberate stop
        assert svc._stock_supervisor.is_running() is False
        active["on"] = True
        with caplog.at_level("INFO", logger="clio.flowcollector.collector"):
            asyncio.run(svc.maybe_manage_streams(time.time()))  # 09:25 restart
        assert broker.stock_streams_made == 2
        assert _dead_msgs(caplog, "ERROR") == []
        infos = [r for r in caplog.records if r.levelname == "INFO" and "after deliberate stop" in r.getMessage()]
        assert len(infos) == 1
    finally:
        svc._ensure_streams_stopped()


def test_deliberate_stop_then_restart_of_the_option_stream_logs_info_not_error(tmp_path, monkeypatch, caplog):
    """R-19 (option half): the option supervisor is recorded as deliberately stopped too, so its 09:25 restart is
    INFO, never the 'found dead' ERROR (the stock-only test cannot see this)."""
    cfg = _cfg(tmp_path)
    broker = StreamCapableFakeBroker()
    today = _et_today()
    broker.contracts["SPY"] = [
        {"symbol": _occ("SPY", today, "call", 580.0), "expiration_date": today.isoformat(), "strike_price": 580.0,
         "right": "call", "open_interest": 100},
    ]
    svc = FlowCollectorService(cfg, broker=broker)
    active = {"on": True}
    monkeypatch.setattr(collector_mod, "is_active_now", lambda cfg, now=None: active["on"])
    try:
        asyncio.run(svc.maybe_manage_streams(time.time()))
        assert broker.option_streams_made == 1 and svc._option_supervisor is not None
        active["on"] = False
        asyncio.run(svc.maybe_manage_streams(time.time()))          # overnight: deliberate stop of BOTH streams
        assert svc._option_supervisor.is_running() is False
        active["on"] = True
        with caplog.at_level("INFO", logger="clio.flowcollector.collector"):
            asyncio.run(svc.maybe_manage_streams(time.time()))      # 09:25 restart
        assert broker.option_streams_made == 2
        assert _dead_msgs(caplog, "ERROR") == []
        infos = [r for r in caplog.records if r.levelname == "INFO" and "option stream supervisor restarting after deliberate stop" in r.getMessage()]
        assert len(infos) == 1
    finally:
        svc._ensure_streams_stopped()


def test_unexpected_death_still_logs_error(tmp_path, monkeypatch, caplog):
    """R-19: a supervisor that dies with NO preceding deliberate stop keeps
    the ERROR -- the downgrade must not swallow real failures."""
    cfg = _cfg(tmp_path)
    broker = StreamCapableFakeBroker()
    svc = FlowCollectorService(cfg, broker=broker)
    monkeypatch.setattr(collector_mod, "is_active_now", lambda cfg, now=None: True)
    try:
        asyncio.run(svc.maybe_manage_streams(time.time()))
        _kill_supervisor_thread_unexpectedly(svc._stock_supervisor)
        with caplog.at_level("INFO", logger="clio.flowcollector.collector"):
            asyncio.run(svc.maybe_manage_streams(time.time()))
        assert broker.stock_streams_made == 2
        assert len(_dead_msgs(caplog, "ERROR")) == 1
    finally:
        svc._ensure_streams_stopped()


def test_deliberate_stop_flag_is_cleared_by_restart(tmp_path, monkeypatch, caplog):
    """R-19: the deliberate-stop record is consumed by the restart. A LATER
    unexpected death (no new stop) must log ERROR again."""
    cfg = _cfg(tmp_path)
    broker = StreamCapableFakeBroker()
    svc = FlowCollectorService(cfg, broker=broker)
    active = {"on": True}
    monkeypatch.setattr(collector_mod, "is_active_now", lambda cfg, now=None: active["on"])
    try:
        asyncio.run(svc.maybe_manage_streams(time.time()))
        active["on"] = False
        asyncio.run(svc.maybe_manage_streams(time.time()))
        active["on"] = True
        asyncio.run(svc.maybe_manage_streams(time.time()))  # deliberate restart (consumes flag)
        assert broker.stock_streams_made == 2
        _kill_supervisor_thread_unexpectedly(svc._stock_supervisor)
        with caplog.at_level("INFO", logger="clio.flowcollector.collector"):
            asyncio.run(svc.maybe_manage_streams(time.time()))
        assert broker.stock_streams_made == 3
        assert len(_dead_msgs(caplog, "ERROR")) == 1
    finally:
        svc._ensure_streams_stopped()


def test_tick_end_to_end_smoke_outside_market_hours(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, aggregation_interval_seconds=1.0, comparison_log_interval_seconds=1.0)
    svc = FlowCollectorService(cfg, broker=FakeBroker())
    monkeypatch.setattr(collector_mod, "is_active_now", lambda cfg, now=None: False)
    result = asyncio.run(svc.tick(time.time()))
    assert result["streams_active"] is False
    assert result["wrote_state"] is True


def test_run_forever_exits_promptly_on_preset_stop_event(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, tick_seconds=5.0)
    svc = FlowCollectorService(cfg, broker=FakeBroker())
    monkeypatch.setattr(collector_mod, "is_active_now", lambda cfg, now=None: False)

    async def _go():
        stop_event = asyncio.Event()
        stop_event.set()  # already stopped before run_forever's first wait
        await asyncio.wait_for(svc.run_forever(stop_event), timeout=5.0)

    asyncio.run(_go())  # must not hang / must not raise
    assert svc._streams_active is False


def test_run_forever_writes_final_state_on_shutdown_marking_streams_inactive(tmp_path, monkeypatch):
    """Item 7: __main__.py's docstring promises "one final state snapshot"
    on shutdown -- before this fix, run_forever's finally block never
    actually wrote one (only the cadence-gated maybe_write_state did, called
    from tick(), which a pre-set stop_event means never runs at all here).
    aggregation_interval_seconds is set absurdly high so the NORMAL periodic
    write path could not possibly be what produced the file -- only the
    shutdown path can."""
    cfg = _cfg(tmp_path, tick_seconds=5.0, aggregation_interval_seconds=999_999.0)
    svc = FlowCollectorService(cfg, broker=FakeBroker())
    monkeypatch.setattr(collector_mod, "is_active_now", lambda cfg, now=None: False)

    async def _go():
        stop_event = asyncio.Event()
        stop_event.set()
        await asyncio.wait_for(svc.run_forever(stop_event), timeout=5.0)

    asyncio.run(_go())
    import json

    with open(cfg.state_file) as f:
        payload = json.load(f)
    assert payload["collector_status"]["streams_active"] is False
    assert payload["collector_status"]["shutdown"] is True
    assert "SPY" in payload["symbols"]


def test_normalize_conditions_bare_str_not_split_into_characters():
    """Item 12: alpaca-py's Trade.conditions is typed as a list, but per the
    SDK's own typing may arrive as a bare str for a single-condition trade;
    tuple(str(c) for c in "AB") would wrongly split it into ('A', 'B')."""
    from agent.flowcollector.collector import _normalize_conditions

    assert _normalize_conditions("AB") == ("AB",)
    assert _normalize_conditions(["A", "B"]) == ("A", "B")
    assert _normalize_conditions(None) == ()
    assert _normalize_conditions(()) == ()
    assert _normalize_conditions(("X",)) == ("X",)


def test_process_option_trade_handles_bare_str_conditions_without_splitting(tmp_path):
    cfg = _cfg(tmp_path)
    svc = FlowCollectorService(cfg, broker=FakeBroker())
    call_sym = _occ("SPY", _et_today(), "call", 580.0)
    asyncio.run(svc._on_option_trade(_trade_obj(call_sym, 1.2, 10, conditions="AB")))
    now = time.time()
    svc.drain_and_process(now)
    events = svc._aggregator.state["SPY"].options.window(now, 300.0)
    assert len(events) == 1
    assert events[0].conditions == ("AB",)


# ═══════════════════════════════════════════════════════════════════════════
#  AlpacaFlowBroker internals (production adapter). No network: clients are
#  pre-seeded fakes on the instance, bypassing the lazy real-SDK construction
#  in _get_trading_client/_get_option_data_client -- but make_option_stream/
#  make_stock_stream construct REAL alpaca-py OptionDataStream/StockDataStream
#  objects (safe: construction alone never touches the network, only .run()
#  does).
# ═══════════════════════════════════════════════════════════════════════════


def _alpaca_cfg(**overrides) -> FlowCollectorConfig:
    base = dict(alpaca_key_id="k", alpaca_secret_key="s")
    base.update(overrides)
    return FlowCollectorConfig(**base)


class _FakeTradingClientForUniverse:
    """Captures every GetOptionContractsRequest it's asked to serve and
    returns a scripted response sequence (one item per call)."""

    def __init__(self, responses):
        self.requests = []
        self._responses = list(responses)

    def get_option_contracts(self, req):
        self.requests.append(req)
        return self._responses.pop(0)


def _contract_obj(symbol, strike, right="call", oi="100", exp="2026-10-02"):
    return SimpleNamespace(
        symbol=symbol, type=right, strike_price=strike, expiration_date=exp,
        open_interest=oi, open_interest_date="2026-09-24",
    )


def test_alpaca_broker_get_option_universe_passes_server_side_band_filter():
    """Item 5: spot/band_pct, when given, must reach Alpaca as a
    server-side strike_price_gte/lte filter -- STRINGS (confirmed
    Optional[str] on the installed alpaca-py's GetOptionContractsRequest),
    padded wider than the client-side band."""
    cfg = _alpaca_cfg(strike_band_pct=0.04)
    broker = AlpacaFlowBroker(cfg)
    resp = SimpleNamespace(option_contracts=[_contract_obj("SPY261002C00580000", 580.0)], next_page_token=None)
    fake_client = _FakeTradingClientForUniverse([resp])
    broker._trading_client = fake_client

    result = asyncio.run(broker.get_option_universe("SPY", date(2026, 9, 25), date(2026, 9, 27), spot=580.0, band_pct=0.04))
    assert result is not None and len(result) == 1
    req = fake_client.requests[0]
    assert isinstance(req.strike_price_gte, str) and isinstance(req.strike_price_lte, str)
    padded = 0.04 * 1.5
    assert req.strike_price_gte == f"{580.0 * (1 - padded):.4f}"
    assert req.strike_price_lte == f"{580.0 * (1 + padded):.4f}"


def test_alpaca_broker_get_option_universe_no_spot_omits_strike_filter():
    cfg = _alpaca_cfg()
    broker = AlpacaFlowBroker(cfg)
    resp = SimpleNamespace(option_contracts=[], next_page_token=None)
    fake_client = _FakeTradingClientForUniverse([resp])
    broker._trading_client = fake_client
    asyncio.run(broker.get_option_universe("SPY", date(2026, 9, 25), date(2026, 9, 27)))
    req = fake_client.requests[0]
    assert req.strike_price_gte is None
    assert req.strike_price_lte is None


def test_alpaca_broker_get_option_universe_warns_when_cap_hit(caplog):
    """Item 5: keep the defensive contract-count cap, but log a WARNING
    (not silently drop) when it's hit."""
    cfg = _alpaca_cfg()
    broker = AlpacaFlowBroker(cfg)
    big_batch = [_contract_obj(f"SPY261002C{i:08d}", float(i)) for i in range(_MAX_UNIVERSE_CONTRACTS + 1)]
    resp = SimpleNamespace(option_contracts=big_batch, next_page_token=None)
    fake_client = _FakeTradingClientForUniverse([resp])
    broker._trading_client = fake_client
    with caplog.at_level("WARNING"):
        result = asyncio.run(broker.get_option_universe("SPY", date(2026, 9, 25), date(2026, 9, 27)))
    assert result is not None
    assert len(result) == _MAX_UNIVERSE_CONTRACTS + 1
    assert any("cap" in rec.message.lower() for rec in caplog.records)


def test_alpaca_broker_get_option_universe_under_cap_no_warning(caplog):
    cfg = _alpaca_cfg()
    broker = AlpacaFlowBroker(cfg)
    resp = SimpleNamespace(option_contracts=[_contract_obj("SPY261002C00580000", 580.0)], next_page_token=None)
    fake_client = _FakeTradingClientForUniverse([resp])
    broker._trading_client = fake_client
    with caplog.at_level("WARNING"):
        asyncio.run(broker.get_option_universe("SPY", date(2026, 9, 25), date(2026, 9, 27)))
    assert not any("cap" in rec.message.lower() for rec in caplog.records)


# ── get_option_greeks chunking (item 8) ────────────────────────────────────


def test_alpaca_broker_get_option_greeks_chunks_merges_and_tolerates_one_chunk_failure(monkeypatch):
    cfg = _alpaca_cfg()
    broker = AlpacaFlowBroker(cfg)
    symbols = [f"SYM{i}" for i in range(_OPTION_GREEKS_CHUNK_SIZE + 10)]  # 2 chunks
    calls = []

    async def fake_chunk(self, chunk):
        calls.append(list(chunk))
        if len(calls) == 1:
            return None  # first chunk fails entirely
        return {s: {"delta": 0.5, "gamma": 0.05} for s in chunk}

    monkeypatch.setattr(AlpacaFlowBroker, "_get_option_greeks_chunk", fake_chunk)
    result = asyncio.run(broker.get_option_greeks(symbols))
    assert len(calls) == 2
    assert len(calls[0]) == _OPTION_GREEKS_CHUNK_SIZE
    assert len(calls[1]) == 10
    assert result is not None
    assert all(s not in result for s in calls[0])  # failed chunk's symbols absent
    assert all(s in result for s in calls[1])       # succeeded chunk's symbols present


def test_alpaca_broker_get_option_greeks_all_chunks_fail_returns_none(monkeypatch):
    cfg = _alpaca_cfg()
    broker = AlpacaFlowBroker(cfg)
    symbols = [f"SYM{i}" for i in range(5)]

    async def fake_chunk(self, chunk):
        return None

    monkeypatch.setattr(AlpacaFlowBroker, "_get_option_greeks_chunk", fake_chunk)
    assert asyncio.run(broker.get_option_greeks(symbols)) is None


def test_alpaca_broker_get_option_greeks_empty_symbols_returns_empty_dict():
    cfg = _alpaca_cfg()
    broker = AlpacaFlowBroker(cfg)
    assert asyncio.run(broker.get_option_greeks([])) == {}


def test_alpaca_broker_get_option_greeks_single_chunk_no_chunking_needed(monkeypatch):
    cfg = _alpaca_cfg()
    broker = AlpacaFlowBroker(cfg)
    calls = []

    async def fake_chunk(self, chunk):
        calls.append(list(chunk))
        return {s: {"delta": 0.1, "gamma": 0.01} for s in chunk}

    monkeypatch.setattr(AlpacaFlowBroker, "_get_option_greeks_chunk", fake_chunk)
    result = asyncio.run(broker.get_option_greeks(["A", "B", "C"]))
    assert len(calls) == 1
    assert set(result.keys()) == {"A", "B", "C"}


# ── Sticky feed downgrade carries into subsequent stream construction (item 6) ──


def test_stream_construction_honors_sticky_feed_downgrade():
    """Item 6: a REST-level 401/403 (get_spot/get_option_greeks) sets the
    sticky downgrade flags; this proves _feed_for_options/_feed_for_stock
    (and therefore make_option_stream/make_stock_stream) honor them on
    every SUBSEQUENT call -- e.g. the next stream a StreamSupervisor
    reconnect builds."""
    cfg = _alpaca_cfg(options_feed="opra", stock_feed="sip")
    broker = AlpacaFlowBroker(cfg)

    opt_stream = broker.make_option_stream(None, None)
    stock_stream = broker.make_stock_stream(None, None)
    assert "opra" in opt_stream._endpoint
    assert opt_stream._endpoint.endswith("/opra")
    assert stock_stream._endpoint.endswith("/sip")

    # Simulate what a REST-level 401/403 sets (see get_option_greeks/get_spot).
    broker._options_feed_downgraded = True
    broker._stock_feed_downgraded = True

    opt_stream2 = broker.make_option_stream(None, None)
    stock_stream2 = broker.make_stock_stream(None, None)
    assert opt_stream2._endpoint.endswith("/indicative")
    assert stock_stream2._endpoint.endswith("/iex")


def test_stock_feed_invalid_value_normalizes_to_iex_defensively(caplog):
    """Item 1: stock_feed="delayed_sip" (or any other unsupported value)
    must never reach StockDataStream's constructor, which raises ValueError
    for anything but IEX/SIP -- normalized to IEX defensively, warned once."""
    cfg = _alpaca_cfg(stock_feed="delayed_sip")
    broker = AlpacaFlowBroker(cfg)
    with caplog.at_level("WARNING"):
        stream = broker.make_stock_stream(None, None)
    assert stream._endpoint.endswith("/iex")
    assert any("not supported by the live stream" in rec.message for rec in caplog.records)

    caplog.clear()
    with caplog.at_level("WARNING"):
        broker.make_stock_stream(None, None)  # second call: warns only once
    assert not any("not supported by the live stream" in rec.message for rec in caplog.records)


def test_zero_total_oi_gates_gex_instead_of_fake_neutral(tmp_path):
    """Contracts present but every OI a literal 0 -> coverage 0.0 (gated),
    not None (which would skip the gate and report GEX 0.0 'neutral') --
    pass-2 review R1 #2."""
    cfg = _cfg(tmp_path, strikes_refresh_seconds=900.0)
    broker = FakeBroker()
    expiry = _et_today() + timedelta(days=1)
    sym = _occ("SPY", expiry, "call", 580.0)
    broker.contracts["SPY"] = [
        {"symbol": sym, "expiration_date": expiry.isoformat(), "strike_price": 580.0, "right": "call", "open_interest": 0},
    ]
    broker.greeks[sym] = {"delta": 0.5, "gamma": 0.05}
    svc = FlowCollectorService(cfg, broker=broker)
    now = time.time()
    asyncio.run(svc.maybe_refresh_universe(now))
    asyncio.run(svc.maybe_refresh_strikes(now))
    st = svc._aggregator.state["SPY"]
    st.spot = 580.0
    assert st.gamma_oi_coverage == 0.0
    snap = svc._aggregator.snapshot("SPY", now)
    assert snap["gex"] is None and snap["gex_regime"] is None


def test_time_to_expiry_computed_once_per_expiry(tmp_path, monkeypatch):
    """T (zoneinfo work) is computed per expiry, not per contract, on the
    event loop -- pass-2 review R1 #4."""
    from agent.flowcollector import collector as collector_mod

    calls = {"n": 0}
    real = collector_mod.greeks_mod.time_to_expiry_years

    def _count(now_dt, expiry):
        calls["n"] += 1
        return real(now_dt, expiry)

    monkeypatch.setattr(collector_mod.greeks_mod, "time_to_expiry_years", _count)
    from datetime import datetime as _dt, timezone as _tz

    memo = collector_mod._t_memo(_dt(2026, 9, 25, 14, 0, tzinfo=_tz.utc))
    exp = date(2026, 9, 28)
    for _ in range(300):
        memo(exp)
    memo(date(2026, 9, 29))
    assert calls["n"] == 2
