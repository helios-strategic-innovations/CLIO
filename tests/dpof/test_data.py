"""Tests for agent/dpof/data.py.

Plugin-free (no pytest-asyncio): async paths are driven with asyncio.run()
inside ordinary test functions. No network, no real files outside tmp_path.
CLIO integrations (agent.integrations.econ_calendar / learning_engine /
uw_signal_bridge / polygon) are replaced wholesale with fake modules injected
into sys.modules (both packages are empty __init__.py files, so this is safe
and needs no real import chain). agent.integrations.uw_flow_intel is used for
real (it has no top-level agent.config/httpx import) with only its network
primitive `_uw_get` monkeypatched, so the pillar-value-mapping in data.py is
exercised against the REAL _compute_nofa/_compute_premium_signals/
_compute_dp_signals/_compute_gex.
"""

import asyncio
import importlib
import subprocess
import sys
import time
import types
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

import agent.dpof.data as data
from agent.dpof.config import JevMode
from agent.dpof.signals import gex_factor as real_gex_factor

ROOT = Path(__file__).resolve().parents[2]


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


# ═══════════════════════════════════════════════════════════════════════════
#  Module isolation
# ═══════════════════════════════════════════════════════════════════════════

def test_importing_data_never_imports_agent_config_or_httpx():
    """Subprocess check (a fresh interpreter, no prior imports) that importing
    agent.dpof.data alone never drags in agent.config or httpx -- the
    load-bearing requirement from the orchestrator's task brief.
    """
    script = (
        "import sys; sys.path.insert(0, %r); import agent.dpof.data; "
        "assert 'agent.config' not in sys.modules, 'agent.config was imported'; "
        "assert 'httpx' not in sys.modules, 'httpx was imported'; "
        "print('ISOLATION_OK')" % str(ROOT)
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "ISOLATION_OK" in result.stdout


def test_sys_modules_check_in_process():
    """Cheaper in-process companion check (subprocess above is the authoritative
    one since this process may have already imported agent.config via other
    test modules in the same session)."""
    assert "agent.dpof.data" in sys.modules
    # sanity: the module itself doesn't define agent.config/httpx as globals
    assert not hasattr(data, "config")
    assert not hasattr(data, "httpx")


# ═══════════════════════════════════════════════════════════════════════════
#  compute_vwap_and_returns (PURE)
# ═══════════════════════════════════════════════════════════════════════════

def _bar(t, o, h, l, c, v):
    return {"t": t, "o": o, "h": h, "l": l, "c": c, "v": v}


def test_vwap_normal_case():
    now = 1000.0
    bars = [
        _bar(now - 900, 100, 101, 99, 100, 1000),
        _bar(now - 600, 100, 102, 100, 101, 1000),
        _bar(now - 300, 101, 103, 101, 102, 1000),
        _bar(now - 60, 102, 103, 102, 103, 1000),
    ]
    out = data.compute_vwap_and_returns(bars, now)
    assert out["last_price"] == 103
    assert out["bars_count"] == 4
    assert out["price_age_seconds"] == 60.0
    assert out["vwap"] is not None
    # ret_5m vs bar at t=now-300 (300s ago, at-or-before now-300)
    assert out["ret_5m"] == pytest.approx((103 - 102) / 102)
    # ret_15m vs bar at t=now-900 (exactly at cutoff)
    assert out["ret_15m"] == pytest.approx((103 - 100) / 100)


def test_vwap_zero_volume_bars_excluded_from_vwap_but_not_price():
    now = 1000.0
    bars = [
        _bar(now - 600, 100, 105, 95, 100, 0),   # zero volume: excluded from VWAP
        _bar(now - 60, 100, 101, 99, 100, 1000),
    ]
    out = data.compute_vwap_and_returns(bars, now)
    # vwap should equal the typical price of the ONLY v>0 bar, not be skewed by the wide zero-vol bar
    assert out["vwap"] == pytest.approx((101 + 99 + 100) / 3.0)
    assert out["last_price"] == 100


def test_vwap_all_zero_volume_returns_none_not_zero():
    now = 1000.0
    bars = [_bar(now - 60, 100, 105, 95, 100, 0)]
    out = data.compute_vwap_and_returns(bars, now)
    assert out["vwap"] is None
    assert out["last_price"] == 100  # last_price still reported


def test_vwap_insufficient_history_returns_none_for_returns():
    now = 1000.0
    bars = [_bar(now - 60, 100, 101, 99, 100, 100)]  # only 1 bar, nothing 5/15m ago
    out = data.compute_vwap_and_returns(bars, now)
    assert out["ret_5m"] is None
    assert out["ret_15m"] is None
    assert out["bars_count"] == 1


def test_vwap_gap_in_history_finds_latest_bar_at_or_before_cutoff():
    now = 1000.0
    # bars at -20min and -1min only; nothing near -5min or -15min exactly
    bars = [
        _bar(now - 1200, 90, 91, 89, 90, 100),
        _bar(now - 60, 100, 101, 99, 100, 100),
    ]
    out = data.compute_vwap_and_returns(bars, now)
    # ret_5m: latest bar at or before now-300 is the -1200s bar
    assert out["ret_5m"] == pytest.approx((100 - 90) / 90)
    assert out["ret_15m"] == pytest.approx((100 - 90) / 90)


def test_vwap_empty_bars_list():
    out = data.compute_vwap_and_returns([], 1000.0)
    assert out == {
        "last_price": None, "vwap": None, "ret_5m": None, "ret_15m": None,
        "bars_count": 0, "price_age_seconds": None,
    }


def test_vwap_unsorted_input_is_sorted_not_rejected():
    now = 1000.0
    bars = [
        _bar(now - 60, 103, 103, 102, 103, 1000),   # out of order: latest first
        _bar(now - 900, 100, 101, 99, 100, 1000),
        _bar(now - 300, 101, 103, 101, 102, 1000),
    ]
    out = data.compute_vwap_and_returns(bars, now)
    # last_price must be the bar with the LATEST t (103), not the last list element
    assert out["last_price"] == 103
    assert out["price_age_seconds"] == 60.0


def test_vwap_malformed_bars_skipped_defensively():
    now = 1000.0
    bars = [
        "not a dict",
        {"t": None, "c": 100, "v": 10},           # missing t
        {"t": now - 120, "c": "garbage", "v": 10},  # non-numeric c
        {"t": now - 60, "c": 100, "v": 10},        # valid
    ]
    out = data.compute_vwap_and_returns(bars, now)
    assert out["bars_count"] == 1
    assert out["last_price"] == 100


def test_vwap_zero_close_guard_no_zerodivisionerror():
    now = 1000.0
    bars = [
        _bar(now - 900, 100, 101, 99, 0, 1000),   # zero close far back
        _bar(now - 60, 100, 101, 99, 100, 1000),
    ]
    out = data.compute_vwap_and_returns(bars, now)
    assert out["ret_15m"] is None  # candidate close==0 -> guarded, not inf/nan


# ═══════════════════════════════════════════════════════════════════════════
#  Fakes for CLIO integrations
# ═══════════════════════════════════════════════════════════════════════════

class FakeBroker:
    def __init__(self, bars_by_symbol=None, raise_exc=None, hang=False, delay=0.0):
        self.bars_by_symbol = bars_by_symbol or {}
        self.raise_exc = raise_exc
        self.hang = hang
        self.delay = delay
        self.calls = []

    async def get_minute_bars(self, symbol, start_utc):
        self.calls.append((symbol, start_utc))
        if self.hang:
            await asyncio.sleep(999)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.raise_exc:
            raise self.raise_exc
        return self.bars_by_symbol.get(symbol)


def _install_fake_module(monkeypatch, dotted_name: str, module_obj) -> None:
    parts = dotted_name.split(".")
    parent_name = ".".join(parts[:-1])
    attr_name = parts[-1]
    monkeypatch.setitem(sys.modules, dotted_name, module_obj)
    if parent_name:
        parent = importlib.import_module(parent_name)
        monkeypatch.setattr(parent, attr_name, module_obj, raising=False)


class Fakes:
    """Mutable handles for every CLIO integration point data.py touches."""

    def __init__(self, monkeypatch, tmp_path):
        self.monkeypatch = monkeypatch

        # ── econ_calendar ──
        econ_mod = types.ModuleType("agent.integrations.econ_calendar")
        self.econ_events: list[dict] = []
        self._econ_path = tmp_path / "econ_calendar.json"
        self._econ_path.write_text("{}")  # present by default
        econ_mod.CALENDAR_CACHE_PATH = str(self._econ_path)
        econ_mod.load_calendar = lambda: self.econ_events
        _install_fake_module(monkeypatch, "agent.integrations.econ_calendar", econ_mod)

        # ── learning_engine ──
        learn_mod = types.ModuleType("agent.integrations.learning_engine")
        self.learning_state: dict = {}
        self._learn_path = tmp_path / "learning_state.json"
        self._learn_path.write_text("{}")  # present by default
        learn_mod.LEARNING_STATE_FILE = str(self._learn_path)
        learn_mod._load_learning_state = lambda: self.learning_state
        _install_fake_module(monkeypatch, "agent.integrations.learning_engine", learn_mod)

        # ── uw_signal_bridge ──
        bridge_mod = types.ModuleType("agent.integrations.uw_signal_bridge")
        self.vol_fresh = True
        self.vol_regime: dict = {}
        outer = self

        class _FakeBridge:
            def is_fresh(inner_self):
                return outer.vol_fresh

            def get_vol_regime(inner_self):
                return outer.vol_regime

        bridge_mod.get_uw_bridge = lambda: _FakeBridge()
        _install_fake_module(monkeypatch, "agent.integrations.uw_signal_bridge", bridge_mod)

        # ── polygon ──
        poly_mod = types.ModuleType("agent.integrations.polygon")
        self.vix_result: dict = {"last_price": 18.5}
        self.vix_raise = None
        self.vix_hang = False

        async def _fake_get_snapshot_raw(symbol):
            if outer.vix_hang:
                await asyncio.sleep(999)
            if outer.vix_raise:
                raise outer.vix_raise
            return outer.vix_result

        poly_mod.get_snapshot_raw = _fake_get_snapshot_raw
        _install_fake_module(monkeypatch, "agent.integrations.polygon", poly_mod)

        # ── global_news_feeds (headlines for Jev) ──
        news_mod = types.ModuleType("agent.integrations.global_news_feeds")
        news_mod.scan_current_window = lambda: []  # default: no headlines, overridden per test
        _install_fake_module(monkeypatch, "agent.integrations.global_news_feeds", news_mod)
        self._news_mod = news_mod

        # ── uw_flow_intel: REAL module, fake only the network primitive ──
        import agent.integrations.uw_flow_intel as uw_mod
        self.uw_mod = uw_mod
        self.uw_calls: list[tuple[str, dict]] = []
        self.uw_responses: dict[str, object] = {}          # endpoint -> response (or callable(params)->response)
        self.uw_exact_responses: dict[tuple, object] = {}  # (endpoint, sorted param items) -> response
        self.uw_hang = False
        self.uw_delay = 0.0

        async def _fake_uw_get(endpoint, params=None, *, skip_moltbot_budget=False):
            outer.uw_calls.append((endpoint, dict(params or {})))
            if outer.uw_hang:
                await asyncio.sleep(999)
            if outer.uw_delay:
                await asyncio.sleep(outer.uw_delay)
            key = (endpoint, tuple(sorted((params or {}).items())))
            if key in outer.uw_exact_responses:
                resp = outer.uw_exact_responses[key]
            elif endpoint in outer.uw_responses:
                resp = outer.uw_responses[endpoint]
            else:
                resp = {"data": []}
            return resp(params) if callable(resp) else resp

        monkeypatch.setattr(uw_mod, "_uw_get", _fake_uw_get)

    def econ_set_present(self, present: bool) -> None:
        if present:
            self._econ_path.write_text("{}")
        elif self._econ_path.exists():
            self._econ_path.unlink()

    def learning_set_present(self, present: bool) -> None:
        if present:
            self._learn_path.write_text("{}")
        elif self._learn_path.exists():
            self._learn_path.unlink()

    def install_headline_scan(self, fn) -> None:
        """fn is a plain SYNC callable (mirrors the real scan_current_window
        signature: no args, returns a list) -- it runs inside a real executor
        thread, exactly like production."""
        self._news_mod.scan_current_window = fn


class FakeHeadline:
    """Minimal stand-in for global_news_feeds.ParsedHeadline: only the
    attributes data.py actually reads (title, source_name, reliability_tier,
    published_utc) -- deliberately NOT importing the real dataclass so this
    test file has zero dependency on global_news_feeds beyond its documented
    attribute names.
    """

    def __init__(self, title, source_name, reliability_tier, published_utc):
        self.title = title
        self.source_name = source_name
        self.reliability_tier = reliability_tier
        self.published_utc = published_utc


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    return Fakes(monkeypatch, tmp_path)


@pytest.fixture(autouse=True)
def _fast_timeouts(monkeypatch):
    """Keep the default per-call timeouts small so any accidental real hang in
    a test fails fast instead of stalling the suite for 8-15s. Individual
    tests that specifically exercise the timeout/budget behaviour override
    these further down to sub-second values.
    """
    monkeypatch.setattr(data, "_UW_CALL_TIMEOUT_SECONDS", 2.0)
    monkeypatch.setattr(data, "_BARS_CALL_TIMEOUT_SECONDS", 2.0)
    monkeypatch.setattr(data, "_VIX_CALL_TIMEOUT_SECONDS", 2.0)
    monkeypatch.setattr(data, "_FILE_GATE_TIMEOUT_SECONDS", 2.0)
    monkeypatch.setattr(data, "_SNAPSHOT_BUDGET_SECONDS", 5.0)


def _small_cfg(cfg):
    """A cfg with a smaller pillar bar so 2-pillar test snapshots can be
    data_healthy, and short-ish cadences for cadence tests."""
    return replace(cfg, min_pillars=2, uw_refresh_seconds=100, bars_refresh_seconds=50, underlyings=("SPY", "QQQ"))


# ═══════════════════════════════════════════════════════════════════════════
#  Per-pillar: missing / empty / unexpected-shape (x4 pillars)
# ═══════════════════════════════════════════════════════════════════════════

def _run(coro):
    return asyncio.run(coro)


def _nofa_call_key(ticker: str):
    return (data._EP_FLOW_ALERTS, (("limit", 200), ("min_premium", 25000), ("size_greater_oi", True), ("ticker_symbol", ticker)))


def _nofa_alert(ticker: str, is_call: bool, premium: float, epoch: float, vol: int = 500) -> dict:
    return {
        "ticker_symbol": ticker, "total_premium": premium, "type": "CALL" if is_call else "PUT",
        "total_size": vol, "size_greater_oi": True, "created_at": _iso(epoch),
    }


def test_nofa_is_one_call_per_ticker_with_ticker_symbol_param(fakes, cfg):
    """The exact production precedent: GET /option-trades/flow-alerts with
    ticker_symbol=<T>, ONE call per underlying -- never a market-wide sample."""
    c = _small_cfg(cfg)
    fakes.uw_responses[data._EP_FLOW_ALERTS] = {"data": []}
    broker = FakeBroker()
    feed = data.DPOFDataFeed(c, broker)
    _run(feed.snapshot(now_epoch=1000.0))
    nofa_calls = [(ep, p) for ep, p in fakes.uw_calls if ep == data._EP_FLOW_ALERTS]
    assert len(nofa_calls) == 2  # SPY + QQQ, one each
    tickers_called = {p.get("ticker_symbol") for _ep, p in nofa_calls}
    assert tickers_called == {"SPY", "QQQ"}
    for _ep, p in nofa_calls:
        assert p == {"limit": 200, "min_premium": 25000, "size_greater_oi": True, "ticker_symbol": p["ticker_symbol"]}


def test_nofa_missing_on_uw_failure_no_data_key(fakes, cfg):
    """{} (no 'data' key) from _uw_get -> pillar missing, never 0.0."""
    c = _small_cfg(cfg)
    fakes.uw_responses[data._EP_FLOW_ALERTS] = {}
    broker = FakeBroker()
    feed = data.DPOFDataFeed(c, broker)
    snap = _run(feed.snapshot(now_epoch=1000.0))
    nofa = snap.underlyings["SPY"].pillars["nofa"]
    assert nofa.available is False
    assert nofa.value is None
    assert nofa.error and "no 'data' key" in nofa.error


def test_nofa_genuine_empty_is_zero_with_empty_flag(fakes, cfg):
    c = _small_cfg(cfg)
    fakes.uw_responses[data._EP_FLOW_ALERTS] = {"data": []}
    broker = FakeBroker()
    feed = data.DPOFDataFeed(c, broker)
    snap = _run(feed.snapshot(now_epoch=1000.0))
    nofa = snap.underlyings["SPY"].pillars["nofa"]
    assert nofa.available is True
    assert nofa.value == 0.0
    assert nofa.detail.get("empty") is True


def test_nofa_unexpected_shape_list_instead_of_dict(fakes, cfg):
    c = _small_cfg(cfg)
    # SPY's call returns a bare list -- the exact Hydrinia "expected dict, got
    # list" failure mode. QQQ's call is independent and still succeeds.
    fakes.uw_exact_responses[_nofa_call_key("SPY")] = []
    fakes.uw_exact_responses[_nofa_call_key("QQQ")] = {"data": []}
    broker = FakeBroker()
    feed = data.DPOFDataFeed(c, broker)
    snap = _run(feed.snapshot(now_epoch=1000.0))
    spy_nofa = snap.underlyings["SPY"].pillars["nofa"]
    qqq_nofa = snap.underlyings["QQQ"].pillars["nofa"]
    assert spy_nofa.available is False
    assert spy_nofa.value is None
    assert "expected dict, got list" in spy_nofa.error
    assert qqq_nofa.available is True  # one ticker's failure never blocks the other


def test_nofa_window_filters_out_old_alerts(fakes, cfg):
    """Alerts older than _NOFA_WINDOW_SECONDS (1800s) must not count."""
    c = _small_cfg(cfg)
    now_epoch = 1_000_000.0
    old_alerts = [_nofa_alert("SPY", True, 50000, now_epoch - 3600) for _ in range(5)]  # 1h old
    fakes.uw_exact_responses[_nofa_call_key("SPY")] = {"data": old_alerts}
    fakes.uw_responses[data._EP_FLOW_ALERTS] = {"data": []}
    broker = FakeBroker()
    feed = data.DPOFDataFeed(c, broker)
    snap = _run(feed.snapshot(now_epoch=now_epoch))
    nofa = snap.underlyings["SPY"].pillars["nofa"]
    assert nofa.available is True
    assert nofa.value == 0.0
    assert nofa.detail.get("empty") is True  # nothing survived the window filter


def test_nofa_drops_mismatched_ticker_defensively(fakes, cfg):
    c = _small_cfg(cfg)
    now_epoch = 1_000_000.0
    alerts = [_nofa_alert("QQQ", True, 50000, now_epoch) for _ in range(5)]  # wrong ticker in the SPY response
    fakes.uw_exact_responses[_nofa_call_key("SPY")] = {"data": alerts}
    fakes.uw_responses[data._EP_FLOW_ALERTS] = {"data": []}
    broker = FakeBroker()
    feed = data.DPOFDataFeed(c, broker)
    snap = _run(feed.snapshot(now_epoch=now_epoch))
    nofa = snap.underlyings["SPY"].pillars["nofa"]
    assert nofa.value == 0.0
    assert nofa.detail.get("empty") is True


def test_nofa_unparseable_timestamp_dropped_and_counted(fakes, cfg):
    c = _small_cfg(cfg)
    now_epoch = 1_000_000.0
    alerts = [
        {"ticker_symbol": "SPY", "total_premium": 50000, "type": "CALL", "total_size": 10, "created_at": "not-a-timestamp"},
        {"ticker_symbol": "SPY", "total_premium": 50000, "type": "CALL", "total_size": 10},  # no timestamp field at all
    ]
    fakes.uw_exact_responses[_nofa_call_key("SPY")] = {"data": alerts}
    fakes.uw_responses[data._EP_FLOW_ALERTS] = {"data": []}
    broker = FakeBroker()
    feed = data.DPOFDataFeed(c, broker)
    snap = _run(feed.snapshot(now_epoch=now_epoch))
    nofa = snap.underlyings["SPY"].pillars["nofa"]
    assert nofa.detail.get("empty") is True
    assert nofa.detail.get("unparsed_ts") == 2


def test_nofa_thin_below_min_alerts_is_zero_not_missing(fakes, cfg):
    c = _small_cfg(cfg)
    now_epoch = 1_000_000.0
    alerts = [_nofa_alert("SPY", True, 100000, now_epoch)]  # only 1, below _NOFA_MIN_ALERTS=3
    fakes.uw_exact_responses[_nofa_call_key("SPY")] = {"data": alerts}
    fakes.uw_responses[data._EP_FLOW_ALERTS] = {"data": []}
    broker = FakeBroker()
    feed = data.DPOFDataFeed(c, broker)
    snap = _run(feed.snapshot(now_epoch=now_epoch))
    nofa = snap.underlyings["SPY"].pillars["nofa"]
    assert nofa.available is True
    assert nofa.value == 0.0
    assert nofa.detail.get("thin") is True
    assert nofa.detail.get("n") == 1


def test_nofa_real_value_via_real_compute_function(fakes, cfg):
    c = _small_cfg(cfg)
    now_epoch = 1_000_000.0
    # distinct premiums: _compute_nofa dedups on (ticker, option_symbol, premium) --
    # identical alerts would collapse to one.
    alerts = [_nofa_alert("SPY", True, 50000 + i, now_epoch) for i in range(4)]  # >= _NOFA_MIN_ALERTS, all calls
    fakes.uw_exact_responses[_nofa_call_key("SPY")] = {"data": alerts}
    fakes.uw_responses[data._EP_FLOW_ALERTS] = {"data": []}
    broker = FakeBroker()
    feed = data.DPOFDataFeed(c, broker)
    snap = _run(feed.snapshot(now_epoch=now_epoch))
    nofa = snap.underlyings["SPY"].pillars["nofa"]
    assert nofa.available is True
    assert nofa.value == pytest.approx(1.0)  # all-call premium -> fully bullish
    assert nofa.detail.get("n") == 4


def test_premium_missing_when_tide_fails(fakes, cfg):
    c = _small_cfg(cfg)
    fakes.uw_responses[data._EP_MARKET_TIDE] = {}  # tide fails -> whole premium pillar down
    broker = FakeBroker()
    feed = data.DPOFDataFeed(c, broker)
    snap = _run(feed.snapshot(now_epoch=1000.0))
    prem = snap.underlyings["SPY"].pillars["premium"]
    assert prem.available is False
    assert "no 'data' key" in prem.error


def test_premium_genuine_empty(fakes, cfg):
    c = _small_cfg(cfg)
    fakes.uw_responses[data._EP_MARKET_TIDE] = {"data": []}
    fakes.uw_responses["/stock/SPY/net-prem-ticks"] = {"data": []}
    fakes.uw_responses["/stock/QQQ/net-prem-ticks"] = {"data": []}
    broker = FakeBroker()
    feed = data.DPOFDataFeed(c, broker)
    snap = _run(feed.snapshot(now_epoch=1000.0))
    prem = snap.underlyings["SPY"].pillars["premium"]
    assert prem.available is True
    assert prem.value == 0.0
    assert prem.detail.get("empty") is True


def test_premium_unexpected_shape(fakes, cfg):
    c = _small_cfg(cfg)
    fakes.uw_responses[data._EP_MARKET_TIDE] = {"data": []}
    fakes.uw_responses["/stock/SPY/net-prem-ticks"] = {"data": "not-a-list"}
    broker = FakeBroker()
    feed = data.DPOFDataFeed(c, broker)
    snap = _run(feed.snapshot(now_epoch=1000.0))
    prem = snap.underlyings["SPY"].pillars["premium"]
    assert prem.available is False
    assert "expected list for 'data'" in prem.error


def test_darkpool_missing_on_failure(fakes, cfg):
    c = _small_cfg(cfg)
    fakes.uw_responses["/darkpool/SPY"] = {}
    broker = FakeBroker()
    feed = data.DPOFDataFeed(c, broker)
    snap = _run(feed.snapshot(now_epoch=1000.0))
    dp = snap.underlyings["SPY"].pillars["darkpool"]
    assert dp.available is False
    assert dp.value is None


def test_darkpool_genuine_empty(fakes, cfg):
    c = _small_cfg(cfg)
    fakes.uw_responses["/darkpool/SPY"] = {"data": []}
    broker = FakeBroker()
    feed = data.DPOFDataFeed(c, broker)
    snap = _run(feed.snapshot(now_epoch=1000.0))
    dp = snap.underlyings["SPY"].pillars["darkpool"]
    assert dp.available is True
    assert dp.value == 0.0
    assert dp.detail.get("empty") is True


def test_darkpool_unexpected_shape(fakes, cfg):
    c = _small_cfg(cfg)
    fakes.uw_responses["/darkpool/SPY"] = []  # bare list, not a dict at all
    broker = FakeBroker()
    feed = data.DPOFDataFeed(c, broker)
    snap = _run(feed.snapshot(now_epoch=1000.0))
    dp = snap.underlyings["SPY"].pillars["darkpool"]
    assert dp.available is False
    assert "expected dict, got list" in dp.error


def test_gex_missing_on_failure(fakes, cfg):
    c = _small_cfg(cfg)
    fakes.uw_responses["/stock/SPY/spot-exposures/strike"] = {}
    broker = FakeBroker()
    feed = data.DPOFDataFeed(c, broker)
    snap = _run(feed.snapshot(now_epoch=1000.0))
    gex = snap.underlyings["SPY"].pillars["gex"]
    assert gex.available is False
    assert gex.value is None
    assert snap.underlyings["SPY"].gex_regime is None


def test_gex_genuine_empty(fakes, cfg):
    c = _small_cfg(cfg)
    fakes.uw_responses["/stock/SPY/spot-exposures/strike"] = {"data": []}
    broker = FakeBroker()
    feed = data.DPOFDataFeed(c, broker)
    snap = _run(feed.snapshot(now_epoch=1000.0))
    gex = snap.underlyings["SPY"].pillars["gex"]
    assert gex.available is True
    assert gex.value == 0.0
    assert gex.detail.get("empty") is True


def test_gex_unexpected_shape(fakes, cfg):
    c = _small_cfg(cfg)
    fakes.uw_responses["/stock/SPY/spot-exposures/strike"] = {"data": {"not": "a list"}}
    broker = FakeBroker()
    feed = data.DPOFDataFeed(c, broker)
    snap = _run(feed.snapshot(now_epoch=1000.0))
    gex = snap.underlyings["SPY"].pillars["gex"]
    assert gex.available is False
    assert "expected list for 'data'" in gex.error


def test_gex_regime_mapping_accelerating_and_stabilizing(fakes, cfg):
    c = _small_cfg(cfg)
    # net_gex < 0 -> ACCELERATING -> +0.5
    fakes.uw_responses["/stock/SPY/spot-exposures/strike"] = {
        "data": [{"strike": 580, "call_gamma_oi": 10, "put_gamma_oi": -50}],
    }
    # net_gex > 0 -> STABILIZING -> -0.3
    fakes.uw_responses["/stock/QQQ/spot-exposures/strike"] = {
        "data": [{"strike": 480, "call_gamma_oi": 100, "put_gamma_oi": -30}],
    }
    broker = FakeBroker()
    feed = data.DPOFDataFeed(c, broker)
    snap = _run(feed.snapshot(now_epoch=1000.0))
    spy_gex = snap.underlyings["SPY"].pillars["gex"]
    qqq_gex = snap.underlyings["QQQ"].pillars["gex"]
    assert spy_gex.value == pytest.approx(0.5)
    assert snap.underlyings["SPY"].gex_regime == "ACCELERATING"
    assert qqq_gex.value == pytest.approx(-0.3)
    assert snap.underlyings["QQQ"].gex_regime == "STABILIZING"


# ═══════════════════════════════════════════════════════════════════════════
#  NOPE / max-pain (decision CONTEXT only -- never a pillar, never a gate)
# ═══════════════════════════════════════════════════════════════════════════

# ── pure parsing helpers ──

def test_nope_parse_picks_latest_by_timestamp():
    entries = [
        {"timestamp": "2026-09-24T14:00:00Z", "nope": "-0.001"},
        {"timestamp": "2026-09-24T14:05:00Z", "nope": "0.002"},  # latest -> wins
        {"timestamp": "2026-09-24T13:55:00Z", "nope": "0.5"},
    ]
    assert data._parse_latest_nope(entries) == pytest.approx(0.002)


def test_nope_parse_uses_nope_field_not_nope_fill():
    entries = [{"timestamp": "2026-09-24T14:00:00Z", "nope": "0.003", "nope_fill": "0.999"}]
    assert data._parse_latest_nope(entries) == pytest.approx(0.003)


def test_nope_parse_empty_or_none_returns_none():
    assert data._parse_latest_nope([]) is None
    assert data._parse_latest_nope(None) is None


def test_nope_parse_malformed_entries_skipped_not_raising():
    entries = [
        "not_a_dict",
        {"timestamp": "not-a-timestamp", "nope": "0.1"},
        {"timestamp": "2026-09-24T14:00:00Z", "nope": None},
        {"timestamp": "2026-09-24T14:00:00Z"},         # no "nope" key at all
        {"nope": "0.1"},                                # no timestamp
        {"timestamp": "2026-09-24T14:00:00Z", "nope": "not_numeric"},
    ]
    assert data._parse_latest_nope(entries) is None


def test_max_pain_parse_picks_nearest_future_expiry():
    entries = [
        {"expiry": "2026-09-30", "max_pain": "480"},
        {"expiry": "2026-09-25", "max_pain": "575"},   # nearest >= today
        {"expiry": "2026-10-05", "max_pain": "600"},
    ]
    val = data._parse_nearest_max_pain(entries, date(2026, 9, 24))
    assert val == pytest.approx(575.0)


def test_max_pain_parse_excludes_past_expiries():
    entries = [
        {"expiry": "2026-09-20", "max_pain": "570"},   # already in the past
        {"expiry": "2026-09-26", "max_pain": "580"},
    ]
    val = data._parse_nearest_max_pain(entries, date(2026, 9, 24))
    assert val == pytest.approx(580.0)


def test_max_pain_parse_today_itself_is_eligible():
    entries = [{"expiry": "2026-09-24", "max_pain": "590"}]
    val = data._parse_nearest_max_pain(entries, date(2026, 9, 24))
    assert val == pytest.approx(590.0)


def test_max_pain_parse_all_past_returns_none():
    entries = [{"expiry": "2026-09-01", "max_pain": "500"}]
    assert data._parse_nearest_max_pain(entries, date(2026, 9, 24)) is None


def test_max_pain_parse_empty_or_none_returns_none():
    assert data._parse_nearest_max_pain([], date(2026, 9, 24)) is None
    assert data._parse_nearest_max_pain(None, date(2026, 9, 24)) is None


def test_max_pain_parse_malformed_entries_skipped_not_raising():
    entries = [
        "not_a_dict",
        {"expiry": None, "max_pain": "500"},
        {"expiry": "bad-date-format", "max_pain": "500"},
        {"max_pain": "500"},                            # no expiry
        {"expiry": "2026-09-25", "max_pain": None},
        {"expiry": "2026-09-25", "max_pain": "not_numeric"},
    ]
    assert data._parse_nearest_max_pain(entries, date(2026, 9, 24)) is None


# ── DPOFDataFeed / snapshot() integration ──

def test_nope_and_max_pain_populated_from_uw(fakes, cfg):
    c = replace(cfg, min_pillars=1, underlyings=("SPY",))
    _healthy_uw_responses(fakes)
    fakes.uw_responses[data._EP_NOPE_TICKER.format(ticker="SPY")] = {
        "data": [{"timestamp": "2026-09-24T14:00:00Z", "nope": "-0.0015"}],
    }
    fakes.uw_responses[data._EP_MAX_PAIN_TICKER.format(ticker="SPY")] = {
        "data": [{"expiry": "2026-09-25", "max_pain": "578"}],
    }
    broker = FakeBroker(bars_by_symbol={"SPY": [_bar(999, 100, 101, 99, 100, 10)]})
    feed = data.DPOFDataFeed(c, broker)
    snap = _run(feed.snapshot(now_epoch=1000.0))
    spy = snap.underlyings["SPY"]
    assert spy.nope == pytest.approx(-0.0015)
    assert spy.max_pain == pytest.approx(578.0)


def test_nope_missing_on_uw_failure(fakes, cfg):
    c = replace(cfg, min_pillars=1, underlyings=("SPY",))
    _healthy_uw_responses(fakes)
    fakes.uw_responses[data._EP_NOPE_TICKER.format(ticker="SPY")] = {}  # no "data" key -> FAILURE
    broker = FakeBroker(bars_by_symbol={"SPY": [_bar(999, 100, 101, 99, 100, 10)]})
    feed = data.DPOFDataFeed(c, broker)
    snap = _run(feed.snapshot(now_epoch=1000.0))
    assert snap.underlyings["SPY"].nope is None


def test_max_pain_missing_on_uw_failure(fakes, cfg):
    c = replace(cfg, min_pillars=1, underlyings=("SPY",))
    _healthy_uw_responses(fakes)
    fakes.uw_responses[data._EP_MAX_PAIN_TICKER.format(ticker="SPY")] = {}  # no "data" key -> FAILURE
    broker = FakeBroker(bars_by_symbol={"SPY": [_bar(999, 100, 101, 99, 100, 10)]})
    feed = data.DPOFDataFeed(c, broker)
    snap = _run(feed.snapshot(now_epoch=1000.0))
    assert snap.underlyings["SPY"].max_pain is None


def test_nope_genuine_empty_data_is_none_never_a_fabricated_zero(fakes, cfg):
    c = replace(cfg, min_pillars=1, underlyings=("SPY",))
    _healthy_uw_responses(fakes)
    fakes.uw_responses[data._EP_NOPE_TICKER.format(ticker="SPY")] = {"data": []}
    fakes.uw_responses[data._EP_MAX_PAIN_TICKER.format(ticker="SPY")] = {"data": []}
    broker = FakeBroker(bars_by_symbol={"SPY": [_bar(999, 100, 101, 99, 100, 10)]})
    feed = data.DPOFDataFeed(c, broker)
    snap = _run(feed.snapshot(now_epoch=1000.0))
    # Unlike a pillar, NOPE/max-pain have no legitimate "zero" default -- a
    # genuinely empty UW response means "no reading", i.e. None, not 0.0.
    assert snap.underlyings["SPY"].nope is None
    assert snap.underlyings["SPY"].max_pain is None


def test_nope_max_pain_stale_after_3x_uw_refresh(fakes, cfg):
    c = replace(cfg, min_pillars=1, underlyings=("SPY",), uw_refresh_seconds=100)
    _healthy_uw_responses(fakes)
    fakes.uw_responses[data._EP_NOPE_TICKER.format(ticker="SPY")] = {
        "data": [{"timestamp": "2026-09-24T14:00:00Z", "nope": "0.001"}],
    }
    fakes.uw_responses[data._EP_MAX_PAIN_TICKER.format(ticker="SPY")] = {
        "data": [{"expiry": "2026-09-25", "max_pain": "578"}],
    }
    broker = FakeBroker(bars_by_symbol={"SPY": [_bar(999, 100, 101, 99, 100, 10)]})
    feed = data.DPOFDataFeed(c, broker)

    snap1 = _run(feed.snapshot(now_epoch=1000.0))
    assert snap1.underlyings["SPY"].nope == pytest.approx(0.001)
    assert snap1.underlyings["SPY"].max_pain == pytest.approx(578.0)

    # Start failing both -- the previous reading is kept and ages, never
    # overwritten with a fail-open default.
    fakes.uw_responses[data._EP_NOPE_TICKER.format(ticker="SPY")] = {}
    fakes.uw_responses[data._EP_MAX_PAIN_TICKER.format(ticker="SPY")] = {}

    snap2 = _run(feed.snapshot(now_epoch=1000.0 + 110))  # age=110s <= 300s (3x100) -> still served
    assert snap2.underlyings["SPY"].nope == pytest.approx(0.001)
    assert snap2.underlyings["SPY"].max_pain == pytest.approx(578.0)

    snap3 = _run(feed.snapshot(now_epoch=1000.0 + 350))  # age > 300s -> None, never fabricated
    assert snap3.underlyings["SPY"].nope is None
    assert snap3.underlyings["SPY"].max_pain is None


def test_nope_max_pain_absence_never_blocks_data_healthy(fakes, cfg):
    """Orchestrator review: NOPE/max-pain are decision CONTEXT only (models.
    UnderlyingSnapshot docstring: 'never a mechanical gate') -- neither
    compute_composite/_composite_computable nor data_healthy may depend on
    them at all."""
    c = replace(cfg, min_pillars=1, underlyings=("SPY",))
    _healthy_uw_responses(fakes)
    fakes.uw_responses[data._EP_NOPE_TICKER.format(ticker="SPY")] = {}       # failing
    fakes.uw_responses[data._EP_MAX_PAIN_TICKER.format(ticker="SPY")] = {}   # failing
    broker = FakeBroker(bars_by_symbol={"SPY": [_bar(999, 100, 101, 99, 100, 10)]})
    feed = data.DPOFDataFeed(c, broker)
    snap = _run(feed.snapshot(now_epoch=1000.0))
    assert snap.underlyings["SPY"].nope is None
    assert snap.underlyings["SPY"].max_pain is None
    assert snap.data_healthy is True


def test_nope_max_pain_refresh_shares_uw_daily_call_cap(fakes, cfg):
    """The nope/max_pain refreshers go through the same _uw_fetch primitive as
    the 4 composite pillars, so they count against (and are skipped by) the
    same daily cap -- never a separate, unbudgeted UW call path."""
    c = replace(cfg, uw_refresh_seconds=10, uw_daily_call_cap=1, min_pillars=1, underlyings=("SPY",))
    fakes.uw_responses[data._EP_FLOW_ALERTS] = {"data": []}
    broker = FakeBroker()
    feed = data.DPOFDataFeed(c, broker)
    _run(feed.snapshot(now_epoch=1000.0))
    nope_calls_before = len([e for e, _p in fakes.uw_calls if e == "/stock/SPY/nope"])
    assert nope_calls_before >= 1  # actually attempted in the first (uncapped) batch

    calls_before = len(fakes.uw_calls)
    snap2 = _run(feed.snapshot(now_epoch=1011.0))  # cap already reached from batch 1
    assert len(fakes.uw_calls) == calls_before  # no new calls, including nope/max_pain
    assert "uw_dpof_cap_reached" in snap2.health_issues


# ═══════════════════════════════════════════════════════════════════════════
#  Cadence caching
# ═══════════════════════════════════════════════════════════════════════════

def test_uw_refresh_cadence_only_fetches_once_per_window(fakes, cfg):
    c = replace(cfg, uw_refresh_seconds=100, min_pillars=2)
    fakes.uw_responses[data._EP_FLOW_ALERTS] = {"data": []}
    fakes.uw_responses[data._EP_MARKET_TIDE] = {"data": []}
    broker = FakeBroker()
    feed = data.DPOFDataFeed(c, broker)

    _run(feed.snapshot(now_epoch=1000.0))
    n_calls_after_first = len(fakes.uw_calls)
    assert n_calls_after_first > 0

    _run(feed.snapshot(now_epoch=1050.0))  # +50s, still inside the 100s window
    assert len(fakes.uw_calls) == n_calls_after_first  # no new calls

    _run(feed.snapshot(now_epoch=1101.0))  # +101s from the first refresh -> due again
    assert len(fakes.uw_calls) > n_calls_after_first


def test_bars_refresh_cadence_and_pin_via_broker_call_count(fakes, cfg):
    c = replace(cfg, bars_refresh_seconds=50, min_pillars=1)
    broker = FakeBroker(bars_by_symbol={"SPY": [_bar(990, 100, 101, 99, 100, 10)], "QQQ": [_bar(990, 400, 401, 399, 400, 10)]})
    feed = data.DPOFDataFeed(c, broker)

    _run(feed.snapshot(now_epoch=1000.0))
    calls_after_first = len(broker.calls)
    assert calls_after_first == 2  # both tickers due on first cycle

    _run(feed.snapshot(now_epoch=1030.0))  # +30s, inside the 50s window
    assert len(broker.calls) == calls_after_first

    _run(feed.snapshot(now_epoch=1051.0))  # due again
    assert len(broker.calls) > calls_after_first


def test_pillar_goes_missing_after_3x_uw_refresh_staleness(fakes, cfg):
    c = replace(cfg, uw_refresh_seconds=100, min_pillars=1)
    fakes.uw_responses[data._EP_FLOW_ALERTS] = {
        "data": [{"ticker_symbol": "SPY", "total_premium": 1000, "type": "CALL", "total_size": 10,
                   "size_greater_oi": True, "is_otm": True}],
    }
    broker = FakeBroker()
    feed = data.DPOFDataFeed(c, broker)

    # cycle 1: successful fetch at t=1000
    snap1 = _run(feed.snapshot(now_epoch=1000.0))
    assert snap1.underlyings["SPY"].pillars["nofa"].available is True

    # start failing all subsequent NOFA fetches, but keep triggering "due" refreshes
    fakes.uw_responses[data._EP_FLOW_ALERTS] = {}

    snap2 = _run(feed.snapshot(now_epoch=1000.0 + 110))    # age=110s <= 300s (3x100) -> still available (stale but within window)
    assert snap2.underlyings["SPY"].pillars["nofa"].available is True

    snap3 = _run(feed.snapshot(now_epoch=1000.0 + 350))    # a 3rd refresh attempt has happened by now; age > 300s -> missing
    nofa3 = snap3.underlyings["SPY"].pillars["nofa"]
    assert nofa3.available is False
    assert "stale" in nofa3.error


# ═══════════════════════════════════════════════════════════════════════════
#  UW daily call cap
# ═══════════════════════════════════════════════════════════════════════════

def test_uw_daily_cap_stops_calls_and_adds_health_issue(fakes, cfg):
    c = replace(cfg, uw_refresh_seconds=10, uw_daily_call_cap=3, min_pillars=1)
    fakes.uw_responses[data._EP_FLOW_ALERTS] = {"data": []}
    fakes.uw_responses[data._EP_MARKET_TIDE] = {"data": []}
    broker = FakeBroker()
    feed = data.DPOFDataFeed(c, broker)

    snap1 = _run(feed.snapshot(now_epoch=1000.0))  # first refresh: several calls, likely already >= cap
    assert len(fakes.uw_calls) >= 3

    calls_before = len(fakes.uw_calls)
    snap2 = _run(feed.snapshot(now_epoch=1011.0))  # next window due, but cap already reached
    assert len(fakes.uw_calls) == calls_before  # no new calls issued
    assert "uw_dpof_cap_reached" in snap2.health_issues


def test_uw_daily_cap_resets_on_et_day_change(fakes, cfg):
    c = replace(cfg, uw_refresh_seconds=10, uw_daily_call_cap=3, min_pillars=1)
    fakes.uw_responses[data._EP_FLOW_ALERTS] = {"data": []}
    fakes.uw_responses[data._EP_MARKET_TIDE] = {"data": []}
    broker = FakeBroker()
    feed = data.DPOFDataFeed(c, broker)

    # Day 1, noon ET (2026-01-05 is a Monday) -> epoch for 2026-01-05T12:00:00-05:00
    import datetime as _dt
    day1_noon = _dt.datetime(2026, 1, 5, 17, 0, 0, tzinfo=_dt.timezone.utc).timestamp()  # 12:00 ET
    day2_noon = _dt.datetime(2026, 1, 6, 17, 0, 0, tzinfo=_dt.timezone.utc).timestamp()

    _run(feed.snapshot(now_epoch=day1_noon))
    snap_capped = _run(feed.snapshot(now_epoch=day1_noon + 11))
    assert "uw_dpof_cap_reached" in snap_capped.health_issues

    calls_before_new_day = len(fakes.uw_calls)
    snap_next_day = _run(feed.snapshot(now_epoch=day2_noon))
    assert len(fakes.uw_calls) > calls_before_new_day  # cap reset -> calls resumed
    assert "uw_dpof_cap_reached" not in snap_next_day.health_issues


# ═══════════════════════════════════════════════════════════════════════════
#  Gates / data_healthy / health_issues
# ═══════════════════════════════════════════════════════════════════════════

def _healthy_uw_responses(fakes):
    fakes.uw_responses[data._EP_FLOW_ALERTS] = {
        "data": [{"ticker_symbol": "SPY", "total_premium": 1000, "type": "CALL", "total_size": 10,
                   "size_greater_oi": True, "is_otm": True}],
    }
    fakes.uw_responses[data._EP_MARKET_TIDE] = {"data": []}
    fakes.uw_responses["/stock/SPY/net-prem-ticks"] = {"data": []}
    fakes.uw_responses["/darkpool/SPY"] = {"data": []}


def test_econ_gate_source_failure_no_longer_forces_unhealthy(fakes, cfg):
    """Task 1 (2026-09-25, operator directive): econ-calendar gate-SOURCE
    availability no longer forces data_healthy False -- data_healthy is
    strictly about the flow pillars/price data DPOF actually trades on
    (unchanged: healthy UW pillars + fresh price here). The health_issues
    note is still recorded (diagnostic only, no longer a veto input either --
    see signals.py); calendar_status (get_intel_ctx) is the advisory home for
    this now."""
    c = replace(cfg, min_pillars=1, underlyings=("SPY",))
    _healthy_uw_responses(fakes)
    fakes.econ_set_present(False)  # gate source down
    broker = FakeBroker(bars_by_symbol={"SPY": [_bar(999, 100, 101, 99, 100, 10)]})
    feed = data.DPOFDataFeed(c, broker)
    snap = _run(feed.snapshot(now_epoch=1000.0))
    assert snap.data_healthy is True
    assert "gate_source_unavailable:econ_calendar" in snap.health_issues
    intel = _run(feed.get_intel_ctx(now_epoch=1000.0))
    assert intel["calendar_status"] == {"status": "missing", "age_hours": None, "sources": ["econ_calendar"]}


def test_stale_econ_calendar_cache_no_longer_forces_unhealthy(fakes, cfg):
    """Same relaxation for a present-but-STALE calendar cache (daily refresh
    broken) -- Task 1: it is calendar_status["stale"] context now, not a
    health/entry block."""
    import os as _os
    import time as _time
    c = replace(cfg, min_pillars=1, underlyings=("SPY",))
    _healthy_uw_responses(fakes)
    old = _time.time() - 3 * 86400
    _os.utime(fakes._econ_path, (old, old))
    broker = FakeBroker(bars_by_symbol={"SPY": [_bar(999, 100, 101, 99, 100, 10)]})
    feed = data.DPOFDataFeed(c, broker)
    snap = _run(feed.snapshot(now_epoch=1000.0))
    assert snap.data_healthy is True
    assert "gate_source_unavailable:econ_calendar" in snap.health_issues
    intel = _run(feed.get_intel_ctx(now_epoch=1000.0))
    assert intel["calendar_status"]["status"] == "stale"
    assert intel["calendar_status"]["age_hours"] > 36.0


def test_vol_regime_stale_is_note_not_blocking_flag(fakes, cfg):
    c = replace(cfg, min_pillars=1, underlyings=("SPY",))
    _healthy_uw_responses(fakes)
    fakes.vol_fresh = False  # stale bridge data
    broker = FakeBroker(bars_by_symbol={"SPY": [_bar(999, 100, 101, 99, 100, 10)]})
    feed = data.DPOFDataFeed(c, broker)
    snap = _run(feed.snapshot(now_epoch=1000.0))
    assert "vol_regime_stale" in snap.health_issues
    assert not any(f.startswith("vol_regime:") for f in snap.risk_flags)
    # staleness alone must not force data_healthy False (required sources ok, price fresh, composite computable)
    assert snap.data_healthy is True


def test_vol_regime_fresh_and_blocked_adds_risk_flag(fakes, cfg):
    c = replace(cfg, min_pillars=1, underlyings=("SPY",))
    _healthy_uw_responses(fakes)
    fakes.vol_fresh = True
    fakes.vol_regime = {"regime": "CRISIS"}
    broker = FakeBroker(bars_by_symbol={"SPY": [_bar(999, 100, 101, 99, 100, 10)]})
    feed = data.DPOFDataFeed(c, broker)
    snap = _run(feed.snapshot(now_epoch=1000.0))
    assert "vol_regime:CRISIS" in snap.risk_flags


def test_breaking_news_flag_and_econ_blackout_flag(fakes, cfg):
    c = replace(cfg, min_pillars=1, underlyings=("SPY",), event_context_window_minutes=30)
    _healthy_uw_responses(fakes)
    fakes.learning_state = {"breaking_news": {"active": True}}
    from datetime import datetime, timezone
    now_epoch = 1_800_000_000.0
    now_et_iso = datetime.fromtimestamp(now_epoch, tz=timezone.utc).astimezone(data.ET).isoformat()
    fakes.econ_events = [{"name": "FOMC Rate Decision", "importance": 3, "datetime_et": now_et_iso}]
    broker = FakeBroker(bars_by_symbol={"SPY": [_bar(now_epoch - 1, 100, 101, 99, 100, 10)]})
    feed = data.DPOFDataFeed(c, broker)
    snap = _run(feed.snapshot(now_epoch=now_epoch))
    assert "breaking_news" in snap.risk_flags
    assert any(f.startswith("econ_blackout:FOMC") for f in snap.risk_flags)
    assert len(snap.econ_events_soon) == 1


# ═══════════════════════════════════════════════════════════════════════════
#  2026-09-25: econ_calendar FRED/FOMC-static fallback -- DPOF-side contract
#  (data.py itself is source-agnostic: it never reads the `source` field, only
#  cache file-presence/freshness and each event's importance/datetime_et. These
#  tests document and pin that contract against the fallback's actual event
#  shape from agent/integrations/econ_calendar.py.)
# ═══════════════════════════════════════════════════════════════════════════

def test_econ_blackout_fires_around_fred_sourced_cpi_event(fakes, cfg):
    """A FRED-fallback-sourced CPI event (source='fred', the econ_calendar
    fallback's standard 08:30 ET release time for release_id 10 -- see
    _FRED_RELEASE_TABLE) blocks DPOF entries exactly like a UW-sourced one."""
    c = replace(cfg, min_pillars=1, underlyings=("SPY",), event_context_window_minutes=30)
    _healthy_uw_responses(fakes)
    cpi_dt = datetime(2026, 10, 13, 8, 30, 0, tzinfo=data.ET)
    now_epoch = cpi_dt.timestamp()  # exactly at the event
    fakes.econ_events = [{
        "name": "CPI (Consumer Price Index)",
        "importance": 3,
        "datetime_et": cpi_dt.isoformat(),
        "source": "fred",
    }]
    broker = FakeBroker(bars_by_symbol={"SPY": [_bar(now_epoch - 1, 100, 101, 99, 100, 10)]})
    feed = data.DPOFDataFeed(c, broker)
    snap = _run(feed.snapshot(now_epoch=now_epoch))
    assert any(f.startswith("econ_blackout:CPI") for f in snap.risk_flags)
    assert len(snap.econ_events_soon) == 1
    assert snap.econ_events_soon[0]["source"] == "fred"


def test_econ_gate_healthy_with_fred_only_sourced_calendar(fakes, cfg):
    """A calendar cache populated entirely by the FRED/FOMC-static fallback
    (zero UW rows) still satisfies the health check -- DPOF only reads
    file-presence + freshness (_econ_calendar_read_sync's docstring), never
    per-event `source`. This is the DPOF-side half of 'a UW outage alone no
    longer makes DPOF unhealthy'; the econ_calendar.py half (that
    run_calendar_refresh actually keeps writing the cache file during a UW
    outage) is covered in tests/test_econ_calendar_fred_fallback.py."""
    c = replace(cfg, min_pillars=1, underlyings=("SPY",))
    _healthy_uw_responses(fakes)
    fakes.econ_events = [{
        "name": "FOMC Rate Decision / Statement", "importance": 3,
        "datetime_et": "2026-01-28T14:00:00-05:00", "source": "fomc_static",
    }]
    broker = FakeBroker(bars_by_symbol={"SPY": [_bar(999, 100, 101, 99, 100, 10)]})
    feed = data.DPOFDataFeed(c, broker)
    snap = _run(feed.snapshot(now_epoch=1000.0))
    assert "gate_source_unavailable:econ_calendar" not in snap.health_issues


def test_calendar_status_missing_with_no_source_at_all(fakes, cfg):
    """Neither UW nor the FRED/FOMC fallback ever wrote a cache file this
    cycle (or ever) -- 'missing is never zero' still applies to the ADVISORY
    intel too: calendar_status is "missing" (never fabricated "ok"), and
    econ_surprise's own calendar-derived fallback path (Builder E, landed)
    degrades to an empty upcoming list rather than raising or guessing."""
    c = replace(cfg, min_pillars=1, underlyings=("SPY",))
    _healthy_uw_responses(fakes)
    fakes.econ_set_present(False)
    broker = FakeBroker(bars_by_symbol={"SPY": [_bar(999, 100, 101, 99, 100, 10)]})
    feed = data.DPOFDataFeed(c, broker)
    snap = _run(feed.snapshot(now_epoch=1000.0))
    assert snap.data_healthy is True  # Task 1: flow/price data healthy is all that matters now
    assert "gate_source_unavailable:econ_calendar" in snap.health_issues
    intel = _run(feed.get_intel_ctx(now_epoch=1000.0))
    assert intel["calendar_status"]["status"] == "missing"
    assert intel["events"]["upcoming"] == []


def test_primary_price_stale_forces_unhealthy(fakes, cfg):
    c = replace(cfg, min_pillars=1, underlyings=("SPY",), bars_refresh_seconds=60)
    _healthy_uw_responses(fakes)
    # bars are very old relative to now -> price_age_seconds > 3*bars_refresh_seconds
    broker = FakeBroker(bars_by_symbol={"SPY": [_bar(0, 100, 101, 99, 100, 10)]})
    feed = data.DPOFDataFeed(c, broker)
    snap = _run(feed.snapshot(now_epoch=100000.0))
    assert snap.data_healthy is False
    assert "primary_price_stale_or_missing" in snap.health_issues


def test_headlines_always_empty(fakes, cfg):
    c = replace(cfg, min_pillars=1, underlyings=("SPY",))
    _healthy_uw_responses(fakes)
    broker = FakeBroker(bars_by_symbol={"SPY": [_bar(999, 100, 101, 99, 100, 10)]})
    feed = data.DPOFDataFeed(c, broker)
    snap = _run(feed.snapshot(now_epoch=1000.0))
    assert snap.headlines == []


# ═══════════════════════════════════════════════════════════════════════════
#  Time budget: hangs must not stall snapshot()
# ═══════════════════════════════════════════════════════════════════════════

def test_snapshot_bounded_by_per_call_timeout_when_uw_hangs(fakes, cfg, monkeypatch):
    monkeypatch.setattr(data, "_UW_CALL_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(data, "_SNAPSHOT_BUDGET_SECONDS", 5.0)  # generous outer budget
    c = replace(cfg, min_pillars=1, underlyings=("SPY",))
    fakes.uw_hang = True  # every UW call hangs forever
    broker = FakeBroker(bars_by_symbol={"SPY": [_bar(999, 100, 101, 99, 100, 10)]})
    feed = data.DPOFDataFeed(c, broker)

    start = time.monotonic()
    snap = _run(feed.snapshot(now_epoch=1000.0))
    elapsed = time.monotonic() - start

    assert elapsed < 2.0, f"snapshot() took {elapsed:.2f}s -- per-call UW timeout did not bound it"
    assert snap.underlyings["SPY"].pillars["nofa"].available is False
    assert snap.data_healthy is False


def test_snapshot_bounded_by_outer_budget_as_defense_in_depth(fakes, cfg, monkeypatch):
    # Set the per-call timeouts artificially large so ONLY the outer wait_for
    # can save us -- proves the outer budget is a real, independent backstop.
    monkeypatch.setattr(data, "_UW_CALL_TIMEOUT_SECONDS", 999.0)
    monkeypatch.setattr(data, "_BARS_CALL_TIMEOUT_SECONDS", 999.0)
    monkeypatch.setattr(data, "_VIX_CALL_TIMEOUT_SECONDS", 999.0)
    monkeypatch.setattr(data, "_SNAPSHOT_BUDGET_SECONDS", 0.2)
    c = replace(cfg, min_pillars=1, underlyings=("SPY",))
    fakes.uw_hang = True
    broker = FakeBroker(bars_by_symbol={"SPY": [_bar(999, 100, 101, 99, 100, 10)]}, hang=True)
    feed = data.DPOFDataFeed(c, broker)

    start = time.monotonic()
    snap = _run(feed.snapshot(now_epoch=1000.0))
    elapsed = time.monotonic() - start

    assert elapsed < 2.0, f"snapshot() took {elapsed:.2f}s -- outer snapshot budget did not bound it"
    assert "snapshot_refresh_timeout" in snap.health_issues
    assert snap.data_healthy is False


# ═══════════════════════════════════════════════════════════════════════════
#  CancelledError must propagate, never be swallowed
# ═══════════════════════════════════════════════════════════════════════════

def test_cancelled_error_propagates_out_of_snapshot(fakes, cfg):
    c = replace(cfg, min_pillars=1, underlyings=("SPY",))

    class CancellingBroker:
        async def get_minute_bars(self, symbol, start_utc):
            raise asyncio.CancelledError()

    feed = data.DPOFDataFeed(c, CancellingBroker())

    async def _go():
        await feed.snapshot(now_epoch=1000.0)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_go())


# ═══════════════════════════════════════════════════════════════════════════
#  Misc pure helpers
# ═══════════════════════════════════════════════════════════════════════════

def test_session_start_utc_is_930_et_converted_to_utc():
    from datetime import datetime, timezone
    now_epoch = datetime(2026, 1, 5, 20, 0, 0, tzinfo=timezone.utc).timestamp()  # ~3pm ET on a Monday
    start = data._session_start_utc(now_epoch)
    start_et = start.astimezone(data.ET)
    assert (start_et.hour, start_et.minute) == (9, 30)
    assert start_et.date() == datetime(2026, 1, 5, 20, 0, tzinfo=timezone.utc).astimezone(data.ET).date()


def test_gex_factor_is_imported_from_signals_not_duplicated():
    """data.gex_factor must be the real agent.dpof.signals.gex_factor object
    (no local duplicate) -- and it must correctly map uw_flow_intel._compute_
    gex's UPPERCASE regime labels via its internal .lower()."""
    assert data.gex_factor is real_gex_factor
    assert data.gex_factor("ACCELERATING", None) == pytest.approx(0.5)
    assert data.gex_factor("STABILIZING", None) == pytest.approx(-0.3)
    assert data.gex_factor(None, None) is None
    assert not hasattr(data, "_gex_factor_from_regime")  # the old local duplicate must be gone


def test_shape_check_uw_response_three_outcomes():
    assert data._shape_check_uw_response({}, "x") == (None, "x: no 'data' key in response")
    assert data._shape_check_uw_response({"data": []}, "x") == ([], "")
    lst, err = data._shape_check_uw_response([1, 2], "x")
    assert lst is None and "expected dict, got list" in err
    lst, err = data._shape_check_uw_response({"data": {"a": 1}}, "x")
    assert lst is None and "expected list for 'data', got dict" in err


# ═══════════════════════════════════════════════════════════════════════════
#  Headlines: detached background refresher
# ═══════════════════════════════════════════════════════════════════════════

def _headline_feed(c, fakes, jev_mode=JevMode.SHADOW):
    c = replace(c, jev_mode=jev_mode, min_pillars=1, underlyings=("SPY",))
    _healthy_uw_responses(fakes)
    broker = FakeBroker(bars_by_symbol={"SPY": [_bar(999, 100, 101, 99, 100, 10)]})
    return data.DPOFDataFeed(c, broker)


def test_headline_refresher_not_started_when_jev_off(fakes, cfg):
    feed = _headline_feed(cfg, fakes, jev_mode=JevMode.OFF)
    calls = {"n": 0}
    fakes.install_headline_scan(lambda: (calls.__setitem__("n", calls["n"] + 1), [])[1])

    async def _go():
        await feed.snapshot(now_epoch=1000.0)
        await asyncio.sleep(0.05)

    asyncio.run(_go())
    assert feed._headline_task is None
    assert calls["n"] == 0


def test_headline_refresher_started_once_not_duplicated(fakes, cfg):
    feed = _headline_feed(cfg, fakes)
    calls = {"n": 0}

    def _scan():
        calls["n"] += 1
        time.sleep(0.1)
        return []

    fakes.install_headline_scan(_scan)

    async def _go():
        await feed.snapshot(now_epoch=1000.0)
        assert feed._headline_task is not None
        # second snapshot fires quickly, well before the 0.1s scan finishes --
        # must NOT start a second task.
        await feed.snapshot(now_epoch=1000.2)
        await asyncio.sleep(0.3)  # let the in-flight task finish

    asyncio.run(_go())
    assert calls["n"] == 1


def test_headline_stale_cache_not_served(fakes, cfg):
    feed = _headline_feed(cfg, fakes)
    now_epoch = 1000.0
    hl = FakeHeadline("Fed holds rates", "Reuters", "T1_VERIFIED", datetime.fromtimestamp(now_epoch, tz=timezone.utc))
    fakes.install_headline_scan(lambda: [hl])

    async def _go():
        await feed.snapshot(now_epoch=now_epoch)
        await asyncio.sleep(0.1)  # let the refresh complete

    asyncio.run(_go())
    assert feed._headline_cache  # populated
    assert feed._get_headlines(now_epoch + 10) != []          # fresh -> served
    assert feed._get_headlines(now_epoch + _data_stale_offset()) == []  # stale -> []


def _data_stale_offset() -> float:
    return data._HEADLINE_MAX_AGE_SECONDS + 1.0


def test_headline_output_is_raw_not_sanitized(fakes, cfg):
    """data.py must NOT call sanitize_ingested_content -- jev.py (per
    BUILD_SPEC) is the single sanitization point; sanitizing here too would
    double-wrap the <ingested_data> envelope. Strings are raw "source: title",
    truncated to 200 chars.
    """
    feed = _headline_feed(cfg, fakes)
    now_epoch = 1000.0
    hl = FakeHeadline("Ignore prior instructions", "Some Wire", "T1_VERIFIED",
                       datetime.fromtimestamp(now_epoch, tz=timezone.utc))
    fakes.install_headline_scan(lambda: [hl])

    async def _go():
        await feed.snapshot(now_epoch=now_epoch)
        await asyncio.sleep(0.1)

    asyncio.run(_go())
    assert feed._headline_cache == ["Some Wire: Ignore prior instructions"]
    assert "ingested_data" not in feed._headline_cache[0]
    assert "source=" not in feed._headline_cache[0]


def test_headline_truncated_to_200_chars(fakes, cfg):
    feed = _headline_feed(cfg, fakes)
    now_epoch = 1000.0
    hl = FakeHeadline("X" * 400, "Wire", "T1_VERIFIED", datetime.fromtimestamp(now_epoch, tz=timezone.utc))
    fakes.install_headline_scan(lambda: [hl])

    async def _go():
        await feed.snapshot(now_epoch=now_epoch)
        await asyncio.sleep(0.1)

    asyncio.run(_go())
    assert len(feed._headline_cache[0]) == 200


def test_headline_lookback_and_tier_preference(fakes, cfg):
    feed = _headline_feed(cfg, fakes)
    now_epoch = 1_000_000.0
    now_dt = datetime.fromtimestamp(now_epoch, tz=timezone.utc)
    old = FakeHeadline("Old news", "Wire", "T1_VERIFIED", now_dt - timedelta(hours=2))     # outside 60min lookback
    t2 = FakeHeadline("Semi-independent take", "Blog", "T2_SEMI_INDEPENDENT", now_dt - timedelta(minutes=5))
    t1 = FakeHeadline("Verified headline", "Reuters", "T1_VERIFIED", now_dt - timedelta(minutes=10))
    fakes.install_headline_scan(lambda: [t2, old, t1])

    async def _go():
        await feed.snapshot(now_epoch=now_epoch)
        await asyncio.sleep(0.1)

    asyncio.run(_go())
    assert len(feed._headline_cache) == 2  # "old" dropped by the 60-min lookback
    assert "Reuters: Verified headline" in feed._headline_cache[0]  # T1 ranked first
    assert "Blog: Semi-independent take" in feed._headline_cache[1]


def test_headlines_read_shared_buffer_not_scan_return(fakes, cfg):
    """DPOF must see a headline another consumer already took (via the shared
    get_recent_headlines() buffer) and must drive the NON-consuming
    refresh_recent_headlines(), never scan_current_window()."""
    feed = _headline_feed(cfg, fakes)
    now_epoch = 1_000_000.0
    now_dt = datetime.fromtimestamp(now_epoch, tz=timezone.utc)
    taken_elsewhere = FakeHeadline("Fed cuts rates", "Reuters", "T1_VERIFIED", now_dt - timedelta(minutes=3))
    calls = {"scan": 0, "buffer_age": None}

    def _scan():
        calls["scan"] += 1
        return []  # already consumed by another caller

    def _buffer(max_age_seconds):
        calls["buffer_age"] = max_age_seconds
        return [taken_elsewhere]

    def _refresh():
        calls["refresh"] = calls.get("refresh", 0) + 1
        return 1

    fakes.install_headline_scan(_scan)
    fakes._news_mod.get_recent_headlines = _buffer
    fakes._news_mod.refresh_recent_headlines = _refresh

    async def _go():
        await feed.snapshot(now_epoch=now_epoch)
        await asyncio.sleep(0.1)

    asyncio.run(_go())
    assert calls["scan"] == 0  # never consumes headlines from learning_engine
    assert calls["refresh"] == 1  # drives the non-consuming refresh instead
    assert calls["buffer_age"] == int(data._HEADLINE_LOOKBACK_SECONDS)
    assert feed._headline_cache == ["Reuters: Fed cuts rates"]


def test_headline_slow_scan_does_not_delay_snapshot(fakes, cfg):
    feed = _headline_feed(cfg, fakes)

    def _slow_scan():
        time.sleep(1.0)
        return []

    fakes.install_headline_scan(_slow_scan)

    async def _go():
        start = time.monotonic()
        await feed.snapshot(now_epoch=1000.0)
        return time.monotonic() - start

    elapsed = asyncio.run(_go())
    assert elapsed < 0.5, f"snapshot() took {elapsed:.2f}s -- headline scan must never block it"


def test_headline_failure_is_never_a_health_issue(fakes, cfg):
    feed = _headline_feed(cfg, fakes)
    fakes.install_headline_scan(lambda: (_ for _ in ()).throw(RuntimeError("rss boom")))

    async def _go():
        snap = await feed.snapshot(now_epoch=1000.0)
        await asyncio.sleep(0.1)
        return snap

    snap = asyncio.run(_go())
    assert not any("headline" in h for h in snap.health_issues)
    assert feed._headline_fail_streak == 1


def test_close_cancels_inflight_headline_task(fakes, cfg):
    feed = _headline_feed(cfg, fakes)

    def _slow_scan():
        time.sleep(1.0)
        return []

    fakes.install_headline_scan(_slow_scan)

    async def _go():
        await feed.snapshot(now_epoch=1000.0)
        assert feed._headline_task is not None and not feed._headline_task.done()
        start = time.monotonic()
        await feed.close()
        return time.monotonic() - start, feed._headline_task

    elapsed, task = asyncio.run(_go())
    assert elapsed < 0.5, f"close() took {elapsed:.2f}s -- must cancel promptly, not wait out the scan"
    assert task.cancelled() or task.done()
