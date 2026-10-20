"""
Tests for the CLIO-native scheduled report rewrite (builder B2c):
  agent/integrations/premarket_scan.py, hourly_debrief.py,
  postmarket_debrief.py, news_update.py.

Hydrinia (/home/agent/tradingbot) is permanently off. These reports must
describe CLIO's own DPOF book (agent/dpof/) — never Hydrinia's frozen
position_log.json / daily_metrics.json. All Hydrinia bridge writes are gated
on agent.hydrinia_bridge.bridge_enabled() (default OFF -> silent no-op).

No network: Alpaca is stubbed via monkeypatched _get_trading_client/
_retry_alpaca_call. All DPOF file I/O goes through a tmp_path
CLIO_DPOF_DATA_DIR — never the real data/ tree. Market regime (when used)
goes through a tmp_path CLIO_MARKET_REGIME_FILE.
"""

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from agent import hydrinia_bridge
from agent.dpof.models import PositionRecord
from agent.integrations import (
    hourly_debrief,
    news_update,
    postmarket_debrief,
    premarket_scan,
)

_ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    """Isolate every test from the real environment (mirrors tests/dpof/conftest.py). Live-state
    paths are redirected, never deleted: unset means the live default (data/dpof, and
    CLIO_MARKET_REGIME_FILE stays on conftest's per-test temp path)."""
    for key in list(os.environ):
        if key.startswith("CLIO_DPOF_") or key.startswith("CLIO_HYDRINIA_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CLIO_DPOF_DATA_DIR", str(tmp_path / "dpof_default"))
    yield


@pytest.fixture
def dpof_dir(tmp_path, monkeypatch):
    d = tmp_path / "dpof"
    d.mkdir()
    monkeypatch.setenv("CLIO_DPOF_DATA_DIR", str(d))
    return d


def _write_positions(ddir, positions: list[PositionRecord]) -> None:
    payload = {"version": 1, "positions": [p.to_dict() for p in positions]}
    with open(os.path.join(str(ddir), "positions.json"), "w") as f:
        json.dump(payload, f, default=str)


def _day_start_et(days_offset: int = 0) -> datetime:
    now = datetime.now(_ET)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return start + timedelta(days=days_offset)


def _mk_position(**kwargs) -> PositionRecord:
    defaults = dict(
        id="pos1",
        underlying="SPY",
        direction="bullish",
        structure="single",
        legs=[{"symbol": "SPY260925C00580000", "side": "buy"}],
        status="open",
        qty_original=2,
        qty_open=2,
        entry_price=5.25,
        mode="paper",
    )
    defaults.update(kwargs)
    return PositionRecord(**defaults)


class _FakeAlpacaPos:
    def __init__(self, symbol, side="long", avg_entry_price=5.25, unrealized_pl=10.0,
                 current_price=6.0, qty="2", change_today=0.01, market_value=1200.0):
        self.symbol = symbol
        self.side = side
        self.avg_entry_price = avg_entry_price
        self.unrealized_pl = unrealized_pl
        self.current_price = current_price
        self.qty = qty
        self.change_today = change_today
        self.market_value = market_value


class _FakeAccount:
    def __init__(self, equity=105000.0, last_equity=104000.0, cash=50000.0):
        self.equity = equity
        self.last_equity = last_equity
        self.cash = cash


async def _fake_retry(fn, *args, **kwargs):
    return fn(*args, **kwargs)


def _patch_alpaca_positions(monkeypatch, positions):
    """Patch agent.integrations.alpaca so client.get_all_positions() returns `positions`."""
    import agent.integrations.alpaca as alpaca_mod

    class _FakeClient:
        def get_all_positions(self):
            return positions

        def get_account(self):
            return _FakeAccount()

    monkeypatch.setattr(alpaca_mod, "_get_trading_client", lambda: _FakeClient())
    monkeypatch.setattr(alpaca_mod, "_retry_alpaca_call", _fake_retry)


def _patch_alpaca_failure(monkeypatch):
    import agent.integrations.alpaca as alpaca_mod

    def _boom():
        raise RuntimeError("alpaca unreachable")

    monkeypatch.setattr(alpaca_mod, "_get_trading_client", _boom)


# ---------------------------------------------------------------------------
# Hydrinia bridge gating — every module's _write_to_bridge must be a silent
# no-op when the bridge is off (default), and only write when explicitly
# re-enabled.
# ---------------------------------------------------------------------------

class TestBridgeGating:
    def test_bridge_disabled_by_default(self):
        assert hydrinia_bridge.bridge_enabled() is False

    def test_bridge_enabled_via_env(self, monkeypatch):
        monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
        assert hydrinia_bridge.bridge_enabled() is True
        monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "false")
        assert hydrinia_bridge.bridge_enabled() is False

    @pytest.mark.parametrize("mod,fn,args", [
        (premarket_scan, "_write_to_bridge", ("report text", [])),
        (hourly_debrief, "_write_to_bridge", ("report text", 0, 0)),
        (postmarket_debrief, "_write_to_bridge", ("report text", 0, 0, 0)),
        (news_update, "_write_to_bridge", ("report text", 0)),
    ])
    def test_write_to_bridge_noop_when_disabled(self, mod, fn, args, tmp_path, monkeypatch):
        reports_dir = tmp_path / "reports"
        monkeypatch.setattr(mod, "REPORTS_DIR", str(reports_dir))
        getattr(mod, fn)(*args)
        assert not reports_dir.exists()

    @pytest.mark.parametrize("mod,fn,args", [
        (premarket_scan, "_write_to_bridge", ("report text", [])),
        (hourly_debrief, "_write_to_bridge", ("report text", 0, 0)),
        (postmarket_debrief, "_write_to_bridge", ("report text", 0, 0, 0)),
        (news_update, "_write_to_bridge", ("report text", 0)),
    ])
    def test_write_to_bridge_writes_when_enabled(self, mod, fn, args, tmp_path, monkeypatch):
        monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
        reports_dir = tmp_path / "reports"
        monkeypatch.setattr(mod, "REPORTS_DIR", str(reports_dir))
        getattr(mod, fn)(*args)
        assert reports_dir.exists()
        written = list(reports_dir.glob("*.json"))
        assert len(written) == 1
        payload = json.loads(written[0].read_text())
        assert payload["content"] == "report text"


# ---------------------------------------------------------------------------
# premarket_scan.py
# ---------------------------------------------------------------------------

class TestPremarketScan:
    def test_open_positions_alpaca_success_enriched_from_dpof(self, dpof_dir, monkeypatch):
        occ = "SPY260925C00580000"
        _write_positions(dpof_dir, [_mk_position(
            id="dpof1", underlying="SPY", direction="bullish", structure="single",
            legs=[{"symbol": occ, "side": "buy"}], status="open",
            entry_price=5.0,
        )])
        _patch_alpaca_positions(monkeypatch, [_FakeAlpacaPos(occ, unrealized_pl=42.0, current_price=6.5)])

        positions = asyncio.run(premarket_scan._get_open_positions())

        assert len(positions) == 1
        pos = positions[0]
        assert pos["symbol"] == "SPY"  # underlying, not OCC
        assert pos["contract"] == "SPY 2026-09-25 580C"
        assert pos["direction"] == "bullish"
        assert pos["entry_price"] == 5.0  # DPOF entry, not Alpaca avg_entry_price
        assert pos["unrealized_pnl"] == 42.0
        assert "note" not in pos

    def test_open_positions_alpaca_failure_falls_back_to_dpof_ledger(self, dpof_dir, monkeypatch):
        _write_positions(dpof_dir, [_mk_position(id="dpof2", underlying="QQQ", direction="bearish")])
        _patch_alpaca_failure(monkeypatch)

        positions = asyncio.run(premarket_scan._get_open_positions())

        assert len(positions) == 1
        assert positions[0]["symbol"] == "QQQ"
        assert positions[0]["note"] == "(DPOF ledger; live prices unavailable)"

    def test_open_positions_flat_and_alpaca_ok_returns_empty(self, dpof_dir, monkeypatch):
        _patch_alpaca_positions(monkeypatch, [])
        positions = asyncio.run(premarket_scan._get_open_positions())
        assert positions == []

    def test_daily_metrics_clio_native(self, dpof_dir, monkeypatch):
        yesterday_close = _mk_position(
            id="closed_y", underlying="SPY", direction="bullish", status="closed",
            entry_time=(_day_start_et(-1) + timedelta(hours=10)).timestamp(),
            closed_time=(_day_start_et(-1) + timedelta(hours=15)).timestamp(),
            realized_pnl=125.50, exit_reason="tp1",
        )
        # Position closed *today* must NOT count toward "yesterday" metrics.
        today_close = _mk_position(
            id="closed_t", underlying="QQQ", direction="bearish", status="closed",
            entry_time=(_day_start_et(0) + timedelta(hours=10)).timestamp(),
            closed_time=(_day_start_et(0) + timedelta(hours=11)).timestamp(),
            realized_pnl=-30.0, exit_reason="stop",
        )
        _write_positions(dpof_dir, [yesterday_close, today_close])
        _patch_alpaca_positions(monkeypatch, [])

        regime_file = dpof_dir.parent / "regime.json"
        regime_file.write_text(json.dumps({
            "regime": "BULL", "confidence": 1.0, "vix": 14.0,
            "spy_close": 500.0, "spy_sma20": 495.0, "spy_sma50": 490.0,
            "as_of": datetime.now(timezone.utc).isoformat(), "source": "clio_native",
        }))
        monkeypatch.setenv("CLIO_MARKET_REGIME_FILE", str(regime_file))

        metrics = asyncio.run(premarket_scan._get_daily_metrics())

        assert metrics["equity"] == 105000.0
        assert metrics["daily_pnl"] == pytest.approx(1000.0)
        assert metrics["dpof_closed_yesterday_count"] == 1
        assert metrics["dpof_closed_yesterday_wins"] == 1
        assert metrics["dpof_realized_yesterday"] == pytest.approx(125.50)
        assert metrics["market_regime"] == "BULL"

    def test_format_positions_table_flat(self):
        assert "flat" in premarket_scan._format_positions_table([]).lower()

    def test_format_metrics_no_data(self):
        assert premarket_scan._format_metrics({}) == "No daily metrics available."


# ---------------------------------------------------------------------------
# hourly_debrief.py
# ---------------------------------------------------------------------------

class TestHourlyDebrief:
    def test_load_position_context_keyed_by_occ_symbol(self, dpof_dir):
        occ = "QQQ260925P00480000"
        _write_positions(dpof_dir, [_mk_position(
            id="h1", underlying="QQQ", direction="bearish", structure="single",
            legs=[{"symbol": occ, "side": "buy"}], status="open",
        )])

        ctx = hourly_debrief._load_position_context()

        assert occ in ctx
        assert ctx[occ]["direction"] == "bearish"
        assert ctx[occ]["underlying"] == "QQQ"
        assert ctx[occ]["dpof_id"] == "h1"

    def test_load_position_context_empty_when_no_dpof_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLIO_DPOF_DATA_DIR", str(tmp_path / "missing"))
        assert hourly_debrief._load_position_context() == {}

    def test_format_trade_data_uses_contracts_not_shares(self, dpof_dir):
        occ = "SPY260925C00580000"
        _write_positions(dpof_dir, [_mk_position(
            id="h2", underlying="SPY", direction="bullish", structure="single",
            legs=[{"symbol": occ, "side": "buy"}], status="open",
        )])
        ctx = hourly_debrief._load_position_context()

        orders = [{"symbol": occ, "side": "buy", "qty": 2.0, "avg_price": 5.0, "filled_at": "x"}]
        positions = [{
            "symbol": occ, "qty": 2.0, "side": "long", "unrealized_pl": 15.0,
            "current_price": 6.0, "avg_entry_price": 5.0, "change_today": 0.02, "market_value": 1200.0,
        }]
        text = hourly_debrief._format_trade_data(orders, positions, {"equity": 100.0, "pnl_today": 5.0}, ctx)

        assert "contracts" in text
        assert "shares" not in text
        assert "CLIO's DPOF book" in text
        assert "SPY 2026-09-25 580C" in text  # readable contract label, not raw OCC


# ---------------------------------------------------------------------------
# postmarket_debrief.py
# ---------------------------------------------------------------------------

class TestPostmarketDebrief:
    def test_get_todays_activity_buckets_by_et_day(self, dpof_dir):
        opened_today = _mk_position(
            id="open_today", underlying="SPY", direction="bullish", status="open",
            entry_time=(_day_start_et(0) + timedelta(hours=10)).timestamp(),
        )
        closed_today = _mk_position(
            id="closed_today", underlying="QQQ", direction="bearish", status="closed",
            entry_time=(_day_start_et(0) + timedelta(hours=9, minutes=45)).timestamp(),
            closed_time=(_day_start_et(0) + timedelta(hours=14)).timestamp(),
            realized_pnl=88.0, exit_reason="eod_flatten",
        )
        closed_yesterday = _mk_position(
            id="closed_yday", underlying="SPY", direction="bearish", status="closed",
            entry_time=(_day_start_et(-1) + timedelta(hours=10)).timestamp(),
            closed_time=(_day_start_et(-1) + timedelta(hours=11)).timestamp(),
            realized_pnl=-12.0,
        )
        _write_positions(dpof_dir, [opened_today, closed_today, closed_yesterday])

        opens, closes, currently_open = postmarket_debrief._get_todays_activity()

        # opened_today() means "entry_time is today", independent of current status —
        # a same-day round trip (closed_today) legitimately appears in both buckets.
        assert {p.id for p in opens} == {"open_today", "closed_today"}
        assert {p.id for p in closes} == {"closed_today"}
        assert {p.id for p in currently_open} == {"open_today"}

    def test_get_todays_activity_degrades_gracefully_without_dpof(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLIO_DPOF_DATA_DIR", str(tmp_path / "nope"))
        opens, closes, currently_open = postmarket_debrief._get_todays_activity()
        assert opens == [] and closes == [] and currently_open == []

    def test_format_open_positions_flat(self):
        text = postmarket_debrief._format_open_positions([], {})
        assert "flat" in text.lower()
        assert "15:50" in text

    def test_format_open_positions_warns_when_not_flat_and_overlays_alpaca(self):
        occ = "SPY260925C00580000"
        pos = _mk_position(
            id="stuck1", underlying="SPY", direction="bullish", status="open",
            legs=[{"symbol": occ, "side": "buy"}],
        )
        text = postmarket_debrief._format_open_positions([pos], {occ: {"unrealized_pl": 17.5}})
        assert "WARNING" in text
        assert "15:50" in text
        assert "$+17.50" in text

    def test_format_todays_trades_none_when_empty(self):
        text = postmarket_debrief._format_todays_trades([], [])
        assert "TRADES OPENED TODAY (0)" in text
        assert "TRADES CLOSED TODAY (0)" in text
        assert text.count("None") == 2

    def test_format_todays_trades_shows_realized_and_duration(self):
        closed = _mk_position(
            id="c1", underlying="SPY", direction="bullish", status="closed",
            entry_time=_day_start_et(0).timestamp() + 9 * 3600,
            closed_time=_day_start_et(0).timestamp() + 11 * 3600,
            realized_pnl=55.0, exit_reason="tp1",
        )
        text = postmarket_debrief._format_todays_trades([], [closed])
        assert "realized $+55.00" in text
        assert "tp1" in text
        assert "held 2.0h" in text

    def test_daily_metrics_clio_native(self, dpof_dir, monkeypatch):
        closed_today = _mk_position(
            id="ct1", underlying="SPY", direction="bullish", status="closed",
            entry_time=_day_start_et(0).timestamp() + 9 * 3600,
            closed_time=_day_start_et(0).timestamp() + 10 * 3600,
            realized_pnl=40.0,
        )
        _write_positions(dpof_dir, [closed_today])
        _patch_alpaca_positions(monkeypatch, [])

        metrics = asyncio.run(postmarket_debrief._get_daily_metrics())

        assert metrics["equity"] == 105000.0
        assert metrics["daily_pnl"] == pytest.approx(1000.0)
        assert metrics["dpof_realized_today"] == pytest.approx(40.0)
        assert metrics["dpof_closed_today_count"] == 1
        # No CLIO_MARKET_REGIME_FILE configured -> key simply absent, never fabricated.
        assert "market_regime" not in metrics

    def test_format_metrics_omits_regime_line_when_absent(self):
        text = postmarket_debrief._format_metrics({"equity": 100.0, "daily_pnl": 5.0})
        assert "Market Regime" not in text


# ---------------------------------------------------------------------------
# news_update.py
# ---------------------------------------------------------------------------

class TestNewsUpdate:
    def test_open_positions_flat_returns_default_universe_watched(self, dpof_dir):
        positions = news_update._get_open_positions()
        symbols = {p["symbol"]: p["held"] for p in positions}
        assert symbols == {"SPY": False, "QQQ": False}

    def test_open_positions_held_deduped_against_universe(self, dpof_dir):
        _write_positions(dpof_dir, [_mk_position(id="n1", underlying="SPY", direction="bullish", status="open")])

        positions = news_update._get_open_positions()

        symbols = {p["symbol"]: p["held"] for p in positions}
        assert symbols["SPY"] is True
        assert symbols["QQQ"] is False
        assert list(symbols).count("SPY") == 1  # no duplicate entry for a held universe ticker

    def test_open_positions_respects_configured_universe(self, dpof_dir, monkeypatch):
        monkeypatch.setenv("CLIO_DPOF_UNDERLYINGS", "IWM")
        positions = news_update._get_open_positions()
        assert [p["symbol"] for p in positions] == ["IWM"]

    def test_open_positions_degrades_gracefully_without_dpof(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLIO_DPOF_DATA_DIR", str(tmp_path / "nope"))
        positions = news_update._get_open_positions()
        assert {p["symbol"] for p in positions} == {"SPY", "QQQ"}
        assert all(p["held"] is False for p in positions)
