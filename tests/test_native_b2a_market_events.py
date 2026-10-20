"""
B2a: market_events.py repointed off Hydrinia's position_log.json.

CLIO-UW-4 Trigger 3 (unusual-ticker overlap with open positions) must build
its held-ticker set from CLIO's own DPOF book (agent/dpof/readmodel.py), not
Hydrinia's frozen/stale state/position_log.json. Also covers the 2027 NYSE
holiday/early-close additions.
"""

import asyncio
import json
import os
from datetime import date

import pytest

from agent.integrations import market_events as me


class FakeUWBridge:
    def __init__(self, unusual_tickers):
        self._unusual = unusual_tickers

    def is_fresh(self):
        return True

    def get_vol_regime(self):
        return {"regime": "CALM", "event_risk_flag": False}

    def get_unusual_tickers(self):
        return self._unusual

    def get_market_context(self):
        return {}

    def get_earnings(self):
        return {"upcoming": []}

    def get_predictions(self):
        return {"unusual_finance": []}

    def get_sector_enhanced(self):
        return {"etf_tide": {}}

    def get_regime_enhanced(self):
        return {"correlation_changes": {}}


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setenv("CLIO_DPOF_DATA_DIR", str(tmp_path / "dpof"))
    monkeypatch.setattr(me, "check_rss_breaking_news", lambda: None)

    async def _fake_fetch_snapshot(symbol):
        if symbol == "SPY":
            return {"change_pct": 0.1}
        return {"last_price": 15.0, "change": 0.2}

    monkeypatch.setattr(me, "_fetch_snapshot", _fake_fetch_snapshot)

    import agent.integrations.learning_engine as le

    monkeypatch.setattr(le, "_load_learning_state", lambda: {"breaking_news": {}})
    monkeypatch.setattr(le, "_save_learning_state", lambda state: None)


def _write_dpof_positions(data_dir, underlyings):
    os.makedirs(data_dir, exist_ok=True)
    positions = [
        {
            "id": f"p_{u}",
            "underlying": u,
            "direction": "bullish",
            "structure": "single",
            "legs": [],
            "status": "open",
            "qty_original": 1,
            "qty_open": 1,
        }
        for u in underlyings
    ]
    with open(os.path.join(data_dir, "positions.json"), "w") as f:
        json.dump({"positions": positions}, f)


class FakeContext:
    bot = None


def test_held_positions_come_from_dpof_not_hydrinia(monkeypatch, tmp_path, caplog):
    import logging

    monkeypatch.setattr(me, "_is_market_hours", lambda: True)
    _write_dpof_positions(str(tmp_path / "dpof"), ["SPY"])

    import agent.integrations.uw_signal_bridge as uwb

    monkeypatch.setattr(uwb, "get_uw_bridge", lambda: FakeUWBridge(unusual_tickers=["SPY"]))

    caplog.set_level(logging.WARNING, logger="openclaw.market_events")
    asyncio.run(me.check_market_events(FakeContext()))

    assert any("UNUSUAL_OVERLAP" in r.message and "SPY" in r.message for r in caplog.records)


def test_no_open_dpof_positions_means_no_overlap(monkeypatch, tmp_path, caplog):
    import logging

    monkeypatch.setattr(me, "_is_market_hours", lambda: True)
    _write_dpof_positions(str(tmp_path / "dpof"), [])  # flat book

    import agent.integrations.uw_signal_bridge as uwb

    monkeypatch.setattr(uwb, "get_uw_bridge", lambda: FakeUWBridge(unusual_tickers=["SPY"]))

    caplog.set_level(logging.WARNING, logger="openclaw.market_events")
    asyncio.run(me.check_market_events(FakeContext()))

    assert not any("UNUSUAL_OVERLAP" in r.message for r in caplog.records)


def test_held_set_source_is_dpof_readmodel():
    """The held-ticker set is built from agent.dpof.readmodel, and the old
    fcntl-locked Hydrinia position_log.json read (open()/flock() calls) is gone."""
    import inspect

    src = inspect.getsource(me.check_market_events)
    assert "readmodel.open_underlyings" in src
    assert "fcntl.flock" not in src
    assert 'open(_pos_log' not in src


# ---------------------------------------------------------------------------
# 2027 NYSE holiday / early-close calendar
# ---------------------------------------------------------------------------

_EXPECTED_2027_HOLIDAYS = {
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26),
    date(2027, 5, 31), date(2027, 6, 18), date(2027, 7, 5), date(2027, 9, 6),
    date(2027, 11, 25), date(2027, 12, 24),
}


def test_2027_holidays_present():
    assert _EXPECTED_2027_HOLIDAYS <= me.MARKET_HOLIDAYS


def test_2027_early_close_day_after_thanksgiving():
    assert date(2027, 11, 26) in me.MARKET_EARLY_CLOSE


def test_2026_dates_unchanged():
    assert date(2026, 12, 25) in me.MARKET_HOLIDAYS
    assert date(2026, 11, 27) in me.MARKET_EARLY_CLOSE


def test_is_market_hours_respects_2027_holiday(monkeypatch):
    from datetime import datetime

    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2027, 1, 18, 10, 0, tzinfo=tz)  # MLK Day 2027, would-be market hours

    monkeypatch.setattr(me, "datetime", _FixedDatetime)
    assert me._is_market_hours() is False
