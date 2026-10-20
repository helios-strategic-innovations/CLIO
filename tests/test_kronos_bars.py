"""agent.kronos.bars -- RTH filtering, early-close/holiday handling,
unclosed-bar exclusion, amount computation, and the AlpacaBarsClient
adapter against a fake data client (no network)."""

import asyncio
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from agent.kronos.bars import (
    BAR_MINUTES,
    AlpacaBarsClient,
    compute_amount,
    filter_rth_bars,
    is_trading_day,
    session_close_et,
    session_open_et,
)
from agent.kronos.config import KronosConfig

_ET = ZoneInfo("America/New_York")

# A normal Tuesday with a full 09:30-16:00 session.
_NORMAL_DAY = date(2026, 9, 22)
# Day-after-Thanksgiving 2026 -- early close at 13:00 (agent.integrations.market_events.MARKET_EARLY_CLOSE).
_EARLY_CLOSE_DAY = date(2026, 11, 27)
# Christmas Day 2026 -- full NYSE holiday.
_HOLIDAY = date(2026, 12, 25)
# A Saturday.
_WEEKEND = date(2026, 9, 26)

_HOLIDAYS = frozenset({_HOLIDAY})
_EARLY_CLOSE = frozenset({_EARLY_CLOSE_DAY})


def _et(d: date, hour: int, minute: int = 0) -> datetime:
    return datetime(d.year, d.month, d.day, hour, minute, tzinfo=_ET)


def _epoch(dt: datetime) -> float:
    return dt.timestamp()


def _bar(t_open_et: datetime, o=100.0, h=101.0, l=99.0, c=100.5, v=1000.0, vw=None) -> dict:
    return {"t": _epoch(t_open_et), "o": o, "h": h, "l": l, "c": c, "v": v, "vw": vw}


# ── compute_amount ───────────────────────────────────────────────────────

def test_compute_amount_uses_vwap_when_present():
    assert compute_amount(1000.0, 50.25, 50.0) == 1000.0 * 50.25


def test_compute_amount_falls_back_to_close_when_vwap_absent():
    assert compute_amount(1000.0, None, 50.0) == 1000.0 * 50.0


def test_compute_amount_zero_volume_is_zero_not_none():
    assert compute_amount(None, None, 50.0) == 0.0


# ── session_open_et / session_close_et / is_trading_day ─────────────────

def test_session_times_normal_day():
    assert session_open_et(_NORMAL_DAY, _HOLIDAYS) == _et(_NORMAL_DAY, 9, 30)
    assert session_close_et(_NORMAL_DAY, _HOLIDAYS, _EARLY_CLOSE) == _et(_NORMAL_DAY, 16, 0)


def test_session_times_early_close_day():
    assert session_close_et(_EARLY_CLOSE_DAY, _HOLIDAYS, _EARLY_CLOSE) == _et(_EARLY_CLOSE_DAY, 13, 0)
    assert session_open_et(_EARLY_CLOSE_DAY, _HOLIDAYS) == _et(_EARLY_CLOSE_DAY, 9, 30)


def test_session_times_holiday_is_none():
    assert session_open_et(_HOLIDAY, _HOLIDAYS) is None
    assert session_close_et(_HOLIDAY, _HOLIDAYS, _EARLY_CLOSE) is None


def test_session_times_weekend_is_none():
    assert session_open_et(_WEEKEND, _HOLIDAYS) is None
    assert is_trading_day(_WEEKEND, _HOLIDAYS) is False


def test_is_trading_day_normal():
    assert is_trading_day(_NORMAL_DAY, _HOLIDAYS) is True


# ── filter_rth_bars ───────────────────────────────────────────────────────

def test_filter_keeps_rth_bars_only():
    raw = [
        _bar(_et(_NORMAL_DAY, 9, 0)),    # pre-market -- excluded
        _bar(_et(_NORMAL_DAY, 9, 30)),   # first RTH bar -- included
        _bar(_et(_NORMAL_DAY, 15, 55)),  # last RTH bar (normal day) -- included
        _bar(_et(_NORMAL_DAY, 16, 0)),   # after-hours (bar OPENS at close) -- excluded
        _bar(_et(_NORMAL_DAY, 18, 0)),   # after-hours -- excluded
    ]
    now_epoch = _epoch(_et(_NORMAL_DAY, 23, 59))
    out = filter_rth_bars(raw, bar_minutes=BAR_MINUTES, now_epoch=now_epoch, holidays=_HOLIDAYS, early_close=_EARLY_CLOSE)
    kept_opens = [b.t_open for b in out]
    assert kept_opens == [_epoch(_et(_NORMAL_DAY, 9, 30)), _epoch(_et(_NORMAL_DAY, 15, 55))]


def test_filter_excludes_weekend_and_holiday_bars():
    raw = [_bar(_et(_WEEKEND, 10, 0)), _bar(_et(_HOLIDAY, 10, 0))]
    now_epoch = _epoch(_et(date(2026, 12, 26), 12, 0))
    out = filter_rth_bars(raw, bar_minutes=BAR_MINUTES, now_epoch=now_epoch, holidays=_HOLIDAYS, early_close=_EARLY_CLOSE)
    assert out == []


def test_filter_early_close_day_excludes_bars_after_1300():
    raw = [
        _bar(_et(_EARLY_CLOSE_DAY, 12, 55)),  # last valid open (bar closes exactly at 13:00) -- included
        _bar(_et(_EARLY_CLOSE_DAY, 13, 0)),   # opens exactly at the early close -- excluded
        _bar(_et(_EARLY_CLOSE_DAY, 13, 30)),  # excluded
    ]
    now_epoch = _epoch(_et(_EARLY_CLOSE_DAY, 20, 0))
    out = filter_rth_bars(raw, bar_minutes=BAR_MINUTES, now_epoch=now_epoch, holidays=_HOLIDAYS, early_close=_EARLY_CLOSE)
    assert len(out) == 1
    assert out[0].t_open == _epoch(_et(_EARLY_CLOSE_DAY, 12, 55))


def test_filter_excludes_still_forming_bar():
    """A bar opening at 15:55 closes at 16:00 -- if 'now' is only 15:57,
    that bar hasn't closed yet and must be excluded."""
    raw = [_bar(_et(_NORMAL_DAY, 15, 50)), _bar(_et(_NORMAL_DAY, 15, 55))]
    now_epoch = _epoch(_et(_NORMAL_DAY, 15, 57))  # bar at 15:55 closes at 16:00 -- still forming
    out = filter_rth_bars(raw, bar_minutes=BAR_MINUTES, now_epoch=now_epoch, holidays=_HOLIDAYS, early_close=_EARLY_CLOSE)
    assert len(out) == 1
    assert out[0].t_open == _epoch(_et(_NORMAL_DAY, 15, 50))


def test_filter_includes_bar_exactly_at_close_boundary():
    """A bar opening at 15:55 closes at exactly 16:00 -- 'now' == 16:00 means it just closed."""
    raw = [_bar(_et(_NORMAL_DAY, 15, 55))]
    now_epoch = _epoch(_et(_NORMAL_DAY, 16, 0))
    out = filter_rth_bars(raw, bar_minutes=BAR_MINUTES, now_epoch=now_epoch, holidays=_HOLIDAYS, early_close=_EARLY_CLOSE)
    assert len(out) == 1


def test_filter_computes_amount_and_defaults_missing_fields():
    raw = [_bar(_et(_NORMAL_DAY, 10, 0), c=200.0, v=500.0, vw=201.0)]
    now_epoch = _epoch(_et(_NORMAL_DAY, 23, 0))
    out = filter_rth_bars(raw, bar_minutes=BAR_MINUTES, now_epoch=now_epoch, holidays=_HOLIDAYS, early_close=_EARLY_CLOSE)
    assert out[0].amount == pytest.approx(500.0 * 201.0)


def test_filter_drops_malformed_bars_without_crashing():
    raw = [{"t": None, "o": 1, "h": 2, "l": 0, "c": 1}, {"t": _epoch(_et(_NORMAL_DAY, 10, 0)), "o": None, "h": 2, "l": 0, "c": 1}]
    now_epoch = _epoch(_et(_NORMAL_DAY, 23, 0))
    out = filter_rth_bars(raw, bar_minutes=BAR_MINUTES, now_epoch=now_epoch, holidays=_HOLIDAYS, early_close=_EARLY_CLOSE)
    assert out == []


def test_filter_dedups_by_t_open_keeping_last():
    t = _et(_NORMAL_DAY, 10, 0)
    raw = [_bar(t, c=100.0), _bar(t, c=999.0)]
    now_epoch = _epoch(_et(_NORMAL_DAY, 23, 0))
    out = filter_rth_bars(raw, bar_minutes=BAR_MINUTES, now_epoch=now_epoch, holidays=_HOLIDAYS, early_close=_EARLY_CLOSE)
    assert len(out) == 1
    assert out[0].c == 999.0


def test_filter_sorts_ascending_even_if_input_is_not():
    raw = [_bar(_et(_NORMAL_DAY, 10, 5)), _bar(_et(_NORMAL_DAY, 9, 30))]
    now_epoch = _epoch(_et(_NORMAL_DAY, 23, 0))
    out = filter_rth_bars(raw, bar_minutes=BAR_MINUTES, now_epoch=now_epoch, holidays=_HOLIDAYS, early_close=_EARLY_CLOSE)
    assert out[0].t_open < out[1].t_open


# ── AlpacaBarsClient against a fake data client ──────────────────────────

class _FakeAlpacaBar:
    def __init__(self, ts, o, h, l, c, v, vw=None):
        self.timestamp = ts
        self.open, self.high, self.low, self.close, self.volume, self.vwap = o, h, l, c, v, vw


class _FakeStockDataClient:
    def __init__(self, bars_by_symbol):
        self._bars_by_symbol = bars_by_symbol
        self.calls = []

    def get_stock_bars(self, req):
        self.calls.append(req)
        return {sym: bars for sym, bars in self._bars_by_symbol.items()}


def test_alpaca_bars_client_filters_and_caps_to_lookback():
    now = _et(_NORMAL_DAY, 12, 0).astimezone(timezone.utc)
    bars = [
        _FakeAlpacaBar(_et(_NORMAL_DAY, 9, 30).astimezone(timezone.utc), 1, 2, 0.5, 1.5, 100, 1.4),
        _FakeAlpacaBar(_et(_NORMAL_DAY, 9, 35).astimezone(timezone.utc), 1.5, 2.5, 1, 2, 200, 1.9),
        _FakeAlpacaBar(_et(_NORMAL_DAY, 11, 55).astimezone(timezone.utc), 2, 3, 1.5, 2.5, 300, None),  # still-forming at "now"=12:00? closes 12:00, included
        _FakeAlpacaBar(_et(_NORMAL_DAY, 11, 55).astimezone(timezone.utc) + timedelta(minutes=10), 5, 6, 4, 5, 999, None),
    ]
    fake_client = _FakeStockDataClient({"SPY": bars})
    cfg = KronosConfig(alpaca_key_id="k", alpaca_secret_key="s", stock_feed="iex")
    client = AlpacaBarsClient(cfg, client=fake_client)

    result = asyncio.run(client.get_bars("SPY", lookback_bars=2, now=now))
    assert result is not None
    assert len(result) == 2  # capped to lookback_bars
    assert result[-1].c == 2.5  # newest of the RTH-eligible, closed bars kept


def test_alpaca_bars_client_downgrades_feed_on_403():
    class _Rejecting:
        def __init__(self):
            self.calls = 0

        def get_stock_bars(self, req):
            self.calls += 1
            if self.calls == 1:
                err = Exception("forbidden")
                err.status_code = 403
                raise err
            return {"SPY": [_FakeAlpacaBar(_et(_NORMAL_DAY, 9, 30).astimezone(timezone.utc), 1, 1, 1, 1, 10, None)]}

    fake_client = _Rejecting()
    cfg = KronosConfig(alpaca_key_id="k", alpaca_secret_key="s", stock_feed="sip")
    client = AlpacaBarsClient(cfg, client=fake_client)
    now = _et(_NORMAL_DAY, 12, 0).astimezone(timezone.utc)

    result = asyncio.run(client.get_bars("SPY", lookback_bars=10, now=now))
    assert result is not None and len(result) == 1
    assert client.feed_status()["downgraded"] is True
    assert client.feed_status()["effective"] == "iex"


def test_alpaca_bars_client_returns_none_on_persistent_failure():
    class _AlwaysFails:
        def get_stock_bars(self, req):
            raise RuntimeError("boom")

    cfg = KronosConfig(alpaca_key_id="k", alpaca_secret_key="s")
    client = AlpacaBarsClient(cfg, client=_AlwaysFails())
    now = _et(_NORMAL_DAY, 12, 0).astimezone(timezone.utc)
    result = asyncio.run(client.get_bars("SPY", lookback_bars=10, now=now))
    assert result is None
