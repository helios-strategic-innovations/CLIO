"""POLY-1 / SUB-1 (S211): market snapshots come from Alpaca Elite, not the cancelled Polygon plan.

No network: a fake Alpaca data client stands in for the SDK client. The adapter must return the
same dict shape as polygon.get_snapshot_raw so market_events / learning_engine change one import.
"""

import asyncio
import math
from types import SimpleNamespace

import pytest

from agent.integrations import alpaca as alpaca_mod
from agent.integrations import market_snapshot as ms
from agent.integrations import polygon as polygon_mod


def _snap(last=500.5, day_close=501.0, prev_close=495.0, high=502.0, low=498.0, volume=1_000_000):
    return SimpleNamespace(
        latest_trade=SimpleNamespace(price=last) if last is not None else None,
        daily_bar=SimpleNamespace(close=day_close, high=high, low=low, volume=volume)
        if day_close is not None
        else None,
        previous_daily_bar=SimpleNamespace(close=prev_close) if prev_close is not None else None,
    )


class _FakeDataClient:
    def __init__(self, snaps):
        self.snaps = snaps
        self.requests = []

    def get_stock_snapshot(self, request):
        self.requests.append(request)
        wanted = request.symbol_or_symbols
        wanted = [wanted] if isinstance(wanted, str) else list(wanted)
        return {s: self.snaps[s] for s in wanted if s in self.snaps}


@pytest.fixture()
def data_client(monkeypatch):
    monkeypatch.delenv("CLIO_MARKET_SNAPSHOT_SOURCE", raising=False)
    fake = _FakeDataClient({"SPY": _snap()})
    monkeypatch.setattr(alpaca_mod, "_get_data_client", lambda: fake)
    return fake


def _run(coro):
    return asyncio.run(coro)


def test_maps_alpaca_snapshot_to_polygon_shape(data_client):
    snap = _run(ms.get_snapshot_raw("spy"))  # case-insensitive
    assert set(snap) == {
        "last_price", "day_close", "prev_close", "change", "change_pct",
        "day_high", "day_low", "volume",
    }
    assert snap["last_price"] == 500.5
    assert snap["day_close"] == 501.0
    assert snap["prev_close"] == 495.0
    assert snap["change"] == pytest.approx(6.0)
    assert snap["change_pct"] == pytest.approx(6.0 / 495.0 * 100)
    assert (snap["day_high"], snap["day_low"], snap["volume"]) == (502.0, 498.0, 1_000_000)
    assert data_client.requests[0].symbol_or_symbols == "SPY"


def test_missing_daily_bar_falls_back_to_latest_trade(data_client):
    data_client.snaps["SPY"] = _snap(last=500.5, day_close=None, prev_close=495.0)
    snap = _run(ms.get_snapshot_raw("SPY"))
    assert snap["day_close"] == 500.5
    assert snap["day_high"] == 0.0 and snap["volume"] == 0.0


def test_missing_previous_close_gives_zero_change_not_a_crash(data_client):
    data_client.snaps["SPY"] = _snap(prev_close=None)
    snap = _run(ms.get_snapshot_raw("SPY"))
    assert snap["prev_close"] == 0.0 and snap["change"] == 0 and snap["change_pct"] == 0


def test_nan_price_is_unknown_not_propagated(data_client):
    data_client.snaps["SPY"] = _snap(last=float("nan"), day_close=501.0)
    snap = _run(ms.get_snapshot_raw("SPY"))
    assert snap["last_price"] == 501.0 and math.isfinite(snap["last_price"])


def test_unknown_symbol_raises_lookup_error(data_client):
    with pytest.raises(LookupError):
        _run(ms.get_snapshot_raw("ZZZZ"))


def test_index_symbols_use_the_yahoo_path_and_never_alpaca(data_client, monkeypatch):
    async def fake_yahoo(symbol):
        return {"last_price": 16.07, "via": "yahoo", "symbol": symbol}

    monkeypatch.setattr(polygon_mod, "_yahoo_index_snapshot", fake_yahoo)
    snap = _run(ms.get_snapshot_raw("vix"))
    assert snap["via"] == "yahoo" and snap["symbol"] == "VIX"
    assert data_client.requests == []


def test_rollback_flag_routes_to_polygon(data_client, monkeypatch):
    async def fake_polygon(symbol):
        return {"via": "polygon", "symbol": symbol}

    monkeypatch.setattr(polygon_mod, "get_snapshot_raw", fake_polygon)
    monkeypatch.setenv("CLIO_MARKET_SNAPSHOT_SOURCE", "polygon")
    assert _run(ms.get_snapshot_raw("SPY"))["via"] == "polygon"
    assert data_client.requests == []


def test_unknown_source_value_defaults_to_alpaca(data_client, monkeypatch):
    monkeypatch.setenv("CLIO_MARKET_SNAPSHOT_SOURCE", "bogus")
    assert _run(ms.get_snapshot_raw("SPY"))["last_price"] == 500.5


def test_blank_symbol_returns_empty(data_client):
    assert _run(ms.get_snapshot_raw("  ")) == {}
    assert data_client.requests == []


def test_get_price_change_shape_and_none_on_empty(data_client, monkeypatch):
    out = _run(ms.get_price_change("SPY"))
    assert out == {"price": 500.5, "change_pct": pytest.approx(6.0 / 495.0 * 100)}

    async def empty(symbol):
        return {}

    monkeypatch.setattr(ms, "get_snapshot_raw", empty)
    assert _run(ms.get_price_change("SPY")) is None


def test_market_events_fetch_snapshot_uses_the_adapter(monkeypatch):
    from agent.integrations import market_events

    async def fake(symbol):
        return {"last_price": 1.0, "symbol": symbol}

    monkeypatch.setattr(ms, "get_snapshot_raw", fake)
    assert _run(market_events._fetch_snapshot("SPY")) == {"last_price": 1.0, "symbol": "SPY"}


def test_learning_snapshot_price_uses_adapter_and_never_polygon(monkeypatch):
    from agent.integrations import learning_engine

    async def polygon_must_not_be_used(*args, **kwargs):
        raise AssertionError("learning snapshot must not touch Polygon")

    monkeypatch.setattr(polygon_mod, "_get", polygon_must_not_be_used)

    async def fake(symbol):
        return {"price": 10.0, "change_pct": 1.5}

    monkeypatch.setattr(ms, "get_price_change", fake)
    assert _run(learning_engine._fetch_snapshot_price("SPY")) == {"price": 10.0, "change_pct": 1.5}

    async def boom(symbol):
        raise RuntimeError("provider down")

    monkeypatch.setattr(ms, "get_price_change", boom)
    assert _run(learning_engine._fetch_snapshot_price("SPY")) is None
