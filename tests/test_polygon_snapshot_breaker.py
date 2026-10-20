"""POLY-1 (S211): the Polygon plan is not entitled to real-time stock snapshots (403 NOT_AUTHORIZED).

Once the snapshot endpoint answers 403, stop sending requests for a window; other endpoints
(aggregates, news) must be unaffected. No network: a fake client stands in for httpx.
"""

import asyncio
import time

import httpx
import pytest

from agent.integrations import polygon

SNAP = "/v2/snapshot/locale/us/markets/stocks/tickers/SPY"
PREV = "/v2/aggs/ticker/SPY/prev"


class _FakeClient:
    def __init__(self, status_by_prefix):
        self.calls = []
        self._status = status_by_prefix

    async def get(self, url, params=None):
        self.calls.append(url)
        status = 403 if "/v2/snapshot/" in url else self._status
        request = httpx.Request("GET", url)
        return httpx.Response(status, json={"status": "X"}, request=request)


@pytest.fixture()
def client(monkeypatch):
    fake = _FakeClient(200)

    async def _get_client(timeout=15.0):
        return fake

    monkeypatch.setattr(polygon, "BASE_URL", "https://api.polygon.test")
    monkeypatch.setattr(polygon, "API_KEY", "FAKE_POLYGON_KEY")
    monkeypatch.setattr(polygon, "_get_client", _get_client)
    monkeypatch.setattr(polygon, "_snapshot_denied_until", 0.0)
    return fake


def _run(coro):
    return asyncio.run(coro)


def test_403_opens_breaker_and_later_calls_skip_http(client):
    with pytest.raises(httpx.HTTPStatusError):
        _run(polygon._get(SNAP))
    assert len(client.calls) == 1
    for _ in range(3):
        with pytest.raises(polygon.PolygonNotEntitled):
            _run(polygon._get(SNAP))
    assert len(client.calls) == 1  # no further HTTP while the breaker is open


def test_breaker_message_carries_no_url_or_key(client):
    with pytest.raises(httpx.HTTPStatusError):
        _run(polygon._get(SNAP))
    with pytest.raises(polygon.PolygonNotEntitled) as exc:
        _run(polygon._get(SNAP))
    assert "FAKE_POLYGON_KEY" not in str(exc.value)
    assert "apiKey" not in str(exc.value)


def test_other_endpoints_unaffected_while_open(client):
    with pytest.raises(httpx.HTTPStatusError):
        _run(polygon._get(SNAP))
    assert _run(polygon._get(PREV)) == {"status": "X"}
    assert any("/aggs/" in c for c in client.calls)


def test_breaker_expires_and_probes_again(client, monkeypatch):
    with pytest.raises(httpx.HTTPStatusError):
        _run(polygon._get(SNAP))
    monkeypatch.setattr(polygon, "_snapshot_denied_until", time.monotonic() - 1.0)  # window elapsed
    with pytest.raises(httpx.HTTPStatusError):  # probe reaches the (still 403) endpoint
        _run(polygon._get(SNAP))
    assert len(client.calls) == 2


def test_breaker_window_env_is_floored_and_tolerates_garbage(monkeypatch):
    monkeypatch.setenv("CLIO_POLYGON_SNAPSHOT_BREAKER_S", "5")
    assert polygon._snapshot_breaker_seconds() == 60.0
    monkeypatch.setenv("CLIO_POLYGON_SNAPSHOT_BREAKER_S", "not-a-number")
    assert polygon._snapshot_breaker_seconds() == 21600.0
    monkeypatch.delenv("CLIO_POLYGON_SNAPSHOT_BREAKER_S", raising=False)
    assert polygon._snapshot_breaker_seconds() == 21600.0
