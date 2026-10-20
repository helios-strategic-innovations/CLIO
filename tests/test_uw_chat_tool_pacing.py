"""S211 (ADR-005): the chat-tool UW client (agent/integrations/unusual_whales.py) has its own httpx client and
does not go through uw_flow_intel._uw_request, so it must take a slot from the shared pacer itself -- otherwise
an LLM tool call can land inside a DPOF burst on the same key. No network; budgets are redirected by conftest."""

import asyncio

import pytest

from agent.integrations import unusual_whales as uwt
from agent.integrations import uw_pacer


class _Resp:
    status_code = 200

    def json(self):
        return {"data": []}

    def raise_for_status(self):
        return None


class _Client:
    def __init__(self):
        self.calls = 0

    async def get(self, *_a, **_k):
        self.calls += 1
        return _Resp()


@pytest.fixture()
def harness(monkeypatch):
    client = _Client()
    sleeps = []

    async def get_client():
        return client

    async def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr(uwt, "API_KEY", "test-key")
    monkeypatch.setattr(uwt, "BASE_URL", "https://uw.invalid")
    monkeypatch.setattr(uwt, "_get_client", get_client)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    uw_pacer.reset_uw_pacer_for_tests()
    return client, sleeps


def test_chat_tool_requests_are_paced_like_everything_else(harness):
    client, sleeps = harness

    async def three():
        return [await uwt._get("/api/market/market-tide") for _ in range(3)]

    assert asyncio.run(three()) == [{"data": []}] * 3
    assert client.calls == 3
    assert sleeps == [pytest.approx(1 / 3, abs=0.02), pytest.approx(2 / 3, abs=0.02)]  # 1st free, then spaced


def test_chat_tool_shares_the_singleton_with_the_flow_client(harness):
    """A slot taken by uw_flow_intel's path delays the chat tool (one bucket, not two)."""
    _, sleeps = harness
    pacer = uw_pacer.get_uw_pacer()
    pacer.reserve()  # e.g. a DPOF request just took the free slot
    asyncio.run(uwt._get("/api/market/market-tide"))
    assert sleeps and sleeps[0] == pytest.approx(1 / 3, abs=0.02)


def test_pacer_disabled_flag_leaves_the_chat_tool_unpaced(harness, monkeypatch):
    _, sleeps = harness
    monkeypatch.setenv("CLIO_UW_PACER_ENABLED", "0")
    uw_pacer.reset_uw_pacer_for_tests()

    async def burst():
        for _ in range(4):
            await uwt._get("/api/market/market-tide")

    asyncio.run(burst())
    assert sleeps == []
