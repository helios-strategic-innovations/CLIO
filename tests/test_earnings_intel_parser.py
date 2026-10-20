"""S215 / R-16: the earnings calendar search asks for 10 results, not 5.

A fake search_web records its kwargs and returns ten snippets, with a bellwether named
only in the 8th result, so the test fails if the parser stops at the first five. The
calendar load/save, budget tracker and search are all faked; nothing reaches the network
or the live earnings_calendar.json.
"""

import asyncio

import pytest

from agent.integrations import earnings_intel
from agent.integrations import web_search


class _Budget:
    async def check_budget(self, name):
        return True, ""

    async def record_usage(self, name):
        return None


@pytest.fixture
def harness(monkeypatch):
    calls = []
    saved = []
    bellwether = next(iter(earnings_intel.EARNINGS_SUPPLY_CHAIN))
    results = [
        {"title": f"Market roundup {i}", "snippet": "no bellwether here", "url": f"https://x/{i}"}
        for i in range(1, 11)
    ]
    results[7] = {
        "title": f"{bellwether} reports next week",
        "snippet": f"{bellwether} earnings on October 15",
        "url": "https://x/8",
    }

    async def fake_search(query, **kwargs):
        calls.append((query, kwargs))
        return results[: kwargs.get("max_results", 5)]

    monkeypatch.setattr(web_search, "search_web", fake_search)
    monkeypatch.setattr(earnings_intel, "budget_tracker", _Budget())
    monkeypatch.setattr(earnings_intel, "_load_calendar", lambda: {})
    monkeypatch.setattr(earnings_intel, "_save_calendar", lambda data: saved.append(data))
    return {"calls": calls, "saved": saved, "bellwether": bellwether}


def test_search_requests_ten_results(harness):
    asyncio.run(earnings_intel._refresh_earnings_calendar())
    assert len(harness["calls"]) == 1
    _query, kwargs = harness["calls"][0]
    assert kwargs.get("max_results") == 10


def test_bellwether_in_eighth_result_is_parsed(harness):
    entries = asyncio.run(earnings_intel._refresh_earnings_calendar())
    tickers = [e["ticker"] for e in entries]
    assert harness["bellwether"] in tickers
    assert harness["saved"] and harness["saved"][0]["refreshed_at"]
