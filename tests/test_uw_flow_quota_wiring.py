import asyncio

from agent.integrations import uw_flow_intel as uw
from agent.integrations.uw_quota_client import QuotaDecision


class FakeResponse:
    def __init__(self, status_code, data=None, headers=None):
        self.status_code = status_code
        self._data = data or {}
        self.headers = headers or {
            "x-uw-daily-req-count": "36490",
            "x-uw-token-req-limit": "160000",
            "x-uw-req-per-minute-remaining": "999999",
        }

    def json(self):
        return self._data


class FakeAsyncClient:
    def __init__(self, responses):
        self.responses = list(responses)

    async def get(self, *_args, **_kwargs):
        return self.responses.pop(0)


class FakeQuota:
    def __init__(self, allowed=True):
        self.allowed = allowed
        self.reservations = []
        self.observations = []
        self.releases = []

    def reserve(self, **kwargs):
        self.reservations.append(kwargs)
        return QuotaDecision(
            allowed=self.allowed,
            request_id=f"00000000-0000-0000-0000-{len(self.reservations):012d}",
            reason_code="P0_RESERVED" if self.allowed else "ACCOUNT_LIMIT",
            source="supabase",
            remaining=123000,
        )

    def observe_response(self, decision, **kwargs):
        self.observations.append((decision, kwargs))
        return kwargs

    def release(self, decision, reason):
        self.releases.append((decision, reason))
        return True


def test_clio_uw_call_reserves_and_observes(monkeypatch):
    quota = FakeQuota()
    client = FakeAsyncClient([FakeResponse(200, {"data": [1]})])
    monkeypatch.setattr(uw, "_get_api_key", lambda: "test-key")
    monkeypatch.setattr(uw, "_get_quota_coordinator", lambda: quota)

    async def get_client():
        return client

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(uw, "_get_client", get_client)
    monkeypatch.setattr(uw.asyncio, "sleep", no_sleep)
    result = asyncio.run(uw._uw_get("/market/market-tide", skip_moltbot_budget=True))

    assert result == {"data": [1]}
    assert quota.reservations[0]["priority"] == 0
    assert quota.observations[0][1]["endpoint"] == "/market/market-tide"
    assert quota.observations[0][1]["status_code"] == 200


def test_clio_uw_denial_avoids_provider_call(monkeypatch):
    quota = FakeQuota(allowed=False)
    client = FakeAsyncClient([FakeResponse(200, {"should": "not-run"})])
    monkeypatch.setattr(uw, "_get_api_key", lambda: "test-key")
    monkeypatch.setattr(uw, "_get_quota_coordinator", lambda: quota)

    async def get_client():
        return client

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(uw, "_get_client", get_client)
    monkeypatch.setattr(uw.asyncio, "sleep", no_sleep)
    result = asyncio.run(uw._uw_get("/congress/recent-trades", skip_moltbot_budget=True))

    assert result == {}
    assert len(client.responses) == 1
    assert quota.reservations[0]["priority"] == 2


def test_clio_uw_429_reserves_and_observes_retry(monkeypatch):
    quota = FakeQuota()
    client = FakeAsyncClient(
        [
            FakeResponse(429, headers={
                "x-uw-daily-req-count": "36500",
                "x-uw-token-req-limit": "160000",
                "x-uw-req-per-minute-reset": "5000",
                "Retry-After": "5",
            }),
            FakeResponse(200, {"data": ["recovered"]}, headers={
                "x-uw-daily-req-count": "36501",
                "x-uw-token-req-limit": "160000",
            }),
        ]
    )
    monkeypatch.setattr(uw, "_get_api_key", lambda: "test-key")
    monkeypatch.setattr(uw, "_get_quota_coordinator", lambda: quota)

    async def get_client():
        return client

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(uw, "_get_client", get_client)
    monkeypatch.setattr(uw.asyncio, "sleep", no_sleep)
    result = asyncio.run(uw._uw_get("/option-trades/flow-alerts", skip_moltbot_budget=True))

    assert result == {"data": ["recovered"]}
    assert len(quota.reservations) == 2
    assert [item[1]["status_code"] for item in quota.observations] == [429, 200]


def test_coordinator_initialization_failure_keeps_local_guard_path(monkeypatch):
    client = FakeAsyncClient([FakeResponse(200, {"data": ["available"]})])
    monkeypatch.setattr(uw, "_get_api_key", lambda: "test-key")
    monkeypatch.setattr(uw, "_get_quota_coordinator", lambda: (_ for _ in ()).throw(RuntimeError("down")))

    async def get_client():
        return client

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(uw, "_get_client", get_client)
    monkeypatch.setattr(uw.asyncio, "sleep", no_sleep)
    result = asyncio.run(uw._uw_get("/market/market-tide", skip_moltbot_budget=True))
    assert result == {"data": ["available"]}
