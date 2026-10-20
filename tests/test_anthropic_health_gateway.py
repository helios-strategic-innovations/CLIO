"""
OP-09: gateway-aware health classification in agent/integrations/anthropic_health.py.

Real SDK exception instances are raised by a fake client, so the isinstance
mapping in _ping_sync is exercised against the genuine class hierarchy.
No network: the client is a fake object; keys are fake.
"""

import httpx
import anthropic
import pytest

from agent.integrations import anthropic_health as ah

_FAKE_KEY = "sk-" + "a" * 64
_REQ = httpx.Request("POST", "https://gateway.invalid/v1/messages")


def _resp(status: int) -> httpx.Response:
    return httpx.Response(status, request=_REQ)


class _FakeMessages:
    def __init__(self, exc=None):
        self._exc = exc

    def create(self, **kwargs):
        if self._exc is not None:
            raise self._exc
        return object()


class _FakeClient:
    def __init__(self, exc=None):
        self.messages = _FakeMessages(exc)


def _ping_with(monkeypatch, exc=None):
    monkeypatch.setattr(ah, "_get_client", lambda: _FakeClient(exc))
    return ah._ping_sync()


def test_healthy_ping_returns_ok_tuple(monkeypatch):
    ok, latency, reason = _ping_with(monkeypatch, None)
    assert ok is True
    assert isinstance(latency, int) and latency >= 0
    assert reason is None


def test_authentication_error_maps_to_auth_invalid(monkeypatch):
    exc = anthropic.AuthenticationError(
        "Invalid API key",
        response=_resp(401),
        body={"type": "error", "error": {"type": "authentication_error", "message": "Invalid API key"}},
    )
    ok, latency, reason = _ping_with(monkeypatch, exc)
    assert (ok, latency, reason) == (False, None, "auth_invalid")


def test_permission_denied_maps_to_key_forbidden_or_expired(monkeypatch):
    exc = anthropic.PermissionDeniedError(
        "API key expired",
        response=_resp(403),
        body={"type": "error", "error": {"type": "permission_error", "message": "API key expired"}},
    )
    assert _ping_with(monkeypatch, exc) == (False, None, "key_forbidden_or_expired")


def test_rate_limit_insufficient_balance_by_message(monkeypatch):
    exc = anthropic.RateLimitError(
        "Insufficient balance on this key",
        response=_resp(429),
        body={"type": "error", "error": {"type": "rate_limit_error", "message": "Insufficient balance"}},
    )
    assert _ping_with(monkeypatch, exc) == (False, None, "credit_balance_exhausted")


def test_rate_limit_insufficient_balance_by_body_type(monkeypatch):
    # Message carries no "insufficient" word; the gateway error.type does.
    exc = anthropic.RateLimitError(
        "quota reached",
        response=_resp(429),
        body={"type": "error", "error": {"type": "insufficient_balance", "code": "insufficient_quota"}},
    )
    assert _ping_with(monkeypatch, exc) == (False, None, "credit_balance_exhausted")


def test_rate_limit_plain_maps_to_rate_limited(monkeypatch):
    exc = anthropic.RateLimitError(
        "Too many requests",
        response=_resp(429),
        body={"type": "error", "error": {"type": "rate_limit_error", "message": "Too many requests"}},
    )
    assert _ping_with(monkeypatch, exc) == (False, None, "rate_limited")


def test_bad_request_credit_marker_still_exhausted(monkeypatch):
    exc = anthropic.BadRequestError(
        "Your credit balance is too low",
        response=_resp(400),
        body={"type": "error", "error": {"type": "invalid_request_error", "message": "credit balance"}},
    )
    assert _ping_with(monkeypatch, exc) == (False, None, "credit_balance_exhausted")


def test_bad_request_other_keeps_generic_prefix(monkeypatch):
    exc = anthropic.BadRequestError(
        "model not found",
        response=_resp(400),
        body={"type": "error", "error": {"type": "invalid_request_error", "message": "model not found"}},
    )
    ok, _, reason = _ping_with(monkeypatch, exc)
    assert ok is False
    assert reason.startswith("bad_request: ")


def test_internal_server_error_maps_to_provider_unavailable(monkeypatch):
    exc = anthropic.InternalServerError(
        "upstream error",
        response=_resp(503),
        body={"type": "error", "error": {"type": "server_error", "message": "upstream error"}},
    )
    assert _ping_with(monkeypatch, exc) == (False, None, "provider_unavailable")


def test_connection_error_maps_to_provider_unavailable(monkeypatch):
    exc = anthropic.APIConnectionError(request=_REQ)
    assert _ping_with(monkeypatch, exc) == (False, None, "provider_unavailable")


def test_timeout_maps_to_provider_unavailable(monkeypatch):
    exc = anthropic.APITimeoutError(request=_REQ)
    assert _ping_with(monkeypatch, exc) == (False, None, "provider_unavailable")


def test_other_exception_keeps_generic_format_and_no_key(monkeypatch):
    exc = ValueError("boom " + _FAKE_KEY)
    ok, _, reason = _ping_with(monkeypatch, exc)
    assert ok is False
    assert reason.startswith("ValueError: ")
    assert len(reason) <= len("ValueError: ") + 120


def test_reasons_never_contain_key_for_auth_errors(monkeypatch):
    exc = anthropic.AuthenticationError(
        "Invalid API key " + _FAKE_KEY,
        response=_resp(401),
        body={"type": "error", "error": {"type": "authentication_error"}},
    )
    _, _, reason = _ping_with(monkeypatch, exc)
    assert _FAKE_KEY not in reason
    assert reason == "auth_invalid"


def test_transport_400_count_reads_shared_status_histogram(monkeypatch):
    from agent import anthropic_gateway as gw

    stats = gw.GatewayStats()
    stats.record_call("claude-opus-5-5", 10, 400)
    stats.record_call("claude-opus-5-5", 10, 400)
    stats.record_call("claude-opus-5-5", 10, 200)
    monkeypatch.setattr(gw, "_STATS", stats)
    assert ah._transport_400_count() == 2


def test_transport_400_count_is_none_when_stats_unavailable(monkeypatch):
    from agent import anthropic_gateway as gw

    def _boom():
        raise RuntimeError("stats unavailable")

    monkeypatch.setattr(gw, "get_stats", _boom)
    assert ah._transport_400_count() is None


def test_check_result_reports_transport_and_keeps_legacy_field(monkeypatch):
    import asyncio

    monkeypatch.setattr(ah, "_ping_sync", lambda: (True, 5, None))
    monkeypatch.setattr(ah, "_tail_lines_sync", lambda path, n: [])
    monkeypatch.setattr(ah, "_transport_400_count", lambda: 3)
    result = asyncio.run(ah.check_anthropic_health())
    assert result["recent_400_count"] == 0
    assert result["transport_400_count"] == 3
    assert result["healthy"] is True


def test_count_credit_markers_new_markers_case_insensitive():
    lines = [
        "INSUFFICIENT BALANCE on key",
        "error insufficient_balance",
        "Api Key Expired at 2027",
        "credit balance is too low",
        "unrelated line",
    ]
    assert ah._count_credit_markers_sync(lines) == 4


def test_new_markers_present_in_constant():
    for marker in ("insufficient balance", "insufficient_balance", "api key expired"):
        assert marker in ah._CREDIT_MARKERS
