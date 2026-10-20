import asyncio
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from starlette.requests import Request

from mission_control.backend import auth
from mission_control.backend.routes.analytics import _compute_metrics
from mission_control.backend.routes.options import _option_snapshot_fields


def _request(method: str, path: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": method,
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [],
            "scheme": "https",
            "server": ("test", 443),
            "client": ("127.0.0.1", 1),
        }
    )


def test_external_api_key_is_method_scoped_read_only(monkeypatch):
    monkeypatch.setattr(auth.mc_config, "external_api_key", "review-only-key")
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="review-only-key")

    allowed = asyncio.run(auth.require_auth(_request("GET", "/api/files/read"), credentials))
    assert allowed == {"sub": "external_api", "scope": "readonly"}

    for method, path in (
        ("POST", "/api/files/write"),
        ("POST", "/api/files/edit"),
        ("POST", "/api/hydrinia/control"),
        ("GET", "/api/portfolio-admin"),
    ):
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(auth.require_auth(_request(method, path), credentials))
        assert exc_info.value.status_code == 403


def test_missing_option_snapshot_is_null_and_stale():
    fields = _option_snapshot_fields(None)
    assert fields["bid"] is None
    assert fields["ask"] is None
    assert fields["delta"] is None
    assert fields["implied_volatility"] is None
    assert fields["stale"] is True
    assert fields["reason_code"] == "NO_SNAPSHOT"


def test_partial_option_snapshot_preserves_real_values_without_fake_greeks():
    timestamp = datetime(2026, 7, 9, 20, 0, tzinfo=timezone.utc)
    snapshot = SimpleNamespace(
        latest_quote=SimpleNamespace(bid_price=4.0, ask_price=4.4, timestamp=timestamp),
        latest_trade=SimpleNamespace(price=4.2, size=8, timestamp=timestamp),
        greeks=None,
    )
    fields = _option_snapshot_fields(snapshot)
    assert fields["mid"] == 4.2
    assert fields["spread"] == 0.4
    assert fields["delta"] is None
    assert fields["stale"] is True
    assert fields["reason_code"] == "MISSING_GREEKS"


def test_short_analytics_history_is_unavailable_not_neutral():
    metrics = _compute_metrics([0.0], [500_000.0], ["2026-07-09"])
    assert metrics["sharpe"] is None
    assert metrics["total_return_pct"] is None
    assert metrics["start_equity"] == 500_000.0
    assert metrics["stale"] is True
    assert metrics["reason_code"] == "INSUFFICIENT_HISTORY"


def test_supplementary_uw_bridge_keeps_persistent_clio_guard():
    source = (
        Path(__file__).parents[1] / "agent" / "integrations" / "uw_signal_bridge.py"
    ).read_text(encoding="utf-8")
    start = source.index("    async def _clio_uw_get(")
    end = source.index("    async def fetch_fundamentals", start)
    block = source[start:end]
    assert "_uw_get(endpoint, params=params)" in block
    assert "skip_moltbot_budget=True" not in block
