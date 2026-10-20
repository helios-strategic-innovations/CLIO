"""
P2-5: Tests for AthenaBridge malformed-JSONL handling and health surfacing
(agent/integrations/athena_bridge.py).

All fixtures live under pytest's tmp_path — never a real production log/vault path.
"""

import json
from datetime import datetime, timezone

import pytest

from agent.integrations import athena_bridge as ab


@pytest.fixture(autouse=True)
def isolated_bridge_paths(monkeypatch, tmp_path):
    """Point the module's vault + bridge-log paths at tmp_path, and enable the bridge."""
    vault_path = tmp_path / "forecasts.jsonl"
    log_path = tmp_path / "athena_bridge_log.jsonl"
    monkeypatch.setattr(ab, "_FORECASTS_JSONL", str(vault_path))
    monkeypatch.setattr(ab, "_BRIDGE_LOG_PATH", str(log_path))
    monkeypatch.setenv("CLIO_ATHENA_ENABLED", "1")
    monkeypatch.setenv("CLIO_ATHENA_LOG_ONLY", "1")
    return {"vault_path": vault_path, "log_path": log_path}


def _write(path, text):
    with open(path, "w") as f:
        f.write(text)


def _valid_ticker_line(ticker="AAPL", alignment=0.5):
    entry = {
        "event_type": "forecast_ticker",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": {
            "ticker": ticker,
            "point_forecast": [100.0, 101.0, 102.0],
            "current_price": 100.0,
            "implied_vol": 0.2,
            "directional_alignment": alignment,
            "horizon_days": 5,
        },
    }
    return json.dumps(entry) + "\n"


def test_malformed_lines_counted_and_surfaced(isolated_bridge_paths):
    vault_path = isolated_bridge_paths["vault_path"]
    _write(
        vault_path,
        _valid_ticker_line("AAPL")
        + "{not valid json\n"
        + "also not json\n"
        + _valid_ticker_line("MSFT"),
    )

    bridge = ab.AthenaBridge()
    status = bridge.get_status()

    assert status["vault_entries"] == 2
    assert status["malformed_drops"] == 2
    assert status["load_failed"] is False
    assert status["available"] is True
    assert any("malformed" in issue.lower() for issue in status["issues"])


def test_all_malformed_vault_reports_load_failed_not_healthy(isolated_bridge_paths):
    vault_path = isolated_bridge_paths["vault_path"]
    _write(vault_path, "{broken\n" + "still broken\n" + "{\"unterminated\":\n")

    bridge = ab.AthenaBridge()
    status = bridge.get_status()

    assert status["vault_entries"] == 0
    assert status["load_failed"] is True
    assert status["available"] is False
    assert any("load failed" in issue.lower() for issue in status["issues"])
    assert bridge.is_available() is False


def test_clean_valid_vault_is_healthy_and_round_trips(isolated_bridge_paths):
    vault_path = isolated_bridge_paths["vault_path"]
    _write(vault_path, _valid_ticker_line("AAPL", alignment=0.6))

    bridge = ab.AthenaBridge()
    status = bridge.get_status()

    assert status["available"] is True
    assert status["load_failed"] is False
    assert status["malformed_drops"] == 0
    assert status["vault_entries"] == 1
    assert not any("malformed" in i.lower() or "load failed" in i.lower() for i in status["issues"])

    forecast = bridge.forecast_ticker("AAPL")
    assert forecast is not None
    assert forecast["ticker"] == "AAPL"
    assert forecast["directional_label"] == "BULLISH"
    assert forecast["current_price"] == 100.0
