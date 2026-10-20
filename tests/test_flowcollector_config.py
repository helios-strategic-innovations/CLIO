"""Tests for agent/flowcollector/config.py. Hermetic: env-only, no I/O."""

import os

import pytest

from agent.flowcollector.config import FlowCollectorConfig, load_config


_ENV_PREFIX_KEYS = [k for k in os.environ if k.startswith("CLIO_FLOWCOLLECTOR_")]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("CLIO_FLOWCOLLECTOR_") or key in (
            "ALPACA_PAPER_KEY_ID", "ALPACA_PAPER_SECRET_KEY", "ALPACA_PAPER",
            "CLIO_SELFFLOW_FILE", "CLIO_SELFFLOW_COMPARE_FILE", "TRADINGBOT_DIR",
        ):
            monkeypatch.delenv(key, raising=False)
    yield


def test_load_config_defaults_match_dataclass_defaults():
    """CLAUDE.md coding standard: dataclass defaults MUST match from_env()
    defaults when no env vars are set."""
    default = FlowCollectorConfig()
    loaded = load_config()
    assert loaded.underlyings == default.underlyings
    assert loaded.strike_band_pct == default.strike_band_pct
    assert loaded.max_dte == default.max_dte
    assert loaded.options_feed == default.options_feed
    assert loaded.stock_feed == default.stock_feed
    assert loaded.alpaca_paper == default.alpaca_paper
    assert loaded.symbol_cap_per_connection == default.symbol_cap_per_connection
    assert loaded.state_file == default.state_file
    assert loaded.compare_file == default.compare_file


def test_load_config_reads_alpaca_paper_keys(monkeypatch):
    monkeypatch.setenv("ALPACA_PAPER_KEY_ID", "AK123")
    monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY", "SECRET456")
    cfg = load_config()
    assert cfg.alpaca_key_id == "AK123"
    assert cfg.alpaca_secret_key == "SECRET456"


def test_load_config_env_overrides():
    import os as _os

    _os.environ["CLIO_FLOWCOLLECTOR_STRIKE_BAND_PCT"] = "0.08"
    _os.environ["CLIO_FLOWCOLLECTOR_MAX_DTE"] = "1"
    _os.environ["CLIO_FLOWCOLLECTOR_UNDERLYINGS"] = "spy, qqq, iwm"
    try:
        cfg = load_config()
        assert cfg.strike_band_pct == 0.08
        assert cfg.max_dte == 1
        assert cfg.underlyings == ("SPY", "QQQ", "IWM")
    finally:
        for k in ("CLIO_FLOWCOLLECTOR_STRIKE_BAND_PCT", "CLIO_FLOWCOLLECTOR_MAX_DTE", "CLIO_FLOWCOLLECTOR_UNDERLYINGS"):
            _os.environ.pop(k, None)


def test_load_config_invalid_numeric_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("CLIO_FLOWCOLLECTOR_MAX_DTE", "not-a-number")
    cfg = load_config()
    assert cfg.max_dte == FlowCollectorConfig().max_dte


def test_selfflow_env_vars_honored():
    import os as _os

    _os.environ["CLIO_SELFFLOW_FILE"] = "/tmp/x/selfflow_state.json"
    _os.environ["CLIO_SELFFLOW_COMPARE_FILE"] = "/tmp/x/selfflow_compare.jsonl"
    try:
        cfg = load_config()
        assert cfg.state_file == "/tmp/x/selfflow_state.json"
        assert cfg.compare_file == "/tmp/x/selfflow_compare.jsonl"
    finally:
        _os.environ.pop("CLIO_SELFFLOW_FILE", None)
        _os.environ.pop("CLIO_SELFFLOW_COMPARE_FILE", None)


def test_validate_flags_bad_config():
    bad = FlowCollectorConfig(underlyings=(), strike_band_pct=2.0, max_dte=-1, symbol_cap_per_connection=0, options_feed="bogus", stock_feed="bogus")
    issues = bad.validate()
    assert len(issues) == 6


def test_validate_default_config_has_no_issues():
    assert FlowCollectorConfig().validate() == []


def test_validate_rejects_delayed_sip_for_stock_feed():
    """Item 1 CRITICAL: alpaca-py's live StockDataStream only accepts
    IEX/SIP (raises ValueError for anything else, confirmed via
    DataFeed.DELAYED_SIP) -- "delayed_sip" is a REST-only concept and must
    never validate as a legal stock_feed."""
    cfg = FlowCollectorConfig(stock_feed="delayed_sip")
    issues = cfg.validate()
    assert any("stock_feed" in i for i in issues)


def test_validate_accepts_sip_and_iex_for_stock_feed():
    assert FlowCollectorConfig(stock_feed="sip").validate() == []
    assert FlowCollectorConfig(stock_feed="iex").validate() == []


def test_universe_empty_retry_seconds_default_matches_load_config():
    default = FlowCollectorConfig()
    assert default.universe_empty_retry_seconds > 0
    assert load_config().universe_empty_retry_seconds == default.universe_empty_retry_seconds


def test_load_config_env_override_universe_empty_retry_seconds(monkeypatch):
    monkeypatch.setenv("CLIO_FLOWCOLLECTOR_UNIVERSE_EMPTY_RETRY_SECONDS", "30")
    assert load_config().universe_empty_retry_seconds == 30.0


def test_validate_rejects_non_positive_universe_empty_retry_seconds():
    assert FlowCollectorConfig(universe_empty_retry_seconds=0.0).validate() != []
    assert FlowCollectorConfig(universe_empty_retry_seconds=-1.0).validate() != []


def test_window_config_fields_were_removed_as_dead():
    """Item 10 (LOW): window_short/medium/long_seconds were never wired into
    Aggregator (which uses its own module-level WINDOW_*_SECONDS constants)
    -- removed rather than fixed, per the orchestrator's stated preference."""
    for dead_field in ("window_short_seconds", "window_medium_seconds", "window_long_seconds"):
        assert not hasattr(FlowCollectorConfig(), dead_field)
