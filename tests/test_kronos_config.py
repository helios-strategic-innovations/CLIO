"""agent.kronos.config -- parsing, clamping, and dataclass/from_env parity
(CLAUDE.md coding standard: "Dataclass defaults MUST match from_env()
defaults")."""

from dataclasses import fields

import pytest

from agent.kronos.config import KronosConfig, config_summary, load_config

_ALL_KRONOS_ENV = (
    "CLIO_KRONOS_SYMBOLS", "CLIO_KRONOS_SAMPLE_PATHS", "CLIO_KRONOS_LOOKBACK_BARS",
    "CLIO_KRONOS_TORCH_THREADS", "CLIO_KRONOS_RUN_TIMEOUT", "CLIO_KRONOS_TEMPERATURE",
    "CLIO_KRONOS_TOP_P", "CLIO_KRONOS_MODEL", "CLIO_KRONOS_MODEL_REVISION",
    "CLIO_KRONOS_TOKENIZER", "CLIO_KRONOS_TOKENIZER_REVISION", "CLIO_KRONOS_VENDOR_COMMIT",
    "CLIO_KRONOS_HF_LOCAL_ONLY", "ALPACA_PAPER_KEY_ID", "ALPACA_PAPER_SECRET_KEY",
    "CLIO_KRONOS_STOCK_FEED", "ALPACA_STOCK_FEED", "CLIO_KRONOS_HISTORY_FILE",
    "CLIO_KRONOS_HISTORY_MAX_LINES",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in _ALL_KRONOS_ENV:
        monkeypatch.delenv(key, raising=False)
    yield


def test_defaults_match_from_env_with_no_env_set():
    """The dataclass's own defaults, and what from_env() falls back to with
    a totally clean environment, must be identical."""
    assert load_config() == KronosConfig()


def test_specific_defaults():
    d = KronosConfig()
    assert d.symbols == ("SPY", "QQQ")
    assert d.sample_paths == 16
    assert d.lookback_bars == 400
    assert d.torch_threads == 6
    assert d.run_timeout_seconds == 240.0
    assert d.model_id == "NeoQuasar/Kronos-small"
    assert d.model_revision == "901c26c1332695a2a8f243eb2f37243a37bea320"
    assert d.tokenizer_id == "NeoQuasar/Kronos-Tokenizer-base"
    assert d.tokenizer_revision == "0e0117387f39004a9016484a186a908917e22426"
    assert d.hf_local_files_only is True
    assert d.stock_feed == "iex"


def test_sample_paths_clamped_2_to_64(monkeypatch):
    monkeypatch.setenv("CLIO_KRONOS_SAMPLE_PATHS", "1000")
    assert load_config().sample_paths == 64
    monkeypatch.setenv("CLIO_KRONOS_SAMPLE_PATHS", "0")
    assert load_config().sample_paths == 2
    monkeypatch.setenv("CLIO_KRONOS_SAMPLE_PATHS", "20")
    assert load_config().sample_paths == 20


def test_lookback_bars_clamped_64_to_512(monkeypatch):
    monkeypatch.setenv("CLIO_KRONOS_LOOKBACK_BARS", "5000")
    assert load_config().lookback_bars == 512
    monkeypatch.setenv("CLIO_KRONOS_LOOKBACK_BARS", "1")
    assert load_config().lookback_bars == 64


def test_torch_threads_clamped_1_to_8(monkeypatch):
    monkeypatch.setenv("CLIO_KRONOS_TORCH_THREADS", "0")
    assert load_config().torch_threads == 1
    monkeypatch.setenv("CLIO_KRONOS_TORCH_THREADS", "99")
    assert load_config().torch_threads == 8


def test_symbols_env_override_uppercases_and_splits(monkeypatch):
    monkeypatch.setenv("CLIO_KRONOS_SYMBOLS", "spy, qqq, iwm")
    assert load_config().symbols == ("SPY", "QQQ", "IWM")


def test_stock_feed_prefers_kronos_specific_over_shared(monkeypatch):
    monkeypatch.setenv("ALPACA_STOCK_FEED", "sip")
    assert load_config().stock_feed == "sip"
    monkeypatch.setenv("CLIO_KRONOS_STOCK_FEED", "iex")
    assert load_config().stock_feed == "iex"


def test_invalid_int_falls_back_to_default(monkeypatch, caplog):
    monkeypatch.setenv("CLIO_KRONOS_SAMPLE_PATHS", "not-a-number")
    assert load_config().sample_paths == KronosConfig().sample_paths


def test_history_file_default_and_override(monkeypatch):
    d = KronosConfig()
    assert d.history_file.endswith("data/intelligence/kronos_history.jsonl")
    monkeypatch.setenv("CLIO_KRONOS_HISTORY_FILE", "/tmp/somewhere/kh.jsonl")
    assert load_config().history_file == "/tmp/somewhere/kh.jsonl"


def test_validate_flags_bad_values():
    bad = KronosConfig(sample_paths=1, lookback_bars=10, torch_threads=0, top_p=2.0, stock_feed="sip9")
    issues = bad.validate()
    assert issues, "expected validation issues"
    assert any("sample_paths" in i for i in issues)
    assert any("lookback_bars" in i for i in issues)
    assert any("torch_threads" in i for i in issues)
    assert any("top_p" in i for i in issues)
    assert any("stock_feed" in i for i in issues)


def test_validate_passes_defaults():
    assert KronosConfig().validate() == []


def test_config_summary_redacts_keys():
    cfg = KronosConfig(alpaca_key_id="AKIA_SECRET", alpaca_secret_key="shh")
    summary = config_summary(cfg)
    assert summary["alpaca_key_id"] == "set"
    assert summary["alpaca_secret_key"] == "set"
    assert "AKIA_SECRET" not in str(summary)
    assert "shh" not in str(summary)


def test_config_summary_covers_every_field():
    cfg = KronosConfig()
    summary = config_summary(cfg)
    for f in fields(cfg):
        assert f.name in summary
