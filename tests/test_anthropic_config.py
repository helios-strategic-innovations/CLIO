"""OP-01: Anthropic routing through the OneProvider gateway (agent/config.py).

All env is controlled with monkeypatch; Config.from_env() is called directly.
Keys below are obviously fake (sk- + 64 chars of a/0).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.config import Config  # noqa: E402

DIRECT_KEY = "sk-" + "a" * 64
GATEWAY_KEY = "sk-" + "0" * 64
GOOD_URL = "https://api.oneprovider.dev"

ENV_KEYS = (
    "ANTHROPIC_API_KEY",
    "ONEPROVIDER_API_KEY",
    "ONEPROVIDER_BASE_URL",
    "CLIO_ANTHROPIC_GATEWAY",
    "DEFAULT_MODEL",
    "FAST_MODEL",
    "JUDGMENT_MODEL",
    "SAMPLING_MODEL",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _from_env(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Config.from_env()


def test_unset_everything_is_direct(monkeypatch):
    cfg = Config.from_env()
    assert cfg.anthropic_gateway is False
    assert cfg.anthropic_base_url == ""
    assert cfg.anthropic_api_key == ""


def test_direct_key_used_when_no_gateway_key(monkeypatch):
    cfg = _from_env(monkeypatch, ANTHROPIC_API_KEY=DIRECT_KEY)
    assert cfg.anthropic_gateway is False
    assert cfg.anthropic_base_url == ""
    assert cfg.anthropic_api_key == DIRECT_KEY


def test_gateway_key_routes_via_oneprovider(monkeypatch):
    cfg = _from_env(
        monkeypatch,
        ANTHROPIC_API_KEY=DIRECT_KEY,
        ONEPROVIDER_API_KEY=GATEWAY_KEY,
    )
    assert cfg.anthropic_gateway is True
    assert cfg.anthropic_base_url == GOOD_URL
    assert cfg.anthropic_api_key == GATEWAY_KEY
    assert cfg.anthropic_api_key != DIRECT_KEY


def test_gateway_key_never_sent_to_anthropic_host(monkeypatch, capsys):
    cfg = _from_env(
        monkeypatch,
        ANTHROPIC_API_KEY=DIRECT_KEY,
        ONEPROVIDER_API_KEY=GATEWAY_KEY,
        ONEPROVIDER_BASE_URL="https://api.anthropic.com",
    )
    assert cfg.anthropic_gateway is True
    assert cfg.anthropic_api_key == ""
    assert cfg.anthropic_base_url == ""
    captured = capsys.readouterr()
    assert "host not in allowlist" in captured.err
    assert GATEWAY_KEY not in captured.err


def test_mode_off_ignores_gateway_key(monkeypatch):
    cfg = _from_env(
        monkeypatch,
        ANTHROPIC_API_KEY=DIRECT_KEY,
        ONEPROVIDER_API_KEY=GATEWAY_KEY,
        CLIO_ANTHROPIC_GATEWAY="off",
    )
    assert cfg.anthropic_gateway is False
    assert cfg.anthropic_base_url == ""
    assert cfg.anthropic_api_key == DIRECT_KEY


def test_mode_on_without_gateway_key_fails_closed(monkeypatch):
    cfg = _from_env(monkeypatch, ANTHROPIC_API_KEY=DIRECT_KEY, CLIO_ANTHROPIC_GATEWAY="on")
    assert cfg.anthropic_gateway is True
    assert cfg.anthropic_api_key == ""
    assert cfg.anthropic_base_url == ""


def test_mode_auto_with_direct_key_only_is_direct(monkeypatch):
    cfg = _from_env(monkeypatch, ANTHROPIC_API_KEY=DIRECT_KEY, CLIO_ANTHROPIC_GATEWAY="auto")
    assert cfg.anthropic_gateway is False
    assert cfg.anthropic_api_key == DIRECT_KEY


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://api.oneprovider.dev",
        "https://evil.example.com",
        "https://api.oneprovider.dev/v1",
        "https://user:pass@api.oneprovider.dev",
        "https://api.oneprovider.dev:8443",
        "https://api.oneprovider.dev?x=1",
        "https://api.oneprovider.dev:",
    ],
)
def test_invalid_gateway_url_sends_no_key(monkeypatch, capsys, bad_url):
    cfg = _from_env(
        monkeypatch,
        ANTHROPIC_API_KEY=DIRECT_KEY,
        ONEPROVIDER_API_KEY=GATEWAY_KEY,
        ONEPROVIDER_BASE_URL=bad_url,
    )
    assert cfg.anthropic_gateway is True
    assert cfg.anthropic_api_key == ""
    assert cfg.anthropic_base_url == ""
    captured = capsys.readouterr()
    assert GATEWAY_KEY not in captured.err
    assert DIRECT_KEY not in captured.err


def test_key_whitespace_and_quotes_cleaned(monkeypatch):
    cfg = _from_env(monkeypatch, ONEPROVIDER_API_KEY=' "' + GATEWAY_KEY + '" ')
    assert cfg.anthropic_gateway is True
    assert cfg.anthropic_api_key == GATEWAY_KEY
    assert cfg.oneprovider_api_key == GATEWAY_KEY


def test_single_quoted_direct_key_cleaned(monkeypatch):
    cfg = _from_env(monkeypatch, ANTHROPIC_API_KEY=" '" + DIRECT_KEY + "' ")
    assert cfg.anthropic_api_key == DIRECT_KEY


def test_bogus_mode_treated_as_auto(monkeypatch, capsys):
    cfg = _from_env(
        monkeypatch,
        ANTHROPIC_API_KEY=DIRECT_KEY,
        ONEPROVIDER_API_KEY=GATEWAY_KEY,
        CLIO_ANTHROPIC_GATEWAY="sometimes",
    )
    assert cfg.anthropic_gateway_mode == "auto"
    assert cfg.anthropic_gateway is True
    assert cfg.anthropic_api_key == GATEWAY_KEY
    captured = capsys.readouterr()
    assert "sometimes" not in captured.err
    assert GATEWAY_KEY not in captured.err


def test_client_kwargs_direct(monkeypatch):
    cfg = _from_env(monkeypatch, ANTHROPIC_API_KEY=DIRECT_KEY)
    kwargs = cfg.anthropic_client_kwargs()
    assert kwargs == {"api_key": DIRECT_KEY, "timeout": 120}


def test_client_kwargs_gateway(monkeypatch):
    monkeypatch.setenv("CLIO_ANTHROPIC_GATEWAY_TRANSPORT", "0")
    cfg = _from_env(monkeypatch, ANTHROPIC_API_KEY=DIRECT_KEY, ONEPROVIDER_API_KEY=GATEWAY_KEY)
    kwargs = cfg.anthropic_client_kwargs(timeout=30)
    assert kwargs == {"api_key": GATEWAY_KEY, "timeout": 30, "base_url": GOOD_URL}


def test_client_kwargs_gateway_transport_shared_client(monkeypatch):
    import anthropic

    monkeypatch.delenv("CLIO_ANTHROPIC_GATEWAY_TRANSPORT", raising=False)
    cfg = _from_env(monkeypatch, ANTHROPIC_API_KEY=DIRECT_KEY, ONEPROVIDER_API_KEY=GATEWAY_KEY)
    first = cfg.anthropic_client_kwargs(timeout=30)
    second = cfg.anthropic_client_kwargs(timeout=30)
    assert isinstance(first["http_client"], anthropic.DefaultHttpxClient)
    assert first["http_client"] is second["http_client"]
    assert first["base_url"] == GOOD_URL


def test_client_kwargs_extra_overrides(monkeypatch):
    cfg = _from_env(monkeypatch, ANTHROPIC_API_KEY=DIRECT_KEY)
    kwargs = cfg.anthropic_client_kwargs(max_retries=0, timeout=5)
    assert kwargs == {"api_key": DIRECT_KEY, "timeout": 5, "max_retries": 0}


def test_dataclass_defaults_match_from_env_defaults(monkeypatch):
    defaults = Config()
    cfg = Config.from_env()
    for name in (
        "oneprovider_api_key",
        "oneprovider_base_url",
        "anthropic_gateway_mode",
        "anthropic_gateway",
        "anthropic_base_url",
        "default_model",
        "fast_model",
        "judgment_model",
        "sampling_model",
    ):
        assert getattr(defaults, name) == getattr(cfg, name), name


def test_no_key_in_stderr_on_gateway_route(monkeypatch, capsys):
    _from_env(monkeypatch, ANTHROPIC_API_KEY=DIRECT_KEY, ONEPROVIDER_API_KEY=GATEWAY_KEY)
    captured = capsys.readouterr()
    assert GATEWAY_KEY not in captured.err
    assert DIRECT_KEY not in captured.err
