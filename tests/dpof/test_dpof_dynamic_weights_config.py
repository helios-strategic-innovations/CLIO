"""Dynamic-weights config knobs: defaults, env normalisation, clamps, dynamic_weights_mode."""
import pytest

from agent.dpof.config import DPOFConfig, dynamic_weights_mode, load_dpof_config

ENV_NAMES = (
    "CLIO_DPOF_DYNAMIC_WEIGHTS",
    "CLIO_DPOF_DW_RING",
    "CLIO_DPOF_DW_DEAD_EPS",
    "CLIO_DPOF_DW_QUALITY_APPLY",
)


def _clear(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_defaults_on_dataclass_and_loader_agree(monkeypatch):
    """Dataclass defaults and load_dpof_config() with the four env vars unset are identical."""
    _clear(monkeypatch)
    d = DPOFConfig()
    cfg = load_dpof_config()
    expected = ("shadow", 40, 0.005, False)
    assert (d.dynamic_weights, d.dw_ring_size, d.dw_dead_eps, d.dw_quality_apply) == expected
    assert (cfg.dynamic_weights, cfg.dw_ring_size, cfg.dw_dead_eps, cfg.dw_quality_apply) == expected


@pytest.mark.parametrize("raw,expected", [
    (" LIVE ", "live"),
    ("Shadow", "shadow"),
    ("off", "off"),
    ("banana", "off"),
    ("", "shadow"),
    ("   ", "shadow"),
    ("1", "off"),
])
def test_mode_env_normalisation(monkeypatch, raw, expected):
    """Mode is stripped and lowercased; an unknown value is off; a blank value is treated as unset (the shadow default)."""
    _clear(monkeypatch)
    monkeypatch.setenv("CLIO_DPOF_DYNAMIC_WEIGHTS", raw)
    assert load_dpof_config().dynamic_weights == expected


@pytest.mark.parametrize("raw,expected", [("5", 10), ("999", 200), ("abc", 40), ("75", 75)])
def test_ring_size_clamped_and_defaulted(monkeypatch, raw, expected):
    """Ring size clamps to [10, 200]; non-integer falls back to 40."""
    _clear(monkeypatch)
    monkeypatch.setenv("CLIO_DPOF_DW_RING", raw)
    assert load_dpof_config().dw_ring_size == expected


@pytest.mark.parametrize("raw,expected", [("0", 0.0005), ("1", 0.05), ("nan", 0.005), ("abc", 0.005),
                                          ("inf", 0.005), ("0.01", 0.01)])
def test_dead_eps_clamped_and_defaulted(monkeypatch, raw, expected):
    """Dead-eps clamps to [0.0005, 0.05]; non-finite or non-numeric falls back to 0.005."""
    _clear(monkeypatch)
    monkeypatch.setenv("CLIO_DPOF_DW_DEAD_EPS", raw)
    assert load_dpof_config().dw_dead_eps == pytest.approx(expected)


@pytest.mark.parametrize("raw,expected", [("1", True), ("true", True), ("0", False)])
def test_quality_apply_flag(monkeypatch, raw, expected):
    """Quality-apply flag parses truthy strings; unset gives False."""
    _clear(monkeypatch)
    monkeypatch.setenv("CLIO_DPOF_DW_QUALITY_APPLY", raw)
    assert load_dpof_config().dw_quality_apply is expected


@pytest.mark.parametrize("value,expected", [
    ("LIVE ", "live"),
    ("x", "off"),
    (None, "off"),
    (5, "off"),
    ("shadow", "shadow"),
])
def test_dynamic_weights_mode_on_directly_built_config(value, expected):
    """dynamic_weights_mode normalises whatever string a DPOFConfig instance carries."""
    assert dynamic_weights_mode(DPOFConfig(dynamic_weights=value)) == expected


def test_validate_adds_no_problem_for_dynamic_knobs():
    """Default config validate() reports nothing about the dynamic-weights knobs."""
    problems = DPOFConfig().validate()
    assert isinstance(problems, list)
    assert not any("dynamic" in p or "dw_" in p for p in problems)
