import math

from agent.integrations import trade_idea_feedback as tif


class FakeGapBridge:
    def __init__(self, *, tier="CAUTION", min_conviction=0.30):
        self._tier = tier
        self._min_conviction = min_conviction

    def get_drawdown_state(self):
        return {}

    def get_min_conviction(self):
        return self._min_conviction

    def get_current_tier(self):
        return self._tier

    def is_leveraged_etf_allowed(self):
        return True

    def get_sector_concentration(self):
        return {}


def _install_gap_bridge(monkeypatch, bridge):
    monkeypatch.setenv("MOLTBOT_GAP_CONVICTION_ENABLED", "1")
    monkeypatch.setenv("MOLTBOT_GAP_CONVICTION_LOG_ONLY", "1")
    monkeypatch.setattr(tif, "_get_gap_bridge", lambda: bridge)


def test_gap_missing_or_nan_conviction_is_dropped_and_flagged(monkeypatch):
    _install_gap_bridge(monkeypatch, FakeGapBridge())

    idea = {"ticker": "SPY", "conviction": math.nan, "confidence": None}
    out = tif._apply_gap_conviction_filter(idea)

    assert out["gap_filtered"] is True
    assert out["gap_adjusted_conviction"] is None
    assert "missing/invalid" in out["gap_conviction_warning"]
    assert out["conviction_bounds_warning"] == out["gap_conviction_warning"]


def test_gap_zero_conviction_remains_valid_not_missing(monkeypatch):
    _install_gap_bridge(monkeypatch, FakeGapBridge(tier="CAUTION", min_conviction=0.30))

    out = tif._apply_gap_conviction_filter({"ticker": "SPY", "conviction": 0.0})

    assert out["gap_filtered"] is True
    assert out["gap_adjusted_conviction"] == 0.0
    assert out.get("conviction_bounds_warning") is None
    assert "below tier minimum" in out["gap_conviction_warning"]


def test_gap_out_of_range_numeric_conviction_clamps_with_attribution(monkeypatch):
    _install_gap_bridge(monkeypatch, FakeGapBridge(tier="NORMAL", min_conviction=0.30))

    out = tif._apply_gap_conviction_filter({"ticker": "SPY", "conviction": 1.4})

    assert out["gap_filtered"] is False
    assert out["gap_adjusted_conviction"] == 1.0
    assert out["conviction_bounds_applied"] is True
    assert out["conviction_bounds_original"] == 1.4
    assert "clamped to 1.0000" in out["conviction_bounds_warning"]


def test_current_adjusted_conviction_skips_null_gap_value():
    value, source = tif._current_adjusted_conviction(
        {"gap_adjusted_conviction": None, "conviction": 0.63, "confidence": 0.12}
    )

    assert value == 0.63
    assert source == "conviction"


def test_conviction_helper_rejects_bool_and_nonfinite_values():
    assert tif._select_idea_conviction({"conviction": True, "confidence": float("inf")})[0] is None
    assert tif._select_idea_conviction({"conviction": -0.2}) == (0.0, "conviction", -0.2, True)
