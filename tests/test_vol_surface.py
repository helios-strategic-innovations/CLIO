"""Tests for agent/integrations/vol_surface.py ("V", INTEL_V3_SPEC.md).

Hermetic: no network. agent.integrations.uw_flow_intel._uw_get is
monkeypatched per-test (it is imported lazily inside vol_surface.py, so
patching the attribute on the real uw_flow_intel module is sufficient and
never touches the network). All file I/O goes through a tmp_path
CLIO_VOL_SURFACE_FILE env override -- never the real data/ tree.
"""

import asyncio
import json
from datetime import date, datetime, timedelta, timezone

import pytest

import agent.integrations.vol_surface as vs
from agent.config import config


def _run(coro):
    return asyncio.run(coro)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


@pytest.fixture(autouse=True)
def _isolated_file(tmp_path, monkeypatch):
    """Every test gets its own state file and a clean read cache."""
    path = tmp_path / "vol_surface.json"
    monkeypatch.setenv("CLIO_VOL_SURFACE_FILE", str(path))
    vs._cache.clear()
    # Freeze "today" for the realized-vol window-freshness check (fixture rows
    # are dated 2026-09-23/24); the refresh-window gate is patched per test.
    from zoneinfo import ZoneInfo
    monkeypatch.setattr(vs, "_now_et", lambda: datetime(2026, 9, 25, 12, 0, tzinfo=ZoneInfo("America/New_York")))
    return path


def _write_state(path, symbols: dict):
    path.write_text(json.dumps({"generated_at": _iso(datetime.now(timezone.utc)), "symbols": symbols}))


def _entry(symbol="SPY", age_seconds=0, **extra):
    e = {
        "symbol": symbol,
        "iv_front": 18.0,
        "iv_30d": 19.0,
        "term_slope": -1.0,
        "term_state": "contango",
        "rv_20d": 15.0,
        "iv_rv_spread": 4.0,
        "iv_rank": 60.0,
        "skew_25d": 2.5,
        "richness": "rich",
        "as_of": _iso(datetime.now(timezone.utc) - timedelta(seconds=age_seconds)),
        "sources": ["term_structure", "vol_stats", "realized", "skew"],
        "quality_notes": [],
    }
    e.update(extra)
    return e


# ═══════════════════════════════════════════════════════════════════════════
#  _compute_term_structure
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeTermStructure:
    def test_empty_returns_all_none(self):
        r = vs._compute_term_structure([])
        assert r == {"iv_front": None, "iv_30d": None, "term_slope": None, "term_state": None}

    def test_none_input_returns_all_none(self):
        r = vs._compute_term_structure(None)
        assert r["iv_front"] is None

    def test_normal_contango(self):
        # front (7dte) cheaper than 30d -> negative slope -> contango
        rows = [
            {"dte": 7, "expiry": "2026-10-02", "volatility": "0.18"},
            {"dte": 30, "expiry": "2026-10-25", "volatility": "0.20"},
            {"dte": 90, "expiry": "2026-12-24", "volatility": "0.22"},
        ]
        r = vs._compute_term_structure(rows)
        assert r["iv_front"] == pytest.approx(18.0)
        assert r["iv_30d"] == pytest.approx(20.0)
        assert r["term_slope"] == pytest.approx(-2.0)
        assert r["term_state"] == "contango"

    def test_normal_backwardation(self):
        rows = [
            {"dte": 3, "expiry": "2026-09-28", "volatility": "0.35"},
            {"dte": 28, "expiry": "2026-10-23", "volatility": "0.22"},
        ]
        r = vs._compute_term_structure(rows)
        assert r["term_slope"] == pytest.approx(13.0)
        assert r["term_state"] == "backwardation"

    def test_flat_within_neutral_band(self):
        rows = [
            {"dte": 5, "expiry": "2026-09-30", "volatility": "0.200"},
            {"dte": 31, "expiry": "2026-10-26", "volatility": "0.203"},
        ]
        r = vs._compute_term_structure(rows)
        assert r["term_slope"] == pytest.approx(-0.3)
        assert r["term_state"] == "flat"

    def test_boundary_exactly_at_band_edge_is_flat(self):
        # exactly +0.5 is NOT > band -> flat (strict inequality)
        rows = [
            {"dte": 5, "expiry": "2026-09-30", "volatility": "0.205"},
            {"dte": 31, "expiry": "2026-10-26", "volatility": "0.200"},
        ]
        r = vs._compute_term_structure(rows)
        assert r["term_slope"] == pytest.approx(0.5)
        assert r["term_state"] == "flat"

    def test_single_entry_gives_zero_slope_flat(self):
        rows = [{"dte": 14, "expiry": "2026-10-09", "volatility": "0.21"}]
        r = vs._compute_term_structure(rows)
        assert r["iv_front"] == r["iv_30d"] == pytest.approx(21.0)
        assert r["term_slope"] == pytest.approx(0.0)
        assert r["term_state"] == "flat"

    def test_zero_or_negative_iv_filtered_out(self):
        rows = [
            {"dte": 5, "expiry": "2026-09-30", "volatility": "0"},
            {"dte": 30, "expiry": "2026-10-25", "volatility": "-0.1"},
            {"dte": 60, "expiry": "2026-11-24", "volatility": "0.25"},
        ]
        r = vs._compute_term_structure(rows)
        assert r["iv_front"] == pytest.approx(25.0)  # only surviving entry

    def test_negative_dte_filtered_out(self):
        rows = [{"dte": -1, "expiry": "2026-09-01", "volatility": "0.2"}]
        r = vs._compute_term_structure(rows)
        assert r["iv_front"] is None

    def test_unparseable_dte_skipped(self):
        rows = [
            {"dte": "not-a-number", "expiry": "2026-10-01", "volatility": "0.2"},
            {"dte": 30, "expiry": "2026-10-25", "volatility": "0.22"},
        ]
        r = vs._compute_term_structure(rows)
        assert r["iv_front"] == pytest.approx(22.0)

    def test_malformed_rows_ignored(self):
        rows = ["not-a-dict", {"dte": 30, "expiry": "2026-10-25", "volatility": "0.2"}]
        r = vs._compute_term_structure(rows)
        assert r["iv_front"] == pytest.approx(20.0)


# ═══════════════════════════════════════════════════════════════════════════
#  _pick_skew_expiry
# ═══════════════════════════════════════════════════════════════════════════


class TestPickSkewExpiry:
    def test_picks_closest_to_30dte(self):
        rows = [
            {"dte": 7, "expiry": "2026-10-02"},
            {"dte": 28, "expiry": "2026-10-23"},
            {"dte": 60, "expiry": "2026-11-24"},
        ]
        assert vs._pick_skew_expiry(rows) == "2026-10-23"

    def test_excludes_near_dte_below_minimum(self):
        rows = [
            {"dte": 1, "expiry": "2026-09-26"},
            {"dte": 3, "expiry": "2026-09-28"},
        ]
        assert vs._pick_skew_expiry(rows) is None

    def test_empty_returns_none(self):
        assert vs._pick_skew_expiry([]) is None
        assert vs._pick_skew_expiry(None) is None

    def test_missing_expiry_field_skipped(self):
        rows = [{"dte": 30}]
        assert vs._pick_skew_expiry(rows) is None

    def test_truncates_datetime_expiry_to_date(self):
        rows = [{"dte": 30, "expiry": "2026-10-25T00:00:00Z"}]
        assert vs._pick_skew_expiry(rows) == "2026-10-25"


# ═══════════════════════════════════════════════════════════════════════════
#  _compute_skew
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeSkew:
    def test_uses_most_recent_row(self):
        rows = [
            {"date": "2026-09-20", "risk_reversal": "-0.01"},
            {"date": "2026-09-24", "risk_reversal": "-0.021"},
        ]
        assert vs._compute_skew(rows) == pytest.approx(-2.1)

    def test_falls_back_past_unparseable_trailing_row(self):
        rows = [
            {"date": "2026-09-20", "risk_reversal": "0.015"},
            {"date": "2026-09-24", "risk_reversal": None},
        ]
        assert vs._compute_skew(rows) == pytest.approx(1.5)

    def test_empty_returns_none(self):
        assert vs._compute_skew([]) is None
        assert vs._compute_skew(None) is None

    def test_all_unparseable_returns_none(self):
        assert vs._compute_skew([{"risk_reversal": "n/a"}]) is None


# ═══════════════════════════════════════════════════════════════════════════
#  _extract_realized_vol — the S177 forward-shift fallback
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractRealizedVol:
    TODAY = date(2026, 9, 25)

    def test_latest_completed_window_used(self):
        rows = [{"date": "2026-08-25", "unshifted_rv_date": "2026-09-24", "realized_volatility": "0.15"}]
        rv, used_fallback = vs._extract_realized_vol(rows, {"rv": 0.99}, today=self.TODAY)
        assert rv == pytest.approx(15.0)
        assert used_fallback is False

    def test_s177_walks_back_past_open_windows(self):
        """S177: the newest rows are null until their window closes; the latest
        non-null row is the most recently completed trailing window."""
        rows = [
            {"date": "2026-08-24", "unshifted_rv_date": "2026-09-23", "realized_volatility": "0.12"},
            {"date": "2026-08-25", "unshifted_rv_date": "2026-09-24", "realized_volatility": "0.14"},
            {"date": "2026-09-23", "unshifted_rv_date": "2026-10-22", "realized_volatility": None},
            {"date": "2026-09-24", "unshifted_rv_date": "2026-10-23", "realized_volatility": None},
        ]
        rv, used_fallback = vs._extract_realized_vol(rows, {"rv": 0.16}, today=self.TODAY)
        assert rv == pytest.approx(14.0)
        assert used_fallback is False

    def test_zero_values_are_skipped(self):
        rows = [
            {"date": "2026-08-25", "unshifted_rv_date": "2026-09-24", "realized_volatility": "0.13"},
            {"date": "2026-09-24", "unshifted_rv_date": "2026-10-23", "realized_volatility": "0"},
        ]
        rv, used_fallback = vs._extract_realized_vol(rows, {"rv": 0.18}, today=self.TODAY)
        assert rv == pytest.approx(13.0)
        assert used_fallback is False

    def test_stale_latest_window_falls_back_to_stats(self):
        """The latest completed window ended too long ago -> not presented as
        current; stats.rv is used and nothing older is dug up."""
        rows = [
            {"date": "2026-07-01", "unshifted_rv_date": "2026-07-31", "realized_volatility": "0.30"},
            {"date": "2026-09-24", "unshifted_rv_date": "2026-10-23", "realized_volatility": None},
        ]
        rv, used_fallback = vs._extract_realized_vol(rows, {"rv": 0.20}, today=self.TODAY)
        assert rv == pytest.approx(20.0)
        assert used_fallback is True

    def test_missing_unshifted_date_uses_row_date(self):
        rows = [{"date": "2026-09-23", "realized_volatility": "0.15"}]
        rv, used_fallback = vs._extract_realized_vol(rows, {"rv": 0.99}, today=self.TODAY)
        assert rv == pytest.approx(15.0) and used_fallback is False

    def test_undatable_row_falls_back(self):
        rows = [{"date": "n/a", "realized_volatility": "0.15"}]
        rv, used_fallback = vs._extract_realized_vol(rows, {"rv": 0.17}, today=self.TODAY)
        assert rv == pytest.approx(17.0) and used_fallback is True

    def test_all_null_falls_back_to_stats(self):
        rows = [{"date": "2026-09-24", "unshifted_rv_date": "2026-10-23", "realized_volatility": None}]
        rv, used_fallback = vs._extract_realized_vol(rows, {"rv": 0.16}, today=self.TODAY)
        assert rv == pytest.approx(16.0) and used_fallback is True

    def test_empty_rows_falls_back_to_stats(self):
        rv, used_fallback = vs._extract_realized_vol([], {"rv": 0.17}, today=self.TODAY)
        assert rv == pytest.approx(17.0)
        assert used_fallback is True

    def test_both_sources_missing_returns_none(self):
        rv, used_fallback = vs._extract_realized_vol([], {"rv": None}, today=self.TODAY)
        assert rv is None
        assert used_fallback is False

    def test_stats_rv_zero_also_treated_as_missing(self):
        rv, used_fallback = vs._extract_realized_vol([], {"rv": 0.0}, today=self.TODAY)
        assert rv is None
        assert used_fallback is False


# ═══════════════════════════════════════════════════════════════════════════
#  _classify_richness
# ═══════════════════════════════════════════════════════════════════════════


class TestClassifyRichness:
    def test_none_spread_is_none(self):
        assert vs._classify_richness(None, 80.0) is None

    def test_rich_high_spread_high_rank(self):
        assert vs._classify_richness(5.0, 70.0) == "rich"

    def test_rich_missing_rank_still_rich(self):
        assert vs._classify_richness(5.0, None) == "rich"

    def test_high_spread_low_rank_is_fair_not_rich(self):
        assert vs._classify_richness(5.0, 20.0) == "fair"

    def test_cheap_low_spread_low_rank(self):
        assert vs._classify_richness(-3.0, 20.0) == "cheap"

    def test_cheap_missing_rank_still_cheap(self):
        assert vs._classify_richness(-3.0, None) == "cheap"

    def test_low_spread_high_rank_is_fair_not_cheap(self):
        assert vs._classify_richness(-3.0, 80.0) == "fair"

    def test_mid_spread_is_fair(self):
        assert vs._classify_richness(0.5, 50.0) == "fair"

    def test_boundary_rich_threshold_inclusive(self):
        assert vs._classify_richness(3.0, 50.0) == "rich"

    def test_boundary_cheap_threshold_inclusive(self):
        assert vs._classify_richness(-2.0, 50.0) == "cheap"


# ═══════════════════════════════════════════════════════════════════════════
#  _fetch_symbol_bundle / _compose_symbol_result — UW layer + assembly
# ═══════════════════════════════════════════════════════════════════════════


def _uw_payloads_full(ticker="SPY"):
    return {
        f"/stock/{ticker}/volatility/term-structure": {
            "data": [
                {"dte": 7, "expiry": "2026-10-02", "volatility": "0.18"},
                {"dte": 30, "expiry": "2026-10-25", "volatility": "0.20"},
            ]
        },
        f"/stock/{ticker}/volatility/stats": {
            "data": {"date": "2026-09-24", "iv": "0.20", "iv_rank": "0.55", "rv": "0.16", "ticker": ticker}
        },
        f"/stock/{ticker}/volatility/realized": {
            "data": [
                {"date": "2026-09-23", "realized_volatility": "0.155"},
                {"date": "2026-09-24", "realized_volatility": "0.16"},
            ]
        },
        f"/stock/{ticker}/historical-risk-reversal-skew": {
            "data": [{"date": "2026-09-24", "delta": 25, "risk_reversal": "0.025"}]
        },
    }


def _install_fake_uw_get(monkeypatch, payloads: dict, *, calls=None, raise_on=None):
    """payloads: {endpoint_prefix: payload}. Matches by exact endpoint (the
    part before '?') since vol_surface always builds f"/stock/{symbol}/...".
    """
    calls = calls if calls is not None else []

    async def _fake(endpoint, params=None, *, skip_moltbot_budget=False):
        calls.append((endpoint, params))
        if raise_on and raise_on in endpoint:
            raise RuntimeError("simulated UW failure")
        return payloads.get(endpoint, {})

    import agent.integrations.uw_flow_intel as uw_flow_intel

    monkeypatch.setattr(uw_flow_intel, "_uw_get", _fake)
    return calls


class TestFetchSymbolBundle:
    def test_full_success_makes_4_calls_all_sources(self, monkeypatch):
        calls = _install_fake_uw_get(monkeypatch, _uw_payloads_full("SPY"))
        bundle = _run(vs._fetch_symbol_bundle("SPY"))
        assert bundle["calls"] == 4
        # "realized" is credited at compose time, only when a row was actually used
        assert set(bundle["sources"]) == {"term_structure", "vol_stats", "skew"}
        entry = vs._compose_symbol_result("SPY", bundle, "2026-09-24T12:00:00+00:00")
        assert "realized" in entry["sources"]
        assert len(calls) == 4
        # skew call carries delta + the expiry derived from term-structure
        skew_call = [c for c in calls if "risk-reversal" in c[0]][0]
        assert skew_call[1] == {"delta": 25, "expiry": "2026-10-25"}

    def test_term_structure_failure_skips_skew_call(self, monkeypatch):
        payloads = _uw_payloads_full("SPY")
        payloads["/stock/SPY/volatility/term-structure"] = {}
        calls = _install_fake_uw_get(monkeypatch, payloads)
        bundle = _run(vs._fetch_symbol_bundle("SPY"))
        assert bundle["calls"] == 3  # skew skipped -- no expiry available
        assert bundle["skew_expiry"] is None
        assert "skew" not in bundle["sources"]
        assert "term_structure" not in bundle["sources"]

    def test_total_failure_yields_zero_sources(self, monkeypatch):
        _install_fake_uw_get(monkeypatch, {})
        bundle = _run(vs._fetch_symbol_bundle("SPY"))
        assert bundle["sources"] == []
        assert bundle["calls"] == 3  # no expiry -> skew skipped too


class TestComposeSymbolResult:
    def test_full_bundle_produces_populated_entry(self, monkeypatch):
        _install_fake_uw_get(monkeypatch, _uw_payloads_full("SPY"))
        bundle = _run(vs._fetch_symbol_bundle("SPY"))
        entry = vs._compose_symbol_result("SPY", bundle, "2026-09-24T12:00:00+00:00")
        assert entry["symbol"] == "SPY"
        assert entry["iv_front"] == pytest.approx(18.0)
        assert entry["iv_30d"] == pytest.approx(20.0)
        assert entry["term_state"] == "contango"
        assert entry["rv_20d"] == pytest.approx(16.0)
        assert entry["iv_rv_spread"] == pytest.approx(4.0)  # 20.0 - 16.0
        assert entry["iv_rank"] == pytest.approx(55.0)
        assert entry["skew_25d"] == pytest.approx(2.5)
        assert entry["richness"] == "rich"
        assert entry["quality_notes"] == []

    def test_empty_bundle_gives_all_none_with_notes(self):
        bundle = {
            "ts_rows": [], "stats_data": {}, "realized_rows": [], "skew_rows": [],
            "skew_expiry": None, "calls": 3, "sources": [],
        }
        entry = vs._compose_symbol_result("SPY", bundle, "2026-09-24T12:00:00+00:00")
        for key in ("iv_front", "iv_30d", "term_slope", "term_state", "rv_20d",
                    "iv_rv_spread", "iv_rank", "skew_25d", "richness"):
            assert entry[key] is None, key
        assert any("term-structure unavailable" in n for n in entry["quality_notes"])
        assert any("skew skipped" in n for n in entry["quality_notes"])

    def test_missing_term_structure_falls_back_to_stats_iv_for_spread(self):
        bundle = {
            "ts_rows": [], "stats_data": {"iv": "0.22", "rv": "0.18", "iv_rank": "0.4"},
            "realized_rows": [], "skew_rows": [], "skew_expiry": None,
            "calls": 3, "sources": ["vol_stats"],
        }
        entry = vs._compose_symbol_result("SPY", bundle, "2026-09-24T12:00:00+00:00")
        assert entry["iv_30d"] is None  # term-structure itself still None
        assert entry["rv_20d"] == pytest.approx(18.0)
        assert entry["iv_rv_spread"] == pytest.approx(4.0)  # 22.0 - 18.0 via stats.iv fallback
        assert any("used volatility/stats.iv" in n for n in entry["quality_notes"])

    def test_s177_fallback_note_present_when_triggered(self):
        bundle = {
            "ts_rows": [{"dte": 30, "expiry": "2026-10-25", "volatility": "0.2"}],
            "stats_data": {"iv": "0.2", "rv": "0.19", "iv_rank": "0.5"},
            "realized_rows": [{"date": "2026-09-24", "realized_volatility": None}],
            "skew_rows": [], "skew_expiry": None, "calls": 3, "sources": ["term_structure", "vol_stats", "realized"],
        }
        entry = vs._compose_symbol_result("SPY", bundle, "2026-09-24T12:00:00+00:00")
        assert entry["rv_20d"] == pytest.approx(19.0)
        assert any("S177" in n for n in entry["quality_notes"])

    def test_skew_with_expiry_but_no_rows_gets_the_empty_or_denied_note(self):
        """R-13: the note no longer blames the endpoint; empty or denied (quota) skew reads the same way."""
        bundle = {
            "ts_rows": [{"dte": 30, "expiry": "2026-10-25", "volatility": "0.2"}],
            "stats_data": {"iv": "0.2", "rv": "0.19", "iv_rank": "0.5"},
            "realized_rows": [], "skew_rows": [], "skew_expiry": "2026-10-25",
            "calls": 4, "sources": ["term_structure", "vol_stats"],
        }
        entry = vs._compose_symbol_result("SPY", bundle, "2026-09-24T12:00:00+00:00")
        assert entry["skew_25d"] is None
        assert "skew unavailable (empty or denied)" in entry["quality_notes"]
        assert not any("risk_reversal" in n for n in entry["quality_notes"])


# ═══════════════════════════════════════════════════════════════════════════
#  _resolve_symbols
# ═══════════════════════════════════════════════════════════════════════════


class TestResolveSymbols:
    def test_default_when_dpof_import_fails(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def _blocked(name, *a, **kw):
            if name == "agent.dpof.config" or name.startswith("agent.dpof"):
                raise ImportError("blocked for test")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", _blocked)
        assert vs._resolve_symbols() == ["SPY", "QQQ"]

    def test_uses_dpof_underlyings_when_available(self, monkeypatch):
        class _FakeCfg:
            underlyings = ("qqq", "spy", "qqq")  # lowercase + dup, order preserved

        monkeypatch.setattr("agent.dpof.config.load_dpof_config", lambda: _FakeCfg())
        assert vs._resolve_symbols() == ["QQQ", "SPY"]

    def test_empty_underlyings_falls_back_to_default(self, monkeypatch):
        class _FakeCfg:
            underlyings = ()

        monkeypatch.setattr("agent.dpof.config.load_dpof_config", lambda: _FakeCfg())
        assert vs._resolve_symbols() == ["SPY", "QQQ"]


# ═══════════════════════════════════════════════════════════════════════════
#  refresh_vol_surface — end to end
# ═══════════════════════════════════════════════════════════════════════════


class TestRefreshVolSurface:
    def _prep(self, monkeypatch, symbols=("SPY", "QQQ")):
        monkeypatch.setattr(vs, "_within_refresh_window", lambda: True)
        monkeypatch.setattr(vs, "_resolve_symbols", lambda: list(symbols))
        monkeypatch.setattr(config, "uw_api_key", "uw-test-key")

    def test_full_success_writes_file_and_counts_calls(self, monkeypatch, _isolated_file):
        self._prep(monkeypatch)
        payloads = {}
        payloads.update(_uw_payloads_full("SPY"))
        payloads.update(_uw_payloads_full("QQQ"))
        _install_fake_uw_get(monkeypatch, payloads)

        result = _run(vs.refresh_vol_surface())
        assert result is not None
        assert sorted(result["refreshed"]) == ["QQQ", "SPY"]
        assert result["errors"] == []
        assert result["api_calls"] == 8

        on_disk = json.loads(_isolated_file.read_text())
        assert set(on_disk["symbols"].keys()) == {"SPY", "QQQ"}
        assert on_disk["symbols"]["SPY"]["richness"] == "rich"
        # no leftover tmp files
        assert not list(_isolated_file.parent.glob("*.tmp"))

    def test_no_uw_key_skips_entirely(self, monkeypatch, _isolated_file):
        monkeypatch.setattr(vs, "_within_refresh_window", lambda: True)
        monkeypatch.setattr(vs, "_resolve_symbols", lambda: ["SPY"])
        monkeypatch.setattr(config, "uw_api_key", "")
        calls = _install_fake_uw_get(monkeypatch, {})

        result = _run(vs.refresh_vol_surface())
        assert result is None
        assert calls == []
        assert not _isolated_file.exists()

    def test_outside_refresh_window_skips_without_calling_uw(self, monkeypatch, _isolated_file):
        monkeypatch.setattr(vs, "_within_refresh_window", lambda: False)
        monkeypatch.setattr(config, "uw_api_key", "uw-test-key")
        calls = _install_fake_uw_get(monkeypatch, {})

        result = _run(vs.refresh_vol_surface())
        assert result is None
        assert calls == []

    def test_partial_failure_preserves_previous_entry_for_failed_symbol(self, monkeypatch, _isolated_file):
        self._prep(monkeypatch, symbols=("SPY", "QQQ"))
        previous_qqq = _entry(symbol="QQQ", richness="cheap")
        _write_state(_isolated_file, {"QQQ": previous_qqq})
        vs._cache.clear()

        payloads = dict(_uw_payloads_full("SPY"))  # QQQ payloads intentionally absent -> {}
        _install_fake_uw_get(monkeypatch, payloads)

        result = _run(vs.refresh_vol_surface())
        assert result is not None
        assert "SPY" in result["refreshed"]
        assert "QQQ" in result["errors"]

        on_disk = json.loads(_isolated_file.read_text())
        assert on_disk["symbols"]["QQQ"] == previous_qqq  # untouched
        assert on_disk["symbols"]["SPY"]["richness"] == "rich"

    def test_total_failure_for_new_symbol_writes_all_none_entry(self, monkeypatch, _isolated_file):
        self._prep(monkeypatch, symbols=("SPY",))
        _install_fake_uw_get(monkeypatch, {})  # every endpoint returns {}

        result = _run(vs.refresh_vol_surface())
        assert result is not None
        assert result["errors"] == ["SPY"]

        on_disk = json.loads(_isolated_file.read_text())
        entry = on_disk["symbols"]["SPY"]
        assert entry["iv_front"] is None
        assert "all UW endpoints unavailable this cycle" in entry["quality_notes"]

    def test_fetch_exception_never_propagates(self, monkeypatch, _isolated_file):
        self._prep(monkeypatch, symbols=("SPY",))
        _install_fake_uw_get(monkeypatch, {}, raise_on="term-structure")

        result = _run(vs.refresh_vol_surface())
        assert result is not None  # never raises
        assert result["errors"] == ["SPY"]

    def test_outer_guard_never_raises_on_write_failure(self, monkeypatch, _isolated_file):
        self._prep(monkeypatch, symbols=("SPY",))
        _install_fake_uw_get(monkeypatch, _uw_payloads_full("SPY"))
        monkeypatch.setattr(vs, "_write_atomic", lambda payload: False)

        result = _run(vs.refresh_vol_surface())
        assert result is None

    def test_outer_guard_never_raises_on_unexpected_error(self, monkeypatch, _isolated_file):
        self._prep(monkeypatch, symbols=("SPY",))

        def boom():
            raise RuntimeError("boom")

        monkeypatch.setattr(vs, "_load_existing", boom)
        result = _run(vs.refresh_vol_surface())
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
#  get_vol_surface — read path
# ═══════════════════════════════════════════════════════════════════════════


class TestGetVolSurface:
    def test_missing_file_returns_none(self):
        assert vs.get_vol_surface("SPY") is None

    def test_empty_symbol_returns_none(self, _isolated_file):
        _write_state(_isolated_file, {"SPY": _entry()})
        assert vs.get_vol_surface("") is None

    def test_fresh_entry_returned(self, _isolated_file):
        _write_state(_isolated_file, {"SPY": _entry(age_seconds=60)})
        data = vs.get_vol_surface("SPY")
        assert data is not None
        assert data["symbol"] == "SPY"
        assert data["richness"] == "rich"

    def test_case_insensitive_lookup(self, _isolated_file):
        _write_state(_isolated_file, {"SPY": _entry(age_seconds=0)})
        assert vs.get_vol_surface("spy") is not None

    def test_unknown_symbol_returns_none(self, _isolated_file):
        _write_state(_isolated_file, {"SPY": _entry()})
        assert vs.get_vol_surface("QQQ") is None

    def test_stale_entry_returns_none(self, _isolated_file):
        _write_state(_isolated_file, {"SPY": _entry(age_seconds=2 * 3600)})
        assert vs.get_vol_surface("SPY", max_age_seconds=3600) is None

    def test_custom_max_age_accepts_older_data(self, _isolated_file):
        _write_state(_isolated_file, {"SPY": _entry(age_seconds=2 * 3600)})
        data = vs.get_vol_surface("SPY", max_age_seconds=24 * 3600)
        assert data is not None

    def test_malformed_json_returns_none(self, _isolated_file):
        _isolated_file.write_text("{not valid json")
        assert vs.get_vol_surface("SPY") is None

    def test_missing_as_of_returns_none(self, _isolated_file):
        entry = _entry()
        del entry["as_of"]
        _write_state(_isolated_file, {"SPY": entry})
        assert vs.get_vol_surface("SPY") is None

    def test_future_as_of_returns_none(self, _isolated_file):
        _write_state(_isolated_file, {"SPY": _entry(age_seconds=-3600)})
        assert vs.get_vol_surface("SPY") is None

    def test_symbols_not_a_dict_returns_none(self, _isolated_file):
        _isolated_file.write_text(json.dumps({"symbols": "nope"}))
        assert vs.get_vol_surface("SPY") is None

    def test_mtime_cache_reflects_updates(self, _isolated_file):
        _write_state(_isolated_file, {"SPY": _entry(richness="rich")})
        assert vs.get_vol_surface("SPY")["richness"] == "rich"
        _write_state(_isolated_file, {"SPY": _entry(richness="cheap")})
        assert vs.get_vol_surface("SPY")["richness"] == "cheap"

    def test_never_raises_on_unreadable_path(self, monkeypatch):
        # point at a directory (not a file): os.stat succeeds, open() raises IsADirectoryError
        monkeypatch.setenv("CLIO_VOL_SURFACE_FILE", "/")
        vs._cache.clear()
        assert vs.get_vol_surface("SPY") is None


# ═══════════════════════════════════════════════════════════════════════════
#  _write_atomic
# ═══════════════════════════════════════════════════════════════════════════


class TestWriteAtomic:
    def test_success_leaves_no_tmp(self, _isolated_file):
        ok = vs._write_atomic({"generated_at": "x", "symbols": {}})
        assert ok is True
        assert _isolated_file.exists()
        assert not list(_isolated_file.parent.glob("*.tmp"))

    def test_failure_returns_false(self, tmp_path, monkeypatch):
        blocker = tmp_path / "blocker"
        blocker.write_text("x")
        bad_path = blocker / "vol_surface.json"
        monkeypatch.setenv("CLIO_VOL_SURFACE_FILE", str(bad_path))
        ok = vs._write_atomic({"symbols": {}})
        assert ok is False


# ═══════════════════════════════════════════════════════════════════════════
#  Advisory-only guarantee (grep-provable, but assert the module surface too)
# ═══════════════════════════════════════════════════════════════════════════


def test_module_never_imports_dpof_execution_paths():
    import ast
    import inspect

    src = inspect.getsource(vs)
    tree = ast.parse(src)
    forbidden = ("agent.dpof.broker", "agent.dpof.orders", "agent.dpof.engine")
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not any(node.module.startswith(f) for f in forbidden), node.module
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not any(alias.name.startswith(f) for f in forbidden), alias.name


# ═══════════════════════════════════════════════════════════════════════════
#  Pass-2 review fixes: source attribution + single-flight refresh
# ═══════════════════════════════════════════════════════════════════════════


def test_realized_not_credited_when_stats_fallback_used():
    bundle = {
        "ts_rows": [{"dte": 30, "expiry": "2026-10-25", "volatility": "0.2"}],
        "stats_data": {"iv": "0.2", "rv": "0.19", "iv_rank": "0.5"},
        "realized_rows": [{"date": "2026-09-24", "unshifted_rv_date": "2026-10-23", "realized_volatility": None}],
        "skew_rows": [], "skew_expiry": None, "calls": 3, "sources": ["term_structure", "vol_stats"],
    }
    entry = vs._compose_symbol_result("SPY", bundle, "2026-09-24T12:00:00+00:00")
    assert entry["rv_20d"] == pytest.approx(19.0)
    assert "realized" not in entry["sources"]


def test_refresh_is_single_flight(monkeypatch):
    """A second call while one refresh is running returns None immediately
    instead of racing the first one's merge-write."""
    started = asyncio.Event()
    release = asyncio.Event()
    calls = {"n": 0}

    async def _slow_locked(context=None):
        calls["n"] += 1
        started.set()
        await release.wait()
        return {"refreshed": ["SPY"]}

    monkeypatch.setattr(vs, "_refresh_vol_surface_locked", _slow_locked)

    async def _go():
        first = asyncio.create_task(vs.refresh_vol_surface())
        await started.wait()
        second = await vs.refresh_vol_surface()
        release.set()
        return await first, second

    first, second = _run(_go())
    assert first == {"refreshed": ["SPY"]}
    assert second is None
    assert calls["n"] == 1
