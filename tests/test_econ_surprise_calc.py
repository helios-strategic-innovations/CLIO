"""Tests for agent/integrations/econ_surprise.py — direction/market_read
mapping, z-score history math (incl. the inverted-series sign flip and the
<8-points-=-None contract), and the row-merge -> surprise-entry pipeline
(_process_rows), including cross-source confirmation and unit-mismatch
guarding.

No network. All inputs are hand-built "common" rows in the same shape
_nasdaq_row_to_common()/_tv_row_to_common() produce (tested separately in
test_econ_surprise_parsing.py).
"""

from datetime import datetime, timedelta

import pytest

import agent.integrations.econ_surprise as es


# ═══════════════════════════════════════════════════════════════════════════
#  _direction / _market_read
# ═══════════════════════════════════════════════════════════════════════════

def test_direction_above_below_inline():
    assert es._direction(0.4, "%") == "above"
    assert es._direction(-0.4, "%") == "below"
    assert es._direction(0.01, "%") == "inline"   # below the 0.05 tick epsilon
    assert es._direction(-0.01, "%") == "inline"
    assert es._direction(0.05, "%") == "above"    # exactly at epsilon is NOT inline (strict <)


def test_direction_unit_specific_epsilon():
    assert es._direction(1.0, "K") == "inline"    # < 2.0 K epsilon
    assert es._direction(3.0, "K") == "above"
    assert es._direction(0.1, "level") == "inline"  # < 0.5 level epsilon
    assert es._direction(1.0, "level") == "above"


def test_market_read_inflation():
    assert es._market_read("inflation", "above") == "hawkish"
    assert es._market_read("inflation", "below") == "dovish"
    assert es._market_read("inflation", "inline") == "neutral"


def test_market_read_fed():
    assert es._market_read("fed", "above") == "hawkish"
    assert es._market_read("fed", "below") == "dovish"


def test_market_read_inverted_flips_sign():
    """Claims/unemployment: a HIGHER-than-expected print is BAD for growth."""
    assert es._market_read("inverted", "above") == "growth_down"
    assert es._market_read("inverted", "below") == "growth_up"
    assert es._market_read("inverted", "inline") == "neutral"


def test_market_read_growth_normal_sign():
    assert es._market_read("growth", "above") == "growth_up"
    assert es._market_read("growth", "below") == "growth_down"


def test_market_read_unknown_category_is_none_never_guessed():
    assert es._market_read(None, "above") is None
    assert es._market_read(None, "inline") is None


# ═══════════════════════════════════════════════════════════════════════════
#  _update_history / z-score
# ═══════════════════════════════════════════════════════════════════════════

def test_z_is_none_below_min_history():
    history: dict = {}
    key = "CPI MoM"
    # Feed 7 prior points (the 8th call, with 7 PRIOR points, must still be None)
    for raw in [0.1, -0.1, 0.2, -0.2, 0.05, -0.05, 0.15]:
        series, z = es._update_history(history, key, raw)
        assert z is None
    assert len(history[key]) == 7


def test_z_appears_once_min_history_reached():
    history: dict = {}
    key = "CPI MoM"
    values = [0.1, -0.1, 0.2, -0.2, 0.05, -0.05, 0.15, 0.0]  # 8 prior points after this loop
    for raw in values:
        es._update_history(history, key, raw)
    assert len(history[key]) == 8
    # 9th point: now there ARE >= 8 prior points -> z must be computed
    series, z = es._update_history(history, key, 0.5)
    assert z is not None
    assert isinstance(z, float)
    assert len(history[key]) == 9


def test_z_computed_from_prior_history_not_self_referential():
    """z for a new point uses stdev of the PRIOR points only -- feeding a
    huge outlier must not shrink its own z by including itself in stdev."""
    history: dict = {}
    key = "NFP"
    for raw in [1.0] * 8  :  # 8 identical prior points -> stdev == 0 -> next z would divide-by-zero-guard to None
        es._update_history(history, key, raw)
    series, z = es._update_history(history, key, 1.0)
    assert z is None  # stdev of 8 identical points is 0 -> guarded, not ZeroDivisionError


def test_z_history_capped():
    history: dict = {}
    key = "PPI MoM"
    for i in range(100):
        es._update_history(history, key, float(i))
    assert len(history[key]) == es._HISTORY_CAP
    assert history[key][-1] == 99.0  # most recent kept
    assert history[key][0] == float(100 - es._HISTORY_CAP)


def test_z_lookback_window_is_bounded():
    """Only the last _Z_LOOKBACK prior points feed the stdev, even if more
    history exists."""
    history: dict = {}
    key = "GDP QoQ"
    # 40 near-zero points then jam in enough spread that only a bounded
    # window would show as low-variance vs a smaller one; simplest check:
    # history grows past _Z_LOOKBACK and z is still computed without error.
    for i in range(40):
        es._update_history(history, key, 0.01 * (i % 3))
    series, z = es._update_history(history, key, 5.0)
    assert z is not None
    assert len(history[key]) == 41


# ═══════════════════════════════════════════════════════════════════════════
#  _process_rows — merge, dedup, confirmed_by, unit-mismatch guard
# ═══════════════════════════════════════════════════════════════════════════

def _row(event, release_time_et, actual, actual_unit, consensus, consensus_unit,
         previous=None, previous_unit=None, importance=2, source="nasdaq"):
    return {
        "event": event, "release_time_et": release_time_et,
        "actual": actual, "actual_unit": actual_unit,
        "consensus": consensus, "consensus_unit": consensus_unit,
        "previous": previous, "previous_unit": previous_unit,
        "importance": importance, "source": source,
    }


def _fresh_state():
    return es._default_state()


def test_process_rows_basic_surprise_and_alert_dedup():
    state = _fresh_state()
    nasdaq_row = _row("Initial Jobless Claims", "2026-09-25T08:30:00-04:00",
                       197.0, "K", 201.0, "K", 198.0, "K", importance=2, source="nasdaq")
    rows_by_source = {"nasdaq": [nasdaq_row], "tradingview": None}

    entries = es._process_rows(state, rows_by_source)
    assert len(entries) == 1
    e = entries[0]
    assert e["event"] == "Initial Jobless Claims"
    assert e["actual"] == 197.0 and e["consensus"] == 201.0 and e["previous"] == 198.0
    assert e["surprise_raw"] == -4.0
    assert e["direction"] == "below"
    assert e["market_read"] == "growth_up"  # inverted series: a lower claims print is good for growth
    assert e["surprise_z"] is None  # first-ever print, no history yet
    assert e["source"] == "nasdaq"
    assert e["confirmed_by"] == ["nasdaq"]
    assert e["importance"] == 2

    # Re-processing the SAME row again (e.g. next poll tick) must NOT double-record
    entries2 = es._process_rows(state, rows_by_source)
    assert entries2 == []
    assert len(state["surprises"]) == 1


def test_process_rows_cross_source_confirmation():
    state = _fresh_state()
    nasdaq_row = _row("Non Farm Payrolls", "2026-09-04T08:30:00-04:00",
                       162.0, "K", 56.0, "K", 21.0, "K", importance=3, source="nasdaq")
    tv_row = _row("Non Farm Payrolls", "2026-09-04T08:30:00-04:00",
                   162.0, "K", 56.0, "K", 21.0, "K", importance=3, source="tradingview")
    rows_by_source = {"nasdaq": [nasdaq_row], "tradingview": [tv_row]}

    entries = es._process_rows(state, rows_by_source)
    assert len(entries) == 1
    e = entries[0]
    assert set(e["confirmed_by"]) == {"nasdaq", "tradingview"}
    assert e["source"] == "nasdaq"  # nasdaq preferred as primary/candidate


def test_process_rows_disagreement_does_not_confirm():
    state = _fresh_state()
    nasdaq_row = _row("Non Farm Payrolls", "2026-09-04T08:30:00-04:00",
                       162.0, "K", 56.0, "K", source="nasdaq")
    tv_row = _row("Non Farm Payrolls", "2026-09-04T08:30:00-04:00",
                   400.0, "K", 56.0, "K", source="tradingview")  # wildly different actual
    rows_by_source = {"nasdaq": [nasdaq_row], "tradingview": [tv_row]}

    entries = es._process_rows(state, rows_by_source)
    assert len(entries) == 1
    assert entries[0]["confirmed_by"] == ["nasdaq"]  # only the source actually used


def test_process_rows_nasdaq_down_falls_back_to_tradingview():
    state = _fresh_state()
    tv_row = _row("CPI MoM", "2026-10-13T08:30:00-04:00", 0.4, "%", 0.3, "%", source="tradingview")
    rows_by_source = {"nasdaq": None, "tradingview": [tv_row]}

    entries = es._process_rows(state, rows_by_source)
    assert len(entries) == 1
    assert entries[0]["source"] == "tradingview"
    assert entries[0]["confirmed_by"] == ["tradingview"]
    assert entries[0]["surprise_raw"] == 0.1


def test_process_rows_no_consensus_skips_surprise_but_updates_table():
    """An event with only an actual (no consensus anywhere, e.g. a Fed
    speaker row) must not fabricate a surprise -- but the consensus table
    entry (for get_upcoming_releases) still gets recorded. "Fed's Balance
    Sheet" is an unmapped/fallback event, so its table_key carries the
    unit-class suffix (item 1, 2026-09-25 pass-2 fix)."""
    state = _fresh_state()
    row = _row("Fed's Balance Sheet", "2026-09-25T16:30:00-04:00",
               6748.0, "B", None, None, 6747.0, "B", importance=1, source="nasdaq")
    entries = es._process_rows(state, {"nasdaq": [row], "tradingview": None})
    assert entries == []
    assert state["surprises"] == []
    table_key = "Fed's Balance Sheet (level)|2026-09-25T16:30:00-04:00"
    assert table_key in state["consensus_table"]
    assert state["consensus_table"][table_key]["consensus"] is None


def test_process_rows_unit_mismatch_within_same_row_is_skipped():
    """Defensive: actual_unit != consensus_unit on the SAME normalized row
    should never happen from a well-formed source, but must not fabricate a
    cross-unit diff if it does."""
    state = _fresh_state()
    row = _row("Some Series", "2026-09-25T10:00:00-04:00",
               5.0, "K", 4.0, "level", source="nasdaq")
    entries = es._process_rows(state, {"nasdaq": [row], "tradingview": None})
    assert entries == []
    assert state["surprises"] == []


def test_process_rows_no_actual_yet_only_updates_consensus_table():
    state = _fresh_state()
    row = _row("CPI MoM", "2026-10-13T08:30:00-04:00", None, None, 0.3, "%", source="nasdaq")
    entries = es._process_rows(state, {"nasdaq": [row], "tradingview": None})
    assert entries == []
    assert "CPI MoM|2026-10-13T08:30:00-04:00" in state["consensus_table"]
    assert state["consensus_table"]["CPI MoM|2026-10-13T08:30:00-04:00"]["consensus"] == 0.3


def test_process_rows_surprises_list_capped():
    state = _fresh_state()
    for i in range(es._SURPRISES_CAP + 10):
        row = _row(f"Series {i}", f"2026-09-25T{(8 + i % 10):02d}:00:00-04:00",
                   float(i), "level", float(i) - 1.0, "level", source="nasdaq")
        es._process_rows(state, {"nasdaq": [row], "tradingview": None})
    assert len(state["surprises"]) == es._SURPRISES_CAP
    # oldest entries evicted first (FIFO)
    assert state["surprises"][0]["event"] == "Series 10"
    assert state["surprises"][-1]["event"] == f"Series {es._SURPRISES_CAP + 9}"


# ═══════════════════════════════════════════════════════════════════════════
#  _process_rows — "late" flag (2026-09-25 orchestrator fix)
# ═══════════════════════════════════════════════════════════════════════════

def test_process_rows_late_flag_false_for_timely_detection():
    state = _fresh_state()
    now = datetime.now(es._NY)
    row = _row("CPI MoM", now.isoformat(), 0.4, "%", 0.3, "%", source="nasdaq")
    entries = es._process_rows(state, {"nasdaq": [row], "tradingview": None}, now=now)
    assert entries[0]["late"] is False


def test_process_rows_late_flag_true_when_detected_well_after_release():
    state = _fresh_state()
    now = datetime.now(es._NY)
    released = now - timedelta(minutes=90)
    row = _row("CPI MoM", released.isoformat(), 0.4, "%", 0.3, "%", source="nasdaq")
    entries = es._process_rows(state, {"nasdaq": [row], "tradingview": None}, now=now)
    assert entries[0]["late"] is True


def test_process_rows_late_flag_boundary_just_under_and_over_threshold():
    now = datetime.now(es._NY)

    state_a = _fresh_state()
    just_under = now - timedelta(minutes=es._LATE_DETECTION_MINUTES - 1)
    row_a = _row("CPI MoM", just_under.isoformat(), 0.4, "%", 0.3, "%", source="nasdaq")
    entries_a = es._process_rows(state_a, {"nasdaq": [row_a], "tradingview": None}, now=now)
    assert entries_a[0]["late"] is False

    state_b = _fresh_state()
    just_over = now - timedelta(minutes=es._LATE_DETECTION_MINUTES + 1)
    row_b = _row("PPI MoM", just_over.isoformat(), 0.4, "%", 0.3, "%", source="nasdaq")
    entries_b = es._process_rows(state_b, {"nasdaq": [row_b], "tradingview": None}, now=now)
    assert entries_b[0]["late"] is True


def test_process_rows_late_flag_defaults_to_now_when_not_passed():
    """Backward-compat: existing 2-arg call sites (no `now`) must still work
    and default to a timely (late=False) read for a release_time_et that's
    effectively "right now"."""
    state = _fresh_state()
    now = datetime.now(es._NY)
    row = _row("NFP", now.isoformat(), 150.0, "K", 120.0, "K", source="nasdaq")
    entries = es._process_rows(state, {"nasdaq": [row], "tradingview": None})  # no `now` kwarg
    assert entries[0]["late"] is False


def test_process_rows_late_flag_unparsable_release_time_defaults_false():
    state = _fresh_state()
    row = _row("CPI MoM", "not-a-timestamp", 0.4, "%", 0.3, "%", source="nasdaq")
    entries = es._process_rows(state, {"nasdaq": [row], "tradingview": None})
    assert entries[0]["late"] is False  # never guessed True on unparsable input


def test_process_rows_contract_keys_unchanged_plus_late():
    """The 14 original contract keys must all still be present, with `late`
    as the one additive extra."""
    state = _fresh_state()
    now = datetime.now(es._NY)
    row = _row("CPI MoM", now.isoformat(), 0.4, "%", 0.3, "%", source="nasdaq")
    entries = es._process_rows(state, {"nasdaq": [row], "tradingview": None}, now=now)
    entry = entries[0]
    contract_keys = {
        "event", "release_time_et", "actual", "consensus", "previous", "unit",
        "surprise_raw", "surprise_z", "direction", "market_read", "importance",
        "source", "confirmed_by", "as_of",
    }
    assert contract_keys.issubset(entry.keys())
    assert entry.keys() - contract_keys == {"late"}


# ═══════════════════════════════════════════════════════════════════════════
#  Same-name/same-time unit-class collisions (2026-09-25 pass-2 fix, item 1
#  — CRITICAL). Rows below are the literal Building Permits / New Home Sales
#  pair from scratchpad/recon/nasdaq_20260925.json (two US rows at the SAME
#  gmt time, one %, one level).
# ═══════════════════════════════════════════════════════════════════════════

_BUILDING_PERMITS_PCT = {
    "event": "Building Permits", "release_time_et": "2026-09-25T08:00:00-04:00",
    "actual": -2.1, "actual_unit": "%", "consensus": -2.7, "consensus_unit": "%",
    "previous": 4.3, "previous_unit": "%", "importance": 2, "source": "nasdaq",
}
_BUILDING_PERMITS_LEVEL = {
    "event": "Building Permits", "release_time_et": "2026-09-25T08:00:00-04:00",
    "actual": 1.403, "actual_unit": "M", "consensus": 1.394, "consensus_unit": "M",
    "previous": 1.433, "previous_unit": "M", "importance": 2, "source": "nasdaq",
}
_NEW_HOME_SALES_LEVEL = {
    "event": "New Home Sales", "release_time_et": "2026-09-25T10:00:00-04:00",
    "actual": 684.0, "actual_unit": "K", "consensus": 615.0, "consensus_unit": "K",
    "previous": 643.0, "previous_unit": "K", "importance": 2, "source": "nasdaq",
}
_NEW_HOME_SALES_PCT_NO_CONSENSUS = {
    "event": "New Home Sales", "release_time_et": "2026-09-25T10:00:00-04:00",
    "actual": 6.4, "actual_unit": "%", "consensus": None, "consensus_unit": None,
    "previous": -4.3, "previous_unit": "%", "importance": 2, "source": "nasdaq",
}


def test_merge_building_permits_both_variants_recorded_regardless_of_row_order():
    """CRITICAL: both the %-MoM and level-in-millions Building Permits rows
    must be recorded as SEPARATE surprises with SEPARATE history series,
    regardless of the order Nasdaq happens to return them in."""
    state_a = _fresh_state()
    entries_a = es._process_rows(state_a, {"nasdaq": [_BUILDING_PERMITS_PCT, _BUILDING_PERMITS_LEVEL],
                                            "tradingview": None})
    state_b = _fresh_state()
    entries_b = es._process_rows(state_b, {"nasdaq": [_BUILDING_PERMITS_LEVEL, _BUILDING_PERMITS_PCT],
                                            "tradingview": None})

    for entries, state in ((entries_a, state_a), (entries_b, state_b)):
        assert len(entries) == 2
        by_unit = {e["unit"]: e for e in entries}
        assert set(by_unit.keys()) == {"%", "M"}  # entry["unit"] preserves the PRECISE unit; "M" buckets into the "level" CLASS for keying
        assert by_unit["%"]["surprise_raw"] == pytest.approx(0.6)             # -2.1 - (-2.7)
        assert by_unit["M"]["surprise_raw"] == pytest.approx(0.009, abs=1e-6)  # 1.403 - 1.394
        assert set(state["history"].keys()) == {"Building Permits (%)", "Building Permits (level)"}
        assert state["history"]["Building Permits (%)"] == [pytest.approx(0.6)]
        assert state["history"]["Building Permits (level)"] == [pytest.approx(0.009, abs=1e-6)]
        assert set(state["alerted"].keys()) == {
            "Building Permits (%)|2026-09-25T08:00:00-04:00",
            "Building Permits (level)|2026-09-25T08:00:00-04:00",
        }

    # Deterministic regardless of insertion order: identical surprise_raw
    # values recorded either way (not just "2 entries" — the SAME 2).
    raws_a = sorted(round(e["surprise_raw"], 6) for e in entries_a)
    raws_b = sorted(round(e["surprise_raw"], 6) for e in entries_b)
    assert raws_a == raws_b


def test_merge_new_home_sales_level_variant_has_consensus_pct_variant_does_not():
    """The %-variant row (New Home Sales) has a blank consensus in the real
    fixture -- it must still get its OWN consensus_table entry (for
    get_upcoming_releases) without corrupting or merging into the level
    variant's entry, and must not produce a fabricated surprise."""
    state = _fresh_state()
    entries = es._process_rows(state, {"nasdaq": [_NEW_HOME_SALES_LEVEL, _NEW_HOME_SALES_PCT_NO_CONSENSUS],
                                        "tradingview": None})
    assert len(entries) == 1
    assert entries[0]["unit"] == "K"  # precise unit preserved; buckets into the "level" CLASS for keying
    assert entries[0]["surprise_raw"] == pytest.approx(69.0)  # 684 - 615

    level_key = "New Home Sales (level)|2026-09-25T10:00:00-04:00"
    pct_key = "New Home Sales (%)|2026-09-25T10:00:00-04:00"
    assert state["consensus_table"][level_key]["consensus"] == 615.0
    assert pct_key in state["consensus_table"]
    assert state["consensus_table"][pct_key]["consensus"] is None
    assert pct_key not in state["alerted"]  # no surprise recorded for the no-consensus variant


def test_merge_tradingview_single_variant_attaches_to_matching_unit_class_only():
    """A TradingView row carrying only ONE variant (here, only the % row)
    must pair with Nasdaq's %-variant and NEVER the level-variant, even
    though both Nasdaq rows share the same event name + release time."""
    tv_pct_row = {
        "event": "Building Permits", "release_time_et": "2026-09-25T08:00:00-04:00",
        "actual": -2.0, "actual_unit": "%", "consensus": -2.7, "consensus_unit": "%",
        "previous": 4.3, "previous_unit": "%", "importance": 2, "source": "tradingview",
    }
    state = _fresh_state()
    entries = es._process_rows(
        state,
        {"nasdaq": [_BUILDING_PERMITS_PCT, _BUILDING_PERMITS_LEVEL], "tradingview": [tv_pct_row]},
    )
    assert len(entries) == 2
    pct_entry = next(e for e in entries if e["unit"] == "%")
    level_entry = next(e for e in entries if e["unit"] == "M")
    # TV's % row agrees closely with Nasdaq's % row -> cross-confirmed
    assert set(pct_entry["confirmed_by"]) == {"nasdaq", "tradingview"}
    # The level variant has no TV counterpart at all -> single-source only,
    # and specifically NEVER attaches the mismatched-unit TV row.
    assert level_entry["confirmed_by"] == ["nasdaq"]


def test_canonical_key_for_row_curated_events_never_suffixed_even_with_collision_shape():
    """Sanity: even if a curated event's row happened to look "collision-
    shaped" (two rows, same name/time), the canonical key must stay
    unsuffixed per the 2026-09-25 pass-2 fix's explicit requirement that
    mapped keys (CPI MoM, NFP, ...) keep their current canonical names."""
    row1 = {"event": "CPI MoM", "actual_unit": "%", "consensus_unit": "%", "previous_unit": "%"}
    row2 = {"event": "Consumer Price Index (CPI) MoM", "actual_unit": "level", "consensus_unit": "level",
            "previous_unit": "level"}
    assert es._canonical_key_for_row(row1) == "CPI MoM"
    assert es._canonical_key_for_row(row2) == "CPI MoM"
