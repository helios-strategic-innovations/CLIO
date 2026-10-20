"""Tests for agent/integrations/econ_surprise.py — value parsing, event-name
canonicalization/category classification, and raw-row normalization for both
sources.

Row fixtures below are literal excerpts transcribed from the CLIO intel v3
recon dumps (scratchpad/recon/nasdaq_20260925.json,
scratchpad/recon/tv_20260925.json, scratchpad/recon/tv_nfp_20260904.json) so
this test module stays hermetic (no filesystem dependency on the scratchpad,
no network).
"""

from datetime import date, datetime

import pytest

import agent.integrations.econ_surprise as es


# ═══════════════════════════════════════════════════════════════════════════
#  parse_value
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("197K", (197.0, "K")),
        ("-0.3%", (-0.3, "%")),
        ("4.1%", (4.1, "%")),
        ("1.2M", (1.2, "M")),
        ("$-78.3B", (-78.3, "B")),
        ("", (None, None)),
        (" ", (None, None)),          # Nasdaq's blank-consensus marker
        ("&nbsp;", (None, None)),     # Nasdaq's blank-actual marker (e.g. Fed speaker rows)
        ("1,719K", (1719.0, "K")),    # comma thousands
        ("6,748B", (6748.0, "B")),
        ("2.969T", (2.969, "T")),
        ("3.850%", (3.85, "%")),
        ("14", (14.0, "level")),      # no suffix -> "level"
        ("-", (None, None)),
        ("N/A", (None, None)),
        (None, (None, None)),
    ],
)
def test_parse_value_strings(raw, expected):
    assert es.parse_value(raw) == expected


def test_parse_value_numeric_passthrough():
    assert es.parse_value(162) == (162.0, "level")
    assert es.parse_value(4.1) == (4.1, "level")
    assert es.parse_value(0) == (0.0, "level")  # real zero must survive, not collapse to None


def test_parse_value_never_returns_zero_for_unparseable():
    """Missing is never zero (CLAUDE.md invariant) — every unparseable input
    must come back as (None, None), never (0.0, ...)."""
    for bad in ("", " ", "&nbsp;", None, "abc%", "K", "-", "+", "."):
        val, unit = es.parse_value(bad)
        assert val is None
        assert val != 0


def test_parse_value_bool_guarded():
    # bool is a subclass of int in Python; must not silently become 1.0/0.0
    assert es.parse_value(True) == (None, None)
    assert es.parse_value(False) == (None, None)


# ═══════════════════════════════════════════════════════════════════════════
#  Canonical key + category classification
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize(
    "name,expected_key,expected_category",
    [
        ("Core CPI MoM", "Core CPI MoM", "inflation"),
        ("CPI YoY", "CPI YoY", "inflation"),
        ("Consumer Price Index (CPI) MoM", "CPI MoM", "inflation"),
        ("Core PCE Price Index MoM", "Core PCE MoM", "inflation"),
        ("PCE Price Index MoM", "PCE MoM", "inflation"),
        ("Producer Price Index (PPI) MoM", "PPI MoM", "inflation"),
        ("Non Farm Payrolls", "NFP", "growth"),
        ("Nonfarm Payrolls", "NFP", "growth"),
        ("Non-Farm Payrolls", "NFP", "growth"),
        ("Non Farm Payrolls Private", "NFP Private", "growth"),
        ("Unemployment Rate", "Unemployment Rate", "inverted"),
        ("Initial Jobless Claims", "Initial Jobless Claims", "inverted"),
        ("Continuing Jobless Claims", "Continuing Jobless Claims", "inverted"),
        ("GDP (QoQ)", "GDP QoQ", "growth"),
        ("Retail Sales MoM", "Retail Sales MoM", "growth"),
        ("ISM Manufacturing PMI", "ISM Manufacturing PMI", "growth"),
        ("ISM Services PMI", "ISM Services PMI", "growth"),
        ("JOLTS Job Openings", "JOLTS", "growth"),
        ("Michigan Consumer Sentiment Final", "UMich Sentiment", "growth"),
        ("FOMC Rate Decision", "Fed Funds Rate decision", "fed"),
        ("Fed Funds Rate", "Fed Funds Rate decision", "fed"),
    ],
)
def test_canonical_key_and_category(name, expected_key, expected_category):
    assert es._canonical_key(name) == expected_key
    assert es._event_category(name) == expected_category


def test_unmatched_event_falls_back_to_raw_name_with_no_category():
    name = "Building Permits"
    assert es._canonical_key(name) == "Building Permits"
    assert es._event_category(name) is None

    name2 = "  KC   Fed Composite Index  "
    assert es._canonical_key(name2) == "KC Fed Composite Index"  # whitespace-normalized
    assert es._event_category(name2) is None


def test_canonical_key_empty_or_bad_input():
    assert es._canonical_key("") == "Unknown"
    assert es._canonical_key(None) == "Unknown"
    assert es._event_category("") is None
    assert es._event_category(None) is None


# ═══════════════════════════════════════════════════════════════════════════
#  Nasdaq row normalization (recon: nasdaq_20260925.json, US rows)
# ═══════════════════════════════════════════════════════════════════════════

_NASDAQ_CLAIMS_ROW = {
    "gmt": "08:30", "country": "United States", "eventName": "Initial Jobless Claims",
    "actual": "197K", "consensus": "201K", "previous": "198K",
}
_NASDAQ_SPEAKER_ROW = {
    "gmt": "04:10", "country": "United States", "eventName": "FOMC Member Williams Speaks",
    "actual": "&nbsp;", "consensus": " ", "previous": "&nbsp;",
}
_NASDAQ_BALANCE_SHEET_ROW = {
    "gmt": "16:30", "country": "United States", "eventName": "Fed's Balance Sheet",
    "actual": "6,748B", "consensus": " ", "previous": "6,747B",
}
_NASDAQ_ALLDAY_ROW = {
    "gmt": "24H", "country": "United Kingdom", "eventName": "Car Registration",
    "actual": "-39.8%", "consensus": " ", "previous": "32.7%",
}


def test_nasdaq_row_to_common_claims():
    ev = es._nasdaq_row_to_common(_NASDAQ_CLAIMS_ROW, date(2026, 9, 25))
    assert ev is not None
    assert ev["event"] == "Initial Jobless Claims"
    assert ev["actual"] == 197.0 and ev["actual_unit"] == "K"
    assert ev["consensus"] == 201.0 and ev["consensus_unit"] == "K"
    assert ev["previous"] == 198.0 and ev["previous_unit"] == "K"
    assert ev["source"] == "nasdaq"
    assert ev["importance"] == 2  # medium (jobless claims) per econ_calendar's classifier
    dt = datetime.fromisoformat(ev["release_time_et"])
    assert dt.year == 2026 and dt.month == 9 and dt.day == 25
    assert dt.hour == 8 and dt.minute == 30
    assert dt.tzinfo is not None


def test_nasdaq_row_blank_actual_and_consensus_are_none_not_zero():
    ev = es._nasdaq_row_to_common(_NASDAQ_SPEAKER_ROW, date(2026, 9, 25))
    assert ev is not None
    assert ev["actual"] is None
    assert ev["consensus"] is None
    assert ev["previous"] is None


def test_nasdaq_row_comma_thousands_billions():
    ev = es._nasdaq_row_to_common(_NASDAQ_BALANCE_SHEET_ROW, date(2026, 9, 25))
    assert ev["actual"] == 6748.0
    assert ev["actual_unit"] == "B"
    assert ev["consensus"] is None  # blank " " consensus


def test_nasdaq_row_allday_gmt_returns_none():
    """gmt='24H' is not a clock time -- row must be dropped, not mis-timed."""
    assert es._nasdaq_row_to_common(_NASDAQ_ALLDAY_ROW, date(2026, 9, 25)) is None


def test_nasdaq_row_missing_event_name_returns_none():
    row = dict(_NASDAQ_CLAIMS_ROW)
    row["eventName"] = ""
    assert es._nasdaq_row_to_common(row, date(2026, 9, 25)) is None


# ═══════════════════════════════════════════════════════════════════════════
#  TradingView row normalization (recon: tv_20260925.json, tv_nfp_20260904.json)
# ═══════════════════════════════════════════════════════════════════════════

_TV_NFP_ROW = {
    "title": "Non Farm Payrolls", "country": "US", "indicator": "Non Farm Payrolls",
    "category": "lbr", "actual": 162, "forecast": 56, "previous": 21, "unit": None,
    "importance": 1, "date": "2026-09-04T12:30:00.000Z",
}
_TV_UNRATE_ROW = {
    "title": "Unemployment Rate", "country": "US", "indicator": "Unemployment Rate",
    "category": "lbr", "actual": 4.1, "forecast": 4.1, "previous": 4.1, "unit": "%",
    "importance": 1, "date": "2026-09-04T12:30:00.000Z",
}
_TV_DURABLES_ROW = {
    "title": "Durable Goods Orders MoM", "country": "US", "indicator": "Durable Goods Orders",
    "category": "gov", "actual": None, "forecast": -0.4, "previous": 1.1, "unit": "%",
    "importance": 1, "date": "2026-09-25T12:30:00.000Z",
}
_TV_UNGA_ROW = {
    "title": "UN General Assembly", "country": "US", "indicator": "Calendar",
    "category": "gov", "actual": None, "forecast": None, "previous": None, "unit": None,
    "importance": 0, "date": "2026-09-25T00:00:00.000Z",
}
_TV_GOVT_PAYROLLS_ROW = {
    "title": "Government Payrolls", "country": "US", "indicator": "Government Payrolls",
    "category": "lbr", "actual": 35, "forecast": None, "previous": -50, "unit": None,
    "importance": -1, "date": "2026-09-04T12:30:00.000Z",
}


def test_tv_row_nfp_implied_thousands_unit():
    """TV's `unit` field is None for NFP but the value (162) means 162K --
    the payrolls-family override must supply unit='K', matching Nasdaq's
    parallel "162K" string representation for the same canonical series."""
    ev = es._tv_row_to_common(_TV_NFP_ROW)
    assert ev is not None
    assert ev["event"] == "Non Farm Payrolls"
    assert ev["actual"] == 162.0 and ev["actual_unit"] == "K"
    assert ev["consensus"] == 56.0 and ev["consensus_unit"] == "K"
    assert ev["previous"] == 21.0 and ev["previous_unit"] == "K"
    assert ev["source"] == "tradingview"
    assert ev["importance"] == 3  # NFP is HIGH per econ_calendar's classifier keywords
    dt = datetime.fromisoformat(ev["release_time_et"])
    # 2026-09-04T12:30:00Z == 08:30 ET (EDT, UTC-4, early September)
    assert dt.hour == 8 and dt.minute == 30


def test_tv_row_percent_unit_respected():
    ev = es._tv_row_to_common(_TV_UNRATE_ROW)
    assert ev["actual"] == 4.1 and ev["actual_unit"] == "%"
    assert ev["consensus"] == 4.1 and ev["consensus_unit"] == "%"
    assert ev["importance"] == 3  # unemployment rate is HIGH


def test_tv_row_null_actual_is_none_not_zero():
    ev = es._tv_row_to_common(_TV_DURABLES_ROW)
    assert ev["actual"] is None
    assert ev["consensus"] == -0.4 and ev["consensus_unit"] == "%"
    assert ev["importance"] == 2  # "durable goods" is a MEDIUM keyword in econ_calendar


def test_tv_row_all_null_still_normalizes_with_none_values():
    ev = es._tv_row_to_common(_TV_UNGA_ROW)
    assert ev is not None
    assert ev["actual"] is None and ev["consensus"] is None and ev["previous"] is None
    assert ev["importance"] == 1  # unclassified, TV importance=0 -> no bump


def test_tv_row_non_nfp_payroll_component_no_thousands_override_needed():
    """"Government Payrolls" also contains "payroll" so gets the same K
    override -- verifies the substring match doesn't accidentally require an
    exact "Non Farm Payrolls" title."""
    ev = es._tv_row_to_common(_TV_GOVT_PAYROLLS_ROW)
    assert ev["actual"] == 35.0 and ev["actual_unit"] == "K"
    assert ev["previous"] == -50.0 and ev["previous_unit"] == "K"


def test_tv_row_missing_title_returns_none():
    row = dict(_TV_UNGA_ROW)
    row["title"] = ""
    row["indicator"] = ""
    assert es._tv_row_to_common(row) is None


def test_tv_row_bad_date_returns_none():
    row = dict(_TV_UNRATE_ROW)
    row["date"] = "not-a-date"
    assert es._tv_row_to_common(row) is None


def test_tv_importance_bump_only_raises_never_lowers():
    """TV importance is an auxiliary signal (per spec: "importance via
    econ_calendar._classify_importance or TradingView importance") -- it can
    bump an unclassified/low event up to medium, but must never override an
    already-HIGH classification derived from the event name."""
    row = dict(_TV_UNGA_ROW)
    row["title"] = "Some Obscure Indicator"
    row["indicator"] = "Some Obscure Indicator"
    row["importance"] = 1
    ev = es._tv_row_to_common(row)
    assert ev["importance"] == 2  # bumped from 1 -> 2

    row2 = dict(_TV_NFP_ROW)
    row2["importance"] = -1  # TV itself thinks it's low importance
    ev2 = es._tv_row_to_common(row2)
    assert ev2["importance"] == 3  # name-based HIGH classification wins regardless


# ═══════════════════════════════════════════════════════════════════════════
#  _row_unit_class / _canonical_key_for_row (2026-09-25 pass-2 fix, item 1 —
#  CRITICAL same-name/same-time collision disambiguation). Rows below are the
#  literal Building Permits / New Home Sales pair from
#  scratchpad/recon/nasdaq_20260925.json.
# ═══════════════════════════════════════════════════════════════════════════

_NASDAQ_BUILDING_PERMITS_PCT_ROW = {
    "gmt": "08:00", "country": "United States", "eventName": "Building Permits",
    "actual": "-2.1%", "consensus": "-2.7%", "previous": "4.3%",
}
_NASDAQ_BUILDING_PERMITS_LEVEL_ROW = {
    "gmt": "08:00", "country": "United States", "eventName": "Building Permits",
    "actual": "1.403M", "consensus": "1.394M", "previous": "1.433M",
}
_NASDAQ_NEW_HOME_SALES_LEVEL_ROW = {
    "gmt": "10:00", "country": "United States", "eventName": "New Home Sales",
    "actual": "684K", "consensus": "615K", "previous": "643K",
}
_NASDAQ_NEW_HOME_SALES_PCT_ROW = {
    "gmt": "10:00", "country": "United States", "eventName": "New Home Sales",
    "actual": "6.4%", "consensus": " ", "previous": "-4.3%",
}


def test_row_unit_class_percent_vs_level_from_real_fixture_rows():
    pct_row = es._nasdaq_row_to_common(_NASDAQ_BUILDING_PERMITS_PCT_ROW, date(2026, 9, 25))
    level_row = es._nasdaq_row_to_common(_NASDAQ_BUILDING_PERMITS_LEVEL_ROW, date(2026, 9, 25))
    assert es._row_unit_class(pct_row) == "%"
    assert es._row_unit_class(level_row) == "level"  # "M" buckets into "level"

    k_row = es._nasdaq_row_to_common(_NASDAQ_NEW_HOME_SALES_LEVEL_ROW, date(2026, 9, 25))
    assert es._row_unit_class(k_row) == "level"  # "K" also buckets into "level"


def test_row_unit_class_prefers_actual_then_consensus_then_previous():
    assert es._row_unit_class({"actual_unit": "%", "consensus_unit": "K", "previous_unit": "M"}) == "%"
    assert es._row_unit_class({"actual_unit": None, "consensus_unit": "K", "previous_unit": "%"}) == "level"
    assert es._row_unit_class({"actual_unit": None, "consensus_unit": None, "previous_unit": "%"}) == "%"


def test_row_unit_class_fully_blank_row_defaults_level():
    assert es._row_unit_class({"actual_unit": None, "consensus_unit": None, "previous_unit": None}) == "level"
    assert es._row_unit_class({}) == "level"


def test_canonical_key_for_row_disambiguates_fallback_events():
    """Both Building Permits variants must get DISTINCT keys -- this is the
    core of the CRITICAL fix (item 1)."""
    pct_row = es._nasdaq_row_to_common(_NASDAQ_BUILDING_PERMITS_PCT_ROW, date(2026, 9, 25))
    level_row = es._nasdaq_row_to_common(_NASDAQ_BUILDING_PERMITS_LEVEL_ROW, date(2026, 9, 25))
    pct_key = es._canonical_key_for_row(pct_row)
    level_key = es._canonical_key_for_row(level_row)
    assert pct_key == "Building Permits (%)"
    assert level_key == "Building Permits (level)"
    assert pct_key != level_key

    nhs_level_row = es._nasdaq_row_to_common(_NASDAQ_NEW_HOME_SALES_LEVEL_ROW, date(2026, 9, 25))
    nhs_pct_row = es._nasdaq_row_to_common(_NASDAQ_NEW_HOME_SALES_PCT_ROW, date(2026, 9, 25))
    assert es._canonical_key_for_row(nhs_level_row) == "New Home Sales (level)"
    assert es._canonical_key_for_row(nhs_pct_row) == "New Home Sales (%)"


def test_canonical_key_for_row_keeps_curated_mapped_names_unsuffixed():
    """Existing mapped keys (CPI MoM, NFP, ...) must keep their current
    canonical names -- never suffixed, per the fix's explicit contract."""
    cpi_row = {"event": "Consumer Price Index (CPI) MoM", "actual_unit": "%",
               "consensus_unit": "%", "previous_unit": "%"}
    assert es._canonical_key_for_row(cpi_row) == "CPI MoM"

    nfp_row = {"event": "Non Farm Payrolls", "actual_unit": "K",
               "consensus_unit": "K", "previous_unit": "K"}
    assert es._canonical_key_for_row(nfp_row) == "NFP"

    claims_row = {"event": "Initial Jobless Claims", "actual_unit": "K",
                  "consensus_unit": "K", "previous_unit": "K"}
    assert es._canonical_key_for_row(claims_row) == "Initial Jobless Claims"
