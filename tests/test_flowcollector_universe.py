"""Tests for agent/flowcollector/universe.py. Hermetic: no network, no SDK objects."""

from datetime import date

from agent.flowcollector.universe import (
    UniverseState,
    build_universe_state,
    dte,
    in_band,
    needs_universe_refresh,
    partition_symbols,
    select_universe,
    session_date_ahead,
    trading_dte,
)


def test_dte_basic():
    today = date(2026, 9, 25)
    assert dte(date(2026, 9, 25), today) == 0
    assert dte(date(2026, 9, 27), today) == 2
    assert dte(date(2026, 9, 24), today) == -1


def test_in_band():
    assert in_band(580.0, 580.0, 0.04) is True
    assert in_band(603.0, 580.0, 0.04) is True   # +3.97%
    assert in_band(604.0, 580.0, 0.04) is False  # +4.14%, just outside
    assert in_band(557.0, 580.0, 0.04) is True   # -3.97%
    assert in_band(556.0, 580.0, 0.04) is False
    assert in_band(580.0, None, 0.04) is False
    assert in_band(580.0, 0.0, 0.04) is False
    assert in_band(580.0, 580.0, -0.01) is False


def _contract(symbol, exp, strike, right="call"):
    return {"symbol": symbol, "expiration_date": exp, "strike_price": strike, "right": right}


def test_select_universe_filters_by_dte_and_band():
    today = date(2026, 9, 25)  # a Friday
    contracts = [
        _contract("A", "2026-09-25", 580.0),   # 0DTE, in band
        _contract("C", "2026-09-28", 590.0),   # Monday: 1 trading session ahead
        _contract("F", "2026-09-29", 585.0),   # Tuesday: 2 sessions
        _contract("G", "2026-09-30", 585.0),   # Wednesday: 3 sessions, out of range
        _contract("D", "2026-09-24", 580.0),   # already expired
        _contract("E", "2026-09-28", 700.0),   # far out of band
    ]
    out = select_universe(contracts, spot=580.0, band_pct=0.04, max_dte=2, today=today)
    symbols = {c["symbol"] for c in out}
    assert symbols == {"A", "C", "F"}


def test_trading_dte_counts_sessions_not_calendar_days():
    fri = date(2026, 9, 25)
    assert trading_dte(fri, fri) == 0
    assert trading_dte(date(2026, 9, 28), fri) == 1      # Fri -> Mon
    assert trading_dte(date(2026, 9, 29), fri) == 2
    assert trading_dte(date(2026, 9, 24), fri) == -1
    # a holiday is not a session: Thanksgiving Thu 2026-11-26
    wed = date(2026, 11, 25)
    assert trading_dte(date(2026, 11, 27), wed, frozenset({date(2026, 11, 26)})) == 1


def test_session_date_ahead():
    fri = date(2026, 9, 25)
    assert session_date_ahead(fri, 0) == fri
    assert session_date_ahead(fri, 1) == date(2026, 9, 28)
    assert session_date_ahead(fri, 2) == date(2026, 9, 29)
    assert session_date_ahead(date(2026, 11, 25), 1, frozenset({date(2026, 11, 26)})) == date(2026, 11, 27)


def test_friday_universe_is_not_0dte_only():
    """2026-09-25: calendar max_dte=2 on a Friday kept only 0DTE contracts
    (Alpaca gives no greeks for same-day expiries -> deltas_loaded 0)."""
    fri = date(2026, 9, 25)
    contracts = [_contract("TODAY", "2026-09-25", 580.0), _contract("MON", "2026-09-28", 580.0)]
    out = {c["symbol"] for c in select_universe(contracts, spot=580.0, band_pct=0.04, max_dte=2, today=fri)}
    assert out == {"TODAY", "MON"}


def test_select_universe_skips_malformed_entries():
    today = date(2026, 9, 25)
    contracts = [
        {"symbol": "OK", "expiration_date": "2026-09-25", "strike_price": 580.0},
        {"symbol": "NOEXP", "strike_price": 580.0},
        {"symbol": "BADEXP", "expiration_date": "not-a-date", "strike_price": 580.0},
        {"symbol": "NOSTRIKE", "expiration_date": "2026-09-25"},
        {"symbol": "BADSTRIKE", "expiration_date": "2026-09-25", "strike_price": "nan-ish"},
        "not-a-dict",
        None,
    ]
    out = select_universe(contracts, spot=580.0, band_pct=0.04, max_dte=2, today=today)
    assert [c["symbol"] for c in out] == ["OK"]


def test_select_universe_no_spot_returns_empty():
    today = date(2026, 9, 25)
    contracts = [_contract("A", "2026-09-25", 580.0)]
    assert select_universe(contracts, spot=None, band_pct=0.04, max_dte=2, today=today) == []
    assert select_universe(contracts, spot=0.0, band_pct=0.04, max_dte=2, today=today) == []


def test_needs_universe_refresh_never_refreshed_yet():
    assert needs_universe_refresh(None, 1000.0, 900.0, None, 580.0, 0.005) is True


def test_needs_universe_refresh_cadence_due():
    assert needs_universe_refresh(1000.0, 1000.0 + 900.0, 900.0, 580.0, 580.0, 0.005) is True
    assert needs_universe_refresh(1000.0, 1000.0 + 899.0, 900.0, 580.0, 580.0, 0.005) is False


def test_needs_universe_refresh_spot_drift():
    # 1% drift with a 0.5% trigger -> due even though cadence not up
    assert needs_universe_refresh(1000.0, 1005.0, 900.0, 580.0, 585.8, 0.005) is True
    # 0.1% drift -> not due
    assert needs_universe_refresh(1000.0, 1005.0, 900.0, 580.0, 580.5, 0.005) is False


def test_needs_universe_refresh_missing_spot_never_forces_by_drift_alone():
    assert needs_universe_refresh(1000.0, 1005.0, 900.0, None, 580.0, 0.005) is False
    assert needs_universe_refresh(1000.0, 1005.0, 900.0, 580.0, None, 0.005) is False


def test_partition_symbols_normal_and_edge_cases():
    syms = [f"S{i}" for i in range(10)]
    parts = partition_symbols(syms, 4)
    assert parts == [syms[0:4], syms[4:8], syms[8:10]]
    assert partition_symbols([], 4) == []
    assert partition_symbols(syms, 0) == [syms]
    assert partition_symbols(syms, -1) == [syms]


def test_universe_state_all_symbols_and_total_count():
    st = UniverseState(
        symbols_by_underlying={"SPY": ["A", "B"], "QQQ": ["C"]},
        contracts_by_symbol={"A": {}, "B": {}, "C": {}},
        built_at=1000.0,
        spot_by_underlying={"SPY": 580.0, "QQQ": 480.0},
    )
    assert sorted(st.all_symbols) == ["A", "B", "C"]
    assert st.total_count == 3


def test_build_universe_state_applies_filter_per_underlying():
    today = date(2026, 9, 25)
    per_underlying = {
        "SPY": [_contract("SA", "2026-09-25", 580.0), _contract("SB", "2026-09-30", 580.0)],
        "QQQ": [_contract("QA", "2026-09-26", 480.0)],
    }
    spots = {"SPY": 580.0, "QQQ": 480.0}
    state = build_universe_state(per_underlying, spots, band_pct=0.04, max_dte=2, today=today, now_epoch=1000.0)
    assert state.symbols_by_underlying["SPY"] == ["SA"]
    assert state.symbols_by_underlying["QQQ"] == ["QA"]
    assert state.built_at == 1000.0
    assert state.spot_by_underlying == spots
    assert "SA" in state.contracts_by_symbol and "SB" not in state.contracts_by_symbol
