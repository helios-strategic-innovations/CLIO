"""Wave 2A: flag-gated hardening of the DPOF UW pillars (premium/darkpool/gex).

Plugin-free (asyncio.run). Only agent.integrations.uw_flow_intel._uw_get is
faked; every compute helper runs for real. No network, no live files
(data_dir is tmp_path via the conftest cfg fixture).
"""

import asyncio
import logging
from dataclasses import replace

import pytest

import agent.dpof.data as data
import agent.integrations.uw_flow_intel as uw_mod
from agent.dpof.config import DPOFConfig, load_dpof_config
from agent.dpof.models import PillarReading, UnderlyingSnapshot
from agent.dpof.signals import compute_composite


class _StubBroker:
    async def get_minute_bars(self, symbol, start_utc):
        return None


@pytest.fixture(autouse=True)
def _clear_drift_dedup():
    data._schema_drift_last_warn.clear()
    yield
    data._schema_drift_last_warn.clear()


@pytest.fixture
def uw(monkeypatch):
    """endpoint -> response (dict, or callable(params)). Unset endpoints return {"data": []}."""
    responses: dict = {}

    async def _fake_uw_get(endpoint, params=None, *, skip_moltbot_budget=False):
        resp = responses.get(endpoint, {"data": []})
        return resp(params) if callable(resp) else resp

    monkeypatch.setattr(uw_mod, "_uw_get", _fake_uw_get)
    return responses


def _cfg(tmp_path, **kw) -> DPOFConfig:
    base = replace(DPOFConfig(), data_dir=str(tmp_path / "dpof"), min_pillars=2,
                   uw_refresh_seconds=100, underlyings=("SPY", "QQQ"))
    return replace(base, **kw)


def _feed(cfg):
    return data.DPOFDataFeed(cfg, _StubBroker())


def _run(coro):
    return asyncio.run(coro)


def _premium_row(nc, npr):
    return {"net_call_premium": nc, "net_put_premium": npr, "tape_time": "2026-10-09T14:00:00Z",
            "net_delta": "0.00", "volume": 1000}


def _buy_print(price="101.00", size=100):
    return {"price": price, "size": size, "nbbo_bid": "99.00", "nbbo_ask": "100.00", "canceled": False}


def _gex_strikes_accel():
    return [{"strike": 580, "call_gamma_oi": 10, "put_gamma_oi": -50}]  # net_gex < 0 -> ACCELERATING


def _gex_strikes_stab():
    return [{"strike": 480, "call_gamma_oi": 100, "put_gamma_oi": -30}]  # net_gex > 0 -> STABILIZING


# ── H1: config knobs ───────────────────────────────────────────────────────

def test_new_flags_default_false_on_dataclass_and_loader(monkeypatch):
    """Both new knobs default False on the dataclass and in load_dpof_config() with env unset; "1" enables them."""
    monkeypatch.delenv("CLIO_DPOF_DARKPOOL_NBBO", raising=False)
    monkeypatch.delenv("CLIO_DPOF_EMPTY_PILLAR_UNAVAILABLE", raising=False)
    assert DPOFConfig().darkpool_nbbo is False
    assert DPOFConfig().empty_pillar_unavailable is False
    c = load_dpof_config()
    assert c.darkpool_nbbo is False and c.empty_pillar_unavailable is False
    monkeypatch.setenv("CLIO_DPOF_DARKPOOL_NBBO", "1")
    monkeypatch.setenv("CLIO_DPOF_EMPTY_PILLAR_UNAVAILABLE", "1")
    c2 = load_dpof_config()
    assert c2.darkpool_nbbo is True and c2.empty_pillar_unavailable is True


# ── H2-H3: premium empty handling ──────────────────────────────────────────

def test_premium_empty_flag_off_keeps_legacy_zero_reading(uw, tmp_path):
    """Flag off: an empty UW tick list is still stored as available 0.0 with detail empty=True."""
    uw[data._EP_MARKET_TIDE] = {"data": []}
    feed = _feed(_cfg(tmp_path))
    _run(feed._refresh_premium(1000.0))
    r = feed._pillar_cache[("SPY", "premium")]
    assert r.available is True and r.value == 0.0 and r.detail.get("empty") is True


def test_premium_empty_flag_on_becomes_unavailable_like_failed_fetch(uw, tmp_path):
    """Flag on: empty ticks write no cache entry and record an 'empty:' last error."""
    uw[data._EP_MARKET_TIDE] = {"data": []}
    feed = _feed(_cfg(tmp_path, empty_pillar_unavailable=True))
    _run(feed._refresh_premium(1000.0))
    assert ("SPY", "premium") not in feed._pillar_cache
    assert ("SPY", "premium") not in feed._pillar_cache_time
    assert feed._pillar_last_error[("SPY", "premium")].startswith("empty:")


# ── H4: real-shape rows through the corrected compute ──────────────────────

def test_premium_real_shape_rows_score_one(uw, tmp_path):
    """20 real-shape rows (nc=1000, np=-200 each) score exactly +1.0 with the flag off."""
    uw[data._EP_MARKET_TIDE] = {"data": []}
    uw["/stock/SPY/net-prem-ticks"] = {"data": [_premium_row("1000.00", "-200.00") for _ in range(20)]}
    feed = _feed(_cfg(tmp_path))
    _run(feed._refresh_premium(1000.0))
    r = feed._pillar_cache[("SPY", "premium")]
    assert r.value == pytest.approx(1.0)
    assert r.available is True


# ── H5: schema canary ──────────────────────────────────────────────────────

def test_premium_schema_drift_flag_off_warns_once_names_fields_not_values(uw, tmp_path, caplog):
    """Flag off: drifted rows warn once (field names only, no values), reading stays legacy 0.0 available."""
    uw[data._EP_MARKET_TIDE] = {"data": []}
    uw["/stock/SPY/net-prem-ticks"] = {"data": [{"foo": 1, "tape_time": "x"}] * 3}
    feed = _feed(_cfg(tmp_path))
    caplog.set_level(logging.INFO, logger="clio.dpof.data")
    _run(feed._refresh_premium(1000.0))
    warns = [r for r in caplog.records if r.levelno == logging.WARNING and "premium schema drift for SPY" in r.getMessage()]
    assert len(warns) == 1
    msg = warns[0].getMessage()
    assert "foo" in msg and "tape_time" in msg
    assert "'x'" not in msg
    r = feed._pillar_cache[("SPY", "premium")]
    assert r.available is True and r.value == 0.0


def test_premium_schema_drift_warning_deduped_within_window(uw, tmp_path, caplog):
    """A second refresh inside the hour window logs no second drift warning."""
    uw[data._EP_MARKET_TIDE] = {"data": []}
    uw["/stock/SPY/net-prem-ticks"] = {"data": [{"foo": 1}] * 3}
    feed = _feed(_cfg(tmp_path))
    caplog.set_level(logging.INFO, logger="clio.dpof.data")
    _run(feed._refresh_premium(1000.0))
    _run(feed._refresh_premium(1100.0))
    warns = [r for r in caplog.records if "premium schema drift for SPY" in r.getMessage()]
    assert len(warns) == 1


def test_premium_schema_drift_flag_on_is_unavailable(uw, tmp_path):
    """Flag on: drifted rows are unavailable (no cache entry, 'empty:' error), like H3."""
    uw[data._EP_MARKET_TIDE] = {"data": []}
    uw["/stock/SPY/net-prem-ticks"] = {"data": [{"foo": 1, "tape_time": "x"}] * 3}
    feed = _feed(_cfg(tmp_path, empty_pillar_unavailable=True))
    _run(feed._refresh_premium(1000.0))
    assert ("SPY", "premium") not in feed._pillar_cache
    assert feed._pillar_last_error[("SPY", "premium")].startswith("empty:")


# ── H6-H7: darkpool NBBO flag ──────────────────────────────────────────────

def test_darkpool_flag_off_uses_legacy_and_logs_one_shadow_line(uw, tmp_path, caplog):
    """Flag off: value equals the legacy bias_score; exactly one INFO shadow line with legacy= and nbbo=."""
    prints = [{"price": "100.00", "size": 100, "nbbo_bid": "99.00", "nbbo_ask": "101.00"} for _ in range(5)]
    uw["/darkpool/SPY"] = {"data": prints}
    expected = uw_mod._compute_dp_signals([], {"SPY": prints})["tickers"]["SPY"]["bias_score"]
    feed = _feed(_cfg(tmp_path))
    caplog.set_level(logging.INFO, logger="clio.dpof.data")
    _run(feed._refresh_darkpool(1000.0))
    assert feed._pillar_cache[("SPY", "darkpool")].value == pytest.approx(max(-1.0, min(1.0, expected)))
    shadow = [r for r in caplog.records if "darkpool shadow" in r.getMessage()]
    assert len(shadow) == 1
    assert shadow[0].levelno == logging.INFO
    assert "legacy=" in shadow[0].getMessage() and "nbbo=" in shadow[0].getMessage()


def test_darkpool_flag_on_clear_buys_score_plus_one(uw, tmp_path):
    """Flag on with >=20 classified prints above the NBBO mid: value 1.0 from the #nbbo source."""
    uw["/darkpool/SPY"] = {"data": [_buy_print() for _ in range(20)]}
    feed = _feed(_cfg(tmp_path, darkpool_nbbo=True))
    _run(feed._refresh_darkpool(1000.0))
    r = feed._pillar_cache[("SPY", "darkpool")]
    assert r.value == pytest.approx(1.0)
    assert r.source.endswith("#nbbo")


def test_darkpool_flag_on_undersampled_is_legacy_empty_when_flag_unavailable_off(uw, tmp_path):
    """Flag on, 5 classified prints (<20): legacy empty 0.0 available reading, never an NBBO value."""
    uw["/darkpool/SPY"] = {"data": [_buy_print() for _ in range(5)]}
    feed = _feed(_cfg(tmp_path, darkpool_nbbo=True))
    _run(feed._refresh_darkpool(1000.0))
    r = feed._pillar_cache[("SPY", "darkpool")]
    assert r.available is True and r.value == 0.0 and r.detail.get("empty") is True
    assert not r.source.endswith("#nbbo")


def test_darkpool_flag_on_undersampled_is_unavailable_when_empty_flag_on(uw, tmp_path):
    """Flag on (both), 5 classified prints: unavailable, no cache entry."""
    uw["/darkpool/SPY"] = {"data": [_buy_print() for _ in range(5)]}
    feed = _feed(_cfg(tmp_path, darkpool_nbbo=True, empty_pillar_unavailable=True))
    _run(feed._refresh_darkpool(1000.0))
    assert ("SPY", "darkpool") not in feed._pillar_cache
    assert feed._pillar_last_error[("SPY", "darkpool")].startswith("empty:")


# ── H8: GEX regime transition log ──────────────────────────────────────────

def test_gex_regime_transitions_logged_only_on_change(uw, tmp_path, caplog):
    """None->ACCELERATING and ACCELERATING->STABILIZING are logged; a repeated STABILIZING is not."""
    feed = _feed(_cfg(tmp_path))
    caplog.set_level(logging.INFO, logger="clio.dpof.data")
    uw["/stock/SPY/spot-exposures/strike"] = {"data": _gex_strikes_accel()}
    _run(feed._refresh_gex(1000.0))
    uw["/stock/SPY/spot-exposures/strike"] = {"data": _gex_strikes_stab()}
    _run(feed._refresh_gex(1100.0))
    _run(feed._refresh_gex(1200.0))
    lines = [r.getMessage() for r in caplog.records if "gex regime SPY" in r.getMessage()]
    assert len(lines) == 2
    assert "None -> ACCELERATING" in lines[0]
    assert "ACCELERATING -> STABILIZING" in lines[1]


# ── H9: composite consequence of the empty-unavailable flag ────────────────

def _snap(premium_ok: bool, darkpool_ok: bool) -> UnderlyingSnapshot:
    pillars = {
        "nofa": PillarReading(name="nofa", value=-0.8, available=True),
        "gex": PillarReading(name="gex", value=-0.3, available=True),
    }
    if premium_ok:
        pillars["premium"] = PillarReading(name="premium", value=0.0, available=True)
    else:
        pillars["premium"] = PillarReading.missing("premium", "empty:empty_ticks")
    if darkpool_ok:
        pillars["darkpool"] = PillarReading(name="darkpool", value=0.0, available=True)
    else:
        pillars["darkpool"] = PillarReading.missing("darkpool", "empty:empty_prints")
    return UnderlyingSnapshot(symbol="SPY", pillars=pillars)


def test_composite_none_when_both_new_pillars_unavailable():
    """Documents the consequence: premium AND darkpool unavailable leaves 2 pillars < min_pillars=3 -> None."""
    comp, _used, reasons = compute_composite(_snap(False, False), DPOFConfig())
    assert comp is None
    assert "insufficient_pillars:2<3" in reasons


def test_composite_is_float_when_only_premium_unavailable():
    """Documents the consequence: with darkpool still available (0.0) the composite stays computable."""
    comp, _used, _reasons = compute_composite(_snap(False, True), DPOFConfig())
    assert isinstance(comp, float)


# ── Reviewer additions: gaps found by independent mutation testing ─────────

def test_darkpool_flag_off_never_uses_nbbo_value_even_with_sufficient_sample(uw, tmp_path, caplog):
    """Flag off, 25 clear buys (NBBO ratio +1.0): the pillar value stays the legacy bias_score; the shadow line shows nbbo=+1.0000."""
    prints = [_buy_print() for _ in range(25)]
    uw["/darkpool/SPY"] = {"data": prints}
    legacy = uw_mod._compute_dp_signals([], {"SPY": prints})["tickers"]["SPY"]["bias_score"]
    feed = _feed(_cfg(tmp_path))
    caplog.set_level(logging.INFO, logger="clio.dpof.data")
    _run(feed._refresh_darkpool(1000.0))
    r = feed._pillar_cache[("SPY", "darkpool")]
    assert r.value == pytest.approx(max(-1.0, min(1.0, legacy)))
    assert abs(r.value) < 0.5  # NOT the NBBO ratio (+1.0)
    assert not r.source.endswith("#nbbo")
    shadow = [m.getMessage() for m in caplog.records if "darkpool shadow" in m.getMessage()]
    assert len(shadow) == 1 and "nbbo=+1.0000" in shadow[0]


def test_darkpool_flag_on_emits_no_shadow_line(uw, tmp_path, caplog):
    """Flag on: the NBBO value is live, so no 'darkpool shadow' readout is emitted."""
    uw["/darkpool/SPY"] = {"data": [_buy_print() for _ in range(25)]}
    feed = _feed(_cfg(tmp_path, darkpool_nbbo=True))
    caplog.set_level(logging.INFO, logger="clio.dpof.data")
    _run(feed._refresh_darkpool(1000.0))
    assert not [r for r in caplog.records if "darkpool shadow" in r.getMessage()]


def test_darkpool_empty_prints_legacy_zero_by_default_unavailable_with_empty_flag(uw, tmp_path):
    """Empty print list: legacy available 0.0 by default; unavailable with reason 'empty:empty_prints' when the empty flag is on."""
    uw["/darkpool/SPY"] = {"data": []}
    off = _feed(_cfg(tmp_path))
    _run(off._refresh_darkpool(1000.0))
    r = off._pillar_cache[("SPY", "darkpool")]
    assert r.available is True and r.value == 0.0 and r.detail.get("empty") is True
    on = _feed(_cfg(tmp_path, empty_pillar_unavailable=True))
    _run(on._refresh_darkpool(1000.0))
    assert ("SPY", "darkpool") not in on._pillar_cache
    assert on._pillar_last_error[("SPY", "darkpool")] == "empty:empty_prints"


def test_premium_genuinely_empty_ticks_are_not_schema_drift(uw, tmp_path, caplog):
    """An empty tick list is 'empty', not 'drift': no WARNING, and the flag-on reason is exactly 'empty:empty_ticks'."""
    uw[data._EP_MARKET_TIDE] = {"data": []}
    uw["/stock/SPY/net-prem-ticks"] = {"data": []}
    feed = _feed(_cfg(tmp_path, empty_pillar_unavailable=True))
    caplog.set_level(logging.INFO, logger="clio.dpof.data")
    _run(feed._refresh_premium(1000.0))
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert feed._pillar_last_error[("SPY", "premium")] == "empty:empty_ticks"


@pytest.mark.parametrize("row", [
    {"net_put_premium": "100.00"},
    {"net_call_premium": "100.00"},
    {"net_premium": "5"},
    {"call_premium": "5"},
    {"put_premium": "5"},
])
def test_premium_any_single_recognized_field_is_not_drift(uw, tmp_path, caplog, row):
    """Rows carrying any ONE recognized premium field (new or legacy name) never trigger the drift canary."""
    uw[data._EP_MARKET_TIDE] = {"data": []}
    uw["/stock/SPY/net-prem-ticks"] = {"data": [dict(row) for _ in range(3)]}
    feed = _feed(_cfg(tmp_path))
    caplog.set_level(logging.INFO, logger="clio.dpof.data")
    _run(feed._refresh_premium(1000.0))
    assert not [r for r in caplog.records if "schema drift" in r.getMessage()]


def test_premium_drift_decided_on_last_20_rows_only(uw, tmp_path, caplog):
    """Recognized rows older than the last 20 do not hide a drifted tail; a recognized tail is not drift."""
    uw[data._EP_MARKET_TIDE] = {"data": []}
    good = [_premium_row("1.00", "1.00") for _ in range(5)]
    bad = [{"foo": 1} for _ in range(20)]
    caplog.set_level(logging.INFO, logger="clio.dpof.data")
    uw["/stock/SPY/net-prem-ticks"] = {"data": good + bad}
    _run(_feed(_cfg(tmp_path))._refresh_premium(1000.0))
    assert len([r for r in caplog.records if "premium schema drift for SPY" in r.getMessage()]) == 1
    data._schema_drift_last_warn.clear()
    caplog.clear()
    uw["/stock/SPY/net-prem-ticks"] = {"data": bad[:5] + [_premium_row("1.00", "1.00") for _ in range(20)]}
    _run(_feed(_cfg(tmp_path))._refresh_premium(1000.0))
    assert not [r for r in caplog.records if "schema drift" in r.getMessage()]


def test_premium_drift_dedup_is_per_ticker(uw, tmp_path, caplog):
    """SPY and QQQ drifting in one refresh each log their own warning (the dedup key includes the ticker)."""
    uw[data._EP_MARKET_TIDE] = {"data": []}
    uw["/stock/SPY/net-prem-ticks"] = {"data": [{"foo": 1}] * 3}
    uw["/stock/QQQ/net-prem-ticks"] = {"data": [{"foo": 1}] * 3}
    caplog.set_level(logging.INFO, logger="clio.dpof.data")
    _run(_feed(_cfg(tmp_path))._refresh_premium(1000.0))
    warns = [r.getMessage() for r in caplog.records if "schema drift" in r.getMessage()]
    assert len(warns) == 2
    assert any("for SPY" in w for w in warns) and any("for QQQ" in w for w in warns)
