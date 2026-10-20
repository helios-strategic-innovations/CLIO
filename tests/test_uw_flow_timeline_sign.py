"""Sign-convention tests for get_ticker_flow_timeline and its net-flow consumer.

Verified convention (2026-10-10): net_call_premium > 0 = calls bought (bullish);
net_put_premium > 0 = puts bought (bearish); net_flow = net_call - net_put.
No network: _uw_get is stubbed on the uw_flow_intel module.
"""

import asyncio

import pytest

import agent.integrations.uw_flow_intel as uw
from agent.integrations.unusual_whales import get_net_flow_timeline

DATE = "2026-10-09"


def _tick(tape_time, **fields):
    """Build a net-prem-ticks row with string numerics like the live API."""
    row = {"tape_time": tape_time, "call_volume": 10, "put_volume": 5,
           "call_volume_ask_side": 6, "put_volume_ask_side": 2}
    row.update(fields)
    return row


def _stub(monkeypatch, rows):
    """Replace _uw_get so the timeline reads the given rows."""
    async def fake_uw_get(endpoint, params=None, *, skip_moltbot_budget=False):
        return {"data": rows}
    monkeypatch.setattr(uw, "_uw_get", fake_uw_get)


def _timeline(monkeypatch, rows):
    """Run get_ticker_flow_timeline for SPY on DATE over the stubbed rows."""
    _stub(monkeypatch, rows)
    return asyncio.run(uw.get_ticker_flow_timeline("SPY", bucket_minutes=5, date=DATE))


def test_bullish_tick_net_flow_is_call_minus_put(monkeypatch):
    """Calls bought (nc>0) and puts sold (np<0) give net_flow = nc - np = 1200."""
    out = _timeline(monkeypatch, [_tick("2026-10-09T09:31:00.000000-04:00",
                                        net_call_premium="1000.00", net_put_premium="-200.00")])
    assert len(out) == 1
    b = out[0]
    assert b["time"] == "09:30"
    assert b["net_call"] == pytest.approx(1000.0)
    assert b["net_put"] == pytest.approx(-200.0)
    assert b["net_flow"] == pytest.approx(1200.0)


def test_bearish_tick_net_flow_is_negative(monkeypatch):
    """Calls sold (nc<0) and puts bought (np>0) give net_flow = -500 - 800 = -1300."""
    out = _timeline(monkeypatch, [_tick("2026-10-09T09:31:00.000000-04:00",
                                        net_call_premium="-500.00", net_put_premium="800.00")])
    b = out[0]
    assert b["net_call"] == pytest.approx(-500.0)
    assert b["net_put"] == pytest.approx(800.0)
    assert b["net_flow"] == pytest.approx(-1300.0)


def test_equal_call_and_put_buying_nets_to_zero(monkeypatch):
    """Equal call and put buying (100 and 100) nets to 0, not 200 (convention pin)."""
    out = _timeline(monkeypatch, [_tick("2026-10-09T09:31:00.000000-04:00",
                                        net_call_premium="100.00", net_put_premium="100.00")])
    assert out[0]["net_flow"] == pytest.approx(0.0)


def test_legacy_net_premium_positive_goes_to_calls(monkeypatch):
    """Legacy-only net_premium=300 gives net_call 300, net_put 0, net_flow 300."""
    out = _timeline(monkeypatch, [_tick("2026-10-09T09:31:00.000000-04:00", net_premium="300")])
    b = out[0]
    assert b["net_call"] == pytest.approx(300.0)
    assert b["net_put"] == pytest.approx(0.0)
    assert b["net_flow"] == pytest.approx(300.0)


def test_legacy_net_premium_negative_goes_to_puts_as_positive(monkeypatch):
    """Legacy-only net_premium=-300 gives net_call 0, net_put +300 (puts bought), net_flow -300."""
    out = _timeline(monkeypatch, [_tick("2026-10-09T09:31:00.000000-04:00", net_premium="-300")])
    b = out[0]
    assert b["net_call"] == pytest.approx(0.0)
    assert b["net_put"] == pytest.approx(300.0)
    assert b["net_flow"] == pytest.approx(-300.0)


def test_ticks_in_same_bucket_sum_and_next_bucket_is_separate(monkeypatch):
    """Two ticks in 09:30 sum (net_flow 125); a 09:36 ET tick gets its own 09:35 bucket, sorted."""
    rows = [
        _tick("2026-10-09T09:36:00.000000-04:00", net_call_premium="10.00", net_put_premium="0.00"),
        _tick("2026-10-09T09:31:00.000000-04:00", net_call_premium="100.00", net_put_premium="0.00"),
        _tick("2026-10-09T09:33:30.000000-04:00", net_call_premium="50.00", net_put_premium="25.00"),
    ]
    out = _timeline(monkeypatch, rows)
    assert [b["time"] for b in out] == ["09:30", "09:35"]
    assert out[0]["net_call"] == pytest.approx(150.0)
    assert out[0]["net_put"] == pytest.approx(25.0)
    assert out[0]["net_flow"] == pytest.approx(125.0)
    assert out[0]["ticks"] == 2
    assert out[1]["net_flow"] == pytest.approx(10.0)


def test_tick_from_other_et_date_is_skipped(monkeypatch):
    """A tick whose ET date differs from the requested date is ignored."""
    rows = [
        _tick("2026-10-09T09:31:00.000000-04:00", net_call_premium="1000.00", net_put_premium="-200.00"),
        _tick("2026-10-08T09:31:00.000000-04:00", net_call_premium="5000.00", net_put_premium="0.00"),
    ]
    out = _timeline(monkeypatch, rows)
    assert len(out) == 1
    assert out[0]["net_flow"] == pytest.approx(1200.0)


@pytest.mark.parametrize("payload", [{"data": []}, {"data": None}, {}])
def test_empty_or_missing_data_returns_empty_list(monkeypatch, payload):
    """Empty, None or missing data yields an empty list."""
    async def fake_uw_get(endpoint, params=None, *, skip_moltbot_budget=False):
        return payload
    monkeypatch.setattr(uw, "_uw_get", fake_uw_get)
    assert asyncio.run(uw.get_ticker_flow_timeline("SPY", 5, DATE)) == []


def test_consumer_bearish_day_reads_put_bias(monkeypatch):
    """Bearish day (nc=-500, np=800) is reported as PUT BIAS with a negative total, never CALL BIAS."""
    _stub(monkeypatch, [_tick("2026-10-09T09:31:00.000000-04:00",
                              net_call_premium="-500.00", net_put_premium="800.00")])
    text = asyncio.run(get_net_flow_timeline({"symbol": "SPY", "date": DATE}))
    assert "PUT BIAS" in text
    assert "CALL BIAS" not in text
    assert "Total: -$" in text


def test_consumer_bullish_day_reads_call_bias(monkeypatch):
    """Bullish day (nc=1000, np=-200) is reported as CALL BIAS with a positive total, never PUT BIAS."""
    _stub(monkeypatch, [_tick("2026-10-09T09:31:00.000000-04:00",
                              net_call_premium="1000.00", net_put_premium="-200.00")])
    text = asyncio.run(get_net_flow_timeline({"symbol": "SPY", "date": DATE}))
    assert "CALL BIAS" in text
    assert "PUT BIAS" not in text
    assert "Total: +$" in text


def test_utc_z_timestamps_are_bucketed_in_et(monkeypatch):
    """UW's real UTC 'Z' stamps bucket in ET: 13:31Z is 09:31 ET, bucket '09:30' (not UTC '13:30')."""
    out = _timeline(monkeypatch, [_tick("2026-10-09T13:31:00.000000Z",
                                        net_call_premium="1000.00", net_put_premium="-200.00")])
    assert [b["time"] for b in out] == ["09:30"]


def test_utc_z_timestamps_use_the_et_date_for_the_day_filter(monkeypatch):
    """21:30 ET on the 9th is 01:30Z on the 10th and must be kept; 23:00 ET on the 8th (03:00Z on the 9th) must be dropped."""
    rows = [
        _tick("2026-10-10T01:30:00.000000Z", net_call_premium="100.00", net_put_premium="0.00"),
        _tick("2026-10-09T03:00:00.000000Z", net_call_premium="999.00", net_put_premium="0.00"),
    ]
    out = _timeline(monkeypatch, rows)
    assert [b["time"] for b in out] == ["21:30"]
    assert out[0]["net_call"] == pytest.approx(100.0)


def test_naive_timestamps_are_treated_as_et(monkeypatch):
    """A tape_time with no offset is taken as ET and buckets unchanged."""
    out = _timeline(monkeypatch, [_tick("2026-10-09T09:31:00", net_call_premium="1.00", net_put_premium="0.00")])
    assert [b["time"] for b in out] == ["09:30"]


def test_consumer_prints_sign_legend_and_et_times(monkeypatch):
    """The consumer text states that Net Put > 0 means puts bought (bearish) and that times are ET."""
    _stub(monkeypatch, [_tick("2026-10-09T13:31:00.000000Z", net_call_premium="-500.00", net_put_premium="800.00")])
    text = asyncio.run(get_net_flow_timeline({"symbol": "SPY", "bucket_minutes": 5, "date": DATE}))
    assert "Net Put > 0 = puts net bought = bearish" in text
    assert "times ET" in text
    assert "09:30" in text and "13:30" not in text


def test_nan_inf_and_oversized_values_never_poison_a_bucket(monkeypatch):
    """'nan', 'inf', '-inf' and a 309-digit int count as 0, so net_call/net_put/net_flow stay finite and exact."""
    import math
    out = _timeline(monkeypatch, [
        _tick("2026-10-09T09:31:00.000000-04:00", net_call_premium="nan", net_put_premium="inf"),
        _tick("2026-10-09T09:32:00.000000-04:00", net_call_premium=10 ** 400, net_put_premium="-inf"),
        _tick("2026-10-09T09:33:00.000000-04:00", net_call_premium="1000.00", net_put_premium="-200.00"),
    ])
    b = out[0]
    assert all(math.isfinite(b[k]) for k in ("net_call", "net_put", "net_flow"))
    assert b["net_call"] == pytest.approx(1000.0) and b["net_put"] == pytest.approx(-200.0) and b["net_flow"] == pytest.approx(1200.0)
