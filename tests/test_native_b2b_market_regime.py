"""
Tests for agent/integrations/market_regime.py (CLIO-native market regime classifier —
replaces Hydrinia's frozen daily_metrics.json["market_regime"]).

No network: refresh_market_regime()'s two blocking/async fetchers
(_fetch_spy_daily_bars, _fetch_vix) are monkeypatched directly. All file I/O goes
through a tmp_path CLIO_MARKET_REGIME_FILE — never the real data/ tree.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from agent.integrations import market_regime as mr


@pytest.fixture(autouse=True)
def _isolated_file(tmp_path, monkeypatch):
    """Every test gets its own regime file and a clean read cache."""
    path = tmp_path / "market_regime.json"
    monkeypatch.setenv("CLIO_MARKET_REGIME_FILE", str(path))
    mr._cache.clear()
    return path


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


_NY = ZoneInfo("America/New_York")


def _today_et():
    return datetime.now(timezone.utc).astimezone(_NY).date()


def _dated(closes, last_day=None):
    """(session_date, close) pairs, oldest first, consecutive calendar days ending at last_day."""
    if last_day is None:
        last_day = _today_et()
    n = len(closes)
    return [(last_day - timedelta(days=n - 1 - i), closes[i]) for i in range(n)]


class _FakeBar:
    def __init__(self, day, close):
        self.timestamp = datetime(day.year, day.month, day.day, 4, 0, tzinfo=timezone.utc)
        self.close = close


class _FakeBarsClient:
    """Mimics Alpaca's bars endpoint: sort=desc returns newest-first, sort absent/asc returns
    oldest-first, and `limit` truncates the TAIL of the requested order (the R-15 mechanism)."""

    def __init__(self, dated_bars):
        self._bars = dated_bars  # oldest first
        self.requests = []

    def get_stock_bars(self, request):
        from alpaca.common.enums import Sort

        self.requests.append(request)
        rows = list(self._bars)
        if request.sort == Sort.DESC:
            rows = list(reversed(rows))[: request.limit]
        else:
            rows = rows[: request.limit]
        return {"SPY": [_FakeBar(d, c) for d, c in rows]}


@pytest.fixture
def fake_bars(monkeypatch):
    """Install a fake Alpaca data client over the given (date, close) bars (oldest first)."""

    def install(dated_bars):
        client = _FakeBarsClient(dated_bars)
        import agent.integrations.alpaca as alp

        monkeypatch.setattr(alp, "_get_data_client", lambda: client)
        return client

    return install


def _write(path, regime="BULL", confidence=1.0, age_seconds=0, **extra):
    payload = {
        "regime": regime,
        "confidence": confidence,
        "vix": extra.pop("vix", 14.0),
        "spy_close": extra.pop("spy_close", 500.0),
        "spy_sma20": extra.pop("spy_sma20", 495.0),
        "spy_sma50": extra.pop("spy_sma50", 490.0),
        "as_of": _iso(datetime.now(timezone.utc) - timedelta(seconds=age_seconds)),
        "source": "clio_native",
    }
    payload.update(extra)
    _write_atomic(path, payload)
    return payload


def _write_atomic(path, payload):
    """Same contract as the production writer (tmp + os.replace): every write is a new inode."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# classify_regime() — classification matrix
# ---------------------------------------------------------------------------


class TestClassifyMatrix:
    def test_crisis_vix_ge_35_overrides_bull_trend(self):
        regime, conf = mr.classify_regime(spy_close=510, spy_sma20=505, spy_sma50=490, vix=36.0)
        assert regime == "CRISIS"
        # trend buckets disagree with risk_off -> only VIX agrees -> 1/3
        assert conf == pytest.approx(round(1 / 3, 4))

    def test_crisis_vix_exactly_35_boundary(self):
        regime, _ = mr.classify_regime(spy_close=480, spy_sma20=470, spy_sma50=490, vix=35.0)
        assert regime == "CRISIS"

    def test_crisis_all_signals_agree_full_confidence(self):
        regime, conf = mr.classify_regime(spy_close=470, spy_sma20=460, spy_sma50=490, vix=40.0)
        assert regime == "CRISIS"
        assert conf == pytest.approx(1.0)

    def test_bear_vix_gt_30(self):
        regime, _ = mr.classify_regime(spy_close=510, spy_sma20=505, spy_sma50=490, vix=31.0)
        assert regime == "BEAR"

    def test_bear_vix_exactly_30_is_not_bear_by_vix_alone(self):
        # vix_gt_30 requires strictly > 30; at exactly 30 with a bull trend and vix>25
        # the CHOPPY branch (vix_gt_25) fires instead.
        regime, _ = mr.classify_regime(spy_close=510, spy_sma20=505, spy_sma50=490, vix=30.0)
        assert regime == "CHOPPY"

    def test_bear_trend_alone_no_vix(self):
        regime, conf = mr.classify_regime(spy_close=480, spy_sma20=485, spy_sma50=495, vix=None)
        assert regime == "BEAR"
        # both trend buckets agree (risk_off); vix abstains -> 2/3
        assert conf == pytest.approx(round(2 / 3, 4))

    def test_choppy_vix_between_25_and_30_with_bull_trend(self):
        regime, _ = mr.classify_regime(spy_close=510, spy_sma20=505, spy_sma50=490, vix=27.0)
        assert regime == "CHOPPY"

    def test_choppy_mixed_trend_no_vix(self):
        # close above sma50 but sma20 below sma50 -> mixed trend, no vix
        regime, _ = mr.classify_regime(spy_close=492, spy_sma20=488, spy_sma50=490, vix=None)
        assert regime == "CHOPPY"

    def test_choppy_equal_smas_is_mixed(self):
        regime, _ = mr.classify_regime(spy_close=490, spy_sma20=490, spy_sma50=490, vix=10.0)
        assert regime == "CHOPPY"

    def test_bull_clean_uptrend_low_vix(self):
        regime, conf = mr.classify_regime(spy_close=510, spy_sma20=505, spy_sma50=490, vix=14.0)
        assert regime == "BULL"
        assert conf == pytest.approx(1.0)

    def test_bull_uptrend_vix_missing_lowers_confidence(self):
        regime, conf = mr.classify_regime(spy_close=510, spy_sma20=505, spy_sma50=490, vix=None)
        assert regime == "BULL"
        # vix abstains (never "agrees") -> capped at 2/3 even though both trend reads agree
        assert conf == pytest.approx(round(2 / 3, 4))

    def test_bull_uptrend_vix_exactly_25_boundary(self):
        regime, _ = mr.classify_regime(spy_close=510, spy_sma20=505, spy_sma50=490, vix=25.0)
        assert regime == "BULL"


# ---------------------------------------------------------------------------
# get_market_regime() / get_market_regime_label() — staleness + malformed data
# ---------------------------------------------------------------------------


class TestReadPath:
    def test_missing_file_returns_none(self, _isolated_file):
        assert mr.get_market_regime() is None
        assert mr.get_market_regime_label() == "CHOPPY"
        assert mr.get_market_regime_label(default="BULL") == "BULL"

    def test_fresh_file_returns_data(self, _isolated_file):
        payload = _write(_isolated_file, regime="BEAR", age_seconds=60)
        data = mr.get_market_regime()
        assert data is not None
        assert data["regime"] == "BEAR"
        assert data["source"] == "clio_native"
        assert mr.get_market_regime_label() == "BEAR"

    def test_stale_file_returns_none(self, _isolated_file):
        _write(_isolated_file, regime="BULL", age_seconds=7 * 3600)  # older than 6h default
        assert mr.get_market_regime() is None
        assert mr.get_market_regime_label(default="CHOPPY") == "CHOPPY"

    def test_custom_max_age_accepts_older_data(self, _isolated_file):
        _write(_isolated_file, regime="BULL", age_seconds=7 * 3600)
        data = mr.get_market_regime(max_age_seconds=24 * 3600)
        assert data is not None
        assert data["regime"] == "BULL"

    def test_malformed_json_returns_none(self, _isolated_file):
        _isolated_file.write_text("{not valid json")
        assert mr.get_market_regime() is None
        assert mr.get_market_regime_label() == "CHOPPY"

    def test_missing_regime_key_returns_none(self, _isolated_file):
        _isolated_file.write_text(json.dumps({"as_of": _iso(datetime.now(timezone.utc))}))
        assert mr.get_market_regime() is None

    def test_invalid_regime_value_returns_none(self, _isolated_file):
        _write(_isolated_file, regime="SIDEWAYS", age_seconds=0)
        assert mr.get_market_regime() is None

    def test_missing_as_of_returns_none(self, _isolated_file):
        _isolated_file.write_text(json.dumps({"regime": "BULL"}))
        assert mr.get_market_regime() is None

    def test_future_as_of_returns_none(self, _isolated_file):
        _write(_isolated_file, regime="BULL", age_seconds=-3600)  # 1h in the future
        assert mr.get_market_regime() is None

    def test_mtime_cache_reflects_updates(self, _isolated_file):
        _write(_isolated_file, regime="BULL", age_seconds=0)
        assert mr.get_market_regime_label() == "BULL"
        _write(_isolated_file, regime="BEAR", age_seconds=0)
        assert mr.get_market_regime_label() == "BEAR"

    def test_cache_sees_a_same_size_rewrite_within_one_mtime_tick(self, _isolated_file):
        """The VPS case (2026-09-27): two writes of the same size landed on the same ext4 mtime
        tick, so an (mtime, size) cache key served the old regime. Forced here on any filesystem."""
        as_of = _iso(datetime.now(timezone.utc))
        base = {"confidence": 1.0, "vix": 14.0, "spy_close": 500.0, "spy_sma20": 495.0,
                "spy_sma50": 490.0, "as_of": as_of, "source": "clio_native"}
        _write_atomic(_isolated_file, {**base, "regime": "BULL"})
        first = os.stat(_isolated_file)
        assert mr.get_market_regime_label() == "BULL"
        _write_atomic(_isolated_file, {**base, "regime": "BEAR"})
        os.utime(_isolated_file, ns=(first.st_atime_ns, first.st_mtime_ns))
        second = os.stat(_isolated_file)
        # Precondition: the old (mtime, size) key really collides.
        assert (second.st_mtime_ns, second.st_size) == (first.st_mtime_ns, first.st_size)
        assert mr.get_market_regime_label() == "BEAR"

    def test_never_raises_on_unreadable_path(self, monkeypatch):
        # point at a directory (not a file) — os.stat succeeds, open() raises IsADirectoryError
        monkeypatch.setenv("CLIO_MARKET_REGIME_FILE", "/")
        mr._cache.clear()
        assert mr.get_market_regime() is None
        assert mr.get_market_regime_label() == "CHOPPY"


# ---------------------------------------------------------------------------
# refresh_market_regime() — atomic write, failure keeps previous file, never raises
# ---------------------------------------------------------------------------


class TestRefresh:
    def _bars(self, n=60, start=480.0, step=0.5):
        return _dated([start + i * step for i in range(n)])

    def test_refresh_writes_atomically_and_is_readable(self, _isolated_file, monkeypatch):
        monkeypatch.setattr(mr, "_fetch_spy_daily_bars", lambda **kw: self._bars())

        async def fake_vix():
            return 16.5

        monkeypatch.setattr(mr, "_fetch_vix", fake_vix)

        result = asyncio.run(mr.refresh_market_regime())
        assert result is not None
        assert result["regime"] in mr._REGIMES
        assert result["source"] == "clio_native"
        assert _isolated_file.exists()
        # no leftover tmp files
        assert not list(_isolated_file.parent.glob("*.tmp"))

        mr._cache.clear()
        data = mr.get_market_regime()
        assert data is not None
        assert data["regime"] == result["regime"]

    def test_refresh_insufficient_bars_keeps_previous_file_and_returns_none(self, _isolated_file, monkeypatch):
        previous = _write(_isolated_file, regime="BULL", age_seconds=0)
        monkeypatch.setattr(mr, "_fetch_spy_daily_bars", lambda **kw: _dated([500.0, 501.0]))  # too few

        async def fake_vix():
            return 15.0

        monkeypatch.setattr(mr, "_fetch_vix", fake_vix)

        result = asyncio.run(mr.refresh_market_regime())
        assert result is None
        on_disk = json.loads(_isolated_file.read_text())
        assert on_disk["regime"] == previous["regime"]

    def test_refresh_fetch_raises_never_propagates(self, _isolated_file, monkeypatch):
        def boom(**kw):
            raise RuntimeError("alpaca down")

        monkeypatch.setattr(mr, "_fetch_spy_daily_bars", boom)
        result = asyncio.run(mr.refresh_market_regime())
        assert result is None  # never raises

    def test_refresh_vix_source_error_still_completes_on_trend_alone(self, _isolated_file, monkeypatch):
        """_fetch_vix() swallows its own errors internally (returns None) — a broken VIX
        source must not block a refresh; it should complete using the SPY trend alone."""
        monkeypatch.setattr(mr, "_fetch_spy_daily_bars", lambda **kw: self._bars())

        async def boom_snapshot(symbol):
            raise RuntimeError("polygon down")

        monkeypatch.setattr("agent.integrations.polygon.get_snapshot_raw", boom_snapshot)

        result = asyncio.run(mr.refresh_market_regime())
        assert result is not None
        assert result["vix"] is None
        assert result["regime"] in mr._REGIMES

    def test_refresh_vix_awaitable_raise_is_still_caught_by_outer_guard(self, _isolated_file, monkeypatch):
        """Even if _fetch_vix itself were replaced with something that raises (bypassing its
        own internal handling), refresh_market_regime's outer guard must never propagate."""
        monkeypatch.setattr(mr, "_fetch_spy_daily_bars", lambda **kw: self._bars())

        async def boom_vix():
            raise RuntimeError("polygon down")

        monkeypatch.setattr(mr, "_fetch_vix", boom_vix)
        result = asyncio.run(mr.refresh_market_regime())
        assert result is None  # outer try/except caught the propagated error; no crash

    def test_refresh_write_failure_keeps_previous_file(self, _isolated_file, monkeypatch):
        previous = _write(_isolated_file, regime="BEAR", age_seconds=0)
        monkeypatch.setattr(mr, "_fetch_spy_daily_bars", lambda **kw: self._bars())

        async def fake_vix():
            return 15.0

        monkeypatch.setattr(mr, "_fetch_vix", fake_vix)
        monkeypatch.setattr(mr, "_write_atomic", lambda payload: False)

        result = asyncio.run(mr.refresh_market_regime())
        assert result is None
        on_disk = json.loads(_isolated_file.read_text())
        assert on_disk["regime"] == previous["regime"]

    def test_refresh_never_raises_even_on_get_running_loop_failure(self, _isolated_file, monkeypatch):
        # Simulate a totally malformed payload path by making the classify step blow up.
        monkeypatch.setattr(mr, "_fetch_spy_daily_bars", lambda **kw: self._bars())

        async def fake_vix():
            return 15.0

        monkeypatch.setattr(mr, "_fetch_vix", fake_vix)

        def boom_classify(*a, **kw):
            raise ValueError("boom")

        monkeypatch.setattr(mr, "classify_regime", boom_classify)
        result = asyncio.run(mr.refresh_market_regime())
        assert result is None


class TestAtomicWriteHelper:
    def test_write_atomic_success_leaves_no_tmp(self, _isolated_file):
        ok = mr._write_atomic({"regime": "BULL", "as_of": _iso(datetime.now(timezone.utc))})
        assert ok is True
        assert _isolated_file.exists()
        assert not list(_isolated_file.parent.glob("*.tmp"))

    def test_write_atomic_failure_returns_false(self, tmp_path, monkeypatch):
        # point the file at a path whose parent cannot be created (a file, not a dir)
        blocker = tmp_path / "blocker"
        blocker.write_text("x")
        bad_path = blocker / "market_regime.json"
        monkeypatch.setenv("CLIO_MARKET_REGIME_FILE", str(bad_path))
        ok = mr._write_atomic({"regime": "BULL"})
        assert ok is False


# ---------------------------------------------------------------------------
# R-15: newest daily bars are requested, bars_as_of is recorded, stale bars are refused
# ---------------------------------------------------------------------------


class TestNewestBars:
    def _closes_80(self, start=400.0):
        # 80 calendar days of bars, close_i = start + i, newest last; ends today (ET)
        return _dated([start + i for i in range(80)])

    def test_request_asks_for_newest_bars_and_closes_last_is_newest(self, _isolated_file, monkeypatch, fake_bars):
        from alpaca.common.enums import Sort

        bars = self._closes_80()
        client = fake_bars(bars)

        async def fake_vix():
            return 15.0

        monkeypatch.setattr(mr, "_fetch_vix", fake_vix)
        result = asyncio.run(mr.refresh_market_regime())

        assert client.requests, "bars endpoint was never called"
        assert client.requests[0].sort == Sort.DESC
        newest_day, newest_close = bars[-1]
        assert result is not None
        assert result["spy_close"] == round(newest_close, 4)
        assert result["bars_as_of"] == newest_day.isoformat()
        on_disk = json.loads(_isolated_file.read_text())
        assert on_disk["bars_as_of"] == newest_day.isoformat()

    def test_fetch_returns_ascending_newest_window(self, fake_bars):
        bars = self._closes_80()
        fake_bars(bars)
        got = mr._fetch_spy_daily_bars()
        assert len(got) == mr._SESSIONS_REQUESTED
        assert got[-1] == bars[-1]
        assert [d for d, _ in got] == sorted(d for d, _ in got)

    def test_stale_last_bar_refuses_and_keeps_old_file_byte_identical(
        self, _isolated_file, monkeypatch, fake_bars, caplog
    ):
        _write(_isolated_file, regime="BEAR", age_seconds=0)
        before = _isolated_file.read_bytes()
        stale_last = _today_et() - timedelta(days=5)
        fake_bars(_dated([400.0 + i for i in range(60)], last_day=stale_last))

        async def fake_vix():
            return 15.0

        monkeypatch.setattr(mr, "_fetch_vix", fake_vix)
        with caplog.at_level(logging.WARNING, logger=mr.logger.name):
            result = asyncio.run(mr.refresh_market_regime())

        assert result is None
        assert _isolated_file.read_bytes() == before
        assert not list(_isolated_file.parent.glob("*.tmp"))
        assert any(
            f"last bar {stale_last.isoformat()} is stale; keeping previous regime" in r.getMessage()
            for r in caplog.records
            if r.levelno == logging.WARNING
        )

    def test_weekend_last_bar_friday_accepted_on_monday_morning(self, _isolated_file, monkeypatch, fake_bars):
        monday_0900_et = datetime(2026, 10, 12, 9, 0, tzinfo=_NY)
        monkeypatch.setattr(mr, "_now_utc", lambda: monday_0900_et.astimezone(timezone.utc))
        friday = datetime(2026, 10, 9).date()
        fake_bars(_dated([400.0 + i for i in range(60)], last_day=friday))

        async def fake_vix():
            return 15.0

        monkeypatch.setattr(mr, "_fetch_vix", fake_vix)
        result = asyncio.run(mr.refresh_market_regime())
        assert result is not None
        assert result["bars_as_of"] == "2026-10-09"
        mr._cache.clear()
        assert mr.get_market_regime() is not None

    @pytest.mark.parametrize("now_et, accepted", [
        (datetime(2026, 10, 13, 9, 0, tzinfo=_NY), True),    # Fri bar, Tue morning = exactly 4 days: Monday-holiday weekend
        (datetime(2026, 10, 14, 9, 0, tzinfo=_NY), False),   # 5 days: refused
    ])
    def test_stale_bar_boundary_is_exactly_four_calendar_days(self, _isolated_file, monkeypatch, fake_bars, now_et, accepted):
        monkeypatch.setattr(mr, "_now_utc", lambda: now_et.astimezone(timezone.utc))
        fake_bars(_dated([400.0 + i for i in range(60)], last_day=datetime(2026, 10, 9).date()))

        async def fake_vix():
            return 15.0

        monkeypatch.setattr(mr, "_fetch_vix", fake_vix)
        result = asyncio.run(mr.refresh_market_regime())
        assert (result is not None) is accepted


    def test_old_file_without_bars_as_of_still_readable(self, _isolated_file):
        _write(_isolated_file, regime="BULL", age_seconds=60)
        assert "bars_as_of" not in json.loads(_isolated_file.read_text())
        data = mr.get_market_regime()
        assert data is not None
        assert data["regime"] == "BULL"

    def test_read_rejects_payload_whose_bars_are_stale_even_if_as_of_is_fresh(self, _isolated_file):
        _write(_isolated_file, regime="BULL", age_seconds=60, bars_as_of=(_today_et() - timedelta(days=5)).isoformat())
        assert mr.get_market_regime() is None
        assert mr.get_market_regime_label(default="CHOPPY") == "CHOPPY"

    def test_read_accepts_payload_with_recent_bars_as_of(self, _isolated_file):
        _write(_isolated_file, regime="BEAR", age_seconds=60, bars_as_of=_today_et().isoformat())
        assert mr.get_market_regime_label() == "BEAR"
