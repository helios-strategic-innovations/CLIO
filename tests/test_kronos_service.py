"""agent.kronos.service -- scheduling math (next_wake_et across boundaries,
early close, weekend/holiday) and one full run() with fakes, writing both
files readable by agent.kronos.state."""

import asyncio
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from agent.kronos import state
from agent.kronos.bars import Bar, BarsClient
from agent.kronos.config import KronosConfig
from agent.kronos.service import KronosService, is_active_now, next_wake_et

_ET = ZoneInfo("America/New_York")

_HOLIDAYS = frozenset({date(2026, 12, 25)})
_EARLY_CLOSE = frozenset({date(2026, 11, 27), date(2026, 12, 24)})


def _et(y, m, d, hh, mm, ss=0) -> datetime:
    return datetime(y, m, d, hh, mm, ss, tzinfo=_ET)


# ── next_wake_et ──────────────────────────────────────────────────────────

def test_next_wake_before_session_open_is_first_wake():
    now = _et(2026, 9, 22, 8, 0)
    wake = next_wake_et(now, _HOLIDAYS, _EARLY_CLOSE)
    assert wake == _et(2026, 9, 22, 9, 35, 20)


def test_next_wake_mid_boundary_rolls_to_next_boundary():
    now = _et(2026, 9, 22, 9, 37, 0)
    wake = next_wake_et(now, _HOLIDAYS, _EARLY_CLOSE)
    assert wake == _et(2026, 9, 22, 9, 40, 20)


def test_next_wake_right_after_a_wake_moves_to_the_following_one():
    now = _et(2026, 9, 22, 9, 40, 25)  # just past the 09:40:20 wake
    wake = next_wake_et(now, _HOLIDAYS, _EARLY_CLOSE)
    assert wake == _et(2026, 9, 22, 9, 45, 20)


def test_next_wake_exactly_at_a_wake_returns_same_instant():
    now = _et(2026, 9, 22, 9, 40, 20)
    wake = next_wake_et(now, _HOLIDAYS, _EARLY_CLOSE)
    assert wake == now


def test_next_wake_last_wake_of_day_is_5_before_close():
    now = _et(2026, 9, 22, 15, 52, 0)  # between the 15:50:20 and 15:55:20 wakes
    wake = next_wake_et(now, _HOLIDAYS, _EARLY_CLOSE)
    assert wake == _et(2026, 9, 22, 15, 55, 20)


def test_next_wake_after_last_wake_rolls_to_next_trading_day():
    now = _et(2026, 9, 22, 15, 56, 0)  # past the last (15:55:20) wake
    wake = next_wake_et(now, _HOLIDAYS, _EARLY_CLOSE)
    assert wake == _et(2026, 9, 23, 9, 35, 20)


def test_next_wake_skips_weekend():
    now = _et(2026, 9, 25, 16, 0, 0)  # Friday, after close
    wake = next_wake_et(now, _HOLIDAYS, _EARLY_CLOSE)
    assert wake == _et(2026, 9, 28, 9, 35, 20)  # Monday


def test_next_wake_from_within_the_weekend_itself():
    now = _et(2026, 9, 26, 10, 0, 0)  # Saturday
    wake = next_wake_et(now, _HOLIDAYS, _EARLY_CLOSE)
    assert wake == _et(2026, 9, 28, 9, 35, 20)


def test_next_wake_early_close_day_last_wake_is_5_before_1300():
    now = _et(2026, 11, 27, 12, 52, 0)  # between the 12:50:20 and 12:55:20 wakes
    wake = next_wake_et(now, _HOLIDAYS, _EARLY_CLOSE)
    assert wake == _et(2026, 11, 27, 12, 55, 20)


def test_next_wake_early_close_day_after_last_wake_rolls_forward():
    now = _et(2026, 11, 27, 13, 0, 0)  # early close already happened
    wake = next_wake_et(now, _HOLIDAYS, _EARLY_CLOSE)
    assert wake > _et(2026, 11, 27, 12, 55, 20)
    assert wake.date() > date(2026, 11, 27)


def test_next_wake_skips_full_holiday():
    # Thursday Dec 24 2026 is an early-close half day; Friday Dec 25 is a
    # full holiday -- the wake after Dec 24's window must skip to the next
    # trading day entirely (not Dec 25).
    now = _et(2026, 12, 24, 13, 0, 0)  # just past Dec 24's 12:55:20 last wake
    wake = next_wake_et(now, _HOLIDAYS, _EARLY_CLOSE)
    assert wake.date() != date(2026, 12, 25)
    assert wake.date() > date(2026, 12, 24)


# ── is_active_now ─────────────────────────────────────────────────────────

def test_is_active_now_true_within_window():
    assert is_active_now(_et(2026, 9, 22, 12, 0), _HOLIDAYS, _EARLY_CLOSE) is True


def test_is_active_now_false_before_window():
    assert is_active_now(_et(2026, 9, 22, 9, 30), _HOLIDAYS, _EARLY_CLOSE) is False


def test_is_active_now_false_on_weekend():
    assert is_active_now(_et(2026, 9, 26, 12, 0), _HOLIDAYS, _EARLY_CLOSE) is False


# ── Full run_once() with fakes ────────────────────────────────────────────

class _FakeBarsClient(BarsClient):
    def __init__(self, bars_by_symbol: dict[str, list[Bar]]):
        self._bars_by_symbol = bars_by_symbol

    async def get_bars(self, symbol, lookback_bars, now=None):
        return self._bars_by_symbol.get(symbol)

    def feed_status(self):
        return {"effective": "iex", "downgraded": False}


class _FakePredictor:
    def __init__(self, close_delta_by_symbol: dict[str, float]):
        self._delta = close_delta_by_symbol
        self.calls = []

    def predict_batch(self, bars_by_symbol, pred_len, seed):
        self.calls.append((set(bars_by_symbol.keys()), pred_len, seed))
        out = {}
        for sym, bars in bars_by_symbol.items():
            anchor = bars[-1].c
            delta = self._delta.get(sym, 0.001)
            # Two paths per symbol: one steadily up, one steadily down --
            # enough for compute_horizon_metrics (needs >= 2 finite samples).
            up_path = [anchor * (1.0 + delta * i) for i in range(1, pred_len + 1)]
            down_path = [anchor * (1.0 - delta * i) for i in range(1, pred_len + 1)]
            out[sym] = [up_path, down_path]
        return out


def _make_bars(day: date, last_open_hh: int, last_open_mm: int, n: int, last_close: float) -> list[Bar]:
    """`n` ascending 5-min RTH bars ending at `last_open_hh:last_open_mm`
    (open time of the LAST bar), closes ramping toward `last_close`."""
    out = []
    t0 = _et(day.year, day.month, day.day, last_open_hh, last_open_mm) - timedelta(minutes=5 * (n - 1))
    for i in range(n):
        t_open = t0 + timedelta(minutes=5 * i)
        c = last_close - (n - 1 - i) * 0.01
        out.append(Bar(t_open=t_open.timestamp(), t_close=t_open.timestamp() + 300, o=c, h=c, l=c, c=c, v=1000.0, amount=1000.0 * c))
    return out


def test_run_once_writes_forecast_and_scores_readable_by_state(tmp_path, monkeypatch):
    forecast_file = tmp_path / "kronos_forecast.json"
    scores_file = tmp_path / "kronos_scores.json"
    history_file = tmp_path / "kronos_history.jsonl"
    monkeypatch.setenv("CLIO_KRONOS_FORECAST_FILE", str(forecast_file))
    monkeypatch.setenv("CLIO_KRONOS_SCORES_FILE", str(scores_file))

    now = _et(2026, 9, 22, 11, 0, 20)  # a normal wake instant
    bars = {
        "SPY": _make_bars(date(2026, 9, 22), 10, 55, 200, 500.0),  # last bar closes at 11:00
        "QQQ": _make_bars(date(2026, 9, 22), 10, 55, 200, 400.0),
    }
    cfg = KronosConfig(symbols=("SPY", "QQQ"), sample_paths=2, lookback_bars=200, history_file=str(history_file))
    service = KronosService(cfg, bars_client=_FakeBarsClient(bars), predictor=_FakePredictor({"SPY": 0.002, "QQQ": 0.002}))

    result = asyncio.run(service.run_once(now=now, dry_run=False))
    assert result["ok_count"] == 2
    assert result["symbols"]["SPY"]["status"] == "ok"
    assert set(result["symbols"]["SPY"]["horizons"].keys()) == {"15", "30", "60"}

    forecast = state.read_forecast(now=now.timestamp())
    assert forecast is not None
    assert forecast["symbols"]["SPY"]["status"] == "ok"
    assert forecast["params"]["sample_paths"] == 2
    assert forecast["model"]["model"] == cfg.model_id

    scores = state.read_scores()
    assert scores is not None
    assert scores["symbols"] == {}  # nothing resolved yet on the very first run

    import os
    assert os.path.exists(history_file)


def test_run_once_dry_run_writes_nothing(tmp_path, monkeypatch):
    forecast_file = tmp_path / "kronos_forecast.json"
    scores_file = tmp_path / "kronos_scores.json"
    history_file = tmp_path / "kronos_history.jsonl"
    monkeypatch.setenv("CLIO_KRONOS_FORECAST_FILE", str(forecast_file))
    monkeypatch.setenv("CLIO_KRONOS_SCORES_FILE", str(scores_file))

    now = _et(2026, 9, 22, 11, 0, 20)
    bars = {"SPY": _make_bars(date(2026, 9, 22), 10, 55, 200, 500.0)}
    cfg = KronosConfig(symbols=("SPY",), sample_paths=2, lookback_bars=200, history_file=str(history_file))
    service = KronosService(cfg, bars_client=_FakeBarsClient(bars), predictor=_FakePredictor({"SPY": 0.002}))

    result = asyncio.run(service.run_once(now=now, dry_run=True))
    assert result["dry_run"] is True
    import os
    assert not os.path.exists(forecast_file)
    assert not os.path.exists(scores_file)
    assert not os.path.exists(history_file)


def test_run_once_marks_no_data_status_when_bars_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIO_KRONOS_FORECAST_FILE", str(tmp_path / "f.json"))
    monkeypatch.setenv("CLIO_KRONOS_SCORES_FILE", str(tmp_path / "s.json"))
    now = _et(2026, 9, 22, 11, 0, 20)
    cfg = KronosConfig(symbols=("SPY",), history_file=str(tmp_path / "h.jsonl"))
    service = KronosService(cfg, bars_client=_FakeBarsClient({}), predictor=_FakePredictor({}))
    result = asyncio.run(service.run_once(now=now, dry_run=False))
    assert result["symbols"]["SPY"]["status"] == "no_data"
    assert result["ok_count"] == 0


def test_run_once_no_horizons_fit_near_close_is_ok_not_error(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIO_KRONOS_FORECAST_FILE", str(tmp_path / "f.json"))
    monkeypatch.setenv("CLIO_KRONOS_SCORES_FILE", str(tmp_path / "s.json"))
    now = _et(2026, 9, 22, 15, 55, 20)  # last wake of the day
    bars = {"SPY": _make_bars(date(2026, 9, 22), 15, 55, 200, 500.0)}  # anchor bar closes at 16:00 (the close itself)
    cfg = KronosConfig(symbols=("SPY",), sample_paths=2, history_file=str(tmp_path / "h.jsonl"))
    service = KronosService(cfg, bars_client=_FakeBarsClient(bars), predictor=_FakePredictor({"SPY": 0.002}))
    result = asyncio.run(service.run_once(now=now, dry_run=False))
    assert result["symbols"]["SPY"]["status"] == "ok"
    assert result["symbols"]["SPY"]["horizons"] == {}


def test_run_once_predictor_failure_marks_error_status(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIO_KRONOS_FORECAST_FILE", str(tmp_path / "f.json"))
    monkeypatch.setenv("CLIO_KRONOS_SCORES_FILE", str(tmp_path / "s.json"))

    class _BrokenPredictor:
        def predict_batch(self, *_args, **_kwargs):
            raise RuntimeError("model exploded")

    now = _et(2026, 9, 22, 11, 0, 20)
    bars = {"SPY": _make_bars(date(2026, 9, 22), 10, 55, 200, 500.0)}
    cfg = KronosConfig(symbols=("SPY",), history_file=str(tmp_path / "h.jsonl"))
    service = KronosService(cfg, bars_client=_FakeBarsClient(bars), predictor=_BrokenPredictor())
    result = asyncio.run(service.run_once(now=now, dry_run=False))
    assert result["symbols"]["SPY"]["status"] == "error"
    assert "model exploded" in result["symbols"]["SPY"]["error"]


def test_startup_check_loads_the_model_and_probes_bars(caplog):
    loaded = []

    class _LoadingPredictor(_FakePredictor):
        def _ensure_loaded(self):
            loaded.append(True)

    bars = {"SPY": _make_bars(date(2026, 9, 25), 15, 55, 5, 580.0)}
    svc = KronosService(KronosConfig(symbols=("SPY", "QQQ")), bars_client=_FakeBarsClient(bars),
                        predictor=_LoadingPredictor({}))

    class _Raising(_FakeBarsClient):
        async def get_bars(self, symbol, lookback_bars, now=None):
            if symbol == "QQQ":
                raise ConnectionError("blip")
            return await super().get_bars(symbol, lookback_bars, now)

    svc._bars_client = _Raising(bars)
    with caplog.at_level("INFO"):
        asyncio.run(svc.startup_check())
    assert loaded == [True]
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "SPY: 5 bars" in text and "get_bars(QQQ) failed" in text  # bars failure logged, tolerated


def test_startup_check_raises_when_the_model_cannot_load():
    from agent.kronos.forecaster import ModelUnavailable

    class _Broken(_FakePredictor):
        def _ensure_loaded(self):
            raise ModelUnavailable("cache empty")

    svc = KronosService(KronosConfig(symbols=("SPY",)), bars_client=_FakeBarsClient({}), predictor=_Broken({}))
    with pytest.raises(ModelUnavailable):
        asyncio.run(svc.startup_check())


def test_batch_seed_depends_on_every_symbols_anchor(tmp_path):
    """The seed must change when ANY symbol's anchor bar changes (not just the first symbol's)."""
    now = _et(2026, 9, 22, 11, 0, 20)
    cfg = KronosConfig(symbols=("SPY", "QQQ"), sample_paths=2, lookback_bars=200,
                       history_file=str(tmp_path / "h.jsonl"))

    def seed_for(qqq_last_open_mm):
        bars = {
            "SPY": _make_bars(date(2026, 9, 22), 10, 55, 200, 500.0),
            "QQQ": _make_bars(date(2026, 9, 22), 10, qqq_last_open_mm, 200, 400.0),
        }
        pred = _FakePredictor({})
        svc = KronosService(cfg, bars_client=_FakeBarsClient(bars), predictor=pred)
        asyncio.run(svc.run_once(now=now, dry_run=True))
        return pred.calls[0][2]

    assert seed_for(55) == seed_for(55)          # same anchors -> same seed
    assert seed_for(50) != seed_for(55)          # only QQQ's anchor moved -> different seed


# ── R-20: per-run timing line + slow-predict WARNING ──────────────────────

class _ClockedPredictor(_FakePredictor):
    """Advances a scripted monotonic clock by `cost_s` inside predict_batch,
    so the predict timing is exact and the test never sleeps."""

    def __init__(self, clock, cost_s: float):
        super().__init__({"SPY": 0.002})
        self._clock = clock
        self._cost_s = cost_s

    def predict_batch(self, bars_by_symbol, pred_len, seed):
        self._clock["t"] += self._cost_s
        return super().predict_batch(bars_by_symbol, pred_len, seed)


def _run_timed(tmp_path, monkeypatch, cost_s: float, caplog):
    import agent.kronos.service as service_mod

    monkeypatch.setenv("CLIO_KRONOS_FORECAST_FILE", str(tmp_path / "f.json"))
    monkeypatch.setenv("CLIO_KRONOS_SCORES_FILE", str(tmp_path / "s.json"))
    clock = {"t": 1000.0}
    monkeypatch.setattr(service_mod, "_clock", lambda: clock["t"])
    now = _et(2026, 9, 22, 11, 0, 20)
    bars = {"SPY": _make_bars(date(2026, 9, 22), 10, 55, 200, 500.0)}
    cfg = KronosConfig(symbols=("SPY",), sample_paths=2, lookback_bars=200, history_file=str(tmp_path / "h.jsonl"))
    service = KronosService(cfg, bars_client=_FakeBarsClient(bars), predictor=_ClockedPredictor(clock, cost_s))
    with caplog.at_level("INFO", logger="clio.kronos.service"):
        asyncio.run(service.run_once(now=now, dry_run=True))
    return [r for r in caplog.records if r.name == "clio.kronos.service"]


def test_run_logs_predict_and_total_timing_at_info(tmp_path, monkeypatch, caplog):
    recs = _run_timed(tmp_path, monkeypatch, 95.0, caplog)
    lines = [r for r in recs if r.levelname == "INFO" and r.getMessage().startswith("kronos run: ")]
    assert len(lines) == 1
    assert lines[0].getMessage() == "kronos run: predict=95.0s total=95.0s asof=2026-09-22T11:00:20-04:00 ok=1/1"


def test_slow_predict_over_240s_warns(tmp_path, monkeypatch, caplog):
    recs = _run_timed(tmp_path, monkeypatch, 250.0, caplog)
    warns = [r for r in recs if r.levelname == "WARNING" and "predict_batch took" in r.getMessage()]
    assert len(warns) == 1
    assert "250.0s" in warns[0].getMessage()
    assert any(r.getMessage().startswith("kronos run: predict=250.0s") for r in recs)


def test_predict_exactly_at_240s_does_not_warn(tmp_path, monkeypatch, caplog):
    recs = _run_timed(tmp_path, monkeypatch, 240.0, caplog)
    assert [r for r in recs if r.levelname == "WARNING" and "predict_batch took" in r.getMessage()] == []
