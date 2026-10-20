"""
CLIO Kronos — service orchestrator (scheduling + one full run).

Mirrors `agent.flowcollector.collector.FlowCollectorService`'s shape: a
duck-typed `PredictorLike`/`agent.kronos.bars.BarsClient` pair production
code wires to the real Alpaca/torch adapters and tests wire to fakes, an
`async def run_once(...)` that does exactly one forecast+score pass, and an
`async def run_forever(stop_event)` outer loop `__main__.py` drives.

Blocking calls (bar fetches, the torch predictor call, the history/state
file I/O) all go through `loop.run_in_executor` — CLAUDE.md anti-pattern
#19 ("never block the event loop thread"); this matters even more here
than in most CLIO services because a stalled CPU-bound Kronos inference
must never freeze SIGTERM handling or the scheduling wait.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import logging
import math
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

from agent.kronos import scorer, state
from agent.kronos.bars import (
    BAR_MINUTES,
    AlpacaBarsClient,
    Bar,
    BarsClient,
    is_trading_day,
    session_close_et,
    session_open_et,
)
from agent.kronos.config import KronosConfig
from agent.kronos.forecaster import KronosPredictor, ModelUnavailable, build_forecast_horizons, resolve_pred_plan, seed_from_anchor

logger = logging.getLogger("clio.kronos.service")

_ET = ZoneInfo("America/New_York")

# 20s after each 5-minute boundary (design doc §3's cadence).
WAKE_DELAY = timedelta(seconds=20)
_STEP = timedelta(minutes=BAR_MINUTES)
# R-20: a predict_batch run that takes longer than this is logged at WARNING
# (documented cost ~91-95 s; the wake cadence is 300 s).
_PREDICT_WARN_SECONDS = 240.0
# Monotonic clock for run timing; a module attribute so tests can script it.
_clock = time.monotonic


def _market_calendar() -> tuple[frozenset, frozenset]:
    from agent.integrations.market_events import MARKET_EARLY_CLOSE, MARKET_HOLIDAYS

    return MARKET_HOLIDAYS, MARKET_EARLY_CLOSE


def next_wake_et(
    now_et: datetime, holidays: frozenset, early_close: frozenset, *, lookahead_days: int = 10,
) -> datetime:
    """PURE. The next wake instant at or after `now_et`: 20s after a
    5-minute boundary, for boundaries from `session_open + 5min` through
    `session_close - 5min` inclusive (09:35-15:55 ET on a normal day;
    early-close-aware -- the window shrinks to 09:35-12:55 ET on a
    half-day), on the next trading day that has at least one such
    boundary. Skips weekends/holidays entirely."""
    day = now_et.date()
    first_day = True
    for _ in range(lookahead_days):
        sess_open = session_open_et(day, holidays)
        sess_close = session_close_et(day, holidays, early_close)
        if sess_open is not None and sess_close is not None:
            first_boundary = sess_open + _STEP
            last_boundary = sess_close - _STEP
            if last_boundary >= first_boundary:
                if first_day:
                    min_boundary_time = now_et - WAKE_DELAY
                    if min_boundary_time <= first_boundary:
                        candidate = first_boundary
                    else:
                        delta_steps = (min_boundary_time - first_boundary) / _STEP
                        candidate = first_boundary + math.ceil(delta_steps) * _STEP
                else:
                    candidate = first_boundary
                if candidate <= last_boundary:
                    return candidate + WAKE_DELAY
        day = day + timedelta(days=1)
        first_day = False
    raise RuntimeError(f"kronos: no trading day found within {lookahead_days} days of {now_et.isoformat()}")


def is_active_now(now_et: datetime, holidays: frozenset, early_close: frozenset) -> bool:
    """True inside the same [session_open+5min, session_close-5min] window
    `next_wake_et` schedules wakes within -- used only for `--once`'s
    informational summary, never to gate whether a forecast is computed."""
    day = now_et.date()
    if not is_trading_day(day, holidays):
        return False
    sess_open = session_open_et(day, holidays)
    sess_close = session_close_et(day, holidays, early_close)
    if sess_open is None or sess_close is None:
        return False
    return sess_open + _STEP <= now_et <= sess_close - _STEP


class KronosService:
    def __init__(self, cfg: KronosConfig, *, bars_client: Optional[BarsClient] = None, predictor: Optional[Any] = None) -> None:
        self._cfg = cfg
        self._bars_client: BarsClient = bars_client if bars_client is not None else AlpacaBarsClient(cfg)
        self._predictor = predictor if predictor is not None else KronosPredictor(cfg)
        # One inference at a time: a run that times out is cancelled at the await, but its torch
        # thread keeps going -- a single-worker pool makes the next run queue behind it instead of
        # running a second model call alongside it.
        self._predict_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="kronos-predict")

    def _feed_used(self) -> str:
        status = getattr(self._bars_client, "feed_status", None)
        if callable(status):
            try:
                return str(status().get("effective", "unknown"))
            except Exception:
                return "unknown"
        return "unknown"

    async def run_once(self, now: Optional[datetime] = None, *, dry_run: bool = False) -> dict:
        """One forecast+score pass over every configured symbol. Never
        raises -- a per-symbol failure downgrades that symbol's status
        (`"no_data"`/`"error"`), the run always completes and writes (or,
        with `dry_run=True`, would-write) both files."""
        loop = asyncio.get_running_loop()
        run_t0 = _clock()
        predict_s = 0.0
        now_dt = now if now is not None else datetime.now(timezone.utc)
        now_epoch = now_dt.timestamp()
        holidays, early_close = _market_calendar()

        bars_by_symbol: dict[str, list[Bar]] = {}
        symbols_out: dict[str, dict] = {}
        anchor_epochs: dict[str, float] = {}
        pred_plans: dict[str, tuple[int, list]] = {}
        latencies: dict[str, int] = {}

        for symbol in self._cfg.symbols:
            t0 = time.time()
            try:
                bars = await self._bars_client.get_bars(symbol, self._cfg.lookback_bars, now=now_dt)
            except Exception as e:
                logger.error(f"kronos: get_bars({symbol}) raised: {type(e).__name__}: {e}")
                bars = None
            latency_ms = int((time.time() - t0) * 1000)
            latencies[symbol] = latency_ms

            if not bars:
                symbols_out[symbol] = {
                    "status": "no_data", "error": "no RTH bars available",
                    "asof_bar": None, "anchor_close": None, "feed": self._feed_used(),
                    "latency_ms": latency_ms, "horizons": {},
                }
                continue

            bars_by_symbol[symbol] = bars
            last_bar = bars[-1]
            anchor_et = datetime.fromtimestamp(last_bar.t_close, tz=timezone.utc).astimezone(_ET)
            close_et = session_close_et(anchor_et.date(), holidays, early_close)
            if close_et is None:
                # Defensive only -- filter_rth_bars already guarantees every
                # returned bar's open time is on a trading day.
                symbols_out[symbol] = {
                    "status": "error", "error": "anchor bar not on a trading day",
                    "asof_bar": anchor_et.isoformat(), "anchor_close": last_bar.c,
                    "feed": self._feed_used(), "latency_ms": latency_ms, "horizons": {},
                }
                continue

            pred_len, horizons_plan = resolve_pred_plan(anchor_et, close_et)
            anchor_epochs[symbol] = last_bar.t_close
            pred_plans[symbol] = (pred_len, horizons_plan)

            if pred_len <= 0 or not horizons_plan:
                # Every horizon would cross the close from here -- nothing
                # to forecast this run (never extrapolate overnight), but
                # this is NOT a data/model error.
                symbols_out[symbol] = {
                    "status": "ok", "error": None, "asof_bar": anchor_et.isoformat(),
                    "anchor_close": last_bar.c, "feed": self._feed_used(),
                    "latency_ms": latency_ms, "horizons": {},
                }

        # One shared batched predictor call across every symbol that still
        # needs one (status not yet decided above).
        need_predict = {s: bars_by_symbol[s] for s in bars_by_symbol if s not in symbols_out}
        if need_predict:
            max_pred_len = max((pred_plans[s][0] for s in need_predict), default=0)
            # One seed for the batched call, derived from EVERY symbol's anchor bar, so the same set of
            # anchors always samples the same way even when one symbol's bars lag behind another's.
            seed = seed_from_anchor(sum(anchor_epochs[s] for s in sorted(need_predict)))
            predict_t0 = _clock()
            try:
                paths_by_symbol = await loop.run_in_executor(
                    self._predict_pool, self._predictor.predict_batch, need_predict, max_pred_len, seed,
                )
            except ModelUnavailable as e:
                logger.warning(f"kronos: model unavailable this run: {e}")
                paths_by_symbol = {}
                for s in need_predict:
                    symbols_out[s] = {
                        "status": "error", "error": str(e),
                        "asof_bar": datetime.fromtimestamp(bars_by_symbol[s][-1].t_close, tz=timezone.utc).astimezone(_ET).isoformat(),
                        "anchor_close": bars_by_symbol[s][-1].c, "feed": self._feed_used(),
                        "latency_ms": latencies.get(s, 0), "horizons": {},
                    }
            except Exception as e:
                logger.error(f"kronos: predictor.predict_batch failed: {type(e).__name__}: {e}")
                paths_by_symbol = {}
                for s in need_predict:
                    symbols_out[s] = {
                        "status": "error", "error": f"{type(e).__name__}: {e}",
                        "asof_bar": datetime.fromtimestamp(bars_by_symbol[s][-1].t_close, tz=timezone.utc).astimezone(_ET).isoformat(),
                        "anchor_close": bars_by_symbol[s][-1].c, "feed": self._feed_used(),
                        "latency_ms": latencies.get(s, 0), "horizons": {},
                    }
            finally:
                predict_s = _clock() - predict_t0

            for symbol, bars in need_predict.items():
                if symbol in symbols_out:
                    continue  # already downgraded to error above
                last_bar = bars[-1]
                anchor_et = datetime.fromtimestamp(last_bar.t_close, tz=timezone.utc).astimezone(_ET)
                close_et = session_close_et(anchor_et.date(), holidays, early_close)
                paths = paths_by_symbol.get(symbol) or []
                horizons = build_forecast_horizons(last_bar.c, anchor_et, close_et, paths)
                symbols_out[symbol] = {
                    "status": "ok", "error": None, "asof_bar": anchor_et.isoformat(),
                    "anchor_close": last_bar.c, "feed": self._feed_used(),
                    "latency_ms": latencies.get(symbol, 0), "horizons": horizons,
                }

        # record_forecast takes keyword args -- run_in_executor needs a
        # plain positional callable, so wrap with functools.partial.
        if not dry_run:
            for symbol, entry in symbols_out.items():
                if entry.get("status") != "ok" or not entry.get("horizons"):
                    continue
                await loop.run_in_executor(
                    None,
                    functools.partial(
                        scorer.record_forecast, self._cfg.history_file, self._cfg.history_max_lines,
                        symbol=symbol, anchor_epoch=anchor_epochs[symbol],
                        anchor_close=entry["anchor_close"], horizons=entry["horizons"],
                    ),
                )

        forecast_payload = {
            "schema": state.SCHEMA,
            "written_at": now_epoch,
            "model": {
                "model": self._cfg.model_id, "model_revision": self._cfg.model_revision,
                "tokenizer": self._cfg.tokenizer_id, "tokenizer_revision": self._cfg.tokenizer_revision,
                "vendor_commit": self._cfg.vendor_commit,
            },
            "params": {
                "bar_minutes": BAR_MINUTES, "lookback_bars": self._cfg.lookback_bars,
                "sample_paths": self._cfg.sample_paths, "temperature": self._cfg.temperature, "top_p": self._cfg.top_p,
            },
            "symbols": symbols_out,
        }

        scores_payload: dict = {"schema": state.SCHEMA, "written_at": now_epoch, "symbols": {}}
        if not dry_run:
            await loop.run_in_executor(None, state.write_json_atomic, state.forecast_path(), forecast_payload)
            scores_payload = await loop.run_in_executor(
                None,
                functools.partial(
                    scorer.resolve_and_score, self._cfg.history_file, self._cfg.history_max_lines,
                    bars_by_symbol=bars_by_symbol, now_epoch=now_epoch,
                ),
            )
            await loop.run_in_executor(None, state.write_json_atomic, state.scores_path(), scores_payload)

        ok_count = sum(1 for e in symbols_out.values() if e.get("status") == "ok")
        total_s = _clock() - run_t0
        asof_et = datetime.fromtimestamp(now_epoch, tz=timezone.utc).astimezone(_ET).isoformat()
        logger.info(
            "kronos run: predict=%.1fs total=%.1fs asof=%s ok=%d/%d",
            predict_s, total_s, asof_et, ok_count, len(self._cfg.symbols),
        )
        if predict_s > _PREDICT_WARN_SECONDS:
            logger.warning(
                "kronos: predict_batch took %.1fs (over the %.0fs warn threshold; wake cadence 300s)",
                predict_s, _PREDICT_WARN_SECONDS,
            )
        return {
            "as_of": now_epoch, "symbols": symbols_out, "scores": scores_payload,
            "ok_count": ok_count, "total": len(self._cfg.symbols), "dry_run": dry_run,
        }

    async def startup_check(self) -> None:
        """Fail fast at service start (not at the next session's first run): load the model -- this
        exercises the sandboxed, offline Hugging Face cache -- and fetch bars once. A model that
        cannot load raises ModelUnavailable (the unit then fails visibly and systemd retries); a bars
        failure is logged and tolerated (it may be a transient network blip)."""
        loop = asyncio.get_running_loop()
        ensure = getattr(self._predictor, "_ensure_loaded", None)
        if callable(ensure):
            await loop.run_in_executor(self._predict_pool, ensure)
        for symbol in self._cfg.symbols:
            try:
                bars = await self._bars_client.get_bars(symbol, self._cfg.lookback_bars)
                logger.info(f"kronos: startup check -- {symbol}: {len(bars)} bars (feed {self._feed_used()})")
            except Exception as e:
                logger.error(f"kronos: startup check -- get_bars({symbol}) failed: {type(e).__name__}: {e}")

    async def run_forever(self, stop_event: asyncio.Event) -> None:
        """`stop_event` is an asyncio.Event a SIGTERM/SIGINT handler sets
        (see `__main__.py`). Sleeps until the next scheduled wake, then runs
        one pass bounded by `cfg.run_timeout_seconds`; never lets a stuck
        run block the next wake indefinitely, and never raises out of the
        loop (a failed run just logs and the loop continues to the next
        wake)."""
        while not stop_event.is_set():
            now_utc = datetime.now(timezone.utc)
            now_et = now_utc.astimezone(_ET)
            holidays, early_close = _market_calendar()
            wake_et = next_wake_et(now_et, holidays, early_close)
            sleep_s = max(0.0, (wake_et - now_et).total_seconds())
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=sleep_s)
                break  # stop requested during the sleep
            except asyncio.TimeoutError:
                pass
            if stop_event.is_set():
                break
            try:
                result = await asyncio.wait_for(self.run_once(), timeout=self._cfg.run_timeout_seconds)
                logger.info(
                    f"kronos: run complete ok={result['ok_count']}/{result['total']} "
                    f"as_of={datetime.fromtimestamp(result['as_of'], tz=timezone.utc).astimezone(_ET).isoformat()}"
                )
            except asyncio.TimeoutError:
                logger.error(f"kronos: run exceeded run_timeout_seconds={self._cfg.run_timeout_seconds}s -- skipped")
            except Exception as e:
                logger.error(f"kronos: run_once failed: {type(e).__name__}: {e}", exc_info=True)
        logger.info("kronos: shutdown complete")
