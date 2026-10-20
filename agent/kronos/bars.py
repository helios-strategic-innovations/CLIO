"""
CLIO Kronos — regular-session 5-minute bar fetch.

Native 5-minute bars via Alpaca's `StockHistoricalDataClient`
(`TimeFrame(5, TimeFrameUnit.Minute)`), NOT 1-minute bars resampled
client-side. Chosen because: (1) upstream Kronos's own example forecasts
5-minute bars directly (docs/design/clio_kronos.md §3), so this is the
"native" shape the model expects; (2) a single request per symbol is
simpler than fetching 5x the 1-minute bars and re-aggregating OHLCV+VWAP,
which would need extra edge-case handling (a partial trailing 1-min bar at
the fetch boundary, re-weighting VWAP across 5 sub-bars) this module would
otherwise have to get right and test. The one thing this choice does NOT
control is which epoch-second boundary Alpaca aligns 5-minute buckets to —
`filter_rth_bars` below never assumes grid alignment; it only checks
wall-clock RTH-window membership and "has this bar's interval elapsed",
so it is correct regardless of the server's alignment choice.

Feed selection mirrors (does not import) `agent.dpof.broker.DPOFBroker`'s
sticky feed-downgrade pattern: a 401/403 on the configured feed (SIP)
downgrades to IEX for the rest of this process — "never go blind" — logged
once. Blocking Alpaca SDK calls run via `loop.run_in_executor`, the same
pattern `agent.dpof.broker.DPOFBroker.get_minute_bars` and
`agent.flowcollector.collector.AlpacaFlowBroker` use (CLAUDE.md anti-pattern
#19: never `fcntl`/blocking I/O directly on the event loop thread — the
same "never block the loop" rule applies to any blocking SDK call).

Keys (`ALPACA_PAPER_KEY_ID`/`ALPACA_PAPER_SECRET_KEY`, via
`KronosConfig.alpaca_key_id`/`alpaca_secret_key`) are read from env and
never logged (see `config.config_summary`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date as _date
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

from agent.kronos.config import KronosConfig

logger = logging.getLogger("clio.kronos.bars")

_ET = ZoneInfo("America/New_York")

BAR_MINUTES = 5
# 09:30-16:00 ET normal day = 6.5h = 78 five-minute bars/session; used only
# to size the calendar-day padding on the fetch window, never for filtering.
_BARS_PER_SESSION = 78


@dataclass(frozen=True)
class Bar:
    t_open: float   # epoch seconds, bar interval start
    t_close: float  # epoch seconds, bar interval end (t_open + BAR_MINUTES*60)
    o: float
    h: float
    l: float
    c: float
    v: float
    amount: float    # volume * vwap when vwap is present, else volume * close


def _f(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def compute_amount(volume: Optional[float], vwap: Optional[float], close: float) -> float:
    """`amount` (dollar volume) = volume * vwap when the bar carries a vwap,
    else volume * close — never a placeholder 0.0 (CLAUDE.md "no fake
    zeros" cousin: this is always a real, if approximate, quantity)."""
    v = volume if volume is not None else 0.0
    if vwap is not None:
        return v * vwap
    return v * close


def _market_calendar() -> tuple[frozenset, frozenset]:
    """Lazy import (mirrors agent.flowcollector.collector._market_calendar)
    -- 2026/2027-only sets per that module's own docstring; extend annually
    there, not here."""
    from agent.integrations.market_events import MARKET_EARLY_CLOSE, MARKET_HOLIDAYS

    return MARKET_HOLIDAYS, MARKET_EARLY_CLOSE


def is_trading_day(day: _date, holidays: frozenset) -> bool:
    return day.weekday() < 5 and day not in holidays


def session_open_et(day: _date, holidays: frozenset) -> Optional[datetime]:
    """09:30 ET on `day`, or None if `day` has no regular session at all
    (weekend or full NYSE holiday)."""
    if not is_trading_day(day, holidays):
        return None
    return datetime(day.year, day.month, day.day, 9, 30, 0, tzinfo=_ET)


def session_close_et(day: _date, holidays: frozenset, early_close: frozenset) -> Optional[datetime]:
    """The regular-session close on `day` (13:00 ET on an early-close
    half-day, else 16:00 ET), or None if `day` has no session at all."""
    if not is_trading_day(day, holidays):
        return None
    close_hour = 13 if day in early_close else 16
    return datetime(day.year, day.month, day.day, close_hour, 0, 0, tzinfo=_ET)


def filter_rth_bars(
    raw_bars: list[dict],
    *,
    bar_minutes: int,
    now_epoch: float,
    holidays: frozenset,
    early_close: frozenset,
) -> list[Bar]:
    """PURE. `raw_bars` items are `{"t": epoch_of_bar_open, "o","h","l","c",
    "v", "vw"(optional)}`. Keeps only bars whose OPEN time falls within
    `[session_open, session_close)` ET on a trading day (so a bar starting
    exactly at an early-close half-day's 12:55 is the last one kept; one
    starting at 13:00 on that day is excluded), AND whose CLOSE time
    (`t_open + bar_minutes*60`) has already elapsed as of `now_epoch` —
    never the still-forming bar. Returns ascending by `t_open`; a duplicate
    `t_open` keeps the LAST occurrence in `raw_bars` (mirrors
    `DPOFBroker.get_minute_bars`'s own sort-then-dedup convention, applied
    here as dedup-then-sort for the same end result). Malformed rows
    (missing/non-numeric `t`/o/h/l/c) are silently dropped, never crash the
    caller — the same "one bad row doesn't cost the batch" discipline as
    `AlpacaFlowBroker.get_option_universe`'s per-contract try/except."""
    bar_seconds = bar_minutes * 60
    by_open: dict[float, Bar] = {}
    for raw in raw_bars or []:
        t_open = _f(raw.get("t"))
        o, h, l, c = _f(raw.get("o")), _f(raw.get("h")), _f(raw.get("l")), _f(raw.get("c"))
        if t_open is None or o is None or h is None or l is None or c is None:
            continue
        t_close = t_open + bar_seconds
        if t_close > now_epoch:
            continue  # still forming (or in the future) -- never a partial bar
        open_dt = datetime.fromtimestamp(t_open, tz=timezone.utc).astimezone(_ET)
        day = open_dt.date()
        sess_open = session_open_et(day, holidays)
        sess_close = session_close_et(day, holidays, early_close)
        if sess_open is None or sess_close is None:
            continue  # weekend/holiday -- no session at all
        if not (sess_open <= open_dt < sess_close):
            continue  # pre-market/after-hours/outside today's (possibly early) close
        v = _f(raw.get("v")) or 0.0
        vw = _f(raw.get("vw"))
        amount = compute_amount(v, vw, c)
        by_open[t_open] = Bar(t_open=t_open, t_close=t_close, o=o, h=h, l=l, c=c, v=v, amount=amount)
    return [by_open[k] for k in sorted(by_open.keys())]


class BarsClient:
    """Duck-typed interface `agent.kronos.service` depends on. Tests inject
    a fake implementing this one async method; `AlpacaBarsClient` below is
    the production adapter (same pattern as `agent.flowcollector.collector.
    FlowBroker`/`AlpacaFlowBroker`)."""

    async def get_bars(self, symbol: str, lookback_bars: int, now: Optional[datetime] = None) -> Optional[list[Bar]]:
        raise NotImplementedError


class AlpacaBarsClient(BarsClient):
    """Production I/O adapter — the only place in this module that
    imports/constructs alpaca-py clients."""

    def __init__(self, cfg: KronosConfig, client: Any = None) -> None:
        self._cfg = cfg
        # Injectable (tests only, mirrors agent.dpof.broker.DPOFBroker's
        # constructor pattern) -- when set, _get_client() never constructs a
        # real StockHistoricalDataClient.
        self._client: Any = client
        self._feed_downgraded = False
        self._feed_downgrade_reason = ""

    def _get_client(self):
        if self._client is None:
            from alpaca.data.historical.stock import StockHistoricalDataClient

            self._client = StockHistoricalDataClient(
                api_key=self._cfg.alpaca_key_id, secret_key=self._cfg.alpaca_secret_key,
            )
        return self._client

    def _feed(self):
        from alpaca.data.enums import DataFeed

        if self._feed_downgraded:
            return DataFeed.IEX
        return DataFeed.SIP if self._cfg.stock_feed == "sip" else DataFeed.IEX

    def feed_status(self) -> dict:
        return {
            "configured": self._cfg.stock_feed,
            "effective": "iex" if self._feed_downgraded else self._cfg.stock_feed,
            "downgraded": self._feed_downgraded,
            "downgrade_reason": self._feed_downgrade_reason,
        }

    @staticmethod
    def _is_feed_subscription_error(e: Exception) -> bool:
        status = getattr(e, "status_code", None)
        if status is None:
            resp = getattr(e, "response", None)
            status = getattr(resp, "status_code", None) if resp is not None else None
        return status in (401, 403)

    async def get_bars(self, symbol: str, lookback_bars: int, now: Optional[datetime] = None) -> Optional[list[Bar]]:
        """Fetches enough calendar history to cover `lookback_bars` RTH
        5-minute bars (padded generously for weekends/holidays), filters to
        closed RTH bars only, and returns the newest `lookback_bars` of
        them (ascending). None on a fetch failure (after one feed-downgrade
        retry) -- the caller treats that as "no data this cycle", never a
        crash."""
        import asyncio

        loop = asyncio.get_running_loop()
        now_dt = now if now is not None else datetime.now(timezone.utc)
        now_epoch = now_dt.timestamp()
        # lookback_bars 5-min RTH bars span lookback_bars/_BARS_PER_SESSION
        # sessions; pad generously (weekends/holidays roughly double a
        # calendar-day count, plus a flat 5-day floor) so a fetch never comes
        # up short of real trading days.
        sessions_needed = (lookback_bars / float(_BARS_PER_SESSION)) + 2.0
        start_utc = now_dt - timedelta(days=sessions_needed * 2.0 + 5.0)

        async def _attempt():
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

            client = self._get_client()
            req = StockBarsRequest(
                symbol_or_symbols=symbol, timeframe=TimeFrame(BAR_MINUTES, TimeFrameUnit.Minute),
                start=start_utc, end=now_dt, feed=self._feed(),
            )

            def _call():
                return client.get_stock_bars(req)

            return await loop.run_in_executor(None, _call)

        try:
            barset = await _attempt()
        except Exception as e:
            if self._is_feed_subscription_error(e) and not self._feed_downgraded:
                status = getattr(e, "status_code", None)
                self._feed_downgraded = True
                self._feed_downgrade_reason = f"http_{status}" if status is not None else "subscription_error"
                logger.warning(
                    f"kronos bars: stock feed {self._cfg.stock_feed!r} rejected "
                    f"({self._feed_downgrade_reason}) -- downgrading to 'iex' for the "
                    f"rest of this process."
                )
                try:
                    barset = await _attempt()
                except Exception as e2:
                    logger.error(f"kronos bars: get_bars({symbol}) failed after downgrade: {type(e2).__name__}: {e2}")
                    return None
            else:
                logger.error(f"kronos bars: get_bars({symbol}) failed: {type(e).__name__}: {e}")
                return None

        try:
            bars = barset[symbol]
        except Exception as e:
            logger.debug(f"kronos bars: no bars for {symbol}: {e}")
            bars = []

        raw: list[dict] = []
        for b in bars or []:
            try:
                ts = getattr(b, "timestamp", None)
                t_epoch = ts.timestamp() if ts is not None else None
                raw.append({
                    "t": t_epoch, "o": getattr(b, "open", None), "h": getattr(b, "high", None),
                    "l": getattr(b, "low", None), "c": getattr(b, "close", None),
                    "v": getattr(b, "volume", None), "vw": getattr(b, "vwap", None),
                })
            except Exception as e:
                logger.debug(f"kronos bars: skipping malformed bar for {symbol}: {e}")

        holidays, early_close = _market_calendar()
        filtered = filter_rth_bars(
            raw, bar_minutes=BAR_MINUTES, now_epoch=now_epoch, holidays=holidays, early_close=early_close,
        )
        return filtered[-lookback_bars:] if len(filtered) > lookback_bars else filtered
