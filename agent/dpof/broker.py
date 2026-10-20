"""
CLIO DPOF — dedicated-account Alpaca adapter (I/O). THE SINGLE ORDER CHOKEPOINT.

Every order this engine ever places goes through :meth:`DPOFBroker.submit`.
Nothing here reads ``agent.config`` directly -- ``cfg.alpaca_key_id`` /
``cfg.alpaca_secret_key`` are resolved once, in
``agent.dpof.config.load_dpof_config()``, to either a dedicated DPOF Alpaca
account (preferred) or, only when the dedicated keys are unset AND the
account is paper (``cfg.alpaca_paper`` True), CLIO's existing shared PAPER
keys (``ALPACA_PAPER_KEY_ID``/``ALPACA_PAPER_SECRET_KEY``) as a same-source
fallback -- see ``cfg.alpaca_key_source`` ("dedicated" | "clio_paper" |
"none"). LIVE never falls back: it always requires its own dedicated keys.

Mode ladder (``cfg.mode``):
  off / log_only  -> submit() never touches the trading client at all.
  paper           -> requires cfg.alpaca_paper True + a reachable, sufficiently
                     provisioned paper account.
  live            -> requires every gate in design doc §7. Any single failing
                     gate collapses `effective_mode()` to LOG_ONLY.

PDT is dead (Alpaca removed pattern_day_trader/daytrade_count/
daytrading_buying_power from the API by 2026-07-06; see recon R4). This
module never reads those fields.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime
from typing import Any, Optional

from agent.dpof.config import DPOFConfig, DpofMode
from agent.dpof.models import GateResult, OptionQuote, OrderIntent, OrderResult

logger = logging.getLogger("clio.dpof.broker")

_GATE_CACHE_TTL_SECONDS = 300.0
# Alpaca's option snapshot endpoint rejects more than 100 symbols per request
# ({"message":"symbol limit is 100"}); a SPY/QQQ chain is routinely several hundred
# contracts. Every multi-symbol snapshot call goes through _get_option_snapshots.
_OPTION_SNAPSHOT_CHUNK = 100
_MAX_RETRY_ATTEMPTS = 3
_RETRY_BASE_DELAY = 1.0

_OCC_RE = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")

_TERMINAL_DEAD_STATUSES = ("canceled", "expired", "rejected")


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_float_or_none(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _safe_int(val: Any, default: int = 0) -> int:
    if val is None:
        return default
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default


def _enum_val(x: Any) -> str:
    if x is None:
        return ""
    return x.value if hasattr(x, "value") else str(x)


def _parse_occ(symbol: str) -> Optional[dict]:
    """Minimal local OCC parser -- deliberately NOT imported from
    agent.integrations.alpaca_options, which pulls in agent.config."""
    m = _OCC_RE.match(symbol)
    if not m:
        return None
    underlying, date_str, type_char, strike_str = m.groups()
    try:
        expiration = datetime.strptime(date_str, "%y%m%d").strftime("%Y-%m-%d")
    except ValueError:
        return None
    return {
        "underlying": underlying,
        "expiration": expiration,
        "type": "call" if type_char == "C" else "put",
        "strike": int(strike_str) / 1000.0,
    }


class BrokerLookupError(Exception):
    """The broker could not tell us an order's state (network/5xx/auth/parse error).

    Deliberately distinct from "not found" (get_order_by_client_id -> None). The
    OrderManager treats a missing order as "never reached the broker" and may
    RESEND; doing that on a mere lookup failure could put a second live order in
    the market (double exit) or orphan a live entry. On this error the caller
    must change nothing and retry next cycle."""


def _is_not_found(e: Exception) -> bool:
    status = getattr(e, "status_code", None)
    if status == 404:
        return True
    msg = str(e).lower()
    return "not found" in msg or "order not found" in msg


def _is_transient(e: Exception) -> bool:
    name = type(e).__name__
    return any(kw in name for kw in ("Timeout", "Connection", "Reset", "BrokenPipe"))


def _is_feed_subscription_error(e: Exception) -> bool:
    """401 (bad/missing key on the data endpoint) or 403 (not entitled to this
    feed -- e.g. OPRA/SIP requested without the add-on subscription active).
    Same status_code lookup pattern as _is_not_found (alpaca-py's APIError
    exposes .status_code; a bare httpx/requests-style exception may too)."""
    status = getattr(e, "status_code", None)
    if status is None:
        resp = getattr(e, "response", None)
        status = getattr(resp, "status_code", None) if resp is not None else None
    return status in (401, 403)


class DPOFBroker:
    def __init__(
        self,
        cfg: DPOFConfig,
        trading_client: Any = None,
        stock_data_client: Any = None,
        option_data_client: Any = None,
    ):
        self._cfg = cfg
        # Track which clients were explicitly injected (tests) so
        # update_config() never silently swaps them out from under a test.
        self._injected_trading = trading_client is not None
        self._injected_stock = stock_data_client is not None
        self._injected_option = option_data_client is not None
        self._trading_client = trading_client
        self._stock_data_client = stock_data_client
        self._option_data_client = option_data_client

        self._last_gate_result: Optional[GateResult] = None
        self._gate_checked_at: Optional[float] = None
        self._gate_cache_mode: Optional[DpofMode] = None

        # Sticky feed downgrade (never re-attempts the premium feed once
        # downgraded, for the life of this instance -- see
        # _maybe_downgrade_options_feed/_maybe_downgrade_stock_feed and
        # feed_status()). "Never go blind": a 401/403 subscription error on
        # OPRA/SIP falls back to indicative/IEX rather than returning no
        # data for the rest of the session.
        self._options_feed_downgraded = False
        self._options_feed_downgrade_reason = ""
        self._stock_feed_downgraded = False
        self._stock_feed_downgrade_reason = ""

    # ------------------------------------------------------------------
    # Config / client lifecycle
    # ------------------------------------------------------------------

    def update_config(self, cfg: DPOFConfig) -> None:
        old_key = (self._cfg.alpaca_key_id, self._cfg.alpaca_secret_key, self._cfg.alpaca_paper)
        new_key = (cfg.alpaca_key_id, cfg.alpaca_secret_key, cfg.alpaca_paper)
        old_options_feed = self._cfg.options_feed
        old_stock_feed = self._cfg.stock_feed
        self._cfg = cfg
        if new_key != old_key:
            if not self._injected_trading:
                self._trading_client = None
            if not self._injected_stock:
                self._stock_data_client = None
            if not self._injected_option:
                self._option_data_client = None
            # Keys/paper changed underneath us -- a cached gate verdict is
            # now meaningless.
            self._last_gate_result = None
            self._gate_checked_at = None
            self._gate_cache_mode = None
        # The operator explicitly re-pointed the feed config (an actual .env
        # edit + restart, or a fresh config object at startup) -- the sticky
        # downgrade from a PRIOR feed choice no longer applies. A cfg that
        # still asks for the same feed keeps its downgrade (never re-attempts
        # the premium feed just because update_config() was called again).
        if cfg.options_feed != old_options_feed:
            self._options_feed_downgraded = False
            self._options_feed_downgrade_reason = ""
        if cfg.stock_feed != old_stock_feed:
            self._stock_feed_downgraded = False
            self._stock_feed_downgrade_reason = ""

    def _get_trading_client(self):
        if self._trading_client is None:
            from alpaca.trading.client import TradingClient

            self._trading_client = TradingClient(
                api_key=self._cfg.alpaca_key_id,
                secret_key=self._cfg.alpaca_secret_key,
                paper=self._cfg.alpaca_paper,
            )
        return self._trading_client

    def _get_stock_data_client(self):
        if self._stock_data_client is None:
            from alpaca.data.historical.stock import StockHistoricalDataClient

            self._stock_data_client = StockHistoricalDataClient(
                api_key=self._cfg.alpaca_key_id,
                secret_key=self._cfg.alpaca_secret_key,
            )
        return self._stock_data_client

    def _get_option_data_client(self):
        if self._option_data_client is None:
            from alpaca.data.historical.option import OptionHistoricalDataClient

            self._option_data_client = OptionHistoricalDataClient(
                api_key=self._cfg.alpaca_key_id,
                secret_key=self._cfg.alpaca_secret_key,
            )
        return self._option_data_client

    def _feed_for_stock(self):
        from alpaca.data.enums import DataFeed

        if self._stock_feed_downgraded:
            return DataFeed.IEX
        mapping = {
            "iex": DataFeed.IEX,
            "sip": DataFeed.SIP,
            "delayed_sip": DataFeed.DELAYED_SIP,
            "otc": DataFeed.OTC,
        }
        return mapping.get(self._cfg.stock_feed, DataFeed.IEX)

    def _feed_for_options(self):
        from alpaca.data.enums import OptionsFeed

        if self._options_feed_downgraded:
            return OptionsFeed.INDICATIVE
        return OptionsFeed.OPRA if self._cfg.options_feed == "opra" else OptionsFeed.INDICATIVE

    # ------------------------------------------------------------------
    # Feed fallback -- "never go blind": a 401/403 subscription error on the
    # configured premium feed (OPRA for options, SIP/delayed_sip for stock)
    # downgrades to the indicative/IEX feed for the rest of this process
    # (never re-attempts the premium feed on its own -- see update_config()
    # for the one case that resets it: the operator explicitly re-pointing
    # cfg.options_feed/stock_feed). Logs exactly ONE warning per downgrade,
    # not once per failed call.
    # ------------------------------------------------------------------

    def _maybe_downgrade_options_feed(self, e: Exception) -> bool:
        """True if `e` triggered (or already triggered) the options downgrade
        and the caller should retry with the now-indicative feed."""
        if self._options_feed_downgraded:
            return False  # already downgraded this process; nothing new to retry
        if self._cfg.options_feed != "opra" or not _is_feed_subscription_error(e):
            return False
        status = getattr(e, "status_code", None)
        self._options_feed_downgraded = True
        self._options_feed_downgrade_reason = f"http_{status}" if status is not None else "subscription_error"
        logger.warning(
            f"DPOF broker: options feed 'opra' rejected ({self._options_feed_downgrade_reason}) -- "
            f"downgrading to 'indicative' for the rest of this process (option quotes/greeks will be "
            f"delayed/indicative from here on, never blind)."
        )
        return True

    def _maybe_downgrade_stock_feed(self, e: Exception) -> bool:
        """True if `e` triggered (or already triggered) the stock-feed downgrade
        and the caller should retry with the now-IEX feed."""
        if self._stock_feed_downgraded:
            return False
        if self._cfg.stock_feed not in ("sip", "delayed_sip") or not _is_feed_subscription_error(e):
            return False
        status = getattr(e, "status_code", None)
        self._stock_feed_downgraded = True
        self._stock_feed_downgrade_reason = f"http_{status}" if status is not None else "subscription_error"
        logger.warning(
            f"DPOF broker: stock feed '{self._cfg.stock_feed}' rejected ({self._stock_feed_downgrade_reason}) -- "
            f"downgrading to 'iex' for the rest of this process (bars/VWAP will be IEX-only from here on, "
            f"never blind)."
        )
        return True

    async def _with_feed_fallback(self, *, is_options: bool, attempt):
        """Run `attempt()` (an async, zero-arg callable using the CURRENT
        self._feed_for_options()/_feed_for_stock() choice); on a 401/403
        subscription error, downgrade the relevant feed and retry `attempt()`
        exactly once with the now-downgraded feed. Any other exception, or a
        retry that fails again, propagates to the caller unchanged."""
        try:
            return await attempt()
        except Exception as e:
            downgraded = (
                self._maybe_downgrade_options_feed(e) if is_options else self._maybe_downgrade_stock_feed(e)
            )
            if not downgraded:
                raise
            return await attempt()

    def feed_status(self) -> dict:
        """Loggable/reportable view of the live feed state -- status
        visibility for the sticky downgrade (never touches the network)."""
        return {
            "options_feed_configured": self._cfg.options_feed,
            "options_feed_effective": "indicative" if self._options_feed_downgraded else self._cfg.options_feed,
            "options_feed_downgraded": self._options_feed_downgraded,
            "options_feed_downgrade_reason": self._options_feed_downgrade_reason,
            "stock_feed_configured": self._cfg.stock_feed,
            "stock_feed_effective": "iex" if self._stock_feed_downgraded else self._cfg.stock_feed,
            "stock_feed_downgraded": self._stock_feed_downgraded,
            "stock_feed_downgrade_reason": self._stock_feed_downgrade_reason,
        }

    # ------------------------------------------------------------------
    # Retry + executor plumbing
    # ------------------------------------------------------------------

    async def _call(self, func, *args, idempotent: bool = True, **kwargs):
        """run_in_executor + bounded retry. Mutating calls (submit/replace) MUST pass
        idempotent=False: a timed-out request may already have reached the broker, and
        a blind resend is a second order. Those get one attempt; the OrderManager
        recovers the truth via get_order_by_client_id on the pre-recorded client id."""
        loop = asyncio.get_running_loop()
        last_exc: Optional[Exception] = None
        max_attempts = _MAX_RETRY_ATTEMPTS if idempotent else 1
        for attempt in range(1, max_attempts + 1):
            try:
                return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
            except Exception as e:
                status = getattr(e, "status_code", None)
                transient = status in (429, 500, 502, 503, 504) or _is_transient(e)
                if transient and attempt < max_attempts:
                    delay = min(_RETRY_BASE_DELAY * (2 ** (attempt - 1)) * 2.0, 30.0)
                    logger.warning(
                        f"DPOF broker retry {attempt}/{max_attempts}: {type(e).__name__}: {e}; sleeping {delay:.1f}s"
                    )
                    last_exc = e
                    await asyncio.sleep(delay)
                    continue
                raise
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("DPOF broker: retry exhausted without an exception")

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    async def get_account(self) -> Optional[dict]:
        try:
            client = self._get_trading_client()
            acct = await self._call(client.get_account)
        except Exception as e:
            logger.error(f"DPOF broker: get_account failed: {e}")
            return None
        try:
            level = getattr(acct, "options_trading_level", None)
            if level is None:
                level = getattr(acct, "options_approved_level", None)
            return {
                "account_number": str(getattr(acct, "account_number", "") or ""),
                "equity": _safe_float_or_none(getattr(acct, "equity", None)),
                "last_equity": _safe_float_or_none(getattr(acct, "last_equity", None)),
                "buying_power": _safe_float_or_none(getattr(acct, "buying_power", None)),
                "options_buying_power": _safe_float_or_none(getattr(acct, "options_buying_power", None)),
                "options_trading_level": _safe_int(level, 0),
                "status": _enum_val(getattr(acct, "status", None)),
                "trading_blocked": bool(getattr(acct, "trading_blocked", False)),
                "account_blocked": bool(getattr(acct, "account_blocked", False)),
            }
        except Exception as e:
            logger.error(f"DPOF broker: get_account parse failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    async def get_minute_bars(self, symbol: str, start_utc: datetime) -> Optional[list[dict]]:
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

            client = self._get_stock_data_client()

            async def _attempt():
                req = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                    start=start_utc,
                    feed=self._feed_for_stock(),
                )
                return await self._call(client.get_stock_bars, req)

            barset = await self._with_feed_fallback(is_options=False, attempt=_attempt)
        except Exception as e:
            logger.error(f"DPOF broker: get_minute_bars failed for {symbol}: {e}")
            return None

        try:
            bars = barset[symbol]
        except Exception as e:
            logger.debug(f"DPOF broker: no bars for {symbol}: {e}")
            bars = []

        out: list[dict] = []
        for b in bars or []:
            try:
                ts = getattr(b, "timestamp", None)
                epoch = ts.timestamp() if ts is not None else None
                out.append(
                    {
                        "t": epoch,
                        "o": _safe_float(getattr(b, "open", None)),
                        "h": _safe_float(getattr(b, "high", None)),
                        "l": _safe_float(getattr(b, "low", None)),
                        "c": _safe_float(getattr(b, "close", None)),
                        "v": _safe_float(getattr(b, "volume", None)),
                        "vw": _safe_float_or_none(getattr(b, "vwap", None)),
                    }
                )
            except Exception as e:
                logger.debug(f"DPOF broker: skipping malformed bar for {symbol}: {e}")
        out.sort(key=lambda x: (x["t"] is None, x["t"]))
        return out

    async def get_option_chain(
        self, underlying: str, exp_min, exp_max, right: str
    ) -> Optional[list[OptionQuote]]:
        try:
            from alpaca.trading.enums import AssetStatus, ContractType
            from alpaca.trading.requests import GetOptionContractsRequest

            client = self._get_trading_client()
            contract_type = ContractType.CALL if right == "call" else ContractType.PUT
            contracts: list = []
            page_token = None
            while True:
                req = GetOptionContractsRequest(
                    underlying_symbols=[underlying],
                    status=AssetStatus.ACTIVE,
                    expiration_date_gte=exp_min,
                    expiration_date_lte=exp_max,
                    type=contract_type,
                    page_token=page_token,
                )
                resp = await self._call(client.get_option_contracts, req)
                batch = getattr(resp, "option_contracts", None) or []
                contracts.extend(batch)
                page_token = getattr(resp, "next_page_token", None)
                if not page_token or not batch or len(contracts) > 5000:
                    break
        except Exception as e:
            logger.error(f"DPOF broker: get_option_chain contracts failed for {underlying}: {e}")
            return None

        if not contracts:
            return []

        symbols = [c.symbol for c in contracts]
        try:
            snapshots = await self._get_option_snapshots(symbols, f"get_option_chain {underlying}")
        except Exception as e:
            logger.error(f"DPOF broker: get_option_chain snapshots failed for {underlying}: {e}")
            snapshots = None
        if snapshots is None:
            # Unavailable, not illiquid: a chain of quote-less contracts reads as all-illiquid
            # and yields "no_candidates" (every entry blocked, 2026-09-24..28).
            return None

        out: list[OptionQuote] = []
        for c in contracts:
            try:
                snap = snapshots.get(c.symbol) if isinstance(snapshots, dict) else None
                q = getattr(snap, "latest_quote", None) if snap else None
                greeks = getattr(snap, "greeks", None) if snap else None
                out.append(
                    OptionQuote(
                        symbol=c.symbol,
                        underlying=underlying,
                        expiration=str(getattr(c, "expiration_date", "")),
                        strike=_safe_float(getattr(c, "strike_price", None), 0.0),
                        right=right,
                        bid=_safe_float_or_none(getattr(q, "bid_price", None)) if q else None,
                        ask=_safe_float_or_none(getattr(q, "ask_price", None)) if q else None,
                        delta=_safe_float_or_none(getattr(greeks, "delta", None)) if greeks else None,
                        theta=_safe_float_or_none(getattr(greeks, "theta", None)) if greeks else None,
                        iv=_safe_float_or_none(getattr(snap, "implied_volatility", None)) if snap else None,
                        open_interest=_safe_int(getattr(c, "open_interest", None), None)
                        if getattr(c, "open_interest", None) is not None
                        else None,
                        volume=None,
                    )
                )
            except Exception as e:
                logger.debug(f"DPOF broker: skipping malformed contract {getattr(c, 'symbol', '?')}: {e}")
        return out

    async def _get_option_snapshots(self, symbols: list[str], label: str) -> Optional[dict]:
        """Option snapshots for `symbols`, requested in chunks of at most
        _OPTION_SNAPSHOT_CHUNK (Alpaca's per-request limit). Each chunk gets the
        usual feed-downgrade retry. If any chunk still fails the whole lookup is
        unavailable (None): a partial result would make the missing symbols look
        absent, and a lookup failure is never "not found"."""
        from alpaca.data.requests import OptionSnapshotRequest

        opt_client = self._get_option_data_client()
        merged: dict = {}
        for start in range(0, len(symbols), _OPTION_SNAPSHOT_CHUNK):
            chunk = symbols[start:start + _OPTION_SNAPSHOT_CHUNK]

            async def _attempt(chunk=chunk):
                snap_req = OptionSnapshotRequest(symbol_or_symbols=chunk, feed=self._feed_for_options())
                return await self._call(opt_client.get_option_snapshot, snap_req)

            try:
                snaps = await self._with_feed_fallback(is_options=True, attempt=_attempt)
            except Exception as e:
                logger.error(
                    f"DPOF broker: {label}: option snapshots failed for symbols {start}-{start + len(chunk) - 1} "
                    f"of {len(symbols)}: {e}"
                )
                return None
            if isinstance(snaps, dict):
                merged.update(snaps)
        return merged

    async def get_option_quotes(self, symbols: list[str]) -> Optional[dict[str, OptionQuote]]:
        if not symbols:
            return {}
        try:
            snapshots = await self._get_option_snapshots(list(symbols), "get_option_quotes")
        except Exception as e:
            logger.error(f"DPOF broker: get_option_quotes failed: {e}")
            return None
        if snapshots is None:
            return None

        out: dict[str, OptionQuote] = {}
        for sym in symbols:
            snap = snapshots.get(sym) if isinstance(snapshots, dict) else None
            if snap is None:
                continue
            parsed = _parse_occ(sym)
            q = getattr(snap, "latest_quote", None)
            greeks = getattr(snap, "greeks", None)
            out[sym] = OptionQuote(
                symbol=sym,
                underlying=parsed["underlying"] if parsed else "",
                expiration=parsed["expiration"] if parsed else "",
                strike=parsed["strike"] if parsed else 0.0,
                right=parsed["type"] if parsed else "",
                bid=_safe_float_or_none(getattr(q, "bid_price", None)) if q else None,
                ask=_safe_float_or_none(getattr(q, "ask_price", None)) if q else None,
                delta=_safe_float_or_none(getattr(greeks, "delta", None)) if greeks else None,
                theta=_safe_float_or_none(getattr(greeks, "theta", None)) if greeks else None,
                iv=_safe_float_or_none(getattr(snap, "implied_volatility", None)),
                open_interest=None,
                volume=None,
            )
        return out

    # ------------------------------------------------------------------
    # Gates
    # ------------------------------------------------------------------

    async def verify_gates(self, now_epoch: float) -> GateResult:
        mode = self._cfg.mode
        if mode in (DpofMode.OFF, DpofMode.LOG_ONLY):
            result = GateResult(allowed=True, reasons=[])
            self._last_gate_result = result
            self._gate_checked_at = now_epoch
            self._gate_cache_mode = mode
            return result

        if (
            self._last_gate_result is not None
            and self._gate_checked_at is not None
            and self._gate_cache_mode == mode
            and (now_epoch - self._gate_checked_at) < _GATE_CACHE_TTL_SECONDS
        ):
            return self._last_gate_result

        if mode == DpofMode.PAPER:
            result = await self._verify_paper_gates()
        elif mode == DpofMode.LIVE:
            result = await self._verify_live_gates()
        else:
            result = GateResult(allowed=False, reasons=[f"unknown_mode:{mode}"])

        self._last_gate_result = result
        self._gate_checked_at = now_epoch
        self._gate_cache_mode = mode
        return result

    async def _verify_paper_gates(self) -> GateResult:
        reasons: list[str] = []
        if not self._cfg.alpaca_paper:
            reasons.append("paper_flag_false")
        if not (self._cfg.alpaca_key_id and self._cfg.alpaca_secret_key):
            reasons.append("keys_not_set")

        account = await self.get_account()
        if account is None:
            reasons.append("account_fetch_failed")
        else:
            # NOT "3 if allow_verticals else min_options_level" -- that would
            # collapse the WHOLE mode to LOG_ONLY (blocking even a plain long
            # call/put) on a level-2 account just because allow_verticals
            # defaults to True. Verticals are gated separately, per-cycle,
            # by engine.py's own runtime check (options_level >= 3 ->
            # verticals offered; otherwise singles only, no error — see
            # contracts.py's module docstring and DPOF_V2_SPEC.md A2:
            # "offered only when the account's options level >= 3", not
            # "trading only allowed when >= 3"). This gate's only job is the
            # account's BASIC ability to trade the configured strategy at
            # all (min_options_level, default 2 = long calls/puts).
            min_level = self._cfg.min_options_level
            level = account.get("options_trading_level") or 0
            if level < min_level:
                reasons.append(f"options_level_too_low:{level}<{min_level}")

        return GateResult(allowed=(len(reasons) == 0), reasons=reasons)

    async def _verify_live_gates(self) -> GateResult:
        reasons: list[str] = []
        if self._cfg.alpaca_paper:
            reasons.append("live_requires_paper_false")
        if not self._cfg.live_ack_ok():
            reasons.append("live_ack_missing")
        if not (self._cfg.alpaca_key_id and self._cfg.alpaca_secret_key):
            reasons.append("keys_not_set")

        try:
            from agent.safety.kill_registry import get_kill_registry

            kill_ok = get_kill_registry().is_enabled("live_trading")
        except Exception as e:
            logger.error(f"DPOF broker: kill registry check failed: {e}")
            kill_ok = False
        if not kill_ok:
            reasons.append("kill_switch_live_trading_disabled")

        account = await self.get_account()
        if account is None:
            reasons.append("account_fetch_failed")
        else:
            if not self._cfg.live_account_number:
                reasons.append("live_account_not_configured")
            elif account.get("account_number") != self._cfg.live_account_number:
                reasons.append("account_number_mismatch")

            # See the identical comment in _verify_paper_gates -- verticals
            # are gated separately (engine.py, per-cycle), never by blocking
            # the whole mode here.
            min_level = self._cfg.min_options_level
            level = account.get("options_trading_level") or 0
            if level < min_level:
                reasons.append(f"options_level_too_low:{level}<{min_level}")

            equity = account.get("equity")
            if equity is None or equity < self._cfg.min_account_equity:
                reasons.append("equity_below_min")

            if account.get("trading_blocked"):
                reasons.append("trading_blocked")
            if account.get("account_blocked"):
                reasons.append("account_blocked")

        return GateResult(allowed=(len(reasons) == 0), reasons=reasons)

    def effective_mode(self) -> DpofMode:
        mode = self._cfg.mode
        if mode in (DpofMode.OFF, DpofMode.LOG_ONLY):
            return mode
        if self._last_gate_result is None or not self._last_gate_result.allowed:
            return DpofMode.LOG_ONLY
        if self._gate_cache_mode != mode:
            # Gates were verified for a different mode than the one now
            # configured -- fail safe rather than trust a stale verdict.
            return DpofMode.LOG_ONLY
        return mode

    # ------------------------------------------------------------------
    # Orders -- the chokepoint
    # ------------------------------------------------------------------

    async def submit(self, intent: OrderIntent) -> OrderResult:
        if not intent.client_order_id.startswith(self._cfg.client_order_prefix):
            logger.error(
                f"DPOF broker: refusing to submit order without the DPOF prefix: {intent.client_order_id!r}"
            )
            return OrderResult(ok=False, client_order_id=intent.client_order_id, error="client_order_id_missing_prefix")

        is_entry = intent.purpose == "entry"
        if is_entry:
            # Entries need every gate (design §7): effective_mode() collapses to
            # LOG_ONLY on any failed/unverified gate.
            mode = self.effective_mode()
            send = mode in (DpofMode.PAPER, DpofMode.LIVE)
        else:
            # Exits are risk-REDUCING. A position opened in paper/live must be
            # closable even while entry gates fail (a transient account fetch, the
            # live_trading kill switch, a mode edit + restart mid-session) —
            # otherwise a gate hiccup at 15:50 traps it overnight. Only requirement:
            # the configured account type still matches the one it was opened in.
            pmode = intent.position_mode
            real = pmode in (DpofMode.PAPER.value, DpofMode.LIVE.value)
            account_matches = self._cfg.alpaca_paper == (pmode == DpofMode.PAPER.value)
            keys = bool(self._cfg.alpaca_key_id and self._cfg.alpaca_secret_key)
            send = real and account_matches and keys
            mode = DpofMode(pmode) if real else self.effective_mode()
            if real and not send:
                logger.error(
                    f"DPOF broker: cannot send exit {intent.client_order_id} for a {pmode} position "
                    f"(account_matches={account_matches}, keys={keys}) — position needs manual attention"
                )
        if not send:
            logger.info(f"DPOF broker: {mode.value} mode -- would submit {intent.client_order_id}, sending nothing")
            return OrderResult(ok=True, client_order_id=intent.client_order_id, would_submit=True)

        try:
            from agent.safety.audit_log import audit_tool_execution

            audit_tool_execution(
                "dpof_submit_order",
                {
                    "client_order_id": intent.client_order_id,
                    "purpose": intent.purpose,
                    "qty": intent.qty,
                    "limit_price": intent.limit_price,
                    "reason": intent.reason,
                    "position_id": intent.position_id,
                    "legs": [{"symbol": l.symbol, "side": l.side, "position_intent": l.position_intent} for l in intent.legs],
                },
                user_id="dpof_engine",
            )
        except Exception as e:
            logger.warning(f"DPOF broker: audit hook failed for {intent.client_order_id}: {e}")
            if is_entry:
                return OrderResult(ok=False, client_order_id=intent.client_order_id, error="audit_hook_failed")
            # Exits never get blocked by an audit-plumbing failure -- a
            # risk-reducing trade must still go out.

        try:
            request = self._build_order_request(intent)
        except Exception as e:
            logger.error(f"DPOF broker: failed to build order request for {intent.client_order_id}: {e}")
            return OrderResult(ok=False, client_order_id=intent.client_order_id, error=f"request_build_failed:{e}")

        try:
            client = self._get_trading_client()
            order = await self._call(client.submit_order, request, idempotent=False)
        except Exception as e:
            logger.error(f"DPOF broker: submit_order failed for {intent.client_order_id}: {e}")
            return OrderResult(ok=False, client_order_id=intent.client_order_id, error=str(e))

        try:
            return self._order_result_from_sdk(order)
        except Exception as e:
            logger.error(f"DPOF broker: failed to parse submitted order for {intent.client_order_id}: {e}")
            return OrderResult(ok=True, client_order_id=intent.client_order_id, error=f"parse_failed:{e}")

    def _build_order_request(self, intent: OrderIntent):
        from alpaca.trading.enums import OrderClass, OrderSide, PositionIntent, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest, OptionLegRequest

        if not intent.legs:
            raise ValueError("OrderIntent has no legs")

        tif = TimeInForce.GTC if intent.tif == "gtc" else TimeInForce.DAY

        if len(intent.legs) > 1:
            if intent.limit_price is None:
                raise ValueError("multi-leg orders require a limit price (mleg has no market order)")
            # Alpaca mleg sign convention (recon R4, docs.alpaca.markets): positive
            # limit_price = net DEBIT, negative = net CREDIT. Opening a debit vertical
            # pays a debit (+); closing it receives a credit (-). Sending a positive
            # price on a close would bid to PAY for the exit and forfeit price
            # protection. Verify in paper before enabling verticals live.
            mleg_price = abs(intent.limit_price)
            if intent.purpose == "exit":
                mleg_price = -mleg_price
            legs = [
                OptionLegRequest(
                    symbol=leg.symbol,
                    ratio_qty=leg.ratio_qty,
                    side=OrderSide.BUY if leg.side == "buy" else OrderSide.SELL,
                    position_intent=PositionIntent(leg.position_intent),
                )
                for leg in intent.legs
            ]
            return LimitOrderRequest(
                qty=intent.qty,
                order_class=OrderClass.MLEG,
                time_in_force=tif,
                limit_price=mleg_price,
                legs=legs,
                client_order_id=intent.client_order_id,
            )

        leg = intent.legs[0]
        side = OrderSide.BUY if leg.side == "buy" else OrderSide.SELL
        pos_intent = PositionIntent(leg.position_intent) if leg.position_intent else None
        if intent.limit_price is None:
            return MarketOrderRequest(
                symbol=leg.symbol,
                qty=intent.qty,
                side=side,
                time_in_force=tif,
                client_order_id=intent.client_order_id,
                position_intent=pos_intent,
            )
        return LimitOrderRequest(
            symbol=leg.symbol,
            qty=intent.qty,
            side=side,
            time_in_force=tif,
            limit_price=intent.limit_price,
            client_order_id=intent.client_order_id,
            position_intent=pos_intent,
        )

    def _order_result_from_sdk(self, order: Any) -> OrderResult:
        return OrderResult(
            ok=True,
            broker_order_id=str(getattr(order, "id", "") or ""),
            client_order_id=str(getattr(order, "client_order_id", "") or ""),
            status=_enum_val(getattr(order, "status", None)),
            filled_qty=_safe_int(getattr(order, "filled_qty", None), 0),
            filled_avg_price=_safe_float_or_none(getattr(order, "filled_avg_price", None)),
        )

    def _order_dict_from_sdk(self, order: Any) -> dict:
        legs = getattr(order, "legs", None) or []
        leg_dicts = []
        for l in legs:
            leg_dicts.append(
                {
                    "symbol": str(getattr(l, "symbol", "")),
                    "side": _enum_val(getattr(l, "side", None)),
                    "filled_qty": _safe_int(getattr(l, "filled_qty", None), 0),
                }
            )
        return {
            "id": str(getattr(order, "id", "") or ""),
            "status": _enum_val(getattr(order, "status", None)),
            "filled_qty": _safe_int(getattr(order, "filled_qty", None), 0),
            "filled_avg_price": _safe_float_or_none(getattr(order, "filled_avg_price", None)),
            "limit_price": _safe_float_or_none(getattr(order, "limit_price", None)),
            "legs": leg_dicts,
        }

    async def get_order_by_client_id(self, client_order_id: str) -> Optional[dict]:
        """dict if found; None ONLY if the broker affirmatively reports no such order
        (404). Any other failure raises BrokerLookupError — never conflate "I couldn't
        ask" with "it doesn't exist" (see BrokerLookupError)."""
        try:
            client = self._get_trading_client()
            order = await self._call(client.get_order_by_client_id, client_order_id)
        except Exception as e:
            if _is_not_found(e):
                logger.info(f"DPOF broker: order {client_order_id} not found at broker")
                return None
            logger.error(f"DPOF broker: get_order_by_client_id failed for {client_order_id}: {e}")
            raise BrokerLookupError(f"lookup_failed:{type(e).__name__}") from e
        try:
            return self._order_dict_from_sdk(order)
        except Exception as e:
            logger.error(f"DPOF broker: failed to parse order for {client_order_id}: {e}")
            raise BrokerLookupError(f"parse_failed:{type(e).__name__}") from e

    async def cancel_order(self, broker_order_id: str, client_order_id: str) -> bool:
        if not client_order_id.startswith(self._cfg.client_order_prefix):
            logger.error(f"DPOF broker: refusing to cancel non-DPOF order {client_order_id!r}")
            return False
        try:
            client = self._get_trading_client()
            await self._call(client.cancel_order_by_id, broker_order_id)
            return True
        except Exception as e:
            logger.error(f"DPOF broker: cancel_order failed for {broker_order_id}: {e}")
            return False

    # NOTE: there is deliberately no replace_order() here any more. Alpaca's
    # PATCH /v2/orders/{id} does NOT edit an order in place -- it creates a
    # NEW order (new id, and a new auto-generated client_order_id unless one
    # is supplied) and the OLD order transitions to status "replaced" with
    # replaced_by=<new id> (alpaca-py Order.replaces/replaced_by,
    # OrderStatus.REPLACED). A previous version of this method assumed
    # in-place PATCH semantics; using it from orders.py would have silently
    # orphaned the ledger's tracked client_order_id on every reprice. The
    # reprice/escalate protocol in orders.py now always CANCELS the stale
    # order, waits for it to confirm terminal via get_order_by_client_id,
    # and SENDS A NEW ORDER with a freshly generated (chain-appended)
    # client_order_id instead. Do not reintroduce a replace-in-place helper
    # without re-verifying Alpaca's actual PATCH semantics against a live
    # paper account first.

    async def list_open_orders(self) -> Optional[list[dict]]:
        try:
            from alpaca.trading.enums import QueryOrderStatus
            from alpaca.trading.requests import GetOrdersRequest

            client = self._get_trading_client()
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            orders = await self._call(client.get_orders, req)
        except Exception as e:
            logger.error(f"DPOF broker: list_open_orders failed: {e}")
            return None
        out = []
        for o in orders or []:
            try:
                out.append(self._order_dict_from_sdk(o))
            except Exception as e:
                logger.debug(f"DPOF broker: skipping malformed order: {e}")
        return out

    async def list_option_positions(self) -> Optional[list[dict]]:
        try:
            client = self._get_trading_client()
            positions = await self._call(client.get_all_positions)
        except Exception as e:
            logger.error(f"DPOF broker: list_option_positions failed: {e}")
            return None
        out = []
        for p in positions or []:
            if _enum_val(getattr(p, "asset_class", None)) != "us_option":
                continue
            try:
                out.append(
                    {
                        "symbol": str(getattr(p, "symbol", "")),
                        "qty": _safe_int(getattr(p, "qty", None), 0),
                        "avg_entry_price": _safe_float_or_none(getattr(p, "avg_entry_price", None)),
                        "market_value": _safe_float_or_none(getattr(p, "market_value", None)),
                        "unrealized_pl": _safe_float_or_none(getattr(p, "unrealized_pl", None)),
                    }
                )
            except Exception as e:
                logger.debug(f"DPOF broker: skipping malformed position: {e}")
        return out
