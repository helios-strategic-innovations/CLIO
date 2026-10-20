"""
OpenClaw/Moltbot — Real-Time Price Streaming
Persistent Alpaca WebSocket streams (stock + options) feeding an in-memory quote cache.
Consumed by Moltbot tools and Mission Control WebSocket.
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from agent.config import config

logger = logging.getLogger("openclaw.integrations.price_stream")

# --- Data Structures ---


@dataclass
class CachedQuote:
    """A cached price quote from the streaming feed."""
    symbol: str
    bid: float
    ask: float
    mid: float
    last_trade_price: float
    last_trade_size: int
    timestamp: float  # epoch seconds
    source: str  # "stock" or "option"


# Cache staleness threshold (seconds)
_CACHE_TTL = 60.0

# Reconnect backoff constants
_RECONNECT_BASE_DELAY = 2.0
_RECONNECT_MAX_DELAY = 60.0


class PriceStreamManager:
    """
    Singleton managing Alpaca WebSocket streams + in-memory quote cache.

    Architecture:
    - StockDataStream.run() is BLOCKING — runs in a stored executor future
    - OptionDataStream.run() — same pattern, separate stored future
    - Quote callbacks fire from the stream's internal thread → threading.Lock for cache
    - Auto-reconnect with exponential backoff on disconnect
    """

    def __init__(self):
        self._cache: dict[str, CachedQuote] = {}
        self._lock = threading.Lock()  # protects cache + callback list
        self._sub_lock = threading.Lock()  # protects subscription sets
        self._stock_task: Optional[asyncio.Task] = None
        self._option_task: Optional[asyncio.Task] = None
        self._stock_run_future: Optional[asyncio.Future] = None
        self._option_run_future: Optional[asyncio.Future] = None
        self._stock_subs: set[str] = set()
        self._option_subs: set[str] = set()
        self._callbacks: list[Callable] = []
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # Stream references for clean shutdown
        self._stock_stream: Any = None
        self._option_stream: Any = None

    # --- Lifecycle ---

    async def start(self) -> None:
        """Start stock + option streams as background tasks."""
        if self._running:
            logger.info("[PriceStream] Already running")
            return

        if not config.alpaca_api_key:
            logger.warning("[PriceStream] No Alpaca API key configured, skipping stream start")
            return

        self._running = True
        self._loop = asyncio.get_running_loop()

        # Auto-subscribe to current portfolio positions
        await self._auto_subscribe_portfolio()

        # Start stream threads
        self._stock_task = asyncio.create_task(self._run_stock_stream())
        self._option_task = asyncio.create_task(self._run_option_stream())
        logger.info("[PriceStream] Started stock + option streams")

    async def stop(self) -> None:
        """Graceful shutdown — close WebSocket connections then cancel tasks."""
        self._running = False
        # Close the blocking stream WebSockets first (this unblocks stream.run())
        # S151-B H-6: Add timeout to prevent hanging on unresponsive streams
        loop = asyncio.get_running_loop()
        run_futures = [
            ("stock", self._stock_run_future),
            ("option", self._option_run_future),
        ]
        for stream_ref in (self._stock_stream, self._option_stream):
            if stream_ref is not None:
                try:
                    await asyncio.wait_for(
                        loop.run_in_executor(None, stream_ref.stop),
                        timeout=10.0,
                    )
                except asyncio.TimeoutError:
                    logger.warning("[PriceStream] Stream stop timed out after 10s, continuing")
                except Exception as e:
                    logger.debug(f"[PriceStream] Stream stop error: {e}")
        for label, future in run_futures:
            if future and not future.done():
                try:
                    await asyncio.wait_for(asyncio.shield(future), timeout=15.0)
                    logger.info(f"[PriceStream] {label} stream runner exited")
                except asyncio.TimeoutError:
                    logger.warning(f"[PriceStream] {label} stream runner did not exit within 15s")
                except Exception as e:
                    logger.debug(f"[PriceStream] {label} stream runner wait error: {e}")
        self._stock_stream = None
        self._option_stream = None
        self._stock_run_future = None
        self._option_run_future = None
        # Cancel the wrapping asyncio tasks (with timeout per task)
        for task in (self._stock_task, self._option_task):
            if task:
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=10)
                except asyncio.TimeoutError:
                    logger.warning("S159 L-3: Stream stop timed out after 10s — continuing shutdown")
                except (asyncio.CancelledError, Exception):
                    pass
        self._stock_task = None
        self._option_task = None
        logger.info("[PriceStream] Stopped")

    @property
    def is_running(self) -> bool:
        """Check if streams are active."""
        return self._running and (self._stock_task is not None or self._option_task is not None)

    # --- Subscriptions ---

    async def subscribe_stock(self, symbol: str) -> None:
        """Add symbol to stock quote subscription."""
        symbol = symbol.upper()
        with self._sub_lock:
            if symbol in self._stock_subs:
                return
            self._stock_subs.add(symbol)
        # Try to subscribe on running stream (if supported)
        if self._stock_stream is not None:
            try:
                self._stock_stream.subscribe_quotes(self._make_stock_quote_handler(), symbol)
                self._stock_stream.subscribe_trades(self._make_stock_trade_handler(), symbol)
            except Exception as e:
                logger.debug(f"[PriceStream] Dynamic stock subscribe failed (will pick up on reconnect): {e}")
        logger.debug(f"[PriceStream] Subscribed stock: {symbol}")

    async def subscribe_option(self, occ_symbol: str) -> None:
        """Add OCC symbol to option quote subscription."""
        occ_symbol = occ_symbol.upper()
        with self._sub_lock:
            if occ_symbol in self._option_subs:
                return
            self._option_subs.add(occ_symbol)
        if self._option_stream is not None:
            try:
                self._option_stream.subscribe_quotes(self._make_option_quote_handler(), occ_symbol)
                self._option_stream.subscribe_trades(self._make_option_trade_handler(), occ_symbol)
            except Exception as e:
                logger.debug(f"[PriceStream] Dynamic option subscribe failed (will pick up on reconnect): {e}")
        logger.debug(f"[PriceStream] Subscribed option: {occ_symbol}")

    async def unsubscribe_stock(self, symbol: str) -> None:
        """Remove symbol from stock subscription."""
        with self._sub_lock:
            self._stock_subs.discard(symbol.upper())

    async def unsubscribe_option(self, occ_symbol: str) -> None:
        """Remove OCC symbol from option subscription."""
        with self._sub_lock:
            self._option_subs.discard(occ_symbol.upper())

    # --- Cache Access ---

    def get_quote(self, symbol: str) -> Optional[CachedQuote]:
        """Read from cache. Thread-safe. Returns None if stale (>60s)."""
        with self._lock:
            quote = self._cache.get(symbol)
            if quote and (time.time() - quote.timestamp) > _CACHE_TTL:
                return None  # stale
            return quote

    def get_all_quotes(self) -> dict[str, CachedQuote]:
        """Snapshot of entire non-stale cache. Thread-safe."""
        now = time.time()
        with self._lock:
            return {
                k: v for k, v in self._cache.items()
                if (now - v.timestamp) <= _CACHE_TTL
            }

    # --- Callbacks ---

    def register_callback(self, fn: Callable) -> None:
        """Register callback for real-time updates (used by Mission Control WS)."""
        with self._lock:
            self._callbacks.append(fn)

    def unregister_callback(self, fn: Callable) -> None:
        """Remove callback on WS disconnect. Thread-safe."""
        with self._lock:
            self._callbacks = [c for c in self._callbacks if c is not fn]

    def _fire_callbacks(self, symbol: str, quote: CachedQuote) -> None:
        """Notify all registered callbacks. Called from stream thread."""
        with self._lock:
            cbs = list(self._callbacks)
        for cb in cbs:
            try:
                cb(symbol, quote)
            except Exception as e:
                logger.debug(f"[PriceStream] Callback error: {e}")

    # --- Internal: Cache Update ---

    def _update_cache(self, symbol: str, bid: float, ask: float,
                       last_price: float, last_size: int, source: str) -> None:
        """Update cache entry. Called from stream thread."""
        mid = round((bid + ask) / 2, 4) if bid > 0 and ask > 0 else last_price
        quote = CachedQuote(
            symbol=symbol,
            bid=bid,
            ask=ask,
            mid=mid,
            last_trade_price=last_price,
            last_trade_size=last_size,
            timestamp=time.time(),
            source=source,
        )
        with self._lock:
            self._cache[symbol] = quote
        self._fire_callbacks(symbol, quote)

    # --- Internal: Stream Runners ---

    async def _run_stock_stream(self) -> None:
        """Run stock stream with auto-reconnect."""
        delay = _RECONNECT_BASE_DELAY
        while self._running:
            with self._sub_lock:
                _has_subs = bool(self._stock_subs)
            if not _has_subs:
                await asyncio.sleep(5)
                continue
            try:
                loop = asyncio.get_running_loop()
                future = loop.run_in_executor(None, self._stock_stream_thread)
                self._stock_run_future = future
                await asyncio.shield(future)
                delay = _RECONNECT_BASE_DELAY  # reset on clean exit
            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self._running:
                    break
                logger.warning(f"[PriceStream] Stock stream disconnected: {e}. Reconnecting in {delay:.0f}s")
                await asyncio.sleep(delay)
                delay = min(delay * 2, _RECONNECT_MAX_DELAY)
            finally:
                future = self._stock_run_future
                if future is not None and future.done():
                    self._stock_run_future = None

    async def _run_option_stream(self) -> None:
        """Run option stream with auto-reconnect."""
        delay = _RECONNECT_BASE_DELAY
        while self._running:
            with self._sub_lock:
                _has_subs = bool(self._option_subs)
            if not _has_subs:
                await asyncio.sleep(5)
                continue
            try:
                loop = asyncio.get_running_loop()
                future = loop.run_in_executor(None, self._option_stream_thread)
                self._option_run_future = future
                await asyncio.shield(future)
                delay = _RECONNECT_BASE_DELAY
            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self._running:
                    break
                logger.warning(f"[PriceStream] Option stream disconnected: {e}. Reconnecting in {delay:.0f}s")
                await asyncio.sleep(delay)
                delay = min(delay * 2, _RECONNECT_MAX_DELAY)
            finally:
                future = self._option_run_future
                if future is not None and future.done():
                    self._option_run_future = None

    # --- Handler factories (reusable for initial + dynamic subscriptions) ---

    def _make_stock_quote_handler(self):
        async def on_quote(data):
            try:
                symbol = str(data.symbol)
                bid = float(getattr(data, "bid_price", 0) or 0)
                ask = float(getattr(data, "ask_price", 0) or 0)
                last_price = round((bid + ask) / 2, 4) if bid > 0 and ask > 0 else 0.0
                self._update_cache(symbol, bid, ask, last_price, 0, "stock")
            except Exception as e:
                logger.debug(f"[PriceStream] Stock quote handler error: {e}")
        return on_quote

    def _make_stock_trade_handler(self):
        async def on_trade(data):
            try:
                symbol = str(data.symbol)
                price = float(getattr(data, "price", 0) or 0)
                size = int(getattr(data, "size", 0) or 0)
                # Full read-modify-write under single lock to avoid TOCTOU
                with self._lock:
                    existing = self._cache.get(symbol)
                    bid = existing.bid if existing else 0.0
                    ask = existing.ask if existing else 0.0
                    mid = round((bid + ask) / 2, 4) if bid > 0 and ask > 0 else price
                    self._cache[symbol] = CachedQuote(
                        symbol=symbol, bid=bid, ask=ask, mid=mid,
                        last_trade_price=price, last_trade_size=size,
                        timestamp=time.time(), source="stock",
                    )
                    quote = self._cache[symbol]
                self._fire_callbacks(symbol, quote)
            except Exception as e:
                logger.debug(f"[PriceStream] Stock trade handler error: {e}")
        return on_trade

    def _make_option_quote_handler(self):
        async def on_quote(data):
            try:
                symbol = str(data.symbol)
                bid = float(getattr(data, "bid_price", 0) or 0)
                ask = float(getattr(data, "ask_price", 0) or 0)
                last_price = round((bid + ask) / 2, 4) if bid > 0 and ask > 0 else 0.0
                self._update_cache(symbol, bid, ask, last_price, 0, "option")
            except Exception as e:
                logger.debug(f"[PriceStream] Option quote handler error: {e}")
        return on_quote

    def _make_option_trade_handler(self):
        async def on_trade(data):
            try:
                symbol = str(data.symbol)
                price = float(getattr(data, "price", 0) or 0)
                size = int(getattr(data, "size", 0) or 0)
                # Full read-modify-write under single lock to avoid TOCTOU
                with self._lock:
                    existing = self._cache.get(symbol)
                    bid = existing.bid if existing else 0.0
                    ask = existing.ask if existing else 0.0
                    mid = round((bid + ask) / 2, 4) if bid > 0 and ask > 0 else price
                    self._cache[symbol] = CachedQuote(
                        symbol=symbol, bid=bid, ask=ask, mid=mid,
                        last_trade_price=price, last_trade_size=size,
                        timestamp=time.time(), source="option",
                    )
                    quote = self._cache[symbol]
                self._fire_callbacks(symbol, quote)
            except Exception as e:
                logger.debug(f"[PriceStream] Option trade handler error: {e}")
        return on_trade

    def _stock_stream_thread(self) -> None:
        """Run StockDataStream in a dedicated thread (blocking)."""
        from alpaca.data.live import StockDataStream
        from alpaca.data.enums import DataFeed

        feed = DataFeed.SIP if config.alpaca_stock_feed.lower() == "sip" else DataFeed.IEX

        stream = StockDataStream(
            api_key=config.alpaca_api_key,
            secret_key=config.alpaca_secret_key,
            feed=feed,
        )
        self._stock_stream = stream  # Store for clean shutdown

        with self._sub_lock:
            symbols = list(self._stock_subs)

        # S129: Guard against empty subscription list (causes 400 error on some Alpaca SDK versions)
        if not symbols:
            logger.warning("[PriceStream] No stock symbols to subscribe — skipping stream start")
            self._stock_stream = None
            return

        logger.info(f"[PriceStream] Subscribing stock stream to {len(symbols)} symbols: "
                     f"{symbols[:10]}{'...' if len(symbols) > 10 else ''} "
                     f"(feed={feed}, paper={config.alpaca_paper})")
        stream.subscribe_quotes(self._make_stock_quote_handler(), *symbols)
        stream.subscribe_trades(self._make_stock_trade_handler(), *symbols)
        try:
            stream.run()  # BLOCKING — runs until disconnect or stop()
        except Exception as e:
            logger.error(f"[PriceStream] Stock stream error (feed={feed}): {type(e).__name__}: {e}")
        finally:
            self._stock_stream = None

    def _option_stream_thread(self) -> None:
        """Run OptionDataStream in a dedicated thread (blocking)."""
        from alpaca.data.live import OptionDataStream

        # Try to construct with feed parameter first
        stream = None
        try:
            from alpaca.data.enums import OptionsFeed
            feed = OptionsFeed.OPRA if config.alpaca_options_feed.lower() == "opra" else OptionsFeed.INDICATIVE
            stream = OptionDataStream(
                api_key=config.alpaca_api_key,
                secret_key=config.alpaca_secret_key,
                feed=feed,
            )
        except (ImportError, TypeError):
            pass
        if stream is None:
            stream = OptionDataStream(
                api_key=config.alpaca_api_key,
                secret_key=config.alpaca_secret_key,
            )
        self._option_stream = stream  # Store for clean shutdown

        with self._sub_lock:
            symbols = list(self._option_subs)

        # Guard against empty subscription list (same pattern as stock stream)
        if not symbols:
            logger.warning("[PriceStream] No option symbols to subscribe — skipping stream start")
            self._option_stream = None
            return

        stream.subscribe_quotes(self._make_option_quote_handler(), *symbols)
        stream.subscribe_trades(self._make_option_trade_handler(), *symbols)

        logger.info(f"[PriceStream] Option stream starting with {len(symbols)} symbols: "
                     f"{symbols[:5]}{'...' if len(symbols) > 5 else ''} "
                     f"(paper={config.alpaca_paper})")
        try:
            stream.run()  # BLOCKING
        except Exception as e:
            logger.error(f"[PriceStream] Option stream error: {type(e).__name__}: {e}")
        finally:
            self._option_stream = None

    # --- Auto-subscribe Portfolio ---

    async def _auto_subscribe_portfolio(self) -> None:
        """Subscribe to all held portfolio positions on startup."""
        try:
            # S121 W-25: Use shared TradingClient singleton from alpaca.py
            from agent.integrations.alpaca import _get_trading_client
            loop = asyncio.get_running_loop()
            client = _get_trading_client()
            positions = await loop.run_in_executor(None, client.get_all_positions)
            with self._sub_lock:
                for pos in positions:
                    symbol = str(pos.symbol)
                    # S159-B H-1: AssetClass is a str-Enum — str() returns
                    # "AssetClass.US_OPTION" not "us_option". Use .value.
                    raw_class = getattr(pos, "asset_class", None)
                    ac = raw_class.value if hasattr(raw_class, "value") else str(raw_class)
                    if ac == "us_option":
                        self._option_subs.add(symbol)
                    else:
                        self._stock_subs.add(symbol)
            logger.info(
                f"[PriceStream] Auto-subscribed: {len(self._stock_subs)} stocks, "
                f"{len(self._option_subs)} options from portfolio"
            )
        except Exception as e:
            logger.debug(f"[PriceStream] Auto-subscribe failed (non-fatal): {e}")


# --- Thread-safe Singleton ---

_stream_manager: Optional[PriceStreamManager] = None
_manager_lock = threading.Lock()


def get_stream_manager() -> PriceStreamManager:
    """Thread-safe singleton for PriceStreamManager."""
    global _stream_manager
    if _stream_manager is None:
        with _manager_lock:
            if _stream_manager is None:
                _stream_manager = PriceStreamManager()
    return _stream_manager
