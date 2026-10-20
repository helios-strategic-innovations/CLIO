"""Shared dependencies — Alpaca clients, Polygon, file readers."""

import json
import os
import fcntl
import logging
import asyncio
import threading
from functools import lru_cache
from typing import Any

from mission_control.backend.config import mc_config

logger = logging.getLogger("mission_control.deps")

# ---------------------------------------------------------------------------
# Alpaca clients (lazy singletons, thread-safe)
# ---------------------------------------------------------------------------
_trading_client = None
_trading_lock = threading.Lock()
_data_client = None
_data_lock = threading.Lock()


def _get_alpaca_config() -> tuple:
    """Read Alpaca config from Moltbot's config module."""
    try:
        from agent.config import config
        return config.alpaca_api_key, config.alpaca_secret_key, config.alpaca_paper, config.alpaca_base_url
    except Exception as e:
        logger.error(f"Failed to import Moltbot config: {e}")
        return "", "", True, "https://api.alpaca.markets"


def get_trading_client():
    global _trading_client
    if _trading_client is None:
        with _trading_lock:
            if _trading_client is None:
                from alpaca.trading.client import TradingClient
                key, secret, paper, url = _get_alpaca_config()
                _trading_client = TradingClient(
                    api_key=key,
                    secret_key=secret,
                    paper=paper,
                    url_override=url,
                )
    return _trading_client


def get_data_client():
    global _data_client
    if _data_client is None:
        with _data_lock:
            if _data_client is None:
                from alpaca.data.historical import StockHistoricalDataClient
                key, secret, _, _ = _get_alpaca_config()
                _data_client = StockHistoricalDataClient(api_key=key, secret_key=secret)
    return _data_client


# ---------------------------------------------------------------------------
# File readers (with locking where needed)
# ---------------------------------------------------------------------------

def read_json(path: str, lock: bool = False) -> Any:
    """Read a JSON file, optionally with shared flock."""
    try:
        if lock:
            lock_path = path + ".lock"
            with open(lock_path, "a") as lf:
                fcntl.flock(lf, fcntl.LOCK_SH)
                try:
                    with open(path) as f:
                        return json.load(f)
                finally:
                    fcntl.flock(lf, fcntl.LOCK_UN)
        else:
            with open(path) as f:
                return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.debug(f"read_json({path}): {e}")
        return None


def read_text(path: str) -> str:
    """Read a text file, return empty string on error."""
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        return ""


def _hydrinia_bridge_enabled() -> bool:
    try:
        from agent.hydrinia_bridge import bridge_enabled

        return bridge_enabled()
    except Exception as e:
        logger.debug(f"deps: bridge_enabled() unavailable ({type(e).__name__}: {e})")
        return False


def load_symbols_universe() -> set:
    """CLIO's trading universe.

    CLIO-native: DPOF's own underlyings (default SPY/QQQ). Falls back to
    Hydrinia's symbols.txt only when the bridge is explicitly re-enabled —
    that file is otherwise frozen/stale (Hydrinia retired, 2026-09-24).
    """
    if not _hydrinia_bridge_enabled():
        try:
            from agent.dpof.config import load_dpof_config

            return set(load_dpof_config().underlyings)
        except Exception as e:
            logger.debug(f"DPOF underlyings unavailable ({type(e).__name__}: {e})")
            return set()
    return _load_symbols_universe_hydrinia()


@lru_cache(maxsize=1)
def _load_symbols_universe_hydrinia() -> set:
    """Hydrinia's trading universe from symbols.txt (bridge-enabled only)."""
    path = os.path.join(mc_config.tradingbot_root, "symbols.txt")
    symbols = set()
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Handle CSV format (symbol,exchange or just symbol)
                sym = line.split(",")[0].strip().upper()
                if sym:
                    symbols.add(sym)
    except FileNotFoundError:
        logger.warning(f"symbols.txt not found at {path}")
    return symbols


async def run_in_executor(fn, *args):
    """Run blocking function in thread executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fn, *args)
