"""
OpenClaw/Moltbot — Rate Limiting, Budget Tracking & Retry Logic

Provides:
  - BudgetTracker: Persistent quota tracking for budget-limited APIs
  - retry_on_transient: Decorator for exponential backoff on transient HTTP errors
  - budget_and_retry: Combined budget check + retry decorator
  - BudgetExhaustedError: Exception for budget exhaustion

Budget caps (Moltbot shares APIs with Hydrinia trading bot):
  - SEC-API: 500/month
  - Unusual Whales: 16,000/day (CLIO runaway guard; ~10% of UW Startup 160K/day)
  - SAM.gov: 500/month
  - Apify: 1,000/day

Retired buckets (kept here as forensic anchors so future audits don't re-add):
  - Trading Economics: 70/month — retired S187 2026-05-03 (econ_calendar migrated to UW)
"""

import asyncio
import fcntl
import functools
import json
import logging
import os
import random
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

_NY = ZoneInfo("America/New_York")
from pathlib import Path
from typing import Any

import aiofiles
import httpx

logger = logging.getLogger("openclaw.rate_limiter")

# CLIO_RATE_LIMITS_FILE overrides the location (tests point it at a temp file
# so a test run can never consume the live VPS budget counters).
QUOTA_FILE = Path(os.environ.get("CLIO_RATE_LIMITS_FILE") or "/home/agent/openclaw/logs/rate_limits.json")

# Transient HTTP status codes worth retrying
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}

# Delay multipliers by error type (inspired by Hydrinia's IntelligentRetryStrategy)
ERROR_DELAY_MULTIPLIERS = {
    429: 5.0,  # Rate limited — back off hard
    500: 3.0,  # Server error
    502: 2.0,  # Bad gateway
    503: 3.0,  # Service unavailable
    504: 2.0,  # Gateway timeout
}


# ============================================================
# Budget Tracking
# ============================================================

class BudgetExhaustedError(Exception):
    """Raised when an API's request budget is exhausted for the current period."""

    def __init__(self, api_name: str, message: str):
        self.api_name = api_name
        super().__init__(message)


class _ApiBudget:
    __slots__ = ("name", "max_requests", "period", "used", "period_start", "warned_80_period")

    def __init__(self, name: str, max_requests: int, period: str):
        self.name = name
        self.max_requests = max_requests
        self.period = period  # "daily" or "monthly"
        self.used = 0
        self.period_start = ""
        self.warned_80_period = ""


class BudgetTracker:
    """Tracks usage quotas for budget-limited APIs. Async-safe via asyncio.Lock."""

    def __init__(self):
        self._budgets: dict[str, _ApiBudget] = {}
        self._lock = asyncio.Lock()
        self._loaded = False

    def register(self, name: str, max_requests: int, period: str) -> None:
        """Register an API budget. Call at module load time, before any requests."""
        self._budgets[name] = _ApiBudget(name, max_requests, period)

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        async with self._lock:
            if self._loaded:
                return
            await self._load()
            self._loaded = True

    async def _load(self) -> None:
        if not QUOTA_FILE.exists():
            logger.info("No rate_limits.json found, starting fresh")
            return
        try:
            async with aiofiles.open(QUOTA_FILE, "r") as f:
                data = json.loads(await f.read())
            for name, saved in data.items():
                if name in self._budgets:
                    self._budgets[name].used = saved.get("used", 0)
                    self._budgets[name].period_start = saved.get("period_start", "")
        except Exception as e:
            logger.error(f"Failed to load rate_limits.json: {e}")

    def _load_sync(self) -> None:
        """Synchronous eager load — called once at module init before event loop exists."""
        if not QUOTA_FILE.exists():
            logger.info("No rate_limits.json found, starting fresh")
            self._loaded = True
            return
        lock_fd = None
        try:
            lock_fd = open(str(QUOTA_FILE) + ".lock", "a")
            fcntl.flock(lock_fd, fcntl.LOCK_SH)
            with open(QUOTA_FILE, "r") as f:
                data = json.load(f)
            for name, saved in data.items():
                if name in self._budgets:
                    self._budgets[name].used = saved.get("used", 0)
                    self._budgets[name].period_start = saved.get("period_start", "")
            self._loaded = True
            logger.info(f"Budget counters loaded from disk ({len(data)} entries)")
        except Exception as e:
            logger.error(f"Failed to load rate_limits.json (sync): {e}")
            self._loaded = True  # Mark loaded to prevent async retry on bad file
        finally:
            if lock_fd:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    lock_fd.close()
                except Exception:
                    pass

    def _save_sync(self, data: dict) -> None:
        """Blocking I/O for rate-limit persistence — runs in a thread executor.
        Never call directly from the asyncio event loop (S163-B invariant)."""
        lock_fd = None
        try:
            QUOTA_FILE.parent.mkdir(parents=True, exist_ok=True)
            lock_fd = open(str(QUOTA_FILE) + ".lock", "a")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            tmp_path = str(QUOTA_FILE) + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, str(QUOTA_FILE))
        except Exception as e:
            logger.error(f"Failed to save rate_limits.json: {e}")
        finally:
            if lock_fd:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    lock_fd.close()
                except Exception:
                    pass

    async def _save(self) -> None:
        """Persist budget data atomically. Dispatches blocking I/O to executor
        to avoid blocking the asyncio event loop (flock-not-on-event-loop)."""
        data = {
            name: {
                "used": b.used,
                "period_start": b.period_start,
                "max_requests": b.max_requests,
                "period": b.period,
            }
            for name, b in self._budgets.items()
        }
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._save_sync, data)

    @staticmethod
    def _current_period_key(period: str) -> str:
        now = datetime.now(_NY)  # Reset at ET midnight, consistent with codebase convention
        if period == "daily":
            return now.strftime("%Y-%m-%d")
        return now.strftime("%Y-%m")  # monthly

    def _maybe_reset(self, b: _ApiBudget) -> None:
        """Reset counter if the period has rolled over."""
        current_key = self._current_period_key(b.period)
        if b.period_start != current_key:
            logger.info(f"Budget reset for {b.name}: {b.period_start} -> {current_key}")
            b.used = 0
            b.period_start = current_key
            b.warned_80_period = ""

    def _maybe_warn_80pct(self, b: _ApiBudget) -> None:
        """Warn once per period when an API crosses 80% of its local budget."""
        if b.max_requests <= 0 or b.warned_80_period == b.period_start:
            return
        if b.used >= int(b.max_requests * 0.8):
            logger.warning(
                "Budget 80%% threshold for %s: %s/%s used this %s",
                b.name,
                b.used,
                b.max_requests,
                "day" if b.period == "daily" else "month",
            )
            b.warned_80_period = b.period_start

    async def check_budget(self, name: str) -> tuple[bool, str]:
        """Pre-flight check. Returns (allowed, message)."""
        await self._ensure_loaded()
        if name not in self._budgets:
            return True, "OK"

        async with self._lock:
            return self._check_budget_unlocked(name)

    def _check_budget_unlocked(self, name: str) -> tuple[bool, str]:
        """Pre-flight check WITHOUT acquiring lock. Caller MUST hold self._lock."""
        if name not in self._budgets:
            return True, "OK"
        b = self._budgets[name]
        self._maybe_reset(b)

        remaining = b.max_requests - b.used
        if remaining <= 0:
            period_label = "today" if b.period == "daily" else "this month"
            msg = (
                f"Budget exhausted for {b.name}: "
                f"{b.used}/{b.max_requests} used {period_label}. "
                f"Resets next {'day' if b.period == 'daily' else 'month'}."
            )
            logger.info(f"Budget exhausted: {b.name} ({b.used}/{b.max_requests} used {period_label})")
            return False, msg

        if remaining <= b.max_requests * 0.1:
            logger.warning(f"Budget low for {b.name}: {remaining}/{b.max_requests} remaining")

        return True, f"OK ({remaining} remaining)"

    async def record_usage(self, name: str, count: int = 1) -> None:
        """Record that `count` requests were made. Persists to disk."""
        await self._ensure_loaded()
        if name not in self._budgets:
            return

        async with self._lock:
            await self._record_usage_unlocked(name, count)

    async def _record_usage_unlocked(self, name: str, count: int = 1) -> None:
        """Record usage WITHOUT acquiring lock. Caller MUST hold self._lock."""
        if name not in self._budgets:
            return
        b = self._budgets[name]
        self._maybe_reset(b)
        b.used += count
        self._maybe_warn_80pct(b)
        await self._save()
        logger.debug(f"Budget {b.name}: {b.used}/{b.max_requests}")

    async def get_status(self) -> dict[str, dict[str, Any]]:
        """Return status of all budgets (for /status command)."""
        await self._ensure_loaded()
        result = {}
        for name, b in self._budgets.items():
            current_key = self._current_period_key(b.period)
            used = b.used if b.period_start == current_key else 0
            result[name] = {
                "used": used,
                "max": b.max_requests,
                "remaining": b.max_requests - used,
                "period": b.period,
            }
        return result


# Singleton instance
budget_tracker = BudgetTracker()


def _register_budgets():
    """Register budgets using config values (supports .env overrides).

    Uses lazy import to avoid circular dependency (config.py does not
    import rate_limiter, so this direction is safe).
    """
    from agent.config import config

    budget_tracker.register("sec_api", max_requests=config.budget_sec_api, period="monthly")
    # trading_economics retired S187 (econ_calendar migrated to UW)
    budget_tracker.register("unusual_whales", max_requests=config.budget_unusual_whales, period="daily")
    budget_tracker.register("sam_gov", max_requests=config.budget_sam_gov, period="monthly")
    budget_tracker.register("apify", max_requests=config.budget_apify, period="daily")
    budget_tracker.register("whisper", max_requests=config.budget_whisper, period="daily")
    budget_tracker.register("email_send", max_requests=config.budget_email_send, period="daily")
    budget_tracker.register("email_read", max_requests=config.budget_email_read, period="daily")
    budget_tracker.register("web_search", max_requests=config.budget_web_search, period="daily")
    budget_tracker.register("web_fetch", max_requests=config.budget_web_fetch, period="daily")
    budget_tracker.register("learning_analysis", max_requests=config.budget_learning_analysis, period="daily")
    budget_tracker.register("learning_synthesis", max_requests=config.budget_learning_synthesis, period="daily")
    budget_tracker.register("polygon_learning", max_requests=config.budget_polygon_learning, period="daily")
    budget_tracker.register("book_analysis", max_requests=config.budget_book_analysis, period="daily")
    budget_tracker.register("book_synthesis", max_requests=config.budget_book_synthesis, period="daily")
    budget_tracker.register("learning_query_gen", max_requests=config.budget_learning_query_gen, period="daily")
    budget_tracker.register("elevenlabs_tts", max_requests=config.budget_elevenlabs_tts, period="daily")
    budget_tracker.register("premarket_scan", max_requests=config.budget_premarket_scan, period="daily")
    budget_tracker.register("hourly_debrief", max_requests=config.budget_hourly_debrief, period="daily")
    budget_tracker.register("postmarket_debrief", max_requests=config.budget_postmarket_debrief, period="daily")
    budget_tracker.register("news_update", max_requests=config.budget_news_update, period="daily")
    budget_tracker.register("report_web_search", max_requests=config.budget_report_web_search, period="daily")
    budget_tracker.register("news_intel", max_requests=config.budget_news_intel, period="daily")
    budget_tracker.register("macro_research", max_requests=config.budget_macro_research, period="daily")
    budget_tracker.register("geopolitical_research", max_requests=config.budget_geopolitical_research, period="daily")
    budget_tracker.register("fed_analysis", max_requests=config.budget_fed_analysis, period="daily")
    budget_tracker.register("earnings_intel", max_requests=config.budget_earnings_intel, period="daily")
    budget_tracker.register("trade_policy", max_requests=config.budget_trade_policy, period="daily")
    budget_tracker.register("commodity_research", max_requests=config.budget_commodity_research, period="daily")
    budget_tracker.register("bond_intel", max_requests=config.budget_bond_intel, period="daily")

    # SEAL enrichment budget (Haiku calls for knowledge implication generation)
    budget_tracker.register("seal_enrichment", max_requests=config.budget_seal_enrichment, period="daily")

    # SEAL retention testing budget (Haiku calls for QA verification)
    budget_tracker.register("seal_retention", max_requests=config.budget_seal_retention, period="daily")

    # Financial ML #25: Adversarial contradiction resolution (Haiku)
    budget_tracker.register("adversarial_resolver", max_requests=config.budget_adversarial_resolver, period="daily")

    # Financial Datasets (Bloomberg-grade fundamentals via financialdatasets.ai)
    budget_tracker.register("financial_datasets", max_requests=config.budget_financial_datasets, period="daily")

    # E.1 social_intel signal extraction (Sonnet call attempts; dollar belt lives in enrich_runs)
    budget_tracker.register("social_enrich", max_requests=config.budget_social_enrich, period="daily")

    # econ_calendar FRED fallback (CLIO's first live FRED API consumer; see agent/integrations/econ_calendar.py)
    budget_tracker.register("fred_api", max_requests=config.budget_fred_api, period="daily")

    # econ_surprise: Nasdaq + TradingView unofficial econ-print endpoints (see agent/integrations/econ_surprise.py)
    budget_tracker.register("econ_surprise", max_requests=config.budget_econ_surprise, period="daily")

    # newsflash_lite: Polygon /v2/reference/news general-market poll (see agent/integrations/newsflash_lite.py)
    budget_tracker.register("newsflash_polygon", max_requests=config.budget_newsflash_polygon, period="daily")

    # DPOF learning loop (Task ORCH-7Q4K-DPOF builder B2, 2026-09-26): F1 post-trade
    # reflection (agent/dpof/reflection.py) + F2 borderline debate (agent/dpof/debate.py)
    budget_tracker.register("dpof_reflection", max_requests=config.budget_dpof_reflection, period="daily")
    budget_tracker.register("dpof_debate", max_requests=config.budget_dpof_debate, period="daily")

    # Eagerly load persisted counters so budgets survive restart
    budget_tracker._load_sync()


_register_budgets()


async def check_and_record(api_name: str) -> None:
    """Pre-flight check + record usage atomically. Raises BudgetExhaustedError if over budget.

    S118 W-008: Holds lock for both check and record to prevent concurrent coroutines
    from passing the same check between the two operations.
    """
    await budget_tracker._ensure_loaded()
    async with budget_tracker._lock:
        allowed, msg = budget_tracker._check_budget_unlocked(api_name)
        if not allowed:
            raise BudgetExhaustedError(api_name, msg)
        await budget_tracker._record_usage_unlocked(api_name)


# ============================================================
# Retry Decorators
# ============================================================

def _compute_delay(attempt: int, status: int | None, base_delay: float, max_delay: float) -> float:
    """Compute backoff delay based on attempt number and error type."""
    multiplier = ERROR_DELAY_MULTIPLIERS.get(status, 2.0) if status else 2.0
    delay = min(base_delay * (2 ** (attempt - 1)) * multiplier, max_delay)
    # P1-9: jitter to de-correlate lockstep retries. Clamp AFTER adding jitter —
    # jitter must never push the returned delay past the caller's max_delay contract.
    return min(delay + random.uniform(0, delay * 0.25), max_delay)


def retry_on_transient(max_attempts: int = 3, base_delay: float = 1.0, max_delay: float = 30.0):
    """Async decorator: retry on transient HTTP errors with exponential backoff.

    Handles httpx.HTTPStatusError (429, 5xx), httpx.TimeoutException,
    and httpx.ConnectError. Non-transient errors (4xx except 429) are raised immediately.
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)

                except httpx.HTTPStatusError as e:
                    status = e.response.status_code
                    if status not in TRANSIENT_STATUS_CODES or attempt == max_attempts:
                        raise
                    delay = _compute_delay(attempt, status, base_delay, max_delay)
                    retry_after = e.response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = min(max(delay, float(retry_after)), max_delay * 2)
                        except ValueError:
                            pass
                    logger.warning(
                        f"Retry {attempt}/{max_attempts} for {func.__name__}: "
                        f"HTTP {status}, sleeping {delay:.1f}s"
                    )
                    last_exception = e
                    await asyncio.sleep(delay)

                except (httpx.TimeoutException, httpx.ConnectError) as e:
                    if attempt == max_attempts:
                        raise
                    delay = _compute_delay(attempt, None, base_delay, max_delay)
                    logger.warning(
                        f"Retry {attempt}/{max_attempts} for {func.__name__}: "
                        f"{type(e).__name__}, sleeping {delay:.1f}s"
                    )
                    last_exception = e
                    await asyncio.sleep(delay)

            # S118 W-009: Guard against last_exception being None (e.g. max_attempts=0)
            if last_exception is not None:
                raise last_exception
            raise RuntimeError(f"Retry exhausted after {max_attempts} attempts")  # S233: Defensive — unreachable unless max_attempts=0

        return wrapper

    return decorator


def budget_and_retry(
    api_name: str, max_attempts: int = 3, base_delay: float = 1.0, max_delay: float = 30.0
):
    """Combined decorator: budget check (once, before retries) + retry with backoff.

    Budget is counted once per logical request — retries do not consume extra budget.
    Raises BudgetExhaustedError if budget is exhausted.
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Budget check happens once, before any attempts
            await check_and_record(api_name)

            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)

                except httpx.HTTPStatusError as e:
                    status = e.response.status_code
                    if status not in TRANSIENT_STATUS_CODES or attempt == max_attempts:
                        raise
                    delay = _compute_delay(attempt, status, base_delay, max_delay)
                    retry_after = e.response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = min(max(delay, float(retry_after)), max_delay * 2)
                        except ValueError:
                            pass
                    logger.warning(
                        f"Retry {attempt}/{max_attempts} for {func.__name__}: "
                        f"HTTP {status}, sleeping {delay:.1f}s"
                    )
                    last_exception = e
                    await asyncio.sleep(delay)

                except (httpx.TimeoutException, httpx.ConnectError) as e:
                    if attempt == max_attempts:
                        raise
                    delay = _compute_delay(attempt, None, base_delay, max_delay)
                    logger.warning(
                        f"Retry {attempt}/{max_attempts} for {func.__name__}: "
                        f"{type(e).__name__}, sleeping {delay:.1f}s"
                    )
                    last_exception = e
                    await asyncio.sleep(delay)

            # S118 W-009: Guard against last_exception being None (e.g. max_attempts=0)
            if last_exception is not None:
                raise last_exception
            raise RuntimeError(f"Retry exhausted after {max_attempts} attempts")  # S233: Defensive — unreachable unless max_attempts=0

        return wrapper

    return decorator
