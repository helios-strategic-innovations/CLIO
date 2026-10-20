"""
CLIO Financial Datasets Client — Bloomberg-Grade Fundamental Data

Direct API client for financialdatasets.ai. Provides CLIO's 12 fundamental
data capabilities: metrics, filings, estimates, profiles, screening, rates,
comparisons, earnings, and deep fundamentals.

Architecture:
  - Thread-safe singleton via get_fd_client()
  - In-memory TTL cache (fundamental data changes slowly)
  - Budget tracking via CLIO's BudgetTracker (consumers call check_and_record)
  - Sync HTTP via httpx (callers use run_in_executor for async)
  - Unwraps FD API nested responses automatically

API Response Wrapper Keys (BC#112 — verified via Hydrinia 157-C2 + CLIO smoke test):
  - /financial-metrics/snapshot     -> "snapshot" (NOT "financial_metrics_snapshot")
  - /earnings                       -> "earnings" (contains "quarterly" + "annual" sub-keys)
  - /filings/items                  -> "items" (NOT "filing_items")
  - /financials/search/screener      -> "results" (NOT "search_results")
  - /financials/search/line-items   -> "search_results"
  - /macro/interest-rates/snapshot  -> returns ALL banks, filter client-side
  - /macro/interest-rates           -> "interest_rates" (ascending sort, oldest first)
  NOTE: Interest rate endpoints are under /macro/ prefix.

CC-CLIO-FD-1 | Atlas Strategies — March 2026
"""

import json
import logging
import os
import random
import threading
import time
from typing import Any, Dict, List, Optional

import httpx

from agent.config import config

logger = logging.getLogger("openclaw.fd_client")

# ============================================================
# Cache TTLs (seconds) — matches Hydrinia's client
# ============================================================

_CACHE_TTLS = {
    "financials": 86400,               # 24hr (quarterly statements)
    "income_statements": 86400,
    "balance_sheets": 86400,
    "cash_flow_statements": 86400,
    "segmented_revenues": 86400,
    "financial_metrics": 3600,         # 1hr (computed from statements + price)
    "financial_metrics_snapshot": 3600,
    "earnings": 3600,
    "filings": 21600,                  # 6hr
    "filing_items": 21600,
    "company_facts": 86400,            # 24hr (near-static)
    "analyst_estimates": 21600,        # 6hr (consensus shifts slowly)
    "interest_rates": 3600,            # 1hr
    "interest_rates_snapshot": 3600,
    "screener": 900,                   # 15min
    "line_items": 900,
}

# Request config
_TIMEOUT = 30.0
_MAX_RETRIES = 2
_RETRY_BACKOFF = 2.0

# ============================================================
# Utilities
# ============================================================


def _safe_float(val, default=0.0) -> float:
    """Parse numeric fields safely."""
    if val is None:
        return default
    try:
        result = float(val)
        if result != result or result == float("inf") or result == float("-inf"):
            return default
        return result
    except (ValueError, TypeError):
        return default


# ============================================================
# Client
# ============================================================


class FinancialDatasetsClient:
    """CLIO's Financial Datasets API client with TTL cache.

    Thread-safe singleton. All CLIO consumers share cached data.
    Lighter than Hydrinia's client — no CLI mode, no format_* methods.

    Budget tracking: Consumers must call check_and_record("financial_datasets")
    before invoking client methods. The client itself does not check budgets
    because BudgetTracker is async and this client is sync.
    """

    def __init__(self):
        self._api_key = config.financial_datasets_api_key
        self._base_url = config.financial_datasets_base_url.rstrip("/")

        if not self._api_key:
            logger.warning("FD-CLIO: No API key (FINANCIAL_DATASETS_API_KEY). Calls will fail.")

        self._headers = {
            "X-API-KEY": self._api_key,
            "Accept": "application/json",
        }

        # P1-8: persistent HTTP client (connection pool + keep-alive) reused across
        # every request and retry, instead of a fresh client (TCP+TLS handshake) per call.
        self._client = httpx.Client(
            timeout=_TIMEOUT, follow_redirects=True, headers=self._headers
        )

        # Thread-safe cache: {key: (data, expiry_ts)}
        self._cache: Dict[str, tuple] = {}
        self._cache_lock = threading.Lock()

        # Counters (informational — GIL-safe increments)
        self._request_count = 0
        self._cache_hits = 0
        self._errors = 0

        logger.info("FD-CLIO: Client initialized (key=%s)", "set" if self._api_key else "MISSING")

    # ── Cache ──────────────────────────────────────────────────

    def _cache_key(self, endpoint: str, params: dict) -> str:
        sorted_params = "&".join(f"{k}={v}" for k, v in sorted(params.items()) if v is not None)
        return f"{endpoint}?{sorted_params}"

    def _get_cached(self, key: str) -> Optional[Any]:
        with self._cache_lock:
            if key in self._cache:
                data, expiry = self._cache[key]
                if time.time() < expiry:
                    self._cache_hits += 1
                    return data
                del self._cache[key]
        return None

    def _set_cached(self, key: str, data: Any, ttl_key: str):
        ttl = _CACHE_TTLS.get(ttl_key, 3600)
        with self._cache_lock:
            self._cache[key] = (data, time.time() + ttl)

    def cache_stats(self) -> Dict[str, Any]:
        with self._cache_lock:
            active = sum(1 for _, (_, exp) in self._cache.items() if time.time() < exp)
        total = self._request_count + self._cache_hits
        return {
            "total_requests": self._request_count,
            "cache_hits": self._cache_hits,
            "hit_rate": f"{(self._cache_hits / max(1, total)) * 100:.1f}%",
            "active_entries": active,
            "errors": self._errors,
        }

    # ── Feature flag helpers (runtime-evaluated, NOT module-level) ──

    @staticmethod
    def _is_enabled() -> bool:
        return os.environ.get("CLIO_FD_ENABLED", "0") == "1"

    @staticmethod
    def _is_log_only() -> bool:
        return os.environ.get("CLIO_FD_LOG_ONLY", "1") == "1"

    # ── HTTP ───────────────────────────────────────────────────

    def _get(self, endpoint: str, params: dict = None, ttl_key: str = None) -> Optional[dict]:
        """Synchronous GET with cache and retries.

        Budget tracking is the caller's responsibility (async check_and_record).
        """
        if not self._is_enabled() and not self._is_log_only():
            return {"_error": "FD client disabled", "_status": 0}

        params = params or {}
        cache_key = self._cache_key(endpoint, params)

        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        url = f"{self._base_url}{endpoint}"

        for attempt in range(_MAX_RETRIES + 1):
            try:
                self._request_count += 1

                resp = self._client.get(url, params=params)

                if resp.status_code == 200:
                    data = resp.json()
                    if ttl_key:
                        self._set_cached(cache_key, data, ttl_key)
                    return data

                elif resp.status_code == 401:
                    logger.error("FD-CLIO: 401 Unauthorized — check API key")
                    self._errors += 1
                    return {"_error": "Unauthorized", "_status": 401}

                elif resp.status_code == 402:
                    logger.error("FD-CLIO: 402 Payment Required")
                    self._errors += 1
                    return {"_error": "Payment required", "_status": 402}

                elif resp.status_code == 404:
                    return {"_error": "No data found", "_status": 404}

                elif resp.status_code == 429:
                    wait = _RETRY_BACKOFF * (attempt + 1)
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        try:
                            wait = min(max(wait, float(retry_after)), _TIMEOUT)
                        except ValueError:
                            pass
                    wait += random.uniform(0, wait * 0.25)  # P1-13: de-correlate concurrent retries
                    logger.warning("FD-CLIO: 429 rate limited, waiting %.1fs", wait)
                    time.sleep(wait)
                    continue

                else:
                    logger.warning("FD-CLIO: HTTP %d for %s", resp.status_code, endpoint)
                    if attempt < _MAX_RETRIES:
                        time.sleep(_RETRY_BACKOFF)
                        continue
                    self._errors += 1
                    return {"_error": f"HTTP {resp.status_code}", "_status": resp.status_code}

            except httpx.TimeoutException:
                logger.warning("FD-CLIO: Timeout for %s (attempt %d)", endpoint, attempt + 1)
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_BACKOFF)
                    continue
                self._errors += 1
                return {"_error": "Timeout", "_status": 0}

            except httpx.HTTPError as e:
                logger.error("FD-CLIO: HTTP error for %s: %s", endpoint, e)
                self._errors += 1
                return {"_error": str(e), "_status": 0}

        return {"_error": "Max retries exceeded", "_status": 0}

    def _post(self, endpoint: str, body: dict, ttl_key: str = None) -> Optional[dict]:
        """Synchronous POST with cache and retries."""
        if not self._is_enabled() and not self._is_log_only():
            return {"_error": "FD client disabled", "_status": 0}

        cache_key = self._cache_key(endpoint, {"_body": json.dumps(body, sort_keys=True)})
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        url = f"{self._base_url}{endpoint}"

        for attempt in range(_MAX_RETRIES + 1):
            try:
                self._request_count += 1

                resp = self._client.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=body,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    if ttl_key:
                        self._set_cached(cache_key, data, ttl_key)
                    return data

                elif resp.status_code == 404:
                    return {"_error": "No data found", "_status": 404}

                elif resp.status_code in (401, 402):
                    msgs = {401: "Unauthorized", 402: "Payment required"}
                    self._errors += 1
                    return {"_error": msgs.get(resp.status_code, "Error"), "_status": resp.status_code}

                elif resp.status_code == 429:
                    wait = _RETRY_BACKOFF * (attempt + 1)
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        try:
                            wait = min(max(wait, float(retry_after)), _TIMEOUT)
                        except ValueError:
                            pass
                    wait += random.uniform(0, wait * 0.25)  # P1-13: de-correlate concurrent retries
                    logger.warning("FD-CLIO: 429 rate limited (POST), waiting %.1fs", wait)
                    time.sleep(wait)
                    continue

                else:
                    logger.warning("FD-CLIO: HTTP %d for POST %s", resp.status_code, endpoint)
                    if attempt < _MAX_RETRIES:
                        time.sleep(_RETRY_BACKOFF)
                        continue
                    self._errors += 1
                    return {"_error": f"HTTP {resp.status_code}", "_status": resp.status_code}

            except (httpx.TimeoutException, httpx.HTTPError) as e:
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_BACKOFF)
                    continue
                self._errors += 1
                return {"_error": str(e), "_status": 0}

        return {"_error": "Max retries exceeded", "_status": 0}

    # ── Unwrap Helper ─────────────────────────────────────────
    # FD API nests responses in various wrapper keys.
    # BC#112: Wrapper keys differ from endpoint names — verified via Hydrinia.

    def _unwrap(self, data: Optional[dict], key: str) -> Any:
        """Unwrap FD API nested response. Returns inner payload or original if not nested."""
        if data is None:
            return None
        if "_error" in data:
            return data
        if key in data:
            return data[key]
        return data

    # ═══════════════════════════════════════════════════════════
    #  1. FINANCIAL METRICS SNAPSHOT (real-time 100+ ratios)
    # ═══════════════════════════════════════════════════════════

    def get_financial_metrics_snapshot(self, ticker: str) -> Optional[dict]:
        """Current PE, PB, EV/EBITDA, ROE, ROA, margins, debt ratios — all at a glance.

        BC#112: Response wrapper is "snapshot" (NOT "financial_metrics_snapshot").
        Field names use full words: price_to_earnings_ratio (not pe_ratio),
        price_to_book_ratio (not price_to_book), etc.
        """
        data = self._get("/financial-metrics/snapshot", {"ticker": ticker.upper()},
                         ttl_key="financial_metrics_snapshot")
        return self._unwrap(data, "snapshot")

    # ═══════════════════════════════════════════════════════════
    #  2. FINANCIAL METRICS (historical, 30yr depth)
    # ═══════════════════════════════════════════════════════════

    def get_financial_metrics(self, ticker: str, period: str = "annual",
                              limit: int = 5) -> Optional[dict]:
        """Historical valuation, profitability, growth metrics over time."""
        data = self._get("/financial-metrics",
                         {"ticker": ticker.upper(), "period": period, "limit": limit},
                         ttl_key="financial_metrics")
        return self._unwrap(data, "financial_metrics")

    # ═══════════════════════════════════════════════════════════
    #  3. SEC FILINGS LIST
    # ═══════════════════════════════════════════════════════════

    def get_sec_filings(self, ticker: str, filing_type: List[str] = None,
                        limit: int = 10) -> Optional[dict]:
        """List SEC filings (10-K, 10-Q, 8-K, 20-F) with dates and accession numbers.

        BC#112: Response uses "report_date" (NOT "report_period").
        """
        params: dict = {"ticker": ticker.upper(), "limit": limit}
        if filing_type:
            params["filing_type"] = ",".join(filing_type)
        data = self._get("/filings", params, ttl_key="filings")
        return self._unwrap(data, "filings")

    # ═══════════════════════════════════════════════════════════
    #  4. SEC FILING TEXT (Item-level extraction)
    # ═══════════════════════════════════════════════════════════

    def get_sec_filing_items(self, ticker: str, filing_type: str, year: int,
                             item: str = None, quarter: int = None) -> Optional[dict]:
        """Read specific sections from SEC filings (Risk Factors, MD&A, etc.).

        BC#112: Response wrapper is "items" (NOT "filing_items").
        Item filter uses hyphen format: "Item-1A" (NOT "Item 1A").
        quarter=4 returns 400 for 10-Q (Q4 covered by annual 10-K).
        year = fiscal year, not calendar year.
        CAUTION: Returns large text. Truncate to 4K chars for CLIO context windows.
        """
        # S159 H-8 + S162 M-10: Reject quarter=4 for 10-Q with explicit error
        if filing_type and "10-Q" in str(filing_type).upper() and quarter == 4:
            raise ValueError("Quarter 4 not available for 10-Q filings (covered by annual 10-K). Use '10-K' or quarter 1-3.")
        params: dict = {"ticker": ticker.upper(), "filing_type": filing_type, "year": year}
        if item:
            params["item"] = item
        if quarter is not None and quarter > 0:
            params["quarter"] = quarter
        data = self._get("/filings/items", params, ttl_key="filing_items")
        return self._unwrap(data, "items")

    # ═══════════════════════════════════════════════════════════
    #  5. SEGMENTED REVENUES (business + geography breakdown)
    # ═══════════════════════════════════════════════════════════

    def get_segmented_revenues(self, ticker: str, period: str = "annual",
                                limit: int = 3) -> Optional[dict]:
        """Revenue breakdown by business segment and geography.

        BC#112: XBRL structure — items have identical "name" fields;
        use segments[].label for human-readable segment names.
        """
        data = self._get("/financials/segmented-revenues",
                         {"ticker": ticker.upper(), "period": period, "limit": limit},
                         ttl_key="segmented_revenues")
        return self._unwrap(data, "segmented_revenues")

    # ═══════════════════════════════════════════════════════════
    #  6. ANALYST ESTIMATES (forward consensus EPS/revenue)
    # ═══════════════════════════════════════════════════════════

    def get_analyst_estimates(self, ticker: str, period: str = "annual",
                              limit: int = 3) -> Optional[dict]:
        """Consensus EPS and revenue estimates for future periods.

        BC#112: Fields are "revenue" and "earnings_per_share"
        (NOT "estimated_revenue" or "estimated_eps"). No fiscal_year field.
        """
        data = self._get("/analyst-estimates",
                         {"ticker": ticker.upper(), "period": period, "limit": limit},
                         ttl_key="analyst_estimates")
        return self._unwrap(data, "analyst_estimates")

    # ═══════════════════════════════════════════════════════════
    #  7. COMPANY PROFILE (facts + segments + metrics combined)
    # ═══════════════════════════════════════════════════════════

    def get_company_facts(self, ticker: str) -> Optional[dict]:
        """Company metadata: sector, industry, SIC, CIK, exchange, location.

        BC#112: Available fields include sic_industry, location, exchange,
        is_active. Does NOT include employees, headquarters, or description.
        """
        data = self._get("/company/facts", {"ticker": ticker.upper()},
                         ttl_key="company_facts")
        return self._unwrap(data, "company_facts")

    def get_company_profile(self, ticker: str) -> dict:
        """Combined profile: facts + revenue segments + metrics snapshot.

        Makes 3 API calls (cached independently). Returns combined dict.
        """
        facts = self.get_company_facts(ticker)
        segments = self.get_segmented_revenues(ticker, period="annual", limit=1)
        metrics = self.get_financial_metrics_snapshot(ticker)

        return {
            "ticker": ticker.upper(),
            "facts": facts if not isinstance(facts, dict) or "_error" not in facts else None,
            "revenue_segments": segments if not isinstance(segments, dict) or "_error" not in segments else None,
            "metrics_snapshot": metrics if not isinstance(metrics, dict) or "_error" not in metrics else None,
        }

    # ═══════════════════════════════════════════════════════════
    #  8. INTEREST RATES (central bank rates: FED, ECB, BOJ, etc.)
    # ═══════════════════════════════════════════════════════════

    def get_interest_rates_snapshot(self, bank: str = "FED") -> Optional[dict]:
        """Current rate for a central bank.

        BC#114: Snapshot endpoint returns ALL ~10 banks regardless of query param.
        Must filter client-side by "bank" field.
        Endpoint: /macro/interest-rates/snapshot (NOT /interest-rates/snapshot).
        """
        data = self._get("/macro/interest-rates/snapshot",
                         {"central_bank": bank.upper()},
                         ttl_key="interest_rates_snapshot")
        # API returns all banks — filter to requested bank
        if data and "_error" not in data:
            unwrapped = self._unwrap(data, "interest_rates")
            if isinstance(unwrapped, list):
                bank_upper = bank.upper()
                for rate_entry in unwrapped:
                    if isinstance(rate_entry, dict) and rate_entry.get("bank", "").upper() == bank_upper:
                        return rate_entry
                return {"_error": f"Bank {bank} not found in snapshot", "_status": 404}
            return unwrapped
        return self._unwrap(data, "interest_rates")

    def get_interest_rates(self, bank: str = "FED", start_date: str = None,
                           end_date: str = None) -> Optional[dict]:
        """Historical rate changes for a central bank.

        BC#113: Returns ASCENDING sort (oldest first, 1954->2026).
        Use rates[-1] for most recent. ~860 monthly entries for FED.
        Endpoint: /macro/interest-rates (NOT /interest-rates).
        """
        params: dict = {"bank": bank.upper()}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        data = self._get("/macro/interest-rates", params, ttl_key="interest_rates")
        rates = self._unwrap(data, "interest_rates")
        # S159 H-9: Validate ascending sort order (oldest first)
        if rates and len(rates) >= 2:
            _first_d = rates[0].get("date", rates[0].get("effective_date", ""))
            _last_d = rates[-1].get("date", rates[-1].get("effective_date", ""))
            if _first_d and _last_d and _first_d > _last_d:
                logger.warning("S159 H-9: FD interest rates not ascending — reversing")
                rates = list(reversed(rates))
        return rates

    # ═══════════════════════════════════════════════════════════
    #  9. STOCK SCREENER (17K+ tickers, fundamental filters)
    # ═══════════════════════════════════════════════════════════

    def screen_stocks(self, filters: List[dict], limit: int = 20) -> Optional[dict]:
        """Screen stocks by fundamental criteria.

        Each filter: {"field": "pe_ratio", "operator": "lte", "value": 15}
        Operators: gt, lt, gte, lte, eq, in

        Endpoint: /financials/search/screener (verified via API test).
        Response wrapper is "results" (NOT "search_results").
        Screener filter fields use short names (pe_ratio, etc.) which
        differ from snapshot response fields (price_to_earnings_ratio).
        Returns ONLY filtered fields — not full financial profiles.
        """
        body = {"filters": filters, "limit": limit}
        data = self._post("/financials/search/screener", body, ttl_key="screener")
        return self._unwrap(data, "results")

    # ═══════════════════════════════════════════════════════════
    #  10. COMPARE FINANCIALS (cross-ticker line items)
    # ═══════════════════════════════════════════════════════════

    def search_line_items(self, line_items: List[str], tickers: List[str],
                          period: str = "annual", limit: int = 1) -> Optional[dict]:
        """Side-by-side financial comparison across tickers.

        line_items: ["revenue", "net_income", "eps_diluted", "free_cash_flow"]
        tickers: ["NVDA", "AMD", "INTC"]

        Endpoint: /financials/search/line-items (verified via API test).
        Response wrapper is "search_results".
        """
        body = {
            "line_items": line_items,
            "tickers": [t.upper() for t in tickers],
            "period": period,
            "limit": limit,
        }
        data = self._post("/financials/search/line-items", body, ttl_key="line_items")
        return self._unwrap(data, "search_results")

    # ═══════════════════════════════════════════════════════════
    #  11. DEEP FUNDAMENTALS (statements + metrics + facts)
    # ═══════════════════════════════════════════════════════════

    def get_deep_fundamentals(self, ticker: str, period: str = "annual",
                               limit: int = 3) -> dict:
        """Comprehensive fundamental analysis — combined from multiple endpoints.

        Returns: facts, income statements, balance sheets, cash flows, metrics, snapshot.
        Makes 6 API calls (all cached independently).
        """
        t = ticker.upper()
        return {
            "ticker": t,
            "facts": self.get_company_facts(t),
            "income_statements": self._unwrap(
                self._get("/financials/income-statements",
                          {"ticker": t, "period": period, "limit": limit},
                          ttl_key="income_statements"),
                "income_statements"),
            "balance_sheets": self._unwrap(
                self._get("/financials/balance-sheets",
                          {"ticker": t, "period": period, "limit": limit},
                          ttl_key="balance_sheets"),
                "balance_sheets"),
            "cash_flows": self._unwrap(
                self._get("/financials/cash-flow-statements",
                          {"ticker": t, "period": period, "limit": limit},
                          ttl_key="cash_flow_statements"),
                "cash_flow_statements"),
            "metrics": self.get_financial_metrics(t, period=period, limit=limit),
            "snapshot": self.get_financial_metrics_snapshot(t),
        }

    # ═══════════════════════════════════════════════════════════
    #  12. EARNINGS DETAIL (surprise, YoY, estimates)
    # ═══════════════════════════════════════════════════════════

    def get_earnings(self, ticker: str, limit: int = 4) -> Optional[dict]:
        """Recent earnings with actual vs estimated, surprise %, and YoY changes.

        Response: {"earnings": {"ticker": ..., "quarterly": {...}, "annual": {...}}}
        Returns the inner "earnings" dict. Access .get("quarterly") or .get("annual").
        Fields: earnings_per_share (not eps_diluted),
        estimated_earnings_per_share (not estimated_eps).
        revenue_surprise and eps_surprise are STRINGS ("BEAT"/"MISS"),
        NOT numeric percentages.
        """
        data = self._get("/earnings", {"ticker": ticker.upper(), "limit": limit},
                         ttl_key="earnings")
        return self._unwrap(data, "earnings")

    def get_earnings_context(self, ticker: str) -> dict:
        """Combined earnings context: latest earnings + forward estimates.

        Single call returns everything needed for pre/post-earnings analysis.
        """
        return {
            "ticker": ticker.upper(),
            "latest_earnings": self.get_earnings(ticker, limit=2),
            "forward_estimates": self.get_analyst_estimates(ticker, period="quarterly", limit=4),
        }

    # ── Convenience: Fundamental Score ────────────────────────

    def fundamental_score(self, ticker: str) -> dict:
        """Quick fundamental quality score for a ticker.

        Returns: {"score": 0.0-1.0, "grade": "A"-"F", "factors": [...]}
        Used by: trade idea feedback, learning engine thesis evaluation.
        """
        snap = self.get_financial_metrics_snapshot(ticker)
        if not snap or (isinstance(snap, dict) and "_error" in snap):
            return {"score": 0.5, "grade": "N/A", "factors": ["No FD data available"]}

        score = 0.5
        factors = []

        # Profitability (field confirmed: return_on_equity)
        roe = snap.get("return_on_equity")
        if roe is not None:
            roe = _safe_float(roe)
            if roe > 0.15:
                score += 0.10
                factors.append(f"Strong ROE: {roe:.1%}")
            elif roe < 0:
                score -= 0.15
                factors.append(f"Negative ROE: {roe:.1%}")

        # Growth (field confirmed: revenue_growth)
        rev_growth = snap.get("revenue_growth")
        if rev_growth is not None:
            rev_growth = _safe_float(rev_growth)
            if rev_growth > 0.10:
                score += 0.10
                factors.append(f"Revenue growing: {rev_growth:.1%}")
            elif rev_growth < -0.10:
                score -= 0.10
                factors.append(f"Revenue declining: {rev_growth:.1%}")

        # Leverage (field confirmed: debt_to_equity)
        dte = snap.get("debt_to_equity")
        if dte is not None:
            dte = _safe_float(dte)
            if dte > 3.0:
                score -= 0.10
                factors.append(f"High leverage: D/E {dte:.1f}")
            elif dte < 0.5:
                score += 0.05
                factors.append(f"Conservative leverage: D/E {dte:.1f}")

        # Margin (field confirmed: net_margin)
        net_margin = snap.get("net_margin")
        if net_margin is not None:
            net_margin = _safe_float(net_margin)
            if net_margin > 0.15:
                score += 0.10
                factors.append(f"Strong margins: {net_margin:.1%}")
            elif net_margin < 0:
                score -= 0.15
                factors.append(f"Unprofitable: {net_margin:.1%}")

        # FCF (BC#112: field is free_cash_flow_yield, NOT free_cash_flow_per_share)
        fcf_yield = snap.get("free_cash_flow_yield")
        if fcf_yield is not None:
            fcf_yield = _safe_float(fcf_yield)
            if fcf_yield > 0.03:
                score += 0.05
                factors.append(f"Positive FCF yield: {fcf_yield:.1%}")
            elif fcf_yield < 0:
                score -= 0.10
                factors.append(f"Negative FCF yield: {fcf_yield:.1%}")

        score = max(0.0, min(1.0, score))
        grade = ("A" if score >= 0.8 else "B" if score >= 0.6 else
                 "C" if score >= 0.4 else "D" if score >= 0.2 else "F")

        return {"score": round(score, 3), "grade": grade, "factors": factors}


# ============================================================
# Singleton
# ============================================================

_instance: Optional[FinancialDatasetsClient] = None
_instance_lock = threading.Lock()


def get_fd_client() -> FinancialDatasetsClient:
    """Get or create singleton FD client."""
    global _instance
    if _instance is not None:
        return _instance
    with _instance_lock:
        if _instance is not None:
            return _instance
        _instance = FinancialDatasetsClient()
        return _instance
