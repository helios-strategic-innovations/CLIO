"""
OpenClaw/Moltbot — Pre-Market Scan (Session 85)

Runs at 9:00 AM ET weekdays. Produces a comprehensive, sector-wide morning brief
with trade ideas, delivered to Telegram (+ the Hydrinia file bridge, only if
explicitly re-enabled via CLIO_HYDRINIA_BRIDGE_ENABLED — see agent/hydrinia_bridge.py).

Coexists with alignment_checker.py (which is position-specific RED/YELLOW/GREEN).
This scan is broader: market overview, sector spotlight, trade ideas, key risks.

Positions/metrics describe CLIO's own DPOF book (SPY/QQQ long options on the CLIO
Alpaca paper account, agent/dpof/) — never Hydrinia's frozen state files.
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import anthropic

from agent.config import config
from agent.integrations.alpaca_values import AlpacaValueError, normalize_position_side
from agent.prompts.system_prompt import cached_system_block
from agent.rate_limiter import budget_tracker
from agent.security import sanitize_ingested_content

logger = logging.getLogger("openclaw.premarket_scan")


def _get_athena():
    """Lazy import for Athena bridge (Phase 1 integration)."""
    try:
        from agent.integrations.athena_bridge import get_athena_bridge
        return get_athena_bridge()
    except Exception:
        return None

_NY = ZoneInfo("America/New_York")

REPORTS_DIR = os.path.join(config.tradingbot_dir, "moltbot_knowledge", "reports", "premarket")

# Cached Anthropic client
_cached_client: anthropic.Anthropic | None = None
_client_lock = __import__("threading").Lock()


def _get_client() -> anthropic.Anthropic:
    global _cached_client
    if _cached_client is None:
        with _client_lock:
            if _cached_client is None:
                _cached_client = anthropic.Anthropic(**config.anthropic_client_kwargs(timeout=120))
    return _cached_client


# Sector mapping for position classification
TICKER_SECTOR = {
    "AAPL": "tech", "MSFT": "tech", "NVDA": "tech", "GOOGL": "tech", "GOOG": "tech",
    "META": "tech", "AMZN": "tech", "TSLA": "consumer", "AMD": "tech", "INTC": "tech",
    "JPM": "financial", "BAC": "financial", "GS": "financial", "MS": "financial",
    "XOM": "energy", "CVX": "energy", "COP": "energy", "SLB": "energy", "OXY": "energy",
    "LMT": "defense", "RTX": "defense", "NOC": "defense", "GD": "defense", "HII": "defense",
    "GLD": "commodities", "SLV": "commodities", "GDX": "commodities", "UNG": "energy",
    "TLT": "bonds", "HYG": "bonds", "LQD": "bonds", "SHY": "bonds", "IEF": "bonds",
    "SPY": "indices", "QQQ": "indices", "IWM": "indices", "DIA": "indices",
    "XLE": "energy", "XLF": "financial", "XLK": "tech", "XLV": "healthcare",
    "XOP": "energy", "BOIL": "energy", "USO": "energy",
    "JNJ": "healthcare", "UNH": "healthcare", "PFE": "healthcare", "MRK": "healthcare",
}

PREMARKET_SYSTEM_PROMPT = """You are a pre-market trading intelligence analyst preparing a comprehensive morning brief for a quantitative trading operation.

Write in plain text only — NO markdown formatting, NO headers with #, NO bold/italic markers.
Be direct, data-driven, and actionable. Every sentence should deliver value.

Structure your brief as follows:

MARKET OVERVIEW
3-4 sentences covering: overnight futures, pre-market sentiment, key overnight moves globally (Asia/Europe), any macro catalysts.

SECTOR SPOTLIGHT
2-3 sectors that are showing notable pre-market activity. For each: what's moving, why, and the trading implication.

POSITION CHECK
For each open position: assess overnight news impact and give an action bias (HOLD/ADD/TRIM/EXIT with brief rationale).

TRADE IDEAS
1-3 actionable trade ideas with: ticker, direction (LONG/SHORT), thesis in 1 sentence, catalyst, and risk.

OPTIONS FLOW CONTEXT (if available)
Highlight any notable options flow signals: unusual NOFA readings, dark pool accumulation/distribution, or premium skew that confirms or contradicts the sector/position thesis above.

KEY RISKS
2-3 risks to monitor today that could disrupt the thesis on current positions or the broader market."""


async def _get_open_positions() -> list[dict]:
    """CLIO's own DPOF positions: live Alpaca account, enriched from the DPOF ledger.

    Alpaca is the source of truth for what's actually held and its live P&L.
    The DPOF readmodel (agent/dpof/readmodel.py — CLIO-native, not Hydrinia)
    supplies direction/structure/entry context for option legs, matched by
    OCC contract symbol. On Alpaca failure, falls back to the DPOF ledger
    alone (no live prices) rather than silently going empty.
    """
    dpof_by_leg: dict[str, object] = {}
    try:
        from agent.dpof import readmodel
        for p in readmodel.open_positions():
            for leg in (p.legs or []):
                sym = leg.get("symbol") if isinstance(leg, dict) else None
                if sym:
                    dpof_by_leg[sym] = p
    except Exception as e:
        logger.debug(f"DPOF readmodel unavailable for position enrichment: {e}")

    result = []
    try:
        from agent.integrations.alpaca import _get_trading_client, _retry_alpaca_call
        from agent.dpof import readmodel
        client = _get_trading_client()
        raw_positions = await _retry_alpaca_call(client.get_all_positions) or []
        for pos in raw_positions:
            symbol = getattr(pos, "symbol", "")
            dpof_pos = dpof_by_leg.get(symbol)
            try:
                live_side = normalize_position_side(getattr(pos, "side", None))
            except AlpacaValueError:
                logger.warning("Pre-market scan skipped unknown side for %s", symbol)
                continue
            try:
                entry_price = float(getattr(pos, "avg_entry_price", 0) or 0)
            except (ValueError, TypeError):
                entry_price = 0.0
            try:
                unrealized_pnl = float(getattr(pos, "unrealized_pl", 0) or 0)
            except (ValueError, TypeError):
                unrealized_pnl = 0.0
            try:
                current_price = float(getattr(pos, "current_price", 0) or 0)
            except (ValueError, TypeError):
                current_price = 0.0
            try:
                contract = readmodel.contract_label(symbol)
            except Exception:
                contract = symbol
            result.append({
                "symbol": getattr(dpof_pos, "underlying", None) or symbol,
                "contract": contract,
                "direction": getattr(dpof_pos, "direction", None) or live_side,
                "structure": getattr(dpof_pos, "structure", ""),
                "entry_price": (
                    dpof_pos.entry_price
                    if dpof_pos is not None and dpof_pos.entry_price is not None
                    else entry_price
                ),
                "unrealized_pnl": unrealized_pnl,
                "current_price": current_price,
                "dpof_id": getattr(dpof_pos, "id", ""),
            })
    except Exception as e:
        logger.warning(f"Alpaca position fetch failed, falling back to DPOF ledger: {e}")
        # Fallback: DPOF ledger only (no live prices)
        try:
            from agent.dpof import readmodel
            for p in readmodel.open_positions():
                result.append({
                    "symbol": p.underlying,
                    "contract": readmodel.position_label(p),
                    "direction": p.direction,
                    "structure": p.structure,
                    "entry_price": p.entry_price or 0,
                    "unrealized_pnl": 0,
                    "current_price": 0,
                    "dpof_id": p.id,
                    "note": "(DPOF ledger; live prices unavailable)",
                })
        except Exception as e2:
            logger.warning(f"DPOF ledger fallback also failed: {e2}")

    return result


async def _get_daily_metrics() -> dict:
    """CLIO-native metrics: Alpaca account equity/last_equity + DPOF's prior-session
    closed trades + market regime. Replaces Hydrinia's daily_metrics.json.
    """
    metrics: dict = {}
    try:
        from agent.integrations.alpaca import _get_trading_client, _retry_alpaca_call
        client = _get_trading_client()
        account = await _retry_alpaca_call(client.get_account)
        if account is not None:
            equity = float(getattr(account, "equity", 0) or 0)
            last_equity = float(getattr(account, "last_equity", 0) or 0)
            metrics["equity"] = equity
            metrics["daily_pnl"] = equity - last_equity
    except Exception as e:
        logger.warning(f"Alpaca account fetch failed for daily metrics: {e}")

    try:
        from agent.dpof import readmodel
        now_et = datetime.now(_NY)
        start_today = now_et.replace(hour=0, minute=0, second=0, microsecond=0)
        start_yesterday = start_today - timedelta(days=1)
        closed = readmodel.closed_between(start_yesterday.timestamp(), start_today.timestamp())
        wins = [p for p in closed if p.realized_pnl > 0]
        metrics["dpof_closed_yesterday_count"] = len(closed)
        metrics["dpof_closed_yesterday_wins"] = len(wins)
        metrics["dpof_realized_yesterday"] = sum(p.realized_pnl for p in closed)
    except Exception as e:
        logger.debug(f"DPOF readmodel unavailable for daily metrics: {e}")

    try:
        from agent.integrations.market_regime import get_market_regime
        regime = get_market_regime()
        if regime:
            metrics["market_regime"] = regime.get("regime")
    except Exception as e:
        logger.debug(f"Market regime unavailable: {e}")

    return metrics


def _get_sectors_from_positions(positions: list[dict]) -> list[str]:
    """Extract unique sectors from open positions."""
    sectors = set()
    for pos in positions:
        symbol = pos.get("symbol", "")
        sector = TICKER_SECTOR.get(symbol, "")
        if sector:
            sectors.add(sector)
    return list(sectors)


async def _fetch_sector_news(positions: list[dict]) -> str:
    """Fetch broad sector and market news via web search."""
    news_parts = []
    date_str = datetime.now(_NY).strftime("%B %d %Y")

    queries = [
        f"stock market pre-market today {date_str}",
        f"S&P 500 futures pre-market {date_str}",
    ]

    # Add sector-specific query for top position sector
    sectors = _get_sectors_from_positions(positions)
    if sectors:
        queries.append(f"{sectors[0]} stocks news today {date_str}")

    # Add geopolitical query if defense/energy positions exist
    geo_sectors = {"defense", "energy", "commodities"}
    if geo_sectors.intersection(sectors):
        queries.append(f"geopolitical market risk today {date_str}")

    for query in queries:
        try:
            allowed, msg = await budget_tracker.check_budget("report_web_search")
            if not allowed:
                logger.warning(f"Web search budget exhausted: {msg}")
                break

            from agent.integrations.web_search import search_web
            results = await search_web(query, max_results=3)
            await budget_tracker.record_usage("report_web_search")

            # Lazy staleness import (consistent with all other modules)
            try:
                from agent.integrations.news_update import _url_is_stale, _snippet_is_stale
                _has_staleness = True
            except ImportError:
                _has_staleness = False

            for r in (results or []):
                title = r.get("title", "")
                snippet = r.get("snippet", r.get("body", ""))
                url = r.get("url", r.get("href", ""))
                if _has_staleness:
                    if _url_is_stale(url):
                        continue
                    if _snippet_is_stale(title, snippet):
                        continue
                if title or snippet:
                    news_parts.append(f"- {title}: {snippet}" + (f" ({url})" if url else ""))

        except Exception as e:
            logger.warning(f"Sector news search failed for '{query}': {e}")

        await asyncio.sleep(0.5)

    raw = "\n".join(news_parts) if news_parts else "No sector news available."
    return sanitize_ingested_content(raw, source="ddg_search") if news_parts else raw


async def _fetch_ticker_news(tickers: list[str]) -> str:
    """Fetch overnight news for specific tickers."""
    news_parts = []
    date_str = datetime.now(_NY).strftime("%B %d %Y")

    for ticker in tickers[:5]:  # Cap at 5 tickers
        try:
            allowed, msg = await budget_tracker.check_budget("report_web_search")
            if not allowed:
                break

            from agent.integrations.web_search import search_web
            query = f"{ticker} stock news today {date_str}"
            results = await search_web(query, max_results=3)
            await budget_tracker.record_usage("report_web_search")

            # Lazy staleness import
            try:
                from agent.integrations.news_update import _url_is_stale, _snippet_is_stale
                _has_staleness = True
            except ImportError:
                _has_staleness = False

            ticker_news = []
            for r in (results or []):
                title = r.get("title", "")
                snippet = r.get("snippet", r.get("body", ""))
                url = r.get("url", r.get("href", ""))
                if _has_staleness:
                    if _url_is_stale(url):
                        continue
                    if _snippet_is_stale(title, snippet):
                        continue
                if title or snippet:
                    ticker_news.append(f"  - {title}: {snippet}")

            if ticker_news:
                news_parts.append(f"{ticker}:\n" + "\n".join(ticker_news))

        except Exception as e:
            logger.warning(f"Ticker news fetch failed for {ticker}: {e}")

        await asyncio.sleep(0.5)

    raw = "\n".join(news_parts) if news_parts else "No ticker-specific news available."
    return sanitize_ingested_content(raw, source="ddg_search") if news_parts else raw


def _format_positions_table(positions: list[dict]) -> str:
    """Format CLIO's open DPOF positions as a readable table for the prompt."""
    if not positions:
        return "No open positions — CLIO's DPOF book is flat."

    lines = []
    for pos in positions:
        symbol = pos.get("symbol", "?")
        contract = pos.get("contract", symbol)
        direction = str(pos.get("direction", "?")).upper()
        entry = pos.get("entry_price", "?")
        pnl = pos.get("unrealized_pnl", 0)
        pnl_str = f"${pnl:+.2f}" if isinstance(pnl, (int, float)) else str(pnl)
        note = f" {pos['note']}" if pos.get("note") else ""
        lines.append(f"  {symbol} {direction} {contract} @ {entry} | P&L: {pnl_str}{note}")

    return "\n".join(lines)


def _format_metrics(metrics: dict) -> str:
    """Format CLIO-native daily metrics (Alpaca account + DPOF ledger) for the prompt."""
    if not metrics:
        return "No daily metrics available."

    equity = metrics.get("equity")
    daily_pnl = metrics.get("daily_pnl")
    equity_str = f"${equity:,.2f}" if isinstance(equity, (int, float)) else "?"
    pnl_str = f"${daily_pnl:+,.2f}" if isinstance(daily_pnl, (int, float)) else "?"

    closed = metrics.get("dpof_closed_yesterday_count", 0)
    wins = metrics.get("dpof_closed_yesterday_wins", 0)
    realized = metrics.get("dpof_realized_yesterday", 0.0)
    realized_str = f"${realized:+,.2f}" if isinstance(realized, (int, float)) else "?"

    lines = [
        f"  Equity: {equity_str}",
        f"  Yesterday P&L: {pnl_str}",
        f"  DPOF trades closed yesterday: {closed} ({wins} wins) | realized: {realized_str}",
    ]
    regime = metrics.get("market_regime")
    if regime:
        lines.append(f"  Market Regime: {regime}")
    return "\n".join(lines)


async def _get_uw_flow_context(tickers: list[str]) -> str:
    """Fetch Unusual Whales flow intelligence for the pre-market brief."""
    try:
        from agent.integrations.uw_flow_intel import get_uw_signals, get_ticker_flow
        data = await get_uw_signals()
        if not data:
            return ""

        parts = []

        # Market Tide (from data["market"] — keys: net_flow, sentiment, ratio)
        market = data.get("market", {})
        if market:
            sentiment = market.get("sentiment", "N/A")
            net_flow = market.get("net_flow", 0)
            ratio = market.get("ratio", "N/A")
            parts.append(
                f"Market Tide: {sentiment} | Net Flow: ${net_flow:,.0f} | C/P Ratio: {ratio}"
            )

        # Top movers from composite signals list
        signals = data.get("signals", [])
        bulls = sorted([s for s in signals if s.get("score", 0) > 0.1],
                       key=lambda s: s.get("score", 0), reverse=True)[:3]
        bears = sorted([s for s in signals if s.get("score", 0) < -0.1],
                       key=lambda s: s.get("score", 0))[:3]
        if bulls:
            bull_str = ", ".join(f"{s.get('ticker', '?')}({s.get('score', 0):+.2f})" for s in bulls)
            parts.append(f"Top Bullish Flow: {bull_str}")
        if bears:
            bear_str = ", ".join(f"{s.get('ticker', '?')}({s.get('score', 0):+.2f})" for s in bears)
            parts.append(f"Top Bearish Flow: {bear_str}")

        # Per-ticker flow for open positions (top 5)
        ticker_flows = []
        for t in tickers[:5]:
            tf = get_ticker_flow(t)
            if tf:
                score = tf.get("score", 0)
                direction = tf.get("dir", "N/A")
                dp_bias = tf.get("dp_bias", "")
                flow_prem = tf.get("flow_prem", 0)
                flow_parts = [f"{direction}({score:+.2f})"]
                if dp_bias:
                    flow_parts.append(f"DP={dp_bias}")
                if flow_prem:
                    flow_parts.append(f"NetPrem=${flow_prem:,.0f}")
                ticker_flows.append(f"  {t}: {', '.join(flow_parts)}")
        if ticker_flows:
            parts.append("Position Flow Signals:\n" + "\n".join(ticker_flows))

        return "\n".join(parts) if parts else ""
    except Exception as e:
        logger.debug(f"UW flow context unavailable: {e}")
        return ""


# CC-CLIO-FD-3: Premarket fundamental snapshot
def _fd_premarket_snapshot(ticker: str) -> str:
    """Quick fundamental snapshot for premarket scan tickers.

    CC-CLIO-FD-3: Fetches key metrics (MCap, PE, ROE, RevG) from
    Financial Datasets API for position tickers in the premarket brief.
    BC#112: Snapshot field names use full words (price_to_earnings_ratio, etc.).
    """
    if os.environ.get("CLIO_FD_PREMARKET_PROFILE", "0") != "1":
        return ""
    try:
        from agent.integrations.fd_client import get_fd_client
        client = get_fd_client()
        snap = client.get_financial_metrics_snapshot(ticker)
        if not snap or (isinstance(snap, dict) and "_error" in snap):
            return ""

        # BC#112: Snapshot uses full-word field names
        # Type guards: API may return strings/None for numeric fields
        parts = []
        mcap = snap.get("market_cap")
        if isinstance(mcap, (int, float)) and mcap > 0:
            parts.append(f"MCap=${mcap / 1e9:.1f}B" if mcap > 1e9 else f"MCap=${mcap / 1e6:.0f}M")
        pe = snap.get("price_to_earnings_ratio")
        if isinstance(pe, (int, float)):
            parts.append(f"PE={pe:.1f}")
        roe = snap.get("return_on_equity")
        if isinstance(roe, (int, float)):
            parts.append(f"ROE={roe:.1%}")
        rev_g = snap.get("revenue_growth")
        if isinstance(rev_g, (int, float)):
            parts.append(f"RevG={rev_g:.1%}")

        return f"[FD: {' | '.join(parts)}]" if parts else ""
    except Exception as e:
        logger.debug("CC-CLIO-FD-3: Premarket snapshot failed for %s: %s", ticker, e)
        return ""


def _build_fd_premarket_section(tickers: list[str]) -> str:
    """Build FD fundamental context for all premarket position tickers. CC-CLIO-FD-3."""
    if os.environ.get("CLIO_FD_PREMARKET_PROFILE", "0") != "1":
        return ""

    if os.environ.get("CLIO_FD_LOG_ONLY", "1") == "1":
        logger.info("CC-CLIO-FD-3 [LOG_ONLY] would inject premarket FD for %s", tickers[:5])
        return ""

    snapshots = []
    for t in tickers[:5]:
        snap = _fd_premarket_snapshot(t)
        if snap:
            snapshots.append(f"  {t}: {snap}")

    if not snapshots:
        return ""

    section = "\n\nFUNDAMENTAL SNAPSHOTS (Financial Datasets):\n" + "\n".join(snapshots)
    logger.info("CC-CLIO-FD-3: Premarket FD section for %d tickers (%d chars)",
                len(snapshots), len(section))
    return section


def _get_athena_macro_outlook(positions: list[dict]) -> str:
    """
    Fetch macro + position forecasts from Athena vault for pre-market context.
    Returns formatted string for injection into user_message.
    Gracefully returns empty string if bridge unavailable or disabled.
    """
    try:
        if os.environ.get("CLIO_ATHENA_ENABLED", "0") != "1":
            return ""

        athena = _get_athena()
        if athena is None or not athena.is_available():
            return ""

        # Fetch all available macro forecasts from vault
        macro_forecasts = athena.forecast_macro()
        if not macro_forecasts:
            return ""

        lines = ["ATHENA FORWARD OUTLOOK (TimesFM vault forecasts):"]

        # Format macro forecasts
        for indicator, forecast in sorted(macro_forecasts.items()):
            if forecast is None:
                continue
            alignment = forecast.get("directional_alignment", 0.0)
            label = forecast.get("directional_label", "NEUTRAL")
            horizon = forecast.get("horizon_days", 0)
            scenarios = forecast.get("scenarios", {})
            worst = scenarios.get("worst", 0.0)
            best = scenarios.get("best", 0.0)
            if isinstance(worst, (int, float)) and isinstance(best, (int, float)):
                lines.append(
                    f"  {indicator}: {label} (alignment={alignment:+.2f}, "
                    f"horizon={horizon}d, worst={worst:.4f}, best={best:.4f})"
                )
            else:
                lines.append(f"  {indicator}: {label} (alignment={alignment:+.2f})")

        # Position alignment checks
        if positions:
            lines.append("  Position Alignment:")
            for pos in positions[:5]:
                ticker = pos.get("symbol", "")
                if not ticker:
                    continue
                forecast = athena.forecast_ticker(ticker)
                if forecast is None:
                    continue
                label = forecast.get("directional_label", "NEUTRAL")
                alignment = forecast.get("directional_alignment", 0.0)
                implied_vol = forecast.get("implied_vol", 0.0)
                if isinstance(alignment, (int, float)) and isinstance(implied_vol, (int, float)):
                    lines.append(
                        f"    {ticker}: {label} (alignment={alignment:+.2f}, "
                        f"IV={implied_vol:.1%})"
                    )

        outlook_text = "\n".join(lines)

        if os.environ.get("CLIO_ATHENA_LOG_ONLY", "1") == "1":
            logger.info("[ATHENA LOG-ONLY] Macro outlook:\n%s", outlook_text)
            return ""  # Don't inject in log-only mode

        return outlook_text

    except Exception as e:
        logger.debug("_get_athena_macro_outlook() failed: %s", e)
        return ""


async def _generate_report(positions: list[dict], metrics: dict,
                           sector_news: str, ticker_news: str,
                           uw_flow: str = "", athena_outlook: str = "") -> str | None:
    """Call Claude Opus to generate the pre-market scan report."""
    allowed, msg = await budget_tracker.check_budget("premarket_scan")
    if not allowed:
        logger.warning(f"Pre-market scan budget exhausted: {msg}")
        return None

    positions_table = _format_positions_table(positions)
    metrics_text = _format_metrics(metrics)
    now_str = datetime.now(_NY).strftime("%A, %B %d, %Y at %I:%M %p ET")

    try:
        from agent.security import sanitize_ingested_content as _sic
        uw_section = f"\n\nOPTIONS FLOW INTELLIGENCE (Unusual Whales):\n{_sic(uw_flow[:2000], source='uw_flow')}" if uw_flow else ""
    except ImportError:
        uw_section = f"\n\nOPTIONS FLOW INTELLIGENCE (Unusual Whales):\n{uw_flow[:2000]}" if uw_flow else ""

    # CC-CLIO-FD-3: Fundamental snapshots for position tickers
    # FD client is sync httpx — MUST use run_in_executor to avoid blocking event loop
    pos_tickers = [p.get("symbol", "") for p in positions if p.get("symbol")]
    try:
        _fd_loop = asyncio.get_running_loop()
        # S159 M-22: Timeout to prevent hung FD calls from blocking premarket scan
        fd_section = await asyncio.wait_for(
            _fd_loop.run_in_executor(None, _build_fd_premarket_section, pos_tickers),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        logger.warning("S159 M-22: FD premarket snapshot timed out after 30s")
        fd_section = ""
    except Exception as e:
        logger.debug("CC-CLIO-FD-3: Premarket FD section failed: %s", e)
        fd_section = ""

    athena_section = f"\n\n{athena_outlook}" if athena_outlook else ""

    user_message = (
        f"Date: {now_str}\n\n"
        f"OPEN POSITIONS (CLIO's own DPOF SPY/QQQ options book):\n{positions_table}\n\n"
        f"YESTERDAY'S METRICS:\n{metrics_text}\n\n"
        f"SECTOR & MARKET NEWS:\n{sector_news[:4000]}\n\n"
        f"TICKER-SPECIFIC NEWS:\n{ticker_news[:3000]}"
        f"{uw_section}"
        f"{fd_section}"
        f"{athena_section}"
    )

    try:
        client = _get_client()
        loop = asyncio.get_running_loop()

        # S134: Escalating retry for 529 Overloaded (SDK default 0.4s+0.9s is too short)
        _RETRY_DELAYS = (10, 30)
        response = None
        for _attempt in range(1 + len(_RETRY_DELAYS)):
            try:
                response = await loop.run_in_executor(None, lambda: client.messages.create(
                    model=config.judgment_model,  # S201: Fable 5 judgment tier (bridge-facing report)
                    max_tokens=4096,
                    system=cached_system_block(PREMARKET_SYSTEM_PROMPT),
                    messages=[{"role": "user", "content": user_message}],
                ))
                break
            except Exception as _e:
                _is_529 = getattr(_e, "status_code", None) == 529
                if _is_529 and _attempt < len(_RETRY_DELAYS):
                    _delay = _RETRY_DELAYS[_attempt]
                    logger.warning(
                        f"Pre-market scan: 529 overloaded (attempt {_attempt + 1}/"
                        f"{1 + len(_RETRY_DELAYS)}), retrying in {_delay}s..."
                    )
                    await asyncio.sleep(_delay)
                else:
                    raise

        if response is None:
            return None

        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

        result = text.strip() if text.strip() else None
        if result:
            await budget_tracker.record_usage("premarket_scan")
        return result

    except Exception as e:
        logger.error(f"Pre-market scan Claude call failed: {e}", exc_info=True)
        return None


def _write_to_bridge(report_text: str, positions: list[dict]) -> None:
    """Write report to the Hydrinia file bridge — no-op unless explicitly re-enabled."""
    try:
        from agent.hydrinia_bridge import bridge_enabled
        if not bridge_enabled():
            return
    except Exception as e:
        logger.debug(f"Hydrinia bridge check failed, skipping bridge write: {e}")
        return
    try:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        timestamp = datetime.now(_NY).strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_premarket.json"

        entry = {
            "id": f"report_{timestamp}_premarket",
            "type": "premarket_scan",
            "created": datetime.now(_NY).isoformat(),
            "content": report_text,
            "positions_count": len(positions),
            "tickers": [p.get("symbol", "") for p in positions],
        }

        file_path = os.path.join(REPORTS_DIR, filename)
        tmp = file_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(entry, f, indent=2, default=str)
        os.replace(tmp, file_path)

        logger.info(f"Pre-market scan written to bridge: {filename}")
    except Exception as e:
        logger.error(f"Failed to write pre-market scan to bridge: {e}")


async def _send_report(context, report_text: str) -> None:
    """Send report to Telegram."""
    try:
        from agent.integrations.scheduled_tasks import _get_chat_ids, _safe_send
        chat_ids = _get_chat_ids()
        if not chat_ids:
            return

        header = f"PRE-MARKET SCAN — {datetime.now(_NY).strftime('%A %b %d, %Y')}\n\n"
        full_text = header + report_text

        if len(full_text) <= 4000:
            for cid in chat_ids:
                await _safe_send(context.bot, cid, text=full_text)
        else:
            chunks = [full_text[i:i + 4000] for i in range(0, len(full_text), 4000)]
            for cid in chat_ids:
                for chunk in chunks:
                    await _safe_send(context.bot, cid, text=chunk)

    except Exception as e:
        logger.warning(f"Failed to send pre-market scan to Telegram: {e}")


async def run_premarket_scan(context) -> None:
    """Main entry point: comprehensive pre-market scan.

    Called by JobQueue at 9:00 AM ET on weekdays.
    """
    now = datetime.now(_NY)
    if now.weekday() >= 5:
        return

    logger.info("Starting pre-market scan...")

    try:
        # Gather data
        positions = await _get_open_positions()
        metrics = await _get_daily_metrics()

        tickers = [p.get("symbol", "") for p in positions if p.get("symbol")]
        # S129: Filter invalid tickers (OCC options format, phantom tickers)
        _valid_ticker_re = re.compile(r'^/?[A-Z]{1,5}(-[A-Z]{1,4})?$')
        tickers = [t for t in tickers if _valid_ticker_re.match(t)]

        # Fetch news in parallel-ish (sequential with small delays for rate limits)
        sector_news = await _fetch_sector_news(positions)
        ticker_news = await _fetch_ticker_news(tickers)

        # Fetch UW flow intelligence (fire-and-forget on failure)
        uw_flow = await _get_uw_flow_context(tickers)

        # Athena forward outlook (Phase 1 — fire-and-forget)
        athena_outlook = _get_athena_macro_outlook(positions)

        # Generate report via Claude Opus
        report = await _generate_report(
            positions, metrics, sector_news, ticker_news,
            uw_flow=uw_flow, athena_outlook=athena_outlook,
        )

        if not report:
            logger.warning("Pre-market scan: no report generated (budget exhausted or API error)")
            return

        # Deliver to both channels
        _write_to_bridge(report, positions)
        await _send_report(context, report)

        logger.info(f"Pre-market scan complete: {len(positions)} positions, report delivered")

    except Exception as e:
        logger.error(f"Pre-market scan failed: {e}", exc_info=True)
