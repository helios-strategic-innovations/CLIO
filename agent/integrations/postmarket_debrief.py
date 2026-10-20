"""
OpenClaw/Moltbot — Post-Market Debrief (Session 85)

Runs at 5:00 PM ET weekdays. Produces a comprehensive end-of-day trading debrief
with reflections and trade ideas, delivered to Telegram (+ the Hydrinia file
bridge, only if explicitly re-enabled via CLIO_HYDRINIA_BRIDGE_ENABLED — see
agent/hydrinia_bridge.py).

This describes CLIO's own DPOF book (SPY/QQQ long options on the CLIO Alpaca
paper account, agent/dpof/, typically 1-day expiries flattened by 15:50 ET) —
never Hydrinia's frozen state files.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import anthropic

from agent.config import config
from agent.integrations.alpaca_values import AlpacaValueError, normalize_position_side
from agent.prompts.system_prompt import cached_system_block
from agent.rate_limiter import budget_tracker
from agent.security import sanitize_ingested_content

logger = logging.getLogger("openclaw.postmarket_debrief")


def _get_athena():
    """Lazy import for Athena bridge (Phase 1 integration)."""
    try:
        from agent.integrations.athena_bridge import get_athena_bridge
        return get_athena_bridge()
    except Exception:
        return None

_NY = ZoneInfo("America/New_York")

KNOWLEDGE_INDEX_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "knowledge", "knowledge_index.json",
)
REPORTS_DIR = os.path.join(config.tradingbot_dir, "moltbot_knowledge", "reports", "postmarket")

_cached_client: anthropic.Anthropic | None = None
_client_lock = __import__("threading").Lock()


def _get_client() -> anthropic.Anthropic:
    global _cached_client
    if _cached_client is None:
        with _client_lock:
            if _cached_client is None:
                _cached_client = anthropic.Anthropic(**config.anthropic_client_kwargs(timeout=120))
    return _cached_client


POSTMARKET_SYSTEM_PROMPT = """You are writing a comprehensive end-of-day trading debrief for a quantitative trading operation. Be reflective, insightful, and data-driven.

Write in plain text only — NO markdown formatting, NO headers with #, NO bold/italic markers.

Structure your debrief as follows:

DAY SUMMARY
2-3 sentences: total P&L, equity change, key performance numbers, overall assessment.

TRADE REVIEW
For each trade today: ticker, direction, entry/exit prices, P&L, duration, and a brief note on what went right or wrong.

MARKET CONTEXT
What drove the market today. Sector performance, key movers, and how it affected our positions.

POSITION OUTLOOK
For each open position: overnight risk assessment and any catalysts to watch.

TRADE IDEAS
1-3 ideas for tomorrow based on today's patterns. Ticker, direction, thesis, catalyst.

REFLECTIONS
2-3 sentences: lessons learned, behavioral observations, what to do differently tomorrow."""


def _get_todays_activity() -> tuple[list, list, list]:
    """CLIO's own DPOF activity for today: opened, closed, currently open.

    CLIO-native (agent/dpof/readmodel.py) — replaces Hydrinia's position_log.json.
    Returns lists of PositionRecord (never dicts); readmodel never raises, so a
    read failure degrades to three empty lists rather than an exception.
    """
    try:
        from agent.dpof import readmodel
        return readmodel.opened_today(), readmodel.closed_today(), readmodel.open_positions()
    except Exception as e:
        logger.debug(f"DPOF readmodel unavailable for today's activity: {e}")
        return [], [], []


async def _get_daily_metrics() -> dict:
    """CLIO-native day summary: Alpaca account equity/last_equity (true daily P&L)
    + DPOF's realized P&L today (readmodel.snapshot) + market regime.
    Replaces Hydrinia's daily_metrics.json.
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
        snap = readmodel.snapshot()
        metrics["dpof_realized_today"] = snap.get("realized_pnl_today", 0.0)
        metrics["dpof_closed_today_count"] = snap.get("closed_today_count", 0)
        metrics["dpof_closed_today_wins"] = snap.get("closed_today_wins", 0)
        metrics["dpof_open_count"] = snap.get("open_count", 0)
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


def _get_recent_knowledge() -> str:
    """Get the last 3 knowledge entries from today."""
    if not os.path.isfile(KNOWLEDGE_INDEX_PATH):
        return "No recent knowledge entries."
    try:
        import fcntl
        lock_path = KNOWLEDGE_INDEX_PATH + ".lock"
        with open(lock_path, "a+") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_SH)
            try:
                with open(KNOWLEDGE_INDEX_PATH, "r") as f:
                    data = json.load(f)
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)

        entries = data.get("entries", []) if isinstance(data, dict) else data
        if not isinstance(entries, list):
            return "No recent knowledge entries."

        today_str = datetime.now(_NY).strftime("%Y-%m-%d")
        recent = [
            e for e in entries
            if e.get("created", "").startswith(today_str)
        ][-3:]

        if not recent:
            return "No knowledge entries from today."

        parts = []
        for e in recent:
            title = e.get("title", "Untitled")
            category = e.get("category", "")
            parts.append(f"  - [{category}] {title}")
        return "\n".join(parts)

    except Exception as e:
        logger.warning(f"Failed to read knowledge index: {e}")
        return "Knowledge index unavailable."


def _format_todays_trades(opens: list, closes: list) -> str:
    """Format today's DPOF trades for the prompt (CLIO-native, readmodel-sourced).

    PositionRecord has no standalone exit price (multi-leg structures make one
    ambiguous) — realized P&L + exit_reason from format_position_line() convey
    the outcome without fabricating a number.
    """
    try:
        from agent.dpof import readmodel
    except Exception as e:
        logger.debug(f"DPOF readmodel unavailable for trade formatting: {e}")
        return f"TRADES OPENED TODAY ({len(opens)}): unavailable\nTRADES CLOSED TODAY ({len(closes)}): unavailable"

    lines = [f"TRADES OPENED TODAY ({len(opens)}):"]
    if opens:
        for p in opens:
            lines.append("  " + readmodel.format_position_line(p))
    else:
        lines.append("  None")

    lines.append(f"\nTRADES CLOSED TODAY ({len(closes)}):")
    if closes:
        for p in closes:
            line = readmodel.format_position_line(p)
            if isinstance(p.entry_time, (int, float)) and isinstance(p.closed_time, (int, float)):
                held_sec = p.closed_time - p.entry_time
                hours = held_sec / 3600
                if hours < 1:
                    duration = f"{int(held_sec / 60)}min"
                elif hours < 24:
                    duration = f"{hours:.1f}h"
                else:
                    duration = f"{int(hours / 24)}d"
                line += f" held {duration}"
            lines.append("  " + line)
    else:
        lines.append("  None")

    return "\n".join(lines)


async def _get_alpaca_unrealized() -> dict[str, dict]:
    """Fetch live Alpaca positions for unrealized P&L."""
    try:
        from agent.integrations.alpaca import _get_trading_client, _retry_alpaca_call
        client = _get_trading_client()
        positions = await _retry_alpaca_call(client.get_all_positions) or []
        result = {}
        for pos in positions:
            symbol = getattr(pos, "symbol", "")
            try:
                side = normalize_position_side(getattr(pos, "side", None))
            except AlpacaValueError:
                logger.warning("Post-market debrief skipped unknown side for %s", symbol)
                continue
            try:
                unrealized_pl = float(getattr(pos, "unrealized_pl", 0) or 0)
            except (ValueError, TypeError):
                unrealized_pl = 0.0
            try:
                current_price = float(getattr(pos, "current_price", 0) or 0)
            except (ValueError, TypeError):
                current_price = 0.0
            result[symbol] = {
                "unrealized_pl": unrealized_pl,
                "current_price": current_price,
                "side": side,
            }
        return result
    except Exception as e:
        logger.debug(f"Alpaca position fetch failed: {e}")
        return {}


def _format_open_positions(positions: list, alpaca_data: dict[str, dict]) -> str:
    """Format CLIO's DPOF open positions for the prompt.

    Normally empty after the 15:50 ET flatten — say so explicitly when flat,
    and flag loudly if anything is still open after close. Overlays live
    Alpaca unrealized P&L, matched by OCC leg symbol.
    """
    if not positions:
        return "Flat — CLIO's DPOF book has no open positions (expected after the 15:50 ET flatten)."

    try:
        from agent.dpof import readmodel
    except Exception as e:
        logger.debug(f"DPOF readmodel unavailable for open-position formatting: {e}")
        readmodel = None

    lines = [
        f"WARNING: {len(positions)} DPOF position(s) still open after close — "
        f"should have been flattened by 15:50 ET."
    ]
    for pos in positions:
        unrealized = None
        for leg in (pos.legs or []):
            sym = leg.get("symbol") if isinstance(leg, dict) else None
            if sym and sym in alpaca_data:
                unrealized = alpaca_data[sym].get("unrealized_pl")
                break
        pnl_str = f"${unrealized:+,.2f}" if isinstance(unrealized, (int, float)) else "unavailable"
        label = readmodel.format_position_line(pos) if readmodel else f"{pos.underlying} {pos.direction}"
        lines.append(f"  {label} | unrealized: {pnl_str}")
    return "\n".join(lines)


def _format_metrics(metrics: dict) -> str:
    """Format CLIO-native daily metrics (Alpaca account + DPOF ledger) for the prompt."""
    if not metrics:
        return "No daily metrics available."
    equity = metrics.get("equity")
    daily_pnl = metrics.get("daily_pnl")
    equity_str = f"${equity:,.2f}" if isinstance(equity, (int, float)) else "?"
    pnl_str = f"${daily_pnl:+,.2f}" if isinstance(daily_pnl, (int, float)) else "?"

    realized = metrics.get("dpof_realized_today", 0.0)
    realized_str = f"${realized:+,.2f}" if isinstance(realized, (int, float)) else "?"
    closed = metrics.get("dpof_closed_today_count", 0)
    wins = metrics.get("dpof_closed_today_wins", 0)

    lines = [
        f"  Equity: {equity_str}",
        f"  Daily P&L: {pnl_str}",
        f"  DPOF trades closed today: {closed} ({wins} wins) | realized: {realized_str}",
    ]
    regime = metrics.get("market_regime")
    if regime:
        lines.append(f"  Market Regime: {regime}")
    return "\n".join(lines)


async def _fetch_market_wrap(closes: list) -> str:
    """Fetch end-of-day market wrap news. `closes` is a list of DPOF PositionRecord."""
    news_parts = []
    date_str = datetime.now(_NY).strftime("%B %d %Y")

    queries = [
        f"stock market close today {date_str}",
        f"S&P 500 close analysis {date_str}",
    ]

    # Add biggest mover query
    if closes:
        biggest = max(closes, key=lambda p: abs(getattr(p, "realized_pnl", 0) or 0))
        ticker = getattr(biggest, "underlying", "")
        if ticker:
            queries.append(f"{ticker} stock today {date_str}")

    for query in queries:
        try:
            allowed, msg = await budget_tracker.check_budget("report_web_search")
            if not allowed:
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
                    news_parts.append(f"- {title}: {snippet}")

        except Exception as e:
            logger.warning(f"Market wrap search failed for '{query}': {e}")

        await asyncio.sleep(0.5)

    raw = "\n".join(news_parts) if news_parts else "No market wrap news available."
    return sanitize_ingested_content(raw, source="ddg_search") if news_parts else raw


def _fd_position_fundamentals(tickers: list) -> str:
    """CC-CLIO-FD-5: Add fundamental snapshots for current portfolio positions."""
    if os.environ.get("CLIO_FD_POSTMARKET_FUNDAMENTALS", "0") != "1" or not tickers:
        return ""
    try:
        from agent.integrations.fd_client import get_fd_client
        client = get_fd_client()

        lines = ["POSITION FUNDAMENTALS (FD):"]
        for ticker in tickers[:10]:
            try:
                snap = client.get_financial_metrics_snapshot(ticker)
                if not snap or (isinstance(snap, dict) and "_error" in snap):
                    continue
                # BC#112: snapshot uses long field names; float() for type safety
                pe = snap.get("price_to_earnings_ratio")
                roe = snap.get("return_on_equity")
                rev_g = snap.get("revenue_growth")
                parts = []
                if isinstance(pe, (int, float)):
                    parts.append(f"PE={pe:.1f}")
                if isinstance(roe, (int, float)):
                    parts.append(f"ROE={roe:.1%}")
                if isinstance(rev_g, (int, float)):
                    parts.append(f"RevG={rev_g:.1%}")
                if parts:
                    lines.append(f"  {ticker}: {' | '.join(parts)}")
            except (ValueError, TypeError, KeyError):
                continue

        return "\n".join(lines) if len(lines) > 1 else ""
    except Exception as e:
        logger.debug("CC-CLIO-FD-5: Position fundamentals failed: %s", e)
        return ""


def _get_overnight_risk_context(open_positions: list) -> str:
    """
    Fetch Athena vault forecasts and exit signals for overnight DPOF positions.
    `open_positions` is a list of DPOF PositionRecord. Returns formatted
    context string or empty if unavailable.
    """
    try:
        if os.environ.get("CLIO_ATHENA_ENABLED", "0") != "1":
            return ""

        athena = _get_athena()
        if athena is None or not athena.is_available():
            return ""

        risk_lines = ["OVERNIGHT RISK CONTEXT (Athena/TimesFM vault forecasts):"]

        # Fetch forecasts and exit signals
        for pos in open_positions[:7]:
            ticker = getattr(pos, "underlying", "")
            if not ticker:
                continue

            forecast = athena.forecast_ticker(ticker)
            exit_signal = athena.check_exit_signal(ticker)

            if forecast is None:
                continue

            label = forecast.get("directional_label", "NEUTRAL")
            alignment = forecast.get("directional_alignment", 0.0)
            implied_vol = forecast.get("implied_vol", 0.0)
            scenarios = forecast.get("scenarios", {})

            # Format position line
            if isinstance(alignment, (int, float)) and isinstance(implied_vol, (int, float)):
                line = f"  {ticker}: {label} (alignment={alignment:+.2f}, IV={implied_vol:.1%})"
                worst = scenarios.get("worst")
                best = scenarios.get("best")
                if isinstance(worst, (int, float)) and isinstance(best, (int, float)):
                    line += f" range=[{worst:+.1%}, {best:+.1%}]"
                risk_lines.append(line)

            # Add exit signal if triggered
            if exit_signal is not None and exit_signal.get("signal") == "EXIT":
                risk_lines.append(
                    f"    EXIT SIGNAL: alignment={exit_signal.get('directional_alignment', 0):.2f}, "
                    f"spread={exit_signal.get('quantile_spread_pct', 0):.1%}"
                )
            elif exit_signal is not None and exit_signal.get("signal") == "HOLD_MONITOR":
                risk_lines.append(
                    f"    MONITOR: alignment weakening ({exit_signal.get('directional_alignment', 0):.2f})"
                )

        # Scenario paths for top 3 positions
        scenario_lines = []
        for pos in open_positions[:3]:
            ticker = getattr(pos, "underlying", "")
            if not ticker:
                continue
            scenario_data = athena.get_scenario_paths(ticker)
            if scenario_data is None:
                continue
            scenarios = scenario_data.get("scenarios", {})
            if scenarios:
                worst = scenarios.get("worst", 0.0)
                base = scenarios.get("base", 0.0)
                best = scenarios.get("best", 0.0)
                if all(isinstance(v, (int, float)) for v in (worst, base, best)):
                    scenario_lines.append(
                        f"  {ticker}: worst={worst:+.1%} base={base:+.1%} best={best:+.1%}"
                    )

        if scenario_lines:
            risk_lines.append("  Scenario Endpoints:")
            risk_lines.extend(scenario_lines)

        if len(risk_lines) <= 1:
            return ""

        risk_text = "\n".join(risk_lines)

        if os.environ.get("CLIO_ATHENA_LOG_ONLY", "1") == "1":
            logger.info("[ATHENA LOG-ONLY] Overnight risk:\n%s", risk_text)
            return ""  # Don't inject in log-only mode

        return risk_text

    except Exception as e:
        logger.debug("_get_overnight_risk_context() failed: %s", e)
        return ""


async def _generate_report(todays_opens: list, todays_closes: list,
                           open_positions: list, metrics: dict,
                           market_wrap: str, knowledge: str,
                           alpaca_data: dict[str, dict],
                           fd_fundamentals: str = "",
                           athena_risk: str = "") -> str | None:
    """Call Claude to generate the post-market debrief for CLIO's own DPOF book."""
    allowed, msg = await budget_tracker.check_budget("postmarket_debrief")
    if not allowed:
        logger.warning(f"Post-market debrief budget exhausted: {msg}")
        return None

    trades_text = _format_todays_trades(todays_opens, todays_closes)
    positions_text = _format_open_positions(open_positions, alpaca_data)
    metrics_text = _format_metrics(metrics)
    now_str = datetime.now(_NY).strftime("%A, %B %d, %Y at %I:%M %p ET")

    user_message = (
        f"Date: {now_str}\n\n"
        f"TODAY'S TRADES (CLIO's own DPOF SPY/QQQ options book):\n{trades_text}\n\n"
        f"DAILY METRICS:\n{metrics_text}\n\n"
        f"END-OF-DAY POSITIONS:\n{positions_text}\n\n"
    )
    if fd_fundamentals:
        user_message += f"{fd_fundamentals}\n\n"
    if athena_risk:
        user_message += f"{athena_risk}\n\n"
    user_message += (
        f"MARKET WRAP NEWS:\n{market_wrap[:4000]}\n\n"
        f"TODAY'S LEARNING INSIGHTS:\n{knowledge}"
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
                    system=cached_system_block(POSTMARKET_SYSTEM_PROMPT),
                    messages=[{"role": "user", "content": user_message}],
                ))
                break
            except Exception as _e:
                _is_529 = getattr(_e, "status_code", None) == 529
                if _is_529 and _attempt < len(_RETRY_DELAYS):
                    _delay = _RETRY_DELAYS[_attempt]
                    logger.warning(
                        f"Post-market debrief: 529 overloaded (attempt {_attempt + 1}/"
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
            await budget_tracker.record_usage("postmarket_debrief")
        return result

    except Exception as e:
        logger.error(f"Post-market debrief Claude call failed: {e}", exc_info=True)
        return None


def _write_to_bridge(report_text: str, opens: int, closes: int, active: int) -> None:
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
        filename = f"{timestamp}_postmarket.json"

        entry = {
            "id": f"report_{timestamp}_postmarket",
            "type": "postmarket_debrief",
            "created": datetime.now(_NY).isoformat(),
            "content": report_text,
            "trades_opened": opens,
            "trades_closed": closes,
            "positions_open": active,
        }

        file_path = os.path.join(REPORTS_DIR, filename)
        tmp = file_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(entry, f, indent=2, default=str)
        os.replace(tmp, file_path)

        logger.info(f"Post-market debrief written to bridge: {filename}")
    except Exception as e:
        logger.error(f"Failed to write post-market debrief to bridge: {e}")


async def _send_report(context, report_text: str) -> None:
    """Send report to Telegram."""
    try:
        from agent.integrations.scheduled_tasks import _get_chat_ids, _safe_send
        chat_ids = _get_chat_ids()
        if not chat_ids:
            return

        header = f"POST-MARKET DEBRIEF — {datetime.now(_NY).strftime('%A %b %d, %Y')}\n\n"
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
        logger.warning(f"Failed to send post-market debrief to Telegram: {e}")


async def run_postmarket_debrief(context) -> None:
    """Main entry point: comprehensive end-of-day debrief.

    Called by JobQueue at 5:00 PM ET on weekdays.
    """
    now = datetime.now(_NY)
    if now.weekday() >= 5:
        return

    logger.info("Starting post-market debrief...")

    try:
        todays_opens, todays_closes, open_positions = _get_todays_activity()
        metrics = await _get_daily_metrics()
        knowledge = _get_recent_knowledge()

        market_wrap = await _fetch_market_wrap(todays_closes)
        alpaca_data = await _get_alpaca_unrealized()

        # CC-CLIO-FD-5: Gather fundamental data for held positions
        fd_fundamentals = ""
        position_tickers = [p.underlying for p in open_positions if getattr(p, "underlying", "")]
        if position_tickers:
            loop = asyncio.get_running_loop()
            try:
                # S159 M-22: Timeout to prevent hung FD calls from blocking postmarket debrief
                fd_fundamentals = await asyncio.wait_for(
                    loop.run_in_executor(None, _fd_position_fundamentals, position_tickers),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                logger.warning("S159 M-22: FD postmarket fundamentals timed out after 30s")
            except Exception as e:
                logger.debug("CC-CLIO-FD-5: Postmarket FD fundamentals failed: %s", e)

        # Athena overnight risk context (Phase 1 — fire-and-forget)
        athena_risk = _get_overnight_risk_context(open_positions)

        report = await _generate_report(
            todays_opens, todays_closes, open_positions,
            metrics, market_wrap, knowledge, alpaca_data,
            fd_fundamentals=fd_fundamentals,
            athena_risk=athena_risk,
        )

        if not report:
            logger.warning("Post-market debrief: no report generated (budget exhausted or API error)")
            return

        _write_to_bridge(report, len(todays_opens), len(todays_closes), len(open_positions))
        await _send_report(context, report)

        logger.info(
            f"Post-market debrief complete: {len(todays_opens)} opens, "
            f"{len(todays_closes)} closes, {len(open_positions)} still open"
        )

    except Exception as e:
        logger.error(f"Post-market debrief failed: {e}", exc_info=True)
