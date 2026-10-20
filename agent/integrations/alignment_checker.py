"""
OpenClaw/Moltbot — Pre-Market Position Alignment Checker (Feature 4)

Runs at 9:00 AM ET weekdays. For each open position, pulls overnight news
and scores alignment between the position thesis and current news.

Severity levels:
  RED    — exit recommended (news strongly contradicts position)
  YELLOW — monitor (ambiguous or weakly contradictory news)
  GREEN  — thesis intact (news supports or is neutral)

Results are appended to learning_log.json under alignment_checks[].
"""

import asyncio
import fcntl
import json
import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import anthropic

from agent.config import config
from agent.rate_limiter import budget_tracker, BudgetExhaustedError
from agent.security import sanitize_ingested_content

logger = logging.getLogger("openclaw.alignment_checker")

_NY = ZoneInfo("America/New_York")

LEARNING_LOG_PATH = os.path.join(config.tradingbot_dir, "state", "learning_log.json")

# Cached Anthropic client
_cached_client: anthropic.Anthropic | None = None
_client_lock = __import__("threading").Lock()


def _get_client() -> anthropic.Anthropic:
    """Get or create cached Anthropic client (thread-safe)."""
    global _cached_client
    if _cached_client is None:
        with _client_lock:
            if _cached_client is None:
                _cached_client = anthropic.Anthropic(**config.anthropic_client_kwargs(timeout=120))
    return _cached_client


ALIGNMENT_PROMPT = """You are a trading analyst checking whether overnight news supports or contradicts an existing position.

POSITION (CLIO's own DPOF book — SPY/QQQ long options):
- Ticker: {ticker}
- Direction: {direction}
- Structure: {category}
- Contract: {contract}
- Entry price: {entry_price}
- Conviction: {conviction}

OVERNIGHT NEWS:
{news}

OPTIONS FLOW SIGNALS (Unusual Whales):
{flow_context}

INSTRUCTIONS:
1. Does this news SUPPORT or CONTRADICT the position thesis?
2. Consider: for a LONG position, bearish news = CONFLICT. For a SHORT position, bullish news = CONFLICT.
3. Factor in options flow signals: dark pool accumulation with premium confirms bullish; distribution with negative premium confirms bearish. Flow that contradicts position direction is a warning signal.
4. "Short-term noise" with no fundamental impact = GREEN.
5. Only flag RED if the news represents a material thesis-breaking development.

Output ONLY valid JSON (no markdown, no explanation):
{{
  "severity": "RED" or "YELLOW" or "GREEN",
  "alignment": "CONFIRM" or "CONFLICT" or "NEUTRAL",
  "rationale": "one sentence explanation",
  "news_summary": "two sentence summary of the relevant news"
}}"""


def _get_open_positions() -> list[dict]:
    """Read open positions from CLIO's own DPOF book (readmodel), one per
    underlying — NOT Hydrinia's position_log.json (Hydrinia is permanently off
    and that file is frozen/stale)."""
    try:
        from agent.dpof import readmodel
    except Exception as e:
        logger.debug(f"DPOF readmodel unavailable: {e}")
        return []

    result = []
    for pos in readmodel.open_positions():
        conviction = None
        try:
            conviction = (pos.entry_signal or {}).get("conviction")
        except AttributeError:
            conviction = None
        result.append({
            "symbol": pos.underlying,
            "direction": pos.direction,
            "category": f"DPOF {pos.structure}" if pos.structure else "DPOF",
            "contract": readmodel.position_label(pos),
            "entry_price": pos.entry_price if pos.entry_price is not None else "unfilled",
            "entry_time": pos.entry_time,
            "conviction": conviction if conviction is not None else "unknown",
            "trade_id": pos.id,
        })
    return result


async def _fetch_news_for_ticker(ticker: str) -> str:
    """Fetch overnight news for a ticker using web search. Returns news text or empty string."""
    try:
        allowed, msg = await budget_tracker.check_budget("report_web_search")
        if not allowed:
            logger.warning(f"Report web search budget exhausted: {msg}")
            return ""
        from agent.integrations.web_search import search_web
        query = f"{ticker} stock news overnight today {datetime.now(_NY).strftime('%B %d %Y')}"
        results = await search_web(query, max_results=3)
        await budget_tracker.record_usage("report_web_search")  # S118 W-010: charge AFTER search

        if not results:
            return ""

        # Filter stale results
        try:
            from agent.integrations.news_update import _url_is_stale, _snippet_is_stale
            results = [
                r for r in results
                if not _url_is_stale(r.get("url", r.get("href", "")))
                and not _snippet_is_stale(r.get("title", ""), r.get("snippet", r.get("body", "")))
            ]
        except ImportError:
            pass

        # Combine search result snippets
        snippets = []
        for r in results:
            title = r.get("title", "")
            snippet = r.get("snippet", r.get("body", ""))
            if title or snippet:
                snippets.append(f"- {title}: {snippet}")

        return "\n".join(snippets) if snippets else ""

    except Exception as e:
        logger.warning(f"News fetch failed for {ticker}: {e}")
        return ""


async def _score_alignment(position: dict, news: str) -> dict:
    """Use Claude Sonnet to score alignment between position and news."""
    ticker = position["symbol"]

    if not news:
        return {
            "ticker": ticker,
            "severity": "YELLOW",
            "alignment": "NEUTRAL",
            "rationale": "No overnight news data available — thesis unverified",
            "news_summary": "No relevant news found for this ticker",
        }

    # Check if position has no documented thesis (orphan position)
    if position.get("category") is None and position.get("conviction") is None:
        return {
            "ticker": ticker,
            "severity": "YELLOW",
            "alignment": "NEUTRAL",
            "rationale": "Orphan position — no documented thesis, manual review required",
            "news_summary": news[:200],
        }

    # Budget check for analysis
    allowed, msg = await budget_tracker.check_budget("learning_analysis")
    if not allowed:
        logger.warning(f"Learning analysis budget exhausted: {msg}")
        return {
            "ticker": ticker,
            "severity": "YELLOW",
            "alignment": "NEUTRAL",
            "rationale": "Budget exhausted — unable to analyze alignment",
            "news_summary": news[:200],
        }

    # Fetch UW flow context for this ticker (fire-and-forget on failure)
    flow_text = "No flow data available."
    try:
        from agent.integrations.uw_flow_intel import get_ticker_flow
        flow = get_ticker_flow(ticker)
        if flow:
            parts = []
            score = flow.get("score", 0)
            direction_label = flow.get("dir", "N/A")
            parts.append(f"Composite: {direction_label} ({score:+.2f})")
            dp_bias = flow.get("dp_bias", "")
            if dp_bias and dp_bias != "N/A":
                parts.append(f"Dark Pool: {dp_bias}")
            flow_prem = flow.get("flow_prem", 0)
            if flow_prem:
                parts.append(f"Flow Premium: ${flow_prem:,.0f}")
            if parts:
                flow_text = "; ".join(parts)
    except Exception as e:
        logger.debug(f"UW flow fetch for {ticker} failed: {e}")

    prompt = ALIGNMENT_PROMPT.format(
        ticker=ticker,
        direction=position.get("direction", "unknown"),
        category=position.get("category", "unknown"),
        contract=position.get("contract", "unknown"),
        entry_price=position.get("entry_price", "unknown"),
        conviction=position.get("conviction", "unknown"),
        news=sanitize_ingested_content(news[:3000], source="ddg_alignment"),
        flow_context=flow_text,
    )

    try:
        client = _get_client()
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, lambda: client.messages.create(
            model=config.fast_model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": "medium"},
        ))
        # S121 W-53: Record budget after API call succeeds (S118 pattern)
        await budget_tracker.record_usage("learning_analysis")

        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

        # Strip code fences if Claude wrapped the JSON
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        result = json.loads(text)
        result["ticker"] = ticker
        # Validate severity
        if result.get("severity") not in ("RED", "YELLOW", "GREEN"):
            result["severity"] = "YELLOW"
        if result.get("alignment") not in ("CONFIRM", "CONFLICT", "NEUTRAL"):
            result["alignment"] = "NEUTRAL"
        return result

    except json.JSONDecodeError as e:
        # S190 H2: JSON parse failure on Claude's alignment response is a real signal
        # degradation — Claude either hit a context limit, hallucinated a non-JSON
        # response, or our prompt template drifted. Previously this silently degraded
        # to YELLOW; now escalates to RED so operator's daily 9 AM check surfaces it.
        # The raw-response prefix gives operator something to investigate.
        logger.warning("Alignment JSON parse failed for %s: %s | raw_prefix: %s",
                       ticker, str(e)[:200], text[:200] if text else "<empty>")
        return {
            "ticker": ticker,
            "severity": "RED",
            "alignment": "NEUTRAL",
            "rationale": "alignment_parse_failure (LLM response not JSON; manual review)",
            "news_summary": news[:200],
        }
    except Exception as e:
        # Non-parse failures (API error, network, etc.) — keep YELLOW posture; this
        # is the existing fallback semantic for transient infrastructure issues.
        logger.warning(f"Alignment scoring failed for {ticker}: {e}")
        return {
            "ticker": ticker,
            "severity": "YELLOW",
            "alignment": "NEUTRAL",
            "rationale": f"Scoring failed: {e}",
            "news_summary": news[:200],
        }


LEARNING_LOG_LOCK = LEARNING_LOG_PATH + ".lock"


def _write_alignment_results(checks: list[dict], total_positions: int) -> None:
    """Append alignment check results to learning_log.json."""
    red_count = sum(1 for c in checks if c.get("severity") == "RED")

    entry = {
        "timestamp": datetime.now(_NY).isoformat(),
        "checks": checks,
        "red_flags": red_count,
        "yellow_flags": sum(1 for c in checks if c.get("severity") == "YELLOW"),
        "green_flags": sum(1 for c in checks if c.get("severity") == "GREEN"),
        "total_positions": total_positions,
    }

    try:
        # Exclusive lock file for atomic read-modify-write (prevents TOCTOU)
        with open(LEARNING_LOG_LOCK, "a") as lockf:
            fcntl.flock(lockf, fcntl.LOCK_EX)

            data = {}
            if os.path.isfile(LEARNING_LOG_PATH):
                with open(LEARNING_LOG_PATH, "r") as f:
                    data = json.load(f)

            if "alignment_checks" not in data:
                data["alignment_checks"] = []
            data["alignment_checks"].append(entry)

            # Cap to last 90 entries (~3 months of daily checks)
            if len(data["alignment_checks"]) > 90:
                data["alignment_checks"] = data["alignment_checks"][-90:]

            # Atomic write
            tmp = LEARNING_LOG_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp, LEARNING_LOG_PATH)

        logger.info(f"Alignment check written: {len(checks)} positions, {red_count} RED flags")

    except Exception as e:
        logger.error(f"Failed to write alignment results: {e}", exc_info=True)


async def _send_red_flag_alerts(context, checks: list[dict]) -> None:
    """Send Telegram alerts for RED-flagged positions."""
    red_flags = [c for c in checks if c.get("severity") == "RED"]
    if not red_flags:
        return

    try:
        from agent.integrations.scheduled_tasks import _get_chat_ids, _safe_send
        chat_ids = _get_chat_ids()
        if not chat_ids:
            return

        lines = [f"Pre-Market Alignment: {len(red_flags)} RED FLAG(s):\n"]
        for flag in red_flags:
            lines.append(
                f"  {flag['ticker']}: {flag.get('alignment', '?')} — "
                f"{flag.get('rationale', 'no rationale')}"
            )

        message = "\n".join(lines)
        if len(message) > 4000:
            message = message[:3997] + "..."

        for chat_id in chat_ids:
            await _safe_send(context.bot, chat_id, text=message)

    except Exception as e:
        logger.warning(f"Failed to send RED flag alerts: {e}")


async def run_alignment_check(context) -> None:
    """Main entry point: run pre-market alignment check for all open positions.

    Called by JobQueue at 9:00 AM ET on weekdays.
    """
    now = datetime.now(ZoneInfo("America/New_York"))
    if now.weekday() >= 5:  # Skip weekends
        return
    logger.info("Starting pre-market alignment check...")

    try:
        positions = _get_open_positions()
        if not positions:
            logger.info("No open positions — skipping alignment check")
            return

        checks = []
        for pos in positions:
            ticker = pos.get("symbol", "")
            if not ticker:
                continue

            # Fetch news then score alignment
            news = await _fetch_news_for_ticker(ticker)
            result = await _score_alignment(pos, news)
            checks.append(result)

            # Small delay between positions to avoid rate limits
            await asyncio.sleep(1.0)

        if checks:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _write_alignment_results, checks, len(positions))
            await _send_red_flag_alerts(context, checks)

        red_count = sum(1 for c in checks if c.get("severity") == "RED")
        logger.info(
            f"Alignment check complete: {len(checks)} positions checked, "
            f"{red_count} RED, "
            f"{sum(1 for c in checks if c.get('severity') == 'YELLOW')} YELLOW, "
            f"{sum(1 for c in checks if c.get('severity') == 'GREEN')} GREEN"
        )

    except Exception as e:
        logger.error(f"Alignment check failed: {e}", exc_info=True)
