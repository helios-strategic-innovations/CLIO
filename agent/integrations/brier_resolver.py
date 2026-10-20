"""
Moltbot — Daily Brier Score Prediction Resolution Check
Runs at 4:30 PM ET weekdays. Resolves matured predictions using market data.
Reads/updates prediction_log.json in Hydrinia's state directory.
Uses position_log.json as primary resolution source (closed positions with P&L).
"""

import asyncio
import fcntl
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from agent.config import config

logger = logging.getLogger("openclaw.brier")
_NY = ZoneInfo("America/New_York")

PREDICTION_LOG_PATH = os.path.join(config.tradingbot_dir, "state", "prediction_log.json")
PREDICTION_LOCK_PATH = os.path.join(config.tradingbot_dir, "state", "prediction_log.lock")
POSITION_LOG_PATH = os.path.join(config.tradingbot_dir, "state", "position_log.json")


def _run_brier_resolution_sync() -> None:
    """Synchronous wrapper that acquires lock and runs resolution."""
    if not os.path.isfile(PREDICTION_LOG_PATH):
        return
    lock_fd = open(PREDICTION_LOCK_PATH, "a")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        _run_brier_resolution_locked()
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


async def run_brier_resolution(context) -> None:
    """Daily resolution check for matured predictions. Called at 4:30 PM ET."""
    # Hydrinia is permanently off (2026-09-24) — this module reads/writes
    # Hydrinia's state/prediction_log.json, which CLIO never writes to.
    # Defense in depth beyond the job-registration gate.
    try:
        from agent.hydrinia_bridge import bridge_enabled
        if not bridge_enabled():
            return
    except Exception as e:
        logger.debug(f"Bridge check failed (treating as disabled): {e}")
        return

    now = datetime.now(ZoneInfo("America/New_York"))
    if now.weekday() >= 5:  # Skip weekends
        return
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _run_brier_resolution_sync)
    except Exception as e:
        logger.error(f"Brier resolution check failed: {e}", exc_info=True)


def _load_closed_positions() -> dict[str, list[dict]]:
    """Load closed/stopped positions from position_log.json, grouped by symbol.

    Returns dict mapping symbol -> list of closed position dicts, sorted by close_time.
    Uses shared lock (LOCK_SH) since this is read-only from Moltbot's perspective.
    """
    result: dict[str, list[dict]] = {}
    if not os.path.isfile(POSITION_LOG_PATH):
        return result
    try:
        _pos_lock_path = POSITION_LOG_PATH + ".lock"
        _pos_lock_fd = open(_pos_lock_path, "a")
        try:
            fcntl.flock(_pos_lock_fd, fcntl.LOCK_SH)
            with open(POSITION_LOG_PATH, "r") as f:
                data = json.load(f)
        finally:
            fcntl.flock(_pos_lock_fd, fcntl.LOCK_UN)
            _pos_lock_fd.close()

        positions = data.get("positions", {})
        for pos in positions.values():
            if not isinstance(pos, dict):
                continue
            if pos.get("status") not in ("closed", "stopped"):
                continue
            symbol = pos.get("symbol", "")
            if not symbol:
                continue
            result.setdefault(symbol, []).append(pos)

        # Sort each symbol's positions by close_time ascending
        # close_time can be epoch float or ISO string — normalize for safe comparison
        def _normalize_close_time(p):
            ct = p.get("close_time", 0)
            if isinstance(ct, str):
                try:
                    dt = datetime.fromisoformat(ct)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=_NY)
                    return dt.timestamp()
                except (ValueError, TypeError):
                    return 0
            return ct if ct is not None else 0  # S233 ME-5: Explicit None check (0 is valid)

        for sym in result:
            result[sym].sort(key=_normalize_close_time)

    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load position_log.json: {e}")
    return result


def _resolve_via_position_log(pred: dict, closed_by_symbol: dict[str, list[dict]]) -> bool:
    """Try to resolve a prediction using closed positions from position_log.

    Matches by symbol. Finds the earliest closed position whose close_time
    is after the prediction's created date.
    Returns True if resolved, False otherwise.
    """
    symbol = pred.get("asset", "")
    if not symbol or symbol not in closed_by_symbol:
        return False

    # Parse prediction created time
    try:
        created_str = pred.get("created", "")
        created = datetime.fromisoformat(created_str)
        if created.tzinfo is None:
            created = created.replace(tzinfo=_NY)
        created_epoch = created.timestamp()
    except (ValueError, TypeError):
        return False

    # Find matching closed position (close_time after prediction created)
    for pos in closed_by_symbol[symbol]:
        close_time = pos.get("close_time")
        if close_time is None:
            continue
        # position_log timestamps are epoch floats; close_time may be str or non-numeric — TypeError/ValueError caught below
        try:
            close_epoch = float(close_time)
        except (ValueError, TypeError):
            continue

        if close_epoch <= created_epoch:
            continue

        # Match found — resolve using position P&L
        pnl = pos.get("pnl")
        if pnl is None:
            continue

        try:
            pnl_val = float(pnl)
        except (ValueError, TypeError):
            continue

        outcome = 1 if pnl_val > 0 else 0
        probability = pred.get("probability", 0.5)
        try:
            probability = float(probability)
        except (ValueError, TypeError):
            probability = 0.5
        # S190 M4: clamp probability to [0, 1] before squaring. Out-of-range input
        # (e.g., 1.5 from a malformed upstream prediction) would produce Brier scores
        # >1.0, which break per-source aggregation downstream (line 261).
        probability = max(0.0, min(1.0, probability))

        pred["outcome"] = outcome
        pred["brier_score"] = round((probability - outcome) ** 2, 4)
        pred["status"] = "resolved"
        pred["resolved_at"] = datetime.now(_NY).isoformat()
        pred["resolved_via"] = "position_log"
        pred["resolution_position_id"] = pos.get("position_id", "")
        return True

    return False


def _run_brier_resolution_locked() -> None:
    """Inner resolution logic — called while holding the prediction_log lock."""
    try:
        with open(PREDICTION_LOG_PATH, "r") as f:
            data = json.load(f)

        today = datetime.now(_NY).strftime("%Y-%m-%d")
        predictions = data.get("predictions", [])

        # Find open predictions with resolution_date <= today
        to_resolve = [p for p in predictions if
                      p.get("status") == "open" and
                      p.get("resolution_date", "9999") <= today]

        if not to_resolve:
            return

        resolved_count = 0
        extended_count = 0
        stale_count = 0

        # Load debriefs ONCE (not per-prediction)
        debriefs_path = os.path.join(config.tradingbot_dir, "state", "trade_debriefs.json")
        debriefs_data = {}
        if os.path.isfile(debriefs_path):
            try:
                with open(debriefs_path) as df:
                    debriefs_data = json.load(df)
            except (json.JSONDecodeError, OSError):
                pass

        # Load closed positions ONCE for fallback resolution
        closed_by_symbol = _load_closed_positions()

        for pred in to_resolve:
            trade_id = pred.get("trade_id", "")

            # 1. Try debrief-based resolution (primary — for when trade_debriefs.json exists)
            if trade_id and debriefs_data:
                try:
                    for d in debriefs_data.get("debriefs", []):
                        if d.get("trade_id") == trade_id:
                            pnl = d.get("pnl", 0)
                            outcome = 1 if float(pnl or 0) > 0 else 0
                            probability = pred.get("probability", 0.5)
                            try:
                                probability = float(probability)
                            except (ValueError, TypeError):
                                probability = 0.5
                            # S190 M4: clamp probability to [0, 1] before squaring (see line ~155 note).
                            probability = max(0.0, min(1.0, probability))
                            pred["outcome"] = outcome
                            pred["brier_score"] = round((probability - outcome) ** 2, 4)
                            pred["status"] = "resolved"
                            pred["resolved_at"] = datetime.now(_NY).isoformat()
                            pred["resolved_via"] = "trade_debrief"
                            resolved_count += 1
                            break
                except Exception as e:
                    logger.debug(f"Debrief match failed for {pred.get('asset', '?')}: {e}")

            # 2. Fallback: resolve via position_log (closed positions by symbol)
            if pred.get("status") == "open":
                if _resolve_via_position_log(pred, closed_by_symbol):
                    resolved_count += 1
                    continue

            # 3. If still open, extend or expire
            if pred.get("status") == "open":
                try:
                    created = datetime.fromisoformat(pred.get("created", ""))
                    if created.tzinfo is None:
                        created = created.replace(tzinfo=_NY)
                    age_days = (datetime.now(_NY) - created).days
                    if age_days > 30:
                        pred["status"] = "expired"
                        pred["resolved_at"] = datetime.now(_NY).isoformat()
                        stale_count += 1
                    else:
                        new_date = datetime.strptime(pred["resolution_date"], "%Y-%m-%d") + timedelta(days=7)
                        pred["resolution_date"] = new_date.strftime("%Y-%m-%d")
                        extended_count += 1
                except (ValueError, TypeError):
                    pred["status"] = "expired"
                    pred["resolved_at"] = datetime.now(_NY).isoformat()
                    stale_count += 1

        # Recompute rolling Brier scores if any resolved
        if resolved_count > 0:
            resolved_preds = [p for p in predictions if
                              p.get("status") == "resolved" and p.get("brier_score") is not None]
            by_source = {}
            for p in resolved_preds:
                src = p.get("source", "unknown")
                if src not in by_source:
                    by_source[src] = []
                by_source[src].append(p["brier_score"])

            rolling = {}
            for src, scores in by_source.items():
                recent = scores[-20:]
                rolling[src] = round(sum(recent) / len(recent), 4) if recent else None

            data["rolling_brier_per_agent"] = rolling
            data["total_resolved"] = sum(1 for p in predictions if p.get("status") == "resolved")

        # Save with atomic write
        data["last_updated"] = datetime.now(_NY).isoformat()
        tmp = PREDICTION_LOG_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, PREDICTION_LOG_PATH)

        if resolved_count > 0 or stale_count > 0:
            logger.info(f"Brier resolution: {resolved_count} resolved, {extended_count} extended, {stale_count} expired")

            # Flag agents with Brier > 0.25 (worse than coin-flip baseline)
            bad_agents = [
                f"{src}: {score:.3f}"
                for src, score in data.get("rolling_brier_per_agent", {}).items()
                if score is not None and score > 0.25
            ]
            if bad_agents:
                logger.warning(f"Brier warning — worse than baseline: {', '.join(bad_agents)}")

    except Exception as e:
        logger.error(f"Brier resolution (locked) failed: {e}", exc_info=True)
