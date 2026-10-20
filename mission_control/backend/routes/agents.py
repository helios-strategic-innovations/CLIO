"""Agent routes - consensus viewer, performance, roles."""

import logging
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query
from typing import Optional

from mission_control.backend.auth import require_auth
from mission_control.backend.deps import read_json
from mission_control.backend.config import mc_config

router = APIRouter(prefix="/api/hydrinia", tags=["agents"])
logger = logging.getLogger("mission_control.agents")

_STATE_DIR = os.path.join(mc_config.tradingbot_root, "state")


def _hydrinia_bridge_enabled() -> bool:
    try:
        from agent.hydrinia_bridge import bridge_enabled

        return bridge_enabled()
    except Exception as e:
        logger.debug(f"agents: bridge_enabled() unavailable ({type(e).__name__}: {e})")
        return False
_BUY_DIRECTIONS = {"BUY", "BULLISH", "LONG", "UP"}
_SELL_DIRECTIONS = {"SELL", "BEARISH", "SHORT", "DOWN"}
_MODEL_DEFAULTS = {
    "deepseek": "deepseek-v4-pro",
    "claude": "claude-sonnet-5-5",
    "xai": "grok-4.5",
}
_MODEL_ENV = {
    "deepseek": "DEEPSEEK_MODEL",
    "claude": "CLAUDE_AGENT_MODEL",
    "xai": "XAI_MODEL",
}


def _empty_vote_counts() -> dict:
    return {"total_votes": 0, "buy_votes": 0, "sell_votes": 0, "hold_votes": 0}


def _file_mtime_iso(path: str) -> Optional[str]:
    try:
        return datetime.fromtimestamp(os.path.getmtime(path), timezone.utc).isoformat()
    except OSError:
        return None


def _configured_model(platform: Optional[str]) -> Optional[str]:
    platform_key = str(platform or "").strip().lower()
    if not platform_key:
        return None
    env_name = _MODEL_ENV.get(platform_key)
    return os.getenv(env_name, _MODEL_DEFAULTS.get(platform_key, "")) or None


def _freshness_ttl_seconds() -> int:
    try:
        return max(60, int(os.getenv("AGENT_PERFORMANCE_STALE_SECONDS", "7200")))
    except (TypeError, ValueError):
        return 7200


def _source_age_seconds(value: Optional[str]) -> Optional[int]:
    epoch = _coerce_epoch(value)
    if epoch is None:
        return None
    return max(0, int(datetime.now(timezone.utc).timestamp() - epoch))


def _as_int(value) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_epoch(value) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            pass
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except (TypeError, ValueError, OSError):
        return None


def _iso_from_epoch(epoch: Optional[float]) -> Optional[str]:
    if epoch is None:
        return None
    try:
        return datetime.fromtimestamp(float(epoch), timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return None


def _direction_from_vote(vote) -> str:
    if isinstance(vote, dict):
        vote = vote.get("direction", vote.get("vote", ""))
    direction = str(vote or "").upper()
    if direction in _BUY_DIRECTIONS:
        return "BUY"
    if direction in _SELL_DIRECTIONS:
        return "SELL"
    if direction:
        return "HOLD"
    return ""


def _add_direction(counts: dict, direction: str) -> None:
    if not direction:
        return
    counts["total_votes"] += 1
    if direction == "BUY":
        counts["buy_votes"] += 1
    elif direction == "SELL":
        counts["sell_votes"] += 1
    else:
        counts["hold_votes"] += 1


def _counts_from_trade_list(trades) -> dict:
    counts = _empty_vote_counts()
    if not isinstance(trades, list):
        return counts
    for trade in trades:
        if not isinstance(trade, dict):
            continue
        _add_direction(counts, _direction_from_vote(trade.get("vote")))
    return counts


def _counts_from_vote_history(vote_history) -> dict:
    agents = defaultdict(_empty_vote_counts)
    platform_counts = defaultdict(Counter)
    specialty_counts = defaultdict(Counter)
    requested_model_counts = defaultdict(Counter)
    served_model_counts = defaultdict(Counter)
    served_model_verified = defaultdict(int)
    last_vote_epoch = {}

    def add_meta(agent_name: str, row: dict) -> None:
        platform = row.get("platform") or row.get("model_provider") or row.get("provider")
        if platform:
            platform_counts[agent_name][str(platform)] += 1
        requested_model = row.get("requested_model") or row.get("model")
        if requested_model:
            requested_model_counts[agent_name][str(requested_model)] += 1
        served_model = row.get("served_model") or row.get("response_model") or row.get("actual_model")
        if served_model:
            served_model_counts[agent_name][str(served_model)] += 1
            if row.get("served_model_verified") is True:
                served_model_verified[agent_name] += 1
        specialty = row.get("specialty")
        if specialty:
            specialty_counts[agent_name][str(specialty)] += 1
        epoch = _coerce_epoch(row.get("timestamp") or row.get("ts") or row.get("created_at") or row.get("time"))
        if epoch is not None and epoch > last_vote_epoch.get(agent_name, 0):
            last_vote_epoch[agent_name] = epoch

    if isinstance(vote_history, list):
        for row in vote_history:
            if not isinstance(row, dict):
                continue
            agent_name = row.get("agent_name") or row.get("agent") or row.get("source")
            if not agent_name:
                continue
            _add_direction(agents[agent_name], _direction_from_vote(row.get("vote", row.get("direction"))))
            add_meta(agent_name, row)
    elif isinstance(vote_history, dict):
        for _trade_id, trade_data in vote_history.items():
            if not isinstance(trade_data, dict):
                continue
            votes = trade_data.get("votes", trade_data)
            if not isinstance(votes, dict):
                continue
            for agent_name, vote in votes.items():
                _add_direction(agents[agent_name], _direction_from_vote(vote))
                metadata = dict(trade_data)
                if isinstance(vote, dict):
                    metadata.update(vote)
                add_meta(agent_name, metadata)

    result = {}
    for agent_name, counts in agents.items():
        row = dict(counts)
        platforms = dict(platform_counts.get(agent_name, {}))
        specialties = dict(specialty_counts.get(agent_name, {}))
        requested_models = dict(requested_model_counts.get(agent_name, {}))
        served_models = dict(served_model_counts.get(agent_name, {}))
        row["platform_counts"] = platforms
        row["platform"] = max(platforms, key=platforms.get) if platforms else None
        row["specialty_counts"] = specialties
        row["specialty"] = max(specialties, key=specialties.get) if specialties else None
        row["requested_model_counts"] = requested_models
        row["requested_model"] = (
            max(requested_models, key=requested_models.get)
            if requested_models
            else _configured_model(row["platform"])
        )
        row["served_model_counts"] = served_models
        row["served_model"] = max(served_models, key=served_models.get) if served_models else None
        row["served_model_verified"] = bool(served_model_verified.get(agent_name, 0))
        row["last_vote_time"] = _iso_from_epoch(last_vote_epoch.get(agent_name))
        result[agent_name] = row
    return result


def _prediction_accuracy(pred_log) -> dict:
    predictions = pred_log.get("predictions", []) if isinstance(pred_log, dict) else []
    agent_correct = defaultdict(lambda: {"correct": 0, "total": 0})
    if not isinstance(predictions, list):
        return {}

    for pred in predictions:
        if not isinstance(pred, dict):
            continue
        agent = pred.get("agent") or pred.get("agent_name") or pred.get("source")
        outcome = pred.get("outcome")
        resolved = pred.get("resolved")
        status = str(pred.get("status") or "").lower()
        resolved_at = pred.get("resolved_at")
        brier_score = pred.get("brier_score")
        if not agent:
            continue
        is_resolved = (
            resolved is not None
            or resolved_at is not None
            or brier_score is not None
            or status in {"resolved", "closed", "settled"}
        )
        if not is_resolved:
            continue
        agent_correct[agent]["total"] += 1
        if resolved is True or outcome is True:
            agent_correct[agent]["correct"] += 1

    return dict(agent_correct)


def _normalise_agent_metrics(metrics: dict, recent_counts: dict, pred_accuracy: dict) -> dict:
    result = {}
    agent_names = set(recent_counts.keys()) | set(pred_accuracy.keys())
    if isinstance(metrics, dict):
        agent_names |= {name for name, data in metrics.items() if isinstance(data, dict)}

    for agent_name in sorted(agent_names):
        raw = metrics.get(agent_name, {}) if isinstance(metrics, dict) else {}
        raw = raw if isinstance(raw, dict) else {}
        stats = dict(raw)
        scored_counts = _counts_from_trade_list(raw.get("trades", []))
        recent = recent_counts.get(agent_name, _empty_vote_counts())

        metric_total = _as_int(raw.get("total_votes"))
        scored_total = metric_total if metric_total is not None else scored_counts["total_votes"]
        recent_total = recent.get("total_votes", 0)
        total_votes = recent_total or scored_total
        stats["total_votes"] = total_votes
        stats["scored_trade_total"] = scored_total

        primary_counts = recent if recent_total else scored_counts
        for field in ("buy_votes", "sell_votes", "hold_votes"):
            stats[field] = primary_counts.get(field, 0)

        stats["platform"] = recent.get("platform") or raw.get("platform")
        stats["platform_counts"] = recent.get("platform_counts", {})
        stats["specialty"] = recent.get("specialty") or raw.get("specialty")
        stats["last_vote_time"] = recent.get("last_vote_time")
        requested_model = (
            recent.get("requested_model")
            or raw.get("requested_model")
            or raw.get("model")
            or _configured_model(stats["platform"])
        )
        stats["model"] = requested_model
        stats["requested_model"] = requested_model
        stats["requested_model_counts"] = recent.get("requested_model_counts", {})
        stats["served_model"] = recent.get("served_model") or raw.get("served_model")
        stats["served_model_counts"] = recent.get("served_model_counts", {})
        stats["served_model_verified"] = bool(
            recent.get("served_model_verified") or raw.get("served_model_verified")
        )
        stats["model_source"] = (
            "vote_history"
            if recent.get("requested_model_counts")
            else "runtime_configuration"
        )
        stats["source"] = "state/vote_history.json"
        stats["source_age"] = stats["last_vote_time"]
        stats["source_age_seconds"] = _source_age_seconds(stats["last_vote_time"])

        if agent_name in pred_accuracy and pred_accuracy[agent_name]["total"]:
            stats["resolved_predictions"] = pred_accuracy[agent_name]["total"]
        # F5 gate: agent scoreability is still pending; do not present raw
        # correct_votes, Brier rows, or direction/probability outcomes as validated accuracy.
        stats["accuracy"] = None
        stats["hit_rate"] = None
        stats["accuracy_source"] = "scoreability_pending"
        stats["accuracy_status"] = "pending"
        stats["recent_vote_total"] = recent_total
        stats["recent_buy_votes"] = recent.get("buy_votes", 0)
        stats["recent_sell_votes"] = recent.get("sell_votes", 0)
        stats["recent_hold_votes"] = recent.get("hold_votes", 0)
        stats["buy_rate"] = (stats["buy_votes"] / total_votes) if total_votes else None
        stats["sell_rate"] = (stats["sell_votes"] / total_votes) if total_votes else None
        stats["hold_rate"] = (stats["hold_votes"] / total_votes) if total_votes else None
        stats["vote_skew"] = (
            (stats["buy_votes"] - stats["sell_votes"]) / total_votes
            if total_votes
            else None
        )
        stats["low_buy_outlier"] = bool(total_votes >= 20 and stats["buy_votes"] <= 1)
        result[agent_name] = stats

    return result


@router.get("/agents")
async def get_agents(name: Optional[str] = None, _=Depends(require_auth)):
    """Agent performance with scored metrics plus normalized vote counts.

    The 12-agent deliberation loop this reads (agent_metrics/vote_history/
    prediction_log) belonged to Hydrinia, which is permanently retired
    (2026-09-24) — CLIO's DPOF engine has no per-trade agent vote. Returns an
    empty, correctly-shaped payload unless the bridge is explicitly re-enabled.
    """
    if not _hydrinia_bridge_enabled():
        empty_provenance = {
            "source": "Hydrinia state/agent_metrics.json + vote_history.json + prediction_log.json",
            "as_of": None,
            "stale": True,
            "reason_code": "HYDRINIA_RETIRED",
            "source_age_seconds": None,
            "freshness_ttl_seconds": _freshness_ttl_seconds(),
            "updated": None,
            "agent_metrics_updated": None,
            "prediction_log_updated": None,
            "vote_window_size": None,
        }
        if name:
            return {"agent": name, "metrics": {}, **empty_provenance}
        return {"agents": {}, "total_agents": 0, **empty_provenance}

    metrics = read_json(os.path.join(_STATE_DIR, "agent_metrics.json")) or {}
    vote_history = read_json(os.path.join(_STATE_DIR, "vote_history.json")) or {}
    pred_log = read_json(os.path.join(_STATE_DIR, "prediction_log.json")) or {}

    recent_counts = _counts_from_vote_history(vote_history)
    pred_accuracy = _prediction_accuracy(pred_log)
    result = _normalise_agent_metrics(metrics, recent_counts, pred_accuracy)

    updated = _file_mtime_iso(os.path.join(_STATE_DIR, "vote_history.json"))
    agent_metrics_updated = _file_mtime_iso(os.path.join(_STATE_DIR, "agent_metrics.json"))
    prediction_log_updated = _file_mtime_iso(os.path.join(_STATE_DIR, "prediction_log.json"))
    vote_window_size = len(vote_history) if isinstance(vote_history, list) else None
    source_epochs = [
        epoch
        for epoch in (
            _coerce_epoch(updated),
            _coerce_epoch(agent_metrics_updated),
            _coerce_epoch(prediction_log_updated),
        )
        if epoch is not None
    ]
    as_of = _iso_from_epoch(max(source_epochs)) if source_epochs else None
    source_age_seconds = _source_age_seconds(as_of)
    stale = source_age_seconds is None or source_age_seconds > _freshness_ttl_seconds()
    reason_code = "SOURCE_MISSING" if source_age_seconds is None else ("SOURCE_STALE" if stale else "OK")
    provenance = {
        "source": "Hydrinia state/agent_metrics.json + vote_history.json + prediction_log.json",
        "as_of": as_of,
        "stale": stale,
        "reason_code": reason_code,
        "source_age_seconds": source_age_seconds,
        "freshness_ttl_seconds": _freshness_ttl_seconds(),
        "updated": updated,
        "agent_metrics_updated": agent_metrics_updated,
        "prediction_log_updated": prediction_log_updated,
        "vote_window_size": vote_window_size,
    }

    if name:
        return {
            "agent": name,
            "metrics": result.get(name, {}),
            **provenance,
        }
    return {
        "agents": result,
        "total_agents": len(result),
        **provenance,
    }


@router.get("/agents/consensus")
async def get_consensus(limit: int = Query(20, ge=1, le=100), _=Depends(require_auth)):
    """Recent voting records - agent consensus on trades (Hydrinia-era; retired)."""
    if not _hydrinia_bridge_enabled():
        return {"consensus": [], "count": 0, "source": "hydrinia_retired"}

    trade_outcomes = read_json(os.path.join(_STATE_DIR, "trade_outcomes.json")) or {}

    if isinstance(trade_outcomes, dict):
        trades = list(trade_outcomes.values())
        trades.sort(key=lambda t: t.get("timestamp", 0), reverse=True)
        return {"consensus": trades[:limit], "count": len(trades[:limit]), "source": "hydrinia_legacy_bridge"}

    return {"consensus": [], "count": 0, "source": "hydrinia_legacy_bridge"}
