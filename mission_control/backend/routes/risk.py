"""Risk routes — regime, exposure, risk rules, theses, alerts."""

import glob
import json
import logging
import os
import re
import time
from collections import deque
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, Query

from agent.integrations.alpaca_values import AlpacaValueError, normalize_position_side
from mission_control.backend.auth import require_auth
from mission_control.backend.deps import read_json, read_text, get_trading_client, run_in_executor
from mission_control.backend.config import mc_config

router = APIRouter(prefix="/api/hydrinia", tags=["risk"])
logger = logging.getLogger("mission_control.risk")

_STATE_DIR = os.path.join(mc_config.tradingbot_root, "state")
_UW_SNAPSHOT_PATH = os.path.join(
    mc_config.tradingbot_root, "moltbot_knowledge", "uw_signals", "snapshot.json"
)
_VOL_SURFACE_PATH = os.path.join(
    mc_config.tradingbot_root, "moltbot_knowledge", "gap_signals", "vol_surface.json"
)
_VOTE_VECTORS_PATH = os.path.join(_STATE_DIR, "logs", "vote_vectors.jsonl")
_DAILY_METRICS_PATH = os.path.join(_STATE_DIR, "daily_metrics.json")
_REGIME_JOURNAL_PATH = os.path.join(_STATE_DIR, "regime_journal.json")
_CHURN_ALERTS_PATH = os.getenv(
    "HYDRINIA_CHURN_ALERTS_PATH",
    "/home/agent/hydrinia_ops/emergency_deploy_churn/alerts.jsonl",
)
_NY = ZoneInfo("America/New_York")
_OFFHOURS_TTL_MULTIPLIER = 6
_STRUCTURED_LOG_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\s*\|")
_BROKER_STATE_TTL_SECONDS = 60
_BROKER_STATE_CACHE: dict[str, object] = {
    "stored_monotonic": -1e9,
    "stored_at": None,
    "data": None,
}
_ALERT_TTL_BY_SEVERITY = {
    "critical": 3600,
    "warning": 1800,
    "info": 900,
}
_ALERT_RESOLUTION_TERMS = (
    "RESOLVED",
    "RECOVERED",
    "CLEARED",
    "RESTORED",
)


def _as_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


_TAIL_VOL_EXPOSURE_SYMBOLS = {"UVIX", "UVXY", "VIXY", "VXX", "VIX"}
_KNOWN_BEARISH_ETFS = {
    "SQQQ", "SPXU", "SH", "PSQ", "QID", "TZA", "TWM", "SRTY", "SDOW", "DOG", "DXD",
    "SOXS", "LABD", "FAZ", "ERY", "TECS", "GLL", "ZSL", "DUST", "GDXD", "NVDD",
    "NVDS", "TSLS", "AAPD", "MSFD", "METD", "GGLS", "AMZD", "AMDS",
}


def _is_bearish_expression_instrument(symbol: str, name: str = "") -> bool:
    sym = (symbol or "").upper().strip()
    if not sym:
        return False
    if sym in _TAIL_VOL_EXPOSURE_SYMBOLS or "VIX" in sym:
        return True
    if sym in _KNOWN_BEARISH_ETFS:
        return True

    text = f"{sym} {name or ''}".upper()
    text = re.sub(r"\bSHORT[-\s]?TERM\b", "", text)
    return any(term in text for term in ("INVERSE", "BEAR", "BEARISH", "SHORT"))


def _option_right(contract: str) -> str | None:
    match = re.match(r"^[A-Z]{1,6}\d{6}([CP])\d{8}$", (contract or "").upper().strip())
    return match.group(1) if match else None


def _option_underlying(contract: str) -> str | None:
    match = re.match(r"^([A-Z]{1,6})\d{6}[CP]\d{8}$", (contract or "").upper().strip())
    return match.group(1) if match else None


def _infer_market_exposure_direction(pos: dict, fallback_direction: str = "") -> str:
    raw_direction = str(
        pos.get("market_exposure_direction")
        or pos.get("market_direction")
        or fallback_direction
        or pos.get("direction")
        or ""
    ).upper()
    if raw_direction in {"BULLISH", "BEARISH"} and (pos.get("market_exposure_direction") or pos.get("market_direction")):
        return raw_direction

    direction = str(fallback_direction or pos.get("direction") or "").lower()
    if direction in {"bearish", "short", "sell", "down"}:
        sign = -1
    elif direction in {"bullish", "long", "buy", "up"}:
        sign = 1
    else:
        sign = -1 if bool(pos.get("is_short")) else 1

    raw_symbol = str(pos.get("symbol") or "")
    contract = str(pos.get("contract") or (raw_symbol if _option_right(raw_symbol) else ""))
    if pos.get("is_options") or _option_right(contract):
        right = _option_right(contract)
        if right:
            sign = 1 if right == "C" else -1

    if bool(pos.get("is_short")):
        sign *= -1

    symbol = _option_underlying(contract) or raw_symbol
    category = str(pos.get("category") or "")
    if _is_bearish_expression_instrument(symbol, category):
        sign *= -1

    return "BEARISH" if sign < 0 else "BULLISH"


def _parse_iso_epoch(value) -> float | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=_NY)
        return parsed.timestamp()
    except (TypeError, ValueError):
        return None


def _file_mtime_epoch(path: str) -> float | None:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def _market_hours_now() -> bool:
    now = datetime.now(_NY)
    if now.weekday() >= 5:
        return False
    open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_time <= now <= close_time


def _source_freshness(path: str, data: dict, default_ttl: int) -> dict:
    exists = os.path.isfile(path)
    timestamp = (
        data.get("timestamp")
        or data.get("last_updated")
        or data.get("updated_at")
        if isinstance(data, dict)
        else None
    )
    timestamp_epoch = _parse_iso_epoch(timestamp)
    mtime_epoch = _file_mtime_epoch(path)
    source_epoch = timestamp_epoch or mtime_epoch
    market_hours = _market_hours_now()
    raw_ttl = _as_float(data.get("freshness_ttl_seconds") if isinstance(data, dict) else None, default_ttl)
    ttl_seconds = int(raw_ttl or default_ttl)
    effective_ttl = ttl_seconds if market_hours else ttl_seconds * _OFFHOURS_TTL_MULTIPLIER
    age_seconds = int(max(0, time.time() - source_epoch)) if source_epoch else None

    if not exists:
        status = "missing"
    elif age_seconds is None:
        status = "unknown"
    elif age_seconds <= effective_ttl:
        status = "fresh"
    else:
        status = "stale" if market_hours else "stale_off_hours"

    return {
        "path": path,
        "exists": exists,
        "status": status,
        "timestamp": timestamp,
        "mtime": datetime.fromtimestamp(mtime_epoch, _NY).isoformat() if mtime_epoch else None,
        "age_seconds": age_seconds,
        "ttl_seconds": ttl_seconds,
        "effective_ttl_seconds": effective_ttl,
        "market_hours": market_hours,
    }


async def _cached_broker_state() -> tuple[dict | None, dict]:
    """Bound account/position reads across all risk-page consumers."""
    now = time.monotonic()
    cached = _BROKER_STATE_CACHE.get("data")
    age = max(0.0, now - float(_BROKER_STATE_CACHE["stored_monotonic"]))
    if isinstance(cached, dict) and age <= _BROKER_STATE_TTL_SECONDS:
        return cached, {
            "source": "alpaca_trading_cached",
            "as_of": _BROKER_STATE_CACHE.get("stored_at"),
            "stale": False,
            "reason_code": "OK",
            "cache_hit": True,
            "cache_age_seconds": round(age, 3),
            "ttl_seconds": _BROKER_STATE_TTL_SECONDS,
        }

    try:
        client = get_trading_client()
        account, positions = await run_in_executor(
            lambda: (client.get_account(), client.get_all_positions())
        )
        stored_at = datetime.now(timezone.utc).isoformat()
        data = {"account": account, "positions": list(positions or [])}
        _BROKER_STATE_CACHE.update({
            "stored_monotonic": now,
            "stored_at": stored_at,
            "data": data,
        })
        return data, {
            "source": "alpaca_trading_cached",
            "as_of": stored_at,
            "stale": False,
            "reason_code": "OK",
            "cache_hit": False,
            "cache_age_seconds": 0.0,
            "ttl_seconds": _BROKER_STATE_TTL_SECONDS,
        }
    except Exception:
        logger.debug("Risk broker cache refresh failed", exc_info=True)

    if isinstance(cached, dict):
        return cached, {
            "source": "alpaca_trading_cached",
            "as_of": _BROKER_STATE_CACHE.get("stored_at"),
            "stale": True,
            "reason_code": "STALE_ALPACA_CACHE",
            "cache_hit": True,
            "cache_age_seconds": round(age, 3),
            "ttl_seconds": _BROKER_STATE_TTL_SECONDS,
        }
    return None, {
        "source": "alpaca_trading_cached",
        "as_of": datetime.now(timezone.utc).isoformat(),
        "stale": True,
        "reason_code": "ALPACA_UNAVAILABLE",
        "cache_hit": False,
        "cache_age_seconds": None,
        "ttl_seconds": _BROKER_STATE_TTL_SECONDS,
    }


def _sentiment_scalar(value):
    if isinstance(value, dict):
        for key in ("sentiment", "direction", "market_tide_direction", "label", "regime"):
            if value.get(key) not in (None, ""):
                return str(value[key])
        return None
    if value in (None, ""):
        return None
    return str(value)


def _uw_sentiment_value(snapshot: dict):
    if not isinstance(snapshot, dict):
        return None
    market_context = snapshot.get("market_context")
    market_tide = snapshot.get("market_tide")
    flow_tide = snapshot.get("_flow_tide")
    candidates = [
        market_context.get("market_tide_direction") if isinstance(market_context, dict) else None,
        market_tide.get("sentiment") if isinstance(market_tide, dict) else None,
        flow_tide.get("sentiment") if isinstance(flow_tide, dict) else None,
        snapshot.get("sentiment"),
    ]
    return next((value for value in (_sentiment_scalar(item) for item in candidates) if value), None)


def _cached_indicator_detail(
    *,
    value,
    source: str,
    source_path: str,
    freshness: dict,
    fallback_reason_code: str | None = None,
) -> dict:
    status = str(freshness.get("status") or "unknown")
    stale = status != "fresh"
    if value is None:
        reason_code = "SOURCE_MISSING" if status == "missing" else "VALUE_UNAVAILABLE"
    elif stale:
        reason_code = "SOURCE_STALE"
    else:
        reason_code = fallback_reason_code or "OK"
    return {
        "value": value,
        "source": source,
        "source_path": source_path,
        "delivery": "cached_file",
        "cached": True,
        "vendor_call": False,
        "as_of": freshness.get("timestamp") or freshness.get("mtime"),
        "stale": stale or value is None,
        "age_seconds": freshness.get("age_seconds"),
        "ttl_seconds": freshness.get("effective_ttl_seconds"),
        "reason_code": reason_code,
    }


def _structured_indicator_detail(
    record: dict,
    *,
    fallback_source: str,
    source_path: str,
    file_freshness: dict,
) -> dict | None:
    """Normalize Hydrinia's cache-health envelope without trusting its scalar fallback."""
    if not isinstance(record, dict) or not record:
        return None

    raw_value = record.get("value")
    valid = bool(record.get("valid", False)) and raw_value not in (None, "")
    inner_stale = bool(record.get("stale", True)) or not bool(record.get("current", False))
    file_stale = str(file_freshness.get("status") or "unknown") != "fresh"
    stale = inner_stale or file_stale or not valid

    raw_reason = str(record.get("reason") or "").strip().upper()
    normalized_reason = re.sub(r"[^A-Z0-9]+", "_", raw_reason).strip("_")
    if not valid:
        reason_code = normalized_reason or "VALUE_UNAVAILABLE"
    elif stale:
        reason_code = "SOURCE_STALE"
    else:
        reason_code = "OK"

    return {
        # A stale or invalid sentinel is diagnostic context, not a live value.
        "value": raw_value if not stale else None,
        "last_value": raw_value,
        "source": str(record.get("source") or fallback_source),
        "source_path": source_path,
        "delivery": "cached_file",
        "cached": True,
        "vendor_call": False,
        "as_of": record.get("timestamp") or file_freshness.get("timestamp") or file_freshness.get("mtime"),
        "stale": stale,
        "age_seconds": record.get("age_seconds"),
        "ttl_seconds": record.get("freshness_ttl_seconds") or record.get("ttl_seconds"),
        "reason_code": reason_code,
        "provenance": record.get("provenance") if isinstance(record.get("provenance"), dict) else {},
    }


def _market_indicator_details(
    daily: dict,
    regime_journal: dict,
    uw_snapshot: dict,
) -> tuple[dict, dict]:
    daily_status = _source_freshness(_DAILY_METRICS_PATH, daily, 300)
    journal_status = _source_freshness(_REGIME_JOURNAL_PATH, regime_journal, 1800)
    uw_status = _source_freshness(_UW_SNAPSHOT_PATH, uw_snapshot, 300)
    transitions = regime_journal.get("transitions", []) if isinstance(regime_journal, dict) else []
    latest = transitions[-1] if isinstance(transitions, list) and transitions else {}
    signals = latest.get("signals", {}) if isinstance(latest, dict) else {}
    if not isinstance(signals, dict):
        signals = {}

    vix_detail = _structured_indicator_detail(
        daily.get("vix_health", {}) if isinstance(daily, dict) else {},
        fallback_source="daily_metrics.vix_health",
        source_path=_DAILY_METRICS_PATH,
        file_freshness=daily_status,
    )
    if vix_detail is not None:
        vix_value = vix_detail["value"]
    else:
        vix_value = _as_float(daily.get("vix")) if isinstance(daily, dict) else None
    vix_source = "daily_metrics"
    vix_path = _DAILY_METRICS_PATH
    vix_status = daily_status
    if vix_detail is None and vix_value is None:
        vix_value = _as_float(signals.get("vix_value", signals.get("vix")))
        vix_source = "regime_journal"
        vix_path = _REGIME_JOURNAL_PATH
        vix_status = journal_status
    if vix_detail is None and vix_value is None:
        vol_regime = uw_snapshot.get("vol_regime", {}) if isinstance(uw_snapshot, dict) else {}
        if not isinstance(vol_regime, dict):
            vol_regime = {}
        vix_value = _as_float(
            vol_regime.get("vix")
            or vol_regime.get("vix_level")
            or vol_regime.get("vix_spot")
        )
        vix_source = "hydrinia_uw_snapshot"
        vix_path = _UW_SNAPSHOT_PATH
        vix_status = uw_status
    if vix_detail is None:
        vix_detail = _cached_indicator_detail(
            value=vix_value,
            source=vix_source,
            source_path=vix_path,
            freshness=vix_status,
        )

    sentiment_detail = _structured_indicator_detail(
        daily.get("sentiment_health", {}) if isinstance(daily, dict) else {},
        fallback_source="daily_metrics.sentiment_health",
        source_path=_DAILY_METRICS_PATH,
        file_freshness=daily_status,
    )
    sentiment_value = sentiment_detail["value"] if sentiment_detail is not None else _uw_sentiment_value(uw_snapshot)
    sentiment_source = "hydrinia_uw_snapshot"
    sentiment_path = _UW_SNAPSHOT_PATH
    sentiment_status = uw_status
    fallback_reason = None
    if sentiment_detail is None and sentiment_value is None:
        sentiment_value = _sentiment_scalar(daily.get("sentiment")) if isinstance(daily, dict) else None
        sentiment_source = "daily_metrics"
        sentiment_path = _DAILY_METRICS_PATH
        sentiment_status = daily_status
        fallback_reason = "UW_SENTIMENT_FALLBACK"
    if sentiment_detail is None and sentiment_value is None:
        sentiment_value = _sentiment_scalar(signals.get("sentiment", signals.get("market_sentiment")))
        sentiment_source = "regime_journal"
        sentiment_path = _REGIME_JOURNAL_PATH
        sentiment_status = journal_status
        fallback_reason = "UW_SENTIMENT_FALLBACK"
    if sentiment_detail is None:
        sentiment_detail = _cached_indicator_detail(
            value=sentiment_value,
            source=sentiment_source,
            source_path=sentiment_path,
            freshness=sentiment_status,
            fallback_reason_code=fallback_reason,
        )
    return vix_detail, sentiment_detail


def _gex_rows(snapshot: dict) -> list[dict]:
    gex = snapshot.get("gex", {}) if isinstance(snapshot, dict) else {}
    if not isinstance(gex, dict):
        return []

    rows = []
    for ticker, raw in sorted(gex.items()):
        if not isinstance(raw, dict):
            continue
        rows.append({
            "ticker": ticker,
            "regime": raw.get("gex_regime") or raw.get("regime") or "UNKNOWN",
            "net_gex": _as_float(raw.get("net_gex")),
            "zero_gamma": _as_float(raw.get("zero_gamma")),
            "max_gex_strike": _as_float(raw.get("max_gex_strike")),
            "pin_risk": raw.get("pin_risk"),
            "source": raw.get("source") or snapshot.get("source") or "hydrinia_uw_snapshot",
        })
    return rows


def _age_label(seconds) -> str:
    if seconds is None:
        return "unknown age"
    try:
        seconds = int(seconds)
    except (TypeError, ValueError):
        return "unknown age"
    if seconds < 60:
        return f"{seconds}s old"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m old"
    return f"{minutes // 60}h old"


def _numeric_gex_completeness(rows: list[dict]) -> dict:
    numeric_fields = ("net_gex", "zero_gamma", "max_gex_strike")
    expected = len(rows) * len(numeric_fields)
    present = 0
    for row in rows:
        present += sum(row.get(field) is not None for field in numeric_fields)
    return {
        "rows": len(rows),
        "numeric_fields_present": present,
        "numeric_fields_expected": expected,
        "complete": expected > 0 and present == expected,
        "partial": present > 0 and present < expected,
    }


def _tail_lines(path: str, max_lines: int) -> list[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return list(deque(f, maxlen=max_lines))
    except OSError:
        return []


def _alert_reason_from_line(line: str) -> str | None:
    match = re.search(r"reason=([^|\s]+)", line)
    if match:
        reason = match.group(1).strip()
        if reason and reason.upper() not in {"N/A", "NA", "NONE", "NULL", "UNKNOWN"}:
            return reason
    upper = line.upper()
    if "CONSENSUS PARALYSIS" in upper:
        return "direction_hold_paralysis"
    if "VETO INVALIDITY" in upper:
        return "veto_invalidity"
    if "UW AUTH FAILED" in upper or "401" in upper and "UW" in upper:
        return "uw_auth_failed"
    if "EMERGENCY DEPLOYMENT" in upper:
        return "emergency_deployment"
    return None


def _log_alert_type(line: str, fallback: str = "system") -> str:
    upper = line.upper()
    if "CONSENSUS PARALYSIS" in upper:
        return "consensus_paralysis"
    if "VETO INVALIDITY" in upper:
        return "veto_invalidity"
    if "UW AUTH FAILED" in upper:
        return "uw_auth_failed"
    if "TRACEBACK" in upper or "EXCEPTION" in upper:
        return "exception"
    if "EMERGENCY" in upper:
        return "emergency"
    return fallback


def _is_structured_log_line(line: str) -> bool:
    return bool(_STRUCTURED_LOG_RE.match(line or ""))


def _dedupe_key(message: str, source: str, alert_type: str) -> str:
    text = re.sub(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\s*\|\s*", "", message or "")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "#", text)
    return f"{source}:{alert_type}:{text[:180]}"


def _dedupe_alerts(alerts: list[dict]) -> list[dict]:
    latest_by_key: dict[str, dict] = {}
    for alert in alerts:
        key = alert.get("dedup_key") or _dedupe_key(
            alert.get("message", ""),
            alert.get("source", ""),
            alert.get("type", "system"),
        )
        alert["dedup_key"] = key
        existing = latest_by_key.get(key)
        if existing is None or (alert.get("timestamp") or "") > (existing.get("timestamp") or ""):
            row = dict(alert)
            row["occurrences"] = int(existing.get("occurrences", 0)) + 1 if existing else 1
            latest_by_key[key] = row
        elif existing is not None:
            existing["occurrences"] = int(existing.get("occurrences", 1)) + 1
    return list(latest_by_key.values())


def _normalized_reason_code(value, fallback: str) -> str:
    raw = str(value or fallback).strip().upper()
    normalized = re.sub(r"[^A-Z0-9]+", "_", raw).strip("_")
    return normalized or fallback


def _alert_epoch(value) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.strptime(text[:19], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_NY)
    return parsed.timestamp()


def _with_alert_lifecycle(alert: dict, *, now: float | None = None) -> dict:
    current = time.time() if now is None else now
    row = {key: value for key, value in alert.items() if not key.startswith("_")}
    severity = str(row.get("severity") or "info").lower()
    alert_type = str(row.get("type") or "system")
    fallback_reason = f"{alert_type}_event"
    reason = row.get("reason") or _alert_reason_from_line(str(row.get("message") or "")) or fallback_reason
    row["reason"] = str(reason)
    row["reason_code"] = _normalized_reason_code(row.get("reason_code") or reason, fallback_reason.upper())

    ttl_value = _as_float(
        row.get("ttl_seconds"),
        _ALERT_TTL_BY_SEVERITY.get(severity, 900),
    )
    ttl_seconds = max(1, int(ttl_value or 900))
    event_epoch = _alert_epoch(row.get("timestamp"))
    age_seconds = int(max(0, current - event_epoch)) if event_epoch is not None else None
    expires_epoch = event_epoch + ttl_seconds if event_epoch is not None else None
    message_upper = str(row.get("message") or "").upper()
    explicit_resolved = bool(row.get("resolved") or row.get("resolved_at")) or any(
        term in message_upper for term in _ALERT_RESOLUTION_TERMS
    )
    state_active = bool(alert.get("_state_active"))

    if state_active:
        status = "active"
        active = True
        resolved = False
        resolution_reason = None
    elif explicit_resolved:
        status = "resolved"
        active = False
        resolved = True
        resolution_reason = "EXPLICIT_RESOLUTION"
    elif severity == "info":
        status = "diagnostic"
        active = False
        resolved = False
        resolution_reason = "INFORMATIONAL_EVENT"
    elif event_epoch is None:
        status = "diagnostic"
        active = False
        resolved = False
        resolution_reason = "TIMESTAMP_UNAVAILABLE"
    elif age_seconds is not None and age_seconds > ttl_seconds:
        status = "resolved"
        active = False
        resolved = True
        resolution_reason = "TTL_EXPIRED"
    else:
        status = "active"
        active = True
        resolved = False
        resolution_reason = None

    row.update({
        "status": status,
        "active": active,
        "resolved": resolved,
        "age_seconds": age_seconds,
        "ttl_seconds": ttl_seconds,
        "expires_at": (
            datetime.fromtimestamp(expires_epoch, tz=timezone.utc).isoformat()
            if expires_epoch is not None
            else None
        ),
        "resolution": {
            "status": "unresolved" if active else status,
            "reason_code": resolution_reason,
            "resolved_at": row.get("resolved_at"),
        },
    })
    return row


def _partition_alerts(
    alerts: list[dict],
    *,
    limit: int,
    now: float | None = None,
) -> tuple[list[dict], list[dict]]:
    classified = [_with_alert_lifecycle(alert, now=now) for alert in _dedupe_alerts(alerts)]
    active = [alert for alert in classified if alert["active"]]
    diagnostics = [alert for alert in classified if not alert["active"]]
    active.sort(key=lambda row: (_alert_priority(row), row.get("timestamp") or ""), reverse=True)
    diagnostics.sort(key=lambda row: (row.get("timestamp") or "", _alert_priority(row)), reverse=True)
    return active[:limit], diagnostics[:limit]


def _alert_priority(row: dict) -> int:
    severity_rank = {"critical": 300, "warning": 200, "info": 100}.get(
        str(row.get("severity") or "").lower(),
        0,
    )
    category = row.get("category")
    alert_type = row.get("type")
    if category == "operator_gate":
        severity_rank += 100
    elif category == "geopolitical":
        severity_rank += 40
    elif alert_type == "emergency_deploy_churn":
        severity_rank += 30
    return severity_rank


def _churn_detector_alerts(limit: int) -> list[dict]:
    rows = []
    if not os.path.isfile(_CHURN_ALERTS_PATH):
        return rows
    for line in _tail_lines(_CHURN_ALERTS_PATH, limit * 2):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or not row.get("alert"):
            continue
        symbols = row.get("symbols_skipped") if isinstance(row.get("symbols_skipped"), list) else []
        symbol_text = ", ".join(str(symbol) for symbol in symbols[:8])
        failed = row.get("failed_cycles_in_window")
        window = row.get("window_minutes")
        budget = row.get("planned_budget_total")
        message = (
            f"Emergency deploy churn: {failed} below-floor cycles in {window}m"
            f" | skipped={symbol_text or 'n/a'}"
            f" | planned_budget_total=${float(budget or 0):,.0f}"
        )
        rows.append({
            "severity": "warning",
            "type": "emergency_deploy_churn",
            "category": "execution_observability",
            "reason": row.get("reason") or "repeated_below_floor_emergency_deploy_cycles",
            "message": message,
            "timestamp": row.get("last_cycle_at") or row.get("recorded_at") or "",
            "source": "emergency_deploy_churn/alerts.jsonl",
            "ttl_seconds": int(_as_float(row.get("ttl_seconds"), 1800) or 1800),
            "resolved": bool(row.get("resolved")),
            "resolved_at": row.get("resolved_at"),
            "details": {
                "first_cycle_at": row.get("first_cycle_at"),
                "last_cycle_at": row.get("last_cycle_at"),
                "failed_cycles_in_window": failed,
                "window_minutes": window,
                "symbols_skipped": symbols,
                "max_floor": row.get("max_floor"),
            },
            "dedup_key": f"emergency_deploy_churn:{row.get('reason')}:{row.get('last_cycle_at')}",
        })
    return rows


def _decision_reason(row: dict) -> str:
    explicit = row.get("reason") or row.get("reason_summary")
    if explicit:
        return str(explicit)
    winner = str(row.get("winner") or row.get("consensus") or row.get("result") or "").upper()
    question_type = str(row.get("question_type") or row.get("phase") or "").upper()
    if row.get("timeout_rejection"):
        return "timeout rejection"
    if row.get("veto_blocked"):
        return "veto blocked"
    breakdown = row.get("vote_breakdown")
    if isinstance(breakdown, dict) and breakdown:
        parts = ", ".join(f"{k}={round(float(v), 2)}" for k, v in breakdown.items() if isinstance(v, (int, float)))
        return f"{question_type or 'decision'} vote breakdown: {parts}" if parts else question_type.lower()
    return question_type.lower() or "decision recorded"


def _platform_vote_counts(platform_votes) -> dict:
    if not isinstance(platform_votes, dict):
        return {}
    counts = {}
    for platform, votes in platform_votes.items():
        if isinstance(votes, list):
            counts[str(platform)] = len(votes)
        elif votes is not None:
            counts[str(platform)] = 1
    return counts


def _vote_vector_paths() -> list[str]:
    """Current vote-vector sink plus recent rotations, newest first."""
    paths = []
    for candidate in [_VOTE_VECTORS_PATH, *glob.glob(f"{_VOTE_VECTORS_PATH}.*")]:
        if os.path.isfile(candidate):
            paths.append(candidate)
    return sorted(set(paths), key=lambda path: _file_mtime_epoch(path) or 0, reverse=True)[:4]


def _recent_decisions(limit: int) -> tuple[list[dict], list[str]]:
    decisions = []
    seen = set()
    source_paths = _vote_vector_paths()
    for path in source_paths:
        for line in _tail_lines(path, max(limit * 80, 800)):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict) or row.get("event") != "decision":
                continue
            row_id = row.get("correlation_id") or (row.get("ts_et"), row.get("symbol"), row.get("question_type"), row.get("winner"))
            if row_id in seen:
                continue
            seen.add(row_id)
            ts = row.get("ts_et")
            if not ts and row.get("ts") is not None:
                try:
                    ts = datetime.fromtimestamp(float(row["ts"]), _NY).isoformat()
                except (TypeError, ValueError, OSError):
                    ts = None
            winner = row.get("winner") or row.get("consensus") or row.get("result")
            question_type = row.get("question_type") or row.get("phase")
            symbol = row.get("symbol")
            severity = "warning" if row.get("timeout_rejection") or row.get("veto_blocked") or str(winner).upper() in {"REJECT", "STRONG_NO"} else "info"
            decisions.append({
                "timestamp": ts,
                "symbol": symbol,
                "category": row.get("category"),
                "phase": question_type,
                "winner": winner,
                "confidence": _as_float(row.get("confidence")),
                "margin": _as_float(row.get("margin")),
                "vote_breakdown": row.get("vote_breakdown") if isinstance(row.get("vote_breakdown"), dict) else {},
                "discrete_vote_counts": row.get("discrete_vote_counts") if isinstance(row.get("discrete_vote_counts"), dict) else {},
                "platform_vote_counts": _platform_vote_counts(row.get("platform_votes")),
                "n_votes": row.get("n_votes"),
                "early_termination": row.get("early_termination"),
                "timeout_rejection": row.get("timeout_rejection"),
                "veto_blocked": row.get("veto_blocked"),
                "reason": _decision_reason(row),
                "severity": severity,
                "message": f"{symbol or 'Unknown'} {question_type or 'decision'} -> {winner}",
                "source_file": os.path.basename(path),
            })
    decisions.sort(key=lambda row: row.get("timestamp") or "", reverse=True)
    return decisions[:limit], source_paths


@router.get("/risk")
async def get_risk(_=Depends(require_auth)):
    """Risk rules and current exposure — CLIO-native (DPOF snapshot + Alpaca)."""
    broker_state, broker_source = await _cached_broker_state()
    equity = 0.0
    last_equity = 0.0
    exposure = {"long": 0.0, "short": 0.0, "symbols": {}}
    position_rejections = []

    if broker_state:
        try:
            equity = float(getattr(broker_state["account"], "equity", 0) or 0)
        except (TypeError, ValueError):
            equity = 0.0
        try:
            last_equity = float(getattr(broker_state["account"], "last_equity", 0) or 0)
        except (TypeError, ValueError):
            last_equity = 0.0
        for position in broker_state.get("positions", []):
            try:
                side = normalize_position_side(getattr(position, "side", None))
                market_value = float(getattr(position, "market_value", 0) or 0)
                symbol = str(getattr(position, "symbol", ""))
                exposure[side] += abs(market_value)
                if equity > 0 and symbol:
                    exposure["symbols"][symbol] = round(abs(market_value) / equity * 100, 1)
            except AlpacaValueError as exc:
                position_rejections.append({
                    "symbol": str(getattr(position, "symbol", "")),
                    "reason_code": "UNKNOWN_POSITION_SIDE",
                    "detail": str(exc),
                })
            except (TypeError, ValueError):
                continue

    max_concentration = max(exposure["symbols"].values()) if exposure["symbols"] else 0

    # CLIO-native risk block: DPOF's own snapshot + Alpaca equity/last_equity.
    # Hydrinia's daily_metrics.json is frozen/stale (retired, 2026-09-24) and is
    # no longer read here at all.
    dpof_snapshot: dict = {}
    try:
        from agent.dpof import readmodel

        dpof_snapshot = await run_in_executor(readmodel.snapshot)
    except Exception as e:
        logger.debug(f"DPOF snapshot unavailable: {e}")

    emergency_stop = bool(dpof_snapshot.get("halt_file") or dpof_snapshot.get("hard_halted"))

    daily_metrics = {
        "source": "clio_dpof",
        "mode": dpof_snapshot.get("mode"),
        "account_equity": equity,
        "account_last_equity": last_equity,
        "daily_pl": equity - last_equity,
        "realized_pnl_today": dpof_snapshot.get("realized_pnl_today"),
        "realized_pnl_week": dpof_snapshot.get("realized_pnl_week"),
        "entries_today": dpof_snapshot.get("entries_today"),
        "open_count": dpof_snapshot.get("open_count"),
        "closed_today_count": dpof_snapshot.get("closed_today_count"),
        "closed_today_wins": dpof_snapshot.get("closed_today_wins"),
        "halt_file": dpof_snapshot.get("halt_file"),
        "hard_halted": dpof_snapshot.get("hard_halted"),
        "halt_reason": dpof_snapshot.get("halt_reason"),
    }

    return {
        "equity": equity,
        "exposure": exposure,
        "max_concentration_pct": max_concentration,
        "daily_metrics": daily_metrics,
        "dpof": dpof_snapshot,
        "emergency_stop": emergency_stop,
        "position_rejections": position_rejections,
        "exposure_provenance": broker_source,
    }


def _empty_regime_indicator(reason_code: str) -> dict:
    return {
        "value": None,
        "source": "unavailable",
        "source_path": None,
        "delivery": "none",
        "cached": False,
        "vendor_call": False,
        "as_of": None,
        "stale": True,
        "age_seconds": None,
        "ttl_seconds": None,
        "reason_code": reason_code,
    }


async def _clio_native_regime() -> Optional[dict]:
    """CLIO-native market regime (agent.integrations.market_regime). None if
    the module is missing or has nothing fresh enough — never raises."""
    try:
        from agent.integrations.market_regime import get_market_regime
    except Exception as e:
        logger.debug(f"market_regime module unavailable: {e}")
        return None
    try:
        return await run_in_executor(get_market_regime)
    except Exception as e:
        logger.debug(f"market_regime.get_market_regime() failed: {e}")
        return None


@router.get("/regime")
async def get_regime(_=Depends(require_auth)):
    """Market regime — CLIO-native (agent.integrations.market_regime) by default.

    Falls back to the legacy Hydrinia chain (daily_metrics/regime_journal/UW
    snapshot) only when CLIO_HYDRINIA_BRIDGE_ENABLED=1 — those files are
    otherwise frozen/stale (Hydrinia retired, 2026-09-24).
    """
    native = await _clio_native_regime()
    if native:
        vix_detail = {
            "value": native.get("vix"),
            "source": "clio_native_market_regime",
            "source_path": None,
            "delivery": "live",
            "cached": False,
            "vendor_call": True,
            "as_of": native.get("as_of"),
            "stale": False,
            "age_seconds": None,
            "ttl_seconds": None,
            "reason_code": "OK" if native.get("vix") is not None else "VALUE_UNAVAILABLE",
        }
        sentiment_detail = _empty_regime_indicator("NOT_TRACKED_CLIO_NATIVE")
        return {
            "regime": native.get("regime") or "UNKNOWN",
            "confidence": native.get("confidence"),
            "confidence_source": "clio_native",
            "vix": native.get("vix"),
            "sentiment": None,
            "source_status": {
                "confidence": "clio_native",
                "vix": "ok" if native.get("vix") is not None else "unavailable",
                "sentiment": "unavailable",
            },
            "vix_detail": vix_detail,
            "uw_sentiment": sentiment_detail,
            "market_indicators": {"vix": vix_detail, "uw_sentiment": sentiment_detail},
            "signals": {
                "spy_close": native.get("spy_close"),
                "spy_sma20": native.get("spy_sma20"),
                "spy_sma50": native.get("spy_sma50"),
            },
            "updated": native.get("as_of"),
            "journal": {},
            "source": native.get("source") or "clio_native",
        }

    bridge_on = False
    try:
        from agent.hydrinia_bridge import bridge_enabled
        bridge_on = bridge_enabled()
    except Exception as e:
        logger.debug(f"bridge_enabled() unavailable: {e}")

    if not bridge_on:
        empty = _empty_regime_indicator("HYDRINIA_RETIRED")
        return {
            "regime": "UNKNOWN",
            "confidence": None,
            "confidence_source": "unavailable",
            "vix": None,
            "sentiment": None,
            "source_status": {"confidence": "unavailable", "vix": "unavailable", "sentiment": "unavailable"},
            "vix_detail": empty,
            "uw_sentiment": empty,
            "market_indicators": {"vix": empty, "uw_sentiment": empty},
            "signals": {},
            "updated": None,
            "journal": {},
            "source": "hydrinia_retired",
        }

    # Legacy Hydrinia chain — bridge re-enabled only.
    daily = read_json(_DAILY_METRICS_PATH) or {}
    regime_journal = read_json(_REGIME_JOURNAL_PATH) or {}
    uw_snapshot = read_json(_UW_SNAPSHOT_PATH) or {}

    transitions = regime_journal.get("transitions", []) if isinstance(regime_journal, dict) else []
    latest_transition = transitions[-1] if isinstance(transitions, list) and transitions else {}
    signals = latest_transition.get("signals", {}) if isinstance(latest_transition, dict) else {}
    if not isinstance(signals, dict):
        signals = {}

    market_regime = daily.get("market_regime", "UNKNOWN")
    if isinstance(market_regime, dict):
        regime_str = market_regime.get("regime", "UNKNOWN")
        confidence = market_regime.get("confidence")
    else:
        regime_str = str(market_regime) if market_regime else "UNKNOWN"
        confidence = daily.get("regime_confidence")

    journal_regime = None
    journal_confidence = None
    if isinstance(regime_journal, dict):
        journal_regime = regime_journal.get("current_regime")
    if isinstance(latest_transition, dict):
        journal_regime = journal_regime or latest_transition.get("new_regime")
        journal_confidence = latest_transition.get("confidence")

    if journal_regime and (not regime_str or regime_str == "UNKNOWN"):
        regime_str = journal_regime

    confidence_source = "daily_metrics" if confidence is not None else "unavailable"
    try:
        confidence_float = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence_float = None
    try:
        journal_confidence_float = float(journal_confidence) if journal_confidence is not None else None
    except (TypeError, ValueError):
        journal_confidence_float = None

    if journal_confidence_float is not None and confidence_float is None:
        confidence = journal_confidence_float
        confidence_source = "regime_journal"

    vix_detail, sentiment_detail = _market_indicator_details(daily, regime_journal, uw_snapshot)
    vix = vix_detail["value"]
    sentiment = sentiment_detail["value"]

    return {
        "regime": regime_str,
        "confidence": confidence,
        "confidence_source": confidence_source,
        "vix": vix,
        "sentiment": sentiment,
        "source_status": {
            "confidence": confidence_source,
            "vix": "unavailable" if vix is None else "stale" if vix_detail["stale"] else "ok",
            "sentiment": (
                "unavailable"
                if sentiment is None
                else "stale" if sentiment_detail["stale"] else "ok"
            ),
        },
        "vix_detail": vix_detail,
        "uw_sentiment": sentiment_detail,
        "market_indicators": {
            "vix": vix_detail,
            "uw_sentiment": sentiment_detail,
        },
        "signals": signals,
        "updated": daily.get("last_updated") or regime_journal.get("regime_start"),
        "journal": regime_journal,
        "source": "hydrinia_legacy_bridge",
    }


def _hydrinia_bridge_enabled() -> bool:
    try:
        from agent.hydrinia_bridge import bridge_enabled

        return bridge_enabled()
    except Exception as e:
        logger.debug(f"bridge_enabled() unavailable: {e}")
        return False


@router.get("/market-structure")
async def get_market_structure(_=Depends(require_auth)):
    """Display-only GEX/dealer-gamma bridge health and latest exported values.

    Hydrinia's UW/vol-surface exports are retired (2026-09-24) and frozen on
    disk — this returns an empty, correctly-shaped payload unless the bridge
    is explicitly re-enabled.
    """
    if not _hydrinia_bridge_enabled():
        empty_freshness = {"path": None, "exists": False, "status": "missing", "timestamp": None,
                            "mtime": None, "age_seconds": None, "ttl_seconds": None,
                            "effective_ttl_seconds": None, "market_hours": _market_hours_now()}
        return {
            "source_status": {"uw_snapshot": empty_freshness, "vol_surface": empty_freshness},
            "data_quality": {
                "gex": {"rows": 0, "numeric_fields_present": 0, "numeric_fields_expected": 0,
                        "complete": False, "partial": False},
                "dealer_gamma_complete": False,
            },
            "gex": [],
            "dealer_gamma": {
                "regime": "INCOMPLETE",
                "net_gamma_dollars": None,
                "total_vega": None,
                "vanna_direction": None,
                "weighted_avg_iv": None,
            },
            "spy_vol_surface": {
                "term_structure": {},
                "skew": {},
                "vov": {},
                "iv_rv_spread": None,
                "data_source": None,
            },
            "warnings": [
                "Hydrinia is retired — UW/vol-surface exports are frozen. "
                "Set CLIO_HYDRINIA_BRIDGE_ENABLED=1 to re-enable this bridge."
            ],
            "source": "hydrinia_retired",
        }

    uw_snapshot = read_json(_UW_SNAPSHOT_PATH) or {}
    vol_surface = read_json(_VOL_SURFACE_PATH) or {}
    uw_status = _source_freshness(_UW_SNAPSHOT_PATH, uw_snapshot, 300)
    vol_status = _source_freshness(_VOL_SURFACE_PATH, vol_surface, 2400)

    portfolio = vol_surface.get("portfolio", {}) if isinstance(vol_surface, dict) else {}
    spy_surface = vol_surface.get("spy", {}) if isinstance(vol_surface, dict) else {}
    term_structure = spy_surface.get("term_structure", {}) if isinstance(spy_surface, dict) else {}
    skew = spy_surface.get("skew", {}) if isinstance(spy_surface, dict) else {}
    vov = spy_surface.get("vov", {}) if isinstance(spy_surface, dict) else {}
    net_gamma = _as_float(portfolio.get("net_gamma_dollars")) if isinstance(portfolio, dict) else None
    total_vega = _as_float(portfolio.get("total_vega")) if isinstance(portfolio, dict) else None
    weighted_avg_iv = _as_float(portfolio.get("weighted_avg_iv")) if isinstance(portfolio, dict) else None
    gex_rows = _gex_rows(uw_snapshot)
    gex_quality = _numeric_gex_completeness(gex_rows)

    dealer_gamma_complete = any(value not in (None, 0, 0.0) for value in (net_gamma, total_vega, weighted_avg_iv))
    if not dealer_gamma_complete:
        gamma_regime = "INCOMPLETE"
    elif net_gamma > 0:
        gamma_regime = "POSITIVE_GAMMA"
    elif net_gamma < 0:
        gamma_regime = "NEGATIVE_GAMMA"
    else:
        gamma_regime = "NEUTRAL"

    warnings = []
    if uw_status["status"] == "missing":
        warnings.append("Hydrinia UW snapshot is missing; GEX will remain unavailable until the next market-hours export.")
    elif uw_status["status"] != "fresh":
        warnings.append(f"Hydrinia UW snapshot is {uw_status['status']} ({_age_label(uw_status.get('age_seconds'))}); use GEX as historical context only.")
    elif gex_rows and not gex_quality["complete"]:
        warnings.append("Hydrinia UW snapshot is fresh but GEX numeric fields are incomplete; regime labels are display-only.")
    if vol_status["status"] == "missing":
        warnings.append("Vol-surface export is missing; dealer-gamma context is unavailable.")
    elif vol_status["status"] != "fresh":
        warnings.append(f"Vol-surface export is {vol_status['status']} ({_age_label(vol_status.get('age_seconds'))}); dealer-gamma context is display-only.")
    elif all(value in (None, 0, 0.0) for value in (net_gamma, total_vega, weighted_avg_iv)):
        warnings.append("Vol-surface export is fresh but dealer-gamma/vega/IV fields are zero or missing; verify upstream portfolio Greeks.")

    return {
        "source_status": {
            "uw_snapshot": uw_status,
            "vol_surface": vol_status,
        },
        "data_quality": {
            "gex": gex_quality,
            "dealer_gamma_complete": dealer_gamma_complete,
        },
        "gex": gex_rows,
        "dealer_gamma": {
            "regime": gamma_regime,
            "net_gamma_dollars": net_gamma,
            "total_vega": total_vega,
            "vanna_direction": portfolio.get("vanna_direction") if isinstance(portfolio, dict) else None,
            "weighted_avg_iv": weighted_avg_iv,
        },
        "spy_vol_surface": {
            "term_structure": term_structure,
            "skew": skew,
            "vov": vov,
            "iv_rv_spread": vol_surface.get("iv_rv_spread") if isinstance(vol_surface, dict) else None,
            "data_source": spy_surface.get("data_source") if isinstance(spy_surface, dict) else None,
        },
        "warnings": warnings,
        "source": "hydrinia_legacy_bridge",
    }


@router.get("/decisions")
async def get_decisions(limit: int = Query(20, ge=1, le=100), _=Depends(require_auth)):
    """Recent structured Hydrinia deliberation decisions from vote_vectors.jsonl.

    Hydrinia's deliberation log is retired (2026-09-24) and frozen on disk —
    returns an empty, correctly-shaped payload unless the bridge is explicitly
    re-enabled. For CLIO's own DPOF decisions, see GET /api/dpof/decisions.
    """
    if not _hydrinia_bridge_enabled():
        return {
            "decisions": [],
            "source": "hydrinia_retired",
            "source_files": [],
            "rotation_aware": True,
            "updated": None,
        }
    decisions, source_paths = _recent_decisions(limit)
    newest_mtime = max((_file_mtime_epoch(path) or 0 for path in source_paths), default=0)
    return {
        "decisions": decisions,
        "source": _VOTE_VECTORS_PATH,
        "source_files": source_paths,
        "rotation_aware": True,
        "updated": datetime.fromtimestamp(newest_mtime, _NY).isoformat() if newest_mtime else None,
    }


def _dpof_thesis_text(p) -> tuple[str, Optional[float]]:
    """(thesis text, conviction) for a DPOF PositionRecord's entry rationale."""
    thesis_text = str(getattr(p, "thesis", "") or "").strip()
    conviction = None
    entry_decision = getattr(p, "entry_decision", None)
    if isinstance(entry_decision, dict):
        conviction = entry_decision.get("conviction")
    entry_signal = getattr(p, "entry_signal", None)
    if isinstance(entry_signal, dict):
        if conviction is None:
            conviction = entry_signal.get("conviction")
        if not thesis_text:
            reasons = entry_signal.get("reasons")
            if isinstance(reasons, list) and reasons:
                thesis_text = "; ".join(str(r) for r in reasons[:3])
    if not isinstance(conviction, (int, float)):
        conviction = None
    return thesis_text, conviction


@router.get("/theses")
async def get_theses(_=Depends(require_auth)):
    """Active investment theses — CLIO's own DPOF open positions, plus any
    other Alpaca account positions outside DPOF's book."""
    theses = []
    position_rejections = []

    try:
        from agent.dpof import readmodel

        dpof_open = await run_in_executor(readmodel.open_positions)
    except Exception as e:
        logger.debug(f"DPOF open positions unavailable: {e}")
        dpof_open = []

    leg_symbols: set = set()
    for pos in dpof_open:
        try:
            from agent.dpof.readmodel import position_label
            contract = position_label(pos)
        except Exception:
            contract = pos.underlying
        for leg in (pos.legs or []):
            if isinstance(leg, dict):
                leg_sym = str(leg.get("symbol") or "").upper()
                if leg_sym:
                    leg_symbols.add(leg_sym)
        thesis_text, conviction = _dpof_thesis_text(pos)
        direction_lower = str(pos.direction or "").lower()
        market_direction = (
            "BULLISH" if direction_lower == "bullish"
            else "BEARISH" if direction_lower == "bearish"
            else "NEUTRAL"
        )
        theses.append({
            "symbol": pos.underlying,
            "contract": contract,
            "direction": market_direction,
            "market_exposure_direction": market_direction,
            "contract_direction": market_direction,
            "thesis": thesis_text or f"{pos.structure} position",
            "conviction": conviction,
            "category": pos.structure,
            "dpof_id": pos.id,
            "status": pos.status,
        })

    # Reuse the bounded broker cache so page polling does not create vendor fan-out.
    broker_state, broker_source = await _cached_broker_state()
    if broker_state:
        live_positions = broker_state.get("positions", [])
        pos_symbols = {str(pos.underlying).upper() for pos in dpof_open} | leg_symbols
        for p in live_positions:
            sym = getattr(p, "symbol", "")
            if str(sym).upper() not in pos_symbols:
                try:
                    side = normalize_position_side(getattr(p, "side", None))
                    contract_direction = "BULLISH" if side == "long" else "BEARISH"
                    market_direction = _infer_market_exposure_direction(
                        {"symbol": sym, "direction": contract_direction, "is_short": side == "short"},
                        contract_direction,
                    )
                    theses.append({
                        "symbol": sym,
                        "contract": sym,
                        "direction": market_direction,
                        "market_exposure_direction": market_direction,
                        "contract_direction": contract_direction,
                        "thesis": f"Active {side} position (outside DPOF)",
                        "conviction": None,
                        "category": "",
                    })
                except AlpacaValueError as exc:
                    position_rejections.append({
                        "symbol": str(sym),
                        "reason_code": "UNKNOWN_POSITION_SIDE",
                        "detail": str(exc),
                    })

    return {
        "theses": theses,
        "position_rejections": position_rejections,
        "broker_provenance": broker_source,
        "source": "clio_dpof",
    }


@router.get("/alerts")
async def get_alerts(limit: int = Query(20, ge=1, le=100), _=Depends(require_auth)):
    """Recent Hydrinia alerts from structured-ish logs and system events.

    Every source this route reads (trades.log, hydrinia.log, EMERGENCY_STOP,
    the emergency-deploy-churn detector, geopolitical_intel.json) lives under
    Hydrinia's retired state tree and is frozen on disk — this whole assembly
    is skipped unless the bridge is explicitly re-enabled.
    """
    alerts = []
    bridge_on = _hydrinia_bridge_enabled()

    # Read trades.log for recent activity
    trades_log = os.path.join(_STATE_DIR, "logs", "trades.log")
    if bridge_on and os.path.isfile(trades_log):
        try:
            # Parse last N lines for alert-worthy events
            for line in _tail_lines(trades_log, limit * 4):
                line = line.strip()
                if not line:
                    continue
                severity = "info"
                if any(kw in line.upper() for kw in ["ERROR", "FAIL", "EMERGENCY", "STOP"]):
                    severity = "critical"
                elif any(kw in line.upper() for kw in ["WARNING", "REJECT", "BLOCKED", "RISK"]):
                    severity = "warning"
                elif any(kw in line.upper() for kw in ["CONSENSUS", "REGIME", "TRADE", "ORDER", "FILL", "FINAL DECISION", "DELIBERATION RESULT", "VETO"]):
                    severity = "info"
                else:
                    continue  # skip non-alert lines

                # Extract timestamp if present
                ts = ""
                if "|" in line and len(line) > 20:
                    ts = line[:19].strip()

                alerts.append({
                    "severity": severity,
                    "type": _log_alert_type(line, "trade_event"),
                    "category": "trades",
                    "reason": _alert_reason_from_line(line),
                    "message": line,
                    "timestamp": ts,
                    "source": "trades.log",
                    "details": {},
                })
        except Exception as e:
            logger.debug(f"trades.log read failed: {e}")

    # Hydrinia's engine warnings/errors live in hydrinia.log/journald, not trades.log.
    # Keep this display-only and selective so routine INFO deliberation chatter does not
    # become an operator alert.
    hydrinia_log = os.path.join(_STATE_DIR, "logs", "hydrinia.log")
    if bridge_on and os.path.isfile(hydrinia_log):
        try:
            for line in _tail_lines(hydrinia_log, max(limit * 50, 1000)):
                line = line.strip()
                if not line:
                    continue
                upper = line.upper()
                structured = _is_structured_log_line(line)
                if not structured and "TRACEBACK" not in upper and "EXCEPTION" not in upper:
                    continue
                parts = [part.strip().upper() for part in line.split("|")]
                level = parts[1] if len(parts) > 1 else ""
                if level == "DEBUG":
                    continue
                if level in ("CRITICAL", "ERROR") or any(kw in upper for kw in ["EXCEPTION", "TRACEBACK", "EMERGENCY STOP"]):
                    severity = "critical"
                elif level == "WARNING" or (
                    structured
                    and any(kw in upper for kw in [
                        "CONSENSUS PARALYSIS", "VETO INVALIDITY", "UW AUTH FAILED", "EMERGENCY STOP"
                    ])
                ):
                    severity = "warning"
                else:
                    continue

                ts = ""
                if "|" in line and len(line) > 20:
                    ts = line[:19].strip()

                alerts.append({
                    "severity": severity,
                    "type": _log_alert_type(line, "system"),
                    "category": "engine",
                    "reason": _alert_reason_from_line(line),
                    "message": line,
                    "timestamp": ts,
                    "source": "hydrinia.log",
                    "details": {},
                })
        except Exception as e:
            logger.debug(f"hydrinia.log read failed: {e}")

    if bridge_on:
        alerts.extend(_churn_detector_alerts(limit))

        # Check EMERGENCY_STOP
        emergency_stop_path = os.path.join(_STATE_DIR, "EMERGENCY_STOP")
        if os.path.exists(emergency_stop_path):
            emergency_mtime = _file_mtime_epoch(emergency_stop_path)
            alerts.insert(0, {
                "severity": "critical",
                "type": "emergency",
                "category": "operator_gate",
                "reason": "emergency_stop_active",
                "message": "EMERGENCY STOP is active — all trading halted",
                "timestamp": (
                    datetime.fromtimestamp(emergency_mtime, _NY).isoformat()
                    if emergency_mtime
                    else ""
                ),
                "source": "state/EMERGENCY_STOP",
                "details": {},
                "_state_active": True,
            })

        # Check geopolitical intel for elevated risk
        geo_path = os.path.join(_STATE_DIR, "geopolitical_intel.json")
        geo = read_json(geo_path) or {}
        geo_freshness = _source_freshness(geo_path, geo, 3600)
        cii_scores = geo.get("cii_scores", {})
        for country, score in cii_scores.items():
            if isinstance(score, (int, float)) and score > 70:
                alerts.append({
                    "severity": "warning",
                    "type": "geopolitical",
                    "category": "geopolitical",
                    "reason": "cii_elevated",
                    "message": f"Elevated geopolitical risk: {country} CII score {score:.0f}",
                    "timestamp": geo_freshness.get("timestamp") or geo_freshness.get("mtime") or "",
                    "source": "geopolitical_intel.json",
                    "details": {"country": country, "cii_score": score},
                    "ttl_seconds": geo_freshness.get("effective_ttl_seconds") or 3600,
                    "_state_active": geo_freshness.get("status") == "fresh",
                })

    active_alerts, diagnostics = _partition_alerts(alerts, limit=limit)
    return {
        # Legacy clients consume `alerts`; it now means genuinely active alerts.
        "alerts": active_alerts,
        "active_alerts": active_alerts,
        "diagnostics": diagnostics,
        "counts": {
            "active": len(active_alerts),
            "diagnostics": len(diagnostics),
        },
        "as_of": datetime.now(timezone.utc).isoformat(),
        "reason_code": "OK" if active_alerts else ("HYDRINIA_RETIRED" if not bridge_on else "NO_ACTIVE_ALERTS"),
        "source": "hydrinia_legacy_bridge" if bridge_on else "hydrinia_retired",
    }
