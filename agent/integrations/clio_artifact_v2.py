"""Immutable, lane-agnostic CLIO artifact v2 contract."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple
import uuid


SCHEMA_VERSION = "clio.artifact.v2"
ARTIFACT_TYPE = "trade_idea_context"
DEFAULT_TTL_SECONDS = 48 * 3600
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
_SECRET_KEY_RE = re.compile(
    r"(api[_-]?key|secret|password|passwd|token|authorization|private[_-]?key)",
    re.IGNORECASE,
)
_SECRET_VALUE_RE = re.compile(
    r"(?i)(?:bearer\s+|sk-[A-Za-z0-9_-]{12,}|xox[baprs]-[A-Za-z0-9-]{12,})"
)
_FORBIDDEN_EXECUTION_KEYS = {
    "broker",
    "cancel_order",
    "client_order_id",
    "execution_allowed",
    "governor",
    "limit_price",
    "notional",
    "order",
    "order_class",
    "order_id",
    "order_type",
    "position_size",
    "position_size_guidance",
    "qty",
    "quantity",
    "risk_limit",
    "sizing_multiplier",
    "stop_price",
    "submit_order",
    "time_in_force",
}
_REQUIRED_ENVELOPE_FIELDS = {
    "schema_version",
    "artifact_id",
    "content_digest",
    "producer",
    "producer_version",
    "source_tags",
    "produced_at",
    "observed_at",
    "expires_at",
    "confidence",
    "redactions",
    "quality",
    "supersedes",
    "payload",
}
_ALLOWED_ENVELOPE_FIELDS = set(_REQUIRED_ENVELOPE_FIELDS)
_REQUIRED_TRADE_FIELDS = {
    "artifact_type",
    "legacy_id",
    "ticker",
    "direction",
    "entry_trigger",
    "stop_loss",
    "take_profit",
    "timeframe",
    "thesis",
    "risk_factors",
    "sector",
    "source_category",
    "source_urls",
}


class ClioArtifactError(ValueError):
    def __init__(self, reason_code: str, detail: str = "") -> None:
        self.reason_code = str(reason_code or "CLIO_ARTIFACT_REJECT")
        self.detail = str(detail or "")
        super().__init__(f"{self.reason_code}:{self.detail}" if self.detail else self.reason_code)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def new_artifact_id() -> str:
    return f"clio2_{time.time_ns():x}_{uuid.uuid4().hex[:16]}"


def normalize_direction(value: Any) -> str:
    direction = str(value or "").strip().upper()
    if direction in {"LONG", "BUY", "BULLISH", "BULL"}:
        return "LONG"
    if direction in {"SHORT", "SELL", "BEARISH", "BEAR"}:
        return "SHORT"
    if direction in {"HOLD", "NEUTRAL", "NONE", "FLAT", "ABSTAIN", "NO_TRADE"}:
        raise ClioArtifactError("NON_DIRECTIONAL", direction)
    raise ClioArtifactError("UNKNOWN_DIRECTION", direction or "missing")


def _parse_utc(value: Any, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ClioArtifactError("INVALID_TIMESTAMP", field) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ClioArtifactError("NAIVE_TIMESTAMP", field)
    return parsed.astimezone(timezone.utc)


def _walk_forbidden(value: Any, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in _FORBIDDEN_EXECUTION_KEYS:
                raise ClioArtifactError("EXECUTION_FIELD_FORBIDDEN", f"{path}.{key}")
            _walk_forbidden(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _walk_forbidden(child, f"{path}[{index}]")


def _redact(value: Any, path: str = "payload") -> Tuple[Any, list[str]]:
    redactions: list[str] = []
    if isinstance(value, Mapping):
        output: Dict[str, Any] = {}
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if _SECRET_KEY_RE.search(str(key)):
                output[str(key)] = "[REDACTED]"
                redactions.append(child_path)
                continue
            output[str(key)], nested = _redact(child, child_path)
            redactions.extend(nested)
        return output, redactions
    if isinstance(value, list):
        output_list = []
        for index, child in enumerate(value):
            cleaned, nested = _redact(child, f"{path}[{index}]")
            output_list.append(cleaned)
            redactions.extend(nested)
        return output_list, redactions
    if isinstance(value, tuple):
        return _redact(list(value), path)
    if isinstance(value, str) and _SECRET_VALUE_RE.search(value):
        return "[REDACTED]", [path]
    return value, redactions


def _finite_confidence(value: Any) -> float:
    if isinstance(value, bool):
        raise ClioArtifactError("INVALID_CONFIDENCE", "boolean")
    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise ClioArtifactError("INVALID_CONFIDENCE", "not_numeric") from exc
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ClioArtifactError("INVALID_CONFIDENCE", str(value))
    return confidence


def _validate_trade_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    _walk_forbidden(payload)
    missing = sorted(_REQUIRED_TRADE_FIELDS - set(payload))
    if missing:
        raise ClioArtifactError("MISSING_PAYLOAD_FIELDS", ",".join(missing))
    unknown = sorted(set(payload) - _REQUIRED_TRADE_FIELDS)
    if unknown:
        raise ClioArtifactError("UNKNOWN_PAYLOAD_FIELDS", ",".join(unknown))
    if payload.get("artifact_type") != ARTIFACT_TYPE:
        raise ClioArtifactError("UNSUPPORTED_ARTIFACT_TYPE", str(payload.get("artifact_type")))
    ticker = str(payload.get("ticker") or "").strip().upper()
    if not _TICKER_RE.fullmatch(ticker):
        raise ClioArtifactError("INVALID_TICKER", ticker or "missing")
    direction = normalize_direction(payload.get("direction"))
    for field in ("entry_trigger", "stop_loss", "take_profit", "timeframe", "thesis"):
        if not str(payload.get(field) or "").strip():
            raise ClioArtifactError("MISSING_PAYLOAD_FIELD", field)
    if not isinstance(payload.get("risk_factors"), list):
        raise ClioArtifactError("INVALID_PAYLOAD_FIELD", "risk_factors")
    if not isinstance(payload.get("source_urls"), list):
        raise ClioArtifactError("INVALID_PAYLOAD_FIELD", "source_urls")
    normalized = dict(payload)
    normalized["ticker"] = ticker
    normalized["direction"] = direction
    normalized["risk_factors"] = [str(item) for item in payload.get("risk_factors", [])]
    normalized["source_urls"] = [str(item) for item in payload.get("source_urls", [])]
    return normalized


def build_trade_idea_artifact(
    idea: Mapping[str, Any],
    *,
    producer: str = "clio.learning_engine",
    producer_version: str = "2",
    source_tags: Iterable[str] = ("trade_idea",),
    produced_at: Optional[datetime] = None,
    observed_at: Any = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    supersedes: Optional[str] = None,
) -> Dict[str, Any]:
    produced = produced_at or utc_now()
    if produced.tzinfo is None or produced.utcoffset() is None:
        raise ClioArtifactError("NAIVE_TIMESTAMP", "produced_at")
    produced = produced.astimezone(timezone.utc)
    observed = _parse_utc(observed_at or idea.get("created") or produced.isoformat(), "observed_at")
    ttl = int(ttl_seconds)
    if ttl <= 0:
        raise ClioArtifactError("INVALID_EXPIRY", str(ttl_seconds))
    tags = sorted({str(tag).strip().lower() for tag in source_tags if str(tag).strip()})
    if not tags:
        raise ClioArtifactError("MISSING_SOURCE_TAGS")

    raw_payload = {
        "artifact_type": ARTIFACT_TYPE,
        "legacy_id": str(idea.get("id") or ""),
        "ticker": str(idea.get("ticker") or ""),
        "direction": str(idea.get("direction") or ""),
        "entry_trigger": str(idea.get("entry_trigger") or ""),
        "stop_loss": str(idea.get("stop_loss") or ""),
        "take_profit": str(idea.get("take_profit") or ""),
        "timeframe": str(idea.get("timeframe") or ""),
        "thesis": str(idea.get("thesis") or ""),
        "risk_factors": list(idea.get("risk_factors") or []),
        "sector": str(idea.get("sector") or ""),
        "source_category": str(idea.get("source_category") or ""),
        "source_urls": list(idea.get("source_urls") or []),
    }
    cleaned, redactions = _redact(raw_payload)
    payload = _validate_trade_payload(cleaned)
    confidence = _finite_confidence(idea.get("confidence"))
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": new_artifact_id(),
        "content_digest": canonical_digest(payload),
        "producer": str(producer or "").strip(),
        "producer_version": str(producer_version or "").strip(),
        "source_tags": tags,
        "produced_at": produced.isoformat(),
        "observed_at": observed.isoformat(),
        "expires_at": (produced + timedelta(seconds=ttl)).isoformat(),
        "confidence": confidence,
        "redactions": redactions,
        "quality": {
            "status": "producer_validated",
            "issues": [],
            "exact_risk_fields": [
                "ticker",
                "direction",
                "entry_trigger",
                "stop_loss",
                "take_profit",
                "timeframe",
            ],
            "scrubber": "clio_artifact_v2",
        },
        "supersedes": str(supersedes).strip() if supersedes else None,
        "payload": payload,
    }
    validate_clio_artifact(artifact, now=produced, allow_expired=True)
    return artifact


def validate_clio_artifact(
    artifact: Mapping[str, Any],
    *,
    now: Optional[datetime] = None,
    allow_expired: bool = False,
) -> Dict[str, Any]:
    if not isinstance(artifact, Mapping):
        raise ClioArtifactError("ARTIFACT_NOT_OBJECT")
    missing = sorted(_REQUIRED_ENVELOPE_FIELDS - set(artifact))
    if missing:
        raise ClioArtifactError("MISSING_ENVELOPE_FIELDS", ",".join(missing))
    unknown = sorted(set(artifact) - _ALLOWED_ENVELOPE_FIELDS)
    if unknown:
        raise ClioArtifactError("UNKNOWN_ENVELOPE_FIELDS", ",".join(unknown))
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ClioArtifactError("UNSUPPORTED_SCHEMA", str(artifact.get("schema_version")))
    if not str(artifact.get("artifact_id") or "").startswith("clio2_"):
        raise ClioArtifactError("INVALID_ARTIFACT_ID")
    if not str(artifact.get("producer") or "").strip():
        raise ClioArtifactError("MISSING_PRODUCER")
    if not str(artifact.get("producer_version") or "").strip():
        raise ClioArtifactError("MISSING_PRODUCER_VERSION")
    tags = artifact.get("source_tags")
    if not isinstance(tags, list) or not tags or any(not str(tag).strip() for tag in tags):
        raise ClioArtifactError("INVALID_SOURCE_TAGS")
    if not isinstance(artifact.get("redactions"), list):
        raise ClioArtifactError("MISSING_REDACTIONS")
    quality = artifact.get("quality")
    if not isinstance(quality, Mapping):
        raise ClioArtifactError("INVALID_QUALITY")
    if quality.get("status") not in {"producer_validated", "legacy_converted_offline"}:
        raise ClioArtifactError("INVALID_QUALITY_STATUS", str(quality.get("status")))
    if not isinstance(quality.get("issues"), list):
        raise ClioArtifactError("INVALID_QUALITY_ISSUES")
    if artifact.get("supersedes") is not None and not str(artifact.get("supersedes")).startswith("clio2_"):
        raise ClioArtifactError("INVALID_SUPERSESSION")

    checked = now or utc_now()
    if checked.tzinfo is None or checked.utcoffset() is None:
        raise ClioArtifactError("NAIVE_TIMESTAMP", "now")
    checked = checked.astimezone(timezone.utc)
    produced = _parse_utc(artifact.get("produced_at"), "produced_at")
    observed = _parse_utc(artifact.get("observed_at"), "observed_at")
    expires = _parse_utc(artifact.get("expires_at"), "expires_at")
    if produced > checked + timedelta(minutes=5) or observed > checked + timedelta(minutes=5):
        raise ClioArtifactError("FUTURE_TIMESTAMP")
    if expires <= produced:
        raise ClioArtifactError("INVALID_EXPIRY")
    if not allow_expired and checked >= expires:
        raise ClioArtifactError("STALE_ARTIFACT", expires.isoformat())
    confidence = _finite_confidence(artifact.get("confidence"))

    payload_raw = artifact.get("payload")
    if not isinstance(payload_raw, Mapping):
        raise ClioArtifactError("INVALID_PAYLOAD")
    payload = _validate_trade_payload(payload_raw)
    digest = str(artifact.get("content_digest") or "")
    if not digest or canonical_digest(payload) != digest:
        raise ClioArtifactError("CONTENT_DIGEST_MISMATCH")

    normalized = deepcopy(dict(artifact))
    normalized["confidence"] = confidence
    normalized["payload"] = payload
    return normalized


def artifact_filename(artifact: Mapping[str, Any]) -> str:
    artifact_id = str(artifact.get("artifact_id") or "")
    if not artifact_id.startswith("clio2_"):
        raise ClioArtifactError("INVALID_ARTIFACT_ID")
    return f"{artifact_id}.json"


def atomic_write_artifact(path: Path | str, artifact: Mapping[str, Any]) -> Path:
    validated = validate_clio_artifact(artifact, allow_expired=True)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(validated, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def quarantine_artifact_file(
    source_path: Path | str,
    *,
    reason_code: str,
    quarantine_dir: Path | str,
    detail: str = "",
) -> Path:
    source = Path(source_path)
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    destination_dir = Path(quarantine_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    if destination.exists():
        destination = destination_dir / f"{source.stem}.{digest[:12]}{source.suffix}"
    os.replace(source, destination)
    record = {
        "schema_version": "clio.quarantine.v1",
        "source_name": source.name,
        "quarantined_name": destination.name,
        "source_sha256": digest,
        "reason_code": str(reason_code or "CLIO_ARTIFACT_REJECT"),
        "detail": str(detail or "")[:1000],
        "quarantined_at": utc_now().isoformat(),
    }
    reason_path = destination.with_suffix(destination.suffix + ".reason.json")
    fd, temporary = tempfile.mkstemp(prefix=f".{reason_path.name}.", dir=str(destination_dir))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True, indent=2, ensure_ascii=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, reason_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


__all__ = [
    "ARTIFACT_TYPE",
    "ClioArtifactError",
    "DEFAULT_TTL_SECONDS",
    "SCHEMA_VERSION",
    "artifact_filename",
    "atomic_write_artifact",
    "build_trade_idea_artifact",
    "canonical_digest",
    "new_artifact_id",
    "normalize_direction",
    "quarantine_artifact_file",
    "validate_clio_artifact",
]
