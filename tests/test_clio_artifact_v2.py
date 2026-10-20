from datetime import datetime, timedelta, timezone
import json

import pytest

from agent.integrations.clio_artifact_v2 import (
    ClioArtifactError,
    atomic_write_artifact,
    build_trade_idea_artifact,
    canonical_digest,
    quarantine_artifact_file,
    validate_clio_artifact,
)


def _idea(created):
    return {
        "id": "ti_example",
        "created": created.isoformat(),
        "ticker": "SPY",
        "direction": "long",
        "entry_trigger": "Enter between $100 and $102",
        "stop_loss": "Thesis invalid below $95",
        "take_profit": "First objective $120",
        "timeframe": "2-4 weeks",
        "position_size_guidance": "2% of NAV",
        "thesis": "Breadth and earnings revisions confirm the setup.",
        "confidence": 0.75,
        "risk_factors": ["Close below $95 within five sessions invalidates."],
        "sector": "Index",
        "source_category": "macro",
        "source_urls": ["https://example.test/source"],
    }


def test_round_trip_is_digest_bound_lane_agnostic_and_execution_free(tmp_path):
    now = datetime(2026, 7, 9, 15, 0, tzinfo=timezone.utc)
    artifact = build_trade_idea_artifact(_idea(now), produced_at=now)
    validated = validate_clio_artifact(artifact, now=now + timedelta(seconds=1))
    assert validated["payload"]["direction"] == "LONG"
    assert validated["content_digest"] == canonical_digest(validated["payload"])
    encoded = json.dumps(validated, sort_keys=True).lower()
    for forbidden in ("lane_id", "position_size_guidance", "execution_allowed", "order_class"):
        assert forbidden not in encoded
    target = tmp_path / "artifact.json"
    atomic_write_artifact(target, artifact)
    assert json.loads(target.read_text(encoding="utf-8"))["artifact_id"] == artifact["artifact_id"]


def test_tamper_missing_redactions_unknown_schema_and_staleness_reject():
    now = datetime(2026, 7, 9, 15, 0, tzinfo=timezone.utc)
    artifact = build_trade_idea_artifact(_idea(now), produced_at=now, ttl_seconds=60)

    tampered = json.loads(json.dumps(artifact))
    tampered["payload"]["thesis"] = "tampered"
    with pytest.raises(ClioArtifactError, match="CONTENT_DIGEST_MISMATCH"):
        validate_clio_artifact(tampered, now=now)

    missing_redactions = dict(artifact)
    missing_redactions.pop("redactions")
    with pytest.raises(ClioArtifactError, match="MISSING_ENVELOPE_FIELDS"):
        validate_clio_artifact(missing_redactions, now=now)

    unknown_schema = dict(artifact)
    unknown_schema["schema_version"] = "clio.artifact.v99"
    with pytest.raises(ClioArtifactError, match="UNSUPPORTED_SCHEMA"):
        validate_clio_artifact(unknown_schema, now=now)

    with pytest.raises(ClioArtifactError, match="STALE_ARTIFACT"):
        validate_clio_artifact(artifact, now=now + timedelta(seconds=61))


def test_unknown_direction_and_execution_fields_reject():
    now = datetime(2026, 7, 9, 15, 0, tzinfo=timezone.utc)
    idea = _idea(now)
    idea["direction"] = "sideways"
    with pytest.raises(ClioArtifactError, match="UNKNOWN_DIRECTION"):
        build_trade_idea_artifact(idea, produced_at=now)

    idea["direction"] = "hold"
    with pytest.raises(ClioArtifactError, match="NON_DIRECTIONAL"):
        build_trade_idea_artifact(idea, produced_at=now)

    artifact = build_trade_idea_artifact(_idea(now), produced_at=now)
    artifact["payload"]["position_size_guidance"] = "5%"
    artifact["content_digest"] = canonical_digest(artifact["payload"])
    with pytest.raises(ClioArtifactError, match="EXECUTION_FIELD_FORBIDDEN"):
        validate_clio_artifact(artifact, now=now)


def test_secret_scrubber_records_redaction():
    now = datetime(2026, 7, 9, 15, 0, tzinfo=timezone.utc)
    idea = _idea(now)
    idea["source_urls"] = ["Bearer " + ("fixture-token-" * 3)]
    artifact = build_trade_idea_artifact(idea, produced_at=now)
    assert artifact["redactions"] == ["payload.source_urls[0]"]
    assert artifact["payload"]["source_urls"] == ["[REDACTED]"]


def test_quarantine_preserves_original_bytes_and_reason(tmp_path):
    source = tmp_path / "clio2_bad.json"
    raw = b'{"bad":true}\n'
    source.write_bytes(raw)
    destination = quarantine_artifact_file(
        source,
        reason_code="UNSUPPORTED_SCHEMA",
        quarantine_dir=tmp_path / "quarantine",
    )
    assert not source.exists()
    assert destination.read_bytes() == raw
    reason = json.loads(destination.with_suffix(".json.reason.json").read_text(encoding="utf-8"))
    assert reason["reason_code"] == "UNSUPPORTED_SCHEMA"
