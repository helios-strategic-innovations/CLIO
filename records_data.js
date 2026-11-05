/* Sanitized selected historical records; no live endpoint. */
const CLIO_RECORDED = {
  "kind": "HISTORICAL_CATEGORICAL_PAPER_EVIDENCE_NOT_LIVE",
  "source_sha256": "1061ef7cb997af7c53a1b6a87f5d764336940a90fc78f9210a16251ff1ca6450",
  "recorded_cutoff": "2026-11-04T12:15:44.005-05:00",
  "running_source_at_recording": "fef1d6516ad93d6879cb3748257ab74a1e521424",
  "review_sha256": "86bd42f28553f3ee6f5a3c16815e05d1918db6cf16b43dba742eb988b1306761",
  "records": [
    {
      "anonymous_case": "Recorded PAPER case A",
      "action": "ENTER",
      "kind": "entry",
      "recorded_model": "muse-wonder-1.0",
      "recorded_provider": "meta",
      "matching_attempts": 1,
      "expected_attempts": 1,
      "attempts": [
        {
          "status": "ok",
          "provider": "meta",
          "final_provider": "meta",
          "requested_model": "muse-wonder-1.0",
          "served_model": "muse-wonder-1.0",
          "fallback_used": false
        }
      ],
      "jev_observed": {
        "called": true,
        "mode": "veto",
        "allow": true
      },
      "recorded_action_enum": "enter",
      "recorded_kind_enum": "entry_deliberation"
    },
    {
      "anonymous_case": "Recorded PAPER case A",
      "action": "TRIM",
      "kind": "review",
      "recorded_model": "muse-spark-1.3",
      "recorded_provider": "meta",
      "matching_attempts": 1,
      "expected_attempts": 1,
      "attempts": [
        {
          "status": "ok",
          "provider": "meta",
          "final_provider": "meta",
          "requested_model": "muse-spark-1.3",
          "served_model": "muse-spark-1.3",
          "fallback_used": false
        }
      ],
      "jev_observed": {
        "called": null,
        "mode": "UNKNOWN",
        "allow": null
      },
      "recorded_action_enum": "trim",
      "recorded_kind_enum": "position_review"
    },
    {
      "anonymous_case": "Recorded PAPER case A",
      "action": "EXIT",
      "kind": "review",
      "recorded_model": "muse-spark-1.3",
      "recorded_provider": "meta",
      "matching_attempts": 1,
      "expected_attempts": 1,
      "attempts": [
        {
          "status": "ok",
          "provider": "meta",
          "final_provider": "meta",
          "requested_model": "muse-spark-1.3",
          "served_model": "muse-spark-1.3",
          "fallback_used": false
        }
      ],
      "jev_observed": {
        "called": null,
        "mode": "UNKNOWN",
        "allow": null
      },
      "recorded_action_enum": "exit",
      "recorded_kind_enum": "position_review"
    },
    {
      "anonymous_case": "Recorded PAPER case B",
      "action": "ENTER",
      "kind": "entry",
      "recorded_model": "muse-wonder-1.0",
      "recorded_provider": "meta",
      "matching_attempts": 1,
      "expected_attempts": 1,
      "attempts": [
        {
          "status": "ok",
          "provider": "meta",
          "final_provider": "meta",
          "requested_model": "muse-wonder-1.0",
          "served_model": "muse-wonder-1.0",
          "fallback_used": false
        }
      ],
      "jev_observed": {
        "called": true,
        "mode": "veto",
        "allow": true
      },
      "recorded_action_enum": "enter",
      "recorded_kind_enum": "entry_deliberation"
    },
    {
      "anonymous_case": "Recorded PAPER case B",
      "action": "TRIM",
      "kind": "review",
      "recorded_model": "muse-spark-1.3",
      "recorded_provider": "meta",
      "matching_attempts": 1,
      "expected_attempts": 1,
      "attempts": [
        {
          "status": "ok",
          "provider": "meta",
          "final_provider": "meta",
          "requested_model": "muse-spark-1.3",
          "served_model": "muse-spark-1.3",
          "fallback_used": false
        }
      ],
      "jev_observed": {
        "called": null,
        "mode": "UNKNOWN",
        "allow": null
      },
      "recorded_action_enum": "trim",
      "recorded_kind_enum": "position_review"
    },
    {
      "anonymous_case": "Recorded PAPER case B",
      "action": "EXIT",
      "kind": "review",
      "recorded_model": "muse-spark-1.3",
      "recorded_provider": "meta",
      "matching_attempts": 1,
      "expected_attempts": 1,
      "attempts": [
        {
          "status": "ok",
          "provider": "meta",
          "final_provider": "meta",
          "requested_model": "muse-spark-1.3",
          "served_model": "muse-spark-1.3",
          "fallback_used": false
        }
      ],
      "jev_observed": {
        "called": null,
        "mode": "UNKNOWN",
        "allow": null
      },
      "recorded_action_enum": "exit",
      "recorded_kind_enum": "position_review"
    }
  ],
  "limitations": [
    "Selected historical PAPER records only; bounded/truncated underlying retention",
    "No financial values, account/order/trade/decision IDs, token counts, raw model output or proprietary prompts copied",
    "Recorded events predate17:46ET reentry shadow installation; not live current Meta call proof",
    "No performance/model superiority/profitability/cost completeness claim",
    "Actual trace joins were independently reviewed; this is a redacted categorical summary, not the original raw trace"
  ]
};
