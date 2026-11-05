/* Captured synthetic results; no model or market call. */
const CLIO_REPLAY = {
  "kind": "RECORDED_SYNTHETIC_REPLAY_NOT_LIVE",
  "source_capture_sha256": "3661341d7f2a30952c0ff4c741dcd2f78d32f9531028c63c384385e6391d9655",
  "source_commit": "46da00609029298c2bca6d7d0f723b103f50272f",
  "cases": [
    {
      "id": "conflicting-flow",
      "input_label": "SYNTHETIC_NOT_MARKET_DATA",
      "inputs": {
        "direction": "bearish",
        "nofa": -0.6,
        "premium": 0.2,
        "cache_age_seconds": 10,
        "retained_history_complete": true,
        "prior_thesis_exit": true,
        "price_below_vwap": true,
        "momentum_falling": true
      },
      "result": {
        "disposition": "WOULD_DEFER",
        "reason_codes": [
          "prior_broken",
          "flow_conflict"
        ],
        "history_coverage": "complete",
        "history_scope": "retained_positions",
        "nofa": "aligned",
        "premium": "opposed",
        "price_confirmed": true,
        "provider_freshness": "unknown"
      }
    },
    {
      "id": "reconfirmed",
      "input_label": "SYNTHETIC_NOT_MARKET_DATA",
      "inputs": {
        "direction": "bearish",
        "nofa": -0.6,
        "premium": -0.2,
        "cache_age_seconds": 10,
        "retained_history_complete": true,
        "prior_thesis_exit": true,
        "price_below_vwap": true,
        "momentum_falling": true
      },
      "result": {
        "disposition": "RECONFIRMED",
        "reason_codes": [
          "prior_broken",
          "reconfirmed"
        ],
        "history_coverage": "complete",
        "history_scope": "retained_positions",
        "nofa": "aligned",
        "premium": "aligned",
        "price_confirmed": true,
        "provider_freshness": "unknown"
      }
    },
    {
      "id": "missing-time",
      "input_label": "SYNTHETIC_NOT_MARKET_DATA",
      "inputs": {
        "direction": "bearish",
        "nofa": -0.6,
        "premium": -0.2,
        "cache_age_seconds": null,
        "retained_history_complete": true,
        "prior_thesis_exit": true,
        "price_below_vwap": true,
        "momentum_falling": true
      },
      "result": {
        "disposition": "INSUFFICIENT",
        "reason_codes": [
          "prior_broken",
          "source_time_unknown"
        ],
        "history_coverage": "complete",
        "history_scope": "retained_positions",
        "nofa": "aligned",
        "premium": "aligned",
        "price_confirmed": true,
        "provider_freshness": "unknown"
      }
    },
    {
      "id": "no-prior-thesis",
      "input_label": "SYNTHETIC_NOT_MARKET_DATA",
      "inputs": {
        "direction": "bearish",
        "nofa": -0.6,
        "premium": -0.2,
        "cache_age_seconds": 10,
        "retained_history_complete": true,
        "prior_thesis_exit": false,
        "price_below_vwap": true,
        "momentum_falling": true
      },
      "result": {
        "disposition": "NO_REENTRY_CONTEXT",
        "reason_codes": [
          "no_prior_close"
        ],
        "history_coverage": "complete",
        "history_scope": "retained_positions",
        "nofa": "unknown",
        "premium": "unknown",
        "price_confirmed": null,
        "provider_freshness": "unknown"
      }
    }
  ]
};
