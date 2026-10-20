"""
ANVIL Error Taxonomy — Atlas Node for Validated Intelligence Loops
==================================================================
Canonical error classification for the error-to-rule pipeline.
Shared between Hydrinia and Moltbot (copy to Moltbot in ANVIL-6).

Every operational error is classified into one of 17 error classes
across 3 categories (CC execution, runtime, trading outcome). Each
class has a default severity, a category, and a description template
used by the Lesson Generator.

Feature flags:
  MOLTBOT_ERROR_PIPELINE          — Master gate (default "1")
  MOLTBOT_ERROR_PIPELINE_LOG_ONLY — Log-only mode (default "1")

Author: Atlas Strategies — Harlan Shepherd (Risk/Legal)
"""

import hashlib
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger("openclaw.anvil.taxonomy")

# ============================================================
# Feature Flags
# ============================================================

def ANVIL_ENABLED():
    """Runtime check — not cached at import time."""
    return os.environ.get("MOLTBOT_ERROR_PIPELINE", "1") == "1"


def ANVIL_LOG_ONLY():
    """Runtime check — not cached at import time."""
    return os.environ.get("MOLTBOT_ERROR_PIPELINE_LOG_ONLY", "1") == "1"

# ============================================================
# Severity Levels
# ============================================================

SEVERITY_CRITICAL = "CRITICAL"  # System down, data corruption, position risk
SEVERITY_HIGH = "HIGH"          # Feature broken, silent failure, wrong output
SEVERITY_MEDIUM = "MEDIUM"      # Degraded functionality, workaround available
SEVERITY_LOW = "LOW"            # Cosmetic, transient, self-healing

SEVERITY_ORDER = {
    SEVERITY_LOW: 0,
    SEVERITY_MEDIUM: 1,
    SEVERITY_HIGH: 2,
    SEVERITY_CRITICAL: 3,
}

# ============================================================
# Error Categories
# ============================================================

CATEGORY_CC = "cc_execution"
CATEGORY_RUNTIME = "runtime"
CATEGORY_TRADING = "trading"

# ============================================================
# Error Class Registry
# ============================================================

ERROR_CLASSES: Dict[str, Dict] = {
    # --- CC Execution Errors ---
    "ANCHOR_MISS": {
        "severity_default": SEVERITY_MEDIUM,
        "category": CATEGORY_CC,
        "description": "grep/find for expected code anchor returned 0 results",
        "template": "CC prompt {cc_id} expected anchor `{anchor}` in {file} but grep returned 0 matches. "
                     "The function/line may have been renamed, moved, or deleted by a prior CC.",
    },
    "SIGNATURE_BREAK": {
        "severity_default": SEVERITY_HIGH,
        "category": CATEGORY_CC,
        "description": "Function signature change broke callers",
        "template": "CC prompt {cc_id} changed the signature of `{function}` in {file} but {caller_count} "
                     "call sites were not updated. py_compile caught: {error_msg}",
    },
    "IMPORT_FAIL": {
        "severity_default": SEVERITY_HIGH,
        "category": CATEGORY_CC,
        "description": "Module import failed after edit",
        "template": "CC prompt {cc_id} introduced an import for `{module}` in {file} that fails: {error_msg}. "
                     "Likely cause: dependency not yet deployed or circular import.",
    },
    "FORMAT_BREAK": {
        "severity_default": SEVERITY_CRITICAL,
        "category": CATEGORY_CC,
        "description": "Cross-system format contract violated",
        "template": "CC prompt {cc_id} modified {file} which is read by {reader_system}. The output format "
                     "changed: {details}. See cross_system.md for format contracts.",
    },
    "LOGIC_ERROR": {
        "severity_default": SEVERITY_HIGH,
        "category": CATEGORY_CC,
        "description": "Tests pass but runtime behavior wrong",
        "template": "CC prompt {cc_id} introduced a logic error in {file}:{function}: {details}. "
                     "py_compile passed but functional behavior is incorrect.",
    },
    "MERGE_CONFLICT": {
        "severity_default": SEVERITY_MEDIUM,
        "category": CATEGORY_CC,
        "description": "Two CC prompts modified overlapping code regions",
        "template": "CC prompts {cc_id} and {cc_id_2} both modify {file} around line {line}. "
                     "Sequential execution required. Apply {cc_id} first, then re-anchor {cc_id_2}.",
    },
    # --- Runtime Errors ---
    "API_FAILURE": {
        "severity_default": SEVERITY_LOW,
        "category": CATEGORY_RUNTIME,
        "description": "External API returned error or timed out",
        "template": "API call to {api_name} failed: HTTP {status_code} — {error_msg}. "
                     "Endpoint: {endpoint}. This has occurred {occurrences} times in the last 24h.",
    },
    "DATA_STALE": {
        "severity_default": SEVERITY_MEDIUM,
        "category": CATEGORY_RUNTIME,
        "description": "Data staleness guard triggered",
        "template": "Staleness guard on {file} triggered: data is {age_minutes:.0f} min old "
                     "(threshold: {threshold_minutes} min). Source: {source}.",
    },
    "BRIDGE_CORRUPT": {
        "severity_default": SEVERITY_CRITICAL,
        "category": CATEGORY_RUNTIME,
        "description": "Cross-system bridge file corrupted or unparseable",
        "template": "Bridge file {file} failed to parse: {error_msg}. Reader: {reader}. "
                     "Possible causes: partial write, encoding issue, schema change.",
    },
    "SERVICE_CRASH": {
        "severity_default": SEVERITY_CRITICAL,
        "category": CATEGORY_RUNTIME,
        "description": "systemd service crashed and restarted",
        "template": "Service {service_name} crashed: {error_msg}. Restart count: {restart_count} "
                     "in last 24h. Exit code: {exit_code}.",
    },
    "FEATURE_FLAG_LEAK": {
        "severity_default": SEVERITY_HIGH,
        "category": CATEGORY_RUNTIME,
        "description": "LOG_ONLY feature produced side effects",
        "template": "Feature {flag_name} is set to LOG_ONLY but produced side effect: {details}. "
                     "This is a safety infrastructure violation — LOG_ONLY must be inert.",
    },
    "RESOURCE_EXHAUST": {
        "severity_default": SEVERITY_HIGH,
        "category": CATEGORY_RUNTIME,
        "description": "System resource limit exceeded",
        "template": "Resource exhaustion: {resource_type} at {current_value} "
                     "(limit: {limit_value}). {details}",
    },
    # --- Trading Outcome Errors ---
    "STOP_HIT": {
        "severity_default": SEVERITY_MEDIUM,
        "category": CATEGORY_TRADING,
        "description": "Position closed at stop-loss",
        "template": "Stop-loss hit on {ticker}: entered at ${entry_price:.2f}, stopped at "
                     "${stop_price:.2f} ({loss_pct:.1f}% loss). Thesis: {thesis_summary}.",
    },
    "THESIS_INVALIDATED": {
        "severity_default": SEVERITY_HIGH,
        "category": CATEGORY_TRADING,
        "description": "Fundamental thesis premise broke",
        "template": "Thesis for {ticker} invalidated: {invalidation_reason}. "
                     "Original thesis: {thesis_summary}. Position impact: {impact}.",
    },
    "CONVICTION_MISCAL": {
        "severity_default": SEVERITY_MEDIUM,
        "category": CATEGORY_TRADING,
        "description": "High-conviction trade lost or low-conviction won",
        "template": "Conviction miscalibration on {ticker}: conviction was {conviction}/100 but "
                     "outcome was {outcome_pct:+.1f}%. Expected direction: {direction}.",
    },
    "TIMING_ERROR": {
        "severity_default": SEVERITY_LOW,
        "category": CATEGORY_TRADING,
        "description": "Direction correct but entry/exit timing wrong",
        "template": "Timing error on {ticker}: direction was correct ({direction}) but entry at "
                     "${entry_price:.2f} was suboptimal. Price moved from ${low:.2f} to ${high:.2f}. "
                     "Captured {captured_pct:.0f}% of the move.",
    },
    "SIZING_ERROR": {
        "severity_default": SEVERITY_MEDIUM,
        "category": CATEGORY_TRADING,
        "description": "Position size inappropriate for risk level",
        "template": "Sizing error on {ticker}: allocated {allocation_pct:.1f}% of portfolio "
                     "({sizing_label}). Risk-adjusted appropriate size: {recommended_pct:.1f}%. "
                     "Reason: {reason}.",
    },
}

# ============================================================
# Promotion Criteria
# ============================================================

PROMOTION_CRITERIA = {
    "min_severity": SEVERITY_HIGH,
    "min_occurrences": 2,
    "must_be_generalizable": True,
}

# ============================================================
# Error Record
# ============================================================


def generate_error_id() -> str:
    """Generate a unique error ID: ERR-YYYYMMDD-HHMMSS-XXXX."""
    now = datetime.now(timezone.utc)
    suffix = hashlib.md5(now.isoformat().encode() + os.urandom(8)).hexdigest()[:4]
    return f"ERR-{now.strftime('%Y%m%d')}-{now.strftime('%H%M%S')}-{suffix}"


def create_error_record(
    error_class: str,
    system: str = "moltbot",
    severity_override: Optional[str] = None,
    affected_files: Optional[List[str]] = None,
    related_cc: Optional[List[str]] = None,
    context: Optional[Dict] = None,
) -> Dict:
    """Create a canonical error record.

    Args:
        error_class: One of the keys in ERROR_CLASSES.
        system: 'hydrinia' | 'moltbot' | 'cross_system'.
        severity_override: Override the default severity for this class.
        affected_files: List of files involved.
        related_cc: List of CC prompt IDs involved.
        context: Template variables for the description template.

    Returns:
        Dict with all fields populated.
    """
    if error_class not in ERROR_CLASSES:
        logger.warning(f"Unknown error class: {error_class}. Using LOGIC_ERROR.")
        error_class = "LOGIC_ERROR"

    cls_info = ERROR_CLASSES[error_class]
    severity = severity_override if severity_override is not None else cls_info["severity_default"]
    now = datetime.now(timezone.utc).isoformat()

    # Render description from template
    description = cls_info["description"]
    if context:
        try:
            description = cls_info["template"].format(**context)
        except (KeyError, ValueError, IndexError) as e:
            logger.warning(f"Template render failed for {error_class}: {e}")
            description = cls_info["description"]

    return {
        "id": generate_error_id(),
        "class": error_class,
        "category": cls_info["category"],
        "severity": severity,
        "system": system,
        "first_seen": now,
        "last_seen": now,
        "occurrences": 1,
        "affected_files": affected_files or [],
        "related_cc": related_cc or [],
        "description": description,
        "context": context or {},
        "lesson_id": None,
        "promoted_to_claude_md": False,
        "resolved": False,
    }


def severity_gte(a: str, b: str) -> bool:
    """Return True if severity a >= severity b."""
    return SEVERITY_ORDER.get(a, 0) >= SEVERITY_ORDER.get(b, 0)


def meets_promotion_criteria(record: Dict) -> bool:
    """Check if an error record meets the promotion criteria for CLAUDE.md."""
    if not severity_gte(record.get("severity", SEVERITY_LOW), PROMOTION_CRITERIA["min_severity"]):
        return False
    if record.get("occurrences", 1) < PROMOTION_CRITERIA["min_occurrences"]:
        return False
    return True
