"""
CLIO Code Sentinel — Automated Code Health Monitoring
======================================================
Scheduled pipeline that scans the openclaw codebase for anti-patterns,
syntax errors, and uncommitted changes. Reports findings via Telegram
and persists them as knowledge entries.

Runs every 4 hours (configurable). On-demand via /codehealth command.

Feature flags:
  CLIO_CODE_SENTINEL_ENABLED       — Master gate (default "1")
  CLIO_CODE_SENTINEL_LOG_ONLY      — Log-only mode (default "1")

Author: Atlas Strategies — Operation FORGE Phase 1
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger("openclaw.code_sentinel")

_NY = ZoneInfo("America/New_York")

# ============================================================
# Feature Flags
# ============================================================
# S190 L10: SENTINEL_ENABLED() and SENTINEL_LOG_ONLY() are FUNCTIONS, not module
# constants. Always call with parens. Calling without parens returns the function
# object (truthy) instead of the runtime flag — silent always-on bug.

def SENTINEL_ENABLED():
    """Runtime check — not cached at import time."""
    return os.environ.get("CLIO_CODE_SENTINEL_ENABLED", "1") == "1"


def SENTINEL_LOG_ONLY():
    """Runtime check — not cached at import time."""
    return os.environ.get("CLIO_CODE_SENTINEL_LOG_ONLY", "1") == "1"

# ============================================================
# Configuration
# ============================================================

OPENCLAW_ROOT = "/home/agent/openclaw"
AGENT_DIR = os.path.join(OPENCLAW_ROOT, "agent")
SCAN_EXTENSIONS = {".py"}
MAX_FILE_SIZE = 500_000  # Skip files > 500KB for regex scanning

# Self-exclude: code_sentinel.py contains its own rule definitions as string
# literals (descriptions, regex patterns). Scanning itself produces ~8 false
# positives where rules match their own definitions. Exclude from pattern scans.
_SELF_EXCLUDE = {"agent/integrations/code_sentinel.py"}

# Severity levels
SEV_CRITICAL = "CRITICAL"
SEV_HIGH = "HIGH"
SEV_MEDIUM = "MEDIUM"
SEV_LOW = "LOW"

SEV_ORDER = {SEV_LOW: 0, SEV_MEDIUM: 1, SEV_HIGH: 2, SEV_CRITICAL: 3}


@dataclass
class Finding:
    """A single code audit finding."""
    file: str
    line: int
    severity: str
    pattern_id: str
    description: str
    rule_source: str
    suggestion: str
    snippet: str = ""


# ============================================================
# Anti-Pattern Registry
# ============================================================

# Each pattern: (compiled_regex, severity, pattern_id, description, rule_source, suggestion)
# Regexes are applied line-by-line for performance.

_ANTI_PATTERNS: list[tuple] = []


def _build_patterns() -> list[tuple]:
    """Build anti-pattern registry from CLAUDE.md rules.

    Called once at module load. Patterns are intentionally specific
    to minimize false positives (lesson from S136: ~40% FP rate from
    broad external auditors).
    """
    patterns = []

    # 1. shell=True in subprocess (CRITICAL — security risk)
    patterns.append((
        re.compile(r'\bsubprocess\.\w+\(.*shell\s*=\s*True', re.IGNORECASE),
        SEV_CRITICAL,
        "SHELL_TRUE",
        "subprocess with shell=True",
        "CLAUDE.md Never Do",
        "Use shell=False + shlex.split(). Never shell=True.",
    ))

    # 2. parse_mode="Markdown" with dynamic content (HIGH)
    patterns.append((
        re.compile(r'parse_mode\s*=\s*["\']Markdown["\']'),
        SEV_HIGH,
        "PARSE_MODE_MARKDOWN",
        "parse_mode='Markdown' risks BadRequest with dynamic content",
        "CLAUDE.md Never Do",
        "Use plain text only via _safe_send(). No parse_mode.",
    ))

    # 3. datetime.utcnow() — deprecated (MEDIUM)
    patterns.append((
        re.compile(r'\bdatetime\.utcnow\s*\('),
        SEV_MEDIUM,
        "UTCNOW_DEPRECATED",
        "datetime.utcnow() is deprecated",
        "CLAUDE.md Safety Subsystem",
        "Use datetime.now(timezone.utc) instead.",
    ))

    # 4. Bare except pass (HIGH)
    # Matches: except Exception: pass / except: pass (but NOT except Exception as e: logger.debug)
    patterns.append((
        re.compile(r'except\s*(Exception\s*)?:\s*(?:pass\s*)?(?:#.*)?$'),
        SEV_HIGH,
        "BARE_EXCEPT",
        "Bare except without logging",
        "CLAUDE.md Patterns to Follow",
        "Use try/except Exception as e: logger.debug(...) for fire-and-forget.",
    ))

    # 5. .lstrip() used for prefix removal (HIGH)
    # Specifically: .lstrip("some_string") where string has 2+ chars (likely prefix intent)
    patterns.append((
        re.compile(r'\.lstrip\(\s*["\'][a-zA-Z_]{2,}["\']\s*\)'),
        SEV_HIGH,
        "LSTRIP_AS_PREFIX",
        "str.lstrip() strips character SET, not prefix string",
        "CLAUDE.md Common Gotchas",
        "Use .removeprefix() for prefix removal.",
    ))

    # 6. subprocess.run() in async context — handled at file level, skipped here

    # 7. Lock file opened with "w" mode (MEDIUM)
    patterns.append((
        re.compile(r'open\s*\([^)]*\.lock["\'][^)]*["\']w["\']'),
        SEV_MEDIUM,
        "LOCK_FILE_WRITE_MODE",
        "Lock file opened with 'w' mode (truncates)",
        "CLAUDE.md Patterns to Follow",
        "Use mode 'a' for lock files — 'w' truncates.",
    ))

    # 8. value or default for numeric (LOW — many are intentional)
    patterns.append((
        re.compile(r'\b(\w+)\s+or\s+(0(?:\.0)?)\b'),
        SEV_LOW,
        "FALSY_NUMERIC_DEFAULT",
        "Possible falsy default for numeric (0/0.0 treated as falsy)",
        "CLAUDE.md Common Gotchas",
        "If 0 is a valid value, use 'if value is None' check instead.",
    ))

    # 9. sanitize_ingested_content without source= (HIGH)
    patterns.append((
        re.compile(r'sanitize_ingested_content\s*\([^)]*\)'),
        SEV_HIGH,
        "SANITIZE_NO_SOURCE",
        "sanitize_ingested_content() may be missing source= param",
        "CLAUDE.md Integration Gotchas",
        "Always pass source='label' as keyword argument.",
    ))

    # 10. "token" in error_msg — overly broad auth error match (MEDIUM)
    patterns.append((
        re.compile(r'["\']token["\']\s+in\s+\w*err'),
        SEV_MEDIUM,
        "BROAD_TOKEN_CHECK",
        "Overly broad 'token' in error check matches auth errors",
        "CLAUDE.md Common Gotchas",
        "Use specific: 'context window', 'context length', 'too many tokens'.",
    ))

    # 11. MoltbookKillSwitch.is_active() with parens (HIGH — is_active is property)
    patterns.append((
        re.compile(r'is_active\s*\(\s*\)'),
        SEV_HIGH,
        "PROPERTY_CALLED_AS_METHOD",
        "is_active is a property, not a method",
        "CLAUDE.md Moltbook Subsystem",
        "Use .is_active (no parentheses).",
    ))

    # 12. .reason on GuardrailResult (should be .blocking_reason) (MEDIUM)
    patterns.append((
        re.compile(r'\bguardrail\w*\.\breason\b(?!_)'),
        SEV_MEDIUM,
        "WRONG_ATTR_GUARDRAIL",
        "GuardrailResult uses .blocking_reason, not .reason",
        "CLAUDE.md Content Filter",
        "Use .blocking_reason attribute.",
    ))

    # 13. fromisoformat on date-only without tz normalization (LOW)
    patterns.append((
        re.compile(r'fromisoformat\s*\('),
        SEV_LOW,
        "FROMISOFORMAT_NAIVE",
        "fromisoformat() on date-only strings returns naive datetime",
        "CLAUDE.md Common Gotchas",
        "Always normalize: if dt.tzinfo is None: dt = dt.replace(tzinfo=_NY)",
    ))

    return patterns


_ANTI_PATTERNS = _build_patterns()


# ============================================================
# Scanner
# ============================================================

def _run_command(cmd: list[str], cwd: str = OPENCLAW_ROOT,
                 timeout: int = 30) -> tuple[int, str, str]:
    """Run a command safely. Returns (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=cwd, timeout=timeout, shell=False,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def _get_uncommitted_files() -> list[str]:
    """Get list of uncommitted/modified .py files in openclaw/."""
    # Unstaged changes
    rc, stdout, _ = _run_command(["git", "diff", "--name-only", "HEAD"])
    files = []
    if rc == 0:
        for line in stdout.strip().split("\n"):
            line = line.strip()
            if line and line.endswith(".py"):
                files.append(line)

    # Staged but not committed
    rc_s, stdout_s, _ = _run_command(["git", "diff", "--name-only", "--cached", "HEAD"])
    if rc_s == 0:
        for line in stdout_s.strip().split("\n"):
            line = line.strip()
            if line and line.endswith(".py"):
                files.append(line)

    # Also check untracked files
    rc2, stdout2, _ = _run_command(["git", "ls-files", "--others", "--exclude-standard"])
    if rc2 == 0:
        for line in stdout2.strip().split("\n"):
            line = line.strip()
            if line and line.endswith(".py"):
                files.append(line)

    return sorted(set(files))


def _get_all_agent_py_files() -> list[str]:
    """Get all .py files under agent/ for full scans."""
    files = []
    for root, _dirs, fnames in os.walk(AGENT_DIR):
        for fn in fnames:
            if fn.endswith(".py") and "__pycache__" not in root:
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, OPENCLAW_ROOT)
                files.append(rel)
    return sorted(files)


def _py_compile_check(filepath: str) -> Optional[Finding]:
    """Run py_compile on a file. Returns Finding if syntax error, else None."""
    full_path = os.path.join(OPENCLAW_ROOT, filepath)
    if not os.path.isfile(full_path):
        return None

    rc, _, stderr = _run_command(
        ["python3", "-m", "py_compile", full_path],
        timeout=15,
    )
    if rc != 0:
        # Extract line number from error if possible
        line_num = 0
        line_match = re.search(r'line (\d+)', stderr)
        if line_match:
            line_num = int(line_match.group(1))

        return Finding(
            file=filepath,
            line=line_num,
            severity=SEV_CRITICAL,
            pattern_id="SYNTAX_ERROR",
            description=f"py_compile failed: {stderr.strip()[:200]}",
            rule_source="CLAUDE.md Post-Edit Verification Tier 1",
            suggestion="Fix syntax error before deploying.",
            snippet=stderr.strip()[:300],
        )
    return None


def _scan_line_patterns(filepath: str) -> list[Finding]:
    """Scan a file for anti-pattern matches line by line."""
    if filepath in _SELF_EXCLUDE:
        return []
    full_path = os.path.join(OPENCLAW_ROOT, filepath)
    if not os.path.isfile(full_path):
        return []

    try:
        size = os.path.getsize(full_path)
        if size > MAX_FILE_SIZE:
            logger.debug(f"Skipping {filepath} — too large ({size} bytes)")
            return []

        with open(full_path, "r", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        logger.debug(f"Cannot read {filepath}: {e}")
        return []

    findings = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        # Skip comments and empty lines
        if not stripped or stripped.startswith("#"):
            continue

        for regex, severity, pat_id, desc, rule, suggestion in _ANTI_PATTERNS:
            if regex.search(stripped):
                # Special handling for SANITIZE_NO_SOURCE: only flag if source= is missing
                if pat_id == "SANITIZE_NO_SOURCE":
                    if "source=" in stripped:
                        continue
                    # source is 2nd positional param — comma means 2+ args
                    idx = stripped.find("sanitize_ingested_content(")
                    if idx >= 0 and "," in stripped[idx:]:
                        continue
                    # Skip empty-paren references (docstrings, comments)
                    if "sanitize_ingested_content()" in stripped:
                        continue

                # Look-ahead: FROMISOFORMAT_NAIVE — skip if tzinfo normalization
                # appears within the next 5 lines, or if .date() strips tz
                if pat_id == "FROMISOFORMAT_NAIVE":
                    # .date() strips timezone — tz irrelevant for date comparison
                    if ".date()" in stripped:
                        continue
                    # Check next 5 lines for tzinfo normalization
                    context_ahead = " ".join(
                        lines[i:i + 5] if i + 5 <= len(lines) else lines[i:]
                    )
                    if "tzinfo" in context_ahead or "replace(tz" in context_ahead:
                        continue

                # Look-ahead: BARE_EXCEPT — skip if next line has logging
                # (the code already logs, just without `as e`)
                if pat_id == "BARE_EXCEPT":
                    next_line = lines[i].strip() if i < len(lines) else ""
                    if "logger." in next_line or "logging." in next_line:
                        continue
                    # Fallback pattern: next line is return statement
                    if (next_line == "return" or
                            next_line.startswith(("return ", "return("))):
                        continue
                    # Lock cleanup: flock/LOCK_UN in preceding 3 lines
                    context_before = " ".join(
                        lines[max(0, i - 4):i - 1]
                    )
                    if "flock" in context_before or "LOCK_UN" in context_before:
                        continue

                # FALSY_NUMERIC_DEFAULT — skip lines that are string literals
                # (prompt text like "0.40 or 0.70" is not code)
                if pat_id == "FALSY_NUMERIC_DEFAULT":
                    if stripped.startswith(('"', "'", "- ", "{")):
                        continue
                    # Skip intentional None-to-0 coercions
                    if "getattr(" in stripped or ".get(" in stripped or "float(" in stripped:
                        continue

                findings.append(Finding(
                    file=filepath,
                    line=i,
                    severity=severity,
                    pattern_id=pat_id,
                    description=desc,
                    rule_source=rule,
                    suggestion=suggestion,
                    snippet=stripped[:150],
                ))

    return findings


def _scan_file_level(filepath: str) -> list[Finding]:
    """File-level checks that need whole-file context."""
    if filepath in _SELF_EXCLUDE:
        return []
    full_path = os.path.join(OPENCLAW_ROOT, filepath)
    if not os.path.isfile(full_path):
        return []

    findings = []
    try:
        with open(full_path, "r", errors="replace") as f:
            content = f.read()
    except Exception as e:
        logger.debug(f"Cannot read {filepath} for file-level check: {e}")
        return []

    # Check: subprocess.run() in async-aware file
    if "import asyncio" in content and "subprocess.run(" in content:
        all_lines = content.split("\n")
        for i, line in enumerate(all_lines, 1):
            if "subprocess.run(" in line and not line.strip().startswith("#"):
                # Walk backwards to find enclosing function def.
                # Only flag if the function is async — sync functions
                # in async modules are safe when called via run_in_executor.
                enclosing_is_async = False
                for j in range(i - 2, -1, -1):
                    fline = all_lines[j].lstrip()
                    if fline.startswith("async def "):
                        enclosing_is_async = True
                        break
                    if fline.startswith("def "):
                        break
                if not enclosing_is_async:
                    continue
                findings.append(Finding(
                    file=filepath,
                    line=i,
                    severity=SEV_HIGH,
                    pattern_id="SYNC_SUBPROCESS_IN_ASYNC",
                    description="subprocess.run() in async module blocks event loop",
                    rule_source="CLAUDE.md Always Do",
                    suggestion="Use asyncio.create_subprocess_exec() or run_in_executor() with timeout=30.",
                    snippet=line.strip()[:150],
                ))

    return findings


def run_full_scan(changed_only: bool = True) -> dict:
    """Run a complete code health scan.

    Args:
        changed_only: If True, only scan uncommitted changes. If False, scan all agent/ files.

    Returns:
        dict with keys: files_scanned, syntax_results, findings, summary, timestamp
    """
    if not SENTINEL_ENABLED():
        return {"enabled": False}

    now = datetime.now(_NY)
    logger.info("Code Sentinel: starting %s scan", "changed-only" if changed_only else "full")

    # Determine files to scan
    if changed_only:
        files = _get_uncommitted_files()
        if not files:
            logger.info("Code Sentinel: no uncommitted changes, running full agent/ scan")
            files = _get_all_agent_py_files()
    else:
        files = _get_all_agent_py_files()

    # Phase 1: Syntax checks
    syntax_results = {}
    syntax_failures = []
    for fp in files:
        finding = _py_compile_check(fp)
        if finding:
            syntax_results[fp] = "FAIL"
            syntax_failures.append(finding)
        else:
            syntax_results[fp] = "PASS"

    # Phase 2: Anti-pattern scan
    pattern_findings = []
    for fp in files:
        pattern_findings.extend(_scan_line_patterns(fp))
        pattern_findings.extend(_scan_file_level(fp))

    # Combine all findings
    all_findings = syntax_failures + pattern_findings

    # Sort by severity (highest first)
    all_findings.sort(key=lambda f: SEV_ORDER.get(f.severity, 0), reverse=True)

    # Summary counts
    summary = {
        "critical": sum(1 for f in all_findings if f.severity == SEV_CRITICAL),
        "high": sum(1 for f in all_findings if f.severity == SEV_HIGH),
        "medium": sum(1 for f in all_findings if f.severity == SEV_MEDIUM),
        "low": sum(1 for f in all_findings if f.severity == SEV_LOW),
    }

    result = {
        "timestamp": now.isoformat(),
        "files_scanned": len(files),
        "syntax_pass": sum(1 for v in syntax_results.values() if v == "PASS"),
        "syntax_fail": sum(1 for v in syntax_results.values() if v == "FAIL"),
        "findings": [asdict(f) for f in all_findings],
        "finding_count": len(all_findings),
        "summary": summary,
        "scan_type": "changed_only" if changed_only else "full",
    }

    logger.info(
        "Code Sentinel: scanned %d files, %d findings (%dC/%dH/%dM/%dL)",
        len(files), len(all_findings),
        summary["critical"], summary["high"], summary["medium"], summary["low"],
    )

    return result


def format_report(scan_result: dict) -> str:
    """Format scan results as a human-readable report."""
    if not scan_result.get("enabled", True):
        return "Code Sentinel is disabled."

    lines = [
        "CLIO Code Health Report",
        "=" * 30,
        f"Scan: {scan_result.get('timestamp', '?')}",
        f"Type: {scan_result.get('scan_type', '?')}",
        f"Files scanned: {scan_result.get('files_scanned', 0)}",
        f"Syntax: {scan_result.get('syntax_pass', 0)}/{scan_result.get('files_scanned', 0)} PASS",
    ]

    findings = scan_result.get("findings", [])
    if findings:
        lines.append("")
        lines.append(f"Findings ({len(findings)}):")
        for f in findings:
            lines.append(f"  [{f['severity']}] {f['file']}:{f['line']}")
            lines.append(f"    {f['description']}")
            lines.append(f"    Fix: {f['suggestion']}")
            lines.append("")
    else:
        lines.append("")
        lines.append("No findings. Codebase is clean.")

    summary = scan_result.get("summary", {})
    lines.append(
        f"Overall: {summary.get('critical', 0)}C / {summary.get('high', 0)}H / "
        f"{summary.get('medium', 0)}M / {summary.get('low', 0)}L"
    )

    return "\n".join(lines)


def _prune_old_scan_files(scan_dir: str, keep: int = 10) -> int:
    """Remove old scan files, keeping only the latest `keep` per scan type.

    S168: Prevents storage bloat from identical scans (only line numbers drift).
    Scans are named: YYYYMMDD_HHMMSS_code_sentinel_scan_*.json (or code_architecture_scan_*).
    The state file (code_sentinel_state.json) is never pruned.
    """
    removed = 0
    try:
        all_files = os.listdir(scan_dir)
    except OSError:
        return 0

    # Group scan files by type prefix
    scan_groups: dict[str, list[str]] = {}
    for fname in all_files:
        if not fname.endswith(".json"):
            continue
        # Skip the state file
        if fname == "code_sentinel_state.json":
            continue
        # Extract scan type: "code_sentinel_scan" or "code_architecture_scan"
        for scan_type in ("code_sentinel_scan", "code_architecture_scan"):
            if scan_type in fname:
                scan_groups.setdefault(scan_type, []).append(fname)
                break

    for scan_type, files in scan_groups.items():
        # Filenames start with YYYYMMDD_HHMMSS — lexicographic sort = chronological
        files.sort()
        to_remove = files[:-keep] if len(files) > keep else []
        for fname in to_remove:
            try:
                os.remove(os.path.join(scan_dir, fname))
                removed += 1
            except OSError as e:
                logger.debug(f"Code Sentinel prune: cannot remove {fname}: {e}")

    if removed:
        logger.info(f"Code Sentinel: pruned {removed} old scan files (kept latest {keep} per type)")
    return removed


async def _save_findings_to_knowledge(scan_result: dict) -> None:
    """Persist findings as knowledge entries for retrieval.

    VALIDATED: Uses handle_save_knowledge(p: dict) — the ONLY save function
    in knowledge_tools.py. NOT save_knowledge_entry (which does not exist).
    """
    findings = scan_result.get("findings", [])
    if not findings:
        return

    # Diff mode: only save if findings changed since last scan
    _sentinel_state_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "knowledge", "system_lessons", "code_sentinel_state.json",
    )
    findings_fingerprint = sorted(set(
        (f.get("pattern_id", ""), f.get("file", "")) for f in findings
    ))
    findings_hash = hashlib.md5(
        json.dumps(findings_fingerprint).encode()
    ).hexdigest()
    try:
        with open(_sentinel_state_path, "r") as _sf:
            _prev = json.load(_sf)
        if _prev.get("findings_hash") == findings_hash:
            # S189 B10: DEBUG → INFO so operator can see liveness in production.
            # Per ADR-002 (log-level discipline) DEBUG is filtered at INFO root level.
            # Without this, Code Sentinel appears dormant (24d+ silent in S189 audit).
            logger.info(
                f"Code Sentinel: findings unchanged (hash {findings_hash[:8]}, "
                f"{_prev.get('finding_count', '?')} findings), skipping save"
            )
            # S168: Still prune old files even when skipping save
            _prune_old_scan_files(os.path.dirname(_sentinel_state_path))
            return
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass

    try:
        from agent.tools.knowledge_tools import handle_save_knowledge

        now_str = datetime.now(_NY).strftime("%Y%m%d_%H%M")
        title = f"Code Sentinel Scan {now_str}"
        content_lines = [f"# {title}", ""]
        for f in findings[:30]:
            content_lines.append(f"## [{f['severity']}] {f['pattern_id']}")
            content_lines.append(f"**File:** {f['file']}:{f['line']}")
            content_lines.append(f"**Issue:** {f['description']}")
            content_lines.append(f"**Fix:** {f['suggestion']}")
            if f.get("snippet"):
                content_lines.append(f"**Code:** `{f['snippet'][:100]}`")
            content_lines.append("")

        summary = scan_result.get("summary", {})
        content_lines.append(
            f"**Summary:** {summary.get('critical', 0)}C / {summary.get('high', 0)}H / "
            f"{summary.get('medium', 0)}M / {summary.get('low', 0)}L"
        )

        # handle_save_knowledge is async and takes a dict parameter
        await handle_save_knowledge({
            "title": title,
            "category": "system_lessons",
            "content": "\n".join(content_lines),
            "tags": ["code_sentinel", "code_audit", "forge"],
            "source": "code_sentinel",
        })
        logger.info("Code Sentinel: findings saved to knowledge base")
        # Save findings hash for diff-mode
        try:
            _tmp_s = _sentinel_state_path + ".tmp"
            with open(_tmp_s, "w") as _sf2:
                json.dump({
                    "findings_hash": findings_hash,
                    "last_saved": datetime.now(_NY).isoformat(),
                    "finding_count": len(findings),
                }, _sf2)
            os.replace(_tmp_s, _sentinel_state_path)
        except OSError as _se:
            logger.debug(f"Code Sentinel state save failed: {_se}")
        # S168: Prune old scan files after successful save
        _prune_old_scan_files(os.path.dirname(_sentinel_state_path))
    except Exception as e:
        logger.debug(f"Code Sentinel: failed to save findings: {e}")


# ============================================================
# CC Execution Recording (for Phase 2 integration)
# ============================================================

async def record_cc_execution(cc_id: str, files_changed: list,
                               findings: list, outcome: str) -> None:
    """Record a CC prompt execution as a knowledge entry.

    Called by Claude Code post-execution (manual trigger via /recordcc command).
    """
    try:
        from agent.tools.knowledge_tools import handle_save_knowledge

        now_str = datetime.now(_NY).strftime("%Y%m%d_%H%M")
        content = (
            f"# CC Execution: {cc_id}\n"
            f"**Time:** {now_str}\n"
            f"**Files changed:** {', '.join(files_changed[:20])}\n"
            f"**Outcome:** {outcome}\n"
        )
        if findings:
            content += "\n## Post-Edit Findings\n"
            for f in findings[:10]:
                content += f"- [{f.get('severity', '?')}] {f.get('file', '?')}:{f.get('line', 0)} — {f.get('description', '')}\n"

        await handle_save_knowledge({
            "title": f"CC Execution {cc_id} {now_str}",
            "category": "system_lessons",
            "content": content,
            "tags": ["cc_execution", "forge", cc_id],
            "source": "code_sentinel",
        })
        logger.info(f"Code Sentinel: CC execution {cc_id} recorded")
    except Exception as e:
        logger.debug(f"Code Sentinel: failed to record CC execution: {e}")


# ============================================================
# Scheduled Task Entry Point
# ============================================================

async def run_code_health_check(context) -> None:
    """Scheduled task entry point for code health monitoring.

    Runs every 4 hours. Scans uncommitted changes first; if none,
    does a full agent/ scan. Alerts via Telegram for HIGH+ findings.
    """
    # Runtime flag check (not module-level constant) for dynamic enable/disable
    if os.environ.get("CLIO_CODE_SENTINEL_ENABLED", "1") != "1":
        return

    try:
        # Run scan in executor to avoid blocking event loop
        scan_result = await asyncio.get_running_loop().run_in_executor(
            None, run_full_scan, True,
        )

        if not scan_result.get("findings"):
            logger.info("Code Sentinel: clean scan, no alerts needed")
            return

        summary = scan_result.get("summary", {})
        has_alertable = summary.get("critical", 0) > 0 or summary.get("high", 0) > 0

        # Save to knowledge base (always, even in LOG_ONLY)
        await _save_findings_to_knowledge(scan_result)

        if os.environ.get("CLIO_CODE_SENTINEL_LOG_ONLY", "1") == "1":
            report = format_report(scan_result)
            logger.info(f"Code Sentinel [LOG_ONLY]: Would send report:\n{report[:500]}")
            return

        # Only alert on HIGH+ findings
        if has_alertable:
            from agent.config import config
            chat_ids = [cid for cid in config.telegram_notify_chat_ids]
            if chat_ids and hasattr(context, "bot"):
                report = format_report(scan_result)
                # Send in chunks to avoid Telegram 4096-char limit
                for cid in chat_ids:
                    remaining = report
                    while remaining:
                        if len(remaining) <= 4000:
                            try:
                                await context.bot.send_message(
                                    chat_id=cid, text=remaining,
                                )
                            except Exception as e:
                                logger.debug(
                                    f"Code Sentinel: send failed for {cid}: {e}"
                                )
                            break
                        # Split at newline near 4000-char boundary
                        split_at = remaining.rfind("\n", 0, 4000)
                        if split_at == -1:
                            split_at = 4000
                        try:
                            await context.bot.send_message(
                                chat_id=cid, text=remaining[:split_at],
                            )
                        except Exception as e:
                            logger.debug(
                                f"Code Sentinel: send failed for {cid}: {e}"
                            )
                            break
                        remaining = remaining[split_at:].lstrip("\n")

    except Exception as e:
        logger.error(f"Code Sentinel scan failed: {e}", exc_info=True)


# ============================================================
# On-Demand Entry Point (for /codehealth command)
# ============================================================

async def handle_codehealth_command(scan_type: str = "changed") -> str:
    """On-demand code health check. Returns formatted report string.

    Args:
        scan_type: "changed" for uncommitted only, "full" for all agent/ files.
    """
    if os.environ.get("CLIO_CODE_SENTINEL_ENABLED", "1") != "1":
        return "Code Sentinel is disabled (CLIO_CODE_SENTINEL_ENABLED=0)."

    full = scan_type.lower() == "full"
    scan_result = await asyncio.get_running_loop().run_in_executor(
        None, run_full_scan, not full,
    )

    report = format_report(scan_result)

    # Save findings regardless of LOG_ONLY (on-demand = operator wants it)
    if scan_result.get("findings"):
        await _save_findings_to_knowledge(scan_result)

    return report
