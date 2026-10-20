"""
OpenClaw/Moltbot — File Operation Tool Handlers

Granular file tools: read, write, edit, list, search, git, run_command.
All paths are sandboxed to allowed directories.

Hydrinia is permanently off (operator direction, 2026-09-24); /home/agent/tradingbot/
is only reachable — as a default base, an allowed root, or a git/run_command cwd —
when the operator explicitly re-enables the bridge (agent/hydrinia_bridge.py).
Otherwise everything defaults to /home/agent/openclaw/, CLIO's own repo.
"""

import asyncio
import fnmatch
import logging
import os
import re
import shlex
from pathlib import Path

from agent.security import log_security_event, sanitize_ingested_content, _is_blocked_secret_file

logger = logging.getLogger("openclaw.file_tools")

DEFAULT_BASE = "/home/agent/openclaw"
_RETIRED_BASE = "/home/agent/tradingbot"


def _bridge_enabled() -> bool:
    try:
        from agent.hydrinia_bridge import bridge_enabled
        return bridge_enabled()
    except Exception as e:
        logger.debug(f"Hydrinia bridge check failed, defaulting to disabled: {e}")
        return False


def _allowed_roots() -> list[str]:
    """ALLOWED_ROOTS, minus the retired Hydrinia root unless the bridge is
    explicitly re-enabled. Computed per-call (not a module constant) so a
    runtime bridge toggle takes effect without a restart."""
    roots = [
        "/home/agent/openclaw/",
        "/home/agent/backups/",
        "/tmp/openclaw/",
    ]
    if _bridge_enabled():
        roots.append(f"{_RETIRED_BASE}/")
    return roots

MAX_FILE_SIZE = 1_000_000  # 1MB read limit
COMMAND_TIMEOUT = 60  # seconds

ALLOWED_COMMANDS = [
    r"^python3?\s+-m\s+(pytest|py_compile)(\s+.*)?$",
    r"^git\s+(log|status|diff|show|branch)(\s+.*)?$",
    r"^git\s+add(\s+.*)?$",
    r"^git\s+commit(\s+.*)?$",
    r"^ls(\s+.*)?$",
    r"^wc(\s+.*)?$",
    r"^pip3?\s+(list|show)(\s+.*)?$",
]


def _resolve_path(file_path: str) -> tuple[bool, str, str]:
    """Resolve and validate a file path against allowed roots.

    Returns (ok, resolved_path, error_message).

    For relative paths, tries DEFAULT_BASE (/home/agent/openclaw) first. The
    retired /home/agent/tradingbot/ is only tried as a fallback when the
    Hydrinia bridge is explicitly re-enabled — otherwise a relative path
    resolves under openclaw (and, if it doesn't exist there, fails downstream
    as "not found" rather than silently reaching into the frozen repo).
    """
    if not file_path:
        return False, "", "Empty file path"

    if os.path.isabs(file_path):
        resolved = os.path.realpath(file_path)
    else:
        # S118 W-059: Prefer openclaw root for agent/ paths
        _OPENCLAW_PREFIXES = ("agent/", "data/", "logs/", "scripts/", "mission_control/")
        candidate = os.path.join(DEFAULT_BASE, file_path)
        if any(file_path.startswith(pfx) for pfx in _OPENCLAW_PREFIXES) or os.path.exists(candidate):
            resolved = os.path.realpath(candidate)
        elif _bridge_enabled():
            alt = os.path.join(_RETIRED_BASE, file_path)
            resolved = os.path.realpath(alt) if os.path.exists(alt) else os.path.realpath(candidate)
        else:
            resolved = os.path.realpath(candidate)

    # Append os.sep so "/home/agent/openclaw" matches "/home/agent/openclaw/"
    # but "/home/agent/openclaw-evil" does not
    resolved_check = resolved if resolved.endswith(os.sep) else resolved + os.sep
    if not any(resolved_check.startswith(root) for root in _allowed_roots()):
        log_security_event("path_traversal_file_tools", path=file_path, resolved=resolved)
        return False, "", f"Path outside allowed directories: {file_path}"

    # Secret-file deny-list (.env / keys / certs) — inherited by read/write/edit/list for
    # free since they all resolve through here. handle_search_files needs its own skip (S-1).
    if _is_blocked_secret_file(resolved):
        log_security_event("blocked_secret_file", path=file_path, resolved=resolved)
        return False, "", f"Blocked: secret file not permitted via file tools: {os.path.basename(resolved)}"

    return True, resolved, ""


async def handle_read_file(p: dict) -> str:
    ok, path, err = _resolve_path(p.get("file_path", ""))
    if not ok:
        return f"ERROR: {err}"

    if not os.path.isfile(path):
        return f"File not found: {p.get('file_path')}"

    try:
        size = os.path.getsize(path)
        if size > MAX_FILE_SIZE:
            return f"ERROR: File too large ({size:,} bytes, limit {MAX_FILE_SIZE:,}). Use start_line/end_line."

        start = p.get("start_line")
        end = p.get("end_line")
        exact_requested = start is not None or end is not None
        if not exact_requested:
            try:
                from agent.prompts.headroom_context import maybe_headroom_file_redirect
                redirect = maybe_headroom_file_redirect(
                    path,
                    display_path=p.get("file_path", ""),
                    exact_requested=False,
                    system_label="CLIO file tool",
                )
                if redirect:
                    logger.info(
                        "HEADROOM_FILE_REDIRECT file=%s bytes=%d mode=whole_file exact_bypass=line_reads",
                        p.get("file_path", ""),
                        size,
                    )
                    return sanitize_ingested_content(redirect, f"file-headroom:{p.get('file_path')}")
            except Exception as e:
                logger.debug(f"Headroom file redirect skipped for {p.get('file_path')}: {e}")

        with open(path, "r", errors="replace") as f:
            lines = f.readlines()

        if start is not None or end is not None:
            start_idx = max(0, (start or 1) - 1)
            end_idx = end or len(lines)
            selected = lines[start_idx:end_idx]
            numbered = [f"{start_idx + i + 1:>6}| {line}" for i, line in enumerate(selected)]
            header = f"File: {p.get('file_path')} (lines {start_idx + 1}-{min(end_idx, len(lines))} of {len(lines)})"
        else:
            numbered = [f"{i + 1:>6}| {line}" for i, line in enumerate(lines)]
            header = f"File: {p.get('file_path')} ({len(lines)} lines)"

        return sanitize_ingested_content(f"{header}\n{''.join(numbered)}", f"file:{p.get('file_path')}")
    except Exception as e:
        return f"ERROR reading file: {e}"


async def handle_write_file(p: dict) -> str:
    ok, path, err = _resolve_path(p.get("file_path", ""))
    if not ok:
        return f"ERROR: {err}"

    content = p.get("content", "")
    commit_msg = p.get("commit_message")

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            f.write(content)
        os.replace(tmp_path, path)

        log_security_event("file_written", path=path, size=len(content))
        result = f"Written {len(content)} chars to {p.get('file_path')}"

        if commit_msg:
            git_result = await _git_commit(path, commit_msg)
            result += f"\n{git_result}"

        return result
    except Exception as e:
        return f"ERROR writing file: {e}"


async def handle_edit_file(p: dict) -> str:
    ok, path, err = _resolve_path(p.get("file_path", ""))
    if not ok:
        return f"ERROR: {err}"

    if not os.path.isfile(path):
        return f"File not found: {p.get('file_path')}"

    # S233 HI-12: Size check matching handle_read_file pattern (prevents OOM on huge files)
    size = os.path.getsize(path)
    if size > MAX_FILE_SIZE:
        return f"ERROR: File too large to edit ({size:,} bytes, limit {MAX_FILE_SIZE:,}). Edit manually."

    old_string = p.get("old_string", "")
    new_string = p.get("new_string", "")
    commit_msg = p.get("commit_message")

    if not old_string:
        return "ERROR: old_string is required"
    if old_string == new_string:
        return "ERROR: old_string and new_string are identical"

    try:
        with open(path, "r") as f:
            content = f.read()

        count = content.count(old_string)
        if count == 0:
            return f"ERROR: old_string not found in {p.get('file_path')}"
        if count > 1:
            return f"ERROR: old_string found {count} times — must be unique. Add more context."

        new_content = content.replace(old_string, new_string, 1)

        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            f.write(new_content)
        os.replace(tmp_path, path)

        log_security_event("file_edited", path=path, old_len=len(old_string), new_len=len(new_string))
        result = f"Edited {p.get('file_path')}: replaced {len(old_string)} chars with {len(new_string)} chars"

        if commit_msg:
            git_result = await _git_commit(path, commit_msg)
            result += f"\n{git_result}"

        return result
    except Exception as e:
        return f"ERROR editing file: {e}"


async def handle_list_files(p: dict) -> str:
    dir_path = p.get("path", DEFAULT_BASE)
    ok, resolved, err = _resolve_path(dir_path)
    if not ok:
        return f"ERROR: {err}"

    pattern = p.get("pattern", "*")

    try:
        if not os.path.isdir(resolved):
            return f"Not a directory: {dir_path}"

        matches = []
        for root, dirs, files in os.walk(resolved):
            # Skip hidden dirs and venv
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "venv" and d != "__pycache__"]
            for fname in files:
                if fnmatch.fnmatch(fname, pattern):
                    rel = os.path.relpath(os.path.join(root, fname), resolved)
                    matches.append(rel)
                    if len(matches) >= 200:
                        break
            if len(matches) >= 200:
                break

        matches.sort()
        if not matches:
            return f"No files matching '{pattern}' in {dir_path}"
        return sanitize_ingested_content(f"Files in {dir_path} (pattern: {pattern}):\n" + "\n".join(matches), f"dir:{dir_path}")
    except Exception as e:
        return f"ERROR listing files: {e}"


async def handle_search_files(p: dict) -> str:
    search_pattern = p.get("pattern", "")
    if not search_pattern:
        return "ERROR: pattern is required"

    dir_path = p.get("path", DEFAULT_BASE)
    ok, resolved, err = _resolve_path(dir_path)
    if not ok:
        return f"ERROR: {err}"

    file_glob = p.get("file_glob", "*.py")
    max_results = min(p.get("max_results", 50), 200)

    try:
        regex = re.compile(search_pattern, re.IGNORECASE)
    except re.error as e:
        return f"ERROR: Invalid regex: {e}"

    results = []
    try:
        for root, dirs, files in os.walk(resolved):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "venv" and d != "__pycache__"]
            for fname in files:
                if not fnmatch.fnmatch(fname, file_glob):
                    continue
                fpath = os.path.join(root, fname)
                # S-1: content-search opens files directly (bypassing _resolve_path per file),
                # so re-check here — never open/return secret-file contents via a walk.
                if _is_blocked_secret_file(fpath):
                    continue
                try:
                    with open(fpath, "r", errors="replace") as f:
                        for line_num, line in enumerate(f, 1):
                            if regex.search(line):
                                rel = os.path.relpath(fpath, resolved)
                                results.append(f"{rel}:{line_num}: {line.rstrip()[:200]}")
                                if len(results) >= max_results:
                                    break
                except (OSError, UnicodeDecodeError):
                    continue
                if len(results) >= max_results:
                    break
            if len(results) >= max_results:
                break
    except Exception as e:
        return f"ERROR searching: {e}"

    if not results:
        return f"No matches for '{search_pattern}' in {dir_path}"
    header = f"Search: '{search_pattern}' in {dir_path} ({len(results)} results"
    if len(results) >= max_results:
        header += f", truncated at {max_results}"
    header += "):\n"
    return sanitize_ingested_content(header + "\n".join(results), f"search:{dir_path}")


async def handle_git_status(p: dict) -> str:
    try:
        proc_status = await asyncio.create_subprocess_exec(
            "git", "status", "--short", cwd=DEFAULT_BASE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        status_out, _ = await asyncio.wait_for(proc_status.communicate(), timeout=10)

        proc_log = await asyncio.create_subprocess_exec(
            "git", "log", "--oneline", "-10", cwd=DEFAULT_BASE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        log_out, _ = await asyncio.wait_for(proc_log.communicate(), timeout=10)

        status_text = status_out.decode(errors="replace") if status_out else "(clean)"
        log_text = log_out.decode(errors="replace") if log_out else "No history."
        return sanitize_ingested_content(f"Git Status:\n{status_text}\n\nRecent Commits:\n{log_text}", "git:status")
    except asyncio.TimeoutError:
        return "ERROR: Git command timed out after 10s"
    except Exception as e:
        return f"ERROR: {e}"


async def handle_git_diff(p: dict) -> str:
    try:
        proc_unstaged = await asyncio.create_subprocess_exec(
            "git", "diff", cwd=DEFAULT_BASE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        unstaged_out, _ = await asyncio.wait_for(proc_unstaged.communicate(), timeout=10)

        proc_staged = await asyncio.create_subprocess_exec(
            "git", "diff", "--cached", cwd=DEFAULT_BASE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        staged_out, _ = await asyncio.wait_for(proc_staged.communicate(), timeout=10)

        parts = []
        staged_text = staged_out.decode(errors="replace") if staged_out else ""
        unstaged_text = unstaged_out.decode(errors="replace") if unstaged_out else ""
        if staged_text:
            parts.append(f"Staged Changes:\n{staged_text[:5000]}")
        if unstaged_text:
            parts.append(f"Unstaged Changes:\n{unstaged_text[:5000]}")
        if not parts:
            return "No changes."
        return sanitize_ingested_content("\n\n".join(parts), "git:diff")
    except asyncio.TimeoutError:
        return "ERROR: Git diff timed out after 10s"
    except Exception as e:
        return f"ERROR: {e}"


async def handle_run_command(p: dict) -> str:
    command = p.get("command", "").strip()
    if not command:
        return "ERROR: command is required"

    cwd = p.get("cwd", DEFAULT_BASE)
    ok, resolved_cwd, err = _resolve_path(cwd)
    if not ok:
        return f"ERROR: {err}"

    # Check against allowlist (patterns use $ anchor to match full command)
    if not any(re.match(pat, command) for pat in ALLOWED_COMMANDS):
        log_security_event("blocked_command", command=command)
        return f"ERROR: Command not in allowlist. Allowed: pytest, python3 -m py_compile, git (log/status/diff/show/branch/add/commit), ls, wc, pip list/show"

    try:
        argv = shlex.split(command)
    except ValueError as e:
        return f"ERROR: Invalid command syntax: {e}"

    # S118 W-060: Validate git add file arguments against allowed roots
    # S148 H-9: Block mass-staging flags that bypass file-path validation
    _BLOCKED_GIT_ADD_FLAGS = {"-A", "--all", "-u", "--update", "--no-all"}
    _BLOCKED_GIT_COMMIT_FLAGS = {"--amend", "--allow-empty", "--reset-author", "--no-verify"}
    if len(argv) >= 3 and argv[0] == "git" and argv[1] == "add":
        for arg in argv[2:]:
            if arg in _BLOCKED_GIT_ADD_FLAGS:
                log_security_event("blocked_git_add_flag", flag=arg)
                return f"ERROR: git add flag '{arg}' not allowed (use specific file paths)"
            if arg == "--" or arg.startswith("-"):
                continue
            _ok, _resolved, _err = _resolve_path(arg)
            if not _ok:
                log_security_event("blocked_git_add_path", path=arg)
                return f"ERROR: git add path not in allowed directories: {arg}"
    if len(argv) >= 3 and argv[0] == "git" and argv[1] == "commit":
        for arg in argv[2:]:
            if arg in _BLOCKED_GIT_COMMIT_FLAGS:
                log_security_event("blocked_git_commit_flag", flag=arg)
                return f"ERROR: git commit flag '{arg}' is blocked for safety"

    try:
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=resolved_cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=COMMAND_TIMEOUT,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return f"ERROR: Command timed out after {COMMAND_TIMEOUT}s"

        output = stdout_bytes.decode(errors="replace") if stdout_bytes else ""
        if stderr_bytes:
            output += f"\nSTDERR:\n{stderr_bytes.decode(errors='replace')}"
        if proc.returncode != 0:
            output += f"\n(exit code {proc.returncode})"
        if len(output) > 10000:
            output = output[:10000] + "\n...(truncated)"
        return sanitize_ingested_content(output or "(no output)", f"cmd:{command[:50]}")
    except Exception as e:
        return f"ERROR: {e}"


async def _git_commit(file_path: str, message: str) -> str:
    """Stage and commit a single file."""
    try:
        # Determine repo root from file path
        if file_path.startswith("/home/agent/tradingbot"):
            cwd = "/home/agent/tradingbot"
        elif file_path.startswith("/home/agent/openclaw"):
            cwd = "/home/agent/openclaw"
        else:
            return "Skipped git commit (path not in a known repo)"

        rel_path = os.path.relpath(file_path, cwd)

        proc_add = await asyncio.create_subprocess_exec(
            "git", "add", "--", rel_path, cwd=cwd,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        _, stderr_add = await asyncio.wait_for(proc_add.communicate(), timeout=10)
        if proc_add.returncode != 0:
            return f"Git add failed: {stderr_add.decode(errors='replace') if stderr_add else 'unknown error'}"

        proc_commit = await asyncio.create_subprocess_exec(
            "git", "commit", "-m", message, cwd=cwd,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        _, stderr_commit = await asyncio.wait_for(proc_commit.communicate(), timeout=10)
        if proc_commit.returncode != 0:
            return f"Git commit failed: {stderr_commit.decode(errors='replace') if stderr_commit else 'unknown error'}"

        return f"Git: committed '{rel_path}' — {message}"
    except asyncio.TimeoutError:
        return "Git commit timed out after 10s"
    except Exception as e:
        return f"Git error: {e}"
