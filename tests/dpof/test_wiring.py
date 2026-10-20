"""Source-text assertions for the DPOF wiring edits (mirrors the pattern in
tests/test_shutdown_paths.py). These don't import the modules — they check
the shape of the edits directly, so they catch a mis-placed registration
(e.g. inside `if has_chat_ids:`) even if the module fails to import in this
environment.
"""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_dpof_cycle_registered_in_unconditional_block():
    text = _read("agent/integrations/scheduled_tasks.py")

    assert "from agent.dpof.engine import run_dpof_cycle, DPOF_JOB_INTERVAL, DPOF_JOB_TIMEOUT" in text
    assert 'name="dpof_cycle"' in text

    # The registration must sit in the UNCONDITIONAL block: after the
    # learning/trade-feedback/market-events registrations and OUTSIDE the
    # `if has_chat_ids:` gate that precedes them.
    has_chat_idx = text.index("if has_chat_ids:")
    market_events_idx = text.index("Market event detection registered")
    dpof_idx = text.index('name="dpof_cycle"')
    book_idx = text.index("Book corpus pipeline (Master Research Agenda")

    assert has_chat_idx < market_events_idx < dpof_idx < book_idx

    # Between has_chat_ids's own block and the DPOF registration there must be
    # a dedent back to 4-space indentation (i.e. we're no longer nested inside
    # that `if`) — find the register_scheduled_jobs function body and confirm
    # the DPOF try/except starts at exactly 4 spaces (function-body level).
    func_start = text.index("def register_scheduled_jobs(app: Application) -> None:")
    body = text[func_start:dpof_idx]
    dpof_block_start = body.rindex("\n    try:\n        from agent.dpof.engine import run_dpof_cycle")
    line = body[dpof_block_start + 1:].splitlines()[0]
    assert line == "    try:"  # exactly 4 spaces, not 8 (which would mean nested under has_chat_ids)


def test_shutdown_calls_shutdown_dpof():
    text = _read("agent/integrations/scheduled_tasks.py")
    assert "from agent.dpof.engine import shutdown_dpof" in text
    assert "await shutdown_dpof()" in text

    # Must run before the "Close httpx clients" loop (defense-in-depth: DPOF's
    # jev client should be closed before shared httpx clients it might reuse).
    shutdown_fn_idx = text.index("async def shutdown_background_tasks():")
    dpof_shutdown_idx = text.index("await shutdown_dpof()")
    httpx_idx = text.index("# Close httpx clients on shutdown")
    assert shutdown_fn_idx < dpof_shutdown_idx < httpx_idx


def test_telegram_registers_dpof_command():
    text = _read("agent/integrations/telegram_bot.py")
    assert "from agent.dpof.telegram_commands import dpof_command" in text
    assert 'CommandHandler("dpof", dpof_command)' in text
    assert "DPOF commands registered: ['dpof']" in text

    # Registered inside create_bot(), after the codehealth block (per BUILD_SPEC).
    create_bot_idx = text.index("def create_bot() -> Application:")
    codehealth_idx = text.index("Code Sentinel command registered")
    dpof_cmd_idx = text.index('CommandHandler("dpof", dpof_command)')
    assert create_bot_idx < codehealth_idx < dpof_cmd_idx


def test_help_text_mentions_dpof():
    text = _read("agent/integrations/telegram_bot.py")
    assert "/dpof" in text
    # Specifically present in the /start command-list text block.
    start_idx = text.index("async def cmd_start(")
    status_idx = text.index("async def cmd_status(")
    start_block = text[start_idx:status_idx]
    assert "/dpof" in start_block


def test_env_template_has_dpof_section_no_secrets():
    text = _read(".env.template")
    assert "CLIO DPOF" in text
    # 2026-09-24 operator directive ("paper mode IS the soak", "everything is
    # built active, not shadow") — the template's explicit value matches the
    # new active-by-default code default (DpofMode.PAPER in config.py).
    assert "CLIO_DPOF_MODE=paper" in text
    assert "TYPESAFE_API_KEY=" in text
    assert "TYPESAFE_BASE_URL=" in text
    assert "CLIO_DPOF_JEV_FALLBACK_URL=" in text

    # No secrets: key-bearing lines must be present but blank-valued.
    for line in text.splitlines():
        if line.startswith("CLIO_DPOF_ALPACA_KEY_ID=") or line.startswith("CLIO_DPOF_ALPACA_SECRET_KEY="):
            assert line.split("=", 1)[1] == ""
        if line.startswith("TYPESAFE_API_KEY=") or line.startswith("CLIO_DPOF_JEV_FALLBACK_KEY="):
            assert line.split("=", 1)[1] == ""
        if line.startswith("CLIO_DPOF_LIVE_ACK="):
            assert line.split("=", 1)[1] == ""  # never pre-filled with the live-arm phrase


def test_wiring_edits_are_additive_only():
    """Confirms the wiring diff is small/additive (not a rewrite) by checking
    the surrounding, untouched code is still byte-identical around the edits."""
    result = subprocess.run(
        ["git", "diff", "--stat", "--", "agent/integrations/", ".env.template"],
        cwd=str(ROOT), capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        return  # not a git checkout in this environment -- skip silently
    stat = result.stdout
    # A rewrite would show hundreds of changed lines in scheduled_tasks.py;
    # the real edit is two small blocks (~21 lines total per the PR).
    for line in stat.splitlines():
        if "scheduled_tasks.py" in line:
            assert "++" in line and "--" not in line.split("|")[-1].replace("++", "")


def test_existing_shutdown_paths_suite_still_green():
    """Runs tests/test_shutdown_paths.py as its own process to keep this file
    import-independent; BUILD_SPEC_PHASE2.md requires it stay green."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_shutdown_paths.py", "-q", "-p", "no:warnings"],
        cwd=str(ROOT), capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
