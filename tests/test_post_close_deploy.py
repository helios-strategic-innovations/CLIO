"""S211: scripts/post_close_deploy.py -- the guarded, unattended post-close backup/prune/deploy/restart job.

Every command goes through an injected fake runner; nothing real is executed, merged, deleted or restarted.
The properties that matter for an unattended job that restarts a trading service:
  * it aborts at the FIRST failed check, and never reaches an irreversible step (prune / merge / restart) after one;
  * rehearsal mode issues no destructive command at all;
  * the order is tests -> prune -> merge -> imports -> (flat re-check) -> restart -> verify;
  * a second run is a no-op (done marker / lock).
"""

import argparse
import importlib.util
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "post_close_deploy", Path(__file__).resolve().parents[1] / "scripts" / "post_close_deploy.py"
)
pcd = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pcd)

NY = ZoneInfo("America/New_York")
TARGET = "a" * 40
HEAD0 = "b" * 40
EVENING = datetime(2026, 9, 29, 16, 30, tzinfo=NY)


# ------------------------------------------------------------------------------------------ pure helpers
def test_in_window():
    assert pcd.in_window(datetime(2026, 9, 29, 16, 15, tzinfo=NY), "16:15", "22:00")
    assert pcd.in_window(datetime(2026, 9, 29, 22, 0, tzinfo=NY), "16:15", "22:00")
    assert not pcd.in_window(datetime(2026, 9, 29, 16, 14, tzinfo=NY), "16:15", "22:00")
    assert not pcd.in_window(datetime(2026, 9, 29, 22, 1, tzinfo=NY), "16:15", "22:00")


def test_dirty_source_paths_allows_only_runtime_state():
    porcelain = (" M data/memory/meta_memory.json\n M data/personality/trait_registry.json\n"
                 "?? logs/x.log\n?? scripts/new.py\n")
    assert pcd.dirty_source_paths(porcelain) == []
    assert pcd.dirty_source_paths(" M agent/core.py\n M data/memory/a.json\n") == ["agent/core.py"]
    assert pcd.dirty_source_paths("R  old.py -> agent/new.py\n") == ["agent/new.py"]
    assert pcd.dirty_source_paths("A  scripts/x.py\n") == ["scripts/x.py"]
    assert pcd.dirty_source_paths("") == []


def test_live_positions():
    data = {"positions": [{"status": "closed"}, {"status": "open"}, {"status": "pending"}, {"status": "closing"}, {}]}
    assert len(pcd.live_positions(data)) == 3
    assert pcd.live_positions({}) == []


def test_failed_test_ids():
    out = ("...F.\nFAILED tests/a.py::test_one - AssertionError: x\nFAILED tests/b.py::TestC::test_two[3] - boom\n"
           "FAILED tests/a.py::test_one - duplicate line\nERROR tests/c.py::test_three - fixture\n1 failed\n")
    assert pcd.failed_test_ids(out) == ["tests/a.py::test_one", "tests/b.py::TestC::test_two[3]"]
    assert pcd.failed_test_ids("3049 passed in 50s") == []


def test_changed_agent_modules():
    names = "agent/dpof/engine.py\nagent/integrations/uw_pacer.py\nscripts/x.py\ntests/t.py\nagent/data.json\ndocs/a.md\n"
    assert pcd.changed_agent_modules(names) == ["agent.dpof.engine", "agent.integrations.uw_pacer"]


# ------------------------------------------------------------------------------------------ harness
class Fake:
    """Records every command; per-test overrides decide return codes/outputs."""

    def __init__(self, serving: Path):
        self.serving = serving
        self.calls: list[list[str]] = []
        self.merged = False
        self.overrides = []          # list of (predicate(cmd), (rc, out))
        self.journal = ""
        self.active = "active"
        self.on_merge = None

    def __call__(self, cmd, timeout=120, cwd=None):
        cmd = list(cmd)
        self.calls.append(cmd)
        for pred, result in self.overrides:
            if pred(cmd):
                return result
        joined = " ".join(cmd)
        if cmd[:2] == ["git", "branch"]:
            return 0, "main\n"
        if cmd[:2] == ["git", "status"]:
            return 0, " M data/memory/meta_memory.json\n?? logs/foo\n"
        if cmd[:2] == ["git", "cat-file"]:
            return 0, ""
        if cmd[:3] == ["git", "rev-parse", "HEAD"]:
            return 0, (TARGET if self.merged else HEAD0) + "\n"
        if cmd[:2] == ["git", "merge-base"]:
            return 0, ""
        if cmd[:2] == ["git", "fetch"]:
            return 0, ""
        if cmd[:2] == ["git", "log"]:
            return 0, "1111111 first commit\n2222222 second commit\n"
        if cmd[:3] == ["git", "worktree", "add"]:
            return 0, ""
        if cmd[:3] == ["git", "worktree", "remove"]:
            return 0, ""
        if cmd[:2] == ["git", "merge"]:
            self.merged = True
            if self.on_merge:
                self.on_merge()
            return 0, "Fast-forward\n"
        if cmd[:2] == ["git", "diff"]:
            return 0, "agent/integrations/uw_quota_client.py\nagent/dpof/engine.py\nscripts/x.py\n"
        if "-c" in cmd and "agent.__file__" in joined:
            return 0, f"{cwd}/agent/__init__.py\n"
        if "-m" in cmd and "pytest" in cmd:
            return 0, "3001 passed, 16 warnings in 50.28s\n"
        if any(part.endswith("uw_ledger_prune.py") for part in cmd):
            return 0, "UW ledger prune\n  DRY RUN or deleted\n"
        if "-c" in cmd and cmd[cmd.index("-c") + 1].startswith("import "):
            return 0, ""
        if cmd[:4] == ["sudo", "-n", "/usr/bin/systemctl", "restart"]:
            return 0, ""
        if cmd[:4] == ["sudo", "-n", "/usr/bin/systemctl", "is-active"]:
            return 0, "active\n"
        if cmd[:2] == ["systemctl", "is-active"]:
            return 0, self.active + "\n"
        if cmd[0] == "journalctl":
            return 0, self.journal
        return 0, ""

    def has(self, *fragments):
        return any(all(f in " ".join(c) for f in fragments) for c in self.calls)

    def index(self, *fragments):
        for i, c in enumerate(self.calls):
            if all(f in " ".join(c) for f in fragments):
                return i
        return None


@pytest.fixture()
def env(tmp_path):
    serving = tmp_path / "serving"
    (serving / "logs").mkdir(parents=True)
    (serving / "data" / "dpof").mkdir(parents=True)
    (serving / "data" / "dpof" / "positions.json").write_text(json.dumps({"positions": [{"status": "closed"}]}))
    return serving, Fake(serving)


def _args(**over):
    base = dict(target_sha=TARGET, apply=False, not_before="16:15", not_after="22:00", wait_flat_until="19:00",
                prune_days=14, remote_branch="fix/clio-uw5-stale-probe", skip_tests=False)
    base.update(over)
    return argparse.Namespace(**base)


def _deployer(env, now=EVENING, **over):
    serving, fake = env
    clock = now if callable(now) else (lambda: now)
    return pcd.Deployer(_args(**over), run=fake, now=clock, sleep=lambda s: None, serving=serving), fake, serving


# ------------------------------------------------------------------------------------------ rehearsal
def test_rehearsal_issues_no_destructive_command(env):
    d, fake, serving = _deployer(env)
    assert d.execute() == 0
    assert not fake.has("merge", "--ff-only")
    assert not fake.has("restart")
    assert fake.has("uw_ledger_prune.py") and not fake.has("--apply")
    assert fake.has("worktree", "add") and fake.has("worktree", "remove")           # temp worktree made and removed
    assert fake.has("pytest")                                                        # tests still run in rehearsal
    assert not (serving / "logs" / "post_close_deploy.done").exists()


def test_rehearsal_does_not_wait_for_flat_or_abort_outside_the_window(env):
    serving, fake = env
    (serving / "data" / "dpof" / "positions.json").write_text(json.dumps({"positions": [{"status": "open"}]}))
    d, fake, _ = _deployer(env, now=datetime(2026, 9, 29, 11, 0, tzinfo=NY))
    assert d.execute() == 0


# ------------------------------------------------------------------------------------------ apply: happy path + order
def test_apply_happy_path_runs_every_step_in_order_and_writes_the_marker(env):
    d, fake, serving = _deployer(env, apply=True)
    assert d.execute() == 0
    order = [fake.index("pytest"), fake.index("uw_ledger_prune.py", "--apply", "--vacuum"),
             fake.index("git", "merge", "--ff-only", TARGET), fake.index("import agent.dpof.engine"),
             fake.index("restart", "moltbot"), fake.index("journalctl")]
    assert None not in order and order == sorted(order), order
    assert fake.has("import agent.integrations.uw_quota_client")                    # every changed module is import-checked
    marker = (serving / "logs" / "post_close_deploy.done").read_text()
    assert TARGET in marker and "2 commits" in marker


def test_second_run_is_a_noop_once_the_marker_exists(env):
    d, fake, serving = _deployer(env, apply=True)
    assert d.execute() == 0
    d2, fake2, _ = _deployer((serving, pcd_fake := Fake(serving)), apply=True)
    assert d2.execute() == 30
    assert pcd_fake.calls == []


def test_a_held_lock_makes_a_concurrent_run_exit(env):
    import fcntl

    serving, _ = env
    holder = open(serving / "logs" / "post_close_deploy.lock", "a")
    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        d, fake, _ = _deployer(env, apply=True)
        assert d.execute() == 30
        assert fake.calls == []
    finally:
        holder.close()


# ------------------------------------------------------------------------------------------ apply: every abort stops before the irreversible steps
def _never(fake, *fragments):
    assert not fake.has(*fragments), f"unexpected command containing {fragments}: {fake.calls}"


def test_outside_the_window_aborts_before_touching_anything(env):
    d, fake, _ = _deployer(env, apply=True, now=datetime(2026, 9, 29, 11, 0, tzinfo=NY))
    assert d.execute() == 10
    assert fake.calls == []


def test_not_flat_by_the_deadline_aborts_without_changes(env):
    serving, _ = env
    (serving / "data" / "dpof" / "positions.json").write_text(json.dumps({"positions": [{"status": "open"}]}))
    times = iter([EVENING] * 3 + [datetime(2026, 9, 29, 19, 1, tzinfo=NY)] * 50)
    d, fake, _ = _deployer(env, apply=True, now=lambda: next(times))
    assert d.execute() == 11
    assert fake.calls == []


def test_a_dirty_source_tree_aborts_before_tests_or_prune(env):
    d, fake, _ = _deployer(env, apply=True)
    fake.overrides.append((lambda c: c[:2] == ["git", "status"], (0, " M agent/core.py\n")))
    assert d.execute() == 12
    _never(fake, "pytest"); _never(fake, "uw_ledger_prune"); _never(fake, "merge", "--ff-only")


def test_wrong_branch_aborts(env):
    d, fake, _ = _deployer(env, apply=True)
    fake.overrides.append((lambda c: c[:2] == ["git", "branch"], (0, "feature-x\n")))
    assert d.execute() == 12
    _never(fake, "pytest")


def test_a_target_that_is_not_a_fast_forward_aborts(env):
    d, fake, _ = _deployer(env, apply=True)
    fake.overrides.append((lambda c: c[:2] == ["git", "merge-base"] and "HEAD" in c, (1, "")))
    assert d.execute() == 12
    _never(fake, "pytest"); _never(fake, "uw_ledger_prune")


def test_a_target_not_on_the_remote_branch_aborts(env):
    d, fake, _ = _deployer(env, apply=True)
    fake.overrides.append((lambda c: c[:2] == ["git", "merge-base"] and "origin/fix/clio-uw5-stale-probe" in c, (1, "")))
    assert d.execute() == 12
    _never(fake, "uw_ledger_prune")


def test_failing_tests_abort_before_prune_and_merge(env):
    d, fake, _ = _deployer(env, apply=True)
    fake.overrides.append((lambda c: "pytest" in c, (1, "2 failed, 2999 passed\n")))
    assert d.execute() == 13
    _never(fake, "uw_ledger_prune"); _never(fake, "merge", "--ff-only"); _never(fake, "restart")
    assert fake.has("worktree", "remove")                                            # the temp worktree is still cleaned up


def test_wrong_test_provenance_aborts(env):
    d, fake, _ = _deployer(env, apply=True)
    fake.overrides.append((lambda c: "agent.__file__" in " ".join(c), (0, "/home/agent/openclaw/agent/__init__.py\n")))
    assert d.execute() == 13
    _never(fake, "pytest")


def _pytest_full(cmd):
    return "pytest" in cmd and "tests" in cmd


def test_a_flaky_failure_that_passes_on_rerun_is_logged_and_the_deploy_continues(env, capsys):
    d, fake, serving = _deployer(env, apply=True)
    fake.overrides.append((_pytest_full, (1, "FAILED tests/t.py::test_timing - assert 0.12 < 0.1\n1 failed, 3048 passed\n")))
    fake.overrides.append((lambda c: "pytest" in c and "tests/t.py::test_timing" in c, (0, "1 passed in 0.2s\n")))
    assert d.execute() == 0
    out = capsys.readouterr().out
    assert "FLAKY: passed on rerun" in out and "tests/t.py::test_timing" in out
    assert fake.has("uw_ledger_prune.py", "--apply") and fake.has("restart", "moltbot")     # the deploy went ahead


def test_a_real_regression_fails_again_on_rerun_and_holds_the_deploy(env, capsys):
    d, fake, _ = _deployer(env, apply=True)
    fake.overrides.append((_pytest_full, (1, "FAILED tests/t.py::test_real - assert False\n1 failed, 3048 passed\n")))
    fake.overrides.append((lambda c: "pytest" in c and "tests/t.py::test_real" in c, (1, "1 failed in 0.2s\n")))
    assert d.execute() == 13
    out = capsys.readouterr().out
    assert "tests/t.py::test_real" in out and "again on rerun" in out
    _never(fake, "uw_ledger_prune"); _never(fake, "merge", "--ff-only"); _never(fake, "restart")


def test_many_failures_are_not_rerun_they_abort(env):
    d, fake, _ = _deployer(env, apply=True)
    many = "".join(f"FAILED tests/t.py::test_{i} - x\n" for i in range(6))
    fake.overrides.append((_pytest_full, (1, many + "6 failed\n")))
    assert d.execute() == 13
    assert not any("tests/t.py::test_0" in " ".join(c) for c in fake.calls)             # no rerun of a big failure set
    _never(fake, "uw_ledger_prune")


def test_a_failure_with_no_identifiable_test_aborts_without_rerun(env):
    d, fake, _ = _deployer(env, apply=True)
    fake.overrides.append((_pytest_full, (2, "INTERNALERROR> boom\n")))
    assert d.execute() == 13
    _never(fake, "uw_ledger_prune")


def test_a_failing_prune_aborts_before_the_merge(env):
    d, fake, _ = _deployer(env, apply=True)
    fake.overrides.append((lambda c: any(p.endswith("uw_ledger_prune.py") for p in c), (3, "ABORTED: lock contention\n")))
    assert d.execute() == 14
    _never(fake, "merge", "--ff-only"); _never(fake, "restart")


def test_a_failed_merge_aborts_before_the_restart(env):
    d, fake, _ = _deployer(env, apply=True)
    fake.overrides.append((lambda c: c[:2] == ["git", "merge"], (1, "not possible to fast-forward")))
    assert d.execute() == 15
    _never(fake, "restart")


def test_an_import_failure_after_the_merge_blocks_the_restart(env):
    d, fake, _ = _deployer(env, apply=True)
    fake.overrides.append((lambda c: "-c" in c and "import agent.dpof.engine" in " ".join(c), (1, "ImportError: boom")))
    assert d.execute() == 15
    _never(fake, "restart")


def test_head_not_at_target_after_merge_blocks_the_restart(env):
    d, fake, _ = _deployer(env, apply=True)
    fake.overrides.append((lambda c: c[:3] == ["git", "rev-parse", "HEAD"] and fake.merged, (0, "c" * 40 + "\n")))
    assert d.execute() == 15
    _never(fake, "restart")


def test_position_opening_between_merge_and_restart_blocks_the_restart(env):
    serving, _ = env
    d, fake, _ = _deployer(env, apply=True)

    def open_position():
        (serving / "data" / "dpof" / "positions.json").write_text(json.dumps({"positions": [{"status": "open"}]}))

    fake.on_merge = open_position
    times = iter([EVENING] * 12 + [datetime(2026, 9, 29, 19, 5, tzinfo=NY)] * 50)
    d.now = lambda: next(times)
    assert d.execute() == 11
    _never(fake, "restart")                                                          # merged but NOT restarted


def test_restart_command_failure(env):
    d, fake, _ = _deployer(env, apply=True)
    fake.overrides.append((lambda c: c[:4] == ["sudo", "-n", "/usr/bin/systemctl", "restart"], (1, "sudo: a password is required")))
    assert d.execute() == 16


def test_service_not_active_after_restart_is_flagged_without_auto_rollback(env, capsys):
    d, fake, serving = _deployer(env, apply=True)
    fake.active = "failed"
    assert d.execute() == 20
    out = capsys.readouterr().out
    assert "NO auto-rollback" in out and "1111111" in out                           # names the commits to revert
    assert not (serving / "logs" / "post_close_deploy.done").exists()
    _never(fake, "reset")                                                            # never a reset --hard


def test_a_traceback_after_restart_is_flagged(env):
    d, fake, _ = _deployer(env, apply=True)
    fake.journal = "Traceback (most recent call last):\n"
    assert d.execute() == 20


# ------------------------------------------------------------------------------------------ CLI validation
def test_main_rejects_a_short_or_non_hex_sha(capsys):
    assert pcd.main(["--target-sha", "abc123"]) == 2
    assert pcd.main(["--target-sha", "z" * 40]) == 2


def test_main_rejects_skip_tests_with_apply():
    assert pcd.main(["--target-sha", TARGET, "--apply", "--skip-tests"]) == 2
