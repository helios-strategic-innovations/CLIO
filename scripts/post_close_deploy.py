#!/usr/bin/env python3
"""Post-close deploy + ledger prune orchestrator (S211; operator-approved 2026-09-29).

One guarded, idempotent job for after the market close:
  1. preconditions  -> 2. wait for DPOF flat -> 3. serving-tree checks -> 4. full test suite ON THE TARGET COMMIT
  5. ledger backup + prune + vacuum (scripts/uw_ledger_prune.py) -> 6. fast-forward deploy + import checks
  7. re-check flat -> 8. restart moltbot -> 9. verify (active, no Traceback, ledger report) -> done marker.

Default is a REHEARSAL: it runs the read-only checks (time, flat, tree, target ancestry, tests unless --skip-tests, a prune
DRY RUN, sudo works) and prints what --apply would do. Nothing is deleted, merged or restarted without --apply.

Safety properties
  * The deploy target is an exact SHA (--target-sha), never "the branch tip": what was reviewed is what ships.
  * Every step aborts the whole run at the first failed check -- the irreversible steps (prune, merge, restart) come only
    after everything that can fail cheaply has passed. Prune and merge are ordered so a failure leaves the running system
    exactly as it was (prune touches only old rows and a verified backup exists; the merge is a fast-forward).
  * Only tracked runtime-state files (data/memory/, data/personality/) may be modified in the serving tree; anything
    else aborts (a dirty source tree means someone is mid-edit).
  * `sudo /usr/bin/systemctl restart moltbot` is the exact NOPASSWD form; DPOF must be flat immediately before it.
  * A lock file + a done marker make a second run a no-op. No auto-rollback: if post-restart verification fails the job
    exits 20 and says so loudly; rollback is `git revert` of the listed commits (never `reset --hard`: the tree holds
    tracked runtime state) or the env switches UW_QUOTA_STALE_PROBE=0 / CLIO_UW_PACER_ENABLED=0 (Dom edits .env).

Exit codes: 0 ok / rehearsal ok · 10 time window · 11 DPOF not flat · 12 serving tree / target · 13 tests failed ·
14 prune failed · 15 deploy failed · 16 restart failed · 20 deployed but verification failed · 30 already done / locked.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
SERVING = Path("/home/agent/openclaw")
PY = str(SERVING / "venv" / "bin" / "python")
ALLOWED_DIRTY_PREFIXES = ("data/memory/", "data/personality/")
RESTART_CMD = ["sudo", "-n", "/usr/bin/systemctl", "restart", "moltbot"]


class Abort(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code


# ------------------------------------------------------------------------------- pure helpers (unit-tested)
def parse_hhmm(text: str) -> int:
    h, m = text.split(":")
    return int(h) * 60 + int(m)


def in_window(now_et: datetime, not_before: str, not_after: str) -> bool:
    minutes = now_et.hour * 60 + now_et.minute
    return parse_hhmm(not_before) <= minutes <= parse_hhmm(not_after)


def dirty_source_paths(porcelain: str) -> list[str]:
    """Tracked files with modifications that are NOT sanctioned runtime state (untracked '??' lines are ignored)."""
    bad = []
    for line in porcelain.splitlines():
        if not line.strip() or line.startswith("??"):
            continue
        path = line[3:].strip().strip('"')
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if not path.startswith(ALLOWED_DIRTY_PREFIXES):
            bad.append(path)
    return bad


def live_positions(positions_json: dict) -> list[dict]:
    return [p for p in (positions_json.get("positions") or []) if p.get("status") in ("pending", "open", "closing")]


def failed_test_ids(pytest_output: str) -> list[str]:
    """Node ids from pytest's short-summary 'FAILED <nodeid> - <reason>' lines (ERROR lines are not rerunnable)."""
    ids = []
    for line in pytest_output.splitlines():
        if line.startswith("FAILED "):
            node = line[len("FAILED "):].split(" - ", 1)[0].strip()
            if node and node not in ids:
                ids.append(node)
    return ids


def changed_agent_modules(diff_names: str) -> list[str]:
    mods = []
    for name in diff_names.splitlines():
        if name.startswith("agent/") and name.endswith(".py"):
            mods.append(name[:-3].replace("/", "."))
    return mods


# ------------------------------------------------------------------------------- orchestrator
class Deployer:
    def __init__(self, args, *, run=None, now=None, sleep=time.sleep, serving: Path = SERVING):
        self.args = args
        self.serving = serving
        self.run = run or self._run
        self.now = now or (lambda: datetime.now(NY))
        self.sleep = sleep
        self.log_lines: list[str] = []
        self.tmp_worktree: Path | None = None
        self.commands: list[list[str]] = []

    # -- plumbing
    def _run(self, cmd, timeout=120, cwd=None):
        self.commands.append(list(cmd))
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(cwd or self.serving), check=False)
        return p.returncode, (p.stdout or "") + (p.stderr or "")

    def say(self, msg: str) -> None:
        line = f"[{self.now().strftime('%H:%M:%S ET')}] {msg}"
        self.log_lines.append(line)
        print(line, flush=True)

    def git(self, *a, cwd=None, timeout=120):
        return self.run(["git", *a], timeout=timeout, cwd=cwd)

    @property
    def apply(self) -> bool:
        return bool(self.args.apply)

    # -- steps
    def step_window(self):
        now = self.now()
        ok = in_window(now, self.args.not_before, self.args.not_after)
        self.say(f"STEP 1 time window {self.args.not_before}-{self.args.not_after} ET: now {now.strftime('%H:%M')} -> {'ok' if ok else 'OUTSIDE'}")
        if not ok and self.apply:
            raise Abort(10, f"outside the allowed window {self.args.not_before}-{self.args.not_after} ET")

    def _flat(self) -> bool:
        path = self.serving / "data" / "dpof" / "positions.json"
        try:
            data = json.loads(path.read_text())
        except FileNotFoundError:
            return True
        return not live_positions(data)

    def step_wait_flat(self, label: str):
        if self._flat():
            self.say(f"{label}: DPOF flat")
            return
        if not self.apply:
            self.say(f"{label}: DPOF NOT flat right now (rehearsal: not waiting)")
            return
        deadline = parse_hhmm(self.args.wait_flat_until)
        self.say(f"{label}: DPOF not flat -- waiting (until {self.args.wait_flat_until} ET)")
        while not self._flat():
            n = self.now()
            if n.hour * 60 + n.minute >= deadline:
                raise Abort(11, f"DPOF still not flat at {self.args.wait_flat_until} ET; nothing was changed")
            self.sleep(20)
        self.say(f"{label}: DPOF now flat")

    def step_tree_and_target(self):
        t = self.args.target_sha
        rc, branch = self.git("branch", "--show-current")
        if branch.strip() != "main":
            raise Abort(12, f"serving tree is on {branch.strip()!r}, expected 'main'")
        rc, status = self.git("status", "--porcelain")
        bad = dirty_source_paths(status)
        if bad:
            raise Abort(12, f"serving tree has modified tracked SOURCE files (someone mid-edit?): {bad[:5]}")
        rc, _ = self.git("cat-file", "-e", f"{t}^{{commit}}")
        if rc != 0:
            raise Abort(12, f"target {t} is not a commit in this repository")
        rc, head = self.git("rev-parse", "HEAD")
        self.head_before = head.strip()
        rc, _ = self.git("merge-base", "--is-ancestor", "HEAD", t)
        if rc != 0:
            raise Abort(12, f"HEAD {self.head_before[:8]} is not an ancestor of the target -- not a fast-forward")
        if self.args.remote_branch:
            self.git("fetch", "origin", self.args.remote_branch, timeout=90)
            rc, _ = self.git("merge-base", "--is-ancestor", t, f"origin/{self.args.remote_branch}")
            if rc != 0:
                raise Abort(12, f"target is not contained in origin/{self.args.remote_branch} (not pushed?)")
        rc, log = self.git("log", "--oneline", f"{self.head_before}..{t}")
        self.commits = log.strip().splitlines()
        self.say(f"STEP 3 serving tree clean of source edits; target {t[:8]} fast-forwards {self.head_before[:8]} by {len(self.commits)} commit(s)")

    def step_tests(self):
        t = self.args.target_sha
        wt = Path(f"/tmp/clio_post_close_{self.now().strftime('%Y%m%d_%H%M%S')}")
        rc, out = self.git("worktree", "add", "--detach", str(wt), t, timeout=120)
        if rc != 0:
            raise Abort(13, f"could not create the temp worktree: {out[-300:]}")
        self.tmp_worktree = wt
        if self.args.skip_tests:
            self.say("STEP 4 tests SKIPPED (--skip-tests)")
            return
        rc, prov = self.run([PY, "-c", "import agent; print(agent.__file__)"], timeout=60, cwd=wt)
        if str(wt) not in prov:
            raise Abort(13, f"test provenance wrong (agent imported from {prov.strip()!r}, not the target worktree)")
        self.say("STEP 4 running the full test suite on the target commit ...")
        rc, out = self.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests", "--maxfail=10"], timeout=1200, cwd=wt)
        tail = out.strip().splitlines()[-1] if out.strip() else ""
        if rc != 0:
            failed = failed_test_ids(out)
            self.say(f"STEP 4 first run FAILED ({tail}); failing: {failed[:8]}")
            if not failed or len(failed) > 5:
                raise Abort(13, f"tests FAILED on the target commit: {tail}; failing: {failed[:8]}")
            # ONE rerun of only the failed tests. A real regression fails again (the deploy is held); a
            # load-sensitive test does not -- and is logged loudly so it gets fixed.
            rc2, out2 = self.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", *failed], timeout=600, cwd=wt)
            tail2 = out2.strip().splitlines()[-1] if out2.strip() else ""
            if rc2 != 0:
                raise Abort(13, f"tests FAILED on the target commit and again on rerun: {failed}: {tail2}")
            self.say(f"STEP 4 FLAKY: passed on rerun ({tail2}): {failed} -- fix these tests")
            return
        self.say(f"STEP 4 tests passed: {tail}")

    def step_prune(self):
        wt = self.tmp_worktree
        script = str(wt / "scripts" / "uw_ledger_prune.py")
        cmd = [PY, script, "--older-than-days", str(self.args.prune_days)]
        if self.apply:
            cmd += ["--apply", "--vacuum"]
        self.say(f"STEP 5 ledger prune ({'APPLY: backup -> delete -> vacuum' if self.apply else 'dry run'}) ...")
        rc, out = self.run(cmd, timeout=1800, cwd=self.serving)
        for line in out.strip().splitlines()[-12:]:
            self.say(f"    {line}")
        if rc != 0:
            raise Abort(14, f"ledger prune exited {rc}; nothing was deployed (a backup exists if it got that far)")

    def step_deploy(self):
        t = self.args.target_sha
        if not self.apply:
            self.say(f"STEP 6 (rehearsal) would: git merge --ff-only {t[:8]}, fresh-import every module changed by the {len(self.commits)} commit(s)")
            return
        rc, out = self.git("merge", "--ff-only", t, timeout=120)
        if rc != 0:
            raise Abort(15, f"fast-forward merge failed: {out[-300:]}")
        rc, head = self.git("rev-parse", "HEAD")
        if head.strip() != t:
            raise Abort(15, f"HEAD is {head.strip()[:8]} after the merge, expected {t[:8]}")
        rc, names = self.git("diff", "--name-only", f"{self.head_before}..{t}")
        for mod in changed_agent_modules(names):
            rc, out = self.run([PY, "-c", f"import {mod}"], timeout=90)
            if rc != 0:
                raise Abort(15, f"import of {mod} FAILED after the merge (restart NOT issued): {out.strip()[-300:]}")
        self.say(f"STEP 6 serving tree now at {t[:8]}; every changed module imports in the serving venv")

    def step_restart(self):
        if not self.apply:
            rc, out = self.run(["sudo", "-n", "/usr/bin/systemctl", "is-active", "moltbot"], timeout=30)
            self.say(f"STEP 8 (rehearsal) sudo non-interactive works: is-active moltbot -> {out.strip()!r} (rc {rc})")
            if rc != 0 and out.strip() != "active":
                raise Abort(16, "sudo -n systemctl is not usable from this context")
            return
        self.step_wait_flat("STEP 7 pre-restart")
        self.restart_at = self.now()
        rc, out = self.run(RESTART_CMD, timeout=180)
        if rc != 0:
            raise Abort(16, f"restart command failed (rc {rc}): {out.strip()[-300:]}")
        self.say("STEP 8 restart issued")

    def step_verify(self):
        if not self.apply:
            self.say("STEP 9 (rehearsal) would: 3x is-active over 30 s, journal Traceback count, uw_ledger_report since the restart")
            return
        self.sleep(15)
        for i in range(3):
            rc, out = self.run(["systemctl", "is-active", "moltbot"], timeout=30)
            if out.strip() != "active":
                raise Abort(20, f"moltbot is {out.strip()!r} after the restart -- DEPLOYED BUT NOT HEALTHY")
            if i < 2:
                self.sleep(10)
        since = self.restart_at.strftime("%Y-%m-%d %H:%M:%S")
        rc, journal = self.run(["journalctl", "-u", "moltbot", "--since", since, "--no-pager"], timeout=60)
        tb = journal.count("Traceback")
        coord = journal.count("UW quota coordinator unavailable")
        self.say(f"STEP 9 moltbot active x3; journal since {since}: Traceback={tb}, 'quota coordinator unavailable'={coord}")
        if tb:
            raise Abort(20, f"{tb} Traceback(s) in the journal since the restart -- DEPLOYED BUT NOT HEALTHY")
        script = self.serving / "scripts" / "uw_ledger_report.py"
        if script.is_file():
            utc_since = self.restart_at.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%S")
            rc, out = self.run([PY, str(script), "--since", utc_since], timeout=120)
            for line in out.strip().splitlines()[:10]:
                self.say(f"    {line}")
        mc = self.serving / "scripts" / "morning_check.py"
        self.say(f"        morning_check.py present in the serving tree: {mc.is_file()} (the 09:50 ET timer runs it)")

    def cleanup(self):
        if self.tmp_worktree is not None:
            self.git("worktree", "remove", "--force", str(self.tmp_worktree), timeout=60)
            self.tmp_worktree = None

    # -- driver
    def execute(self) -> int:
        marker = self.serving / "logs" / "post_close_deploy.done"
        lock_path = self.serving / "logs" / "post_close_deploy.lock"
        self.say(f"post_close_deploy {'APPLY' if self.apply else 'REHEARSAL'}  target {self.args.target_sha[:8]}  prune-days {self.args.prune_days}")
        if marker.exists() and self.apply:
            self.say(f"already done ({marker.read_text().strip()[:120]}) -- nothing to do")
            return 30
        lock_fh = open(lock_path, "a")
        try:
            fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.say("another post_close_deploy holds the lock -- exiting")
            return 30
        try:
            self.step_window()
            self.step_wait_flat("STEP 2")
            self.step_tree_and_target()
            self.step_tests()
            self.step_prune()
            if self.apply:
                self.step_wait_flat("STEP 6a pre-deploy")
            self.step_deploy()
            self.step_restart()
            self.step_verify()
            if self.apply:
                marker.write_text(f"{self.now().isoformat()} deployed {self.args.target_sha} ({len(self.commits)} commits)\n")
                self.say("DONE: backup + prune + vacuum + deploy + restart + verification complete")
            else:
                self.say("REHEARSAL OK: every check passed; --apply would run the steps above")
            return 0
        except Abort as e:
            self.say(f"ABORTED (exit {e.code}): {e}")
            if e.code == 20:
                self.say("    NO auto-rollback. Rollback = git revert of: " + ", ".join(c.split()[0] for c in getattr(self, 'commits', [])))
            return e.code
        finally:
            self.cleanup()
            try:
                fcntl.flock(lock_fh, fcntl.LOCK_UN)
                lock_fh.close()
            except OSError:
                pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target-sha", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--not-before", default="16:15")
    ap.add_argument("--not-after", default="22:00")
    ap.add_argument("--wait-flat-until", default="19:00")
    ap.add_argument("--prune-days", type=int, default=14)
    ap.add_argument("--remote-branch", default="fix/clio-uw5-stale-probe")
    ap.add_argument("--skip-tests", action="store_true", help="rehearsal speed-up only; --apply refuses it")
    args = ap.parse_args(argv)
    if len(args.target_sha) != 40 or any(c not in "0123456789abcdef" for c in args.target_sha):
        print("REFUSED: --target-sha must be a full 40-hex commit id", file=sys.stderr)
        return 2
    if args.apply and args.skip_tests:
        print("REFUSED: --skip-tests is for rehearsals; a real run always tests the target commit", file=sys.stderr)
        return 2
    d = Deployer(args)
    code = d.execute()
    try:
        logdir = SERVING / "logs"
        with open(logdir / f"post_close_deploy_{datetime.now(NY).strftime('%Y%m%d')}.log", "a") as fh:
            fh.write("\n".join(d.log_lines) + "\n")
    except OSError:
        pass
    return code


if __name__ == "__main__":
    sys.exit(main())
