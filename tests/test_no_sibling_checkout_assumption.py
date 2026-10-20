"""S211: a test must exercise the checkout it lives in, never "a sibling directory named openclaw".

Two tests computed ``Path(__file__).resolve().parents[2] / "openclaw"``. From tests/ that is the checkout's PARENT plus
"openclaw" -- right only inside the serving tree itself. In every git worktree it silently pointed at the live serving
tree (the test checked the wrong code and passed), and in a checkout under /tmp it failed with FileNotFoundError; that
was found when the post-close deploy job rehearsed the suite on the exact target commit.
"""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIBLING = re.compile(r"""(?:ROOT|parents\[2\])\s*/\s*["']openclaw["']""")


def _tracked_tests():
    out = subprocess.run(["git", "ls-files", "tests"], cwd=ROOT, capture_output=True, text=True, check=False)
    if out.returncode != 0:
        return None
    return [ROOT / p for p in out.stdout.splitlines() if p.endswith(".py")]


def test_no_test_resolves_a_sibling_openclaw_directory():
    files = _tracked_tests()
    if files is None:
        return  # not a git checkout here
    offenders = []
    for f in files:
        if f.name == Path(__file__).name:
            continue
        for n, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
            if SIBLING.search(line):
                offenders.append(f"{f.relative_to(ROOT)}:{n}: {line.strip()}")
    assert not offenders, "tests must use the checkout that contains them:\n" + "\n".join(offenders)


def test_top_level_tests_do_not_use_parents_2_as_the_repo_root():
    """From tests/x.py, parents[1] is the repo; parents[2] is its parent. (tests/<sub>/x.py legitimately uses [2].)"""
    files = _tracked_tests()
    if files is None:
        return
    offenders = []
    for f in files:
        if f.parent != ROOT / "tests" or f.name == Path(__file__).name:
            continue
        for n, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
            if re.search(r"__file__.*parents\[2\]", line):
                offenders.append(f"{f.relative_to(ROOT)}:{n}: {line.strip()}")
    assert not offenders, "top-level tests must not treat parents[2] as the repo root:\n" + "\n".join(offenders)
