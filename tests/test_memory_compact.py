"""scripts/memory_compact.py: move-only MEMORY.md compaction (2026-09-28).

Every test builds a synthetic memory dir under tmp_path; nothing here reads or writes a real
Claude Code memory directory. The invariants: nothing in the memory dir is deleted, rewritten or
overwritten; the original is reconstructible byte-for-byte; every refusal leaves the memory dir
exactly as it was; a concurrent edit is never lost. The oracles are independent of the tool: a
full lstat snapshot (mode, inode, mtime, symlinks, dirs) and a blind reconstruction that reads
only the files, never the manifest.
"""

import errno
import json
import os
import random
import re
import stat
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from scripts import memory_compact as mc

REPO_ROOT = Path(__file__).resolve().parents[1]
MARKER_RE = re.compile(rb"^<!-- memory_compact:block (\d+) lines (\d+)-(\d+) sha256=([0-9a-f]{64}) -->\n$")


# ---------------------------------------------------------------------------
# fixtures and independent oracles
# ---------------------------------------------------------------------------

def _sample() -> str:
    lines = ["# Memory index", "", "- [Feedback X](feedback_x.md) — a rule", "", "## Active State"]
    lines += [f"- current thing {n}" for n in range(10)]
    lines += ["", "## History"]
    lines += [f"- S{n}: something happened in session {n}, see [x](feedback_x.md)" for n in range(100, 160)]
    lines += ["", "## Old detail"]
    lines += [f"detail line {n} " + "x" * 60 for n in range(40)]
    lines += ["", "## Footer", "- keep this line"]
    return "\n".join(lines) + "\n"


def _mk_memory(tmp_path: Path, text, mode: int = 0o600) -> Path:
    mem = tmp_path / "memory"
    mem.mkdir()
    data = text.encode() if isinstance(text, str) else text
    (mem / "MEMORY.md").write_bytes(data)
    os.chmod(mem / "MEMORY.md", mode)
    (mem / "feedback_x.md").write_text("---\nname: feedback_x\n---\nkeep me\n")
    (mem / "sub").mkdir()
    (mem / "sub" / "nested.md").write_text("nested\n")
    (mem / "emptydir").mkdir()
    os.symlink("feedback_x.md", mem / "link_to_feedback.md")
    return mem


def _snap(d: Path) -> dict:
    """Every entry under d (not d itself): type, mode, inode, mtime_ns, size, content hash, link target."""
    out = {}
    for root, dirs, files in os.walk(d, followlinks=False):
        for name in sorted(dirs) + sorted(files):
            p = Path(root) / name
            st = os.lstat(p)
            entry = {"mode": st.st_mode, "ino": st.st_ino, "mtime": st.st_mtime_ns, "size": st.st_size}
            if stat.S_ISLNK(st.st_mode):
                entry["target"] = os.readlink(p)
            elif stat.S_ISREG(st.st_mode):
                entry["sha"] = mc.sha256_bytes(p.read_bytes())
            out[str(p.relative_to(d))] = entry
    return out


def _blind_reconstruct(index: bytes, archives: list[bytes]) -> bytes:
    """Rebuild the original from files alone: markers carry original line ranges and a sha."""
    blocks = []
    for data in archives:
        lines = data.splitlines(keepends=True)
        marks = [(i, MARKER_RE.match(ln)) for i, ln in enumerate(lines) if MARKER_RE.match(ln)]
        for k, (i, m) in enumerate(marks):
            nxt = marks[k + 1][0] if k + 1 < len(marks) else len(lines)
            body = b"".join(lines[i + 1:nxt])
            sha = m.group(4).decode()
            if mc.sha256_bytes(body) != sha:
                assert body.endswith(b"\n") and mc.sha256_bytes(body[:-1]) == sha, "block does not match its sha"
                body = body[:-1]
            blocks.append((int(m.group(2)), int(m.group(3)), body))
    blocks.sort()
    idx = index.splitlines(keepends=True)
    out, shift, pos = [], 0, 0
    for s, e, body in blocks:
        p = s - 1 - shift  # 0-based index line holding this block's pointer
        out.extend(idx[pos:p])
        out.append(body)
        pos = p + 1
        shift += e - s
    out.extend(idx[pos:])
    return b"".join(out)


def _plan(mem: Path, moves, archives=None, window=None, path=None, sha=None) -> Path:
    sha = sha or mc.sha256_file(mem / "MEMORY.md")
    archives = archives or {m["archive"]: {"description": f"moved {m['archive']}"} for m in moves}
    plan = {"source_sha256": sha, "archives": archives, "moves": moves}
    if window:
        plan["window"] = window
    path = path or mem.parent / f"plan_{len(list(mem.parent.glob('plan_*.json')))}.json"
    path.write_text(json.dumps(plan))
    return path


def _line_of(mem: Path, prefix: str) -> int:
    for i, ln in enumerate((mem / "MEMORY.md").read_text().splitlines(), 1):
        if ln.startswith(prefix):
            return i
    raise AssertionError(prefix)


def _std_moves(mem: Path):
    h, o, f = _line_of(mem, "## History"), _line_of(mem, "## Old detail"), _line_of(mem, "## Footer")
    return [
        {"start": h + 1, "end": o - 2, "archive": "archive_session_history.md",
         "pointer": "- [Session history S100-S159](archive_session_history.md) — per-session notes"},
        {"start": o + 1, "end": f - 2, "archive": "archive_old_detail.md",
         "pointer": "- [Old detail](archive_old_detail.md) -- superseded detail"},
    ]


def _staged(mem, tmp_path, moves=None, name="stage", **kw):
    stage = tmp_path / name
    m = mc.stage(mem, [_plan(mem, moves or _std_moves(mem))], stage, kw.get("max_bytes", 20000),
                 kw.get("max_lines", 180), kw.get("min_lines", 20))
    return stage, m


def _live_archives(mem: Path) -> list[bytes]:
    return [p.read_bytes() for p in sorted(mem.glob("archive_*.md"))]


@pytest.fixture(params=["exchange", "fallback"])
def swap_mode(request, monkeypatch):
    if request.param == "fallback":
        monkeypatch.setattr(mc, "_EXCHANGE_DISABLED", True)
        monkeypatch.setattr(mc, "ALLOW_FALLBACK_DEFAULT", True)
    elif not _exchange_supported():
        pytest.skip("RENAME_EXCHANGE not supported on this filesystem")
    return request.param


def _exchange_supported() -> bool:
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        a, b = Path(d) / "a", Path(d) / "b"
        a.write_text("a")
        b.write_text("b")
        return mc._rename_exchange(a, b) and a.read_text() == "b"


# ---------------------------------------------------------------------------
# survey + stage
# ---------------------------------------------------------------------------

def test_survey_is_read_only_and_windows_cover_every_line(tmp_path):
    mem = _mk_memory(tmp_path, _sample())
    before = _snap(mem)
    s = mc.survey(mem, 3, 20000, 180)
    assert _snap(mem) == before
    assert s["sha256"] == mc.sha256_file(mem / "MEMORY.md") and s["lone_cr"] == 0
    wins = s["windows"]
    assert wins[0]["start"] == 1 and wins[-1]["end"] == s["lines"]
    for a, b in zip(wins, wins[1:]):
        assert b["start"] == a["end"] + 1
    heading_lines = {h["line"] for h in s["headings"]}
    assert all(w["start"] == 1 or w["start"] in heading_lines for w in wins)


def test_stage_moves_verbatim_and_never_touches_the_memory_dir(tmp_path):
    mem = _mk_memory(tmp_path, _sample())
    before = _snap(mem)
    original = (mem / "MEMORY.md").read_bytes()
    stage, m = _staged(mem, tmp_path)
    assert _snap(mem) == before
    assert m["ok"] and m["budget"]["within"] and m["budget"]["floor_ok"]
    index = (stage / "MEMORY.md").read_bytes()
    assert _blind_reconstruct(index, [(stage / n).read_bytes() for n in sorted(m["archives"])]) == original
    assert b"- keep this line" in index and b"- current thing 9" in index
    hist = (stage / "archive_session_history.md").read_text()
    assert hist.startswith('---\nname: archive_session_history\ndescription: "moved archive_session_history.md"\n')
    assert "- S100: something happened in session 100, see [x](feedback_x.md)\n" in hist
    report = (stage / "report.md").read_text()
    assert all(f"pointer: {mv['pointer']}" in report for mv in _std_moves(mem))


def _refusal_cases(mem: Path):
    good = _std_moves(mem)
    g0 = good[0]
    ptr = lambda p: [{**g0, "pointer": p}]  # noqa: E731
    return {
        "sha": ({"moves": good, "sha": "0" * 64}, "source_sha256 does not match"),
        "overlap": ({"moves": [g0, {**g0, "archive": "archive_b.md", "pointer": "- [b](archive_b.md) — b"}]}, "overlap"),
        "traversal": ({"moves": [{**g0, "archive": "../archive_x.md", "pointer": "- [x](../archive_x.md) — x"}]}, "must match"),
        "not_archive_prefix": ({"moves": [{**g0, "archive": "notes.md", "pointer": "- [x](notes.md) — x"}]}, "must match"),
        "newline_in_name": ({"moves": [{**g0, "archive": "archive_a.md\n", "pointer": "- [x](archive_a.md) — x"}]}, "must match"),
        "overwrite_existing": ({"moves": [{**g0, "archive": "archive_exists.md",
                                           "pointer": "- [x](archive_exists.md) — x"}]}, "already exists"),
        "pointer_without_link": ({"moves": ptr("- no link here")}, "pointer must be exactly"),
        "pointer_wrong_archive": ({"moves": ptr("- [x](archive_other.md) — x")}, "pointer must be exactly"),
        "pointer_html_comment": ({"moves": ptr("- [x](archive_session_history.md) — hi <!-- sneaky")}, "pointer must be exactly"),
        "pointer_backtick": ({"moves": ptr("- [x](archive_session_history.md) — run `rm`")}, "pointer must be exactly"),
        "pointer_bidi": ({"moves": ptr("- [x](archive_session_history.md) — a‮b")}, "control/format"),
        "pointer_line_sep": ({"moves": ptr("- [x](archive_session_history.md) — a b")}, "control/format"),
        "pointer_surrogate": ({"moves": ptr("- [x](archive_session_history.md) — a\ud800")}, "control/format"),
        "multiline_pointer": ({"moves": ptr(g0["pointer"] + "\nextra")}, "control/format"),
        "description_lt": ({"moves": [g0], "archives": {g0["archive"]: {"description": "a <b>"}}}, "control/format"),
        "unhashable_archive": ({"moves": [{**g0, "archive": ["archive_session_history.md"]}],
                                "archives": {g0["archive"]: {"description": "d"}}}, "not declared"),
        "line_one": ({"moves": [{**g0, "start": 1}]}, "line 1"),
        "out_of_window": ({"moves": good, "window": {"start": 1, "end": 20}}, "outside"),
        "past_eof": ({"moves": [{**g0, "end": 10_000}]}, "outside"),
        "unused_archive": ({"moves": good, "archives": {**{m["archive"]: {"description": "d"} for m in good},
                                                        "archive_unused.md": {"description": "d"}}}, "never used"),
        "bool_line": ({"moves": [{**g0, "start": True}]}, "integers"),
        "extra_key": ({"moves": [{**g0, "delete": True}]}, "must be"),
    }


def test_stage_refusals_give_the_intended_reason_and_touch_nothing(tmp_path):
    mem = _mk_memory(tmp_path, _sample())
    (mem / "archive_exists.md").write_text("pre-existing\n")
    before = _snap(mem)
    for name, (spec, reason) in _refusal_cases(mem).items():
        p = _plan(mem, spec["moves"], archives=spec.get("archives"), window=spec.get("window"),
                  path=tmp_path / f"bad_{name}.json", sha=spec.get("sha"))
        stage = tmp_path / f"stage_{name}"
        with pytest.raises(mc.Refused, match=re.escape(reason)):
            mc.stage(mem, [p], stage, 20000, 180)
        assert _snap(mem) == before, name
        assert not stage.exists(), name


def test_hostile_json_is_refused_not_a_traceback(tmp_path):
    mem = _mk_memory(tmp_path, _sample())
    sha = mc.sha256_file(mem / "MEMORY.md")
    before = _snap(mem)
    dup = tmp_path / "dup.json"
    dup.write_text('{"source_sha256": "%s", "moves": [], "moves": []}' % sha)
    deep = tmp_path / "deep.json"
    deep.write_text("[" * 100000 + "]" * 100000)
    for p, reason in ((dup, "duplicate JSON keys"), (deep, "cannot read plan JSON")):
        with pytest.raises(mc.Refused, match=reason):
            mc.stage(mem, [p], tmp_path / f"s_{p.stem}", 20000, 180)
    assert _snap(mem) == before


def test_code_fence_split_bare_cr_and_marker_lines_are_refused(tmp_path):
    fenced = "# T\nkeep\n```\ncode a\ncode b\n```\nafter\n" + "".join(f"k{n}\n" for n in range(30))
    (tmp_path / "a").mkdir()
    mem = _mk_memory(tmp_path / "a", fenced)
    moves = [{"start": 3, "end": 4, "archive": "archive_a.md", "pointer": "- [a](archive_a.md) — a"}]
    with pytest.raises(mc.Refused, match="code fence"):
        mc.stage(mem, [_plan(mem, moves)], tmp_path / "s1", 20000, 180)
    whole = [{"start": 3, "end": 6, "archive": "archive_a.md", "pointer": "- [a](archive_a.md) — a"}]
    assert mc.stage(mem, [_plan(mem, whole)], tmp_path / "s2", 20000, 180)["ok"]
    (tmp_path / "b").mkdir()
    mem_b = _mk_memory(tmp_path / "b", b"# T\nline\rmore\nx\n")
    with pytest.raises(mc.Refused, match="bare CR"):
        mc.stage(mem_b, [_plan(mem_b, moves[:1])], tmp_path / "s3", 20000, 180)
    (tmp_path / "c").mkdir()
    mem_c = _mk_memory(tmp_path / "c", "# T\n<!-- memory_compact:block 0 lines 1-2 sha256=x -->\nmore\n")
    with pytest.raises(mc.Refused, match="block markers"):
        mc.stage(mem_c, [_plan(mem_c, [{**moves[0], "start": 2, "end": 3}])], tmp_path / "s4", 20000, 180)


def test_stage_dir_inside_the_memory_dir_is_refused_before_writing(tmp_path):
    mem = _mk_memory(tmp_path, _sample())
    plan = _plan(mem, _std_moves(mem))
    before = _snap(mem)
    for bad in (mem / "stage", tmp_path):
        with pytest.raises(mc.Refused, match="outside the memory dir"):
            mc.stage(mem, [plan], bad, 20000, 180)
        assert _snap(mem) == before
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "x").write_text("x")
    with pytest.raises(mc.Refused, match="new or empty"):
        mc.stage(mem, [plan], busy, 20000, 180)
    assert (busy / "x").read_text() == "x"


def test_two_parallel_plans_merge_and_overlap_across_plans_is_refused(tmp_path):
    mem = _mk_memory(tmp_path, _sample())
    a, b = _std_moves(mem)
    pa, pb = _plan(mem, [a], path=tmp_path / "pa.json"), _plan(mem, [b], path=tmp_path / "pb.json")
    m = mc.stage(mem, [pa, pb], tmp_path / "s1", 20000, 180)
    assert [blk["start"] for blk in m["blocks"]] == [a["start"], b["start"]]
    clash = _plan(mem, [{**b, "start": a["end"] - 1}], path=tmp_path / "pc.json")
    with pytest.raises(mc.Refused, match="overlap"):
        mc.stage(mem, [pa, clash], tmp_path / "s2", 20000, 180)


def test_over_budget_or_under_floor_is_not_ok_and_apply_refuses(tmp_path):
    mem = _mk_memory(tmp_path, _sample())
    stage, m = _staged(mem, tmp_path, _std_moves(mem)[:1], max_bytes=500)
    assert not m["ok"] and m["problems"] == ["over budget"]
    everything = [{"start": 2, "end": _line_of(mem, "- keep this line"), "archive": "archive_all.md",
                   "pointer": "- [All](archive_all.md) — everything"}]
    stage2, m2 = _staged(mem, tmp_path, everything, name="stage2")
    assert not m2["ok"] and "fewer than 20 lines kept" in m2["problems"]
    before = _snap(mem)
    for st in (stage, stage2):
        with pytest.raises(mc.Refused, match="not OK"):
            mc.apply(mem, st, tmp_path / "backups")
    assert _snap(mem) == before


# ---------------------------------------------------------------------------
# apply / rollback
# ---------------------------------------------------------------------------

@pytest.fixture
def umask_022():
    old = os.umask(0o022)
    yield
    os.umask(old)


def test_apply_and_rollback_round_trip(tmp_path, swap_mode, umask_022):
    mem = _mk_memory(tmp_path, _sample(), mode=0o664)
    original = (mem / "MEMORY.md").read_bytes()
    before = _snap(mem)
    stage, m = _staged(mem, tmp_path)
    run = mc.apply(mem, stage, tmp_path / "backups")
    run_dir = Path(run["run_dir"])
    after = _snap(mem)
    unchanged = {k: v for k, v in before.items() if k != "MEMORY.md"}
    assert {k: after[k] for k in unchanged} == unchanged
    assert set(after) - set(before) == set(m["archives"])
    assert stat.S_IMODE(os.stat(mem / "MEMORY.md").st_mode) == 0o664
    assert all(stat.S_IMODE(os.stat(mem / n).st_mode) == 0o664 for n in m["archives"])
    assert _blind_reconstruct((mem / "MEMORY.md").read_bytes(), _live_archives(mem)) == original
    with tarfile.open(run_dir / "memory_dir.tar.gz") as tar:
        files = {x.name: tar.extractfile(x).read() for x in tar.getmembers() if x.isfile()}
    assert set(files) == {k for k, v in before.items() if "sha" in v}
    assert files["MEMORY.md"] == original
    assert (run_dir / "MEMORY.md.orig").read_bytes() == original
    assert Path(run["displaced"]).read_bytes() == original  # the old file is kept, never unlinked
    assert json.loads((run_dir / "run.json").read_text())["state"] == "applied"
    res = mc.rollback(mem, run_dir)
    assert res["restored"] and (mem / "MEMORY.md").read_bytes() == original
    assert stat.S_IMODE(os.stat(mem / "MEMORY.md").st_mode) == 0o664
    restored = _snap(mem)
    assert {k: restored[k] for k in unchanged} == unchanged and set(restored) == set(before)
    assert all((run_dir / "rolled_back" / n).is_file() for n in m["archives"])
    again = mc.rollback(mem, run_dir)  # idempotent
    assert not again["restored"] and again["moved"] == []


def test_apply_refuses_when_memory_changed_after_stage(tmp_path):
    mem = _mk_memory(tmp_path, _sample())
    stage, _ = _staged(mem, tmp_path)
    with open(mem / "MEMORY.md", "a") as f:
        f.write("- appended by another session\n")
    before = _snap(mem)
    with pytest.raises(mc.Refused, match="changed since staging"):
        mc.apply(mem, stage, tmp_path / "backups")
    assert _snap(mem) == before


def test_apply_refuses_a_tampered_stage(tmp_path):
    mem = _mk_memory(tmp_path, _sample())
    stage, m = _staged(mem, tmp_path)
    name = next(iter(m["archives"]))
    (stage / name).write_bytes((stage / name).read_bytes() + b"tampered\n")
    before = _snap(mem)
    with pytest.raises(mc.Refused, match="edited after staging"):
        mc.apply(mem, stage, tmp_path / "b1")
    assert _snap(mem) == before


def test_an_archive_name_that_appears_mid_apply_is_never_overwritten(tmp_path, monkeypatch):
    mem = _mk_memory(tmp_path, _sample())
    stage, m = _staged(mem, tmp_path)
    name = list(m["archives"])[-1]  # written last
    real = mc._new_run_dir

    def racing_run_dir(root, memdir):
        out = real(root, memdir)
        (mem / name).write_text("someone else's file\n")  # after every pre-check
        return out
    monkeypatch.setattr(mc, "_new_run_dir", racing_run_dir)
    original = (mem / "MEMORY.md").read_bytes()
    with pytest.raises(mc.Refused, match="aborted and undone"):
        mc.apply(mem, stage, tmp_path / "backups")
    assert (mem / name).read_text() == "someone else's file\n"
    assert (mem / "MEMORY.md").read_bytes() == original
    assert [p.name for p in mem.glob("archive_*.md")] == [name]


def test_edit_during_backup_is_refused(tmp_path, monkeypatch):
    mem = _mk_memory(tmp_path, _sample())
    stage, _ = _staged(mem, tmp_path)
    real_open = mc.tarfile.open

    def editing_open(path, mode="r", *a, **kw):
        if mode.startswith("w"):
            with open(mem / "MEMORY.md", "a") as f:
                f.write("- edit during backup\n")
        return real_open(path, mode, *a, **kw)
    monkeypatch.setattr(mc.tarfile, "open", editing_open)
    with pytest.raises(mc.Refused, match="changed during the backup|does not verify"):
        mc.apply(mem, stage, tmp_path / "backups")
    assert (mem / "MEMORY.md").read_text().endswith("- edit during backup\n")
    assert not list(mem.glob("archive_*.md"))


def test_a_corrupt_backup_is_refused(tmp_path, monkeypatch):
    import io
    mem = _mk_memory(tmp_path, _sample())
    stage, _ = _staged(mem, tmp_path)
    real_add = tarfile.TarFile.add

    def bad_add(self, name, arcname=None, recursive=True, **kw):
        if arcname == "MEMORY.md":
            data = b"not the original\n"
            info = tarfile.TarInfo("MEMORY.md")
            info.size = len(data)
            return self.addfile(info, io.BytesIO(data))
        return real_add(self, name, arcname=arcname, recursive=recursive, **kw)
    monkeypatch.setattr(tarfile.TarFile, "add", bad_add)
    before = _snap(mem)
    with pytest.raises(mc.Refused, match="does not verify"):
        mc.apply(mem, stage, tmp_path / "backups")
    assert _snap(mem) == before


def test_a_concurrent_edit_at_the_swap_is_never_lost(tmp_path, monkeypatch, swap_mode):
    mem = _mk_memory(tmp_path, _sample())
    stage, _ = _staged(mem, tmp_path)
    edit = b"- written by another session at the last instant\n"
    if swap_mode == "exchange":
        real = mc._rename_exchange
        calls = []

        def racing_exchange(a, b):
            if Path(b).name != "MEMORY.md":  # the capability probe
                return real(a, b)
            if not calls:
                with open(mem / "MEMORY.md", "ab") as f:
                    f.write(edit)
            calls.append(1)
            return real(a, b)
        monkeypatch.setattr(mc, "_rename_exchange", racing_exchange)
        with pytest.raises(mc.Refused, match="swapped back, nothing lost"):
            mc.apply(mem, stage, tmp_path / "backups")
        assert (mem / "MEMORY.md").read_bytes().endswith(edit)  # the edit is back in place
        assert not list(mem.glob("archive_*.md")) and not mc._leftover_tmps(mem)
    else:
        real_replace = os.replace

        def racing_replace(src, dst):
            if Path(dst).name == "MEMORY.md":
                with open(dst, "ab") as f:  # in-place write to the old inode
                    f.write(edit)
            return real_replace(src, dst)
        monkeypatch.setattr(mc.os, "replace", racing_replace)
        with pytest.raises(mc.AppliedWithProblems, match="an edit landed"):
            mc.apply(mem, stage, tmp_path / "backups")
        run_dir = next((tmp_path / "backups").iterdir())
        displaced = next(run_dir.glob("MEMORY.md.displaced.*"))
        assert displaced.read_bytes().endswith(edit)  # saved, not lost


def test_failure_mid_apply_undoes_everything(tmp_path, monkeypatch, swap_mode):
    mem = _mk_memory(tmp_path, _sample())
    stage, m = _staged(mem, tmp_path)
    second = list(m["archives"])[-1]  # written last
    real = mc.write_exclusive

    def failing(path, data, like=None, on_created=None):
        if Path(path).name == second:
            real(path, data[: len(data) // 2], like, on_created)  # a partial file exists...
            raise OSError(errno.ENOSPC, "No space left on device")  # ...then the disk fills
        return real(path, data, like, on_created)
    monkeypatch.setattr(mc, "write_exclusive", failing)
    before = _snap(mem)
    with pytest.raises(mc.Refused, match="aborted and undone"):
        mc.apply(mem, stage, tmp_path / "backups")
    assert _snap(mem) == before
    run_dir = next((tmp_path / "backups").iterdir())
    assert {p.name for p in run_dir.glob("undone_*")} == {f"undone_{n}" for n in m["archives"]}


def test_ctrl_c_mid_apply_undoes_and_propagates(tmp_path, monkeypatch):
    mem = _mk_memory(tmp_path, _sample())
    stage, m = _staged(mem, tmp_path)
    real = mc.write_exclusive
    calls = []

    def interrupted(path, data, like=None, on_created=None):
        real(path, data, like, on_created)
        if Path(path).parent == mem:
            calls.append(path)
            if len(calls) == 1:
                raise KeyboardInterrupt  # after the first archive exists in the memory dir
    monkeypatch.setattr(mc, "write_exclusive", interrupted)
    before = _snap(mem)
    with pytest.raises(KeyboardInterrupt):
        mc.apply(mem, stage, tmp_path / "backups")
    assert _snap(mem) == before


def test_interrupted_apply_is_cleaned_by_rollback(tmp_path, monkeypatch):
    """A SIGKILL mid-apply cannot run the undo: simulate it by disabling the undo."""
    mem = _mk_memory(tmp_path, _sample())
    stage, m = _staged(mem, tmp_path)
    second = list(m["archives"])[-1]  # written last
    real = mc.write_exclusive

    def killed(path, data, like=None, on_created=None):
        if Path(path).name == second:
            real(path, data[:10], like, on_created)
            raise SystemExit("killed")
        return real(path, data, like, on_created)
    monkeypatch.setattr(mc, "write_exclusive", killed)
    monkeypatch.setattr(mc, "undo_created", lambda *a: [])
    before = _snap(mem)
    with pytest.raises(SystemExit):
        mc.apply(mem, stage, tmp_path / "backups")
    assert set(_snap(mem)) - set(before) == set(m["archives"])  # leftovers, MEMORY.md untouched
    monkeypatch.undo()
    run_dir = next((tmp_path / "backups").iterdir())
    assert json.loads((run_dir / "run.json").read_text())["state"] == "pending"
    res = mc.rollback(mem, run_dir)
    assert not res["restored"] and set(res["moved"]) == set(m["archives"])
    assert _snap(mem) == before
    run2 = mc.apply(mem, stage, tmp_path / "backups")  # the stage is still good
    assert Path(run2["run_dir"]).is_dir()


def test_a_failure_after_the_swap_is_reported_and_rollback_still_works(tmp_path, monkeypatch):
    mem = _mk_memory(tmp_path, _sample())
    original = (mem / "MEMORY.md").read_bytes()
    stage, m = _staged(mem, tmp_path)
    real = mc.write_json_atomic

    def failing(path, obj):
        if Path(path).name == "run.json" and obj.get("state") == "applied":
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(path, obj)
    monkeypatch.setattr(mc, "write_json_atomic", failing)
    with pytest.raises(mc.AppliedWithProblems, match="post-apply step failed"):
        mc.apply(mem, stage, tmp_path / "backups")
    assert mc.sha256_file(mem / "MEMORY.md") == m["index_sha256"]
    monkeypatch.undo()
    run_dir = next((tmp_path / "backups").iterdir())
    assert mc.rollback(mem, run_dir)["restored"]
    assert (mem / "MEMORY.md").read_bytes() == original


def test_leftover_tmp_and_cross_device_backup_are_refused(tmp_path):
    mem = _mk_memory(tmp_path, _sample())
    stage, _ = _staged(mem, tmp_path)
    (mem / ".MEMORY.md.memory_compact.old.tmp").write_text("left by a killed run\n")
    before = _snap(mem)
    with pytest.raises(mc.Refused, match="leftover temp"):
        mc.apply(mem, stage, tmp_path / "backups")
    assert _snap(mem) == before
    os.rename(mem / ".MEMORY.md.memory_compact.old.tmp", tmp_path / "moved_out.tmp")
    shm = Path("/dev/shm")
    if shm.is_dir() and os.access(shm, os.W_OK) and os.stat(shm).st_dev != os.stat(mem).st_dev:
        before = _snap(mem)
        with pytest.raises(mc.Refused, match="same filesystem"):
            mc.apply(mem, stage, shm / f"mc_test_{os.getpid()}")
        assert _snap(mem) == before
    with pytest.raises(mc.Refused, match="backup dir must be outside"):
        mc.apply(mem, stage, mem / "backups")


def test_rollback_refuses_when_memory_edited_after_apply(tmp_path):
    mem = _mk_memory(tmp_path, _sample())
    stage, _ = _staged(mem, tmp_path)
    run = mc.apply(mem, stage, tmp_path / "backups")
    with open(mem / "MEMORY.md", "a") as f:
        f.write("- new note after compaction\n")
    before = _snap(mem)
    with pytest.raises(mc.Refused, match="edited after apply"):
        mc.rollback(mem, Path(run["run_dir"]))
    assert _snap(mem) == before


def test_crlf_bom_non_utf8_and_no_trailing_newline_round_trip(tmp_path, swap_mode):
    text = b"\xef\xbb\xbf# T\r\n" + b"".join(b"keep %d\r\n" % n for n in range(25)) + \
        b"## Tail\r\nmove \xff\xfe a\r\nmove b\r\nlast line no newline"
    mem = _mk_memory(tmp_path, text)
    n = len(text.splitlines())
    moves = [{"start": n - 2, "end": n, "archive": "archive_tail.md", "pointer": "- [Tail](archive_tail.md) — tail"}]
    stage, m = _staged(mem, tmp_path, moves)
    assert m["ok"] and m["blocks"][0]["eof_no_newline"]
    run = mc.apply(mem, stage, tmp_path / "backups")
    assert _blind_reconstruct((mem / "MEMORY.md").read_bytes(), _live_archives(mem)) == text
    mc.rollback(mem, Path(run["run_dir"]))
    assert (mem / "MEMORY.md").read_bytes() == text


def test_property_random_files_and_plans(tmp_path, swap_mode):
    rnd = random.Random(1234 if swap_mode == "exchange" else 4321)
    for it in range(60):
        base = tmp_path / f"it{it}"
        base.mkdir()
        n = rnd.randint(25, 90)
        endings = [b"\n", b"\r\n"]
        body = [b"# Title\n"] + [rnd.choice([b"- item %d" % i, b"## H%d" % i, b"", bytes([rnd.randint(0x80, 0xff)])])
                                 + rnd.choice(endings) for i in range(n - 1)]
        if rnd.random() < 0.3:
            body[-1] = body[-1].rstrip(b"\r\n")
        text = b"".join(body)
        mem = _mk_memory(base, text)
        total = len(text.splitlines())
        cuts = sorted(rnd.sample(range(2, total + 1), k=min(total - 1, rnd.choice([2, 4, 6]))))
        moves = []
        for k in range(0, len(cuts) - 1, 2):
            s, e = cuts[k], cuts[k + 1]
            if (total - (e - s)) < 20:
                continue
            moves.append({"start": s, "end": e, "archive": f"archive_p{k % 3}.md",
                          "pointer": f"- [Part {k}](archive_p{k % 3}.md) — lines {s}-{e}"})
            total -= e - s
        if not moves:
            continue
        before = _snap(mem)
        stage, m = _staged(mem, base, moves)
        assert m["ok"], m["problems"]
        run = mc.apply(mem, stage, base / "backups")
        assert _blind_reconstruct((mem / "MEMORY.md").read_bytes(), _live_archives(mem)) == text
        after = _snap(mem)
        assert {k: after[k] for k in before if k != "MEMORY.md"} == {k: v for k, v in before.items() if k != "MEMORY.md"}
        mc.rollback(mem, Path(run["run_dir"]))
        assert (mem / "MEMORY.md").read_bytes() == text
        restored = _snap(mem)
        assert set(restored) == set(before)
        assert {k: restored[k] for k in before if k != "MEMORY.md"} == {k: v for k, v in before.items() if k != "MEMORY.md"}


# ---------------------------------------------------------------------------
# CLI + static
# ---------------------------------------------------------------------------

def test_cli_exit_codes_and_no_body_lines_in_survey(tmp_path):
    mem = _mk_memory(tmp_path, _sample())
    script = str(REPO_ROOT / "scripts" / "memory_compact.py")
    run = lambda *a: subprocess.run([sys.executable, script, *a], capture_output=True, text=True, timeout=60)  # noqa: E731
    r = run("survey", "--memory-dir", str(mem), "--windows", "2")
    assert r.returncode == 0 and "something happened in session" not in r.stdout
    plan = _plan(mem, _std_moves(mem)[:1])
    r = run("stage", "--memory-dir", str(mem), "--plan", str(plan), "--stage-dir", str(tmp_path / "s"), "--max-bytes", "100")
    assert r.returncode == 3
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    r = run("stage", "--memory-dir", str(mem), "--plan", str(bad), "--stage-dir", str(tmp_path / "s2"))
    assert r.returncode == 2 and "REFUSED" in r.stderr and "Traceback" not in r.stderr
    weird = _plan(mem, [{**_std_moves(mem)[0], "archive": {"x": 1}}], path=tmp_path / "weird.json",
                  archives={"archive_session_history.md": {"description": "d"}})
    r = run("stage", "--memory-dir", str(mem), "--plan", str(weird), "--stage-dir", str(tmp_path / "s3"))
    assert r.returncode == 2 and "Traceback" not in r.stderr


def test_the_tool_has_no_delete_or_copy_calls():
    """Move-only by construction: no unlink/remove/rmtree/shutil; os.replace only in two known places."""
    src = (REPO_ROOT / "scripts" / "memory_compact.py").read_text()
    for forbidden in ("os.remove(", "os.unlink(", ".unlink(", "shutil", "rmtree(", "rmdir(", "truncate("):
        assert forbidden not in src, forbidden
    assert src.count("os.replace(") == 2  # write_json_atomic (own files) + the no-exchange swap fallback


def test_an_edit_while_the_new_index_is_written_is_caught_before_the_swap(tmp_path, monkeypatch, swap_mode):
    mem = _mk_memory(tmp_path, _sample())
    stage, _ = _staged(mem, tmp_path)
    real = mc.write_exclusive

    def editing(path, data, like=None, on_created=None):
        real(path, data, like, on_created)
        if Path(path).name.startswith(mc.TMP_PREFIX):
            with open(mem / "MEMORY.md", "a") as f:
                f.write("- edit while the tmp was written\n")
    monkeypatch.setattr(mc, "write_exclusive", editing)
    with pytest.raises(mc.Refused, match="changed just before the swap"):
        mc.apply(mem, stage, tmp_path / "backups")
    assert (mem / "MEMORY.md").read_text().endswith("- edit while the tmp was written\n")
    assert not list(mem.glob("archive_*.md")) and not mc._leftover_tmps(mem)


def test_another_memory_file_changing_during_apply_is_reported(tmp_path, monkeypatch):
    mem = _mk_memory(tmp_path, _sample())
    stage, _ = _staged(mem, tmp_path)
    real = mc.swap_index

    def swap_then_edit(*a, **kw):
        out = real(*a, **kw)
        with open(mem / "feedback_x.md", "a") as f:
            f.write("- another session wrote here\n")
        return out
    monkeypatch.setattr(mc, "swap_index", swap_then_edit)
    with pytest.raises(mc.AppliedWithProblems, match="feedback_x.md"):
        mc.apply(mem, stage, tmp_path / "backups")


def test_rollback_refuses_when_an_archive_was_edited_after_apply(tmp_path):
    mem = _mk_memory(tmp_path, _sample())
    stage, m = _staged(mem, tmp_path)
    run = mc.apply(mem, stage, tmp_path / "backups")
    name = next(iter(m["archives"]))
    with open(mem / name, "a") as f:
        f.write("- note added to the archive later\n")
    before = _snap(mem)
    with pytest.raises(mc.Refused, match="edited after apply"):
        mc.rollback(mem, Path(run["run_dir"]))
    assert _snap(mem) == before


def test_ctrl_c_right_after_the_swap_keeps_the_archives(tmp_path, monkeypatch, swap_mode):
    """The new index points at the archives: an interrupt after the swap must not undo them."""
    mem = _mk_memory(tmp_path, _sample())
    original = (mem / "MEMORY.md").read_bytes()
    before = _snap(mem)
    stage, m = _staged(mem, tmp_path)
    if swap_mode == "exchange":
        real = mc._rename_exchange

        def swap_then_interrupt(a, b):
            out = real(a, b)
            if Path(b).name != "MEMORY.md":  # the capability probe
                return out
            raise KeyboardInterrupt
        monkeypatch.setattr(mc, "_rename_exchange", swap_then_interrupt)
    else:
        real_replace = os.replace

        def replace_then_interrupt(src, dst):
            real_replace(src, dst)
            if Path(dst).name == "MEMORY.md":
                raise KeyboardInterrupt
        monkeypatch.setattr(mc.os, "replace", replace_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        mc.apply(mem, stage, tmp_path / "backups")
    monkeypatch.undo()
    assert mc.sha256_file(mem / "MEMORY.md") == m["index_sha256"]
    assert all((mem / n).is_file() for n in m["archives"])  # still there for the new index's pointers
    assert _blind_reconstruct((mem / "MEMORY.md").read_bytes(), _live_archives(mem)) == original
    run_dir = next((tmp_path / "backups").iterdir())
    assert mc.rollback(mem, run_dir)["restored"]
    assert (mem / "MEMORY.md").read_bytes() == original
    assert set(_snap(mem)) == set(before)


# ---------------------------------------------------------------------------
# review pass 2 (N1-N12)
# ---------------------------------------------------------------------------

def test_a_second_run_on_the_same_memory_dir_is_refused_while_one_holds_the_lock(tmp_path):
    import fcntl
    mem = _mk_memory(tmp_path, _sample())
    stage, _ = _staged(mem, tmp_path)
    fd = os.open(mem, os.O_RDONLY | os.O_DIRECTORY)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # another apply/rollback in progress
    try:
        before = _snap(mem)
        with pytest.raises(mc.Refused, match="another memory_compact"):
            mc.apply(mem, stage, tmp_path / "backups")
        assert _snap(mem) == before
    finally:
        os.close(fd)
    assert mc.apply(mem, stage, tmp_path / "backups")["state"] == "applied"


def test_a_move_starting_inside_a_code_fence_is_refused_even_with_even_fence_count(tmp_path):
    text = "# T\n```\ncode1\n```\ntext\n```\ncode2\n```\n" + "".join(f"k{n}\n" for n in range(30))
    mem = _mk_memory(tmp_path, text)
    moves = [{"start": 3, "end": 6, "archive": "archive_a.md", "pointer": "- [a](archive_a.md) — a"}]
    with pytest.raises(mc.Refused, match="inside a code fence"):
        mc.stage(mem, [_plan(mem, moves)], tmp_path / "s", 20000, 180)


def test_a_stage_that_changes_nothing_is_not_ok(tmp_path):
    ptr = "- [a](archive_a.md) — the same line"
    text = "# T\n" + ptr + "\n" + "".join(f"k{n}\n" for n in range(30))
    mem = _mk_memory(tmp_path, text)
    stage, m = _staged(mem, tmp_path, [{"start": 2, "end": 2, "archive": "archive_a.md", "pointer": ptr}])
    assert not m["ok"] and any("no change" in p for p in m["problems"])


def test_rollback_refuses_a_tampered_run_json(tmp_path):
    mem = _mk_memory(tmp_path, _sample())
    stage, _ = _staged(mem, tmp_path)
    run_dir = Path(mc.apply(mem, stage, tmp_path / "backups")["run_dir"])
    good = json.loads((run_dir / "run.json").read_text())
    before = _snap(mem)
    for bad in ({**good, "tmp_prefix": ""}, {**good, "created": {"../x": "0" * 64}}, {**good, "run_id": "../x"},
                {k: v for k, v in good.items() if k != "created"}):
        (run_dir / "run.json").write_text(json.dumps(bad))
        with pytest.raises(mc.Refused, match="run.json"):
            mc.rollback(mem, run_dir)
        assert _snap(mem) == before


def test_a_rollback_rerun_leaves_a_file_that_reuses_an_archive_name_alone(tmp_path):
    mem = _mk_memory(tmp_path, _sample())
    stage, m = _staged(mem, tmp_path)
    run_dir = Path(mc.apply(mem, stage, tmp_path / "backups")["run_dir"])
    mc.rollback(mem, run_dir)
    assert json.loads((run_dir / "run.json").read_text())["state"] == "rolled_back"
    name = next(iter(m["archives"]))
    (mem / name).write_text("a new file that happens to reuse the name\n")
    res = mc.rollback(mem, run_dir)
    assert res["left_alone"] == [name] and (mem / name).read_text().startswith("a new file")


def test_fallback_keeps_the_old_inode_so_a_late_fd_writer_is_not_lost(tmp_path, monkeypatch):
    monkeypatch.setattr(mc, "_EXCHANGE_DISABLED", True)
    monkeypatch.setattr(mc, "ALLOW_FALLBACK_DEFAULT", True)
    mem = _mk_memory(tmp_path, _sample())
    stage, _ = _staged(mem, tmp_path)
    holder = open(mem / "MEMORY.md", "ab")  # another session holds the old file open
    try:
        run = mc.apply(mem, stage, tmp_path / "backups")
        holder.write(b"- written after the swap through an old handle\n")
        holder.flush()
    finally:
        holder.close()
    assert Path(run["displaced"]).read_bytes().endswith(b"- written after the swap through an old handle\n")


def test_a_write_into_the_displaced_original_before_apply_returns_is_reported(tmp_path, monkeypatch, swap_mode):
    mem = _mk_memory(tmp_path, _sample())
    stage, _ = _staged(mem, tmp_path)
    real = mc.write_json_atomic

    def late_writer(path, obj):
        if Path(path).name == "run.json" and obj.get("state") == "applied":
            with open(obj["displaced"], "ab") as f:  # an fd-holder writing into the old file
                f.write(b"- late write\n")
        return real(path, obj)
    monkeypatch.setattr(mc, "write_json_atomic", late_writer)
    with pytest.raises(mc.AppliedWithProblems, match="write landed in the displaced original"):
        mc.apply(mem, stage, tmp_path / "backups")


def test_without_rename_exchange_apply_refuses_unless_the_fallback_is_allowed(tmp_path, monkeypatch):
    monkeypatch.setattr(mc, "_EXCHANGE_DISABLED", True)
    mem = _mk_memory(tmp_path, _sample())
    stage, m = _staged(mem, tmp_path)
    before = _snap(mem)
    with pytest.raises(mc.Refused, match="RENAME_EXCHANGE"):
        mc.apply(mem, stage, tmp_path / "backups")
    assert _snap(mem) == before
    script = str(REPO_ROOT / "scripts" / "memory_compact.py")
    run = mc.apply(mem, stage, tmp_path / "backups", allow_fallback=True)
    assert mc.sha256_file(mem / "MEMORY.md") == m["index_sha256"] and Path(run["run_dir"]).is_dir()
    assert "--allow-fallback-swap" in subprocess.run([sys.executable, script, "apply", "-h"],
                                                       capture_output=True, text=True, timeout=60).stdout
