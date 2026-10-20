"""
S215 R-30(b): scripts/kb_orphan_report.py is READ-ONLY. Tested only on synthetic
tmp_path trees; it never reads or writes data/knowledge.
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "kb_orphan_report", str(_REPO / "scripts" / "kb_orphan_report.py"))
kor = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(kor)


def _touch(root: Path, rel: str, body: str = "{}") -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "knowledge"
    root.mkdir()
    live = {"version": 1, "entries": [
        {"id": "a", "file_path": "trading/a/a.json"},
        {"id": "b", "file_path": "trading/b/b.json"},  # missing on disk
    ]}
    archive = {"version": 1, "entries": [
        {"id": "c", "file_path": "personal/c/c.json"},
        {"id": "d", "file_path": "archive/d.json"},  # missing on disk
    ]}
    (root / "knowledge_index.json").write_text(json.dumps(live))
    (root / "knowledge_index_archive.json").write_text(json.dumps(archive))
    _touch(root, "trading/a/a.json")
    _touch(root, "trading/x/orphan1.json")
    _touch(root, "qa_verification/q1.json")
    _touch(root, "qa_verification/q2.json")
    _touch(root, "personal/c/c.json")
    _touch(root, "trade_ideas/ti_1.json")          # trade-idea artifact: skipped by default
    _touch(root, "learning_state.json")            # root-level state: never an entry
    _touch(root, "trading/x/notes.txt", "not json")  # non-json: ignored
    return root


def _snapshot(root: Path) -> dict:
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            p = os.path.join(dirpath, name)
            st = os.lstat(p)
            out[p] = (st.st_mode, st.st_size, st.st_mtime_ns, st.st_ino)
    return out


class TestBuildReport:
    def test_counts_and_grouping(self, tree):
        rep = kor.build_report(str(tree))
        assert rep["live_entries"] == 2
        assert rep["archive_entries"] == 2
        assert rep["files_on_disk"] == 5
        assert rep["orphan_files"] == 3
        assert rep["orphans_by_top_dir"] == {"qa_verification": 2, "trading": 1}
        assert sorted(rep["orphan_samples"]) == [
            "qa_verification/q1.json", "qa_verification/q2.json", "trading/x/orphan1.json",
        ]

    def test_missing_targets_per_index(self, tree):
        rep = kor.build_report(str(tree))
        assert rep["missing_targets"] == {
            "live": ["trading/b/b.json"],
            "archive": ["archive/d.json"],
        }

    def test_trade_ideas_included_only_when_asked(self, tree):
        rep = kor.build_report(str(tree), include_trade_ideas=True)
        assert rep["files_on_disk"] == 6
        assert rep["orphans_by_top_dir"]["trade_ideas"] == 1

    def test_archive_indexed_file_is_not_orphan(self, tree):
        rep = kor.build_report(str(tree))
        assert "personal/c/c.json" not in rep["orphan_samples"]

    def test_absolute_file_path_in_index_is_relativised(self, tree):
        live = json.loads((tree / "knowledge_index.json").read_text())
        live["entries"].append({"id": "e", "file_path": str(tree / "qa_verification" / "q1.json")})
        (tree / "knowledge_index.json").write_text(json.dumps(live))
        rep = kor.build_report(str(tree))
        assert "qa_verification/q1.json" not in rep["orphan_samples"]
        assert rep["orphan_files"] == 2

    def test_missing_indexes_mean_no_entries(self, tmp_path):
        root = tmp_path / "bare"
        _touch(root, "trading/x/only.json")
        rep = kor.build_report(str(root))
        assert rep["live_entries"] == 0 and rep["archive_entries"] == 0
        assert rep["orphan_files"] == 1

    def test_corrupt_index_raises(self, tree):
        (tree / "knowledge_index.json").write_text("{not json")
        with pytest.raises(kor.IndexFileError):
            kor.build_report(str(tree))

    def test_entries_not_a_list_raises(self, tree):
        (tree / "knowledge_index.json").write_text(json.dumps({"entries": "oops"}))
        with pytest.raises(kor.IndexFileError):
            kor.build_report(str(tree))


class TestMain:
    def test_prints_summary_and_returns_zero(self, tree, capsys):
        assert kor.main(["--root", str(tree)]) == 0
        out = capsys.readouterr().out
        assert "orphans (in neither index): 3" in out
        assert "qa_verification: 2" in out
        assert "live index entries whose file is missing: 1" in out
        assert "archive index entries whose file is missing: 1" in out

    def test_missing_root_returns_two(self, tmp_path, capsys):
        assert kor.main(["--root", str(tmp_path / "nope")]) == 2
        assert "not a directory" in capsys.readouterr().err

    def test_corrupt_index_returns_two_not_false_orphans(self, tree, capsys):
        (tree / "knowledge_index_archive.json").write_text("][")
        assert kor.main(["--root", str(tree)]) == 2
        captured = capsys.readouterr()
        assert "unreadable index" in captured.err
        assert "orphans" not in captured.out


class TestReadOnly:
    def test_writes_nothing_on_healthy_tree(self, tree, capsys):
        before = _snapshot(tree)
        assert kor.main(["--root", str(tree), "--include-trade-ideas", "--limit", "99"]) == 0
        capsys.readouterr()
        assert _snapshot(tree) == before

    def test_writes_nothing_on_corrupt_tree(self, tree, capsys):
        (tree / "knowledge_index.json").write_text("{bad")
        before = _snapshot(tree)
        assert kor.main(["--root", str(tree)]) == 2
        capsys.readouterr()
        assert _snapshot(tree) == before

    def test_no_lock_or_tmp_files_created(self, tree, capsys):
        kor.main(["--root", str(tree)])
        capsys.readouterr()
        names = {p.name for p in tree.rglob("*")}
        assert not any(n.endswith((".lock", ".tmp")) for n in names)
