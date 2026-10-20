"""R-32 (S215): memory write-site hygiene.

(a) core.py stores a structured calibration description (not assistant prose) on each confidence prediction.
(b) WorkingMemory.add() coerces priority to a finite float before the entry is persisted, so the
    working_memory.json snapshot keeps numeric types.

Nothing here touches live state: WORKING_MEMORY_PATH is redirected to tmp_path, and the flusher thread
is stopped at teardown. The calibration check reads core.py as text; it does not import or run the agent.
"""

import json
import re
from pathlib import Path

import pytest

from agent.memory import working_memory as wm_mod

CORE_PY = Path(__file__).resolve().parents[1] / "agent" / "core.py"


@pytest.fixture
def wm(tmp_path, monkeypatch):
    monkeypatch.setattr(wm_mod, "WORKING_MEMORY_PATH", tmp_path / "working_memory.json")
    mem = wm_mod.WorkingMemory()
    yield mem
    mem.stop()


def _persisted(path):
    return json.loads(Path(path).read_text())


def test_string_priority_is_coerced_to_float_before_persist(wm, tmp_path):
    entry_id = wm.add(content="user prefers concise trade summaries", domain="trading", priority="0.7")
    assert entry_id
    assert isinstance(wm.get(entry_id).priority, float)
    assert wm.get(entry_id).priority == pytest.approx(0.7)
    wm.flush()
    on_disk = _persisted(tmp_path / "working_memory.json")
    (row,) = on_disk.values()
    assert isinstance(row["priority"], float)
    assert row["priority"] == pytest.approx(0.7)


def test_unparseable_priority_falls_back_to_default(wm):
    entry_id = wm.add(content="a note with a garbage priority value", domain="general", priority="high")
    assert wm.get(entry_id).priority == 0.5


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "nan"])
def test_non_finite_priority_falls_back_to_default(wm, bad):
    entry_id = wm.add(content=f"a note with priority {bad!r} here", domain="general", priority=bad)
    assert wm.get(entry_id).priority == 0.5


def test_numeric_priority_is_unchanged(wm):
    entry_id = wm.add(content="a plain numeric priority entry", domain="general", priority=0.25)
    assert wm.get(entry_id).priority == 0.25


def test_calibration_description_is_a_structured_one_line_string():
    src = CORE_PY.read_text(encoding="utf-8")
    lines = src.splitlines()
    idx = [i for i, ln in enumerate(lines) if 'description=f"{task_type} conf={_conf}",' in ln]
    assert len(idx) == 1, idx
    # The adjacent record_prediction argument lines are untouched.
    block = "\n".join(lines[idx[0] - 6: idx[0] + 2])
    assert 'stated_confidence=_conf,' in block
    assert 'domain=task_type,' in block
    assert '_pred_id = f"cal_{hashlib.md5(final_text[:100].encode()).hexdigest()[:12]}"' in block
    assert "description=final_text[:200]" not in src
    assert not re.search(r"description=final_text", src)
