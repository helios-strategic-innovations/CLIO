#!/usr/bin/env python3
"""RECALL-1 contract tests for serving-prompt episodic recall.

All episodic fixtures live under pytest's tmp_path. These tests must never read
or mutate the real CLIO memory store.

Vocabulary provenance: V2-CLIO RF capture
docs/aster/screen_evidence/20260823T_v2clio_recall1_candidate_return_settled_capture/
identified production rows as trade|trade_idea_outcome plus memory|general,
memory|trade_decision, promoted|general, and
promoted_on_expire|trade_decision.
"""

import json
import os
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

try:
    import fcntl  # noqa: F401
except ModuleNotFoundError:
    sys.modules["fcntl"] = types.SimpleNamespace(
        LOCK_EX=1,
        LOCK_SH=2,
        LOCK_UN=8,
        flock=lambda *_args, **_kwargs: None,
    )

import agent.memory.episodic as episodic_mod  # noqa: E402
from agent.memory.episodic import EpisodicMemory  # noqa: E402
from agent.memory.memory_engine import MemoryEngine  # noqa: E402


def _episodic_store(monkeypatch, tmp_path: Path) -> EpisodicMemory:
    data_dir = tmp_path / "memory"
    monkeypatch.setattr(episodic_mod, "DATA_DIR", str(data_dir))
    return EpisodicMemory()


def _write_trade_heavy_august_distribution(store: EpisodicMemory) -> None:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    month_file = Path(store._dir) / f"episodic_{now.strftime('%Y-%m')}.jsonl"
    month_file.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    for i in range(62):
        entries.append({
            "timestamp": (now - timedelta(seconds=i)).isoformat(),
            "event_type": "trade",
            "content": f"trade outcome {i:03d}: UVIX spread bookkeeping result",
            "category": "trade_idea_outcome",
            "domain": "trade_idea_outcome",
            "outcome": "neutral",
            "importance": 0.7,
            "metadata": {"fixture": "recall1_trade_heavy"},
        })
    memory_shapes = [
        ("memory", "general"),
        ("memory", "trade_decision"),
        ("promoted", "general"),
        ("promoted_on_expire", "trade_decision"),
    ]
    for i in range(19):
        event_type, category = memory_shapes[i % len(memory_shapes)]
        entries.append({
            "timestamp": (now - timedelta(minutes=30, seconds=i)).isoformat(),
            "event_type": event_type,
            "content": f"Dom conversation memory {i:03d}: use source-linked HOLD wording",
            "category": category,
            "domain": category,
            "outcome": None,
            "importance": 0.8,
            "metadata": {"source": "telegram_direct", "fixture": "recall1_memory"},
        })

    with open(month_file, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def test_serving_prompt_prioritizes_dom_memory_over_trade_outcomes(monkeypatch, tmp_path):
    store = _episodic_store(monkeypatch, tmp_path)
    _write_trade_heavy_august_distribution(store)

    engine = MemoryEngine.__new__(MemoryEngine)
    engine.episodic = store

    context = MemoryEngine._get_episodic_context(engine, max_chars=500)

    assert "Dom conversation memory" in context, (
        "serving prompt should surface memory-type conversation events even when "
        "newer trade outcomes dominate the episodic store"
    )
    assert context.splitlines()[0].startswith("- memory: Dom conversation memory")
    assert "- promoted: Dom conversation memory" in context
    assert "- promoted_on_expire: Dom conversation memory" in context


def test_default_get_recent_limit_one_keeps_newest_entry(monkeypatch, tmp_path):
    store = _episodic_store(monkeypatch, tmp_path)
    _write_trade_heavy_august_distribution(store)

    recent = store.get_recent(limit=1)

    assert len(recent) == 1
    assert recent[0]["event_type"] == "trade"
    assert recent[0]["category"] == "trade_idea_outcome"
    assert recent[0]["content"].startswith("trade outcome 000")


def test_serving_prompt_budget_ends_on_whole_line(monkeypatch, tmp_path):
    store = _episodic_store(monkeypatch, tmp_path)
    _write_trade_heavy_august_distribution(store)

    engine = MemoryEngine.__new__(MemoryEngine)
    engine.episodic = store

    context = MemoryEngine._get_episodic_context(engine, max_chars=500)

    assert len(context) <= 500
    assert context == "\n".join(context.splitlines())
    assert all(line.startswith("- ") for line in context.splitlines())
    assert context.splitlines()[-1].endswith(("wording", "result"))


def test_health_and_liveness_still_call_get_recent_with_limit_one():
    class StubWorking:
        def stats(self):
            return {"total_entries": 0}

        def persist_health(self):
            return {"broken": False}

    class StubEpisodic:
        def __init__(self):
            self.calls = []

        def get_recent(self, limit=1):
            self.calls.append({"limit": limit})
            return [{"timestamp": datetime.now(timezone.utc).isoformat()}]

        def stats(self):
            return {"total_size_kb": 1.0}

    class StubProcedural:
        def stats(self):
            return {"total_patterns": 0}

        def get_all_patterns(self):
            return []

    class StubMeta:
        def get_counts(self):
            return {"corrections": 0, "gaps": 0}

    engine = MemoryEngine.__new__(MemoryEngine)
    engine.working = StubWorking()
    engine.episodic = StubEpisodic()
    engine.procedural = StubProcedural()
    engine.meta = StubMeta()
    engine._last_working_probe = {"status": "LIVE"}

    report = MemoryEngine._collect_liveness(engine)
    health = MemoryEngine.health_check(engine)

    assert report["degraded"] is False
    assert health["overall_healthy"] is True
    assert engine.episodic.calls == [{"limit": 1}, {"limit": 1}]
