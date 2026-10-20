"""
Tests for Hydrinia-bridge write gating in agent/tools/knowledge_tools.py,
agent/memory/memory_engine.py, and agent/personality/engine.py (B1 scope).

Hydrinia is permanently off (operator direction, 2026-09-24). Every write into
its (frozen) bridge directories must be a no-op when
CLIO_HYDRINIA_BRIDGE_ENABLED is unset/false, and must still work when the
operator explicitly re-enables it. CLIO's own writes/internal flushes are
never affected by the gate.

knowledge_tools.py's writes are redirected into tmp_path — never the real
data/ tree. The memory_engine/personality "bridge on" tests likewise monkeypatch
their mirror-dir module constants (HYDRINIA_HOT_CONTEXT_MIRROR_DIR /
HYDRINIA_PERSONALITY_MIRROR_DIR) to a tmp_path — /home/agent/tradingbot/ is a
real, live directory on the production VPS and must never be touched by a test.
"""

import json
import os

import pytest

import agent.tools.knowledge_tools as kt
import agent.memory.memory_engine as me_mod
import agent.personality.engine as pe_mod


# ---------------------------------------------------------------------------
# knowledge_tools.py — mirror gating
# ---------------------------------------------------------------------------

CATEGORY = "trading_market_analysis"


@pytest.fixture
def kt_paths(tmp_path, monkeypatch):
    base = tmp_path / "kb"
    mirror = tmp_path / "mirror"
    monkeypatch.setattr(kt, "KNOWLEDGE_BASE_DIR", str(base))
    monkeypatch.setattr(kt, "KNOWLEDGE_INDEX_FILE", str(base / "knowledge_index.json"))
    monkeypatch.setattr(kt, "KNOWLEDGE_ARCHIVE_INDEX_FILE", str(base / "knowledge_index_archive.json"))
    monkeypatch.setattr(kt, "_INDEX_LOCK_FILE", str(base / "knowledge_index.json.lock"))
    monkeypatch.setattr(kt, "TRADINGBOT_MIRROR_DIR", str(mirror))
    monkeypatch.setattr(kt, "URGENT_MIRROR_DIR", str(mirror / "urgent"))
    monkeypatch.setattr(kt, "CURRENT_PRIORS_PATH", str(base / "current_priors.md"))
    # Stub fire-and-forget integrations the save path may lazily import, so
    # this stays a pure unit test of the mirror gate (matches the pattern in
    # tests/test_knowledge_save_deadlock.py).
    import sys
    import types
    cd_stub = types.ModuleType("agent.integrations.contradiction_detector")
    cd_stub.check_new_entry_contradictions = lambda entry: []
    monkeypatch.setitem(sys.modules, "agent.integrations.contradiction_detector", cd_stub)
    seal_stub = types.ModuleType("agent.tools.seal_knowledge_enhancer")
    seal_stub.SEAL_ENABLED = lambda: False
    seal_stub.SEAL_LOG_ONLY = lambda: True
    seal_stub.get_implication_engine = lambda: None
    monkeypatch.setitem(sys.modules, "agent.tools.seal_knowledge_enhancer", seal_stub)
    return base, mirror


import asyncio


def _run(coro):
    return asyncio.run(coro)


class TestKnowledgeMirrorGate:
    def test_ensure_dirs_skips_mirror_when_bridge_off(self, kt_paths, monkeypatch):
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
        base, mirror = kt_paths
        kt._ensure_dirs()
        assert os.path.isdir(base)
        assert not os.path.exists(mirror)

    def test_ensure_dirs_creates_mirror_when_bridge_on(self, kt_paths, monkeypatch):
        monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
        base, mirror = kt_paths
        kt._ensure_dirs()
        assert os.path.isdir(mirror)

    def test_save_knowledge_no_mirror_file_when_bridge_off(self, kt_paths, monkeypatch):
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
        base, mirror = kt_paths
        result = _run(kt.handle_save_knowledge({
            "title": "Bridge-off mirror probe",
            "category": CATEGORY,
            "content": "no prices here",
            "tags": ["probe"],
            "source": "test",
        }))
        assert result.startswith("Saved:")
        # CLIO's own KB entry always lands...
        rel_dir = kt.CATEGORY_DIRS[CATEGORY]
        kb_files = list((base / rel_dir).glob("*.json")) if (base / rel_dir).exists() else []
        assert kb_files, "entry should be written to CLIO's own knowledge base"
        # ...but nothing is mirrored to the retired Hydrinia bridge dir.
        assert not os.path.exists(mirror)

    def test_save_knowledge_writes_mirror_file_when_bridge_on(self, kt_paths, monkeypatch):
        monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
        base, mirror = kt_paths
        result = _run(kt.handle_save_knowledge({
            "title": "Bridge-on mirror probe",
            "category": CATEGORY,
            "content": "no prices here",
            "tags": ["probe"],
            "source": "test",
        }))
        assert result.startswith("Saved:")
        assert "Mirrored to:" in result
        rel_dir = kt.CATEGORY_DIRS[CATEGORY]
        mirror_subdir = rel_dir.split("/", 1)[1] if "/" in rel_dir else rel_dir
        mirror_files = list((mirror / mirror_subdir).glob("*.json"))
        assert mirror_files, "entry should be mirrored to the Hydrinia bridge dir when enabled"

    def test_update_priors_no_mirror_when_bridge_off(self, kt_paths, monkeypatch):
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
        base, mirror = kt_paths
        os.makedirs(base, exist_ok=True)
        result = _run(kt.handle_update_priors({"content": "SPY gamma at 580"}))
        assert "Operational priors updated" in result
        assert os.path.isfile(base / "current_priors.md")
        assert not os.path.exists(mirror / "current_priors.md")

    def test_update_priors_mirrors_when_bridge_on(self, kt_paths, monkeypatch):
        monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
        base, mirror = kt_paths
        os.makedirs(base, exist_ok=True)
        os.makedirs(mirror, exist_ok=True)  # pre-existing behavior: no _ensure_dirs() call in this path
        result = _run(kt.handle_update_priors({"content": "SPY gamma at 580"}))
        assert "Operational priors updated" in result
        assert os.path.isfile(mirror / "current_priors.md")


# ---------------------------------------------------------------------------
# memory_engine.py — export_hot_context() mirror gating
# ---------------------------------------------------------------------------

class _StubTier:
    def __init__(self):
        self._dirty = False
        self.flushed = False

    def flush(self):
        self.flushed = True

    def stats(self):
        return {}

    def _purge_expired(self):
        pass


class TestMemoryEngineMirrorGate:
    def _make_engine(self):
        engine = object.__new__(me_mod.MemoryEngine)
        engine.working = _StubTier()
        engine.meta = _StubTier()
        engine.procedural = _StubTier()
        engine.episodic = _StubTier()
        engine.get_full_context = lambda: "stub context"
        engine._collect_liveness = lambda: {}
        return engine

    def test_mirror_skipped_when_bridge_off(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
        # Redirect the mirror dir constant to tmp_path regardless — belt and
        # braces alongside the os.makedirs guard below (never touch the real
        # /home/agent/tradingbot/ tree from a test).
        monkeypatch.setattr(me_mod, "HYDRINIA_HOT_CONTEXT_MIRROR_DIR", str(tmp_path / "mirror"))
        engine = self._make_engine()
        # Sanity: internal tier flush still runs even though the mirror is skipped.
        engine.export_hot_context()
        assert engine.working.flushed is True
        assert engine.meta.flushed is True
        assert engine.procedural.flushed is True
        # Assert the write is never attempted: patch os.makedirs to explode if
        # called with the (redirected) mirror dir.
        def _boom(path, *a, **kw):
            if "mirror" in str(path):
                raise AssertionError("mirror dir must not be created when bridge is off")
            return real_makedirs(path, *a, **kw)

        real_makedirs = os.makedirs
        monkeypatch.setattr(me_mod.os, "makedirs", _boom)
        engine.export_hot_context()  # must not raise

    def test_mirror_written_when_bridge_on(self, tmp_path, monkeypatch):
        """When the bridge is on, export_hot_context() reaches its mirror write
        path. The mirror-dir module constant is redirected to tmp_path — the
        real /home/agent/tradingbot/ tree is a live directory on the production
        VPS and must never be written to by a test."""
        monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
        mirror_dir = str(tmp_path / "moltbot_knowledge" / "working_memory")
        monkeypatch.setattr(me_mod, "HYDRINIA_HOT_CONTEXT_MIRROR_DIR", mirror_dir)
        mirror_file = os.path.join(mirror_dir, "hot_context.json")

        engine = self._make_engine()
        engine.export_hot_context()
        assert os.path.isfile(mirror_file), "mirror file should be written when bridge is on"
        with open(mirror_file) as f:
            written = json.load(f)
        assert written.get("full_context") == "stub context"


# ---------------------------------------------------------------------------
# personality/engine.py — export_snapshot() gating
# ---------------------------------------------------------------------------

class TestPersonalityExportSnapshotGate:
    def _make_engine(self):
        engine = object.__new__(pe_mod.PersonalityEngine)
        return engine

    def test_export_snapshot_noop_when_bridge_off(self, monkeypatch):
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
        engine = self._make_engine()

        def _boom():
            raise AssertionError("registry must not be touched when bridge is off")

        # If export_snapshot proceeds past the gate it will try self.registry,
        # which doesn't exist on this bare instance -> AttributeError. The gate
        # must return before that point, so no exception at all.
        engine.export_snapshot()  # must not raise

    def test_export_snapshot_writes_when_bridge_on(self, tmp_path, monkeypatch):
        """When the bridge is on, export_snapshot() reaches its mirror write
        path. The mirror-dir module constant is redirected to tmp_path — the
        real /home/agent/tradingbot/ tree is a live directory on the production
        VPS and must never be written to by a test."""
        monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
        engine = self._make_engine()

        class _FakeConfig:
            enabled = True

        class _FakeRegistry:
            def get_confirmed_traits(self):
                return []

            def get_proposed_traits(self):
                return []

        engine.registry = _FakeRegistry()
        engine.get_snapshot_context = lambda: "stub"
        engine.config = _FakeConfig()
        monkeypatch.setattr(
            "agent.personality.config.get_personality_config", lambda: _FakeConfig(),
        )

        snapshot_dir = str(tmp_path / "moltbot_knowledge" / "personality")
        monkeypatch.setattr(pe_mod, "HYDRINIA_PERSONALITY_MIRROR_DIR", snapshot_dir)
        snapshot_file = os.path.join(snapshot_dir, "personality_snapshot.json")

        engine.export_snapshot()
        assert os.path.isfile(snapshot_file), "snapshot file should be written when bridge is on"
        with open(snapshot_file) as f:
            written = json.load(f)
        assert written.get("system_prompt_context") == "stub"
