"""
R-24: a Living Perspective synthesis checkpoint (every SYNTHESIS_EVERY_N_CYCLES)
that lands on an early-exit cycle must stay due and run on the next completed cycle.

Everything external is faked: search, fetch, analysis, KB save, synthesis, budget,
bandit, archival. Learning state is written only to tmp_path. No network, no real
data/ or state/ paths.
"""

import asyncio
import json
from datetime import datetime as _real_datetime
from zoneinfo import ZoneInfo

import pytest

import agent.integrations.news_update as _news_update
import agent.integrations.seal_category_bandit as _bandit_mod
import agent.integrations.source_authority as _source_authority
import agent.tools.knowledge_tools as _knowledge_tools
from agent.integrations import learning_engine as le

_FROZEN_ET = _real_datetime(2026, 10, 12, 10, 0, tzinfo=ZoneInfo("America/New_York"))


class _FrozenDT(_real_datetime):
    @classmethod
    def now(cls, tz=None):
        return _FROZEN_ET.astimezone(tz) if tz is not None else _FROZEN_ET.replace(tzinfo=None)


_FAKE_CATS = [
    {"name": "fake_a", "knowledge_category": "trading_market_analysis", "queries": ["alpha query one"]},
    {"name": "fake_b", "knowledge_category": "trading_market_analysis", "queries": ["beta query two"]},
]

_RESULTS = [{"url": "https://example.test/a", "title": "A", "snippet": "s"}]


class _FakeSched:
    def get_mode(self):
        return "normal", {}


class _FakeBandit:
    def select_category(self, regime=None, base_weights=None):
        return ""


class _Harness:
    def __init__(self, monkeypatch, tmp_path):
        self.search_on = True
        self.search_calls = 0
        self.synth_calls = 0
        self.state_file = tmp_path / "learning_state.json"
        monkeypatch.setattr(le, "LEARNING_STATE_FILE", str(self.state_file))
        monkeypatch.setattr(le, "LEARNING_CATEGORIES", _FAKE_CATS)
        monkeypatch.setattr(le, "datetime", _FrozenDT)
        monkeypatch.setenv("MOLTBOT_COST_AWARE_ENABLED", "0")
        monkeypatch.setattr(le, "_category_cooldowns", {})
        monkeypatch.setattr(le, "_query_failure_counts", {})
        monkeypatch.setattr(le, "_update_category_weights", lambda state, cat, analysis: None)
        monkeypatch.setattr(le, "_get_current_market_regime", lambda default=None: "CHOPPY")
        monkeypatch.setattr(le, "_get_learning_scheduler", lambda: _FakeSched())
        monkeypatch.setattr(le, "_extract_knowledge_gaps", lambda: [])
        monkeypatch.setattr(le, "_get_portfolio_symbols", lambda: [])

        async def _no_adaptive(*a, **k):
            return None

        async def _budget_ok(*a, **k):
            return True, "OK"

        async def _search(query, max_results=0):
            self.search_calls += 1
            return list(_RESULTS) if self.search_on else []

        async def _fetch(url, max_chars=0):
            return "page body text"

        async def _analyze(content, category, urls, pre_dedup=False):
            return {"quality_score": 9}

        async def _save(analysis, cat, kcat, urls):
            return True

        async def _synth(context):
            self.synth_calls += 1

        monkeypatch.setattr(le, "_generate_adaptive_query", _no_adaptive)
        monkeypatch.setattr(le, "_preflight_budget_check", _budget_ok)
        monkeypatch.setattr(le, "_load_url_history", lambda: {})
        monkeypatch.setattr(le, "_save_url_history", lambda h: None)
        monkeypatch.setattr(le, "_cleanup_url_history", lambda h: h)
        monkeypatch.setattr(le, "_is_url_seen", lambda u, h: False)
        monkeypatch.setattr(le, "_record_url", lambda *a, **k: None)
        monkeypatch.setattr(le, "search_web", _search)
        monkeypatch.setattr(le, "fetch_page_content", _fetch)
        monkeypatch.setattr(le, "_analyze_with_claude", _analyze)
        monkeypatch.setattr(le, "_save_analysis_to_knowledge", _save)
        monkeypatch.setattr(le, "run_synthesis", _synth)
        monkeypatch.setattr(_news_update, "_url_is_stale", lambda u: False)
        monkeypatch.setattr(_news_update, "_snippet_is_stale", lambda t, s: False)
        monkeypatch.setattr(_source_authority, "filter_and_rank_results", lambda r: r)
        monkeypatch.setattr(_knowledge_tools, "run_knowledge_archival", lambda: {"archived": 0})
        monkeypatch.setattr(_bandit_mod, "get_category_bandit", lambda: _FakeBandit())

    def seed(self, cycle_count: int):
        self.state_file.write_text(json.dumps({
            "category_index": 0,
            "query_indices": {},
            "cycle_count": cycle_count,
            "last_run": "",
        }))

    def run(self, results: bool):
        self.search_on = results
        asyncio.run(le.run_learning_cycle(None))

    def state(self) -> dict:
        return json.loads(self.state_file.read_text())


@pytest.fixture
def h(monkeypatch, tmp_path):
    return _Harness(monkeypatch, tmp_path)


class TestSynthesisCheckpointDue:
    def test_early_exit_on_checkpoint_sets_flag_without_synthesis(self, h):
        h.seed(7)
        h.run(results=False)  # no search results -> early return after counter advanced
        st = h.state()
        assert st["cycle_count"] == 8
        assert st["synthesis_due"] is True
        assert h.synth_calls == 0

    def test_next_completed_cycle_runs_synthesis_once_and_clears_flag(self, h):
        h.seed(7)
        h.run(results=False)
        h.run(results=True)
        st = h.state()
        assert st["cycle_count"] == 9
        assert h.synth_calls == 1
        assert st["synthesis_due"] is False

    def test_synthesis_not_repeated_on_following_completed_cycle(self, h):
        h.seed(7)
        h.run(results=False)
        h.run(results=True)
        h.run(results=True)
        assert h.synth_calls == 1
        assert h.state()["cycle_count"] == 10

    def test_flag_survives_several_early_exits_then_runs_once(self, h):
        h.seed(7)
        h.run(results=False)  # 8, due
        h.run(results=False)  # 9, still due
        assert h.synth_calls == 0
        assert h.state()["synthesis_due"] is True
        h.run(results=True)   # 10, completed -> runs
        assert h.synth_calls == 1
        assert h.state()["synthesis_due"] is False

    def test_normal_checkpoint_cycle_still_synthesises(self, h):
        h.seed(7)
        h.run(results=True)   # completed cycle lands on 8
        st = h.state()
        assert st["cycle_count"] == 8
        assert h.synth_calls == 1
        assert st["synthesis_due"] is False

    def test_non_checkpoint_completed_cycle_does_not_synthesise(self, h):
        h.seed(3)
        h.run(results=True)
        assert h.state()["cycle_count"] == 4
        assert h.synth_calls == 0
        assert not h.state().get("synthesis_due")


class TestPreflightHonoursDueFlag:
    def _budget(self, monkeypatch):
        names: list[str] = []

        class _Tracker:
            async def check_budget(self, name):
                names.append(name)
                return True, "OK"

        monkeypatch.setattr(le, "budget_tracker", _Tracker())
        return names

    def test_due_flag_adds_synthesis_budget_check(self, monkeypatch):
        names = self._budget(monkeypatch)
        asyncio.run(le._preflight_budget_check(9, True))
        assert "learning_synthesis" in names

    def test_no_flag_off_checkpoint_skips_synthesis_budget_check(self, monkeypatch):
        names = self._budget(monkeypatch)
        asyncio.run(le._preflight_budget_check(9, False))
        assert "learning_synthesis" not in names
