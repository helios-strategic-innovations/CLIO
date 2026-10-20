"""
S215 R-30(a) + (c).

(a) knowledge index entries get an additive `quality_imputed: true` ONLY when the 5.0
    default was applied because no quality_score was supplied.
(c) the SEAL implication (IMPL) generator prompt carries the no-unsourced-numbers rule.

Every knowledge path is redirected into tmp_path; nothing touches data/knowledge.
"""

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

from agent.tools import knowledge_tools as kt

# Load the real enhancer by file path, private name, NOT via sys.modules: another test
# module (test_knowledge_save_deadlock) installs a permanent stub under that name.
_SKE_PATH = Path(__file__).resolve().parents[1] / "agent" / "tools" / "seal_knowledge_enhancer.py"
_SKE_SPEC = importlib.util.spec_from_file_location("_real_seal_knowledge_enhancer_s215", str(_SKE_PATH))
ske = importlib.util.module_from_spec(_SKE_SPEC)
_SKE_SPEC.loader.exec_module(ske)


@pytest.fixture
def kb(monkeypatch, tmp_path):
    base = tmp_path / "knowledge"
    base.mkdir()
    monkeypatch.setattr(kt, "KNOWLEDGE_BASE_DIR", str(base))
    monkeypatch.setattr(kt, "KNOWLEDGE_INDEX_FILE", str(base / "knowledge_index.json"))
    # _INDEX_LOCK_FILE is derived at import time; it must be redirected with the index.
    monkeypatch.setattr(kt, "_INDEX_LOCK_FILE", str(base / "knowledge_index.json.lock"))
    monkeypatch.setattr(kt, "KNOWLEDGE_ARCHIVE_INDEX_FILE", str(base / "knowledge_index_archive.json"))
    monkeypatch.setattr(kt, "TRADINGBOT_MIRROR_DIR", str(tmp_path / "mirror"))
    monkeypatch.setenv("MOLTBOT_SEAL_ENABLED", "0")
    monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
    return base


def _save(payload):
    return asyncio.run(kt.handle_save_knowledge(payload))


def _index_entries(base):
    with open(base / "knowledge_index.json") as f:
        return json.load(f)["entries"]


def _entry_for(base, title):
    matches = [e for e in _index_entries(base) if e.get("title") == title]
    assert len(matches) == 1, matches
    return matches[0]


BASE_PAYLOAD = {
    "category": "trading_market_analysis",
    "content": "Summary: SPY closed at 512 on a 1.2% rally with breadth improving.",
    "tags": ["macro"],
    "source": "autonomous_learning",
}


class TestQualityImputedFlag:
    def test_flag_set_when_score_defaulted(self, kb):
        _save({**BASE_PAYLOAD, "title": "Imputed Score Entry"})
        e = _entry_for(kb, "Imputed Score Entry")
        assert e["quality_score"] == 5.0
        assert e["quality_imputed"] is True

    def test_flag_absent_when_real_score_given(self, kb):
        _save({**BASE_PAYLOAD, "title": "Real Score Entry", "quality_score": 7})
        e = _entry_for(kb, "Real Score Entry")
        assert e["quality_score"] == 7
        assert "quality_imputed" not in e

    def test_zero_is_a_real_score_not_imputed(self, kb):
        _save({**BASE_PAYLOAD, "title": "Zero Score Entry", "quality_score": 0})
        e = _entry_for(kb, "Zero Score Entry")
        assert e["quality_score"] == 0
        assert "quality_imputed" not in e


class TestImplPromptRule:
    RULE = ('Every price, level, date or target you state must appear in the source text; '
            'if the source gives none write "no level in source".')

    def test_rule_present_in_enrichment_prompt(self):
        assert self.RULE in ske._ENRICHMENT_PROMPT

    def test_rule_survives_prompt_formatting(self):
        rendered = ske._ENRICHMENT_PROMPT.format(
            category="c", title="t", content="x", ticker_line="", max_impl=5,
            max_facts=5, max_qa=3,
        )
        assert self.RULE in rendered
