"""R-17 (S215): a stored trade-policy candidate is re-checked at READ time before it counts as researched.

The live file kept a forced 2026-10-06 candidate 'European Union / unspecified goods'. The gate is OR-logic, so a
rate alone passes it; the read-time view (_usable_research_state) re-applies the concrete-story gate to the stored
fields and denies placeholder goods tokens. Nothing is removed from the file itself.
"""
import asyncio
import json
import logging
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from agent.integrations import trade_policy as tp

NY = ZoneInfo("America/New_York")
FNAME = "trade_policy_state.json"


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(tp, "_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(tp, "RESEARCH_STATE_PATH", str(tmp_path / FNAME))
    return tmp_path


def _run(coro):
    return asyncio.run(coro)


def _now():
    return datetime.now(NY).isoformat()


def _entry(**overrides):
    """Shape of the live stored entry (policy_type, country, goods, researched_at, findings_count)."""
    entry = {
        "policy_type": "tariff_announcement",
        "country": "European Union",
        "goods": "unspecified goods",
        "researched_at": _now(),
        "findings_count": 18,
    }
    entry.update(overrides)
    return entry


# ------------------------------------------------------------------ the read-time filter

def test_the_live_stored_eu_unspecified_goods_candidate_is_dropped_with_one_info_line(caplog):
    state = {"researched_policies": [_entry()]}
    with caplog.at_level(logging.INFO, logger=tp.logger.name):
        usable = tp._usable_research_state(state)
    assert usable["researched_policies"] == []
    lines = [r for r in caplog.records if "dropped 1 stored candidate" in r.getMessage()]
    assert len(lines) == 1 and lines[0].levelno == logging.INFO
    assert "European Union" in lines[0].getMessage()


@pytest.mark.parametrize("goods", ["", "   "])
def test_a_stored_candidate_with_empty_goods_is_dropped(goods):
    usable = tp._usable_research_state({"researched_policies": [_entry(goods=goods, rate="25%")]})
    assert usable["researched_policies"] == []


def test_the_concrete_story_gate_is_re_applied_not_just_the_goods_deny_set():
    # Real goods + a known country, but a policy_type that is not a policy: the re-applied gate rejects it.
    bogus = _entry(policy_type="weather_report", goods="steel", rate="25%")
    assert tp._stored_candidate_is_concrete(bogus) is False
    # The same entry with a real policy type is concrete (so the rejection above is the gate, not the goods).
    assert tp._stored_candidate_is_concrete(_entry(goods="steel", rate="25%")) is True


def test_a_concrete_stored_candidate_is_kept_and_nothing_is_logged(caplog):
    state = {"researched_policies": [_entry(goods="steel")]}
    with caplog.at_level(logging.INFO, logger=tp.logger.name):
        usable = tp._usable_research_state(state)
    assert [e["goods"] for e in usable["researched_policies"]] == ["steel"]
    assert not [r for r in caplog.records if "dropped" in r.getMessage()]


@pytest.mark.parametrize("extra", [
    {"rate": "25%"},                                                  # rate alone is what let it pass the OR-gate
    {"rate": "25%", "authority": "Section 232 (national security)"},
])
def test_unspecified_goods_is_denied_even_when_a_rate_or_instrument_is_present(extra):
    assert tp._stored_candidate_is_concrete(_entry(**extra)) is False


@pytest.mark.parametrize("goods", ["unknown", "n/a", "none", "Unspecified", "unspecified goods"])
def test_every_denied_goods_token_is_dropped_even_with_a_rate(goods):
    assert tp._stored_candidate_is_concrete(_entry(goods=goods, rate="25%")) is False


@pytest.mark.parametrize("entry", [
    _entry(goods="steel"),
    _entry(goods="semiconductors", rate="25%", authority="Section 232 (national security)"),
    _entry(country="China", policy_type="export_controls", goods="rare earth"),
])
def test_concrete_stored_candidates_pass_the_regate(entry):
    assert tp._stored_candidate_is_concrete(entry) is True


def test_an_unknown_country_is_dropped():
    assert tp._stored_candidate_is_concrete(_entry(country="Atlantis", goods="steel")) is False


def test_a_legacy_entry_without_a_goods_field_is_not_judged():
    # the pre-goods shape used by the existing dedup tests: left to the old 5-day rule
    legacy = {"policy_type": "tariff_announcement", "country": "China", "researched_at": _now()}
    assert tp._usable_research_state({"researched_policies": [legacy]})["researched_policies"] == [legacy]


# ------------------------------------------------------------------ escalate(): the dedup path

def _forced_wiring(monkeypatch, story, gathered):
    async def pick():
        return story, [{"title": "t", "snippet": "s", "url": "https://e.test/b"}]

    async def gather(*_a, **_k):
        gathered.append(True)
        return [{"title": "r", "snippet": "s", "url": "https://e.test/r", "query_type": "x"}]

    async def analysis(**_k):
        return "impact chain text"

    delivered = []

    async def deliver(context, policy_type, analysis_text, urls):
        delivered.append(policy_type)

    monkeypatch.setattr(tp, "_forced_policy_pick", pick)
    monkeypatch.setattr(tp, "_gather_impact_chain_research", gather)
    monkeypatch.setattr(tp, "_recall_historical_analogs", lambda country, policy_type: "")
    monkeypatch.setattr(tp, "_assess_policy_validity", lambda policy_type, details: "")
    monkeypatch.setattr(tp, "_generate_impact_chain_analysis", analysis)
    monkeypatch.setattr(tp, "_deliver_trade_policy_intel", deliver)
    return delivered


EU_STEEL_STORY = "European Union announced a 25% tariff on steel imports."


def test_a_stored_unspecified_goods_candidate_does_not_block_a_real_eu_story(state_dir, monkeypatch):
    (state_dir / FNAME).write_text(json.dumps({"researched_policies": [_entry()]}))
    gathered = []
    delivered = _forced_wiring(monkeypatch, EU_STEEL_STORY, gathered)
    _run(tp.escalate(None, "", [], forced=True))
    assert gathered == [True]                                   # pipeline ran: the bogus entry was not "researched"
    assert delivered == ["tariff_announcement"]
    st = json.loads((state_dir / FNAME).read_text())
    # the read-time filter never edits the file: the bogus entry is still there, the new one is appended
    assert len(st["researched_policies"]) == 2
    assert st["researched_policies"][0]["goods"] == "unspecified goods"
    assert st["researched_policies"][1]["goods"] == "steel" and st["researched_policies"][1]["rate"] == "25%"


def test_a_genuinely_researched_candidate_still_dedups(state_dir, monkeypatch):
    (state_dir / FNAME).write_text(json.dumps({"researched_policies": [_entry(goods="steel")]}))
    gathered = []
    delivered = _forced_wiring(monkeypatch, EU_STEEL_STORY, gathered)
    _run(tp.escalate(None, "", [], forced=True))
    assert gathered == [] and delivered == []
