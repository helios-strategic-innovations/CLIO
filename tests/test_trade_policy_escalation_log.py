"""TP-1 (S211): a trade-policy escalation that is skipped as 'not a specific policy' must say so at INFO.

news_intel's S166 forced escalation (trade_policy_state.json >14 days stale) fires only when the keyword
gate failed; trade_policy.escalate then demands a country + an action verb and used to return with a
DEBUG line (invisible at the default INFO level, ADR-002), so the state file was never produced and
nothing explained why. No network, no state I/O: the research-state loader is booby-trapped.
"""

import asyncio
import logging

import pytest

from agent.integrations import trade_policy as tp

GENERAL = "Markets were mixed on Tuesday as investors weighed the outlook for growth and inflation."
SPECIFIC = "China announced new tariffs on steel imports, effective next month."


@pytest.fixture()
def no_state_io(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("a skipped escalation must not touch the research state")

    monkeypatch.setattr(tp, "_load_research_state", boom)
    monkeypatch.setattr(tp, "_save_research_state", boom)


def test_specificity_gate_examples():
    assert tp._is_specific_policy(SPECIFIC) is True
    assert tp._is_specific_policy(GENERAL) is False
    assert tp._is_specific_policy("China is a large economy.") is False         # country, no action
    assert tp._is_specific_policy("A new tariff was announced.") is False       # action, no country


def test_general_text_skip_is_logged_at_info_and_touches_nothing(no_state_io, caplog):
    with caplog.at_level(logging.INFO, logger=tp.logger.name):
        assert asyncio.run(tp.escalate(None, GENERAL, [])) is None
    skips = [r for r in caplog.records
             if r.levelno == logging.INFO and "Trade policy escalation skipped" in r.getMessage()]
    assert len(skips) == 1
    assert "not a specific policy action" in skips[0].getMessage()


def test_empty_analysis_is_also_a_visible_skip(no_state_io, caplog):
    with caplog.at_level(logging.INFO, logger=tp.logger.name):
        asyncio.run(tp.escalate(None, "", []))
    assert any("Trade policy escalation skipped" in r.getMessage() for r in caplog.records)
