"""
Test for agent/core.py::classify_task's "dpof" keyword addition (B1 scope).

CLIO trades its own book via DPOF now; messages mentioning "dpof" should
classify as trade_decision alongside the existing "hydrinia" keyword (kept
for historical/legacy mentions).
"""

from agent.core import classify_task


def test_dpof_mentions_classify_as_trade_decision():
    assert classify_task("what is DPOF doing right now?") == "trade_decision"
    assert classify_task("check dpof status") == "trade_decision"


def test_hydrinia_keyword_still_classifies_as_trade_decision():
    assert classify_task("is hydrinia running?") == "trade_decision"


def test_unrelated_message_still_classifies_general():
    assert classify_task("what's the weather like today?") == "general"
