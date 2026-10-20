"""R-18 (S215): CLIO_DPOF_NEWSFLASH_REQUIRE_LINK -- how the DPOF decider context treats severity-3 newsflash events that
name no market link. log (default; also any unset/bad value) must leave the context byte-identical to today."""
import asyncio
import json
import logging

import pytest

from agent.dpof import data as dd
from agent.integrations import newsflash_lite as nl

ENV = "CLIO_DPOF_NEWSFLASH_REQUIRE_LINK"


def _event(eid, headline, severity, tickers, direction, **extra):
    ev = {"id": eid, "headline": headline, "category": "war_attack", "tier": 1, "severity": severity,
          "confidence": 0.75, "source": "unusual_whales", "ts": "2026-10-12T10:00:00+00:00",
          "tickers": tickers, "direction_hint": direction, "sources": ["unusual_whales"]}
    ev.update(extra)
    return ev


def _state(events):
    return {"active": True, "max_severity": max(e["severity"] for e in events), "events": events,
            "as_of": "2026-10-12T10:00:00+00:00", "stale": False}


# The shape the decider sees today (no market_linked key anywhere).
TODAY_EVENTS = [
    _event("a", "Russian drone strikes kill woman in Kharkiv", 3, [], "risk_off"),
    _event("b", "Saudi attack on oil facility", 3, [], "risk_off"),
]
TODAY = _state(TODAY_EVENTS)


def _with_links():
    """The same events as newsflash_lite now persists them: market_linked on each."""
    return _state([
        _event("a", "Russian drone strikes kill woman in Kharkiv", 3, [], "risk_off", market_linked=False),
        _event("b", "Saudi attack on oil facility", 3, [], "risk_off", market_linked=True),
    ])


@pytest.fixture
def feed(monkeypatch):
    def install(state):
        monkeypatch.setattr(nl, "get_breaking_state", lambda max_age_seconds=900: state)
        return asyncio.run(dd.DPOFDataFeed._fetch_breaking(None, 0.0))
    return install


def _mode(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)


# ------------------------------------------------------------------ golden: log mode leaves the context text alone

@pytest.mark.parametrize("value", [None, "log", "LOG", " log ", "banana", "", "yes"])
def test_log_mode_and_bad_values_leave_the_decider_context_byte_identical(monkeypatch, feed, value):
    _mode(monkeypatch, value)
    out = feed(_with_links())
    assert json.dumps(out) == json.dumps(TODAY)
    assert "market_linked" not in json.dumps(out)


def test_log_mode_logs_the_would_cap_count_once_and_changes_nothing(monkeypatch, feed, caplog):
    _mode(monkeypatch, "log")
    with caplog.at_level(logging.INFO, logger=dd.logger.name):
        out = feed(_with_links())
    assert out["events"][0]["severity"] == 3
    lines = [r.getMessage() for r in caplog.records if "would cap sev3->2 (no market link)" in r.getMessage()]
    assert len(lines) == 1 and "on 1 newsflash event(s)" in lines[0]


def test_only_severity_3_unlinked_events_are_counted_as_would_cap(monkeypatch, feed, caplog):
    # A severity-2 unlinked event is not a candidate for the cap: it must not be counted.
    _mode(monkeypatch, "log")
    state = _state([
        _event("a", "Russian drone strikes kill woman in Kharkiv", 3, [], "risk_off", market_linked=False),
        _event("c", "Local election result", 2, [], "mixed", market_linked=False),
    ])
    with caplog.at_level(logging.INFO, logger=dd.logger.name):
        feed(state)
    lines = [r.getMessage() for r in caplog.records if "would cap sev3->2 (no market link)" in r.getMessage()]
    assert len(lines) == 1 and "on 1 newsflash event(s)" in lines[0]


def test_mode_0_is_off_and_logs_nothing(monkeypatch, feed, caplog):
    _mode(monkeypatch, "0")
    with caplog.at_level(logging.INFO, logger=dd.logger.name):
        out = feed(_with_links())
    assert json.dumps(out) == json.dumps(TODAY)
    assert "would cap" not in caplog.text


# ------------------------------------------------------------------ mode 1: DPOF text only

def test_mode_1_shows_the_unlinked_severity_3_event_at_2_and_leaves_the_linked_one_alone(monkeypatch, feed):
    _mode(monkeypatch, "1")
    state = _with_links()
    out = feed(state)
    by_id = {e["id"]: e for e in out["events"]}
    assert by_id["a"]["severity"] == 2                     # no link -> capped
    assert by_id["b"]["severity"] == 3                     # linked -> untouched
    assert out["max_severity"] == 3
    assert "market_linked" not in by_id["a"] and "market_linked" not in by_id["b"]
    assert by_id["a"]["headline"] == TODAY_EVENTS[0]["headline"]
    assert state["events"][0]["severity"] == 3             # the source dict is never edited


def test_mode_1_recomputes_max_severity_when_the_only_sev3_was_capped(monkeypatch, feed):
    _mode(monkeypatch, "1")
    out = feed(_state([_event("a", "Russian drone strikes kill woman in Kharkiv", 3, [], "risk_off", market_linked=False)]))
    assert out["events"][0]["severity"] == 2 and out["max_severity"] == 2


def test_mode_1_does_not_cap_an_event_whose_link_is_unknown(monkeypatch, feed):
    _mode(monkeypatch, "1")                                # pre-R-18 persisted event: no market_linked key
    out = feed(TODAY)
    assert [e["severity"] for e in out["events"]] == [3, 3]


def test_mode_1_logs_what_it_capped(monkeypatch, feed, caplog):
    _mode(monkeypatch, "1")
    with caplog.at_level(logging.INFO, logger=dd.logger.name):
        feed(_with_links())
    assert "capped 1 sev3 newsflash event(s)" in caplog.text
    assert "would cap" not in caplog.text


# ------------------------------------------------------------------ env parsing

@pytest.mark.parametrize("raw,expected", [
    ("1", "1"), (" 1 ", "1"), ("0", "0"), ("log", "log"), ("LOG", "log"),
    (None, "log"), ("", "log"), ("banana", "log"), ("2", "log"), ("true", "log"),
])
def test_mode_parsing_defaults_bad_values_to_log(monkeypatch, raw, expected):
    _mode(monkeypatch, raw)
    assert dd._newsflash_require_link_mode() == expected
