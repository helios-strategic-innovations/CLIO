"""R-18 (S215): classify_headline() carries an additive market_linked flag; nothing else about the result changes."""
from agent.integrations import newsflash_lite as nl


def test_casualty_headline_is_severity_3_war_attack_with_no_market_link():
    r = nl.classify_headline("Russian drone strikes kill woman in Kharkiv")
    assert r is not None
    assert (r["category"], r["tier"], r["severity"]) == ("war_attack", 1, 3)
    assert r["tickers"] == []
    assert r["market_linked"] is False


def test_attack_on_an_oil_facility_is_market_linked():
    r = nl.classify_headline("Saudi attack on oil facility")
    assert r is not None and r["severity"] == 3
    assert r["market_linked"] is True


def test_drone_strike_on_a_refinery_with_a_benchmark_is_market_linked():
    r = nl.classify_headline("Brent jumps as drone strike hits refinery")
    assert r is not None and r["severity"] == 3
    assert r["market_linked"] is True


def test_a_named_ticker_alone_makes_a_headline_market_linked():
    r = nl.classify_headline("NVDA plunges after guidance warns of slowdown")
    assert r is not None
    assert "NVDA" in r["tickers"]
    assert r["market_linked"] is True


def test_market_link_comes_from_the_summary_too():
    r = nl.classify_headline("Russian drone strikes kill woman in Kharkiv", "Oil and gold rose on the news")
    assert r is not None and r["market_linked"] is True


def test_unrelated_headline_returns_none_as_before():
    assert nl.classify_headline("I fed the dog before work this morning") is None


def test_public_event_carries_market_linked_additively():
    ev = {"id": "h", "headline": "x", "category": "war_attack", "tier": 1, "severity": 3,
          "tickers": [], "direction_hint": "risk_off", "market_linked": False}
    pub = nl._public_event(ev)
    assert pub["market_linked"] is False
    assert pub["severity"] == 3 and pub["direction_hint"] == "risk_off"
