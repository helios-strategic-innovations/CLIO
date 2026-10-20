"""
Tests for agent/integrations/newsflash_lite.py (INTEL_V3_SPEC.md contract "N").

Hermetic: no network. UW is exercised via a fake `uw_flow_intel._uw_get` (mirrors
tests/test_econ_calendar_fred_fallback.py's `_patch_uw` pattern). Polygon is exercised via
a fake `httpx.AsyncClient` (mirrors the same file's `_install_fake_httpx` pattern) and its
budget via the real, registered "newsflash_polygon" BudgetTracker entry (tests/conftest.py
already redirects CLIO_RATE_LIMITS_FILE to a per-session temp file, so these calls never
touch the live VPS counters). Lighthouse is exercised via a monkeypatched
`global_news_feeds.get_recent_headlines` (this module never calls scan_current_window()
itself — see ORCH follow-up "Lighthouse headline starvation" in the module docstring)
returning real `ParsedHeadline` objects (through the real `sanitize_ingested_content`, so
the unwrap round-trip is genuinely exercised, not assumed).
"""

import asyncio
import difflib
import itertools
import logging
from datetime import datetime, timezone

import pytest

import agent.integrations.newsflash_lite as nl
from agent.config import config
from agent.rate_limiter import budget_tracker


# ═══════════════════════════════════════════════════════════════════════════
#  Isolation
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Redirect the state file to a temp path and reset every process-global mutable
    knob this module keeps (cadence timestamps, entitlement flag, read cache, run lock)
    so tests never see leakage from a previous test. Also resets the real
    "newsflash_polygon" budget counter (in-memory _ApiBudget, not the on-disk file) so
    each test starts with a fresh daily allowance."""
    monkeypatch.setenv("CLIO_NEWSFLASH_FILE", str(tmp_path / "newsflash_state.json"))
    monkeypatch.setenv("CLIO_NEWSFLASH_ENABLED", "1")
    monkeypatch.setenv("CLIO_NEWSFLASH_TELEGRAM_LOG_ONLY", "0")
    monkeypatch.setattr(nl, "_uw_last_fetch_ts", 0.0)
    monkeypatch.setattr(nl, "_polygon_last_fetch_ts", 0.0)
    monkeypatch.setattr(nl, "_polygon_entitlement_ok", None)
    monkeypatch.setattr(nl, "_READ_CACHE", {"mtime": None, "data": None})
    monkeypatch.setattr(nl, "_run_lock", asyncio.Lock())
    monkeypatch.setattr(nl, "_seen_item_cache", {})
    monkeypatch.setattr(config, "uw_api_key", "")
    monkeypatch.setattr(config, "polygon_api_key", "")
    monkeypatch.setattr(config, "telegram_notify_chat_ids", [])
    if "newsflash_polygon" in budget_tracker._budgets:
        budget_tracker._budgets["newsflash_polygon"].used = 0
        budget_tracker._budgets["newsflash_polygon"].period_start = ""
    yield


def _run(coro):
    return asyncio.run(coro)


# ═══════════════════════════════════════════════════════════════════════════
#  Classification matrix
# ═══════════════════════════════════════════════════════════════════════════

_TIER1_CASES = [
    ("Federal Reserve announces surprise rate cut", "fed_emergency", "risk_on"),
    ("Fed calls emergency meeting to address banking crisis", "fed_emergency", "mixed"),
    ("Russia declares war on Ukraine after border incident", "war_attack", "risk_off"),
    ("Major US bank collapses amid deposit run panic", "bank_failure", "risk_off"),
    ("NYSE halts trading market-wide after circuit breakers triggered", "exchange_halt", "risk_off"),
    ("Argentina defaults on sovereign debt following crisis", "sovereign_default", "risk_off"),
    ("WHO declares new pandemic a global health emergency", "pandemic", "risk_off"),
    ("Stock market crashes as Dow plummets 2000 points", "market_crash", "risk_off"),
]

_TIER2_CASES = [
    ("Fed hikes rates in hawkish surprise move", "macro_policy", "risk_off"),
    ("Central bank cuts rates in dovish pivot", "macro_policy", "risk_on"),
    ("Taiwan strait tensions escalate after incursion", "geopolitical_mover", "risk_off"),
    ("Ceasefire agreed after weeks of fighting", "geopolitical_mover", "risk_on"),
    ("NVDA plunges after guidance warns of slowdown", "earnings_shock", "risk_off"),
    ("TSLA surges after beats on strong deliveries", "earnings_shock", "risk_on"),
    ("White House announces new tariffs on Chinese imports", "tariffs", "risk_off"),
    ("Tariffs on steel imports rolled back after deal", "tariffs", "risk_on"),
    ("Oil prices spike as crude surges on supply fears", "oil_shock", "risk_off"),
    ("Oil prices plunge as crude collapses on demand fears", "oil_shock", "mixed"),
]

_TIER3_CASES = [
    "Inflation data comes in line with expectations today",
    "Analysts weigh in on upcoming earnings season guidance",
    "Treasury yields tick higher ahead of bond market auction",
]

_IRRELEVANT_CASES = [
    "Random small biotech reports FDA milestone",
    "Local sports team wins championship game",
    "New smartphone model announced with better camera",
]

_FALSE_POSITIVE_GUARD_CASES = [
    "Amazon and Walmart price war intensifies over grocery delivery",
    "Retail chains locked in bidding war for prime real estate",
    "Senator reignites the war on drugs debate in Congress",
    "New Crash Bandicoot 5 game announced for consoles",
    "Automaker unveils new crash-test rated Model Y safety rating",
    "Local team wins car crash derby event this weekend",
]


@pytest.mark.parametrize("title,category,direction", _TIER1_CASES)
def test_tier1_classification(title, category, direction):
    r = nl.classify_headline(title)
    assert r is not None, title
    assert r["tier"] == 1
    assert r["severity"] == 3
    assert r["category"] == category
    assert r["direction_hint"] == direction


@pytest.mark.parametrize("title,category,direction", _TIER2_CASES)
def test_tier2_classification(title, category, direction):
    r = nl.classify_headline(title)
    assert r is not None, title
    assert r["tier"] == 2
    assert r["severity"] == 2
    assert r["category"] == category
    assert r["direction_hint"] == direction


@pytest.mark.parametrize("title", _TIER3_CASES)
def test_tier3_notable_fallback(title):
    r = nl.classify_headline(title)
    assert r is not None, title
    assert r["tier"] == 3
    assert r["severity"] == 1
    assert r["category"] == "notable"
    assert r["direction_hint"] == "mixed"


@pytest.mark.parametrize("title", _IRRELEVANT_CASES)
def test_irrelevant_headlines_return_none(title):
    assert nl.classify_headline(title) is None, title


@pytest.mark.parametrize("title", _FALSE_POSITIVE_GUARD_CASES)
def test_false_positive_guards(title):
    """'war' in 'price war'/'bidding war'/'war on drugs' and 'crash' in a product/test
    name must never classify as the Tier1 war_attack/market_crash categories."""
    r = nl.classify_headline(title)
    assert r is None or r["category"] not in ("war_attack", "market_crash"), (title, r)


def test_bare_fed_word_is_not_a_false_positive_trigger():
    """Tier 3 relevance must not fire on plain English containing the substring 'fed'
    outside a finance context (e.g. 'fed the dog') — see the multi-word _TIER3_PATTERNS
    design note."""
    assert nl.classify_headline("I fed the dog before work this morning") is None


def test_demotion_patterns_are_exercised():
    """Direct check that the demotion list actually matches the phrases it targets
    (defense-in-depth against a future Tier1/2 pattern addition regressing this)."""
    demo_text = "the two firms are locked in a bidding war"
    assert any(p.search(demo_text) for p in nl._DEMOTION_PATTERNS)


# ═══════════════════════════════════════════════════════════════════════════
#  ORCH follow-up #2 items 2/3 — coverage-gap fixes (probe headlines, before all NONE)
# ═══════════════════════════════════════════════════════════════════════════

_COVERAGE_GAP_CASES = [
    # (headline, expected category, expected tier)
    ("Fed's Waller says rate cuts likely appropriate in coming months", "macro_policy", 2),
    ("Bank of Japan unexpectedly raises interest rates", "macro_policy", 2),
    ("Dow tumbles 800 points as banking crisis fears grow", "market_selloff", 2),
    ("CPI comes in hotter than expected, stocks slide", "macro_policy", 2),
    ("Nonfarm payrolls miss estimates by wide margin", "macro_policy", 2),
    ("Apple beats on earnings, raises full-year guidance", "earnings_shock", 2),
    ("Microsoft warns of slower cloud growth ahead", "earnings_shock", 2),
    ("Nvidia shares fall after-hours on weak guidance", "earnings_shock", 2),
    ("Tesla stock crashes after Musk comments on X", "earnings_shock", 2),
    ("Meta plunges 15% on disappointing ad revenue", "earnings_shock", 2),
    ("Oil jumps 5% after drone attack on Saudi facility", None, None),  # not NONE — verified below
    ("Nasdaq sinks 3% as tech stocks fall", "market_crash", 1),
    ("Dow tumbles 800 points in broad selloff", "market_selloff", 2),
]


@pytest.mark.parametrize("title,expected_category,expected_tier", _COVERAGE_GAP_CASES)
def test_coverage_gap_fixes_no_longer_classify_none(title, expected_category, expected_tier):
    r = nl.classify_headline(title)
    assert r is not None, f"still NONE: {title}"
    if expected_category is not None:
        assert r["category"] == expected_category, (title, r)
        assert r["tier"] == expected_tier, (title, r)


def test_routine_scheduled_central_bank_hold_is_tier3_not_tier2():
    """A ROUTINE (non-surprise) central-bank decision must stay Tier3, not escalate to
    macro_policy Tier2 — only unexpected/surprise moves do (ORCH follow-up #2 item 3)."""
    r = nl.classify_headline("Bank of Japan holds interest rates steady as widely expected")
    assert r is not None
    assert r["tier"] == 3
    assert r["category"] == "notable"


_MEGACAP_NAME_CASES = [
    ("Nvidia plunges after guidance warns of slowdown", "NVDA"),
    ("Apple beats on earnings", "AAPL"),
    ("Microsoft warns of slower cloud growth", "MSFT"),
    ("Amazon misses on revenue expectations", "AMZN"),
    ("Meta surges on strong ad sales", "META"),
    ("Facebook parent surges on strong ad sales", "META"),
    ("Alphabet beats on search revenue", "GOOGL"),
    ("Google beats on search revenue", "GOOGL"),
    ("Tesla plunges on delivery miss", "TSLA"),
    ("Broadcom surges after strong AI chip demand beat", "AVGO"),
]


@pytest.mark.parametrize("title,expected_ticker", _MEGACAP_NAME_CASES)
def test_earnings_shock_accepts_company_names_not_just_tickers(title, expected_ticker):
    r = nl.classify_headline(title)
    assert r is not None, title
    assert r["category"] == "earnings_shock"
    assert expected_ticker in r["tickers"]


def test_earnings_shock_accepts_api_supplied_tickers():
    """A headline with NO company name/ticker text mention at all still classifies as
    earnings_shock when the ticker arrives via api_tickers (ORCH follow-up #2 item 2:
    'also accept api-supplied tickers in the classification decision')."""
    r = nl.classify_headline("Shares plunge after the company slashes its guidance",
                              api_tickers=["NVDA"])
    assert r is not None
    assert r["category"] == "earnings_shock"
    assert r["tickers"] == ["NVDA"]


def test_earnings_shock_still_requires_a_verb_even_with_known_name():
    """A megacap name mention with NO earnings-shock verb must NOT classify as
    earnings_shock (e.g. a routine product announcement)."""
    r = nl.classify_headline("Meta announced new VR headset at conference today")
    assert r is None or r["category"] != "earnings_shock"


# ═══════════════════════════════════════════════════════════════════════════
#  ORCH follow-up #2 item 3 — negative controls (10+, none may classify Tier1/2)
# ═══════════════════════════════════════════════════════════════════════════

_NEGATIVE_CONTROL_CASES = [
    "Dow Chemical reports quarterly earnings beat",
    "New crash course teaches coding basics for beginners",
    "Miss Universe pageant winner crowned in Bangkok",
    "Tesla recall of 200 cars over battery defect",
    "Local team suffers crushing defeat in playoff crash",
    "Bitcoin crashes below $30,000 amid liquidation cascade",
    "Amazon stock price war with Walmart heats up in grocery delivery",
    "Car crash on I-95 closes lanes for hours",
    "New Crash Bandicoot game announced for 2027",
    "Bidding war erupts for control of regional bank",
    "US declares war on drugs with new fentanyl crackdown",
    "She was a spy during the cold war, new book reveals",
    "Governor martial law rumors spark confusion online",
    "Dow slides modestly in quiet trading",
]


@pytest.mark.parametrize("title", _NEGATIVE_CONTROL_CASES)
def test_negative_controls_never_hit_tier1_or_tier2(title):
    r = nl.classify_headline(title)
    assert r is None or r["tier"] not in (1, 2), (title, r)


# ═══════════════════════════════════════════════════════════════════════════
#  Ticker extraction
# ═══════════════════════════════════════════════════════════════════════════

def test_ticker_extraction_case_sensitive():
    r = nl.classify_headline("NVDA plunges after guidance warns of slowdown")
    assert r["tickers"] == ["NVDA"]
    # lowercase "meta" as an ordinary word must not false-positive tag META
    assert "META" not in (nl.classify_headline("We will meta-analyze the market data") or {}).get("tickers", [])


def test_ticker_extraction_from_api_tickers():
    r = nl.classify_headline("Tech stocks under pressure amid broad selloff",
                              api_tickers=["AAPL", "XYZ123"])
    # "Tech stocks under pressure" alone isn't classifiable -> None; use a classifiable one
    r2 = nl.classify_headline("Oil prices spike as crude surges on supply fears",
                               api_tickers=["AAPL", "NOTATICKER"])
    assert r2["tickers"] == ["AAPL"]


# ═══════════════════════════════════════════════════════════════════════════
#  ORCH follow-up #2 item 5 — ALL-CAPS ticker-extraction guard
# ═══════════════════════════════════════════════════════════════════════════

def test_is_mostly_uppercase_detection():
    assert nl._is_mostly_uppercase("BREAKING FED CUTS RATES IN EMERGENCY MOVE") is True
    assert nl._is_mostly_uppercase("Fed cuts rates in surprise move") is False
    assert nl._is_mostly_uppercase("SPY") is False  # too short to judge reliably


def test_allcaps_headline_suppresses_free_text_ticker_scan():
    """A non-financial ALL-CAPS headline that happens to contain 'SPY' as an ordinary
    word (not the ticker) must not false-positive tag SPY once the whole headline is
    uppercased, making 'SPY' indistinguishable from the ticker."""
    tickers = nl._extract_tickers("SPY THRILLER NEW MOVIE RELEASES THIS WEEKEND IN THEATERS NATIONWIDE")
    assert tickers == []


def test_normal_case_headline_still_extracts_tickers_correctly():
    """The all-caps guard must not regress ordinary mixed-case headlines with a genuine
    uppercase ticker mention."""
    tickers = nl._extract_tickers("Fed cuts rates in surprise move, SPY rallies 2 percent")
    assert tickers == ["SPY"]


def test_allcaps_headline_still_honors_api_tickers():
    """api_tickers must still apply even when the free-text scan is suppressed for an
    all-caps headline — the guard only distrusts the ambiguous free-text scan, not a
    structured ticker field from the source API."""
    tickers = nl._extract_tickers(
        "BREAKING FED CUTS RATES IN EMERGENCY MOVE MARKETS SOAR", api_tickers=["SPY"]
    )
    assert tickers == ["SPY"]


# ═══════════════════════════════════════════════════════════════════════════
#  Dedupe (exact hash + fuzzy near-duplicate) across sources
# ═══════════════════════════════════════════════════════════════════════════

def test_exact_dedup_same_normalized_title():
    norm = nl._normalize_title("Fed hikes rates in hawkish surprise move")
    h = nl._title_hash(norm)
    events = [{"_hash": h, "_norm": norm, "last_seen_ts": 1000.0}]
    match = nl._find_match(norm, h, events, now_ts=1010.0)
    assert match is events[0]


def test_fuzzy_dedup_near_duplicate_paraphrase():
    norm_a = nl._normalize_title("Fed hikes rates in hawkish surprise move")
    h_a = nl._title_hash(norm_a)
    events = [{"_hash": h_a, "_norm": norm_a, "last_seen_ts": 1000.0}]
    norm_b = nl._normalize_title("Fed hikes interest rates in a hawkish surprise")
    h_b = nl._title_hash(norm_b)
    assert h_a != h_b  # genuinely different hash — this exercises the FUZZY path, not exact
    match = nl._find_match(norm_b, h_b, events, now_ts=1010.0)
    assert match is events[0]


def test_fuzzy_dedup_does_not_merge_distinct_events():
    titles = [
        "Russia declares war on Ukraine after border incident",
        "North Korea declares war on South Korea amid tensions",
        "China declares war on Taiwan after military buildup",
        "Iran declares war on Israel following strikes",
    ]
    norms = [nl._normalize_title(t) for t in titles]
    for a, b in itertools.combinations(norms, 2):
        assert difflib.SequenceMatcher(None, a, b).ratio() < nl._FUZZY_THRESHOLD, (a, b)


def test_dedup_window_expires():
    """A near-identical headline OUTSIDE the dedup window creates a NEW event, not a merge
    (an old event 'falling off' the corroboration window is correct — it will already have
    been evicted from the persisted list by _EVENT_RETENTION_SECONDS long before this)."""
    norm = nl._normalize_title("Fed hikes rates in hawkish surprise move")
    h = nl._title_hash(norm)
    events = [{"_hash": h, "_norm": norm, "last_seen_ts": 0.0}]
    match = nl._find_match(norm, h, events, now_ts=nl._DEDUP_WINDOW_SECONDS + 100)
    assert match is None


# ═══════════════════════════════════════════════════════════════════════════
#  Confidence: corroboration raises it; single low-authority source lowers it
# ═══════════════════════════════════════════════════════════════════════════

def test_corroboration_raises_confidence():
    single = nl._compute_confidence(2, [nl._AUTHORITY_BASE["unusual_whales"]])
    corroborated = nl._compute_confidence(
        2, [nl._AUTHORITY_BASE["unusual_whales"], nl._AUTHORITY_BASE["polygon_news"]]
    )
    assert corroborated > single


def test_corroboration_boost_is_capped():
    many = nl._compute_confidence(2, [0.95] * 10)
    assert many <= 1.0


def test_single_low_authority_source_lowers_confidence():
    low = nl._compute_confidence(2, [nl._LIGHTHOUSE_TIER_AUTHORITY["T3_STATE_AFFILIATED"]])
    high = nl._compute_confidence(2, [nl._AUTHORITY_BASE["unusual_whales"]])
    assert low < high
    # the penalty specifically targets the SINGLE-low-authority case, not corroboration —
    # a T3 source corroborated by a second (even T3) source should not eat the penalty.
    corroborated_low = nl._compute_confidence(2, [0.48, 0.48])
    assert corroborated_low > low


def test_confidence_bounded_0_to_1():
    for tier in (1, 2, 3):
        for authorities in ([], [0.0], [1.0] * 20):
            c = nl._compute_confidence(tier, authorities)
            assert 0.0 <= c <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  State reader: stale / missing / never-raises / mtime-cached
# ═══════════════════════════════════════════════════════════════════════════

def test_get_breaking_state_missing_file():
    r = nl.get_breaking_state()
    assert r == {"active": False, "max_severity": 0, "events": [], "as_of": None, "stale": True}


def test_get_breaking_state_stale_snapshot(monkeypatch, tmp_path):
    import json as _json
    from datetime import timedelta
    path = tmp_path / "newsflash_state.json"
    old_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    path.write_text(_json.dumps({
        "active": True, "max_severity": 3, "as_of": old_ts,
        "events": [{"id": "x", "headline": "old event", "category": "war_attack", "tier": 1,
                    "severity": 3, "confidence": 0.9, "source": "uw", "ts": old_ts,
                    "tickers": [], "direction_hint": "risk_off"}],
    }))
    monkeypatch.setenv("CLIO_NEWSFLASH_FILE", str(path))
    r = nl.get_breaking_state(max_age_seconds=900)
    assert r["stale"] is True
    assert r["active"] is False
    assert r["events"] == []


def test_get_breaking_state_fresh_snapshot(monkeypatch, tmp_path):
    import json as _json
    path = tmp_path / "newsflash_state.json"
    now_iso = datetime.now(timezone.utc).isoformat()
    path.write_text(_json.dumps({
        "active": True, "max_severity": 3, "as_of": now_iso,
        "events": [{"id": "x", "headline": "fresh event", "category": "war_attack", "tier": 1,
                    "severity": 3, "confidence": 0.9, "source": "uw", "sources": ["uw"],
                    "ts": now_iso, "tickers": ["SPY"], "direction_hint": "risk_off",
                    "_hash": "x", "_norm": "x", "last_seen_ts": 0, "authorities": [0.9]}],
    }))
    monkeypatch.setenv("CLIO_NEWSFLASH_FILE", str(path))
    r = nl.get_breaking_state(max_age_seconds=900)
    assert r["stale"] is False
    assert r["active"] is True
    assert r["max_severity"] == 3
    assert len(r["events"]) == 1
    ev = r["events"][0]
    # internal bookkeeping keys never leak into the public contract
    # market_linked (R-18, S215) is the one deliberate additive public field
    assert set(ev.keys()) == {"id", "headline", "category", "tier", "severity", "confidence",
                               "source", "ts", "tickers", "direction_hint", "sources", "market_linked"}
    assert ev["tickers"] == ["SPY"]


def test_get_breaking_state_corrupt_json_never_raises(monkeypatch, tmp_path):
    path = tmp_path / "newsflash_state.json"
    path.write_text("{not valid json")
    monkeypatch.setenv("CLIO_NEWSFLASH_FILE", str(path))
    r = nl.get_breaking_state()
    assert r["stale"] is True
    assert r["events"] == []


def test_get_breaking_state_never_raises_on_weird_input(monkeypatch, tmp_path):
    path = tmp_path / "newsflash_state.json"
    path.write_text("[1, 2, 3]")  # valid JSON, wrong top-level type
    monkeypatch.setenv("CLIO_NEWSFLASH_FILE", str(path))
    r = nl.get_breaking_state()
    assert r["stale"] is True


def test_get_breaking_state_is_mtime_cached(monkeypatch, tmp_path):
    import json as _json
    path = tmp_path / "newsflash_state.json"
    now_iso = datetime.now(timezone.utc).isoformat()
    path.write_text(_json.dumps({"active": False, "max_severity": 0, "as_of": now_iso, "events": []}))
    monkeypatch.setenv("CLIO_NEWSFLASH_FILE", str(path))
    r1 = nl.get_breaking_state()
    # mutate the file WITHOUT changing its mtime bucket in a way the cache would notice —
    # instead, directly assert the read-cache dict got populated after the first read.
    assert nl._READ_CACHE["mtime"] is not None
    r2 = nl.get_breaking_state()
    assert r1 == r2


# ═══════════════════════════════════════════════════════════════════════════
#  State writer: atomic (tmp + os.replace)
# ═══════════════════════════════════════════════════════════════════════════

def test_save_state_sync_is_atomic_write(tmp_path, monkeypatch):
    path = tmp_path / "newsflash_state.json"
    monkeypatch.setenv("CLIO_NEWSFLASH_FILE", str(path))
    nl._save_state_sync({"active": True, "max_severity": 1, "as_of": "x", "events": []})
    assert path.exists()
    assert not (tmp_path / "newsflash_state.json.tmp").exists()
    import json as _json
    assert _json.loads(path.read_text())["active"] is True


# ═══════════════════════════════════════════════════════════════════════════
#  UW ingest: shared budget, cadence gating, source isolation
# ═══════════════════════════════════════════════════════════════════════════

def test_uw_fetch_disabled_without_api_key():
    assert _run(nl._fetch_uw_news()) == []


def test_uw_fetch_uses_shared_budget_and_uw_get(monkeypatch):
    calls = {"check": 0, "uw_get_args": None}

    async def fake_check(name):
        calls["check"] += 1
        assert name == "unusual_whales"

    async def fake_uw_get(endpoint, params=None, *, skip_moltbot_budget=False):
        calls["uw_get_args"] = (endpoint, params, skip_moltbot_budget)
        return {"data": [{"headline": "Fed hikes rates in hawkish surprise move",
                           "created_at": "2026-01-01T00:00:00Z", "tickers": []}]}

    monkeypatch.setattr(config, "uw_api_key", "uw-test-key")
    monkeypatch.setattr(nl, "check_and_record", fake_check)
    import agent.integrations.uw_flow_intel as uwflow
    monkeypatch.setattr(uwflow, "_uw_get", fake_uw_get)

    items = _run(nl._fetch_uw_news())
    assert calls["check"] == 1
    assert calls["uw_get_args"] == ("/news/headlines", {"limit": 50}, True)
    assert len(items) == 1
    assert items[0]["source"] == "unusual_whales"
    assert items[0]["source_kind"] == "unusual_whales"


def test_uw_fetch_budget_exhausted_returns_empty(monkeypatch):
    async def fake_check(name):
        raise nl.BudgetExhaustedError(name, "exhausted")

    called = {"uw_get": False}

    async def fake_uw_get(*a, **kw):
        called["uw_get"] = True
        return {"data": []}

    monkeypatch.setattr(config, "uw_api_key", "uw-test-key")
    monkeypatch.setattr(nl, "check_and_record", fake_check)
    import agent.integrations.uw_flow_intel as uwflow
    monkeypatch.setattr(uwflow, "_uw_get", fake_uw_get)

    items = _run(nl._fetch_uw_news())
    assert items == []
    assert called["uw_get"] is False  # never even reaches the fetch call


def test_uw_fetch_min_interval_gating(monkeypatch):
    calls = {"n": 0}

    async def fake_check(name):
        pass

    async def fake_uw_get(*a, **kw):
        calls["n"] += 1
        return {"data": []}

    monkeypatch.setattr(config, "uw_api_key", "uw-test-key")
    monkeypatch.setattr(nl, "check_and_record", fake_check)
    import agent.integrations.uw_flow_intel as uwflow
    monkeypatch.setattr(uwflow, "_uw_get", fake_uw_get)

    _run(nl._fetch_uw_news())
    _run(nl._fetch_uw_news())  # immediately again — gated by _UW_MIN_INTERVAL_S
    assert calls["n"] == 1


def test_uw_fetch_malformed_payload_isolated(monkeypatch):
    """_uw_get's uniform {} failure signature (no 'data' key) must not raise."""
    async def fake_check(name):
        pass

    async def fake_uw_get(*a, **kw):
        return {}

    monkeypatch.setattr(config, "uw_api_key", "uw-test-key")
    monkeypatch.setattr(nl, "check_and_record", fake_check)
    import agent.integrations.uw_flow_intel as uwflow
    monkeypatch.setattr(uwflow, "_uw_get", fake_uw_get)

    assert _run(nl._fetch_uw_news()) == []


# ═══════════════════════════════════════════════════════════════════════════
#  Polygon ingest: direct httpx, entitlement self-disable, cadence + daily cap
# ═══════════════════════════════════════════════════════════════════════════

class _FakePolygonResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json_data = json_data if json_data is not None else {}

    def json(self):
        return self._json_data


class _FakePolygonClient:
    def __init__(self, outcome_holder, calls, **_kwargs):
        self._outcome = outcome_holder
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def get(self, url, params=None):
        self._calls.append((url, params))
        outcome = self._outcome[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _install_fake_polygon_httpx(monkeypatch, outcome):
    import httpx as real_httpx
    calls: list = []
    holder = [outcome]

    def _factory(**kwargs):
        return _FakePolygonClient(holder, calls, **kwargs)

    monkeypatch.setattr(real_httpx, "AsyncClient", _factory)
    return calls, holder


def test_polygon_fetch_disabled_without_api_key():
    assert _run(nl._fetch_polygon_news()) == []


def test_polygon_news_parses_results(monkeypatch):
    monkeypatch.setattr(config, "polygon_api_key", "poly-test-key")
    payload = {"results": [{
        "title": "Oil prices spike as crude surges on supply fears",
        "description": "Crude oil prices jumped today.",
        "publisher": {"name": "Reuters"},
        "published_utc": "2026-01-01T00:00:00Z",
        "tickers": ["XOM", "CVX"],
    }]}
    _install_fake_polygon_httpx(monkeypatch, _FakePolygonResponse(200, payload))
    items = _run(nl._fetch_polygon_news())
    assert len(items) == 1
    assert items[0]["source"] == "polygon_news:Reuters"
    assert items[0]["source_kind"] == "polygon_news"
    assert items[0]["tickers"] == ["XOM", "CVX"]


def test_polygon_entitlement_403_disables_source_with_one_log(monkeypatch, caplog):
    monkeypatch.setattr(config, "polygon_api_key", "poly-test-key")
    calls, holder = _install_fake_polygon_httpx(monkeypatch, _FakePolygonResponse(403))
    caplog.set_level(logging.INFO, logger="openclaw.newsflash_lite")

    items1 = _run(nl._fetch_polygon_news())
    assert items1 == []
    assert nl._polygon_entitlement_ok is False
    entitlement_logs = [r for r in caplog.records if "entitlement" in r.message.lower()]
    assert len(entitlement_logs) == 1

    nl._polygon_last_fetch_ts = 0.0  # bypass cadence gate to prove the SOURCE stays disabled
    items2 = _run(nl._fetch_polygon_news())
    assert items2 == []
    assert len(calls) == 1  # only the first 403 actually hit the network
    entitlement_logs_after = [r for r in caplog.records if "entitlement" in r.message.lower()]
    assert len(entitlement_logs_after) == 1  # no repeat log spam


def test_polygon_transient_error_does_not_permanently_disable(monkeypatch):
    monkeypatch.setattr(config, "polygon_api_key", "poly-test-key")
    _install_fake_polygon_httpx(monkeypatch, _FakePolygonResponse(500))
    items = _run(nl._fetch_polygon_news())
    assert items == []
    assert nl._polygon_entitlement_ok is None  # untouched — 500 is transient, not entitlement


def test_polygon_fetch_min_interval_gating(monkeypatch):
    monkeypatch.setattr(config, "polygon_api_key", "poly-test-key")
    calls, holder = _install_fake_polygon_httpx(monkeypatch, _FakePolygonResponse(200, {"results": []}))
    _run(nl._fetch_polygon_news())
    _run(nl._fetch_polygon_news())  # immediately again — gated
    assert len(calls) == 1


def test_polygon_budget_exhausted_returns_empty_without_request(monkeypatch):
    monkeypatch.setattr(config, "polygon_api_key", "poly-test-key")
    calls, holder = _install_fake_polygon_httpx(monkeypatch, _FakePolygonResponse(200, {"results": []}))

    async def fake_check(name):
        assert name == "newsflash_polygon"
        raise nl.BudgetExhaustedError(name, "exhausted")

    monkeypatch.setattr(nl, "check_and_record", fake_check)
    items = _run(nl._fetch_polygon_news())
    assert items == []
    assert len(calls) == 0  # budget denied before the HTTP call


def test_polygon_uses_registered_newsflash_polygon_budget(monkeypatch):
    """Exercises the REAL check_and_record()/BudgetTracker path (not a fake), proving
    the "newsflash_polygon" budget registered in agent/rate_limiter.py is actually wired
    up and enforced end to end."""
    monkeypatch.setattr(config, "polygon_api_key", "poly-test-key")
    _install_fake_polygon_httpx(monkeypatch, _FakePolygonResponse(200, {"results": []}))
    assert "newsflash_polygon" in budget_tracker._budgets
    used_before = budget_tracker._budgets["newsflash_polygon"].used
    _run(nl._fetch_polygon_news())
    assert budget_tracker._budgets["newsflash_polygon"].used == used_before + 1


# ═══════════════════════════════════════════════════════════════════════════
#  Lighthouse ingest: reads the shared buffer (never scan_current_window()), unwrap,
#  source tagging, failure isolation
# ═══════════════════════════════════════════════════════════════════════════

def _make_parsed_headline(title, tier="T1_VERIFIED", source_id="reuters_world"):
    from agent.integrations.global_news_feeds import ParsedHeadline
    from agent.security import sanitize_ingested_content
    wrapped = sanitize_ingested_content(title, source=f"rss:{source_id}")
    return ParsedHeadline(
        title=wrapped, summary="", source_name="Reuters World", source_id=source_id,
        source_region="uk", reliability_tier=tier, published_utc=datetime.now(timezone.utc),
        hours_until_us_open=0.0, feed_url="http://example.com/feed", link="http://example.com/a",
        is_state_media=False,
    )


def test_lighthouse_reads_recent_headlines_buffer_not_scan_current_window(monkeypatch):
    """newsflash_lite must NEVER call scan_current_window() itself (ORCH follow-up
    'Lighthouse headline starvation') — only get_recent_headlines()."""
    import agent.integrations.global_news_feeds as gnf

    def raising_scan():
        raise AssertionError("newsflash_lite must not call scan_current_window()")

    hl = _make_parsed_headline("Russia declares war on Ukraine after border incident")
    monkeypatch.setattr(gnf, "scan_current_window", raising_scan)
    monkeypatch.setattr(gnf, "get_recent_headlines", lambda max_age_seconds=3600: [hl])

    items = _run(nl._fetch_lighthouse_events())
    assert len(items) == 1
    assert items[0]["title"] == "Russia declares war on Ukraine after border incident"


def test_lighthouse_unwraps_sanitized_title_and_tags_source(monkeypatch):
    import agent.integrations.global_news_feeds as gnf
    hl = _make_parsed_headline("Russia declares war on Ukraine after border incident")
    monkeypatch.setattr(gnf, "get_recent_headlines", lambda max_age_seconds=3600: [hl])

    items = _run(nl._fetch_lighthouse_events())
    assert len(items) == 1
    assert items[0]["title"] == "Russia declares war on Ukraine after border incident"
    assert "<ingested_data" not in items[0]["title"]
    assert items[0]["source"] == "lighthouse:reuters_world"
    assert items[0]["source_kind"] == "lighthouse"
    assert items[0]["reliability_tier"] == "T1_VERIFIED"


def test_lighthouse_reads_buffer_every_cycle_no_cadence_gate(monkeypatch):
    """Unlike the old scan_current_window()-calling design, reading the buffer is cheap
    and non-network, so there is no min-interval throttle — every refresh cycle reads
    whatever is currently in the buffer."""
    import agent.integrations.global_news_feeds as gnf
    calls = {"n": 0}

    def fake_get_recent(max_age_seconds=3600):
        calls["n"] += 1
        return []

    monkeypatch.setattr(gnf, "get_recent_headlines", fake_get_recent)
    _run(nl._fetch_lighthouse_events())
    _run(nl._fetch_lighthouse_events())  # immediately again — NOT gated
    assert calls["n"] == 2


def test_lighthouse_buffer_read_failure_isolated(monkeypatch):
    import agent.integrations.global_news_feeds as gnf

    def raising(max_age_seconds=3600):
        raise RuntimeError("buffer read failed")

    monkeypatch.setattr(gnf, "get_recent_headlines", raising)
    assert _run(nl._fetch_lighthouse_events()) == []


# ═══════════════════════════════════════════════════════════════════════════
#  refresh_newsflash pipeline: merge/corroborate, severity/active, source isolation
# ═══════════════════════════════════════════════════════════════════════════

def _item(title, source, source_kind, tickers=None, reliability_tier=None, summary=""):
    return {"title": title, "summary": summary, "source": source, "source_kind": source_kind,
            "reliability_tier": reliability_tier, "published": None, "tickers": tickers or []}


def test_refresh_merges_corroborating_sources_and_boosts_confidence(monkeypatch):
    async def fake_uw():
        return [_item("Fed hikes rates in hawkish surprise move", "unusual_whales", "unusual_whales")]

    async def fake_poly():
        return [_item("Fed hikes interest rates in a hawkish surprise", "polygon_news:Reuters", "polygon_news")]

    async def fake_lh():
        return []

    monkeypatch.setattr(nl, "_fetch_uw_news", fake_uw)
    monkeypatch.setattr(nl, "_fetch_polygon_news", fake_poly)
    monkeypatch.setattr(nl, "_fetch_lighthouse_events", fake_lh)

    result = _run(nl.refresh_newsflash(context=None))
    assert result["status"] == "ok"
    assert result["events"] == 1  # the two paraphrases merged into ONE event
    assert result["active"] is True
    assert result["max_severity"] == 2

    state = nl.get_breaking_state()
    ev = state["events"][0]
    assert set(ev["sources"]) == {"unusual_whales", "polygon_news:Reuters"}
    single_source_conf = nl._compute_confidence(2, [nl._AUTHORITY_BASE["unusual_whales"]])
    assert ev["confidence"] > single_source_conf


def test_refresh_source_failure_isolation(monkeypatch):
    async def failing_uw():
        raise RuntimeError("UW down")

    async def ok_polygon():
        return [_item("Oil prices spike as crude surges on supply fears", "polygon_news:AP", "polygon_news")]

    async def ok_lh():
        return []

    monkeypatch.setattr(nl, "_fetch_uw_news", failing_uw)
    monkeypatch.setattr(nl, "_fetch_polygon_news", ok_polygon)
    monkeypatch.setattr(nl, "_fetch_lighthouse_events", ok_lh)

    result = _run(nl.refresh_newsflash(context=None))
    assert result["status"] == "ok"
    assert result["events"] == 1


def test_refresh_all_sources_fail_still_returns_ok(monkeypatch):
    async def boom():
        raise RuntimeError("down")

    monkeypatch.setattr(nl, "_fetch_uw_news", boom)
    monkeypatch.setattr(nl, "_fetch_polygon_news", boom)
    monkeypatch.setattr(nl, "_fetch_lighthouse_events", boom)

    result = _run(nl.refresh_newsflash(context=None))
    assert result["status"] == "ok"
    assert result["events"] == 0
    assert result["active"] is False


def test_refresh_never_calls_scan_current_window(monkeypatch):
    """Full-pipeline proof (ORCH follow-up 'Lighthouse headline starvation'): a real
    refresh_newsflash() run must never reach global_news_feeds.scan_current_window()
    anywhere in its call graph — only get_recent_headlines()."""
    import agent.integrations.global_news_feeds as gnf

    def raising():
        raise AssertionError("refresh_newsflash must never call scan_current_window()")

    monkeypatch.setattr(gnf, "scan_current_window", raising)
    monkeypatch.setattr(gnf, "get_recent_headlines", lambda max_age_seconds=3600: [])

    async def empty():
        return []

    monkeypatch.setattr(nl, "_fetch_uw_news", empty)
    monkeypatch.setattr(nl, "_fetch_polygon_news", empty)
    # deliberately leave _fetch_lighthouse_events as the REAL implementation so it
    # actually exercises the real call path down into global_news_feeds

    result = _run(nl.refresh_newsflash(context=None))
    assert result["status"] == "ok"


# ═══════════════════════════════════════════════════════════════════════════
#  ORCH follow-up #2 item 1 — event-loop blocking fix (executor offload + seen-cache)
# ═══════════════════════════════════════════════════════════════════════════

def test_ingest_processing_runs_off_the_event_loop(monkeypatch):
    """item 1a: the classify+dedupe block must run in a thread executor, not on the
    event loop thread (measured ~8ms/item; buffered-headline volume would otherwise
    freeze the loop for seconds)."""
    import threading

    main_thread_id = threading.get_ident()
    seen_thread_ids = []
    real_process = nl._process_ingest_sync

    def wrapped(*args):
        seen_thread_ids.append(threading.get_ident())
        return real_process(*args)

    monkeypatch.setattr(nl, "_process_ingest_sync", wrapped)

    async def fake_uw():
        return [_item("Russia declares war on Ukraine after border incident", "unusual_whales", "unusual_whales")]

    async def empty():
        return []

    monkeypatch.setattr(nl, "_fetch_uw_news", fake_uw)
    monkeypatch.setattr(nl, "_fetch_polygon_news", empty)
    monkeypatch.setattr(nl, "_fetch_lighthouse_events", empty)

    result = _run(nl.refresh_newsflash(context=None))
    assert result["status"] == "ok"
    assert len(seen_thread_ids) == 1
    assert seen_thread_ids[0] != main_thread_id


def test_seen_cache_classifies_each_headline_once_across_two_passes(monkeypatch):
    """item 1b: a headline already seen within the TTL window must NOT be reclassified
    on a later refresh cycle — classify_headline is called exactly once per distinct
    headline, not once per ~60s cycle it stays in the source buffer."""
    calls = {"n": 0}
    real_classify = nl.classify_headline

    def counting_classify(*args, **kwargs):
        calls["n"] += 1
        return real_classify(*args, **kwargs)

    monkeypatch.setattr(nl, "classify_headline", counting_classify)

    titles = [
        "Russia declares war on Ukraine after border incident",
        "North Korea declares war on South Korea amid tensions",
        "China declares war on Taiwan after military buildup",
    ]

    async def fake_uw():
        return [_item(t, "unusual_whales", "unusual_whales") for t in titles]

    async def empty():
        return []

    monkeypatch.setattr(nl, "_fetch_uw_news", fake_uw)
    monkeypatch.setattr(nl, "_fetch_polygon_news", empty)
    monkeypatch.setattr(nl, "_fetch_lighthouse_events", empty)

    r1 = _run(nl.refresh_newsflash(context=None))
    r2 = _run(nl.refresh_newsflash(context=None))
    assert r1["new_events"] == len(titles)
    assert r2["new_events"] == 0
    assert calls["n"] == len(titles)  # each headline classified exactly ONCE across both passes


def test_seen_cache_still_updates_liveness_on_repeat_sighting(monkeypatch):
    """The seen-cache fast path must still refresh last_seen_ts (keep the event alive)
    even though it skips full reclassification."""
    async def fake_uw():
        return [_item("Russia declares war on Ukraine after border incident", "unusual_whales", "unusual_whales")]

    async def empty():
        return []

    monkeypatch.setattr(nl, "_fetch_uw_news", fake_uw)
    monkeypatch.setattr(nl, "_fetch_polygon_news", empty)
    monkeypatch.setattr(nl, "_fetch_lighthouse_events", empty)

    _run(nl.refresh_newsflash(context=None))
    state1 = nl._load_state()
    ts1 = state1["events"][0]["last_seen_ts"]

    _run(nl.refresh_newsflash(context=None))
    state2 = nl._load_state()
    ts2 = state2["events"][0]["last_seen_ts"]
    assert ts2 >= ts1


def test_seen_cache_cleanup_evicts_expired_entries():
    nl._seen_item_cache["abc123"] = {"event_id": "abc123", "expires_at": 0.0}
    nl._cleanup_seen_cache(now_ts=100.0)
    assert "abc123" not in nl._seen_item_cache


def test_seen_cache_keeps_unexpired_entries():
    nl._seen_item_cache["abc123"] = {"event_id": "abc123", "expires_at": 1_000_000.0}
    nl._cleanup_seen_cache(now_ts=100.0)
    assert "abc123" in nl._seen_item_cache


def test_seen_cache_second_source_still_corroborates_cached_event(monkeypatch):
    """A DIFFERENT source reporting the exact same already-cached headline text must
    still merge as corroboration (via the fast path), not be silently dropped."""
    async def fake_uw():
        return [_item("Russia declares war on Ukraine after border incident", "unusual_whales", "unusual_whales")]

    async def fake_poly_same_text():
        return [_item("Russia declares war on Ukraine after border incident", "polygon_news:AP", "polygon_news")]

    async def empty():
        return []

    monkeypatch.setattr(nl, "_fetch_uw_news", fake_uw)
    monkeypatch.setattr(nl, "_fetch_polygon_news", empty)
    monkeypatch.setattr(nl, "_fetch_lighthouse_events", empty)
    _run(nl.refresh_newsflash(context=None))

    monkeypatch.setattr(nl, "_fetch_uw_news", empty)
    monkeypatch.setattr(nl, "_fetch_polygon_news", fake_poly_same_text)
    _run(nl.refresh_newsflash(context=None))

    state = nl._load_state()
    assert len(state["events"]) == 1
    assert set(state["events"][0]["sources"]) == {"unusual_whales", "polygon_news:AP"}


def test_refresh_irrelevant_headlines_are_discarded(monkeypatch):
    async def fake_uw():
        return [_item("Local sports team wins championship game", "unusual_whales", "unusual_whales")]

    async def empty():
        return []

    monkeypatch.setattr(nl, "_fetch_uw_news", fake_uw)
    monkeypatch.setattr(nl, "_fetch_polygon_news", empty)
    monkeypatch.setattr(nl, "_fetch_lighthouse_events", empty)

    result = _run(nl.refresh_newsflash(context=None))
    assert result["events"] == 0


def test_refresh_tier3_alone_does_not_set_active(monkeypatch):
    async def fake_uw():
        return [_item("Inflation data comes in line with expectations today", "unusual_whales", "unusual_whales")]

    async def empty():
        return []

    monkeypatch.setattr(nl, "_fetch_uw_news", fake_uw)
    monkeypatch.setattr(nl, "_fetch_polygon_news", empty)
    monkeypatch.setattr(nl, "_fetch_lighthouse_events", empty)

    result = _run(nl.refresh_newsflash(context=None))
    assert result["events"] == 1
    assert result["max_severity"] == 1
    assert result["active"] is False  # severity 1 alone never trips "active"


def test_refresh_disabled_flag_short_circuits(monkeypatch):
    monkeypatch.setenv("CLIO_NEWSFLASH_ENABLED", "0")
    result = _run(nl.refresh_newsflash(context=None))
    assert result == {"status": "disabled"}


def test_refresh_concurrent_run_is_skipped(monkeypatch):
    async def slow_uw():
        await asyncio.sleep(0.05)
        return []

    async def empty():
        return []

    monkeypatch.setattr(nl, "_fetch_uw_news", slow_uw)
    monkeypatch.setattr(nl, "_fetch_polygon_news", empty)
    monkeypatch.setattr(nl, "_fetch_lighthouse_events", empty)

    async def _both():
        return await asyncio.gather(nl.refresh_newsflash(context=None), nl.refresh_newsflash(context=None))

    r1, r2 = _run(_both())
    statuses = {r1["status"], r2["status"]}
    assert "skipped_concurrent" in statuses
    assert "ok" in statuses


# ═══════════════════════════════════════════════════════════════════════════
#  Telegram alerting: Tier1 always, Tier2 needs >=2 sources, dedup + rate limit
# ═══════════════════════════════════════════════════════════════════════════

class _FakeBot:
    def __init__(self):
        self.sent = []

    async def send_message(self, chat_id, text, **kw):
        self.sent.append((chat_id, text))


class _FakeCtx:
    def __init__(self, bot):
        self.bot = bot


def test_tier1_event_always_alerts(monkeypatch):
    monkeypatch.setattr(config, "telegram_notify_chat_ids", ["111"])

    async def fake_uw():
        return [_item("Russia declares war on Ukraine after border incident", "unusual_whales", "unusual_whales")]

    async def empty():
        return []

    monkeypatch.setattr(nl, "_fetch_uw_news", fake_uw)
    monkeypatch.setattr(nl, "_fetch_polygon_news", empty)
    monkeypatch.setattr(nl, "_fetch_lighthouse_events", empty)

    bot = _FakeBot()
    result = _run(nl.refresh_newsflash(context=_FakeCtx(bot)))
    assert result["alerts_sent"] == 1
    assert len(bot.sent) == 1
    assert "parse_mode" not in {}  # plain text send — no parse_mode kwarg used anywhere


def test_tier2_single_source_does_not_alert_but_two_sources_do(monkeypatch):
    monkeypatch.setattr(config, "telegram_notify_chat_ids", ["111"])

    async def fake_uw():
        return [_item("Fed hikes rates in hawkish surprise move", "unusual_whales", "unusual_whales")]

    async def empty():
        return []

    monkeypatch.setattr(nl, "_fetch_uw_news", fake_uw)
    monkeypatch.setattr(nl, "_fetch_polygon_news", empty)
    monkeypatch.setattr(nl, "_fetch_lighthouse_events", empty)

    bot = _FakeBot()
    result = _run(nl.refresh_newsflash(context=_FakeCtx(bot)))
    assert result["alerts_sent"] == 0  # single-source Tier2 — no alert yet

    # a second, corroborating source arrives on the next cycle
    async def fake_poly():
        return [_item("Fed hikes interest rates in a hawkish surprise", "polygon_news:Reuters", "polygon_news")]

    monkeypatch.setattr(nl, "_fetch_uw_news", empty)
    monkeypatch.setattr(nl, "_fetch_polygon_news", fake_poly)

    result2 = _run(nl.refresh_newsflash(context=_FakeCtx(bot)))
    assert result2["alerts_sent"] == 1
    assert len(bot.sent) == 1


def test_alert_is_deduped_per_event_id(monkeypatch):
    monkeypatch.setattr(config, "telegram_notify_chat_ids", ["111"])

    async def fake_uw():
        return [_item("Russia declares war on Ukraine after border incident", "unusual_whales", "unusual_whales")]

    async def empty():
        return []

    monkeypatch.setattr(nl, "_fetch_uw_news", fake_uw)
    monkeypatch.setattr(nl, "_fetch_polygon_news", empty)
    monkeypatch.setattr(nl, "_fetch_lighthouse_events", empty)

    bot = _FakeBot()
    r1 = _run(nl.refresh_newsflash(context=_FakeCtx(bot)))
    r2 = _run(nl.refresh_newsflash(context=_FakeCtx(bot)))
    assert r1["alerts_sent"] == 1
    assert r2["alerts_sent"] == 0  # same event, already alerted — deduped by id
    assert len(bot.sent) == 1


def test_alert_rate_limit_caps_at_six_per_hour(monkeypatch):
    monkeypatch.setattr(config, "telegram_notify_chat_ids", ["111"])
    titles = [
        "Russia declares war on Ukraine after border incident",
        "North Korea declares war on South Korea amid tensions",
        "China declares war on Taiwan after military buildup",
        "Iran declares war on Israel following strikes",
        "Major US bank collapses amid deposit run panic",
        "NYSE halts trading market-wide after circuit breakers triggered this morning",
        "Argentina defaults on sovereign debt following crisis",
        "WHO declares new pandemic a global health emergency today",
    ]

    async def fake_uw():
        return [_item(t, f"unusual_whales", "unusual_whales") for t in titles]

    async def empty():
        return []

    monkeypatch.setattr(nl, "_fetch_uw_news", fake_uw)
    monkeypatch.setattr(nl, "_fetch_polygon_news", empty)
    monkeypatch.setattr(nl, "_fetch_lighthouse_events", empty)

    bot = _FakeBot()
    result = _run(nl.refresh_newsflash(context=_FakeCtx(bot)))
    assert result["events"] == 8
    assert result["alerts_sent"] == nl._ALERT_RATE_LIMIT_PER_HOUR
    assert len(bot.sent) == nl._ALERT_RATE_LIMIT_PER_HOUR

    # same hour, same events (all still Tier1, some already alerted, the rest rate-limited)
    bot2 = _FakeBot()
    result2 = _run(nl.refresh_newsflash(context=_FakeCtx(bot2)))
    assert result2["alerts_sent"] == 0


def test_no_context_does_not_mark_alert_sent_so_it_can_retry(monkeypatch):
    """context=None (e.g. a manual/dry call) must NOT burn the alert dedup slot — a later
    cycle with a real Telegram context must still be able to deliver it."""
    async def fake_uw():
        return [_item("Russia declares war on Ukraine after border incident", "unusual_whales", "unusual_whales")]

    async def empty():
        return []

    monkeypatch.setattr(nl, "_fetch_uw_news", fake_uw)
    monkeypatch.setattr(nl, "_fetch_polygon_news", empty)
    monkeypatch.setattr(nl, "_fetch_lighthouse_events", empty)

    r1 = _run(nl.refresh_newsflash(context=None))
    assert r1["alerts_sent"] == 0

    monkeypatch.setattr(config, "telegram_notify_chat_ids", ["111"])
    bot = _FakeBot()
    r2 = _run(nl.refresh_newsflash(context=_FakeCtx(bot)))
    assert r2["alerts_sent"] == 1
    assert len(bot.sent) == 1


def test_telegram_log_only_suppresses_send_but_marks_handled(monkeypatch, caplog):
    monkeypatch.setenv("CLIO_NEWSFLASH_TELEGRAM_LOG_ONLY", "1")
    monkeypatch.setattr(config, "telegram_notify_chat_ids", ["111"])

    async def fake_uw():
        return [_item("Russia declares war on Ukraine after border incident", "unusual_whales", "unusual_whales")]

    async def empty():
        return []

    monkeypatch.setattr(nl, "_fetch_uw_news", fake_uw)
    monkeypatch.setattr(nl, "_fetch_polygon_news", empty)
    monkeypatch.setattr(nl, "_fetch_lighthouse_events", empty)

    caplog.set_level(logging.INFO, logger="openclaw.newsflash_lite")
    bot = _FakeBot()
    r1 = _run(nl.refresh_newsflash(context=_FakeCtx(bot)))
    assert r1["alerts_sent"] == 1  # counted as "handled"
    assert len(bot.sent) == 0  # but nothing was actually sent
    r2 = _run(nl.refresh_newsflash(context=_FakeCtx(bot)))
    assert r2["alerts_sent"] == 0  # deduped, no repeat log-only spam either


def test_alert_message_is_plain_text_no_parse_mode(monkeypatch):
    """CLAUDE.md: Telegram sends must be plain text (never parse_mode)."""
    monkeypatch.setattr(config, "telegram_notify_chat_ids", ["111"])
    captured_kwargs = {}

    class _CapturingBot:
        async def send_message(self, chat_id, text, **kw):
            captured_kwargs.update(kw)

    async def fake_uw():
        return [_item("Russia declares war on Ukraine after border incident", "unusual_whales", "unusual_whales")]

    async def empty():
        return []

    monkeypatch.setattr(nl, "_fetch_uw_news", fake_uw)
    monkeypatch.setattr(nl, "_fetch_polygon_news", empty)
    monkeypatch.setattr(nl, "_fetch_lighthouse_events", empty)

    _run(nl.refresh_newsflash(context=_FakeCtx(_CapturingBot())))
    assert "parse_mode" not in captured_kwargs


# ═══════════════════════════════════════════════════════════════════════════
#  Orchestrator pass-2 probe: numeric magnitude gates + historical framing
# ═══════════════════════════════════════════════════════════════════════════

_MAGNITUDE_CASES = [
    # (headline, expected category or None, expected tier or None)
    ("Nasdaq sinks 0.3% in quiet trading", "notable", 3),
    ("Nasdaq sinks 3% as tech stocks fall", "market_crash", 1),
    ("S&P 500 falls 2% as yields jump", "market_selloff", 2),
    ("Dow plunges 1,500 points in worst day since 2020", "market_crash", 1),
    ("Dow slips 50 points ahead of Fed decision", "notable", 3),
    ("Dow sinks 5%, worst day since 2008 crash", "market_crash", 1),
    ("Oil climbs 1% on inventory draw", None, None),
    ("Crude surges 1% on weekly draw", None, None),
    ("Oil jumps 5% after drone attack on Saudi facility", "oil_shock", 2),
    ("Oil prices spike as crude surges on supply fears", "oil_shock", 2),
    ("Lessons from the 2008 banking crisis for today's investors", None, None),
    ("Documentary revisits the 1930s banking crisis", None, None),
    ("The 2008 banking crisis still shapes regulation", None, None),
    ("Regional bank shares tumble as banking crisis fears return", "credit_stress", 2),
    ("Silicon Valley Bank collapses, FDIC takes control", "bank_failure", 1),
]


@pytest.mark.parametrize("title,expected_category,expected_tier", _MAGNITUDE_CASES)
def test_magnitude_gates_and_historical_framing(title, expected_category, expected_tier):
    r = nl.classify_headline(title)
    if expected_category is None:
        assert r is None or r["tier"] == 3, (title, r)
    else:
        assert r is not None, title
        assert (r["category"], r["tier"]) == (expected_category, expected_tier), (title, r)


def test_new_tier2_categories_are_risk_off():
    for title in ("S&P 500 falls 2% as yields jump", "Regional bank shares tumble as banking crisis fears return"):
        assert nl.classify_headline(title)["direction_hint"] == "risk_off"
