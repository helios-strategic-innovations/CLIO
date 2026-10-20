"""TP-1 (S211): trade-policy weekly check + heartbeat.

trade_policy_state.json was never produced: news_intel's S166 forced escalation handed over the topic's GENERAL
analysis and trade_policy's specificity gate (country + action verb) always rejected it; "no file" was then read as
"never checked" (sentinel MISSING x13/day, a forced attempt every day). Now a forced check runs its own targeted
search and either escalates a genuinely specific story through the UNCHANGED gate or records a heartbeat
("checked, nothing specific": no model call, no Telegram). No network, no live state: everything under tmp_path.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from agent.integrations import news_intel as ni
from agent.integrations import staleness_sentinel as sentinel
from agent.integrations import trade_policy as tp

NY = ZoneInfo("America/New_York")
SPECIFIC = "China announced new tariffs on steel imports, effective next month."
GENERAL = "Markets were mixed on Tuesday as investors weighed growth and inflation."
FNAME = "trade_policy_state.json"


@pytest.fixture()
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(tp, "_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(tp, "RESEARCH_STATE_PATH", str(tmp_path / FNAME))
    monkeypatch.setattr(ni, "_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(sentinel, "_INTEL_DIR", str(tmp_path))
    return tmp_path


def _state(state_dir):
    return json.loads((state_dir / FNAME).read_text())


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def no_pipeline(monkeypatch):
    """Any attempt to run the expensive pipeline (research / model / delivery) is a test failure."""
    async def boom(*_a, **_k):
        raise AssertionError("the pipeline must not run")

    for name in ("_gather_impact_chain_research", "_generate_impact_chain_analysis", "_deliver_trade_policy_intel"):
        monkeypatch.setattr(tp, name, boom)


# ------------------------------------------------------------------ the forced pick
@pytest.fixture()
def search(monkeypatch):
    calls = []
    hits = {"list": []}

    async def fake_search_web(query, max_results=5):
        calls.append(query)
        return list(hits["list"])

    async def allow(_name):
        return True, ""

    async def record(_name):
        return None

    async def no_sleep(_s):
        return None

    from agent.integrations import news_update, web_search

    monkeypatch.setattr(web_search, "search_web", fake_search_web)
    monkeypatch.setattr(tp.budget_tracker, "check_budget", allow)
    monkeypatch.setattr(tp.budget_tracker, "record_usage", record)
    monkeypatch.setattr(tp.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(news_update, "_url_is_stale", lambda url: False)
    monkeypatch.setattr(news_update, "_snippet_is_stale", lambda title, snippet: False)
    return SimpleNamespace(calls=calls, hits=hits)


def test_pick_returns_the_first_specific_hit_and_puts_it_first(search):
    search.hits["list"] = [
        {"title": "Stocks drift", "body": GENERAL, "href": "https://e.test/a"},
        {"title": "China tariffs", "body": "China announced new tariffs on steel imports, effective next month.", "href": "https://e.test/b"},
    ]
    text, results = _run(tp._forced_policy_pick())
    assert "China" in text and "tariffs" in text
    assert results[0]["url"] == "https://e.test/b"                       # the chosen item leads
    assert len(search.calls) == len(tp._FORCED_QUERIES)                  # two targeted queries, no more
    assert all(r["query_type"] == "forced_check" for r in results)


def test_pick_is_none_when_nothing_specific(search):
    search.hits["list"] = [{"title": "Stocks drift", "body": GENERAL, "href": "https://e.test/a"}]
    assert _run(tp._forced_policy_pick()) is None


def test_pick_is_none_on_empty_or_failing_search(search, monkeypatch):
    assert _run(tp._forced_policy_pick()) is None                        # no hits at all
    from agent.integrations import web_search

    async def broken(*_a, **_k):
        raise RuntimeError("search down")

    monkeypatch.setattr(web_search, "search_web", broken)
    assert _run(tp._forced_policy_pick()) is None                        # failures degrade to "nothing found"


def test_pick_stops_when_the_search_budget_is_exhausted(search, monkeypatch):
    async def deny(_n):
        return False, "exhausted"

    monkeypatch.setattr(tp.budget_tracker, "check_budget", deny)
    assert _run(tp._forced_policy_pick()) is None
    assert search.calls == []


# ------------------------------------------------------------------ the forced gate is stricter than escalate()'s
# 2026-09-29: the first live check picked a hit that had only a country and the word "tariff" (the query's own
# word), ran the full pipeline on it and sent Telegram a speculative "EU tariff, rate/goods/authority all
# unspecified" analysis. A candidate now needs an event verb AND a concrete parameter.
WEAK = "EU tariff: what businesses need to know. Trade tensions between the US and the European Union continue."


@pytest.mark.parametrize("text", [
    "China announced new tariffs on steel imports, effective next month.",                       # verb + goods
    "The US imposed a 25% tariff on Chinese imports.",                                           # verb + rate
    "The White House announced tariffs on Japan under Section 232 of the trade act.",            # verb + instrument
    "Trump signed an executive order imposing duties on Canada.",                                # verb + instrument
    "Mexico said it would impose 15 percent duties on imports from Brazil.",                     # rate spelled out
    "Section 301 tariffs on China were announced.",                                              # instrument only
    "The White House imposed a tariff on Canada under IEEPA.",                                   # named instrument: IEEPA
    "A proclamation setting a tariff for Canada was announced.",                                 # named instrument: proclamation
    "Trump signed a Section 232 tariff proclamation for Canada.",                                # 'signed' is the only verb
    "Trump announces 25% tariffs on imported steel from Canada",                                 # plural + 'announces'
    "Commerce Department raises tariff on Chinese solar panels to 50%",                          # 'raises'
    "India hikes tariff on US almonds and apples to 20 percent",                                 # 'hikes'
    "China imposes export controls on rare earth materials, effective December 1",               # policy term + goods
    "Treasury sanctions Russian oil companies in new executive order",                           # sanctions + instrument
    "Brazil levies duties of 30% on imported chemicals",                                         # 'levies' + duties
    "Washington slaps sanctions on Iranian steel exporters",                                     # 'slaps'
    "Mexico imposed a 10% duty on Brazilian imports.",                                           # policy term: duty
    "Japan announced import quotas on Korean steel.",                                            # policy term: quota
    "Russia announced an embargo on Polish dairy.",                                              # policy term: embargo
    "China announces an import ban on Japanese steel.",                                          # policy term: import ban
    "Canada levied a 15% levy on American goods.",                                               # policy term: levy
    "Washington slaps a 25% tariff on Brazilian steel",                                          # 'slaps' is the only verb
])
def test_concrete_gate_accepts_a_story_with_an_event_verb_and_a_parameter(text):
    assert tp._is_concrete_policy(text) is True


@pytest.mark.parametrize("text", [
    WEAK,                                                                                        # country + "tariff", no verb
    "The European Union said a tariff was announced.",                                            # verb but nothing concrete
    "China announced new measures.",                                                             # verb, no tariff keyword
    "Steel prices rose as demand for steel in China recovered.",                                 # goods, no verb/keyword
    "The EU announced a tariff affecting billions of dollars of trade.",                         # 'bill' in 'billions' is not a parameter
    "China banking regulators discussed a tariff on steel.",                                     # 'banking' is not 'ban'
    "China tariff rate of 25% on steel exports remains a talking point.",                        # parameters but no event verb
    "A 25% tariff was imposed on steel imports.",                                                # no country
    "Indiana steelmakers welcomed a 25% tariff that was imposed on imports.",                    # 'indian' inside 'Indiana' is not India
    "China announced new steel production targets, effective 2027",                              # parameters + verb, no policy term
    "Trump threatens new tariffs on Mexico over border policy",                                  # a threat: no event verb, no parameter
    "Section 232 investigation into semiconductor imports opens, tariffs possible",              # an investigation, not an action
    "Markets were mixed on Tuesday as investors weighed growth and inflation.",                  # general text
    "",                                                                                          # empty
])
def test_concrete_gate_rejects_weak_or_general_text(text):
    assert tp._is_concrete_policy(text) is False


def test_concrete_gate_is_independent_of_the_exact_form_escalation_gate():
    assert tp._is_specific_policy(WEAK) is True and tp._is_concrete_policy(WEAK) is False   # the gap this closes
    plural = "Trump announces 25% tariffs on imported steel from Canada"
    assert tp._is_concrete_policy(plural) is True and tp._is_specific_policy(plural) is False   # what layering rejected


def test_pick_skips_a_weak_hit_and_takes_a_later_concrete_one(search):
    search.hits["list"] = [
        {"title": "EU tariffs explained", "body": WEAK, "href": "https://e.test/weak"},
        {"title": "US tariff", "body": "The US imposed a 25% tariff on Chinese imports.", "href": "https://e.test/strong"},
    ]
    text, results = _run(tp._forced_policy_pick())
    assert "25%" in text and results[0]["url"] == "https://e.test/strong"


def test_a_weak_only_result_is_a_heartbeat_not_a_pipeline_run(search, state_dir, no_pipeline, caplog):
    search.hits["list"] = [{"title": "EU tariffs explained", "body": WEAK, "href": "https://e.test/weak"}]
    with caplog.at_level(logging.INFO, logger="openclaw.trade_policy"):
        _run(tp.escalate(SimpleNamespace(bot=None), GENERAL, [], forced=True))
    st = _state(state_dir)
    assert st["last_checked"]["result"] == "none_specific" and st["last_checked"]["forced"] is True
    assert "researched_policies" not in st or st["researched_policies"] == []
    assert "no specific policy action found" in caplog.text


# ------------------------------------------------------------------ escalate(forced=True)
def test_forced_check_with_nothing_specific_writes_a_heartbeat_and_nothing_else(state_dir, no_pipeline, monkeypatch, caplog):
    async def nothing():
        return None

    monkeypatch.setattr(tp, "_forced_policy_pick", nothing)
    with caplog.at_level(logging.INFO, logger=tp.logger.name):
        assert _run(tp.escalate(None, "", [], forced=True)) is None
    st = _state(state_dir)
    assert st["last_checked"]["result"] == "none_specific" and st["last_checked"]["forced"] is True
    assert st["researched_policies"] == []
    datetime.fromisoformat(st["last_updated"])                            # parseable ISO
    assert "heartbeat recorded" in caplog.text


@pytest.mark.parametrize("story,country", [
    (SPECIFIC, "china"),
    ("Trump announces 25% tariffs on imported steel from Canada", "canada"),   # plural/'announces': the old gate rejects it
])
def test_forced_check_with_a_specific_story_runs_the_pipeline_and_keeps_the_heartbeat(state_dir, monkeypatch, story, country):
    async def pick():
        return story, [{"title": "t", "snippet": "s", "url": "https://e.test/b"}]

    delivered = []

    async def gather(*_a, **_k):
        return [{"title": "r", "snippet": "s", "url": "https://e.test/r", "query_type": "x"}]

    async def analysis(**_k):
        return "impact chain text"

    async def deliver(context, policy_type, analysis_text, urls):
        delivered.append((policy_type, analysis_text, urls))

    monkeypatch.setattr(tp, "_forced_policy_pick", pick)
    monkeypatch.setattr(tp, "_gather_impact_chain_research", gather)
    monkeypatch.setattr(tp, "_recall_historical_analogs", lambda country, policy_type: "")
    monkeypatch.setattr(tp, "_assess_policy_validity", lambda policy_type, details: "")
    monkeypatch.setattr(tp, "_generate_impact_chain_analysis", analysis)
    monkeypatch.setattr(tp, "_deliver_trade_policy_intel", deliver)
    _run(tp.escalate(None, "", [], forced=True))

    assert len(delivered) == 1 and delivered[0][1] == "impact chain text"
    st = _state(state_dir)
    assert len(st["researched_policies"]) == 1
    assert st["last_checked"]["result"] == "candidate_found"              # not clobbered by the final save
    assert st["researched_policies"][0]["country"].lower() == country


def test_forced_check_that_exits_early_still_counts_as_checked(state_dir, no_pipeline, monkeypatch):
    """A specific candidate that was researched recently: the pipeline returns early, the check still counts."""
    text = SPECIFIC
    seed = {"researched_policies": [{
        "policy_type": tp.classify_policy(text), "country": tp._extract_policy_details(text)["country"],
        "researched_at": datetime.now(NY).isoformat(),
    }]}
    (state_dir / FNAME).write_text(json.dumps(seed))

    async def pick():
        return text, [{"title": "t", "snippet": "s", "url": "https://e.test/b"}]

    monkeypatch.setattr(tp, "_forced_policy_pick", pick)
    _run(tp.escalate(None, "", [], forced=True))                          # no_pipeline: any research/LLM call fails
    st = _state(state_dir)
    assert st["last_checked"]["result"] == "candidate_found"
    assert len(st["researched_policies"]) == 1                            # nothing new was added


def test_unforced_general_text_touches_no_state(state_dir, no_pipeline):
    _run(tp.escalate(None, GENERAL, []))
    assert not (state_dir / FNAME).exists()


# ------------------------------------------------------------------ state helpers
def test_heartbeat_preserves_existing_analyses_and_survives_prune(state_dir):
    recent = datetime.now(NY).isoformat()
    old = (datetime.now(NY) - timedelta(days=45)).isoformat()
    (state_dir / FNAME).write_text(json.dumps({"researched_policies": [
        {"policy_type": "tariff", "country": "China", "researched_at": recent},
        {"policy_type": "tariff", "country": "Mexico", "researched_at": old},       # pruned (>30 d)
    ]}))
    tp._record_check_sync("none_specific")
    st = _state(state_dir)
    assert [p["country"] for p in st["researched_policies"]] == ["China"]
    assert st["last_checked"]["result"] == "none_specific" and "last_updated" in st
    tp._record_check_sync("candidate_found")                                # a second heartbeat replaces, never accumulates
    assert _state(state_dir)["last_checked"]["result"] == "candidate_found"
    assert len(_state(state_dir)["researched_policies"]) == 1


# ------------------------------------------------------------------ the two staleness readers
def test_news_intel_staleness_sees_the_heartbeat(state_dir):
    assert ni._is_trade_policy_stale(7) is True                            # no file
    tp._record_check_sync("none_specific")
    assert ni._is_trade_policy_stale(7) is False                           # heartbeat refreshed the mtime
    ten_days = time.time() - 10 * 86400
    os.utime(state_dir / FNAME, (ten_days, ten_days))
    assert ni._is_trade_policy_stale(7) is True                            # a real stall still shows


def test_sentinel_reads_the_heartbeat_as_fresh_and_a_stall_as_stale(state_dir):
    assert sentinel._scan_one(FNAME)["status"] == "MISSING"
    tp._record_check_sync("none_specific")
    rep = sentinel._scan_one(FNAME)
    assert rep["status"] == "FRESH" and rep["source"] == "internal:last_updated"
    st = _state(state_dir)
    st["last_updated"] = (datetime.now(NY) - timedelta(days=8)).isoformat()   # SLA is 7 days
    (state_dir / FNAME).write_text(json.dumps(st))
    assert sentinel._scan_one(FNAME)["status"] == "STALE"


# ------------------------------------------------------------------ the once-per-cycle trigger
@pytest.fixture()
def trigger(monkeypatch):
    monkeypatch.setattr(ni, "_escalation_counts", {})
    calls = []

    async def fake_escalate(context, analysis, results, forced=False):
        calls.append((context, analysis, results, forced))

    monkeypatch.setattr(tp, "escalate", fake_escalate)
    return calls


def test_check_runs_forced_when_stale_and_charges_the_daily_counter(trigger, monkeypatch):
    monkeypatch.setattr(ni, "_is_trade_policy_stale", lambda days=7: days == 7)
    _run(ni._maybe_check_trade_policy("CTX"))
    assert trigger == [("CTX", "", [], True)]
    assert ni._escalation_counts["trade_policy"][0] == 1


def test_check_is_skipped_when_fresh(trigger, monkeypatch):
    monkeypatch.setattr(ni, "_is_trade_policy_stale", lambda days=7: False)
    _run(ni._maybe_check_trade_policy("CTX"))
    assert trigger == [] and "trade_policy" not in ni._escalation_counts


def test_check_respects_the_daily_escalation_limit(trigger, monkeypatch):
    monkeypatch.setattr(ni, "_is_trade_policy_stale", lambda days=7: True)
    for _ in range(ni._MAX_ESCALATIONS_PER_DAY):
        _run(ni._maybe_check_trade_policy("CTX"))
    _run(ni._maybe_check_trade_policy("CTX"))                              # over the limit
    assert len(trigger) == ni._MAX_ESCALATIONS_PER_DAY


def test_check_never_raises_and_logs_a_warning(monkeypatch, caplog):
    monkeypatch.setattr(ni, "_escalation_counts", {})
    monkeypatch.setattr(ni, "_is_trade_policy_stale", lambda days=7: True)

    async def boom(*_a, **_k):
        raise RuntimeError("search exploded")

    monkeypatch.setattr(tp, "escalate", boom)
    with caplog.at_level(logging.WARNING, logger=ni.logger.name):
        _run(ni._maybe_check_trade_policy("CTX"))
    assert "weekly trade-policy check failed" in caplog.text


def test_topic_monitor_runs_the_check_before_the_topic_loop_even_with_no_topics(monkeypatch):
    """The old forced path only ran after a topic delivered content (three `continue` exits precede it)."""
    order = []

    class FixedDT(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 29, 10, 0, tzinfo=tz)

    async def check(context):
        order.append("check")

    monkeypatch.setattr(ni, "config", SimpleNamespace(news_intel_enabled=True))
    monkeypatch.setattr(ni, "datetime", FixedDT)
    monkeypatch.setattr(ni, "_load_intel_state", lambda: {"items": [], "last_topics": []})
    monkeypatch.setattr(ni, "_prune_intel_state", lambda s: s)
    monkeypatch.setattr(ni, "_get_next_topics", lambda s: order.append("topics") or [])
    monkeypatch.setattr(ni, "_save_intel_state", lambda s: None)
    monkeypatch.setattr(ni, "_maybe_check_trade_policy", check)
    _run(ni.run_topic_monitor(None))
    assert order == ["topics", "check"]


def test_the_old_content_dependent_forced_branch_is_gone():
    import inspect

    src = inspect.getsource(ni.run_topic_monitor)
    assert "Forcing trade_policy escalation" not in src
    assert "_maybe_check_trade_policy(context)" in src
