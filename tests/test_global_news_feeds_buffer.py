"""
Tests for the recent-headlines buffer added to agent/integrations/global_news_feeds.py
(ORCH-7Q4K-DPOF follow-up "Lighthouse headline starvation", 2026-09-25).

Multiple callers (learning_engine, market_events.check_rss_breaking_news, DPOF's Jev
refresher, and formerly a 4th racer in newsflash_lite.py) all called
scan_current_window(), whose return value only ever contains headlines no earlier caller
has already been handed (HeadlineManager._seen_keys). get_recent_headlines() is a
separate, non-consuming, ADDITIVE view populated as a side effect of scan_current_window()
so every caller sees the same shared recent-headline set regardless of who triggered the
underlying scan first.

Hermetic: no network. Builds real ParsedHeadline objects directly and, for the
"return value unchanged" test, drives scan_current_window()'s real code path with a fake
parser/scheduler rather than mocking scan_current_window() itself.
"""

from datetime import datetime, timedelta, timezone

import pytest

import agent.integrations.global_news_feeds as gnf
from agent.security import sanitize_ingested_content


def _make_headline(title, source_id="reuters_world", source_name="Reuters World",
                    tier="T1_VERIFIED", age_hours=0.0):
    wrapped = sanitize_ingested_content(title, source=f"rss:{source_id}")
    published = datetime.now(timezone.utc) - timedelta(hours=age_hours)
    return gnf.ParsedHeadline(
        title=wrapped, summary="", source_name=source_name, source_id=source_id,
        source_region="uk", reliability_tier=tier, published_utc=published,
        hours_until_us_open=0.0, feed_url="http://example.com/feed", link="http://example.com/a",
        is_state_media=False, relevance_tags=["geopolitical"],
    )


@pytest.fixture(autouse=True)
def _reset_buffer():
    """The buffer is module-level/process-global by design (mirrors HeadlineManager's own
    _seen_keys) — reset it before and after every test so tests never leak into each other."""
    with gnf._recent_buffer_lock:
        gnf._recent_buffer.clear()
        gnf._last_scan_time = None
    yield
    with gnf._recent_buffer_lock:
        gnf._recent_buffer.clear()
        gnf._last_scan_time = None


# ═══════════════════════════════════════════════════════════════════════════
#  _add_to_recent_buffer / get_recent_headlines / get_last_scan_time
# ═══════════════════════════════════════════════════════════════════════════

def test_buffer_accumulates_across_two_scans():
    """Two separate scan-equivalent calls (simulating two DIFFERENT callers'
    scan_current_window() invocations, e.g. learning_engine then market_events) both
    land in the shared buffer — the second caller's scan does not need to have
    re-discovered the first caller's headline."""
    gnf._add_to_recent_buffer([_make_headline("Russia declares war on Ukraine")])
    gnf._add_to_recent_buffer([_make_headline("Fed hikes rates", source_id="bbc_world",
                                               source_name="BBC World")])
    recent = gnf.get_recent_headlines(max_age_seconds=3600)
    assert len(recent) == 2
    assert {h.source_id for h in recent} == {"reuters_world", "bbc_world"}


def test_buffer_dedupes_by_dedup_key():
    hl = _make_headline("Russia declares war on Ukraine")
    gnf._add_to_recent_buffer([hl])
    gnf._add_to_recent_buffer([hl])  # e.g. a later scan re-parses the same still-live item
    assert len(gnf.get_recent_headlines()) == 1


def test_buffer_dedupes_across_calls_with_differently_cased_whitespace():
    """dedup_key normalizes case/whitespace — two ParsedHeadline objects for what is
    really the same story (e.g. re-fetched with trailing whitespace) must not duplicate."""
    hl_a = _make_headline("Russia declares war on Ukraine")
    hl_b = _make_headline("Russia   declares WAR on Ukraine  ")
    gnf._add_to_recent_buffer([hl_a])
    gnf._add_to_recent_buffer([hl_b])
    assert len(gnf.get_recent_headlines()) == 1


def test_buffer_ages_out_old_headlines():
    gnf._add_to_recent_buffer([_make_headline("Old story", age_hours=gnf._MAX_HEADLINE_AGE_HOURS + 1)])
    assert gnf.get_recent_headlines(max_age_seconds=999999) == []


def test_get_recent_headlines_respects_max_age_seconds_param():
    gnf._add_to_recent_buffer([_make_headline("Recent story", source_id="a", age_hours=0.05)])
    gnf._add_to_recent_buffer([_make_headline("Older story", source_id="b", age_hours=2.0)])
    within_1h = gnf.get_recent_headlines(max_age_seconds=3600)
    assert [h.source_id for h in within_1h] == ["a"]
    within_3h = gnf.get_recent_headlines(max_age_seconds=3 * 3600)
    assert {h.source_id for h in within_3h} == {"a", "b"}


def test_get_recent_headlines_newest_first():
    gnf._add_to_recent_buffer([_make_headline("Oldest", source_id="a", age_hours=2.0)])
    gnf._add_to_recent_buffer([_make_headline("Newest", source_id="b", age_hours=0.0)])
    gnf._add_to_recent_buffer([_make_headline("Middle", source_id="c", age_hours=1.0)])
    recent = gnf.get_recent_headlines(max_age_seconds=999999)
    assert [h.source_id for h in recent] == ["b", "c", "a"]


def test_buffer_capped_at_max_size(monkeypatch):
    monkeypatch.setattr(gnf, "_RECENT_BUFFER_CAP", 5)
    headlines = [_make_headline(f"Distinct market story number {i}", source_id=f"feed{i}")
                 for i in range(10)]
    gnf._add_to_recent_buffer(headlines)
    assert len(gnf.get_recent_headlines(max_age_seconds=999999)) == 5


def test_get_recent_headlines_never_raises_on_lock_failure(monkeypatch):
    monkeypatch.setattr(gnf, "_recent_buffer_lock", None)  # 'with None:' would raise
    assert gnf.get_recent_headlines() == []


def test_get_recent_headlines_returns_empty_list_when_buffer_empty():
    assert gnf.get_recent_headlines() == []


def test_last_scan_time_none_before_any_scan():
    assert gnf.get_last_scan_time() is None


def test_last_scan_time_updated_after_add():
    gnf._add_to_recent_buffer([])
    assert gnf.get_last_scan_time() is not None


def test_last_scan_time_advances_on_each_add():
    gnf._add_to_recent_buffer([])
    t1 = gnf.get_last_scan_time()
    gnf._add_to_recent_buffer([])
    t2 = gnf.get_last_scan_time()
    assert t2 >= t1


# ═══════════════════════════════════════════════════════════════════════════
#  scan_current_window() itself: return value + behavior unchanged for existing callers
# ═══════════════════════════════════════════════════════════════════════════

def test_scan_current_window_return_value_unchanged(monkeypatch):
    """The recent-headlines buffer is a pure side effect — scan_current_window()'s
    return value must still be exactly what HeadlineManager.process(raw) produces,
    with the buffer populated ADDITIONALLY, not instead."""
    monkeypatch.setenv("CLIO_LIGHTHOUSE_ENABLED", "1")
    monkeypatch.setenv("CLIO_RSS_FEED_PARSER_ENABLED", "1")
    monkeypatch.setattr(gnf._scheduler, "get_active_window", lambda: "premarket")
    fake_feed_meta = {"id": "reuters_world_test", "name": "Reuters World", "region": "uk",
                       "tier": "T1_VERIFIED"}
    monkeypatch.setattr(gnf._scheduler, "get_feeds_for_window", lambda w: [fake_feed_meta])
    monkeypatch.setattr(gnf, "_feed_health", {})  # fresh — not quarantined from other tests

    hl = _make_headline("Russia declares war on Ukraine", source_id="reuters_world_test")

    class _FakeParser:
        def fetch_feed(self, feed_meta):
            assert feed_meta is fake_feed_meta
            return [hl]

    monkeypatch.setattr(gnf, "_get_parser", lambda: _FakeParser())
    monkeypatch.setattr(gnf, "_headline_mgr", gnf.HeadlineManager())  # fresh seen-state

    result = gnf.scan_current_window()

    assert result == [hl]  # byte-identical to the pre-buffer behavior
    # AND the buffer side effect also happened
    assert gnf.get_recent_headlines()[0].source_id == "reuters_world_test"


def test_scan_current_window_empty_result_still_updates_last_scan_time(monkeypatch):
    monkeypatch.setenv("CLIO_LIGHTHOUSE_ENABLED", "1")
    monkeypatch.setenv("CLIO_RSS_FEED_PARSER_ENABLED", "1")
    monkeypatch.setattr(gnf._scheduler, "get_active_window", lambda: "premarket")
    fake_feed_meta = {"id": "reuters_world_test2", "name": "Reuters World", "region": "uk",
                       "tier": "T1_VERIFIED"}
    monkeypatch.setattr(gnf._scheduler, "get_feeds_for_window", lambda w: [fake_feed_meta])
    monkeypatch.setattr(gnf, "_feed_health", {})

    class _EmptyParser:
        def fetch_feed(self, feed_meta):
            return []

    monkeypatch.setattr(gnf, "_get_parser", lambda: _EmptyParser())
    monkeypatch.setattr(gnf, "_headline_mgr", gnf.HeadlineManager())

    result = gnf.scan_current_window()
    assert result == []
    assert gnf.get_last_scan_time() is not None


def test_scan_current_window_disabled_flag_does_not_touch_buffer(monkeypatch):
    monkeypatch.setenv("CLIO_LIGHTHOUSE_ENABLED", "0")
    result = gnf.scan_current_window()
    assert result == []
    assert gnf.get_last_scan_time() is None  # never actually scanned — no side effect


# ═══════════════════════════════════════════════════════════════════════════
#  refresh_recent_headlines() — side-effect-only driver, NEVER touches _seen_keys
#  (ORCH follow-up #2 item 4: "residual starvation for learning_engine")
# ═══════════════════════════════════════════════════════════════════════════

def test_refresh_recent_headlines_does_not_consume_seen_keys(monkeypatch):
    """refresh_recent_headlines() must NOT touch HeadlineManager._seen_keys — a
    SUBSEQUENT scan_current_window() call (e.g. learning_engine's own scan) must still
    return the SAME headlines, proving they were never 'consumed' by the refresh."""
    monkeypatch.setenv("CLIO_LIGHTHOUSE_ENABLED", "1")
    monkeypatch.setenv("CLIO_RSS_FEED_PARSER_ENABLED", "1")
    monkeypatch.setattr(gnf._scheduler, "get_active_window", lambda: "premarket")
    fake_feed_meta = {"id": "test_feed_refresh", "name": "Test Feed", "region": "uk", "tier": "T1_VERIFIED"}
    monkeypatch.setattr(gnf._scheduler, "get_feeds_for_window", lambda w: [fake_feed_meta])
    monkeypatch.setattr(gnf, "_feed_health", {})
    monkeypatch.setattr(gnf, "_headline_mgr", gnf.HeadlineManager())

    hl = _make_headline("Russia declares war on Ukraine", source_id="test_feed_refresh")

    class _FakeParser:
        def fetch_feed(self, feed_meta):
            return [hl]

    monkeypatch.setattr(gnf, "_get_parser", lambda: _FakeParser())

    added = gnf.refresh_recent_headlines()
    assert added == 1
    assert len(gnf.get_recent_headlines()) == 1

    # a SUBSEQUENT scan_current_window() (simulating learning_engine's own call) STILL
    # returns it -- it was never marked "seen" by refresh_recent_headlines().
    result = gnf.scan_current_window()
    assert result == [hl]


def test_refresh_recent_headlines_applies_age_and_relevance_filtering(monkeypatch):
    monkeypatch.setenv("CLIO_LIGHTHOUSE_ENABLED", "1")
    monkeypatch.setenv("CLIO_RSS_FEED_PARSER_ENABLED", "1")
    monkeypatch.setattr(gnf._scheduler, "get_active_window", lambda: "premarket")
    fake_feed_meta = {"id": "test_feed_filter", "name": "Test Feed", "region": "uk", "tier": "T1_VERIFIED"}
    monkeypatch.setattr(gnf._scheduler, "get_feeds_for_window", lambda w: [fake_feed_meta])
    monkeypatch.setattr(gnf, "_feed_health", {})

    fresh_relevant = _make_headline("Fresh relevant story", source_id="a")
    stale = _make_headline("Stale story", source_id="b", age_hours=gnf._MAX_HEADLINE_AGE_HOURS + 1)
    no_relevance = _make_headline("Irrelevant story", source_id="c")
    no_relevance.relevance_tags = []

    class _FakeParser:
        def fetch_feed(self, feed_meta):
            return [fresh_relevant, stale, no_relevance]

    monkeypatch.setattr(gnf, "_get_parser", lambda: _FakeParser())

    added = gnf.refresh_recent_headlines()
    assert added == 1
    buffered = gnf.get_recent_headlines(max_age_seconds=999999)
    assert [h.source_id for h in buffered] == ["a"]


def test_refresh_recent_headlines_disabled_flag_returns_zero(monkeypatch):
    monkeypatch.setenv("CLIO_LIGHTHOUSE_ENABLED", "0")
    assert gnf.refresh_recent_headlines() == 0


def test_refresh_recent_headlines_no_active_window_returns_zero(monkeypatch):
    monkeypatch.setenv("CLIO_LIGHTHOUSE_ENABLED", "1")
    monkeypatch.setenv("CLIO_RSS_FEED_PARSER_ENABLED", "1")
    monkeypatch.setattr(gnf._scheduler, "get_active_window", lambda: None)
    assert gnf.refresh_recent_headlines() == 0


def test_refresh_recent_headlines_never_raises_on_fetch_error(monkeypatch):
    monkeypatch.setenv("CLIO_LIGHTHOUSE_ENABLED", "1")
    monkeypatch.setenv("CLIO_RSS_FEED_PARSER_ENABLED", "1")
    monkeypatch.setattr(gnf._scheduler, "get_active_window", lambda: "premarket")

    def raising(w):
        raise RuntimeError("boom")

    monkeypatch.setattr(gnf._scheduler, "get_feeds_for_window", raising)
    assert gnf.refresh_recent_headlines() == 0


def test_add_to_recent_buffer_returns_new_count():
    hl1 = _make_headline("Russia declares war on Ukraine")
    hl2 = _make_headline("Fed hikes rates", source_id="bbc_world")
    assert gnf._add_to_recent_buffer([hl1]) == 1
    assert gnf._add_to_recent_buffer([hl1]) == 0  # duplicate — nothing new
    assert gnf._add_to_recent_buffer([hl1, hl2]) == 1  # only hl2 is new
