"""
Tests that agent/integrations/market_events.py::check_rss_breaking_news() builds its
crisis cluster from the shared recent-headlines buffer (global_news_feeds.
get_recent_headlines()) and drives the scan via refresh_recent_headlines() rather than
scan_current_window() (ORCH-7Q4K-DPOF follow-up "Lighthouse headline starvation", plus
follow-up #2 item 4 "residual starvation for learning_engine", 2026-09-25).

Hermetic: no network. refresh_recent_headlines() is monkeypatched as a no-op driver (its
own fetch/filter behavior is covered by tests/test_global_news_feeds_buffer.py), while
the buffer is populated directly via _add_to_recent_buffer() (simulating an EARLIER
caller, e.g. learning_engine's own scan_current_window() call, having already scanned and
"consumed" the same headlines via HeadlineManager._seen_keys).
"""

from datetime import datetime, timezone

import pytest

import agent.integrations.global_news_feeds as gnf
import agent.integrations.market_events as me
from agent.security import sanitize_ingested_content


def _make_crisis_headline(title, source_id, source_name, tier="T1_VERIFIED"):
    wrapped = sanitize_ingested_content(title, source=f"rss:{source_id}")
    return gnf.ParsedHeadline(
        title=wrapped, summary="", source_name=source_name, source_id=source_id,
        source_region="uk", reliability_tier=tier, published_utc=datetime.now(timezone.utc),
        hours_until_us_open=0.0, feed_url="http://example.com/feed", link="http://example.com/a",
        is_state_media=False,
    )


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setenv("CLIO_RSS_BREAKING_ENABLED", "1")
    monkeypatch.setenv("CLIO_RSS_BREAKING_LOG_ONLY", "0")
    monkeypatch.setattr(me, "_is_market_hours", lambda: True)
    monkeypatch.setattr(gnf, "refresh_recent_headlines", lambda: 0)  # default no-op driver
    with gnf._recent_buffer_lock:
        gnf._recent_buffer.clear()
        gnf._last_scan_time = None
    yield
    with gnf._recent_buffer_lock:
        gnf._recent_buffer.clear()
        gnf._last_scan_time = None


def test_cluster_sees_headlines_already_consumed_by_another_caller():
    """The exact starvation scenario: an earlier caller (e.g. learning_engine) already
    'consumed' these headlines via its own scan_current_window() call, but both T1 crisis
    headlines are still visible via the shared buffer, so the cluster check still fires."""
    h1 = _make_crisis_headline("Nuclear strike reported in border region",
                                "reuters_world", "Reuters World")
    h2 = _make_crisis_headline("Military strike confirmed near capital",
                                "bbc_world", "BBC World")
    gnf._add_to_recent_buffer([h1, h2])

    result = me.check_rss_breaking_news()
    assert result is not None
    assert result["trigger"] == "rss_breaking_news"
    assert result["headline_count"] == 2
    assert set(result["sources"]) == {"Reuters World", "BBC World"}


def test_refresh_recent_headlines_still_invoked_as_the_scan_driver(monkeypatch):
    """refresh_recent_headlines() must still be called — it's the market-hours scan
    driver that actually triggers the live RSS fetch / populates the buffer for
    everyone (item 4: unlike the old scan_current_window() driver, it never touches
    HeadlineManager._seen_keys, so it doesn't steal from learning_engine)."""
    calls = {"n": 0}

    def counted_refresh():
        calls["n"] += 1
        return 0

    monkeypatch.setattr(gnf, "refresh_recent_headlines", counted_refresh)
    me.check_rss_breaking_news()
    assert calls["n"] == 1


def test_no_longer_calls_scan_current_window(monkeypatch):
    """ORCH follow-up #2 item 4: market_events must no longer call scan_current_window()
    at all — that's exactly what stole headlines from learning_engine."""
    def raising():
        raise AssertionError("market_events must not call scan_current_window() (item 4 fix)")

    monkeypatch.setattr(gnf, "scan_current_window", raising)
    me.check_rss_breaking_news()  # must not raise


def test_single_t1_source_below_threshold_returns_none():
    h1 = _make_crisis_headline("Nuclear strike reported in border region",
                                "reuters_world", "Reuters World")
    gnf._add_to_recent_buffer([h1])

    assert me.check_rss_breaking_news() is None  # only 1 T1 source — below _RSS_BREAKING_T1_THRESHOLD


def test_non_t1_sources_excluded_from_cluster():
    h1 = _make_crisis_headline("Nuclear strike reported in border region",
                                "reuters_world", "Reuters World", tier="T1_VERIFIED")
    h2 = _make_crisis_headline("Military strike confirmed near capital",
                                "some_blog", "Some Blog", tier="T2_SEMI_INDEPENDENT")
    gnf._add_to_recent_buffer([h1, h2])

    assert me.check_rss_breaking_news() is None  # only 1 T1 among the 2


def test_log_only_mode_returns_none_but_still_logs(monkeypatch, caplog):
    import logging
    monkeypatch.setenv("CLIO_RSS_BREAKING_LOG_ONLY", "1")
    h1 = _make_crisis_headline("Nuclear strike reported in border region",
                                "reuters_world", "Reuters World")
    h2 = _make_crisis_headline("Military strike confirmed near capital",
                                "bbc_world", "BBC World")
    gnf._add_to_recent_buffer([h1, h2])

    caplog.set_level(logging.INFO, logger="openclaw.market_events")
    assert me.check_rss_breaking_news() is None
    assert any("LOG_ONLY" in r.message for r in caplog.records)


def test_no_headlines_in_buffer_returns_none():
    assert me.check_rss_breaking_news() is None
