"""S212 OUTBOX-1: an isolated Supabase outbox failure is an INFO line; only a sustained run is the WARNING.

A failed flush leaves its rows unsynced and the next flush (1 s later) retries them, so a single ReadTimeout heals in
seconds (4 in 21.5 h on 2026-09-29/30, 1-2 rows each). The `UW_SUPABASE_OUTBOX_DEGRADED` token -- which the morning
check and operators grep -- now marks 3+ consecutive failures only; the INFO line deliberately does NOT contain it.
All ledgers live under tmp_path; requests.post is faked (no network).
"""
import logging

import pytest

from agent.integrations import uw_quota_client as quota_module
from tests.test_uw_quota_hygiene import _FakeHttpResponse, _client, _observe, _reserve

TOKEN = "UW_SUPABASE_OUTBOX_DEGRADED"


@pytest.fixture()
def post(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://supabase.invalid")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-role-key")
    state = {"fail": True, "exc": quota_module.requests.exceptions.ReadTimeout("read timed out"), "calls": 0}

    def fake_post(url, **kwargs):
        state["calls"] += 1
        if state["fail"]:
            raise state["exc"]
        return _FakeHttpResponse()

    monkeypatch.setattr(quota_module.requests, "post", fake_post)
    return state


def _queued_client(tmp_path):
    client = _client(tmp_path)
    _observe(client, _reserve(client), 10)          # one outbox row
    return client


def _flush(client, caplog):
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="UWQuota"):
        n = client.flush_outbox_once()
    return n, [r for r in caplog.records if r.name == "UWQuota"]


def test_first_and_second_consecutive_failures_are_info_without_the_degraded_token(tmp_path, post, caplog):
    client = _queued_client(tmp_path)
    for i in (1, 2):
        n, recs = _flush(client, caplog)
        assert n == 0 and len(recs) == 1
        assert recs[0].levelno == logging.INFO
        assert TOKEN not in recs[0].getMessage()
        assert "ReadTimeout" in recs[0].getMessage() and f"{i} in a row" in recs[0].getMessage()


def test_third_consecutive_failure_is_the_warning_with_the_token(tmp_path, post, caplog):
    client = _queued_client(tmp_path)
    for _ in range(2):
        _flush(client, caplog)
    n, recs = _flush(client, caplog)
    assert n == 0 and len(recs) == 1 and recs[0].levelno == logging.WARNING
    assert TOKEN in recs[0].getMessage() and "error=ReadTimeout" in recs[0].getMessage() and "consecutive=3" in recs[0].getMessage()


def test_it_keeps_warning_while_the_outage_continues(tmp_path, post, caplog):
    client = _queued_client(tmp_path)
    for _ in range(3):
        _flush(client, caplog)
    _, recs = _flush(client, caplog)
    assert recs[0].levelno == logging.WARNING and "consecutive=4" in recs[0].getMessage()


def test_success_resets_the_count_and_logs_one_recovery_line(tmp_path, post, caplog):
    client = _queued_client(tmp_path)
    _flush(client, caplog)
    _flush(client, caplog)
    post["fail"] = False
    n, recs = _flush(client, caplog)
    assert n == 1                                                          # the row stayed queued and went out
    assert [r.levelno for r in recs] == [logging.INFO] and "recovered after 2 failed flush(es)" in recs[0].getMessage()
    post["fail"] = True
    _observe(client, _reserve(client), 11)
    _, recs = _flush(client, caplog)
    assert recs[0].levelno == logging.INFO and "1 in a row" in recs[0].getMessage()   # counting starts over


def test_a_single_failure_followed_by_success_is_recovered_and_reset(tmp_path, post, caplog):
    client = _queued_client(tmp_path)
    _flush(client, caplog)
    post["fail"] = False
    n, recs = _flush(client, caplog)
    assert n == 1 and "recovered after 1 failed flush(es)" in recs[0].getMessage()
    post["fail"] = True
    _observe(client, _reserve(client), 11)
    _, recs = _flush(client, caplog)
    assert "1 in a row" in recs[0].getMessage()                               # the single failure did not linger


def test_a_success_with_no_prior_failure_logs_nothing(tmp_path, post, caplog):
    post["fail"] = False
    client = _queued_client(tmp_path)
    n, recs = _flush(client, caplog)
    assert n == 1 and recs == []


def test_http_errors_count_like_timeouts(tmp_path, post, caplog):
    post["exc"] = quota_module.requests.exceptions.HTTPError("503 Server Error")
    client = _queued_client(tmp_path)
    for _ in range(2):
        _flush(client, caplog)
    _, recs = _flush(client, caplog)
    assert recs[0].levelno == logging.WARNING and "error=HTTPError" in recs[0].getMessage()


def test_the_count_is_per_client(tmp_path, post, caplog):
    a = _queued_client(tmp_path / "a")
    b = _queued_client(tmp_path / "b")
    _flush(a, caplog)
    _flush(a, caplog)
    _, recs = _flush(b, caplog)
    assert recs[0].levelno == logging.INFO and "1 in a row" in recs[0].getMessage()


def test_failed_rows_are_still_marked_failed_every_time(tmp_path, post, caplog, monkeypatch):
    seen = []
    real = quota_module._SQLiteQuotaLedger.mark_outbox_failed

    def spy(self, event_ids, error):
        seen.append((list(event_ids), error))
        return real(self, event_ids, error)

    monkeypatch.setattr(quota_module._SQLiteQuotaLedger, "mark_outbox_failed", spy)
    client = _queued_client(tmp_path)
    for _ in range(4):
        _flush(client, caplog)
    assert len(seen) == 4 and all(err == "ReadTimeout" and len(ids) == 1 for ids, err in seen)


def test_the_threshold_constant_is_three():
    assert quota_module._OUTBOX_WARN_AFTER == 3
