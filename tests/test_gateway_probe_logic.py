"""S214: scripts/gateway_probe.py judge thresholds (pure) + one fake-client models run.

No network. The probe module is imported by path; importing it does not touch
.env or agent.config (those happen only inside main()).
"""

import datetime as dt
import importlib.util
import os
import sys
from types import SimpleNamespace as NS

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_spec = importlib.util.spec_from_file_location("gateway_probe", os.path.join(ROOT, "scripts", "gateway_probe.py"))
gp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gp)

NOW = dt.datetime(2026, 10, 11, 12, 0, tzinfo=dt.timezone.utc)


# ---- judge_models -----------------------------------------------------------

def _m(status, error="", served=None, model="claude-opus-5-5"):
    return {model: {"status": status, "served": served, "error": error}}


def test_models_200_pass():
    assert gp.judge_models(_m(200, served="claude-opus-5-5"))[0][0] == "PASS"


def test_models_haiku_5_5_400_is_warn_not_fail():
    assert gp.judge_models(_m(400, model=gp.HAIKU_5_5))[0][0] == "WARN"


def test_models_other_400_is_fail():
    assert gp.judge_models(_m(400))[0][0] == "FAIL"


def test_models_401_and_403_fail():
    assert gp.judge_models(_m(401))[0][0] == "FAIL"
    assert gp.judge_models(_m(403))[0][0] == "FAIL"


def test_models_429_insufficient_fails_plain_429_warns():
    assert gp.judge_models(_m(429, error="Insufficient balance"))[0][0] == "FAIL"
    assert gp.judge_models(_m(429, error="rate limit exceeded"))[0][0] == "WARN"


def test_models_500_and_connection_failure_fail():
    assert gp.judge_models(_m(500))[0][0] == "FAIL"
    assert gp.judge_models(_m(None))[0][0] == "FAIL"


# ---- judge_structured (need ceil(7n/8)) -----------------------------------

def test_structured_seven_of_eight_passes():
    r = gp.judge_structured({"decider_entry": [True] * 7 + [False]}, 8)
    assert r[0][0] == "PASS"


def test_structured_six_of_eight_fails():
    r = gp.judge_structured({"decider_entry": [True] * 6 + [False] * 2}, 8)
    assert r[0][0] == "FAIL"


def test_structured_eight_of_eight_passes():
    assert gp.judge_structured({"s": [True] * 8}, 8)[0][0] == "PASS"


def test_structured_missing_results_count_as_failures():
    # only 6 results recorded out of n=8 -> 6 valid, below 7
    assert gp.judge_structured({"s": [True] * 6}, 8)[0][0] == "FAIL"


def test_structured_n5_boundary_four_fails_five_passes():
    assert gp.judge_structured({"s": [True] * 4 + [False]}, 5)[0][0] == "FAIL"
    assert gp.judge_structured({"s": [True] * 5}, 5)[0][0] == "PASS"


def test_structured_each_surface_judged_separately():
    r = gp.judge_structured({"a": [True] * 8, "b": [False] * 8}, 8)
    assert [x[0] for x in r] == ["PASS", "FAIL"]


# ---- judge_toolchoice (TEXT >= 9/10) ---------------------------------------

def test_toolchoice_nine_of_ten_passes():
    assert gp.judge_toolchoice(9, 10)[0] == "PASS"


def test_toolchoice_eight_of_ten_fails():
    assert gp.judge_toolchoice(8, 10)[0] == "FAIL"


def test_toolchoice_no_calls_fails():
    assert gp.judge_toolchoice(0, 0)[0] == "FAIL"


# ---- judge_empty (final==0 PASS; rate<=1% WARN; else FAIL) -----------------

def test_empty_zero_final_passes_even_with_retries():
    assert gp.judge_empty({"empty_final": 0, "empty_retries": 5}, 72)[0] == "PASS"


def test_empty_one_of_72_is_fail_by_rate():
    # 1/72 = 1.39% > 1% -> FAIL
    assert gp.judge_empty({"empty_final": 1}, 72)[0] == "FAIL"


def test_empty_exactly_one_percent_is_warn():
    assert gp.judge_empty({"empty_final": 1}, 100)[0] == "WARN"


def test_empty_just_over_one_percent_is_fail():
    assert gp.judge_empty({"empty_final": 2}, 100)[0] == "FAIL"


def test_empty_no_calls_fails():
    assert gp.judge_empty({"empty_final": 0}, 0)[0] == "FAIL"


# ---- judge_latency ----------------------------------------------------------

def test_latency_all_within_deadline_passes_including_equal():
    assert gp.judge_latency([3.0, 20.0, 12.5], 20.0)[0] == "PASS"


def test_latency_one_over_deadline_warns():
    assert gp.judge_latency([3.0, 20.01], 20.0)[0] == "WARN"


def test_latency_no_samples_warns():
    assert gp.judge_latency([], 20.0)[0] == "WARN"


def test_latency_detail_reports_share_within_20s():
    detail = gp.judge_latency([5.0, 25.0], 30.0)[1]
    assert "within_20s=1/2" in detail


# ---- judge_balance ----------------------------------------------------------

def test_balance_200_active_passes_and_reports_days():
    payload = {"status": 200, "body": {"is_active": True, "balance_usd": 42.5, "expires_at": "2026-10-21T12:00:00Z"}}
    level, detail = gp.judge_balance(payload, NOW)
    assert level == "PASS"
    assert "days_to_expiry=10" in detail and "balance_usd=42.50" in detail


def test_balance_401_fails():
    assert gp.judge_balance({"status": 401, "body": None}, NOW)[0] == "FAIL"


def test_balance_inactive_fails():
    assert gp.judge_balance({"status": 200, "body": {"is_active": False}}, NOW)[0] == "FAIL"


def test_balance_active_not_reported_warns():
    assert gp.judge_balance({"status": 200, "body": {}}, NOW)[0] == "WARN"


# ---- end-to-end models suite with a fake client ----------------------------

class _FakeAPIError(Exception):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class _FakeMessages:
    def __init__(self, table):
        self.table = table
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        out = self.table[kw["model"]]
        if isinstance(out, Exception):
            raise out
        return out


class _FakeClient:
    def __init__(self, table):
        self.messages = _FakeMessages(table)


def test_models_suite_end_to_end_with_fake_client_and_key_scrub():
    key = "SECRET-KEY-VALUE-123"
    table = {
        "claude-opus-5-5": NS(model="claude-opus-5-5", content=[NS(type="text", text="p")]),
        "claude-sonnet-5-5": _FakeAPIError(403, f"forbidden for key {key}"),
        "claude-haiku-4-5-20251001": NS(model="claude-haiku-4-5-20251001", content=[]),
    }
    client = _FakeClient(table)
    checks = gp.suite_models(lambda: client, key)
    levels = [c[0] for c in checks]
    assert levels == ["PASS", "FAIL", "PASS"]
    assert key not in checks[1][1]
    assert "***" in checks[1][1]
    # each ping is one tiny request with max_tokens 16 and no tools
    assert all(c.get("max_tokens") == 16 and "tools" not in c for c in client.messages.calls)
    assert [c["model"] for c in client.messages.calls] == gp.MODELS_PING


def test_print_checks_reports_failure_flag(capsys):
    fails = [("PASS", "a"), ("FAIL", "b")]
    assert gp._print_checks("x", fails) is True
    out = capsys.readouterr().out
    assert "PASS x: a" in out and "FAIL x: b" in out
    assert gp._print_checks("x", [("PASS", "a"), ("WARN", "c")]) is False
