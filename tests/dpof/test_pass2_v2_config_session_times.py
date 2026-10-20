"""Pass-2 verifier V2 regression tests — DPOFConfig.validate() session-time
ordering (orchestrator follow-up, quoted token ORCH-7Q4K-DPOF).

V1 found validate() had no check that entry_start_et < entry_end_et <
flatten_et. An inverted or malformed window would silently pass validate()
and only surface as confusing runtime behavior (e.g. an entry window that
never opens, or an entry window that outlives the forced flatten).
"""

from __future__ import annotations

from dataclasses import replace

from agent.dpof.config import DPOFConfig


def test_default_config_has_no_session_time_problems():
    problems = DPOFConfig().validate()
    assert not any("entry_start_et" in p or "entry_end_et" in p or "flatten_et" in p for p in problems)


def test_entry_start_at_or_after_entry_end_is_a_problem():
    cfg = replace(DPOFConfig(), entry_start_et="15:15", entry_end_et="15:15")  # equal -> not strictly before
    problems = cfg.validate()
    assert any("entry_start_et < entry_end_et" in p for p in problems)

    cfg2 = replace(DPOFConfig(), entry_start_et="15:20", entry_end_et="15:15")  # start after end
    problems2 = cfg2.validate()
    assert any("entry_start_et < entry_end_et" in p for p in problems2)


def test_entry_end_at_or_after_flatten_is_a_problem():
    cfg = replace(DPOFConfig(), entry_end_et="15:50", flatten_et="15:50")  # equal -> not strictly before
    problems = cfg.validate()
    assert any("entry_end_et < flatten_et" in p for p in problems)

    cfg2 = replace(DPOFConfig(), entry_end_et="16:00", flatten_et="15:50")  # end after flatten
    problems2 = cfg2.validate()
    assert any("entry_end_et < flatten_et" in p for p in problems2)


def test_valid_strictly_increasing_window_has_no_ordering_problems():
    cfg = replace(DPOFConfig(), entry_start_et="09:45", entry_end_et="15:15", flatten_et="15:50")
    problems = cfg.validate()
    assert not any("entry_start_et" in p or "entry_end_et" in p or "flatten_et" in p for p in problems)


def test_malformed_hhmm_strings_are_flagged_and_skip_ordering_checks():
    cfg = replace(DPOFConfig(), entry_start_et="not-a-time", entry_end_et="15:15", flatten_et="15:50")
    problems = cfg.validate()
    assert any("entry_start_et is not a valid HH:MM string" in p for p in problems)
    # Ordering checks that would need entry_start_et must not fire (no crash,
    # no comparison against an unparseable value).
    assert not any("entry_start_et < entry_end_et" in p for p in problems)


def test_all_three_malformed_reports_all_three_and_no_ordering_crash():
    cfg = replace(DPOFConfig(), entry_start_et="", entry_end_et="25:99", flatten_et="abc")
    problems = cfg.validate()  # must never raise
    assert any("entry_start_et is not a valid HH:MM string" in p for p in problems)
    assert any("entry_end_et is not a valid HH:MM string" in p for p in problems)
    assert any("flatten_et is not a valid HH:MM string" in p for p in problems)
    assert not any("<" in p and ("entry_start_et" in p or "entry_end_et" in p) for p in problems if "not a valid" not in p)


def test_out_of_range_hhmm_is_flagged_as_invalid():
    # datetime.time() raises ValueError for hour=25 -- same failure mode as
    # a non-numeric string, both must land in the "not a valid HH:MM" bucket.
    cfg = replace(DPOFConfig(), flatten_et="25:00")
    problems = cfg.validate()
    assert any("flatten_et is not a valid HH:MM string" in p for p in problems)


def test_single_digit_hour_is_accepted_like_risk_parse_hhmm():
    # agent.dpof.risk.parse_hhmm does int(hh)/int(mm) with no zero-padding
    # requirement -- config.py's own copy must accept the same inputs.
    cfg = replace(DPOFConfig(), entry_start_et="9:45", entry_end_et="15:15", flatten_et="15:50")
    problems = cfg.validate()
    assert not any("entry_start_et" in p for p in problems)
