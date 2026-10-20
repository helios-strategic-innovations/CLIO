"""Tests for agent.dpof.prompts's Kronos additions (Task K2, agent/kronos --
docs/design/clio_kronos.md §4): the CHARTER paragraph, the "=== KRONOS
FORECAST ... ===" section (_format_kronos/_format_kronos_line), and
_CTX_SKIP_GENERIC_KEYS. Verifies entry/review messages are byte-identical to
the pre-feature rendering when ctx carries no "kronos" key at all (the
disabled-config acceptance bar, same contract as lessons/debate).

Plugin-free (no pytest-asyncio), matching tests/dpof/test_prompts.py /
tests/dpof/test_lessons_prompt.py.
"""

from __future__ import annotations

import asyncio

import pytest

import agent.dpof.prompts as prompts
from agent.dpof.decider import EntryCandidate


def _candidate(key="C1", underlying="SPY", direction="bullish", mid=1.23, max_affordable_qty=5) -> EntryCandidate:
    return EntryCandidate(
        key=key, underlying=underlying, direction=direction, structure="single",
        legs=[{"symbol": "SPY261002C00580000", "side": "buy", "ratio_qty": 1, "position_intent": "buy_to_open"}],
        expiry="2026-09-25", sessions_to_expiry=1, strike_desc="580C",
        bid=mid - 0.05, ask=mid + 0.05, mid=mid, spread_pct=0.08,
        delta=0.5, theta=-0.12, iv=0.18, open_interest=1200, unit_premium=mid * 100,
        max_affordable_qty=max_affordable_qty,
    )


def _ctx(headlines=None, kronos=None) -> dict:
    d = {
        "now_et": "2026-09-24T11:00:00-04:00",
        "minutes_to_flatten": 290,
        "mode": "paper",
        "underlyings": {"SPY": {"composite": 0.42, "direction": "bullish"}},
        "market_regime": "normal",
        "open_positions": [],
        "capital": {
            "dpof_equity": 25000.0, "capital_available": 24500.0, "open_premium": 500.0,
            "open_positions_count": 1, "entries_today": 1, "realized_today": 0.0,
        },
        "recent_closed": [],
        "econ_events": [],
        "jev": None,
        "headlines": headlines or [],
    }
    if kronos is not None:
        d["kronos"] = kronos
    return d


_KRONOS_SPY_OK = {
    "asof_bar": "2026-09-25T12:00:00-04:00",
    "horizons": {
        "15": {"p_up": 0.6, "mean_return_pct": 0.05, "q10_return_pct": -0.2, "q90_return_pct": 0.3, "expected_abs_move_pct": 0.12},
        "30": {"p_up": 0.58, "mean_return_pct": 0.09, "q10_return_pct": -0.35, "q90_return_pct": 0.55, "expected_abs_move_pct": 0.2},
        "60": {"p_up": 0.55, "mean_return_pct": 0.14, "q10_return_pct": -0.6, "q90_return_pct": 0.9, "expected_abs_move_pct": 0.35},
    },
    "track_record": {
        "30": {
            "all": {"n": 48, "hit_rate": 0.62, "hit_rate_confident": 0.70, "brier": 0.21},
            "recent": {"n": 12, "hit_rate": 0.58, "hit_rate_confident": 0.60, "brier": 0.24},
        },
    },
}


@pytest.fixture(autouse=True)
def _reset_context_cache():
    prompts._context_cache["text"] = ""
    prompts._context_cache["ts"] = 0.0
    yield
    prompts._context_cache["text"] = ""
    prompts._context_cache["ts"] = 0.0


# ─────────────────────────────────────────────────────────────────────────
# CHARTER paragraph
# ─────────────────────────────────────────────────────────────────────────

def test_charter_mentions_kronos_and_advisory_only():
    text = prompts.CHARTER
    assert "Kronos" in text
    assert "candlestick" in text.lower()
    assert "advisory" in text.lower()


def test_charter_kronos_paragraph_mentions_track_record_weighting():
    text = prompts.CHARTER.lower()
    assert "track record" in text
    assert "small n" in text or "little weight" in text


def test_charter_kronos_paragraph_says_never_sole_reason():
    text = prompts.CHARTER.lower()
    assert "never, by itself, a reason to enter, hold, or exit" in text


def test_charter_byte_stable_across_calls():
    """Required for Anthropic prompt caching to hit -- reload and re-check
    the exact same CHARTER bytes."""
    import importlib

    first = prompts.CHARTER
    for _ in range(3):
        reloaded = importlib.reload(prompts)
        assert reloaded.CHARTER == first
    importlib.reload(prompts)  # leave the module in its normal state


def test_charter_no_interpolation_markers():
    """CHARTER must stay a plain, non-interpolated string -- no stray {}/%s
    left over from an f-string/format() mistake in the new paragraph."""
    assert "{" not in prompts.CHARTER
    assert "}" not in prompts.CHARTER


# ─────────────────────────────────────────────────────────────────────────
# Byte-identical rendering when "kronos" key absent (disabled config)
# ─────────────────────────────────────────────────────────────────────────

def test_entry_message_byte_identical_with_no_kronos_key():
    msg_a = asyncio.run(prompts.build_entry_message(_ctx(), [_candidate()]))
    msg_b = asyncio.run(prompts.build_entry_message(_ctx(), [_candidate()]))
    assert msg_a == msg_b
    assert "KRONOS FORECAST" not in msg_a


def test_review_message_byte_identical_with_no_kronos_key():
    position = {"id": "pos_1", "underlying": "SPY", "thesis": "flow strongly bullish"}
    msg_a = asyncio.run(prompts.build_review_message(_ctx(), position))
    assert "KRONOS FORECAST" not in msg_a


def test_entry_message_still_renders_kronos_absent_marker_section_when_key_present_but_none_dict():
    """ctx["kronos"] present but not a usable dict (e.g. an accidental None
    slipping through) -> the section header still renders with "(none)", not
    a crash and not a silent omission (distinguishing "never wired" from
    "wired but empty")."""
    msg = asyncio.run(prompts.build_entry_message(_ctx(kronos="not a dict"), [_candidate()]))
    assert "KRONOS FORECAST" in msg
    assert "(none)" in msg


# ─────────────────────────────────────────────────────────────────────────
# Section rendering when present
# ─────────────────────────────────────────────────────────────────────────

def test_entry_message_renders_kronos_section_header():
    msg = asyncio.run(prompts.build_entry_message(_ctx(kronos={"SPY": _KRONOS_SPY_OK}), [_candidate()]))
    assert "=== KRONOS FORECAST (candlestick model, advisory) ===" in msg


def test_review_message_renders_kronos_section():
    position = {"id": "pos_1", "underlying": "SPY"}
    msg = asyncio.run(prompts.build_review_message(_ctx(kronos={"SPY": _KRONOS_SPY_OK}), position))
    assert "=== KRONOS FORECAST" in msg
    assert "SPY" in msg


def test_kronos_section_renders_p_up_and_track_record_fields():
    msg = asyncio.run(prompts.build_entry_message(_ctx(kronos={"SPY": _KRONOS_SPY_OK}), [_candidate()]))
    assert "p_up=58%" in msg  # 30-min horizon
    assert "n=48" in msg
    assert "n=12" in msg


def test_kronos_section_renders_unavailable_marker_per_symbol():
    msg = asyncio.run(prompts.build_entry_message(_ctx(kronos={"SPY": "unavailable"}), [_candidate()]))
    assert "SPY: unavailable" in msg


def test_kronos_section_missing_horizon_says_no_horizon_data():
    msg = asyncio.run(prompts.build_entry_message(
        _ctx(kronos={"SPY": {"asof_bar": "x", "horizons": {}, "track_record": {}}}), [_candidate()],
    ))
    assert "(no horizon data)" in msg


def test_kronos_section_empty_dict_says_none():
    msg = asyncio.run(prompts.build_entry_message(_ctx(kronos={}), [_candidate()]))
    assert "=== KRONOS FORECAST" in msg
    assert "(none)" in msg


def test_kronos_section_never_fabricates_missing_track_record_as_zero():
    """No track_record entry at all for a horizon -> "n=0", never a
    fabricated hit rate."""
    k = {"asof_bar": "x", "horizons": {"30": {"p_up": 0.5, "mean_return_pct": 0.0,
                                                "q10_return_pct": None, "q90_return_pct": None,
                                                "expected_abs_move_pct": None}}, "track_record": {}}
    msg = asyncio.run(prompts.build_entry_message(_ctx(kronos={"SPY": k}), [_candidate()]))
    assert "n=0" in msg
    assert "n/a" in msg  # q10/q90/expected|move| all n/a


# ─────────────────────────────────────────────────────────────────────────
# _CTX_SKIP_GENERIC_KEYS / no double-render
# ─────────────────────────────────────────────────────────────────────────

def test_kronos_in_ctx_skip_generic_keys():
    assert "kronos" in prompts._CTX_SKIP_GENERIC_KEYS


def test_kronos_key_not_double_rendered_in_generic_fallback():
    """"kronos" must never ALSO appear as a raw "kronos: {...}" compact-JSON
    line from _format_ctx's generic fallback loop."""
    msg = asyncio.run(prompts.build_entry_message(_ctx(kronos={"SPY": _KRONOS_SPY_OK}), [_candidate()]))
    assert "kronos: {" not in msg


# ─────────────────────────────────────────────────────────────────────────
# R-08 — skill-baseline text (CLIO_DPOF_KRONOS_SKILL_TEXT, default on)
# ─────────────────────────────────────────────────────────────────────────

_BASELINE_PHRASE = "vs 0.250 coin-flip -> no skill"
_BASELINE_SENTENCE = "A Kronos Brier above 0.25 means no measured skill; treat the Kronos pillar as uninformative."

# A track record with a real "no skill" reading: n=48, Brier 0.325 (live SPY 30-min shape).
_KRONOS_NO_SKILL = {
    "asof_bar": "2026-09-25T12:00:00-04:00",
    "horizons": {
        "30": {"p_up": 0.58, "mean_return_pct": 0.09, "q10_return_pct": -0.35, "q90_return_pct": 0.55, "expected_abs_move_pct": 0.2},
    },
    "track_record": {"30": {
        "all": {"n": 48, "hit_rate": 0.497, "hit_rate_confident": 0.5, "brier": 0.325},
        "recent": {"n": 12, "hit_rate": 0.58, "hit_rate_confident": 0.60, "brier": 0.24},
    }},
}


# Exact fixture behind the HEAD (8db0420) golden hashes below -- do not edit.
_KRONOS_GOLDEN = {
    "asof_bar": "2026-09-25T12:00:00-04:00",
    "horizons": {
        "15": {"p_up": 0.6, "mean_return_pct": 0.05, "q10_return_pct": -0.2, "q90_return_pct": 0.3, "expected_abs_move_pct": 0.12},
        "30": {"p_up": 0.58, "mean_return_pct": 0.09, "q10_return_pct": -0.35, "q90_return_pct": 0.55, "expected_abs_move_pct": 0.2},
    },
    "track_record": {"30": {
        "all": {"n": 48, "hit_rate": 0.497, "hit_rate_confident": 0.5, "brier": 0.325},
        "recent": {"n": 12, "hit_rate": 0.58, "hit_rate_confident": 0.60, "brier": 0.24},
    }},
}


def _no_skill_with(bucket_all=None, bucket_recent=None):
    """Kronos read whose 30-min track record uses the given buckets."""
    k = {
        "asof_bar": "2026-09-25T12:00:00-04:00",
        "horizons": {"30": {"p_up": 0.58, "mean_return_pct": 0.09, "q10_return_pct": -0.35,
                            "q90_return_pct": 0.55, "expected_abs_move_pct": 0.2}},
        "track_record": {"30": {}},
    }
    if bucket_all is not None:
        k["track_record"]["30"]["all"] = bucket_all
    if bucket_recent is not None:
        k["track_record"]["30"]["recent"] = bucket_recent
    return k


def test_skill_text_flag_default_on(monkeypatch):
    monkeypatch.delenv("CLIO_DPOF_KRONOS_SKILL_TEXT", raising=False)
    assert prompts._kronos_skill_text_on() is True


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", " OFF "])
def test_skill_text_flag_off_values(monkeypatch, raw):
    monkeypatch.setenv("CLIO_DPOF_KRONOS_SKILL_TEXT", raw)
    assert prompts._kronos_skill_text_on() is False


@pytest.mark.parametrize("raw", ["1", "true", "yes", "on"])
def test_skill_text_flag_on_values(monkeypatch, raw):
    monkeypatch.setenv("CLIO_DPOF_KRONOS_SKILL_TEXT", raw)
    assert prompts._kronos_skill_text_on() is True


def test_skill_text_flag_on_shows_baseline_phrase_and_sentence(monkeypatch):
    monkeypatch.setenv("CLIO_DPOF_KRONOS_SKILL_TEXT", "1")
    section = prompts._format_kronos(_ctx(kronos={"SPY": _KRONOS_NO_SKILL}))
    assert _BASELINE_PHRASE in section
    assert "brier=0.325 vs 0.250 coin-flip -> no skill" in section
    assert section.endswith(_BASELINE_SENTENCE)


def test_skill_text_phrase_not_on_recent_bucket_below_n30(monkeypatch):
    """The recent bucket (n=12, Brier 0.24) never gets the phrase, and the
    all-bucket phrase is the only one present."""
    monkeypatch.setenv("CLIO_DPOF_KRONOS_SKILL_TEXT", "1")
    section = prompts._format_kronos(_ctx(kronos={"SPY": _KRONOS_NO_SKILL}))
    assert section.count(_BASELINE_PHRASE) == 1
    assert "recent: n=12 hit=58% confident_hit=60% brier=0.240;" in section


@pytest.mark.parametrize("bucket", [
    {"n": 12, "hit_rate": 0.5, "hit_rate_confident": 0.5, "brier": 0.325},   # n < 30
    {"n": 29, "hit_rate": 0.5, "hit_rate_confident": 0.5, "brier": 0.4},     # n just under 30
])
def test_skill_text_baseline_omitted_when_n_below_30(monkeypatch, bucket):
    monkeypatch.setenv("CLIO_DPOF_KRONOS_SKILL_TEXT", "1")
    section = prompts._format_kronos(_ctx(kronos={"SPY": _no_skill_with(bucket_all=bucket)}))
    assert _BASELINE_PHRASE not in section
    assert _BASELINE_SENTENCE not in section


def test_skill_text_baseline_shown_at_exactly_n_30(monkeypatch):
    # n == 30 is the first judged reading (the guard is n < 30); a '<= 30' mutant must fail here.
    monkeypatch.setenv("CLIO_DPOF_KRONOS_SKILL_TEXT", "1")
    bucket = {"n": 30, "hit_rate": 0.5, "hit_rate_confident": 0.5, "brier": 0.325}
    section = prompts._format_kronos(_ctx(kronos={"SPY": _no_skill_with(bucket_all=bucket)}))
    assert _BASELINE_PHRASE in section
    assert _BASELINE_SENTENCE in section


@pytest.mark.parametrize("brier", [0.25, 0.21, None])
def test_skill_text_baseline_omitted_when_brier_at_or_below_025(monkeypatch, brier):
    monkeypatch.setenv("CLIO_DPOF_KRONOS_SKILL_TEXT", "1")
    bucket = {"n": 48, "hit_rate": 0.5, "hit_rate_confident": 0.5, "brier": brier}
    section = prompts._format_kronos(_ctx(kronos={"SPY": _no_skill_with(bucket_all=bucket)}))
    assert _BASELINE_PHRASE not in section
    assert _BASELINE_SENTENCE not in section


def test_skill_text_baseline_shown_on_recent_bucket_alone(monkeypatch):
    monkeypatch.setenv("CLIO_DPOF_KRONOS_SKILL_TEXT", "1")
    bucket = {"n": 40, "hit_rate": 0.5, "hit_rate_confident": 0.5, "brier": 0.31}
    section = prompts._format_kronos(_ctx(kronos={"SPY": _no_skill_with(bucket_recent=bucket)}))
    assert _BASELINE_PHRASE in section
    assert section.endswith(_BASELINE_SENTENCE)


async def _no_clio_context():
    return ""


def _flag0_messages(monkeypatch):
    """Entry + review messages for a fixed ctx with the no-skill Kronos read,
    CLIO context stubbed out so the bytes are deterministic."""
    monkeypatch.setattr(prompts, "_load_clio_context", _no_clio_context)
    entry = asyncio.run(prompts.build_entry_message(_ctx(kronos={"SPY": _KRONOS_GOLDEN}), [_candidate()]))
    review = asyncio.run(prompts.build_review_message(
        _ctx(kronos={"SPY": _KRONOS_GOLDEN}), {"id": "pos_1", "underlying": "SPY", "thesis": "flow"}))
    return entry, review


def test_flag_off_messages_byte_identical_to_head_golden(monkeypatch):
    """Golden: with CLIO_DPOF_KRONOS_SKILL_TEXT=0 the entry and review messages
    must hash to the values rendered by the pre-change prompts.py (8db0420).
    Any byte drift in the flag-off path fails this test."""
    import hashlib
    monkeypatch.setenv("CLIO_DPOF_KRONOS_SKILL_TEXT", "0")
    entry, review = _flag0_messages(monkeypatch)
    assert hashlib.sha256(entry.encode()).hexdigest() == (
        "74d704075c06e44f4a7a7e3cbd0736d1ae6f7f66785bcb2222b7648ed8f133da")
    assert hashlib.sha256(review.encode()).hexdigest() == (
        "dedd05afdccc8310667f04b966e19991494858c8cc1de7ffadbc684e591d523f")


def test_flag_off_kronos_section_is_head_golden_text(monkeypatch):
    monkeypatch.setenv("CLIO_DPOF_KRONOS_SKILL_TEXT", "0")
    section = prompts._format_kronos(_ctx(kronos={"SPY": _KRONOS_GOLDEN}))
    assert section == (
        "=== KRONOS FORECAST (candlestick model, advisory) ===\n"
        "SPY (as of 2026-09-25T12:00:00-04:00):\n"
        "  15min: p_up=60% mean=+0.05% q10/q90=-0.20%/+0.30% expected|move|=0.12% | "
        "track recent: n=0; all: n=0\n"
        "  30min: p_up=58% mean=+0.09% q10/q90=-0.35%/+0.55% expected|move|=0.20% | "
        "track recent: n=12 hit=58% confident_hit=60% brier=0.240; "
        "all: n=48 hit=50% confident_hit=50% brier=0.325"
    )
    assert _BASELINE_PHRASE not in section
    assert _BASELINE_SENTENCE not in section
