"""Advisory size anchor (operator 2026-09-29): a REFERENCE scale for the decision model, never a cap.

Recent entries landed at ~4-5% of the slice whatever the setup. The operator asked for ~10% of dpof_equity in premium
at conviction 0.50 and ~20% at 0.80. agent.dpof.sizing.anchor_* computes it, the engine adds a "size_anchor" ctx key only
when enabled, and prompts.py renders one advisory section plus a per-candidate contracts-by-conviction line. The
solvency clamp is untouched and nothing here can reject or shrink a decision.
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import replace

import pytest

from agent.dpof import config as config_mod
from agent.dpof import prompts, sizing
from agent.dpof.config import DPOFConfig, load_dpof_config
from agent.dpof.models import RiskState
from tests.dpof.test_engine import _base_dt
from tests.dpof.test_engine_judgment import _engine
from tests.dpof.test_prompts import _candidate, _ctx

pytestmark = pytest.mark.filterwarnings("ignore")


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _reset_context_cache():
    prompts._context_cache["text"] = ""
    prompts._context_cache["ts"] = 0.0
    yield
    prompts._context_cache["text"] = ""
    prompts._context_cache["ts"] = 0.0


def _anchor_ctx(equity=20000.0, c50=0.10, c80=0.20):
    ctx = _ctx()
    ctx["capital"] = {**ctx["capital"], "dpof_equity": equity}
    ctx["size_anchor"] = {"share_c50": c50, "share_c80": c80}
    return ctx


# ─────────────────────────────────────────────────────────────────────────
# sizing.anchor_share / anchor_qty (pure)
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("conv,want", [
    (0.50, 0.10), (0.80, 0.20), (0.65, 0.15), (0.40, 0.10 - 0.10 / 3), (0.90, 0.20 + 0.10 / 3),
])
def test_anchor_share_is_linear_through_the_two_requested_points(conv, want):
    assert sizing.anchor_share(conv, 0.10, 0.20) == pytest.approx(want)


def test_anchor_share_is_clamped_at_both_ends():
    assert sizing.anchor_share(0.0, 0.10, 0.20) == sizing.ANCHOR_SHARE_FLOOR        # raw line is negative here
    assert sizing.anchor_share(0.2, 0.10, 0.20) == sizing.ANCHOR_SHARE_FLOOR        # raw line is exactly 0
    assert sizing.anchor_share(1.0, 0.40, 0.50) == sizing.ANCHOR_SHARE_CEILING      # raw line is ~0.57


def test_anchor_share_is_monotonic_in_conviction():
    shares = [sizing.anchor_share(c / 20, 0.10, 0.20) for c in range(0, 21)]
    assert shares == sorted(shares) and shares[0] < shares[-1]


@pytest.mark.parametrize("conv", [None, "0.5", True, float("nan"), float("inf"), -0.01, 1.01])
def test_anchor_share_none_on_unusable_conviction(conv):
    assert sizing.anchor_share(conv, 0.10, 0.20) is None


@pytest.mark.parametrize("c50,c80", [(None, 0.2), (0.1, None), (float("nan"), 0.2), (0.1, float("inf")), ("a", 0.2)])
def test_anchor_share_none_on_unusable_shares(c50, c80):
    assert sizing.anchor_share(0.5, c50, c80) is None


def test_anchor_qty_floors_and_is_bounded_by_solvency():
    assert sizing.anchor_qty(0.10, 20000.0, 240.0, 50) == 8          # 2000 / 240 = 8.33 -> 8, never rounded up
    assert sizing.anchor_qty(0.10, 20000.0, 240.0, 5) == 5           # max_affordable_qty wins
    assert sizing.anchor_qty(0.10, 20000.0, 240.0, 0) == 0           # nothing affordable -> 0, not None
    assert sizing.anchor_qty(0.02, 1000.0, 240.0, 9) == 0            # one contract exceeds the budget -> 0


def test_anchor_qty_exact_boundary_is_not_lost_to_float_noise():
    assert sizing.anchor_qty(0.10, 2400.0, 240.0, 9) == 1
    assert sizing.anchor_qty(0.30, 1000.0, 100.0, 9) == 3
    # the designed tolerance: a budget short of one contract by float dust (< 1e-9 dollars) still buys it, but a real
    # shortfall (1e-6 dollars) does not -- the epsilon is noise cover, not a fudge
    assert sizing.anchor_qty(0.30, 1000.0, 300.0 + 5e-11, 9) == 1
    assert sizing.anchor_qty(0.30, 1000.0, 300.0 + 1e-6, 9) == 0


def test_anchor_qty_never_negative_even_with_a_negative_solvency_bound():
    assert sizing.anchor_qty(0.10, 20000.0, 240.0, -3) == 0


@pytest.mark.parametrize("args", [
    (None, 20000.0, 240.0, 5), (0.1, None, 240.0, 5), (0.1, 0.0, 240.0, 5), (0.1, -5.0, 240.0, 5),
    (0.1, 20000.0, None, 5), (0.1, 20000.0, 0.0, 5), (0.1, 20000.0, float("nan"), 5), (0.1, 20000.0, 240.0, None),
    (0.0, 20000.0, 240.0, 5), (0.1, float("inf"), 240.0, 5),
])
def test_anchor_qty_none_on_unusable_inputs(args):
    assert sizing.anchor_qty(*args) is None


# ─────────────────────────────────────────────────────────────────────────
# config: defaults, env, normalisation (a bad value must never stop the cycle)
# ─────────────────────────────────────────────────────────────────────────

_ENVS = ("CLIO_DPOF_SIZE_ANCHOR", "CLIO_DPOF_ANCHOR_SHARE_C50", "CLIO_DPOF_ANCHOR_SHARE_C80")


@pytest.fixture()
def clean_env(monkeypatch):
    for k in _ENVS:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_config_defaults_match_the_dataclass(clean_env):
    d = DPOFConfig()
    assert (d.size_anchor_enabled, d.size_anchor_share_c50, d.size_anchor_share_c80) == (True, 0.10, 0.20)
    c = load_dpof_config()
    assert (c.size_anchor_enabled, c.size_anchor_share_c50, c.size_anchor_share_c80) == (True, 0.10, 0.20)


def test_config_flag_off_and_custom_shares(clean_env):
    clean_env.setenv("CLIO_DPOF_SIZE_ANCHOR", "0")
    clean_env.setenv("CLIO_DPOF_ANCHOR_SHARE_C50", "0.15")
    clean_env.setenv("CLIO_DPOF_ANCHOR_SHARE_C80", "0.30")
    c = load_dpof_config()
    assert c.size_anchor_enabled is False and (c.size_anchor_share_c50, c.size_anchor_share_c80) == (0.15, 0.30)


@pytest.mark.parametrize("var", ["CLIO_DPOF_ANCHOR_SHARE_C50", "CLIO_DPOF_ANCHOR_SHARE_C80"])
@pytest.mark.parametrize("raw", ["abc", "0", "-0.1", "0.9", "0.5000001", "1", "nan", "inf", "-inf"])
def test_config_bad_share_falls_back_to_the_default_not_to_a_stopped_cycle(clean_env, var, raw):
    clean_env.setenv(var, raw)
    c = load_dpof_config()
    assert c.size_anchor_share_c50 == 0.10 and c.size_anchor_share_c80 == 0.20
    assert not any("anchor" in p for p in c.validate())


def test_config_share_at_the_upper_bound_is_accepted(clean_env):
    clean_env.setenv("CLIO_DPOF_ANCHOR_SHARE_C80", "0.5")
    assert load_dpof_config().size_anchor_share_c80 == 0.5


def test_config_inverted_shares_reset_both_to_defaults(clean_env):
    clean_env.setenv("CLIO_DPOF_ANCHOR_SHARE_C50", "0.30")
    clean_env.setenv("CLIO_DPOF_ANCHOR_SHARE_C80", "0.10")
    c = load_dpof_config()
    assert (c.size_anchor_share_c50, c.size_anchor_share_c80) == (0.10, 0.20)


# ─────────────────────────────────────────────────────────────────────────
# engine._build_ctx: the key exists only when enabled
# ─────────────────────────────────────────────────────────────────────────

def _build_ctx(cfg):
    eng = _engine(cfg)
    eng._last_intel_ctx = {}
    return eng._build_ctx(
        now_et=_base_dt(), cfg=cfg, snap=None, underlyings_ctx={}, market_regime=None,
        risk=RiskState(), open_positions=[], dpof_equity=10_000.0, open_premium=0.0,
    )


def test_engine_ctx_carries_the_anchor_shares_when_enabled(cfg):
    ctx = _build_ctx(replace(cfg, size_anchor_enabled=True, size_anchor_share_c50=0.12, size_anchor_share_c80=0.25))
    assert ctx["size_anchor"] == {"share_c50": 0.12, "share_c80": 0.25}


def test_engine_ctx_omits_the_key_entirely_when_disabled(cfg):
    assert "size_anchor" not in _build_ctx(replace(cfg, size_anchor_enabled=False))


# ─────────────────────────────────────────────────────────────────────────
# prompts: one advisory section + a per-candidate line, only when on and equity is known
# ─────────────────────────────────────────────────────────────────────────

def _msg(ctx, candidates=None):
    return _run(prompts.build_entry_message(ctx, [_candidate(mid=2.40)] if candidates is None else candidates))


def test_message_has_the_advisory_section_before_the_candidates():
    msg = _msg(_anchor_ctx())
    assert "=== SIZE ANCHOR (advisory reference scale -- NOT a limit and NOT an instruction) ===" in msg
    assert msg.index("=== SIZE ANCHOR") < msg.index("=== CANDIDATES OFFERED")
    assert "0.50 -> 10.0%" in msg and "0.80 -> 20.0%" in msg and "0.65 -> 15.0%" in msg
    assert "your call" in msg and "passing is still fine" in msg          # advisory, never an order
    assert "floor 2%" in msg and "ceiling 30%" in msg


def test_candidate_line_lists_contracts_by_conviction_from_the_slice():
    msg = _msg(_anchor_ctx(equity=20000.0), [_candidate(mid=2.40, max_affordable_qty=50)])
    # unit_premium = 240.0; shares 6.67/10/15/20/23.33% of 20000 -> 1333/2000/3000/4000/4666 -> 5/8/12/16/19
    assert "size_anchor (advisory contracts by conviction): 0.40->5  0.50->8  0.65->12  0.80->16  0.90->19" in msg


def test_candidate_contracts_never_exceed_max_affordable_qty():
    msg = _msg(_anchor_ctx(equity=20000.0), [_candidate(mid=2.40, max_affordable_qty=3)])
    assert "0.40->3  0.50->3  0.65->3  0.80->3  0.90->3" in msg


def test_each_candidate_gets_its_own_line():
    msg = _msg(_anchor_ctx(), [_candidate(key="C1", mid=2.40), _candidate(key="C2", mid=4.80, max_affordable_qty=50)])
    lines = [l for l in msg.splitlines() if "size_anchor (advisory" in l]
    assert len(lines) == 2 and lines[0] != lines[1]


def test_disabled_or_absent_key_renders_no_anchor_and_the_same_message():
    base = _ctx()
    with_none = {**_ctx(), "size_anchor": None}
    m_base, m_none = _msg(base), _msg(with_none)
    assert m_base == m_none
    assert "SIZE ANCHOR" not in m_base and "size_anchor" not in m_base


def test_turning_the_anchor_on_only_adds_lines():
    off = _msg(_ctx()).splitlines()
    on_ctx = _ctx()
    on_ctx["size_anchor"] = {"share_c50": 0.10, "share_c80": 0.20}
    on = _msg(on_ctx).splitlines()
    it = iter(on)
    assert all(any(line == other for other in it) for line in off)         # `off` is a subsequence of `on`: nothing removed or reordered
    assert len(on) > len(off)


@pytest.mark.parametrize("equity", [None, 0.0, -100.0, float("nan"), "n/a"])
def test_unknown_or_unusable_equity_shows_no_anchor_anywhere(equity):
    msg = _msg(_anchor_ctx(equity=equity))
    assert "SIZE ANCHOR" not in msg and "size_anchor (advisory" not in msg


def test_missing_capital_block_shows_no_anchor():
    ctx = _anchor_ctx()
    del ctx["capital"]
    assert "size_anchor" not in _msg(ctx).replace("size_anchor_", "")


@pytest.mark.parametrize("anchor", [{}, {"share_c50": 0.1}, {"share_c50": "x", "share_c80": 0.2}, "on", 1])
def test_malformed_anchor_dict_is_ignored_not_raised(anchor):
    ctx = _anchor_ctx()
    ctx["size_anchor"] = anchor
    assert "SIZE ANCHOR" not in _msg(ctx)                                  # must not raise either


def test_candidate_without_a_unit_premium_gets_no_line_but_the_section_stays():
    cand = _candidate(mid=2.40)
    cand.unit_premium = None
    msg = _msg(_anchor_ctx(), [cand])
    assert "=== SIZE ANCHOR" in msg and "size_anchor (advisory" not in msg


def test_review_message_never_shows_the_anchor():
    review = _run(prompts.build_review_message(_anchor_ctx(), {"id": "p1", "underlying": "SPY", "direction": "bullish"}))
    assert "SIZE ANCHOR" not in review and "size_anchor (advisory" not in review


def test_the_static_charter_is_untouched_so_the_prompt_cache_stays_valid():
    assert "anchor" not in prompts.CHARTER.lower()


def test_the_anchor_is_advisory_it_adds_no_code_side_cap():
    """The only code-side size bound stays solvency: clamp_to_solvency ignores the anchor entirely."""
    assert sizing.clamp_to_solvency(40, 20000.0, 2.40) == (40, [])         # 40 contracts (~$9.6K) >> the 8-contract anchor
    assert sizing.clamp_to_solvency(100, 20000.0, 2.40) == (83, ["clamped_by_solvency:100->83"])
