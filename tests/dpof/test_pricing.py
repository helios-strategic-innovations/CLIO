"""Tests for agent/dpof/pricing.py (existing module; DO NOT EDIT pricing.py)."""

from dataclasses import replace

from agent.dpof.pricing import entry_limit_price, exit_limit_price, reprice_entry, reprice_exit


# ── entry_limit_price ───────────────────────────────────────────────────────


def test_entry_limit_price_capped_at_ask(cfg):
    # Wide spread + big lean pushes the raw lean price past ask; slippage
    # ceiling is loosened so ask is the binding cap.
    c = replace(cfg, entry_lean=1.5, entry_max_slippage_pct=1.0)
    bid, ask = 1.00, 1.20
    price = entry_limit_price(bid, ask, c)
    assert price == ask


def test_entry_limit_price_capped_at_slippage_ceiling(cfg):
    # Default slippage (5%) binds before the wide-spread lean price reaches ask.
    c = replace(cfg, entry_lean=1.0)
    bid, ask = 1.00, 2.00
    mid = (bid + ask) / 2.0
    price = entry_limit_price(bid, ask, c)
    expected_cap = round(mid * (1.0 + c.entry_max_slippage_pct), 2)
    assert price == expected_cap
    assert price < ask


def test_entry_limit_price_uncapped_lean(cfg):
    c = replace(cfg, entry_lean=0.25, entry_max_slippage_pct=1.0)
    bid, ask = 1.00, 1.20
    mid = (bid + ask) / 2.0
    price = entry_limit_price(bid, ask, c)
    assert price == round(mid + 0.25 * (ask - bid), 2)


def test_entry_limit_price_unusable_quote_none(cfg):
    assert entry_limit_price(None, 1.20, cfg) is None
    assert entry_limit_price(1.00, None, cfg) is None
    assert entry_limit_price(1.00, 0.0, cfg) is None
    assert entry_limit_price(-0.01, 1.20, cfg) is None


def test_entry_limit_price_crossed_quote_none(cfg):
    assert entry_limit_price(1.20, 1.00, cfg) is None  # bid > ask


def test_entry_limit_price_locked_market_ok(cfg):
    # bid == ask is unusual but not "crossed" — should still price.
    price = entry_limit_price(1.00, 1.00, cfg)
    assert price == 1.00


# ── exit_limit_price ────────────────────────────────────────────────────────


def test_exit_limit_price_floored_at_bid(cfg):
    c = replace(cfg, exit_lean=1.0)
    bid, ask = 1.00, 1.20
    price = exit_limit_price(bid, ask, c)
    assert price == bid


def test_exit_limit_price_uncapped_lean(cfg):
    c = replace(cfg, exit_lean=0.25)
    bid, ask = 1.00, 1.20
    mid = (bid + ask) / 2.0
    price = exit_limit_price(bid, ask, c)
    assert price == round(mid - 0.25 * (ask - bid), 2)


def test_exit_limit_price_unusable_quote_none(cfg):
    assert exit_limit_price(None, 1.20, cfg) is None
    assert exit_limit_price(1.00, None, cfg) is None
    assert exit_limit_price(1.00, 0.0, cfg) is None
    assert exit_limit_price(-0.01, 1.20, cfg) is None


def test_exit_limit_price_crossed_quote_none(cfg):
    assert exit_limit_price(1.20, 1.00, cfg) is None


# ── reprice_entry ────────────────────────────────────────────────────────────


def test_reprice_entry_monotonic_and_reaches_cap_on_last_attempt(cfg):
    c = replace(cfg, entry_lean=0.1, entry_max_reprices=3, entry_max_slippage_pct=0.05)
    bid, ask = 1.00, 1.50
    mid = (bid + ask) / 2.0
    cap = round(min(mid * (1.0 + c.entry_max_slippage_pct), ask), 2)

    prices = [reprice_entry(bid, ask, attempt, c) for attempt in range(c.entry_max_reprices + 1)]
    assert all(p is not None for p in prices)
    # Non-decreasing walk toward the cap.
    for earlier, later in zip(prices, prices[1:]):
        assert later >= earlier
    assert prices[0] == entry_limit_price(bid, ask, c)
    assert prices[-1] == cap


def test_reprice_entry_attempt_zero_equals_base(cfg):
    bid, ask = 1.00, 1.20
    assert reprice_entry(bid, ask, 0, cfg) == entry_limit_price(bid, ask, cfg)


def test_reprice_entry_negative_attempt_clamped_to_zero(cfg):
    bid, ask = 1.00, 1.20
    assert reprice_entry(bid, ask, -5, cfg) == reprice_entry(bid, ask, 0, cfg)


def test_reprice_entry_unusable_quote_none(cfg):
    assert reprice_entry(None, 1.20, 1, cfg) is None
    assert reprice_entry(1.20, 1.00, 1, cfg) is None  # crossed


# ── reprice_exit ─────────────────────────────────────────────────────────────


def test_reprice_exit_monotonic_and_reaches_bid_on_last_attempt(cfg):
    c = replace(cfg, exit_lean=0.1, exit_max_reprices=3)
    bid, ask = 1.00, 1.50

    prices = [reprice_exit(bid, ask, attempt, c) for attempt in range(c.exit_max_reprices + 1)]
    assert all(p is not None for p in prices)
    # Non-increasing walk toward the bid.
    for earlier, later in zip(prices, prices[1:]):
        assert later <= earlier
    assert prices[0] == exit_limit_price(bid, ask, c)
    assert prices[-1] == bid


def test_reprice_exit_attempt_zero_equals_base(cfg):
    bid, ask = 1.00, 1.20
    assert reprice_exit(bid, ask, 0, cfg) == exit_limit_price(bid, ask, cfg)


def test_reprice_exit_unusable_quote_none(cfg):
    assert reprice_exit(None, 1.20, 1, cfg) is None
    assert reprice_exit(1.20, 1.00, 1, cfg) is None  # crossed
