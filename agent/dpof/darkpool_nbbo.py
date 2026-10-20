"""PURE. NBBO-midpoint signed-flow ratio for UW dark-pool prints.

Mirrors the rule in agent/flowcollector/aggregator.py::compute_darkpool and
agent/flowcollector/classifier.py::classify_dark_trade_side: each print is
classified against the NBBO midpoint (strictly above = buy, strictly below =
sell, exactly at = mid, skipped), and the ratio is signed / gross classified
notional in [-1, 1]. Replaces the median-based bias in
agent/integrations/uw_flow_intel.py::_compute_dp_signals, which cannot leave
a tiny band around 0. NOT YET WIRED: Wave 2 wires it behind a default-off
flag. Unverified assumptions: UW's ext_hour_sold_codes semantics are not
used; UW nbbo_bid/nbbo_ask are assumed to be the quote at print time.
"""

import math


def _num(v):
    """Finite float or None (strings allowed; bool/None/garbage -> None)."""
    if isinstance(v, bool):  # float(True) == 1.0 would otherwise pass as a price/size
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def nbbo_signed_ratio(prints, *, min_classified: int = 20) -> dict | None:
    """Signed/gross NBBO-midpoint ratio over UW dark prints, or None.

    Returns None for non-list/tuple input or when fewer than `min_classified`
    prints classify (never a fabricated 0.0 from a thin window).
    """
    if not isinstance(prints, (list, tuple)):
        return None
    signed = 0.0
    gross = 0.0
    classified = 0
    at_mid = 0
    no_nbbo = 0
    canceled = 0
    bad_px = 0
    non_dict = 0
    for p in prints:
        if not isinstance(p, dict):
            non_dict += 1
            continue
        cx = p.get("canceled")
        if cx is True or str(cx).strip().lower() == "true":
            canceled += 1
            continue
        price = _num(p.get("price"))
        size = _num(p.get("size"))
        if price is None or size is None or price <= 0 or size <= 0:
            bad_px += 1
            continue
        bid = _num(p.get("nbbo_bid"))
        ask = _num(p.get("nbbo_ask"))
        if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:
            no_nbbo += 1
            continue
        mid = (bid + ask) / 2.0
        # Tolerance: prices arrive as strings, so a print at the midpoint can
        # carry float noise that would otherwise misclassify it as buy/sell.
        if abs(price - mid) <= 1e-9:
            at_mid += 1
            continue
        side = 1 if price > mid else -1
        notional = price * size
        signed += side * notional
        gross += notional
        classified += 1
    if classified < min_classified or gross <= 0:
        return None
    return {
        "ratio": max(-1.0, min(1.0, signed / gross)),
        "signed_notional": signed,
        "gross_notional": gross,
        "classified": classified,
        "at_mid": at_mid,
        "no_nbbo": no_nbbo,
        "canceled": canceled,
        "bad_px": bad_px,
        "non_dict": non_dict,
        "total_prints": len(prints),
    }
