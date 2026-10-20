"""READ-ONLY reconciliation of self-computed GEX against Unusual Whales GEX.

Reads a selfflow comparison JSONL file (default:
data/intelligence/selfflow_compare.jsonl; one row per symbol per collector
comparison tick, each with a `self` block and a `uw` block) and reports, per
underlying:

  - n          rows where both self GEX and UW net GEX are finite and non-zero
  - median_ratio   median of |self_gex| / |uw_net_gex|
  - sign_agree     share of rows where self and UW GEX have the same sign
  - proposed_scale the normalisation the self GEX would need to match UW's
                   magnitude (= median_ratio; self GEX / proposed_scale)

It NEVER writes anything: the input is opened with mode 'r' only, and the
report goes to stdout. It does not change any production value.

Usage:
    python3 scripts/selfflow_gex_reconcile.py [--input PATH]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from typing import Iterable, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT = os.path.join(_REPO_ROOT, "data", "intelligence", "selfflow_compare.jsonl")


def _finite_nonzero(value) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    if not math.isfinite(f) or f == 0.0:
        return None
    return f


def _pair_from_row(row) -> Optional[tuple[str, float, float]]:
    if not isinstance(row, dict):
        return None
    symbol = row.get("underlying")
    if not isinstance(symbol, str) or not symbol:
        return None
    self_block = row.get("self")
    uw_block = row.get("uw")
    if not isinstance(self_block, dict) or not isinstance(uw_block, dict):
        return None
    uw_gex = uw_block.get("gex")
    if not isinstance(uw_gex, dict):
        return None
    self_gex = _finite_nonzero(self_block.get("gex"))
    uw_net = _finite_nonzero(uw_gex.get("net_gex"))
    if self_gex is None or uw_net is None:
        return None
    return symbol, self_gex, uw_net


def load_pairs(path: str) -> tuple[dict[str, list[tuple[float, float]]], int]:
    """Return ({symbol: [(self_gex, uw_net_gex), ...]}, skipped_line_count).
    Lines that fail to parse or lack a usable pair are counted, not raised."""
    pairs: dict[str, list[tuple[float, float]]] = {}
    skipped = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                skipped += 1
                continue
            parsed = _pair_from_row(row)
            if parsed is None:
                skipped += 1
                continue
            symbol, self_gex, uw_net = parsed
            pairs.setdefault(symbol, []).append((self_gex, uw_net))
    return pairs, skipped


def summarise(pairs: Iterable[tuple[float, float]]) -> dict:
    """PURE. Per-symbol summary. The ratio is the MEDIAN (not the mean) of
    |self|/|uw| so a few outlier ticks cannot dominate the proposed scale."""
    items = list(pairs)
    if not items:
        return {"n": 0, "median_ratio": None, "sign_agree": None, "proposed_scale": None}
    ratios = [abs(s) / abs(u) for s, u in items]
    agree = sum(1 for s, u in items if (s > 0) == (u > 0))
    median_ratio = statistics.median(ratios)
    return {
        "n": len(items),
        "median_ratio": median_ratio,
        "sign_agree": agree / len(items),
        "proposed_scale": median_ratio,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only self vs UW GEX reconciliation.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="selfflow comparison JSONL (read only)")
    args = parser.parse_args(argv)
    pairs, skipped = load_pairs(args.input)
    print(f"input: {args.input}")
    print(f"skipped lines (unparsable or no usable self/UW GEX pair): {skipped}")
    for symbol in sorted(pairs):
        s = summarise(pairs[symbol])
        print(
            f"{symbol}: n={s['n']} median_ratio(self/UW)={s['median_ratio']:.2f} "
            f"sign_agree={s['sign_agree'] * 100:.1f}% "
            f"proposed_scale={s['proposed_scale']:.2f} (normalise self GEX by this to match UW magnitude)"
        )
    if not pairs:
        print("no symbol has a usable self/UW GEX pair")
    return 0


if __name__ == "__main__":
    sys.exit(main())
