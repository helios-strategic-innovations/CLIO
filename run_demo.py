"""Offline synthetic re-entry demonstration; this does not run CLIO or call a model."""
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import reset_tzpath

# Use the bundled IANA timezone data on systems without a system timezone database.
reset_tzpath([str(Path(__file__).resolve().parent / 'tzdata')])
from demo_core import Direction, evaluate_reentry, validate_reentry

NOW = datetime(2026, 11, 4, 15, 0, tzinfo=timezone.utc).timestamp()


def run_case(key, premium, *, age=10, complete=True, prior=True):
    underlying = SimpleNamespace(symbol='SPY', pillars={
        'nofa': SimpleNamespace(name='nofa', value=-0.6, available=True, age_seconds=age),
        'premium': SimpleNamespace(name='premium', value=premium, available=True, age_seconds=age),
    }, last_price=99, vwap=100, ret_15m=-0.01, price_age_seconds=10)
    snapshot = SimpleNamespace(asof=NOW, underlyings={'SPY': underlying}, data_healthy=True)
    anchor = SimpleNamespace(id='synthetic-prior-thesis', underlying='SPY', direction='bearish',
        structure='single', status='closed', closed_time=NOW - 120,
        exit_reason='decider_exit', last_review={'action': 'exit', 'thesis_status': 'broken'}, mode='paper')
    result = evaluate_reentry(symbol='SPY', direction=Direction.BEARISH, captured_at=NOW,
        snapshot=snapshot, history=[anchor] if prior else [], history_complete=complete)
    if validate_reentry(result) != result:
        raise ValueError('synthetic result failed validation')
    return {'id': key, 'input_label': 'SYNTHETIC_NOT_MARKET_DATA',
            'inputs': {'direction': 'bearish', 'nofa': -0.6, 'premium': premium,
                       'cache_age_seconds': age, 'retained_history_complete': complete,
                       'prior_thesis_exit': prior, 'price_below_vwap': True, 'momentum_falling': True},
            'result': result}


def main():
    cases = [run_case('conflicting-flow', 0.2), run_case('reconfirmed', -0.2),
             run_case('missing-time', -0.2, age=None), run_case('no-prior-thesis', -0.2, prior=False)]
    print(json.dumps({'kind': 'DERIVED_OFFLINE_DEMO_SYNTHETIC_NOT_LIVE', 'cases': cases}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
