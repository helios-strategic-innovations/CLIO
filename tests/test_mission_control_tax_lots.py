import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_tax_lots_estimate_options_contract_multiplier():
    from mission_control.backend.routes.portfolio import _build_tax_lots_from_positions

    now = 1_800_000_000
    result = _build_tax_lots_from_positions({
        "positions": {
            "pos_opt": {
                "symbol": "SPY",
                "contract": "SPY260821C00600000",
                "is_options": True,
                "quantity": 2,
                "entry_price": 1.25,
                "entry_time": now - 3600,
                "status": "open",
            }
        }
    }, limit=10, now=now)

    assert result["broker_verified"] is False
    assert result["open_count"] == 1
    assert result["lots"][0]["estimated_cost_basis"] == 250.0
    assert result["lots"][0]["holding_term"] == "short_term"


def test_tax_lots_flags_recent_loss_wash_sale_window():
    from mission_control.backend.routes.portfolio import _build_tax_lots_from_positions

    day = 86400
    now = 1_800_000_000
    result = _build_tax_lots_from_positions({
        "positions": {
            "loss": {
                "symbol": "ABC",
                "contract": "ABC",
                "is_options": False,
                "quantity": 10,
                "entry_price": 10,
                "entry_time": now - 20 * day,
                "close_time": now - 5 * day,
                "status": "closed",
                "pnl": -25,
            },
            "replacement": {
                "symbol": "ABC",
                "contract": "ABC",
                "is_options": False,
                "quantity": 5,
                "entry_price": 11,
                "entry_time": now - 2 * day,
                "status": "open",
            },
        }
    }, limit=10, now=now)

    lot = next(lot for lot in result["lots"] if lot["lot_id"] == "loss")
    assert lot["wash_sale_risk"] is True
    assert lot["wash_sale_reason"] == "same_symbol_open"


def test_tax_lots_status_filter_and_limit():
    from mission_control.backend.routes.portfolio import _build_tax_lots_from_positions

    result = _build_tax_lots_from_positions({
        "positions": {
            "old": {"symbol": "AAA", "entry_time": 1, "status": "closed"},
            "open": {"symbol": "BBB", "entry_time": 2, "status": "open"},
        }
    }, limit=1, status="open", now=10)

    assert result["total_count"] == 1
    assert result["lots"][0]["lot_id"] == "open"
