"""Portfolio routes — positions, P&L, account, equity history."""

import math
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from agent.integrations.alpaca_values import AlpacaValueError, normalize_position_side
from mission_control.backend.auth import require_auth
from mission_control.backend.deps import get_trading_client, run_in_executor

router = APIRouter(prefix="/api", tags=["portfolio"])
logger = logging.getLogger("mission_control.portfolio")

_PORTFOLIO_HISTORY_PERIODS = {
    "1D": {"canonical": "1D", "alpaca_period": "1D"},
    "5D": {"canonical": "5D", "alpaca_period": "1W"},
    # Legacy aliases remain accepted while clients migrate to the canonical set.
    "1W": {"canonical": "5D", "alpaca_period": "1W"},
    "1M": {"canonical": "1M", "alpaca_period": "1M"},
    "3M": {"canonical": "3M", "alpaca_period": "3M"},
    "6M": {"canonical": "6M", "alpaca_period": "6M"},
    "1Y": {"canonical": "1Y", "alpaca_period": "1A"},
    "1A": {"canonical": "1Y", "alpaca_period": "1A"},
    "YTD": {"canonical": "YTD", "alpaca_period": None},
    "all": {"canonical": "all", "alpaca_period": "all"},
}


def _float_or_none(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _epoch_iso(value: Any) -> str | None:
    ts = _float_or_none(value)
    if ts is None or ts <= 0:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _history_timestamp_iso(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        numeric = _float_or_none(value)
        if numeric is not None:
            try:
                return datetime.fromtimestamp(numeric, tz=timezone.utc).isoformat()
            except (OSError, OverflowError, ValueError):
                return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return str(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _portfolio_history_request_kwargs(
    requested_period: str,
    timeframe: str,
    *,
    now: datetime | None = None,
) -> tuple[str, dict[str, Any]]:
    config = _PORTFOLIO_HISTORY_PERIODS[requested_period]
    kwargs: dict[str, Any] = {"timeframe": timeframe}
    if requested_period == "YTD":
        current = now or datetime.now(timezone.utc)
        kwargs["start"] = datetime(current.year, 1, 1, tzinfo=timezone.utc)
    else:
        kwargs["period"] = config["alpaca_period"]
    return str(config["canonical"]), kwargs


def _portfolio_history_points(history: Any) -> tuple[list[dict[str, Any]], dict[str, list[Any]]]:
    legacy = {
        "timestamps": list(getattr(history, "timestamp", None) or []),
        "equity": list(getattr(history, "equity", None) or []),
        "profit_loss": list(getattr(history, "profit_loss", None) or []),
        "profit_loss_pct": list(getattr(history, "profit_loss_pct", None) or []),
    }
    point_count = max((len(values) for values in legacy.values()), default=0)
    points = []
    for index in range(point_count):
        value = lambda key: legacy[key][index] if index < len(legacy[key]) else None
        points.append({
            "timestamp": _history_timestamp_iso(value("timestamps")),
            "equity": _float_or_none(value("equity")),
            "profit_loss": _float_or_none(value("profit_loss")),
            "profit_loss_pct": _float_or_none(value("profit_loss_pct")),
        })
    return points, legacy


def _portfolio_history_response(
    *,
    points: list[dict[str, Any]],
    legacy: dict[str, list[Any]],
    requested_period: str,
    period: str,
    timeframe: str,
    reason_code: str,
    stale: bool,
    as_of: str | None = None,
) -> dict[str, Any]:
    resolved_as_of = as_of or next(
        (point.get("timestamp") for point in reversed(points) if point.get("timestamp")),
        datetime.now(timezone.utc).isoformat(),
    )
    return {
        "points": points,
        "source": "alpaca_portfolio_history",
        "as_of": resolved_as_of,
        "stale": stale,
        "reason_code": reason_code,
        "requested_period": requested_period,
        "period": period,
        "timeframe": timeframe,
        # Additive compatibility for the current dashboard and older API clients.
        "bars": points,
        **legacy,
    }


def _holding_days(entry_time: Any, close_time: Any, *, now: float) -> int | None:
    entry_ts = _float_or_none(entry_time)
    end_ts = _float_or_none(close_time) or now
    if entry_ts is None or end_ts < entry_ts:
        return None
    return int((end_ts - entry_ts) // 86400)


def _normalized_status(value: Any) -> str:
    status = str(value or "unknown").strip().lower()
    return {"OPEN": "open", "CLOSED": "closed"}.get(status, status)


def _dpof_positions_to_legacy_lot_dict(positions: list) -> dict[str, Any]:
    """Translate DPOF PositionRecords into the `{"positions": {id: {...}}}`
    shape `_build_tax_lots_from_positions` already knows how to read, so the
    estimator logic (cost basis, wash-sale heuristics, holding term) is shared
    rather than duplicated. DPOF trades options only, so `is_options` is
    always True and `is_short` always False (long premium only)."""
    try:
        from agent.dpof.readmodel import position_label
    except Exception:
        position_label = None

    out: dict[str, Any] = {}
    for p in positions:
        # Skip pending (order not yet filled) / cancelled positions — they never
        # held real risk, and showing them as phantom "open" lots (qty_open is
        # typically 0 for these) would clutter an otherwise accurate estimate.
        if p.status not in ("open", "closing", "closed"):
            continue
        try:
            contract = position_label(p) if position_label else p.underlying
        except Exception:
            contract = p.underlying
        quantity = p.qty_open if p.status != "closed" else p.qty_original
        out[p.id] = {
            "symbol": p.underlying,
            "contract": contract,
            "is_options": True,
            "is_short": False,
            "status": "closed" if p.status == "closed" else "open",
            "quantity": quantity,
            "entry_price": p.entry_price,
            "close_price": None,
            "pnl": p.realized_pnl if p.status == "closed" else None,
            "entry_time": p.entry_time,
            "close_time": p.closed_time,
            "category": p.structure,
            "direction": p.direction,
            "trade_id": p.id,
        }
    return {"positions": out}


def _build_tax_lots_from_positions(
    position_log: dict[str, Any] | None,
    *,
    limit: int,
    status: str = "all",
    now: float | None = None,
) -> dict[str, Any]:
    """Build a read-only estimated tax-lot view from a position-log-shaped dict
    (either Hydrinia's legacy position_log.json, or DPOF positions translated
    via `_dpof_positions_to_legacy_lot_dict`)."""
    now_ts = now if now is not None else time.time()
    positions = (position_log or {}).get("positions") or {}
    position_items = [
        (str(pid), pos) for pid, pos in positions.items()
        if isinstance(pos, dict)
    ]

    same_symbol_entries: dict[str, list[tuple[str, float | None, str]]] = {}
    for position_id, pos in position_items:
        symbol = str(pos.get("symbol") or "").upper()
        if not symbol:
            continue
        same_symbol_entries.setdefault(symbol, []).append((
            position_id,
            _float_or_none(pos.get("entry_time")),
            _normalized_status(pos.get("status")),
        ))

    lots: list[dict[str, Any]] = []
    open_count = 0
    closed_count = 0

    for position_id, pos in position_items:
        lot_status = _normalized_status(pos.get("status"))
        if lot_status == "open":
            open_count += 1
        elif lot_status == "closed":
            closed_count += 1
        if status != "all" and lot_status != status:
            continue

        symbol = str(pos.get("symbol") or "").upper()
        contract = str(pos.get("contract") or symbol).upper()
        is_options = bool(pos.get("is_options"))
        quantity = _float_or_none(pos.get("quantity")) or 0.0
        entry_price = _float_or_none(pos.get("entry_price"))
        close_price = _float_or_none(pos.get("close_price"))
        cost_basis = _float_or_none(pos.get("cost_basis"))
        if cost_basis is None and entry_price is not None:
            multiplier = 100.0 if is_options else 1.0
            cost_basis = abs(quantity) * entry_price * multiplier

        pnl = _float_or_none(pos.get("pnl"))
        if pnl is None:
            pnl = _float_or_none(pos.get("realized_pnl"))
        entry_ts = _float_or_none(pos.get("entry_time"))
        close_ts = _float_or_none(pos.get("close_time"))
        held_days = _holding_days(entry_ts, close_ts, now=now_ts)

        wash_sale_risk = False
        wash_sale_reason = None
        if pnl is not None and pnl < 0 and lot_status == "closed":
            open_same_symbol = any(
                other_id != position_id and other_status == "open"
                for other_id, _, other_status in same_symbol_entries.get(symbol, [])
            )
            replacement_within_window = any(
                other_id != position_id
                and other_entry is not None
                and close_ts is not None
                and close_ts <= other_entry <= close_ts + 30 * 86400
                for other_id, other_entry, _ in same_symbol_entries.get(symbol, [])
            )
            forward_window_open = close_ts is not None and now_ts <= close_ts + 30 * 86400
            wash_sale_risk = bool(open_same_symbol or replacement_within_window or forward_window_open)
            if open_same_symbol:
                wash_sale_reason = "same_symbol_open"
            elif replacement_within_window:
                wash_sale_reason = "same_symbol_replacement_within_30d"
            elif forward_window_open:
                wash_sale_reason = "loss_window_still_open"

        lots.append({
            "lot_id": position_id,
            "position_id": position_id,
            "symbol": symbol,
            "contract": contract,
            "is_options": is_options,
            "is_short": bool(pos.get("is_short")),
            "status": lot_status,
            "quantity": quantity,
            "entry_price": entry_price,
            "close_price": close_price,
            "estimated_cost_basis": cost_basis,
            "pnl": pnl,
            "pnl_pct": _float_or_none(pos.get("pnl_pct")),
            "entry_time": _epoch_iso(entry_ts),
            "close_time": _epoch_iso(close_ts),
            "entry_time_epoch": entry_ts,
            "close_time_epoch": close_ts,
            "holding_days": held_days,
            "holding_term": "long_term" if held_days is not None and held_days >= 365 else "short_term",
            "wash_sale_risk": wash_sale_risk,
            "wash_sale_reason": wash_sale_reason,
            "category": pos.get("category"),
            "direction": pos.get("direction"),
            "trade_id": pos.get("trade_id"),
            "broker_verified": False,
            "source": "clio_dpof_ledger",
        })

    lots.sort(
        key=lambda lot: (
            lot.get("entry_time_epoch") is not None,
            lot.get("entry_time_epoch") or 0,
        ),
        reverse=True,
    )

    return {
        "lots": lots[:limit],
        "total_count": len(lots),
        "open_count": open_count,
        "closed_count": closed_count,
        "limit": limit,
        "source": "clio_dpof_ledger",
        "broker_verified": False,
        "generated_at": datetime.fromtimestamp(now_ts, tz=timezone.utc).isoformat(),
        "warning": (
            "Display estimate only: derived from CLIO's own DPOF ledger "
            "(agent/dpof/ positions.json), not broker-confirmed tax records."
        ),
    }


@router.get("/portfolio")
async def get_portfolio(_=Depends(require_auth)):
    """Current positions, unrealized P&L, account summary."""
    try:
        client = get_trading_client()
        account, positions = await run_in_executor(
            lambda: (client.get_account(), client.get_all_positions())
        )

        pos_list = []
        position_rejections = []
        for p in positions:
            try:
                side = normalize_position_side(getattr(p, "side", None))
                pos_list.append({
                    "symbol": p.symbol,
                    "qty": float(p.qty or 0),
                    "side": f"PositionSide.{side.upper()}",
                    "side_normalized": side,
                    "avg_entry_price": float(p.avg_entry_price or 0),
                    "current_price": float(p.current_price or 0),
                    "market_value": float(p.market_value or 0),
                    "unrealized_pl": float(p.unrealized_pl or 0),
                    "unrealized_plpc": float(p.unrealized_plpc or 0) * 100,
                    "asset_class": str(getattr(p, "asset_class", "us_equity")),
                })
            except AlpacaValueError as exc:
                position_rejections.append({
                    "symbol": str(getattr(p, "symbol", "")),
                    "reason_code": "UNKNOWN_POSITION_SIDE",
                    "detail": str(exc),
                })
            except (TypeError, ValueError):
                continue

        return {
            "account": {
                "equity": float(account.equity or 0),
                "cash": float(account.cash or 0),
                "buying_power": float(account.buying_power or 0),
                "portfolio_value": float(account.portfolio_value or 0),
                "last_equity": float(account.last_equity or 0),
                "day_pl": float(account.equity or 0) - float(account.last_equity or 0),
                "day_pl_pct": (
                    (float(account.equity or 0) - float(account.last_equity or 0))
                    / float(account.last_equity or 1) * 100
                ),
                "status": str(account.status),
                "pattern_day_trader": bool(account.pattern_day_trader),
                "trading_blocked": bool(account.trading_blocked),
            },
            "positions": pos_list,
            "position_count": len(pos_list),
            "position_rejections": position_rejections,
        }
    except Exception as e:
        logger.error(f"Portfolio fetch failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/portfolio/tax-lots")
async def get_tax_lots(
    limit: int = Query(80, ge=1, le=500),
    status: str = Query("all", pattern="^(all|open|closed)$"),
    _=Depends(require_auth),
):
    """Estimated tax lots derived from CLIO's own DPOF position ledger."""
    try:
        from agent.dpof import readmodel

        positions = await run_in_executor(readmodel.load_positions)
        pos_log = _dpof_positions_to_legacy_lot_dict(positions)
        result = _build_tax_lots_from_positions(pos_log, limit=limit, status=status)
        result["source_path"] = os.path.join(await run_in_executor(readmodel.data_dir), "positions.json")
        return result
    except Exception as e:
        logger.error(f"Tax-lot view failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/portfolio/history")
async def get_portfolio_history(
    period: str = Query("1M", pattern="^(1D|5D|1W|1M|3M|6M|1Y|1A|YTD|all)$"),
    timeframe: str = Query("1D", pattern="^(1Min|5Min|15Min|1H|1D)$"),
    _=Depends(require_auth),
):
    """Historical equity curve."""
    canonical_period, request_kwargs = _portfolio_history_request_kwargs(period, timeframe)
    try:
        from alpaca.trading.requests import GetPortfolioHistoryRequest
        client = get_trading_client()
        history = await run_in_executor(
            lambda: client.get_portfolio_history(
                GetPortfolioHistoryRequest(**request_kwargs)
            )
        )
        points, legacy = _portfolio_history_points(history)
        return _portfolio_history_response(
            points=points,
            legacy=legacy,
            requested_period=period,
            period=canonical_period,
            timeframe=timeframe,
            reason_code="OK" if points else "NO_PORTFOLIO_HISTORY",
            stale=not bool(points),
        )
    except Exception as e:
        logger.error(f"Portfolio history failed: {e}", exc_info=True)
        return _portfolio_history_response(
            points=[],
            legacy={
                "timestamps": [],
                "equity": [],
                "profit_loss": [],
                "profit_loss_pct": [],
            },
            requested_period=period,
            period=canonical_period,
            timeframe=timeframe,
            reason_code="ALPACA_PORTFOLIO_HISTORY_UNAVAILABLE",
            stale=True,
        )
