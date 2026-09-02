"""Performance summaries for the FX forward-test dashboard."""

from __future__ import annotations

import math
from datetime import datetime
from statistics import mean, pstdev
from typing import Any

from fxbot.journal import StructuredJournal, row_to_dict


def performance_summary(
    journal: StructuredJournal,
    *,
    instrument: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> dict[str, Any]:
    trades = journal.filtered_trades(instrument=instrument, start=start, end=end, limit=10_000)
    equity = journal.latest_equity(limit=10_000)
    orders = journal.recent_orders(limit=10_000)
    risk_by_trade = {
        str(order.broker_trade_id): float(order.risk_amount or 0.0)
        for order in orders
        if order.broker_trade_id
    }

    closed = [trade for trade in trades if trade.state == "closed"]
    open_trades = [trade for trade in trades if trade.state == "open"]
    trade_pnls = [float(trade.realized_pl or 0.0) + float(trade.financing or 0.0) for trade in closed]
    wins = [value for value in trade_pnls if value > 0]
    losses = [value for value in trade_pnls if value < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    r_multiples = [
        pnl / risk_by_trade.get(str(trade.broker_trade_id), 0.0)
        for pnl, trade in zip(trade_pnls, closed)
        if risk_by_trade.get(str(trade.broker_trade_id), 0.0) > 0
    ]
    equity_values = [float(row.equity or 0.0) for row in equity]
    returns = _equity_returns(equity_values)

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
    return {
        "trade_count": len(closed),
        "open_trade_count": len(open_trades),
        "win_rate": (len(wins) / len(closed)) if closed else 0.0,
        "profit_factor": profit_factor,
        "total_pnl": sum(trade_pnls),
        "realized_pnl": sum(float(trade.realized_pl or 0.0) for trade in closed),
        "financing": sum(float(trade.financing or 0.0) for trade in closed + open_trades),
        "max_drawdown": _max_drawdown(equity_values),
        "sharpe_ratio": _sharpe(returns),
        "sortino_ratio": _sortino(returns),
        "average_r_multiple": mean(r_multiples) if r_multiples else 0.0,
        "average_rr": mean(r_multiples) if r_multiples else 0.0,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "equity_points": len(equity_values),
    }


def live_snapshot(journal: StructuredJournal, config_payload: dict[str, Any]) -> dict[str, Any]:
    equity = journal.latest_equity(limit=300)
    return {
        "status": row_to_dict(journal.get_state()),
        "positions": [row_to_dict(row) for row in journal.current_positions()],
        "equity_curve": [row_to_dict(row) for row in equity],
        "performance": performance_summary(journal),
        "recent_trades": [row_to_dict(row) for row in journal.recent_trades(limit=50)],
        "recent_signals": [row_to_dict(row) for row in journal.recent_signals(limit=50)],
        "config": config_payload,
    }


def _equity_returns(values: list[float]) -> list[float]:
    returns: list[float] = []
    for previous, current in zip(values, values[1:]):
        if previous > 0:
            returns.append((current - previous) / previous)
    return returns


def _max_drawdown(values: list[float]) -> dict[str, float]:
    peak = 0.0
    max_amount = 0.0
    max_pct = 0.0
    for value in values:
        peak = max(peak, value)
        if peak <= 0:
            continue
        amount = peak - value
        pct = amount / peak
        if pct > max_pct:
            max_pct = pct
            max_amount = amount
    return {"amount": max_amount, "pct": max_pct}


def _sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    volatility = pstdev(returns)
    if volatility <= 0:
        return 0.0
    return mean(returns) / volatility * math.sqrt(len(returns))


def _sortino(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    downside = [value for value in returns if value < 0]
    downside_dev = pstdev(downside) if len(downside) > 1 else 0.0
    if downside_dev <= 0:
        return 0.0
    return mean(returns) / downside_dev * math.sqrt(len(returns))
