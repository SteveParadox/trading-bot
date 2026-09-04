from __future__ import annotations

from forex_agent.data.schemas import TradeRecord


def calculate_max_drawdown(trades: list[TradeRecord]) -> dict:
    closed = [t for t in trades if t.exit_price is not None]
    if not closed:
        return {"max_drawdown": 0.0, "max_drawdown_pct": 0.0, "peak_balance": 0.0}

    balance = closed[0].account_balance
    peak = balance
    max_dd = 0.0
    max_dd_pct = 0.0

    for t in closed:
        balance += t.pnl
        if balance > peak:
            peak = balance
        dd = peak - balance
        dd_pct = dd / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

    return {
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct,
        "peak_balance": peak,
    }


def calculate_tail_risk(trades: list[TradeRecord]) -> dict:
    closed = [t for t in trades if t.exit_price is not None]
    if not closed:
        return {"var_95": 0.0, "cvar_95": 0.0, "worst_trade": 0.0}

    pnls = sorted(t.pnl for t in closed)
    n = len(pnls)
    var_idx = int(n * 0.05)
    var_95 = pnls[var_idx] if var_idx < n else pnls[0]

    tail = pnls[: var_idx + 1] if var_idx < n else pnls[:1]
    cvar_95 = sum(tail) / len(tail) if tail else 0.0

    return {
        "var_95": var_95,
        "cvar_95": cvar_95,
        "worst_trade": pnls[0],
    }


def calculate_position_sizing(
    account_balance: float,
    risk_pct: float,
    entry_price: float,
    stop_loss: float,
) -> dict:
    risk_amount = account_balance * (risk_pct / 100.0)
    risk_pips = abs(entry_price - stop_loss)
    if risk_pips == 0:
        return {"risk_amount": 0.0, "position_size": 0.0, "risk_pips": 0.0}
    position_size = risk_amount / (risk_pips * 100000)
    return {
        "risk_amount": risk_amount,
        "position_size": position_size,
        "risk_pips": risk_pips,
    }
