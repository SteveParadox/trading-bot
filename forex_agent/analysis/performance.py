from __future__ import annotations

from typing import Optional

from forex_agent.data.schemas import PerformanceMetrics, TradeRecord
from forex_agent.data.ingestion import compute_r_multiple, compute_trade_duration


def calculate_performance_metrics(trades: list[TradeRecord]) -> PerformanceMetrics:
    closed = [t for t in trades if t.exit_price is not None]
    if not closed:
        return PerformanceMetrics()

    winners = [t for t in closed if t.is_winner]
    losers = [t for t in closed if not t.is_winner]

    total_pnl = sum(t.pnl for t in closed)
    win_rate = len(winners) / len(closed) if closed else 0.0

    gross_profit = sum(t.pnl for t in winners) if winners else 0.0
    gross_loss = abs(sum(t.pnl for t in losers)) if losers else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

    avg_win = gross_profit / len(winners) if winners else 0.0
    avg_loss = gross_loss / len(losers) if losers else 0.0

    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    r_multiples = []
    for t in closed:
        r = compute_r_multiple(t)
        if r is not None:
            r_multiples.append(r)

    avg_r = sum(r_multiples) / len(r_multiples) if r_multiples else 0.0

    durations = []
    for t in closed:
        d = compute_trade_duration(t)
        if d is not None:
            durations.append(d)
    avg_duration = sum(durations) / len(durations) if durations else 0.0

    pnl_list = [t.pnl for t in closed]
    max_dd, max_dd_pct = _compute_max_drawdown(pnl_list, closed[0].account_balance)

    avg_rr = avg_win / avg_loss if avg_loss > 0 else 0.0

    consecutive_wins, consecutive_losses = _compute_consecutive(closed)

    return PerformanceMetrics(
        total_trades=len(closed),
        winning_trades=len(winners),
        losing_trades=len(losers),
        win_rate=win_rate,
        profit_factor=profit_factor,
        expectancy=expectancy,
        average_win=avg_win,
        average_loss=avg_loss,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        total_pnl=total_pnl,
        avg_r_multiple=avg_r,
        avg_rr_ratio=avg_rr,
        largest_win=max(pnl_list) if pnl_list else 0.0,
        largest_loss=min(pnl_list) if pnl_list else 0.0,
        consecutive_wins=consecutive_wins,
        consecutive_losses=consecutive_losses,
        avg_duration_minutes=avg_duration,
    )


def calculate_rolling_expectancy(
    trades: list[TradeRecord], window: int = 20
) -> list[float]:
    closed = [t for t in trades if t.exit_price is not None]
    results: list[float] = []
    for i in range(len(closed)):
        start = max(0, i - window + 1)
        subset = closed[start : i + 1]
        wins = [t for t in subset if t.is_winner]
        losses = [t for t in subset if not t.is_winner]
        win_rate = len(wins) / len(subset)
        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0.0
        avg_loss = abs(sum(t.pnl for t in losses) / len(losses)) if losses else 0.0
        exp = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        results.append(exp)
    return results


def _compute_max_drawdown(
    pnl_list: list[float], initial_balance: float
) -> tuple[float, float]:
    if not pnl_list:
        return 0.0, 0.0
    equity = initial_balance
    peak = equity
    max_dd = 0.0
    max_dd_pct = 0.0
    for pnl in pnl_list:
        equity += pnl
        if equity > peak:
            peak = equity
        dd = peak - equity
        dd_pct = dd / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
    return max_dd, max_dd_pct


def _compute_consecutive(trades: list[TradeRecord]) -> tuple[int, int]:
    max_wins = 0
    max_losses = 0
    cur_wins = 0
    cur_losses = 0
    for t in trades:
        if t.is_winner:
            cur_wins += 1
            cur_losses = 0
        else:
            cur_losses += 1
            cur_wins = 0
        max_wins = max(max_wins, cur_wins)
        max_losses = max(max_losses, cur_losses)
    return max_wins, max_losses
