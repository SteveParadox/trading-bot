from __future__ import annotations

from collections import defaultdict

from forex_agent.data.schemas import TradeRecord
from forex_agent.data.ingestion import compute_r_multiple, extract_session, extract_day_of_week


def analyze_r_distribution(trades: list[TradeRecord]) -> dict:
    closed = [t for t in trades if t.exit_price is not None]
    r_values = []
    for t in closed:
        r = compute_r_multiple(t)
        if r is not None:
            r_values.append(r)
    if not r_values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    import math
    n = len(r_values)
    mean = sum(r_values) / n
    variance = sum((x - mean) ** 2 for x in r_values) / n if n > 1 else 0.0
    std = math.sqrt(variance)
    return {
        "mean": mean,
        "std": std,
        "min": min(r_values),
        "max": max(r_values),
        "count": n,
        "positive_r_pct": sum(1 for r in r_values if r > 0) / n,
    }


def performance_by_instrument(trades: list[TradeRecord]) -> dict[str, dict]:
    by_symbol: dict[str, list[TradeRecord]] = defaultdict(list)
    for t in trades:
        by_symbol[t.symbol].append(t)

    results = {}
    for symbol, sym_trades in by_symbol.items():
        closed = [t for t in sym_trades if t.exit_price is not None]
        wins = [t for t in closed if t.is_winner]
        total_pnl = sum(t.pnl for t in closed)
        results[symbol] = {
            "total_trades": len(closed),
            "win_rate": len(wins) / len(closed) if closed else 0.0,
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / len(closed) if closed else 0.0,
        }
    return results


def performance_by_session(trades: list[TradeRecord]) -> dict[str, dict]:
    by_session: dict[str, list[TradeRecord]] = defaultdict(list)
    for t in trades:
        session = extract_session(t.entry_time)
        by_session[session].append(t)

    results = {}
    for session, sess_trades in by_session.items():
        closed = [t for t in sess_trades if t.exit_price is not None]
        wins = [t for t in closed if t.is_winner]
        total_pnl = sum(t.pnl for t in closed)
        results[session] = {
            "total_trades": len(closed),
            "win_rate": len(wins) / len(closed) if closed else 0.0,
            "total_pnl": total_pnl,
        }
    return results


def performance_by_day(trades: list[TradeRecord]) -> dict[str, dict]:
    by_day: dict[str, list[TradeRecord]] = defaultdict(list)
    for t in trades:
        day = extract_day_of_week(t.entry_time)
        by_day[day].append(t)

    results = {}
    for day, day_trades in by_day.items():
        closed = [t for t in day_trades if t.exit_price is not None]
        wins = [t for t in closed if t.is_winner]
        total_pnl = sum(t.pnl for t in closed)
        results[day] = {
            "total_trades": len(closed),
            "win_rate": len(wins) / len(closed) if closed else 0.0,
            "total_pnl": total_pnl,
        }
    return results
