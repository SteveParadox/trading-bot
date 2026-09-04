from __future__ import annotations

from forex_agent.data.schemas import TradeRecord


def analyze_spread_quality(trades: list[TradeRecord]) -> dict:
    closed = [t for t in trades if t.exit_price is not None]
    if not closed:
        return {"avg_spread": 0.0, "max_spread": 0.0, "spread_impact": 0.0}

    spreads = [t.spread_at_entry for t in closed]
    avg_spread = sum(spreads) / len(spreads)
    max_spread = max(spreads)
    total_spread_cost = sum(
        t.spread_at_entry * t.position_size * 100000 * 0.00001 for t in closed
    )
    return {
        "avg_spread": avg_spread,
        "max_spread": max_spread,
        "spread_impact": total_spread_cost,
        "spread_count": len(spreads),
    }


def analyze_slippage(trades: list[TradeRecord]) -> dict:
    closed = [t for t in trades if t.exit_price is not None]
    if not closed:
        return {"avg_slippage": 0.0, "max_slippage": 0.0, "total_slippage_cost": 0.0}

    slippages = [abs(t.slippage_pips) for t in closed]
    avg_slip = sum(slippages) / len(slippages)
    max_slip = max(slippages)
    total_cost = sum(
        t.slippage_pips * t.position_size * 100000 * 0.00001 for t in closed
    )
    return {
        "avg_slippage": avg_slip,
        "max_slippage": max_slip,
        "total_slippage_cost": total_cost,
        "slippage_count": len(slippages),
    }


def analyze_execution_quality(trades: list[TradeRecord]) -> dict:
    spread_stats = analyze_spread_quality(trades)
    slippage_stats = analyze_slippage(trades)
    total_cost = spread_stats["spread_impact"] + slippage_stats["total_slippage_cost"]
    return {
        "spread": spread_stats,
        "slippage": slippage_stats,
        "total_execution_cost": total_cost,
    }
