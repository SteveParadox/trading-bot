from __future__ import annotations

from typing import Optional

from forex_agent.data.schemas import TradeRecord


def calculate_mfe_mae(
    entry_price: float,
    direction: str,
    price_path: list[float],
) -> tuple[Optional[float], Optional[float]]:
    """Calculate Maximum Favorable and Adverse Excursion from a price path.

    Args:
        entry_price: Price at entry.
        direction: "LONG" or "SHORT".
        price_path: Sequence of prices observed while the trade was open
            (excluding the entry price itself).

    Returns:
        (mfe, mae) as absolute price distances (favorable positive,
        adverse positive magnitude). Returns (None, None) if path is empty.
    """
    if not price_path:
        return None, None

    if direction == "LONG":
        favorable = max((p - entry_price) for p in price_path)
        adverse = max((entry_price - p) for p in price_path)
    else:
        favorable = max((entry_price - p) for p in price_path)
        adverse = max((p - entry_price) for p in price_path)

    return max(favorable, 0.0), max(adverse, 0.0)


def mfe_mae_in_r(trade: TradeRecord) -> tuple[Optional[float], Optional[float]]:
    """Convert MFE/MAE to R-multiple terms (relative to stop distance)."""
    risk = abs(trade.entry_price - trade.stop_loss) if trade.stop_loss else 0.0
    if risk == 0:
        return None, None
    mfe_r = trade.mfe / risk if trade.mfe is not None else None
    mae_r = trade.mae / risk if trade.mae is not None else None
    return mfe_r, mae_r


def was_take_profit_touched(trade: TradeRecord) -> Optional[bool]:
    """Whether MFE indicates the take-profit was reached at some point."""
    if trade.mfe is None or trade.take_profit is None:
        return None
    if trade.direction == "LONG":
        return (trade.entry_price + trade.mfe) >= trade.take_profit
    return (trade.entry_price - trade.mfe) <= trade.take_profit


def stopped_out_at_full_distance(trade: TradeRecord) -> Optional[bool]:
    """Whether the exit looks like a full stop hit (MAE reaches stop)."""
    if trade.mae is None or trade.stop_loss is None:
        return None
    if trade.direction == "LONG":
        return (trade.entry_price - trade.mae) <= trade.stop_loss
    return (trade.entry_price + trade.mae) >= trade.stop_loss


def efficiency_ratio(trade: TradeRecord) -> Optional[float]:
    """Efficiency = favorable excursion / total excursion.

    A value near 1 means price moved efficiently toward target; a low value
    means churn / poor market microstructure for the trade.
    """
    if trade.mfe is None or trade.mae is None:
        return None
    total = trade.mfe + trade.mae
    if total == 0:
        return 0.0
    return trade.mfe / total


def price_recovered_from_mae(trade: TradeRecord) -> Optional[bool]:
    """Whether price recovered substantially from its worst adverse excursion."""
    if trade.mae is None or trade.exit_price is None:
        return None
    if trade.direction == "LONG":
        worst = trade.entry_price - trade.mae
        return trade.exit_price > worst + (trade.mae * 0.3)
    worst = trade.entry_price + trade.mae
    return trade.exit_price < worst - (trade.mae * 0.3)
