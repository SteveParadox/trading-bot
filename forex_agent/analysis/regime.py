from __future__ import annotations

from forex_agent.data.schemas import MarketRegime, TradeRecord


def detect_regime(
    prices: list[float], short_window: int = 10, long_window: int = 50
) -> MarketRegime:
    if len(prices) < long_window:
        return MarketRegime.UNKNOWN

    trend = classify_trend(prices, short_window, long_window)
    vol_class = classify_volatility(prices, long_window)

    if vol_class == "high":
        return MarketRegime.HIGH_VOLATILITY
    elif vol_class == "low":
        return MarketRegime.LOW_VOLATILITY

    if trend == "up":
        return MarketRegime.TRENDING_UP
    elif trend == "down":
        return MarketRegime.TRENDING_DOWN

    return MarketRegime.RANGING


def classify_volatility(prices: list[float], window: int = 20) -> str:
    if len(prices) < window:
        return "unknown"
    subset = prices[-window:]
    vol = _compute_volatility(subset)
    if vol > 0.015:
        return "high"
    elif vol < 0.005:
        return "low"
    return "normal"


def classify_trend(
    prices: list[float], short_window: int = 10, long_window: int = 50
) -> str:
    if len(prices) < long_window:
        return "unknown"
    short_ma = sum(prices[-short_window:]) / short_window
    long_ma = sum(prices[-long_window:]) / long_window
    diff_pct = (short_ma - long_ma) / long_ma
    if diff_pct > 0.001:
        return "up"
    elif diff_pct < -0.001:
        return "down"
    return "sideways"


def performance_by_regime(trades: list[TradeRecord]) -> dict[str, dict]:
    by_regime: dict[str, list[TradeRecord]] = {}
    for t in trades:
        regime = t.regime.value if t.regime else "unknown"
        if regime not in by_regime:
            by_regime[regime] = []
        by_regime[regime].append(t)

    results = {}
    for regime, reg_trades in by_regime.items():
        closed = [t for t in reg_trades if t.exit_price is not None]
        wins = [t for t in closed if t.is_winner]
        total_pnl = sum(t.pnl for t in closed)
        results[regime] = {
            "total_trades": len(closed),
            "win_rate": len(wins) / len(closed) if closed else 0.0,
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / len(closed) if closed else 0.0,
        }
    return results


def _compute_volatility(prices: list[float]) -> float:
    if len(prices) < 2:
        return 0.0
    mean = sum(prices) / len(prices)
    variance = sum((p - mean) ** 2 for p in prices) / len(prices)
    return variance ** 0.5 / mean if mean != 0 else 0.0
