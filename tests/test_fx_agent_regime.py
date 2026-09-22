from __future__ import annotations

import pytest

from forex_agent.data.schemas import MarketRegime, TradeRecord
from forex_agent.analysis.regime import (
    classify_trend,
    classify_volatility,
    detect_regime,
    performance_by_regime,
)


def make_trade(**overrides) -> TradeRecord:
    defaults = dict(
        trade_id="T001",
        symbol="EURUSD",
        direction="LONG",
        entry_price=1.1000,
        exit_price=1.1050,
        stop_loss=1.0950,
        take_profit=1.1100,
        position_size=1.0,
        account_balance=10000.0,
        risk_amount=50.0,
    )
    defaults.update(overrides)
    return TradeRecord(**defaults)


class TestClassifyVolatility:
    def test_high_volatility(self):
        prices = [1.1000 + i * 0.01 * ((-1) ** i) for i in range(30)]
        result = classify_volatility(prices, window=20)
        assert result == "high"

    def test_low_volatility(self):
        prices = [1.1000 + i * 0.00001 for i in range(30)]
        result = classify_volatility(prices, window=20)
        assert result == "low"

    def test_normal_volatility(self):
        prices = [1.1000 + 0.008 * ((-1) ** i) for i in range(30)]
        result = classify_volatility(prices, window=20)
        assert result == "normal"

    def test_insufficient_data(self):
        prices = [1.1000, 1.1001, 1.1002]
        result = classify_volatility(prices, window=20)
        assert result == "unknown"

    def test_empty_prices(self):
        result = classify_volatility([], window=20)
        assert result == "unknown"


class TestClassifyTrend:
    def test_uptrend(self):
        prices = [1.1000 + i * 0.001 for i in range(60)]
        result = classify_trend(prices, short_window=10, long_window=50)
        assert result == "up"

    def test_downtrend(self):
        prices = [1.1200 - i * 0.001 for i in range(60)]
        result = classify_trend(prices, short_window=10, long_window=50)
        assert result == "down"

    def test_sideways(self):
        prices = [1.1000 + 0.0001 * ((-1) ** i) for i in range(60)]
        result = classify_trend(prices, short_window=10, long_window=50)
        assert result == "sideways"

    def test_insufficient_data(self):
        prices = [1.1000] * 10
        result = classify_trend(prices, short_window=10, long_window=50)
        assert result == "unknown"

    def test_exact_boundary_uptrend(self):
        prices = [1.1000 + i * 0.002 for i in range(60)]
        result = classify_trend(prices, short_window=10, long_window=50)
        assert result == "up"


class TestDetectRegime:
    def test_trending_up(self):
        prices = [1.1000 + i * 0.001 for i in range(60)]
        regime = detect_regime(prices, short_window=10, long_window=50)
        assert regime == MarketRegime.TRENDING_UP

    def test_trending_down(self):
        prices = [1.1200 - i * 0.001 for i in range(60)]
        regime = detect_regime(prices, short_window=10, long_window=50)
        assert regime == MarketRegime.TRENDING_DOWN

    def test_ranging(self):
        prices = [1.1000 + 0.008 * ((-1) ** i) for i in range(60)]
        regime = detect_regime(prices, short_window=10, long_window=50)
        assert regime == MarketRegime.RANGING

    def test_high_volatility(self):
        prices = [1.1000 + 0.02 * ((-1) ** i) for i in range(60)]
        regime = detect_regime(prices, short_window=10, long_window=50)
        assert regime == MarketRegime.HIGH_VOLATILITY

    def test_low_volatility(self):
        prices = [1.1000 + i * 0.00001 for i in range(60)]
        regime = detect_regime(prices, short_window=10, long_window=50)
        assert regime == MarketRegime.LOW_VOLATILITY

    def test_unknown_short_data(self):
        prices = [1.1000, 1.1001, 1.1002]
        regime = detect_regime(prices, short_window=10, long_window=50)
        assert regime == MarketRegime.UNKNOWN


class TestPerformanceByRegime:
    def test_mixed_regimes(self):
        trades = [
            make_trade(trade_id="T1", regime=MarketRegime.TRENDING_UP, entry_price=1.1000, exit_price=1.1050),
            make_trade(trade_id="T2", regime=MarketRegime.TRENDING_UP, entry_price=1.1000, exit_price=1.1050),
            make_trade(trade_id="T3", regime=MarketRegime.RANGING, entry_price=1.1000, exit_price=1.0950),
            make_trade(trade_id="T4", regime=MarketRegime.HIGH_VOLATILITY, entry_price=1.1000, exit_price=1.0950),
        ]
        results = performance_by_regime(trades)
        assert "trending_up" in results
        assert "ranging" in results
        assert "high_volatility" in results
        assert results["trending_up"]["total_trades"] == 2
        assert results["trending_up"]["win_rate"] == pytest.approx(1.0)
        assert results["ranging"]["win_rate"] == pytest.approx(0.0)

    def test_all_same_regime(self):
        trades = [
            make_trade(trade_id=f"T{i}", regime=MarketRegime.TRENDING_UP,
                       entry_price=1.1000, exit_price=1.1050)
            for i in range(5)
        ]
        results = performance_by_regime(trades)
        assert len(results) == 1
        assert results["trending_up"]["total_trades"] == 5

    def test_no_regime(self):
        trades = [
            make_trade(trade_id="T1", regime=None, entry_price=1.1000, exit_price=1.1050),
        ]
        results = performance_by_regime(trades)
        assert "unknown" in results

    def test_empty_trades(self):
        results = performance_by_regime([])
        assert results == {}
