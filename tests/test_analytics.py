from __future__ import annotations

import unittest

import pandas as pd

from backtester.analytics import PerformanceAnalyzer
from backtester.config import BacktestConfig
from backtester.models import PortfolioSnapshot, Side, TradeRecord


class AnalyticsTests(unittest.TestCase):
    def test_performance_metrics_include_profitability_and_risk(self) -> None:
        config = BacktestConfig.from_dict(
            {
                "data": {"base_timeframe": "5m", "symbols": ["BTCUSDT"], "timeframes": ["5m"]},
                "risk": {"initial_equity": 1000.0},
            }
        )
        snapshots = [
            PortfolioSnapshot(pd.Timestamp("2025-01-01T00:00:00Z"), 1000, 1000, 0, 0, 0, 0, 0, 0, 0),
            PortfolioSnapshot(pd.Timestamp("2025-01-01T00:05:00Z"), 1100, 1100, 0, 100, 0, 0, 0, 0, 0),
            PortfolioSnapshot(pd.Timestamp("2025-01-01T00:10:00Z"), 1050, 1050, 0, 50, 0, 0.04545, 0, 0, 0),
        ]
        trades = [
            TradeRecord(
                trade_id="1",
                symbol="BTCUSDT",
                side=Side.LONG,
                entry_time=pd.Timestamp("2025-01-01T00:00:00Z"),
                exit_time=pd.Timestamp("2025-01-01T00:05:00Z"),
                entry_price=100.0,
                exit_price=110.0,
                qty=1.0,
                gross_pnl=10.0,
                fees=1.0,
                slippage=0.5,
                net_pnl=9.0,
                risk_amount=5.0,
                exit_reason="take_profit",
                bars_held=1,
            ),
            TradeRecord(
                trade_id="2",
                symbol="BTCUSDT",
                side=Side.SHORT,
                entry_time=pd.Timestamp("2025-01-01T00:05:00Z"),
                exit_time=pd.Timestamp("2025-01-01T00:10:00Z"),
                entry_price=110.0,
                exit_price=115.0,
                qty=1.0,
                gross_pnl=-5.0,
                fees=1.0,
                slippage=0.5,
                net_pnl=-6.0,
                risk_amount=5.0,
                exit_reason="stop_loss",
                bars_held=1,
            ),
        ]

        metrics = PerformanceAnalyzer(config).calculate(
            snapshots=snapshots,
            trades=trades,
            fills=[],
            execution_stats={"fees_paid": 2.0, "slippage_paid": 1.0},
        )

        self.assertEqual(metrics["profitability"]["trade_count"], 2)
        self.assertAlmostEqual(metrics["profitability"]["win_rate"], 0.5)
        self.assertGreater(metrics["risk"]["max_drawdown"], 0)
        self.assertEqual(metrics["execution"]["fee_impact"], 2.0)


if __name__ == "__main__":
    unittest.main()
