from __future__ import annotations

import unittest

import pandas as pd

from backtester.config import BacktestConfig
from backtester.execution import SimulatedExchange
from backtester.models import OrderSide, OrderType, Side


def make_test_config() -> BacktestConfig:
    return BacktestConfig.from_dict(
        {
            "data": {"symbols": ["BTCUSDT"], "base_timeframe": "5m", "timeframes": ["5m"]},
            "execution": {
                "taker_fee_rate": 0.0,
                "maker_fee_rate": 0.0,
                "spread_bps": 0.0,
                "market_slippage_bps": 0.0,
                "stop_slippage_bps": 0.0,
                "limit_slippage_bps": 0.0,
                "market_latency_candles": 1,
                "resting_order_latency_candles": 0,
                "limit_fill_probability": 1.0,
                "partial_fill_probability": 0.0,
                "conservative_intrabar_priority": True,
            },
            "risk": {"initial_equity": 1000.0, "leverage": 2.0},
            "instruments": {
                "BTCUSDT": {
                    "tick_size": 0.1,
                    "qty_step": 0.001,
                    "min_qty": 0.001,
                    "min_notional": 1.0,
                    "max_leverage": 50.0,
                }
            },
        }
    )


def candle(open_: float, high: float, low: float, close: float) -> pd.Series:
    return pd.Series(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1.0,
        }
    )


class ExecutionEngineTests(unittest.TestCase):
    def test_market_order_executes_on_next_candle(self) -> None:
        exchange = SimulatedExchange(make_test_config())
        exchange.submit_market_entry(
            symbol="BTCUSDT",
            side=Side.LONG,
            qty=1.0,
            timestamp=pd.Timestamp("2025-01-01T00:00:00Z"),
            current_index=0,
            leverage=2.0,
            metadata={},
        )
        self.assertEqual(
            exchange.process_candle(
                symbol="BTCUSDT",
                timestamp=pd.Timestamp("2025-01-01T00:00:00Z"),
                candle=candle(100.0, 101.0, 99.0, 100.0),
                bar_index=0,
            ),
            [],
        )
        fills = exchange.process_candle(
            symbol="BTCUSDT",
            timestamp=pd.Timestamp("2025-01-01T00:05:00Z"),
            candle=candle(105.0, 106.0, 104.0, 105.0),
            bar_index=1,
        )

        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0].price, 105.0)
        self.assertTrue(exchange.has_position("BTCUSDT", Side.LONG))

    def test_reduce_only_order_cannot_flip_position(self) -> None:
        exchange = SimulatedExchange(make_test_config())
        exchange.submit_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            qty=1.0,
            timestamp=pd.Timestamp("2025-01-01T00:00:00Z"),
            current_index=-1,
            leverage=2.0,
        )
        flat_candle = candle(100.0, 101.0, 99.0, 100.0)
        exchange.process_candle(
            symbol="BTCUSDT",
            timestamp=pd.Timestamp("2025-01-01T00:00:00Z"),
            candle=flat_candle,
            bar_index=0,
        )
        exchange.submit_order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            qty=2.0,
            timestamp=pd.Timestamp("2025-01-01T00:05:00Z"),
            current_index=0,
            reduce_only=True,
        )
        exchange.process_candle(
            symbol="BTCUSDT",
            timestamp=pd.Timestamp("2025-01-01T00:05:00Z"),
            candle=flat_candle,
            bar_index=1,
        )

        self.assertFalse(exchange.has_position("BTCUSDT"))
        self.assertEqual(exchange.open_position_count(), 0)

    def test_stop_fills_before_target_when_both_touched(self) -> None:
        exchange = SimulatedExchange(make_test_config())
        exchange.submit_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            qty=1.0,
            timestamp=pd.Timestamp("2025-01-01T00:00:00Z"),
            current_index=-1,
            leverage=2.0,
            metadata={"risk_amount": 10.0},
        )
        exchange.process_candle(
            symbol="BTCUSDT",
            timestamp=pd.Timestamp("2025-01-01T00:00:00Z"),
            candle=candle(100.0, 101.0, 99.0, 100.0),
            bar_index=0,
        )
        exchange.submit_order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP_MARKET,
            qty=1.0,
            trigger_price=95.0,
            timestamp=pd.Timestamp("2025-01-01T00:00:00Z"),
            current_index=0,
            reduce_only=True,
            metadata={"exit_reason": "stop_loss"},
        )
        exchange.submit_order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            qty=1.0,
            price=110.0,
            timestamp=pd.Timestamp("2025-01-01T00:00:00Z"),
            current_index=0,
            reduce_only=True,
            metadata={"exit_reason": "take_profit"},
        )
        exchange.process_candle(
            symbol="BTCUSDT",
            timestamp=pd.Timestamp("2025-01-01T00:05:00Z"),
            candle=candle(100.0, 111.0, 94.0, 100.0),
            bar_index=1,
        )

        self.assertEqual(len(exchange.trades), 1)
        self.assertEqual(exchange.trades[0].exit_reason, "stop_loss")

    def test_realized_pnl_includes_allocated_entry_and_exit_fees(self) -> None:
        raw = make_test_config().to_dict()
        raw["execution"]["taker_fee_rate"] = 0.01
        exchange = SimulatedExchange(BacktestConfig.from_dict(raw))
        exchange.submit_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            qty=1.0,
            timestamp=pd.Timestamp("2025-01-01T00:00:00Z"),
            current_index=-1,
            leverage=2.0,
        )
        exchange.process_candle(
            symbol="BTCUSDT",
            timestamp=pd.Timestamp("2025-01-01T00:00:00Z"),
            candle=candle(100.0, 101.0, 99.0, 100.0),
            bar_index=0,
        )
        exchange.submit_order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            qty=1.0,
            timestamp=pd.Timestamp("2025-01-01T00:05:00Z"),
            current_index=0,
            reduce_only=True,
        )
        fills = exchange.process_candle(
            symbol="BTCUSDT",
            timestamp=pd.Timestamp("2025-01-01T00:05:00Z"),
            candle=candle(110.0, 111.0, 109.0, 110.0),
            bar_index=1,
        )

        self.assertEqual(len(exchange.trades), 1)
        self.assertAlmostEqual(exchange.trades[0].net_pnl, 7.9)
        self.assertAlmostEqual(exchange.realized_pnl, 7.9)
        self.assertAlmostEqual(exchange.cash, 1007.9)
        self.assertAlmostEqual(fills[0].realized_pnl, 7.9)


if __name__ == "__main__":
    unittest.main()
