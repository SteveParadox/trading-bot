from __future__ import annotations

import unittest

import pandas as pd

from backtester.config import BacktestConfig
from backtester.execution import SimulatedExchange
from backtester.models import OrderSide, OrderType, Side


def test_config() -> BacktestConfig:
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


class ExecutionEngineTests(unittest.TestCase):
    def test_market_order_executes_on_next_candle(self) -> None:
        exchange = SimulatedExchange(test_config())
        exchange.submit_market_entry(
            symbol="BTCUSDT",
            side=Side.LONG,
            qty=1.0,
            timestamp=pd.Timestamp("2025-01-01T00:00:00Z"),
            current_index=0,
            leverage=2.0,
            metadata={},
        )
        candle0 = pd.Series({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0})
        self.assertEqual(exchange.process_candle(symbol="BTCUSDT", timestamp=pd.Timestamp("2025-01-01T00:00:00Z"), candle=candle0, bar_index=0), [])
        candle1 = pd.Series({"open": 105.0, "high": 106.0, "low": 104.0, "close": 105.0, "volume": 1.0})
        fills = exchange.process_candle(symbol="BTCUSDT", timestamp=pd.Timestamp("2025-01-01T00:05:00Z"), candle=candle1, bar_index=1)

        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0].price, 105.0)
        self.assertTrue(exchange.has_position("BTCUSDT", Side.LONG))

    def test_reduce_only_order_cannot_flip_position(self) -> None:
        exchange = SimulatedExchange(test_config())
        exchange.submit_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            qty=1.0,
            timestamp=pd.Timestamp("2025-01-01T00:00:00Z"),
            current_index=-1,
            leverage=2.0,
        )
        candle = pd.Series({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0})
        exchange.process_candle(symbol="BTCUSDT", timestamp=pd.Timestamp("2025-01-01T00:00:00Z"), candle=candle, bar_index=0)
        exchange.submit_order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            qty=2.0,
            timestamp=pd.Timestamp("2025-01-01T00:05:00Z"),
            current_index=0,
            reduce_only=True,
        )
        exchange.process_candle(symbol="BTCUSDT", timestamp=pd.Timestamp("2025-01-01T00:05:00Z"), candle=candle, bar_index=1)

        self.assertFalse(exchange.has_position("BTCUSDT"))
        self.assertEqual(exchange.open_position_count(), 0)

    def test_stop_fills_before_target_when_both_touched(self) -> None:
        exchange = SimulatedExchange(test_config())
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
        entry_candle = pd.Series({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0})
        exchange.process_candle(symbol="BTCUSDT", timestamp=pd.Timestamp("2025-01-01T00:00:00Z"), candle=entry_candle, bar_index=0)
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
        wide_candle = pd.Series({"open": 100.0, "high": 111.0, "low": 94.0, "close": 100.0, "volume": 1.0})
        exchange.process_candle(symbol="BTCUSDT", timestamp=pd.Timestamp("2025-01-01T00:05:00Z"), candle=wide_candle, bar_index=1)

        self.assertEqual(len(exchange.trades), 1)
        self.assertEqual(exchange.trades[0].exit_reason, "stop_loss")


if __name__ == "__main__":
    unittest.main()
