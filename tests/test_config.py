from __future__ import annotations

import unittest

from backtester.config import BacktestConfig, DataConfig


class ConfigTests(unittest.TestCase):
    def test_data_config_normalizes_timeframes_and_resample_source(self) -> None:
        config = DataConfig(
            symbols=["btcusdt"],
            base_timeframe="5",
            timeframes=["60"],
            resample_from="5",
        )

        self.assertEqual(config.symbols, ["BTCUSDT"])
        self.assertEqual(config.base_timeframe, "5m")
        self.assertEqual(config.resample_from, "5m")
        self.assertIn("1h", config.timeframes)

    def test_config_rejects_invalid_execution_probability(self) -> None:
        with self.assertRaises(ValueError):
            BacktestConfig.from_dict(
                {
                    "execution": {
                        "limit_fill_probability": 1.5,
                    }
                }
            )

    def test_config_rejects_invalid_strategy_bounds(self) -> None:
        with self.assertRaises(ValueError):
            BacktestConfig.from_dict(
                {
                    "strategy": {
                        "min_atr_pct": 0.10,
                        "max_atr_pct": 0.05,
                    }
                }
            )


if __name__ == "__main__":
    unittest.main()
