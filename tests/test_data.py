from __future__ import annotations

import unittest

import pandas as pd

from backtester.config import DataConfig
from backtester.data import DataPortal, detect_gaps, normalize_ohlcv, resample_ohlcv


class DataEngineTests(unittest.TestCase):
    def test_normalize_resample_and_gap_detection(self) -> None:
        raw = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=12, freq="5min", tz="UTC"),
                "open": range(100, 112),
                "high": range(101, 113),
                "low": range(99, 111),
                "close": range(100, 112),
                "volume": [10.0] * 12,
            }
        )
        config = DataConfig(symbols=["BTCUSDT"], base_timeframe="5m", timeframes=["5m", "1h"])
        frame = normalize_ohlcv(raw, config)
        hourly = resample_ohlcv(frame, "1h")

        self.assertEqual(len(hourly), 1)
        self.assertEqual(float(hourly.iloc[0]["open"]), 100.0)
        self.assertEqual(float(hourly.iloc[0]["high"]), 112.0)
        self.assertEqual(float(hourly.iloc[0]["volume"]), 120.0)

        gappy = frame.drop(frame.index[3])
        report = detect_gaps(gappy, "BTCUSDT", "5m")
        self.assertTrue(report.has_gaps)
        self.assertEqual(report.missing_candles, 1)

    def test_history_uses_only_closed_higher_timeframe_bars(self) -> None:
        config = DataConfig(symbols=["BTCUSDT"], base_timeframe="5m", timeframes=["5m", "1h"])
        portal = DataPortal(config)
        index = pd.date_range("2025-01-01", periods=24, freq="5min", tz="UTC")
        frame = pd.DataFrame(
            {
                "open": [100.0] * 24,
                "high": [101.0] * 24,
                "low": [99.0] * 24,
                "close": [100.0] * 24,
                "volume": [1.0] * 24,
            },
            index=index,
        )
        portal.set_frame("BTCUSDT", "5m", frame)
        portal.set_frame("BTCUSDT", "1h", resample_ohlcv(frame, "1h"))

        history = portal.history("BTCUSDT", "1h", pd.Timestamp("2025-01-01 00:55:00Z"))
        self.assertEqual(len(history), 0)
        history = portal.history("BTCUSDT", "1h", pd.Timestamp("2025-01-01 01:00:00Z"))
        self.assertEqual(len(history), 1)

    def test_set_frame_rejects_gaps_above_configured_tolerance(self) -> None:
        config = DataConfig(
            symbols=["BTCUSDT"],
            base_timeframe="5m",
            timeframes=["5m"],
            max_gap_candles=0,
        )
        portal = DataPortal(config)
        index = pd.to_datetime(
            [
                "2025-01-01T00:00:00Z",
                "2025-01-01T00:10:00Z",
            ]
        )
        frame = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.0, 101.0],
                "volume": [1.0, 1.0],
            },
            index=index,
        )

        with self.assertRaises(ValueError):
            portal.set_frame("BTCUSDT", "5m", frame)


if __name__ == "__main__":
    unittest.main()
