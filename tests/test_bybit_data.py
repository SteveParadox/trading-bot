from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

import pandas as pd

from backtester.bybit_data import (
    BybitCandleStore,
    BybitLiveCandleRecorder,
    bybit_interval,
)
from backtester.config import DataConfig
from backtester.data import DataPortal


class FakeKlineClient:
    def __init__(self, rows: list[list[Any]]) -> None:
        self.rows = rows
        self.calls: list[dict[str, Any]] = []

    def get_kline(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        start = int(kwargs["start"])
        end = int(kwargs["end"])
        limit = int(kwargs["limit"])
        selected = [row for row in self.rows if start <= int(row[0]) <= end]
        # Bybit returns newest-first rows. Return the newest page in the range.
        page = list(reversed(selected[-limit:]))
        return {"retCode": 0, "result": {"list": page}}


def row(timestamp: str, open_: float) -> list[str]:
    start_ms = int(pd.Timestamp(timestamp, tz="UTC").timestamp() * 1000)
    high = open_ + 1.0
    low = open_ - 1.0
    close = open_ + 0.25
    return [
        str(start_ms),
        str(open_),
        str(high),
        str(low),
        str(close),
        "10",
        str(close * 10),
    ]


def rows(count: int = 5) -> list[list[str]]:
    index = pd.date_range("2025-01-01", periods=count, freq="5min", tz="UTC")
    return [row(str(timestamp), 100.0 + position) for position, timestamp in enumerate(index)]


class BybitDataTests(unittest.TestCase):
    def test_bybit_interval_conversion(self) -> None:
        self.assertEqual(bybit_interval("5m"), "5")
        self.assertEqual(bybit_interval("1h"), "60")
        self.assertEqual(bybit_interval("4h"), "240")
        self.assertEqual(bybit_interval("1d"), "D")

    def test_backfill_saves_paginated_bybit_rows_for_data_portal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = DataConfig(
                data_path=temp_dir,
                symbols=["BTCUSDT"],
                base_timeframe="5m",
                timeframes=["5m"],
                provider="bybit",
            )
            store = BybitCandleStore(
                config,
                client=FakeKlineClient(rows(5)),
                request_limit=2,
                request_delay_seconds=0,
            )

            report = store.backfill(
                "BTCUSDT",
                "5m",
                "2025-01-01T00:00:00Z",
                "2025-01-01T00:20:00Z",
                replace=True,
            )

            self.assertEqual(report.rows_downloaded, 5)
            self.assertEqual(report.rows_after, 5)
            portal = DataPortal.from_config(config)
            self.assertEqual(len(portal.get_frame("BTCUSDT", "5m")), 5)

    def test_incremental_update_appends_from_last_local_candle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = DataConfig(
                data_path=temp_dir,
                symbols=["BTCUSDT"],
                base_timeframe="5m",
                timeframes=["5m"],
                provider="bybit",
            )
            all_rows = rows(4)
            store = BybitCandleStore(
                config,
                client=FakeKlineClient(all_rows),
                request_limit=2,
                request_delay_seconds=0,
            )
            store.backfill(
                "BTCUSDT",
                "5m",
                "2025-01-01T00:00:00Z",
                "2025-01-01T00:05:00Z",
                replace=True,
            )

            report = store.incremental_update(
                "BTCUSDT",
                "5m",
                end="2025-01-01T00:15:00Z",
            )

            self.assertIsNotNone(report)
            self.assertEqual(report.rows_downloaded, 2)
            self.assertEqual(len(store.load_local("BTCUSDT", "5m")), 4)

    def test_repair_redownloads_gap_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = DataConfig(
                data_path=temp_dir,
                symbols=["BTCUSDT"],
                base_timeframe="5m",
                timeframes=["5m"],
                provider="bybit",
                max_gap_candles=10,
            )
            all_rows = rows(3)
            store = BybitCandleStore(
                config,
                client=FakeKlineClient(all_rows),
                request_limit=2,
                request_delay_seconds=0,
            )
            path = Path(temp_dir) / "BTCUSDT_5m.csv"
            pd.DataFrame([all_rows[0], all_rows[2]], columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "turnover",
            ]).to_csv(path, index=False)

            issues = store.detect_local_issues("BTCUSDT", "5m")
            self.assertEqual(len([issue for issue in issues if issue.issue_type == "gap"]), 1)

            report = store.repair("BTCUSDT", "5m", padding_candles=0)

            self.assertEqual(len(report.issues), 1)
            self.assertEqual(len(store.load_local("BTCUSDT", "5m")), 3)

    def test_websocket_recorder_writes_confirmed_candles_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = DataConfig(
                data_path=temp_dir,
                symbols=["BTCUSDT"],
                base_timeframe="5m",
                timeframes=["5m"],
                provider="bybit",
            )
            store = BybitCandleStore(
                config,
                client=FakeKlineClient([]),
                request_delay_seconds=0,
            )
            recorder = BybitLiveCandleRecorder(store)
            recorder.handle_message(
                {
                    "topic": "kline.5.BTCUSDT",
                    "data": [
                        {
                            "start": "1735689600000",
                            "open": "100",
                            "high": "101",
                            "low": "99",
                            "close": "100.5",
                            "volume": "10",
                            "turnover": "1005",
                            "confirm": False,
                        },
                        {
                            "start": "1735689900000",
                            "open": "101",
                            "high": "102",
                            "low": "100",
                            "close": "101.5",
                            "volume": "11",
                            "turnover": "1116.5",
                            "confirm": True,
                        },
                    ],
                }
            )

            frame = store.load_local("BTCUSDT", "5m")
            self.assertEqual(len(frame), 1)
            self.assertAlmostEqual(float(frame.iloc[0]["close"]), 101.5)


if __name__ == "__main__":
    unittest.main()
