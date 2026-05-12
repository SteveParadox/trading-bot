"""Example backtest run.

Expected input files by default:

    data/SAGAUSDT_5m.csv
    data/BUSDT_5m.csv

The config derives 1h/4h frames from 5m candles.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path so backtester module can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtester import BacktestConfig, BacktestEngine, DataPortal
from backtester.cli import setup_logging


def main() -> None:
    config = BacktestConfig.from_json("configs/backtest_config.json")
    setup_logging(config)
    data = DataPortal.from_config(config.data)
    result = BacktestEngine(config=config, data=data).run()

    print("Report written to:", config.analytics.output_dir)
    print("Profitability:", result.metrics["profitability"])
    print("Risk:", result.metrics["risk"])
    print("Execution:", result.metrics["execution"])


if __name__ == "__main__":
    main()
