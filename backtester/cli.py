"""Command-line entry point for running backtests."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from backtester.config import BacktestConfig
from backtester.data import DataPortal
from backtester.engine import BacktestEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Bybit USDT perpetual futures backtest.")
    parser.add_argument("--config", default="configs/backtest_config.json", help="Path to JSON backtest config")
    parser.add_argument("--no-export", action="store_true", help="Disable report export for this run")
    args = parser.parse_args()

    config = BacktestConfig.from_json(args.config)
    setup_logging(config)
    if args.no_export:
        raw = config.to_dict()
        raw["analytics"]["export_json"] = False
        raw["analytics"]["export_csv"] = False
        config = BacktestConfig.from_dict(raw)

    data = DataPortal.from_config(config.data)
    result = BacktestEngine(config=config, data=data).run()
    profitability = result.metrics["profitability"]
    risk = result.metrics["risk"]
    print(
        "Backtest complete | "
        f"net={profitability['net_profit']:.2f} USDT | "
        f"return={profitability['net_return_pct']:.2%} | "
        f"trades={profitability['trade_count']} | "
        f"max_dd={risk['max_drawdown']:.2%} | "
        f"sharpe={risk['sharpe_ratio']:.2f}"
    )


def setup_logging(config: BacktestConfig) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if config.logging.file_path:
        path = Path(config.logging.file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, config.logging.level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
        handlers=handlers,
        force=True,
    )


if __name__ == "__main__":
    main()
