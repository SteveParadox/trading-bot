"""Example optimization workflow."""

from __future__ import annotations

from backtester import BacktestConfig
from backtester.optimization import OptimizationRunner


def main() -> None:
    config = BacktestConfig.from_json("configs/backtest_config.json")
    runner = OptimizationRunner(config)

    grid = {
        "strategy.min_risk_reward": [1.2, 1.45, 1.8],
        "risk.risk_per_trade_pct": [0.0025, 0.005, 0.0075],
        "execution.market_slippage_bps": [2.0, 4.0, 6.0]
    }
    results = runner.grid_search(grid)
    for result in results[:10]:
        print(result.objective_value, result.params, result.metrics["profitability"])

    walk_forward = runner.walk_forward(
        {
            "strategy.min_risk_reward": [1.2, 1.45, 1.8],
            "strategy.stop_mode": ["ma", "atr"]
        }
    )
    print("Walk-forward segments:", len(walk_forward))


if __name__ == "__main__":
    main()
