"""Parameter optimization and robustness tooling."""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from backtester.config import BacktestConfig
from backtester.data import DataPortal
from backtester.engine import BacktestEngine, BacktestResult


@dataclass(frozen=True)
class OptimizationResult:
    params: dict[str, Any]
    objective_value: float
    metrics: dict[str, Any]
    result: BacktestResult | None = None


class OptimizationRunner:
    """Run grid/random/walk-forward searches without leaking train into test."""

    def __init__(
        self,
        base_config: BacktestConfig,
        data_factory: Callable[[BacktestConfig], DataPortal] | None = None,
        objective: str | None = None,
    ) -> None:
        self.base_config = base_config
        self.data_factory = data_factory or (lambda cfg: DataPortal.from_config(cfg.data))
        self.objective = objective or base_config.optimization.objective

    def grid_search(
        self,
        param_grid: dict[str, Iterable[Any]],
        *,
        keep_results: bool = False,
    ) -> list[OptimizationResult]:
        results: list[OptimizationResult] = []
        keys = list(param_grid)
        for values in itertools.product(*(list(param_grid[key]) for key in keys)):
            params = dict(zip(keys, values))
            results.append(self._run_params(params, keep_result=keep_results))
        return sorted(results, key=lambda item: item.objective_value, reverse=True)

    def random_search(
        self,
        search_space: dict[str, Iterable[Any] | tuple[float, float]],
        iterations: int | None = None,
        *,
        keep_results: bool = False,
    ) -> list[OptimizationResult]:
        rng = random.Random(self.base_config.optimization.random_seed)
        iterations = iterations or self.base_config.optimization.random_iterations
        results: list[OptimizationResult] = []
        for _ in range(iterations):
            params = {}
            for key, values in search_space.items():
                if isinstance(values, tuple) and len(values) == 2:
                    params[key] = rng.uniform(float(values[0]), float(values[1]))
                else:
                    params[key] = rng.choice(list(values))
            results.append(self._run_params(params, keep_result=keep_results))
        return sorted(results, key=lambda item: item.objective_value, reverse=True)

    def walk_forward(
        self,
        param_grid: dict[str, Iterable[Any]],
        *,
        train_bars: int | None = None,
        test_bars: int | None = None,
    ) -> list[dict[str, Any]]:
        train_bars = train_bars or self.base_config.optimization.walk_forward_train_bars
        test_bars = test_bars or self.base_config.optimization.walk_forward_test_bars
        data = self.data_factory(self.base_config)
        segments = data.walk_forward_segments(train_bars, test_bars)
        segment_reports: list[dict[str, Any]] = []

        for segment in segments:
            train_config = config_with_params(
                self.base_config,
                {
                    "data.start": segment.train_start.isoformat(),
                    "data.end": segment.train_end.isoformat(),
                },
            )
            train_runner = OptimizationRunner(train_config, self.data_factory, self.objective)
            train_results = train_runner.grid_search(param_grid)
            best = train_results[0]

            test_config = config_with_params(
                self.base_config,
                {
                    **best.params,
                    "data.start": segment.test_start.isoformat(),
                    "data.end": segment.test_end.isoformat(),
                },
            )
            test_result = self._run_config(test_config)
            segment_reports.append(
                {
                    "segment": segment.index,
                    "train_start": segment.train_start.isoformat(),
                    "train_end": segment.train_end.isoformat(),
                    "test_start": segment.test_start.isoformat(),
                    "test_end": segment.test_end.isoformat(),
                    "best_params": best.params,
                    "train_objective": best.objective_value,
                    "test_objective": objective_value(test_result.metrics, self.objective),
                    "test_metrics": test_result.metrics,
                }
            )
        return segment_reports

    def parameter_sensitivity(
        self,
        parameter: str,
        values: Iterable[Any],
    ) -> pd.DataFrame:
        rows = []
        for value in values:
            result = self._run_params({parameter: value}, keep_result=False)
            rows.append(
                {
                    "parameter": parameter,
                    "value": value,
                    "objective": result.objective_value,
                    "net_profit": result.metrics["profitability"]["net_profit"],
                    "max_drawdown": result.metrics["risk"]["max_drawdown"],
                    "trades": result.metrics["profitability"]["trade_count"],
                }
            )
        return pd.DataFrame(rows)

    def _run_params(self, params: dict[str, Any], *, keep_result: bool) -> OptimizationResult:
        config = config_with_params(self.base_config, params)
        result = self._run_config(config)
        return OptimizationResult(
            params=params,
            objective_value=objective_value(result.metrics, self.objective),
            metrics=result.metrics,
            result=result if keep_result else None,
        )

    def _run_config(self, config: BacktestConfig) -> BacktestResult:
        data = self.data_factory(config)
        engine = BacktestEngine(config=config, data=data)
        return engine.run()


def monte_carlo_trade_paths(
    trades: pd.DataFrame,
    *,
    initial_equity: float,
    iterations: int = 500,
    random_seed: int = 7,
) -> pd.DataFrame:
    """Bootstrap trade order to estimate path dependency and drawdown risk."""

    if trades.empty or "net_pnl" not in trades:
        return pd.DataFrame()
    rng = np.random.default_rng(random_seed)
    pnl = trades["net_pnl"].to_numpy(dtype="float64")
    rows = []
    for iteration in range(iterations):
        sample = rng.choice(pnl, size=len(pnl), replace=True)
        equity = initial_equity + np.cumsum(sample)
        peak = np.maximum.accumulate(equity)
        drawdown = np.where(peak > 0, (peak - equity) / peak, 0.0)
        rows.append(
            {
                "iteration": iteration,
                "final_equity": float(equity[-1]),
                "net_profit": float(equity[-1] - initial_equity),
                "max_drawdown": float(np.max(drawdown)),
                "min_equity": float(np.min(equity)),
            }
        )
    return pd.DataFrame(rows)


def config_with_params(config: BacktestConfig, params: dict[str, Any]) -> BacktestConfig:
    raw = config.to_dict()
    for path, value in params.items():
        target = raw
        parts = path.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    return BacktestConfig.from_dict(raw)


def objective_value(metrics: dict[str, Any], objective: str) -> float:
    objective = objective.lower()
    if objective == "net_profit":
        return float(metrics["profitability"]["net_profit"])
    if objective == "profit_factor":
        return float(metrics["profitability"]["profit_factor"])
    if objective == "sharpe":
        return float(metrics["risk"]["sharpe_ratio"])
    if objective == "sortino":
        return float(metrics["risk"]["sortino_ratio"])
    if objective == "calmar":
        return float(metrics["risk"]["calmar_ratio"])
    if objective == "expectancy":
        return float(metrics["profitability"]["expectancy"])
    raise ValueError(f"Unknown optimization objective: {objective}")
