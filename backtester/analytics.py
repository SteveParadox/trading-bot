"""Performance, risk, trade, and execution analytics."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtester.config import BacktestConfig
from backtester.models import Fill, PortfolioSnapshot, TradeRecord, annualization_factor


class PerformanceAnalyzer:
    def __init__(self, config: BacktestConfig) -> None:
        self.config = config

    def calculate(
        self,
        *,
        snapshots: list[PortfolioSnapshot],
        trades: list[TradeRecord],
        fills: list[Fill],
        execution_stats: dict[str, float],
    ) -> dict[str, Any]:
        equity = snapshots_to_frame(snapshots)
        trade_frame = records_to_frame(trades)
        fill_frame = records_to_frame(fills)

        profitability = self._profitability(equity, trade_frame)
        risk = self._risk(equity)
        trade_analytics = self._trade_analytics(trade_frame)
        execution = self._execution_analytics(fill_frame, execution_stats, profitability)
        return {
            "profitability": profitability,
            "risk": risk,
            "trades": trade_analytics,
            "execution": execution,
            "portfolio": {
                "initial_equity": self.config.risk.initial_equity,
                "final_equity": (
                    float(equity["equity"].iloc[-1])
                    if not equity.empty
                    else self.config.risk.initial_equity
                ),
                "open_positions": int(equity["open_positions"].iloc[-1]) if not equity.empty else 0,
                "max_margin_used": float(equity["margin_used"].max()) if not equity.empty else 0.0,
                "max_gross_exposure": float(equity["gross_exposure"].max()) if not equity.empty else 0.0,
            },
        }

    def equity_curve(self, snapshots: list[PortfolioSnapshot]) -> pd.DataFrame:
        return snapshots_to_frame(snapshots)

    def monthly_returns(self, snapshots: list[PortfolioSnapshot]) -> pd.Series:
        equity = snapshots_to_frame(snapshots)
        if equity.empty:
            return pd.Series(dtype="float64")
        monthly = equity["equity"].resample("ME").last().pct_change().dropna()
        return monthly

    def export(
        self,
        *,
        output_dir: str | Path,
        snapshots: list[PortfolioSnapshot],
        trades: list[TradeRecord],
        fills: list[Fill],
        metrics: dict[str, Any],
        execution_stats: dict[str, float],
    ) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        equity = snapshots_to_frame(snapshots)
        trade_frame = records_to_frame(trades)
        fill_frame = records_to_frame(fills)
        if self.config.analytics.export_csv:
            equity.to_csv(output / "equity_curve.csv")
            trade_frame.to_csv(output / "trades.csv", index=False)
            fill_frame.to_csv(output / "fills.csv", index=False)
            self.monthly_returns(snapshots).to_csv(output / "monthly_returns.csv", header=["return"])

        if self.config.analytics.export_json:
            report = {
                "config": self.config.to_dict(),
                "metrics": metrics,
                "execution_stats": execution_stats,
            }
            (output / "report.json").write_text(
                json.dumps(to_jsonable(report), indent=2),
                encoding="utf-8",
            )

    def _profitability(self, equity: pd.DataFrame, trades: pd.DataFrame) -> dict[str, Any]:
        initial = self.config.risk.initial_equity
        final = float(equity["equity"].iloc[-1]) if not equity.empty else initial
        net_profit = final - initial
        if trades.empty:
            return {
                "net_profit": net_profit,
                "net_return_pct": net_profit / initial if initial else 0.0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0,
                "average_r_multiple": 0.0,
                "trade_count": 0,
            }
        wins = trades[trades["net_pnl"] > 0]
        losses = trades[trades["net_pnl"] < 0]
        gross_profit = float(wins["net_pnl"].sum())
        gross_loss = float(losses["net_pnl"].sum())
        profit_factor = gross_profit / abs(gross_loss) if gross_loss < 0 else math.inf if gross_profit > 0 else 0.0
        return {
            "net_profit": net_profit,
            "net_return_pct": net_profit / initial if initial else 0.0,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "win_rate": float(len(wins) / len(trades)) if len(trades) else 0.0,
            "profit_factor": profit_factor,
            "expectancy": float(trades["net_pnl"].mean()),
            "average_r_multiple": float(trades["r_multiple"].mean()) if "r_multiple" in trades else 0.0,
            "trade_count": int(len(trades)),
        }

    def _risk(self, equity: pd.DataFrame) -> dict[str, Any]:
        if equity.empty:
            return {
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "calmar_ratio": 0.0,
                "volatility": 0.0,
                "equity_curve_stability": 0.0,
            }
        returns = equity["equity"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        periods = annualization_factor(self.config.data.base_timeframe)
        risk_free_per_period = self.config.analytics.risk_free_rate / periods
        excess = returns - risk_free_per_period
        sharpe = safe_ratio(float(excess.mean()), float(excess.std(ddof=1))) * math.sqrt(periods)
        downside = excess[excess < 0]
        sortino = safe_ratio(float(excess.mean()), float(downside.std(ddof=1))) * math.sqrt(periods)
        max_dd = float(equity["drawdown_pct"].max())
        total_return = (float(equity["equity"].iloc[-1]) / self.config.risk.initial_equity) - 1.0
        calmar = safe_ratio(total_return, max_dd)
        volatility = float(returns.std(ddof=1) * math.sqrt(periods)) if len(returns) > 1 else 0.0
        return {
            "max_drawdown": max_dd,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "volatility": volatility,
            "equity_curve_stability": equity_curve_stability(equity["equity"]),
        }

    def _trade_analytics(self, trades: pd.DataFrame) -> dict[str, Any]:
        if trades.empty:
            return {
                "average_trade_duration_bars": 0.0,
                "best_trade": 0.0,
                "worst_trade": 0.0,
                "consecutive_wins": 0,
                "consecutive_losses": 0,
                "long_performance": {},
                "short_performance": {},
                "symbol_performance": {},
                "timeframe_performance": {},
            }

        return {
            "average_trade_duration_bars": float(trades["bars_held"].mean()),
            "best_trade": float(trades["net_pnl"].max()),
            "worst_trade": float(trades["net_pnl"].min()),
            "consecutive_wins": max_streak(trades["net_pnl"] > 0),
            "consecutive_losses": max_streak(trades["net_pnl"] < 0),
            "long_performance": grouped_trade_stats(trades[trades["side"] == "LONG"]),
            "short_performance": grouped_trade_stats(trades[trades["side"] == "SHORT"]),
            "symbol_performance": {
                str(symbol): grouped_trade_stats(group)
                for symbol, group in trades.groupby("symbol")
            },
            "timeframe_performance": {
                self.config.strategy.entry_timeframe: grouped_trade_stats(trades)
            },
        }

    def _execution_analytics(
        self,
        fills: pd.DataFrame,
        execution_stats: dict[str, float],
        profitability: dict[str, Any],
    ) -> dict[str, Any]:
        fees = float(execution_stats.get("fees_paid") or 0.0)
        slippage = float(execution_stats.get("slippage_paid") or 0.0)
        gross_abs = abs(float(profitability.get("gross_profit") or 0.0)) + abs(
            float(profitability.get("gross_loss") or 0.0)
        )
        return {
            "fee_impact": fees,
            "fee_impact_pct_of_gross_pnl": safe_ratio(fees, gross_abs),
            "slippage_impact": slippage,
            "slippage_impact_pct_of_gross_pnl": safe_ratio(slippage, gross_abs),
            "missed_fills": int(execution_stats.get("missed_fills") or 0),
            "partial_fills": int(execution_stats.get("partial_fills") or 0),
            "orders_expired": int(execution_stats.get("orders_expired") or 0),
            "average_fill_notional": float((fills["qty"] * fills["price"]).mean()) if not fills.empty else 0.0,
        }


def snapshots_to_frame(snapshots: list[PortfolioSnapshot]) -> pd.DataFrame:
    frame = records_to_frame(snapshots)
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.set_index("timestamp").sort_index()


def records_to_frame(records: list[Any]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    rows = []
    for record in records:
        if is_dataclass(record):
            row = asdict(record)
        else:
            row = dict(record)
        if "side" in row and hasattr(row["side"], "value"):
            row["side"] = row["side"].value
        if "liquidity" in row and hasattr(row["liquidity"], "value"):
            row["liquidity"] = row["liquidity"].value
        if "r_multiple" not in row and hasattr(record, "r_multiple"):
            row["r_multiple"] = record.r_multiple
        rows.append(row)
    return pd.DataFrame(rows)


def grouped_trade_stats(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {"trades": 0, "net_pnl": 0.0, "win_rate": 0.0, "profit_factor": 0.0}
    wins = trades[trades["net_pnl"] > 0]
    losses = trades[trades["net_pnl"] < 0]
    gross_profit = float(wins["net_pnl"].sum())
    gross_loss = float(losses["net_pnl"].sum())
    return {
        "trades": int(len(trades)),
        "net_pnl": float(trades["net_pnl"].sum()),
        "win_rate": float(len(wins) / len(trades)),
        "profit_factor": gross_profit / abs(gross_loss) if gross_loss < 0 else math.inf if gross_profit > 0 else 0.0,
        "average_r": float(trades["r_multiple"].mean()) if "r_multiple" in trades else 0.0,
    }


def max_streak(values: pd.Series) -> int:
    best = current = 0
    for value in values.astype(bool):
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def equity_curve_stability(equity: pd.Series) -> float:
    if len(equity) < 3:
        return 0.0
    y = equity.to_numpy(dtype="float64")
    x = np.arange(len(y), dtype="float64")
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - safe_ratio(ss_res, ss_tot) if ss_tot > 0 else 0.0


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0 or not math.isfinite(denominator):
        return 0.0
    result = numerator / denominator
    return result if math.isfinite(result) else 0.0


def to_jsonable(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        return None
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value
