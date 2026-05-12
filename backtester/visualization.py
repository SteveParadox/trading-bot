"""Exportable visualization helpers for backtest reports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtester.analytics import records_to_frame, snapshots_to_frame
from backtester.models import PortfolioSnapshot, TradeRecord


class BacktestVisualizer:
    """Create charts with matplotlib by default and Plotly for candlesticks."""

    def __init__(self, output_dir: str | Path = "reports/latest/charts") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def equity_curve(self, snapshots: list[PortfolioSnapshot], filename: str = "equity_curve.png") -> Path:
        plt = _matplotlib()
        equity = snapshots_to_frame(snapshots)
        path = self.output_dir / filename
        fig, ax = plt.subplots(figsize=(12, 5))
        equity["equity"].plot(ax=ax, color="#2563eb", linewidth=1.6)
        ax.set_title("Equity Curve")
        ax.set_ylabel("USDT")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def drawdown(self, snapshots: list[PortfolioSnapshot], filename: str = "drawdown.png") -> Path:
        plt = _matplotlib()
        equity = snapshots_to_frame(snapshots)
        path = self.output_dir / filename
        fig, ax = plt.subplots(figsize=(12, 4))
        (-equity["drawdown_pct"] * 100).plot(ax=ax, color="#dc2626", linewidth=1.2)
        ax.fill_between(equity.index, -equity["drawdown_pct"] * 100, 0, color="#fecaca", alpha=0.5)
        ax.set_title("Drawdown")
        ax.set_ylabel("%")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def monthly_returns(self, snapshots: list[PortfolioSnapshot], filename: str = "monthly_returns.png") -> Path:
        plt = _matplotlib()
        equity = snapshots_to_frame(snapshots)
        monthly = equity["equity"].resample("ME").last().pct_change().dropna() * 100
        path = self.output_dir / filename
        fig, ax = plt.subplots(figsize=(12, 4))
        colors = ["#16a34a" if value >= 0 else "#dc2626" for value in monthly]
        monthly.plot(kind="bar", ax=ax, color=colors)
        ax.set_title("Monthly Returns")
        ax.set_ylabel("%")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def win_loss_distribution(self, trades: list[TradeRecord], filename: str = "win_loss_distribution.png") -> Path:
        plt = _matplotlib()
        frame = records_to_frame(trades)
        path = self.output_dir / filename
        fig, ax = plt.subplots(figsize=(10, 4))
        if not frame.empty:
            frame["net_pnl"].plot(kind="hist", bins=40, ax=ax, color="#475569", alpha=0.85)
        ax.set_title("Win/Loss Distribution")
        ax.set_xlabel("Net PnL")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def trade_heatmap(self, trades: list[TradeRecord], filename: str = "trade_heatmap.png") -> Path:
        plt = _matplotlib()
        frame = records_to_frame(trades)
        path = self.output_dir / filename
        fig, ax = plt.subplots(figsize=(10, 5))
        if not frame.empty:
            frame["exit_time"] = pd.to_datetime(frame["exit_time"], utc=True)
            pivot = frame.pivot_table(
                index=frame["exit_time"].dt.day_name(),
                columns=frame["exit_time"].dt.hour,
                values="net_pnl",
                aggfunc="sum",
                fill_value=0.0,
            )
            ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            pivot = pivot.reindex([day for day in ordered_days if day in pivot.index])
            image = ax.imshow(pivot, aspect="auto", cmap="RdYlGn")
            ax.set_yticks(range(len(pivot.index)), pivot.index)
            ax.set_xticks(range(len(pivot.columns)), pivot.columns)
            fig.colorbar(image, ax=ax, label="Net PnL")
        ax.set_title("Trade Heatmap by Exit Hour")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def candlestick_overlay(
        self,
        candles: pd.DataFrame,
        trades: list[TradeRecord],
        symbol: str,
        filename: str | None = None,
    ) -> Path:
        go = _plotly()
        filename = filename or f"{symbol}_trade_overlay.html"
        path = self.output_dir / filename
        trade_frame = records_to_frame([trade for trade in trades if trade.symbol == symbol])
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=candles.index,
                    open=candles["open"],
                    high=candles["high"],
                    low=candles["low"],
                    close=candles["close"],
                    name=symbol,
                )
            ]
        )
        if not trade_frame.empty:
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(trade_frame["entry_time"], utc=True),
                    y=trade_frame["entry_price"],
                    mode="markers",
                    marker={"symbol": "triangle-up", "size": 9, "color": "#16a34a"},
                    name="Entries",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(trade_frame["exit_time"], utc=True),
                    y=trade_frame["exit_price"],
                    mode="markers",
                    marker={"symbol": "x", "size": 9, "color": "#dc2626"},
                    name="Exits",
                )
            )
        fig.update_layout(title=f"{symbol} Trade Overlay", xaxis_rangeslider_visible=False)
        fig.write_html(path)
        return path


def _matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Visualization requires matplotlib. Install it to export PNG charts.") from exc
    return plt


def _plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("Candlestick overlays require plotly. Install it to export HTML charts.") from exc
    return go
