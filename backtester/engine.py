"""Backtest orchestration loop."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from backtester.analytics import PerformanceAnalyzer
from backtester.config import BacktestConfig
from backtester.data import DataPortal
from backtester.execution import SimulatedExchange
from backtester.models import (
    Fill,
    OrderSide,
    OrderType,
    PortfolioSnapshot,
    SignalIntent,
    Side,
    timeframe_to_timedelta,
)
from backtester.risk import RiskManager
from backtester.strategy import IndicatorSignalStrategy, Strategy, StrategyContext

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestResult:
    config: BacktestConfig
    snapshots: list[PortfolioSnapshot]
    fills: list[Fill]
    trades: list
    orders: list
    metrics: dict[str, Any]
    execution_stats: dict[str, float]

    def export(self, output_dir: str | Path) -> None:
        analyzer = PerformanceAnalyzer(self.config)
        analyzer.export(
            output_dir=output_dir,
            snapshots=self.snapshots,
            trades=self.trades,
            fills=self.fills,
            metrics=self.metrics,
            execution_stats=self.execution_stats,
        )


class BacktestEngine:
    """Candle-by-candle engine with delayed execution and portfolio accounting."""

    def __init__(
        self,
        config: BacktestConfig,
        data: DataPortal,
        strategy: Strategy | None = None,
        exchange: SimulatedExchange | None = None,
        risk_manager: RiskManager | None = None,
    ) -> None:
        self.config = config
        self.data = data
        self.strategy = strategy or IndicatorSignalStrategy(config.strategy)
        self.exchange = exchange or SimulatedExchange(config)
        self.risk = risk_manager or RiskManager(config)
        self._prepared = False

    def run(self) -> BacktestResult:
        self.prepare()
        base_tf = self.config.data.base_timeframe
        base_delta = pd.Timedelta(timeframe_to_timedelta(base_tf))

        for bar_index, (bar_open, candles) in enumerate(self.data.iter_symbol_candles(base_tf)):
            for symbol, candle in candles.items():
                fills = self.exchange.process_candle(
                    symbol=symbol,
                    timestamp=bar_open,
                    candle=candle,
                    bar_index=bar_index,
                )
                for fill in fills:
                    self._handle_fill(fill, bar_open, bar_index)

            decision_time = pd.Timestamp(bar_open) + base_delta
            self.exchange.mark_to_market(decision_time)

            context = StrategyContext(
                timestamp=decision_time,
                bar_index=bar_index,
                data=self.data,
                exchange=self.exchange,
            )
            intents = self.strategy.generate_intents(context)
            self._submit_intents(intents, decision_time, bar_index)

        metrics = PerformanceAnalyzer(self.config).calculate(
            snapshots=self.exchange.snapshots,
            trades=self.exchange.trades,
            fills=self.exchange.fills,
            execution_stats=self.exchange.stats,
        )
        result = BacktestResult(
            config=self.config,
            snapshots=self.exchange.snapshots,
            fills=self.exchange.fills,
            trades=self.exchange.trades,
            orders=list(self.exchange.orders.values()),
            metrics=metrics,
            execution_stats=self.exchange.stats.copy(),
        )
        if self.config.analytics.export_json or self.config.analytics.export_csv:
            result.export(self.config.analytics.output_dir)
        return result

    def prepare(self) -> None:
        if self._prepared:
            return
        self.strategy.prepare_data(self.data)
        self._prepared = True

    def _submit_intents(
        self,
        intents: list[SignalIntent],
        decision_time: pd.Timestamp,
        bar_index: int,
    ) -> None:
        for intent in intents:
            decision = self.risk.evaluate_intent(intent, self.exchange, decision_time)
            if not decision.allowed or decision.exit_plan is None:
                log.debug("%s rejected: %s", intent.symbol, decision.reason)
                continue
            metadata = {
                **intent.metadata,
                "signal_row": intent.signal_row,
                "side": intent.side.value,
                "risk_amount": decision.risk_amount,
                "pretrade_exit_plan": asdict(decision.exit_plan),
                "entry_price_hint": intent.entry_price_hint,
                "create_bracket": True,
            }
            metadata.update(decision.metadata)
            self.exchange.submit_market_entry(
                symbol=intent.symbol,
                side=intent.side,
                qty=decision.qty,
                timestamp=decision_time,
                current_index=bar_index,
                leverage=float(decision.metadata.get("leverage") or self.config.risk.leverage),
                metadata=metadata,
            )

    def _handle_fill(self, fill: Fill, timestamp: pd.Timestamp, bar_index: int) -> None:
        if fill.reduce_only or not fill.metadata.get("create_bracket"):
            return
        side_value = str(fill.metadata.get("side") or "")
        side = Side.LONG if side_value == Side.LONG.value else Side.SHORT
        signal_row = dict(fill.metadata.get("signal_row") or {})
        intent = SignalIntent(
            symbol=fill.symbol,
            side=side,
            timestamp=fill.timestamp,
            entry_price_hint=fill.price,
            signal_row=signal_row,
            score=float(fill.metadata.get("score") or 0.0),
            metadata=fill.metadata,
        )
        instrument = self.config.instrument_for(fill.symbol)
        exit_plan = self.risk.build_exit_plan(intent, instrument)
        if exit_plan is None:
            log.warning("%s entry filled but protective plan is invalid", fill.symbol)
            if self.config.execution.close_on_protection_failure:
                self.exchange.submit_order(
                    symbol=fill.symbol,
                    side=OrderSide.SELL if side is Side.LONG else OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    qty=fill.qty,
                    timestamp=timestamp,
                    current_index=bar_index,
                    reduce_only=True,
                    metadata={"exit_reason": "protection_failure"},
                )
            return

        risk_amount = fill.qty * exit_plan.risk_distance
        position = self.exchange.position_for(fill.symbol, side)
        if position:
            position.metadata["risk_amount"] = risk_amount
            position.metadata["actual_exit_plan"] = asdict(exit_plan)
        self.exchange.submit_protective_bracket(
            symbol=fill.symbol,
            position_side=side,
            qty=fill.qty,
            entry_price=fill.price,
            exit_plan=exit_plan,
            timestamp=timestamp,
            current_index=bar_index,
            leverage=float(fill.metadata.get("leverage") or self.config.risk.leverage),
            risk_amount=risk_amount,
            metadata={"source_entry_order": fill.order_id},
        )
