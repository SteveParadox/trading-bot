"""Portfolio and order risk controls for simulated futures trading."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd

from backtester.config import BacktestConfig, RiskConfig, StrategyConfig
from backtester.models import ExitPlan, InstrumentSpec, RiskDecision, Side, SignalIntent

log = logging.getLogger(__name__)


class ExchangeRiskView(Protocol):
    cash: float
    realized_pnl: float

    def equity(self) -> float: ...

    def open_position_count(self) -> int: ...

    def has_position(self, symbol: str, side: Side | None = None) -> bool: ...

    def gross_exposure(self) -> float: ...

    def symbol_exposure(self, symbol: str) -> float: ...

    def margin_used(self) -> float: ...

    def portfolio_heat(self) -> float: ...


@dataclass(frozen=True)
class MarketQualityDecision:
    allowed: bool
    reason: str
    metadata: dict[str, Any]


class RiskManager:
    """Centralized trade admission, sizing, and liquidation estimates."""

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config
        self.risk: RiskConfig = config.risk
        self.strategy: StrategyConfig = config.strategy
        self._daily_start_equity: dict[pd.Timestamp, float] = {}
        self._halted_days: set[pd.Timestamp] = set()
        self.peak_equity = config.risk.initial_equity

    def evaluate_intent(
        self,
        intent: SignalIntent,
        exchange: ExchangeRiskView,
        timestamp: pd.Timestamp,
    ) -> RiskDecision:
        """Return a fully sized order decision or a rejection reason."""

        equity = exchange.equity()
        self.peak_equity = max(self.peak_equity, equity)

        hard_stop = self._portfolio_hard_stop(exchange, timestamp)
        if hard_stop:
            return RiskDecision(False, hard_stop)

        instrument = self.config.instrument_for(intent.symbol)
        if exchange.open_position_count() >= self.risk.max_open_positions:
            return RiskDecision(False, "max_open_positions")
        if self.risk.max_symbol_positions <= 1 and exchange.has_position(intent.symbol):
            return RiskDecision(False, "symbol_position_already_open")

        quality = self.validate_market_quality(intent)
        if not quality.allowed:
            return RiskDecision(False, quality.reason, metadata=quality.metadata)

        exit_plan = self.build_exit_plan(intent, instrument)
        if exit_plan is None:
            return RiskDecision(False, "exit_plan_unavailable")

        leverage = min(self.risk.leverage, self.risk.max_leverage, instrument.max_leverage)
        risk_budget = equity * self.risk.risk_per_trade_pct
        atr_pct = float(intent.signal_row.get("atr_pct") or 0.0)
        if self.risk.volatility_target_atr_pct and atr_pct > 0:
            volatility_scalar = min(1.0, self.risk.volatility_target_atr_pct / atr_pct)
            risk_budget *= volatility_scalar

        available_heat = max(0.0, (equity * self.risk.max_portfolio_heat_pct) - exchange.portfolio_heat())
        risk_budget = min(risk_budget, available_heat)
        if risk_budget <= 0:
            return RiskDecision(False, "portfolio_heat_limit")

        entry_price = intent.entry_price_hint
        if entry_price <= 0 or not math.isfinite(entry_price):
            return RiskDecision(False, "invalid_entry_price", exit_plan=exit_plan)

        symbol_exposure = exchange.symbol_exposure(intent.symbol)
        symbol_exposure_limit = equity * self.risk.max_symbol_exposure_pct
        if symbol_exposure >= symbol_exposure_limit:
            return RiskDecision(False, "symbol_exposure_limit")

        gross_exposure_remaining = max(
            0.0,
            equity * self.risk.max_gross_exposure_pct - exchange.gross_exposure(),
        )
        symbol_exposure_remaining = max(0.0, symbol_exposure_limit - symbol_exposure)
        notional_cap = min(
            self.risk.max_trade_notional,
            symbol_exposure_remaining,
            gross_exposure_remaining,
        )
        if notional_cap <= 0:
            return RiskDecision(False, "exposure_limit")

        qty_by_risk = risk_budget / exit_plan.risk_distance
        qty_by_notional = notional_cap / entry_price
        qty = instrument.round_qty(min(qty_by_risk, qty_by_notional))
        if qty < instrument.min_qty:
            return RiskDecision(False, "quantity_below_minimum", exit_plan=exit_plan)

        notional = qty * entry_price
        if notional < instrument.min_notional:
            return RiskDecision(False, "notional_below_minimum", qty=qty, notional=notional, exit_plan=exit_plan)

        margin_required = notional / max(leverage, 1e-12)
        if exchange.cash < self.risk.min_balance_usdt:
            return RiskDecision(False, "min_balance")
        free_equity = max(0.0, equity - exchange.margin_used())
        if margin_required > free_equity:
            return RiskDecision(False, "insufficient_free_equity_for_margin")

        risk_amount = qty * exit_plan.risk_distance
        liquidation = estimate_liquidation_price(
            side=intent.side,
            entry_price=entry_price,
            leverage=leverage,
            maintenance_margin_rate=self.risk.maintenance_margin_rate,
        )
        return RiskDecision(
            True,
            "accepted",
            qty=qty,
            notional=notional,
            margin_required=margin_required,
            risk_amount=risk_amount,
            exit_plan=exit_plan,
            metadata={
                "leverage": leverage,
                "liquidation_price_estimate": liquidation,
                "portfolio_heat_after": exchange.portfolio_heat() + risk_amount,
                **quality.metadata,
            },
        )

    def validate_market_quality(self, intent: SignalIntent) -> MarketQualityDecision:
        row = intent.signal_row
        atr_pct = float(row.get("atr_pct") or 0.0)
        if atr_pct < self.strategy.min_atr_pct or atr_pct > self.strategy.max_atr_pct:
            return MarketQualityDecision(
                False,
                "atr_filter",
                {
                    "atr_pct": atr_pct,
                    "min_atr_pct": self.strategy.min_atr_pct,
                    "max_atr_pct": self.strategy.max_atr_pct,
                },
            )

        last_close = float(row.get("close") or intent.entry_price_hint)
        if last_close > 0:
            deviation = abs(intent.entry_price_hint - last_close) / last_close
            if deviation > self.strategy.max_entry_deviation_pct:
                return MarketQualityDecision(
                    False,
                    "entry_deviation_filter",
                    {"entry_deviation_pct": deviation},
                )
        return MarketQualityDecision(True, "ok", {"atr_pct": atr_pct})

    def build_exit_plan(self, intent: SignalIntent, instrument: InstrumentSpec) -> ExitPlan | None:
        entry_price = intent.entry_price_hint
        row = intent.signal_row
        if entry_price <= 0 or not math.isfinite(entry_price):
            return None

        if self.strategy.stop_mode == "atr":
            atr = float(row.get("atr") or float("nan"))
            if not math.isfinite(atr) or atr <= 0:
                return None
            raw_stop = (
                entry_price - (atr * self.strategy.atr_sl_multiplier)
                if intent.side is Side.LONG
                else entry_price + (atr * self.strategy.atr_sl_multiplier)
            )
        else:
            ma28 = float(row.get("ma28") or float("nan"))
            if not math.isfinite(ma28) or ma28 <= 0:
                return None
            raw_stop = ma28

        if intent.side is Side.LONG:
            stop_loss = instrument.round_price(raw_stop, up=False)
            if stop_loss >= entry_price:
                return None
            risk_distance = entry_price - stop_loss
            take_profit = self._rounded_target_price(
                side=intent.side,
                entry_price=entry_price,
                risk_distance=risk_distance,
                instrument=instrument,
            )
            reward_distance = take_profit - entry_price
        else:
            stop_loss = instrument.round_price(raw_stop, up=True)
            if stop_loss <= entry_price:
                return None
            risk_distance = stop_loss - entry_price
            take_profit = self._rounded_target_price(
                side=intent.side,
                entry_price=entry_price,
                risk_distance=risk_distance,
                instrument=instrument,
            )
            reward_distance = entry_price - take_profit

        if risk_distance <= 0 or reward_distance <= 0:
            return None
        risk_reward = reward_distance / risk_distance
        if risk_reward < self.strategy.min_risk_reward:
            return None
        return ExitPlan(
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_distance=risk_distance,
            reward_distance=reward_distance,
            risk_reward=risk_reward,
        )

    def _rounded_target_price(
        self,
        *,
        side: Side,
        entry_price: float,
        risk_distance: float,
        instrument: InstrumentSpec,
    ) -> float:
        """Round TP conservatively while preserving the configured minimum RR."""

        min_reward = risk_distance * self.strategy.min_risk_reward
        tick = instrument.tick_size
        raw_target = entry_price + min_reward if side is Side.LONG else entry_price - min_reward
        for _ in range(20):
            target = instrument.round_price(raw_target, up=side is Side.SHORT)
            reward = target - entry_price if side is Side.LONG else entry_price - target
            if risk_distance > 0 and reward / risk_distance >= self.strategy.min_risk_reward:
                return target
            raw_target = raw_target + tick if side is Side.LONG else raw_target - tick
        return instrument.round_price(raw_target, up=side is Side.SHORT)

    def _portfolio_hard_stop(self, exchange: ExchangeRiskView, timestamp: pd.Timestamp) -> str | None:
        equity = exchange.equity()
        if equity <= 0:
            return "equity_depleted"

        drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        if drawdown >= self.risk.max_drawdown_pct:
            return "max_drawdown_halt"

        day = pd.Timestamp(timestamp).normalize()
        if day in self._halted_days:
            return "daily_loss_halt"
        self._daily_start_equity.setdefault(day, equity)
        day_start = self._daily_start_equity[day]
        if day_start > 0 and (day_start - equity) / day_start >= self.risk.daily_loss_limit_pct:
            self._halted_days.add(day)
            return "daily_loss_halt"
        return None


def estimate_liquidation_price(
    *,
    side: Side,
    entry_price: float,
    leverage: float,
    maintenance_margin_rate: float = 0.005,
) -> float:
    initial_margin_rate = 1.0 / max(leverage, 1e-12)
    if side is Side.LONG:
        return entry_price * (1.0 - initial_margin_rate + maintenance_margin_rate)
    return entry_price * (1.0 + initial_margin_rate - maintenance_margin_rate)
