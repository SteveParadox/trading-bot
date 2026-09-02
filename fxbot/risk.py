"""FX-specific exits, position sizing, and portfolio risk controls."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from fxbot.config import RiskSettings, StrategySettings
from fxbot.instruments import (
    FxInstrument,
    margin_required_home,
    pips_between,
    position_value_home,
    quote_to_home_factor,
)
from fxbot.models import FxPortfolioState, FxSignalIntent, Side


@dataclass(frozen=True)
class FxExitPlan:
    stop_loss: float
    take_profit: float
    risk_distance: float
    reward_distance: float
    risk_reward: float
    stop_pips: float
    take_profit_pips: float


@dataclass(frozen=True)
class FxRiskDecision:
    allowed: bool
    reason: str
    units: float = 0.0
    signed_units: float = 0.0
    position_value: float = 0.0
    margin_required: float = 0.0
    risk_amount: float = 0.0
    exit_plan: FxExitPlan | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class FxRiskManager:
    def __init__(self, risk: RiskSettings, strategy: StrategySettings) -> None:
        self.risk = risk
        self.strategy = strategy
        self.peak_equity = 0.0
        self._daily_start_equity: dict[str, float] = {}
        self._halted_days: set[str] = set()

    def evaluate_intent(
        self,
        intent: FxSignalIntent,
        instrument: FxInstrument,
        portfolio: FxPortfolioState,
        *,
        conversion_rates: dict[str, float] | None = None,
        snapshot_quote_factor: float | None = None,
        now: datetime | None = None,
    ) -> FxRiskDecision:
        self.peak_equity = max(self.peak_equity, portfolio.equity)
        hard_stop = self._hard_stop(portfolio, now or intent.timestamp)
        if hard_stop:
            return FxRiskDecision(False, hard_stop)
        if portfolio.balance < self.risk.min_balance:
            return FxRiskDecision(False, "min_balance")
        if portfolio.open_positions >= self.risk.max_open_positions:
            return FxRiskDecision(False, "max_open_positions")

        exit_plan = self.build_exit_plan(intent, instrument)
        if exit_plan is None:
            return FxRiskDecision(False, "exit_plan_unavailable")

        entry = intent.entry_price
        quote_factor = quote_to_home_factor(
            instrument,
            self.risk.account_currency,
            entry,
            conversion_rates=conversion_rates,
            snapshot_factor=snapshot_quote_factor,
        )
        risk_per_unit = exit_plan.risk_distance * quote_factor
        if risk_per_unit <= 0 or not math.isfinite(risk_per_unit):
            return FxRiskDecision(False, "invalid_risk_per_unit", exit_plan=exit_plan)

        risk_budget = portfolio.equity * self.risk.risk_per_trade_pct
        heat_remaining = max(0.0, portfolio.equity * self.risk.max_portfolio_risk_pct - portfolio.portfolio_risk)
        risk_budget = min(risk_budget, heat_remaining)
        if risk_budget <= 0:
            return FxRiskDecision(False, "portfolio_risk_limit", exit_plan=exit_plan)

        position_value_per_unit = entry * quote_factor
        margin_per_unit = position_value_per_unit * instrument.margin_rate
        if margin_per_unit <= 0:
            return FxRiskDecision(False, "invalid_margin_rate", exit_plan=exit_plan)

        gross_remaining = max(0.0, portfolio.equity * self.risk.max_gross_exposure_pct - portfolio.gross_exposure)
        pair_current = portfolio.pair_exposures.get(instrument.name, 0.0)
        pair_remaining = max(0.0, portfolio.equity * self.risk.max_pair_exposure_pct - abs(pair_current))
        margin_budget = max(0.0, portfolio.free_margin - portfolio.equity * self.risk.min_free_margin_pct)

        max_units = min(
            self.risk.max_units_per_trade,
            instrument.maximum_order_units,
            risk_budget / risk_per_unit,
            gross_remaining / position_value_per_unit if position_value_per_unit > 0 else 0.0,
            pair_remaining / position_value_per_unit if position_value_per_unit > 0 else 0.0,
            margin_budget / margin_per_unit if margin_per_unit > 0 else 0.0,
        )
        max_units = min(max_units, self._currency_limited_units(intent, instrument, portfolio, position_value_per_unit))
        units = instrument.round_units(max_units)
        if units < instrument.minimum_trade_size:
            return FxRiskDecision(False, "units_below_minimum", exit_plan=exit_plan, metadata={"raw_units": max_units})

        position_value = position_value_home(
            instrument,
            units,
            entry,
            self.risk.account_currency,
            conversion_rates=conversion_rates,
            snapshot_factor=snapshot_quote_factor,
        )
        margin_required = margin_required_home(
            instrument,
            units,
            entry,
            self.risk.account_currency,
            conversion_rates=conversion_rates,
            snapshot_factor=snapshot_quote_factor,
        )
        risk_amount = units * risk_per_unit
        return FxRiskDecision(
            True,
            "accepted",
            units=units,
            signed_units=units * intent.side.broker_units_sign,
            position_value=position_value,
            margin_required=margin_required,
            risk_amount=risk_amount,
            exit_plan=exit_plan,
            metadata={
                "quote_to_home_factor": quote_factor,
                "pip_value_per_unit": instrument.pip_size * quote_factor,
                "portfolio_risk_after": portfolio.portfolio_risk + risk_amount,
                "gross_exposure_after": portfolio.gross_exposure + position_value,
                "margin_rate": instrument.margin_rate,
            },
        )

    def build_exit_plan(self, intent: FxSignalIntent, instrument: FxInstrument) -> FxExitPlan | None:
        entry = intent.entry_price
        row = intent.signal_row
        if entry <= 0 or not math.isfinite(entry):
            return None
        if self.strategy.stop_mode == "atr":
            atr = float(row.get("atr") or 0.0)
            if atr <= 0 or not math.isfinite(atr):
                return None
            raw_distance = atr * self.strategy.atr_sl_multiplier
            raw_stop = entry - raw_distance if intent.side is Side.LONG else entry + raw_distance
        else:
            ma28 = float(row.get("ma28") or 0.0)
            if ma28 <= 0 or not math.isfinite(ma28):
                return None
            raw_stop = ma28

        if intent.side is Side.LONG:
            stop_loss = instrument.round_price(raw_stop)
            if stop_loss >= entry:
                return None
            risk_distance = entry - stop_loss
            take_profit = self._target_price(intent.side, entry, risk_distance, instrument)
            reward_distance = take_profit - entry
        else:
            stop_loss = instrument.round_price(raw_stop)
            if stop_loss <= entry:
                return None
            risk_distance = stop_loss - entry
            take_profit = self._target_price(intent.side, entry, risk_distance, instrument)
            reward_distance = entry - take_profit

        stop_pips = pips_between(instrument, entry, stop_loss)
        if stop_pips < self.strategy.min_stop_pips or stop_pips > self.strategy.max_stop_pips:
            return None
        if instrument.minimum_stop_distance and risk_distance < instrument.minimum_stop_distance:
            return None
        if reward_distance <= 0 or risk_distance <= 0:
            return None
        risk_reward = reward_distance / risk_distance
        if risk_reward < self.strategy.min_risk_reward:
            return None
        return FxExitPlan(
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_distance=risk_distance,
            reward_distance=reward_distance,
            risk_reward=risk_reward,
            stop_pips=stop_pips,
            take_profit_pips=reward_distance / instrument.pip_size,
        )

    def _target_price(
        self,
        side: Side,
        entry_price: float,
        risk_distance: float,
        instrument: FxInstrument,
    ) -> float:
        raw = (
            entry_price + risk_distance * self.strategy.min_risk_reward
            if side is Side.LONG
            else entry_price - risk_distance * self.strategy.min_risk_reward
        )
        tick = 10.0 ** (-instrument.display_precision)
        for _ in range(20):
            target = instrument.round_price(raw)
            reward = target - entry_price if side is Side.LONG else entry_price - target
            if risk_distance > 0 and reward / risk_distance >= self.strategy.min_risk_reward:
                return target
            raw = raw + tick if side is Side.LONG else raw - tick
        return instrument.round_price(raw)

    def _currency_limited_units(
        self,
        intent: FxSignalIntent,
        instrument: FxInstrument,
        portfolio: FxPortfolioState,
        position_value_per_unit: float,
    ) -> float:
        limit = portfolio.equity * self.risk.max_currency_exposure_pct
        if limit <= 0 or position_value_per_unit <= 0:
            return 0.0
        upper = math.inf
        base_per_unit = position_value_per_unit * intent.side.sign
        quote_per_unit = -position_value_per_unit * intent.side.sign
        for currency, per_unit in (
            (instrument.base_currency, base_per_unit),
            (instrument.quote_currency, quote_per_unit),
        ):
            current = portfolio.currency_exposures.get(currency, 0.0)
            candidate_upper = _max_units_for_absolute_limit(current, per_unit, limit)
            upper = min(upper, candidate_upper)
        return max(0.0, upper)

    def _hard_stop(self, portfolio: FxPortfolioState, timestamp: datetime) -> str | None:
        if portfolio.equity <= 0:
            return "equity_depleted"
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - portfolio.equity) / self.peak_equity
            if drawdown >= self.risk.max_drawdown_pct:
                return "max_drawdown_halt"
        day = timestamp.date().isoformat()
        if day in self._halted_days:
            return "daily_loss_halt"
        self._daily_start_equity.setdefault(day, portfolio.equity)
        start = self._daily_start_equity[day]
        if start > 0 and (start - portfolio.equity) / start >= self.risk.max_daily_loss_pct:
            self._halted_days.add(day)
            return "daily_loss_halt"
        return None


def _max_units_for_absolute_limit(current: float, per_unit: float, limit: float) -> float:
    if per_unit == 0:
        return math.inf
    if per_unit > 0:
        return max(0.0, (limit - current) / per_unit)
    return max(0.0, (-limit - current) / per_unit)
