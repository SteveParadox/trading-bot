"""Core domain models shared by the backtester subsystems."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import Any

import pandas as pd


class Side(str, Enum):
    """Trade direction in position terms."""

    LONG = "LONG"
    SHORT = "SHORT"

    @property
    def order_side(self) -> "OrderSide":
        return OrderSide.BUY if self is Side.LONG else OrderSide.SELL

    @property
    def close_order_side(self) -> "OrderSide":
        return OrderSide.SELL if self is Side.LONG else OrderSide.BUY


class OrderSide(str, Enum):
    BUY = "Buy"
    SELL = "Sell"

    @property
    def position_side(self) -> Side:
        return Side.LONG if self is OrderSide.BUY else Side.SHORT


class OrderType(str, Enum):
    MARKET = "Market"
    LIMIT = "Limit"
    STOP_MARKET = "StopMarket"
    STOP_LIMIT = "StopLimit"


class OrderStatus(str, Enum):
    NEW = "New"
    PARTIALLY_FILLED = "PartiallyFilled"
    FILLED = "Filled"
    CANCELED = "Canceled"
    REJECTED = "Rejected"
    EXPIRED = "Expired"


class PositionMode(str, Enum):
    ONE_WAY = "one_way"
    HEDGE = "hedge"


class Liquidity(str, Enum):
    MAKER = "maker"
    TAKER = "taker"


TIMEFRAME_ALIASES: dict[str, str] = {
    "1": "1m",
    "3": "3m",
    "5": "5m",
    "15": "15m",
    "30": "30m",
    "60": "1h",
    "120": "2h",
    "240": "4h",
    "D": "1d",
}

TIMEFRAME_DELTAS: dict[str, timedelta] = {
    "1m": timedelta(minutes=1),
    "3m": timedelta(minutes=3),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "2h": timedelta(hours=2),
    "4h": timedelta(hours=4),
    "1d": timedelta(days=1),
}


def normalize_timeframe(timeframe: str) -> str:
    """Normalize Bybit-style intervals into compact pandas-friendly labels."""

    value = str(timeframe).strip()
    return TIMEFRAME_ALIASES.get(value, value.lower())


def timeframe_to_timedelta(timeframe: str) -> timedelta:
    normalized = normalize_timeframe(timeframe)
    try:
        return TIMEFRAME_DELTAS[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported timeframe: {timeframe}") from exc


def timeframe_to_pandas_rule(timeframe: str) -> str:
    """Return a pandas resampling rule for a normalized timeframe."""

    normalized = normalize_timeframe(timeframe)
    if normalized.endswith("m"):
        return f"{normalized[:-1]}min"
    if normalized.endswith("h"):
        return f"{normalized[:-1]}h"
    if normalized.endswith("d"):
        return f"{normalized[:-1]}D"
    raise ValueError(f"Unsupported timeframe for resampling: {timeframe}")


def annualization_factor(timeframe: str) -> float:
    """Approximate periods per year for return metrics."""

    seconds = timeframe_to_timedelta(timeframe).total_seconds()
    if seconds <= 0:
        return 1.0
    return (365.0 * 24.0 * 60.0 * 60.0) / seconds


def direction_multiplier(side: Side) -> int:
    return 1 if side is Side.LONG else -1


def round_to_step(value: float, step: float, *, up: bool = False) -> float:
    """Round a float to an exchange increment without relying on binary modulo."""

    if step <= 0:
        return value
    units = value / step
    rounded_units = math.ceil(units) if up else math.floor(units)
    decimals = max(0, len(f"{step:.16f}".rstrip("0").split(".")[-1]))
    return round(rounded_units * step, decimals)


@dataclass(frozen=True)
class InstrumentSpec:
    """Exchange constraints required for Bybit-like order rounding and sizing."""

    symbol: str
    tick_size: float = 0.0001
    qty_step: float = 0.001
    min_qty: float = 0.001
    min_notional: float = 5.0
    max_leverage: float = 100.0

    def round_qty(self, qty: float) -> float:
        return round_to_step(qty, self.qty_step, up=False)

    def round_price(self, price: float, *, up: bool = False) -> float:
        return round_to_step(price, self.tick_size, up=up)


@dataclass
class Order:
    """Simulated Bybit order.

    `activation_index` models latency: an order submitted after bar N cannot
    interact with candle N unless explicitly configured to do so.
    """

    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    qty: float
    submitted_at: pd.Timestamp
    activation_index: int
    price: float | None = None
    trigger_price: float | None = None
    reduce_only: bool = False
    close_on_trigger: bool = False
    leverage: float = 1.0
    status: OrderStatus = OrderStatus.NEW
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    expires_at_index: int | None = None
    order_link_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def remaining_qty(self) -> float:
        return max(0.0, self.qty - self.filled_qty)

    @property
    def is_active(self) -> bool:
        return self.status in {OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED}

    @property
    def is_stop(self) -> bool:
        return self.order_type in {OrderType.STOP_MARKET, OrderType.STOP_LIMIT}

    @property
    def is_limit(self) -> bool:
        return self.order_type in {OrderType.LIMIT, OrderType.STOP_LIMIT}

    def record_fill(self, qty: float, price: float) -> None:
        new_total = self.filled_qty + qty
        if new_total <= 0:
            return
        self.avg_fill_price = (
            (self.avg_fill_price * self.filled_qty) + (price * qty)
        ) / new_total
        self.filled_qty = new_total
        self.status = (
            OrderStatus.FILLED
            if self.remaining_qty <= max(1e-12, self.qty * 1e-9)
            else OrderStatus.PARTIALLY_FILLED
        )


@dataclass(frozen=True)
class Fill:
    order_id: str
    symbol: str
    side: OrderSide
    qty: float
    price: float
    fee: float
    liquidity: Liquidity
    timestamp: pd.Timestamp
    reduce_only: bool
    slippage: float = 0.0
    realized_pnl: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def notional(self) -> float:
        return self.qty * self.price


@dataclass
class Position:
    symbol: str
    side: Side
    qty: float
    avg_price: float
    leverage: float
    opened_at: pd.Timestamp
    updated_at: pd.Timestamp
    entry_fees: float = 0.0
    entry_slippage: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def notional(self) -> float:
        return self.qty * self.avg_price

    @property
    def margin(self) -> float:
        leverage = max(self.leverage, 1e-12)
        return self.notional / leverage

    def unrealized_pnl(self, mark_price: float) -> float:
        if self.side is Side.LONG:
            return (mark_price - self.avg_price) * self.qty
        return (self.avg_price - mark_price) * self.qty

    def liquidation_price(self, maintenance_margin_rate: float = 0.005) -> float:
        """Approximate isolated-margin liquidation price.

        This is a conservative estimate, not an exchange risk-engine clone.
        Funding, fee-to-close, tier changes, and cross-collateral are purposely
        outside the formula so reports label this as an estimate.
        """

        initial_margin_rate = 1.0 / max(self.leverage, 1e-12)
        if self.side is Side.LONG:
            return self.avg_price * (1.0 - initial_margin_rate + maintenance_margin_rate)
        return self.avg_price * (1.0 + initial_margin_rate - maintenance_margin_rate)


@dataclass(frozen=True)
class TradeRecord:
    trade_id: str
    symbol: str
    side: Side
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: float
    gross_pnl: float
    fees: float
    slippage: float
    net_pnl: float
    risk_amount: float
    exit_reason: str
    bars_held: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def r_multiple(self) -> float:
        if self.risk_amount <= 0:
            return 0.0
        return self.net_pnl / self.risk_amount


@dataclass(frozen=True)
class PortfolioSnapshot:
    timestamp: pd.Timestamp
    equity: float
    cash: float
    unrealized_pnl: float
    realized_pnl: float
    margin_used: float
    drawdown_pct: float
    open_positions: int
    gross_exposure: float
    net_exposure: float


@dataclass(frozen=True)
class SignalIntent:
    """A strategy request to enter a position."""

    symbol: str
    side: Side
    timestamp: pd.Timestamp
    entry_price_hint: float
    signal_row: dict[str, Any]
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExitPlan:
    stop_loss: float
    take_profit: float
    risk_distance: float
    reward_distance: float
    risk_reward: float


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    reason: str
    qty: float = 0.0
    notional: float = 0.0
    margin_required: float = 0.0
    risk_amount: float = 0.0
    exit_plan: ExitPlan | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
