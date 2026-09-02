"""Shared FX domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"

    @property
    def sign(self) -> int:
        return 1 if self is Side.LONG else -1

    @property
    def broker_units_sign(self) -> int:
        return 1 if self is Side.LONG else -1


class BotRunState(str, Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    HALTED = "halted"


@dataclass(frozen=True)
class FxSignalIntent:
    instrument: str
    side: Side
    timestamp: datetime
    entry_price: float
    signal_row: dict[str, Any]
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FxPosition:
    instrument: str
    side: Side
    units: float
    avg_price: float
    unrealized_pl: float
    margin_used: float
    opened_at: datetime | None = None
    trade_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FxPortfolioState:
    equity: float
    balance: float
    margin_used: float
    open_positions: int
    account_currency: str = "USD"
    portfolio_risk: float = 0.0
    gross_exposure: float = 0.0
    pair_exposures: dict[str, float] = field(default_factory=dict)
    currency_exposures: dict[str, float] = field(default_factory=dict)

    @property
    def free_margin(self) -> float:
        return max(0.0, self.equity - self.margin_used)
