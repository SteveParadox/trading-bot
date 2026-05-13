"""Configuration models for the backtesting framework."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

from backtester.models import InstrumentSpec, PositionMode, normalize_timeframe

T = TypeVar("T")

VALID_DATA_ALIGNMENTS = {"intersection", "union"}
VALID_STOP_MODES = {"ma", "atr"}
VALID_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}


def _finite_number(name: str, value: float | int) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _require_positive(name: str, value: float | int) -> None:
    if _finite_number(name, value) <= 0:
        raise ValueError(f"{name} must be positive")


def _require_non_negative(name: str, value: float | int) -> None:
    if _finite_number(name, value) < 0:
        raise ValueError(f"{name} cannot be negative")


def _require_probability(name: str, value: float | int) -> None:
    number = _finite_number(name, value)
    if number < 0 or number > 1:
        raise ValueError(f"{name} must be between 0 and 1")


def _require_min_int(name: str, value: int, minimum: int) -> None:
    if int(value) < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _require_choice(name: str, value: str, choices: set[str]) -> None:
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of: {allowed}")


@dataclass(frozen=True)
class DataConfig:
    """Historical data configuration.

    `data_path` may point at either a directory or a single CSV/Parquet file.
    Directory loading uses `file_pattern`, for example
    `{symbol}_{timeframe}.csv`.
    """

    data_path: str = "data"
    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT"])
    base_timeframe: str = "5m"
    timeframes: list[str] = field(default_factory=lambda: ["5m", "1h"])
    file_pattern: str = "{symbol}_{timeframe}.csv"
    timestamp_column: str = "timestamp"
    timezone: str = "UTC"
    start: str | None = None
    end: str | None = None
    alignment: str = "intersection"
    fill_missing: bool = False
    max_gap_candles: int = 1
    resample_from: str | None = None
    validate_candles: bool = True

    def __post_init__(self) -> None:
        base_timeframe = normalize_timeframe(self.base_timeframe)
        resample_from = normalize_timeframe(self.resample_from) if self.resample_from else None
        alignment = self.alignment.lower()
        symbols = [str(symbol).strip().upper() for symbol in self.symbols if str(symbol).strip()]
        if not symbols:
            raise ValueError("data.symbols must contain at least one symbol")
        _require_choice("data.alignment", alignment, VALID_DATA_ALIGNMENTS)
        _require_min_int("data.max_gap_candles", self.max_gap_candles, 0)
        object.__setattr__(self, "base_timeframe", base_timeframe)
        object.__setattr__(
            self,
            "timeframes",
            sorted({normalize_timeframe(tf) for tf in self.timeframes + [base_timeframe]}),
        )
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "alignment", alignment)
        object.__setattr__(self, "resample_from", resample_from)


@dataclass(frozen=True)
class ExecutionConfig:
    """Exchange simulation knobs.

    Defaults are deliberately conservative: market orders execute on the next
    candle, pay taker fees, cross half-spread, and receive extra slippage.
    """

    taker_fee_rate: float = 0.00055
    maker_fee_rate: float = 0.00020
    spread_bps: float = 2.0
    market_slippage_bps: float = 3.0
    stop_slippage_bps: float = 5.0
    limit_slippage_bps: float = 0.0
    market_latency_candles: int = 1
    resting_order_latency_candles: int = 1
    partial_fill_probability: float = 0.0
    partial_fill_min_pct: float = 0.25
    partial_fill_max_pct: float = 0.75
    limit_fill_probability: float = 0.85
    stop_fill_probability: float = 1.0
    missed_fill_probability: float = 0.0
    conservative_intrabar_priority: bool = True
    order_timeout_candles: int | None = None
    close_on_protection_failure: bool = True
    position_mode: PositionMode = PositionMode.ONE_WAY
    random_seed: int = 7

    def __post_init__(self) -> None:
        if not isinstance(self.position_mode, PositionMode):
            object.__setattr__(self, "position_mode", PositionMode(str(self.position_mode)))
        _require_non_negative("execution.taker_fee_rate", self.taker_fee_rate)
        _require_non_negative("execution.maker_fee_rate", self.maker_fee_rate)
        _require_non_negative("execution.spread_bps", self.spread_bps)
        _require_non_negative("execution.market_slippage_bps", self.market_slippage_bps)
        _require_non_negative("execution.stop_slippage_bps", self.stop_slippage_bps)
        _require_non_negative("execution.limit_slippage_bps", self.limit_slippage_bps)
        _require_min_int("execution.market_latency_candles", self.market_latency_candles, 0)
        _require_min_int("execution.resting_order_latency_candles", self.resting_order_latency_candles, 0)
        _require_probability("execution.partial_fill_probability", self.partial_fill_probability)
        _require_probability("execution.partial_fill_min_pct", self.partial_fill_min_pct)
        _require_probability("execution.partial_fill_max_pct", self.partial_fill_max_pct)
        if self.partial_fill_min_pct > self.partial_fill_max_pct:
            raise ValueError("execution.partial_fill_min_pct cannot exceed partial_fill_max_pct")
        _require_probability("execution.limit_fill_probability", self.limit_fill_probability)
        _require_probability("execution.stop_fill_probability", self.stop_fill_probability)
        _require_probability("execution.missed_fill_probability", self.missed_fill_probability)
        if self.order_timeout_candles is not None:
            _require_min_int("execution.order_timeout_candles", self.order_timeout_candles, 0)


@dataclass(frozen=True)
class StrategyConfig:
    """Default live-bot strategy adapter settings."""

    name: str = "indicator_signal_strategy"
    entry_timeframe: str = "5m"
    htf_timeframe: str = "1h"
    min_risk_reward: float = 1.45
    stop_mode: str = "ma"
    atr_sl_multiplier: float = 1.5
    partial_tp_enabled: bool = True
    tp1_qty_pct: float = 0.50
    tp2_multiplier: float = 2.0
    max_entry_deviation_pct: float = 0.003
    min_atr_pct: float = 0.001
    max_atr_pct: float = 0.08
    max_spread_bps: float = 12.0
    max_signals_per_bar: int = 2
    use_signal_ranking: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "entry_timeframe", normalize_timeframe(self.entry_timeframe))
        object.__setattr__(self, "htf_timeframe", normalize_timeframe(self.htf_timeframe))
        stop_mode = self.stop_mode.lower()
        _require_choice("strategy.stop_mode", stop_mode, VALID_STOP_MODES)
        _require_positive("strategy.min_risk_reward", self.min_risk_reward)
        _require_positive("strategy.atr_sl_multiplier", self.atr_sl_multiplier)
        _require_probability("strategy.tp1_qty_pct", self.tp1_qty_pct)
        if self.partial_tp_enabled and self.tp1_qty_pct in {0.0, 1.0}:
            raise ValueError("strategy.tp1_qty_pct must leave quantity for both partial targets")
        _require_positive("strategy.tp2_multiplier", self.tp2_multiplier)
        _require_non_negative("strategy.max_entry_deviation_pct", self.max_entry_deviation_pct)
        _require_non_negative("strategy.min_atr_pct", self.min_atr_pct)
        _require_positive("strategy.max_atr_pct", self.max_atr_pct)
        if self.min_atr_pct > self.max_atr_pct:
            raise ValueError("strategy.min_atr_pct cannot exceed max_atr_pct")
        _require_non_negative("strategy.max_spread_bps", self.max_spread_bps)
        _require_min_int("strategy.max_signals_per_bar", self.max_signals_per_bar, 0)
        object.__setattr__(self, "stop_mode", stop_mode)


@dataclass(frozen=True)
class RiskConfig:
    initial_equity: float = 1_000.0
    risk_per_trade_pct: float = 0.005
    max_trade_notional: float = 250.0
    leverage: float = 2.0
    max_leverage: float = 10.0
    min_balance_usdt: float = 10.0
    max_open_positions: int = 3
    max_symbol_positions: int = 1
    max_drawdown_pct: float = 0.20
    daily_loss_limit_pct: float = 0.03
    max_portfolio_heat_pct: float = 0.06
    max_symbol_exposure_pct: float = 0.35
    max_gross_exposure_pct: float = 1.5
    volatility_target_atr_pct: float | None = None
    maintenance_margin_rate: float = 0.005

    def __post_init__(self) -> None:
        _require_positive("risk.initial_equity", self.initial_equity)
        _require_non_negative("risk.risk_per_trade_pct", self.risk_per_trade_pct)
        _require_positive("risk.max_trade_notional", self.max_trade_notional)
        _require_positive("risk.leverage", self.leverage)
        _require_positive("risk.max_leverage", self.max_leverage)
        _require_non_negative("risk.min_balance_usdt", self.min_balance_usdt)
        _require_min_int("risk.max_open_positions", self.max_open_positions, 0)
        _require_min_int("risk.max_symbol_positions", self.max_symbol_positions, 0)
        _require_probability("risk.max_drawdown_pct", self.max_drawdown_pct)
        _require_probability("risk.daily_loss_limit_pct", self.daily_loss_limit_pct)
        _require_non_negative("risk.max_portfolio_heat_pct", self.max_portfolio_heat_pct)
        _require_non_negative("risk.max_symbol_exposure_pct", self.max_symbol_exposure_pct)
        _require_non_negative("risk.max_gross_exposure_pct", self.max_gross_exposure_pct)
        if self.volatility_target_atr_pct is not None:
            _require_positive("risk.volatility_target_atr_pct", self.volatility_target_atr_pct)
        _require_non_negative("risk.maintenance_margin_rate", self.maintenance_margin_rate)


@dataclass(frozen=True)
class AnalyticsConfig:
    output_dir: str = "reports/latest"
    export_json: bool = True
    export_csv: bool = True
    risk_free_rate: float = 0.0

    def __post_init__(self) -> None:
        _finite_number("analytics.risk_free_rate", self.risk_free_rate)


@dataclass(frozen=True)
class OptimizationConfig:
    objective: str = "calmar"
    train_pct: float = 0.70
    walk_forward_train_bars: int = 2_000
    walk_forward_test_bars: int = 500
    random_iterations: int = 25
    monte_carlo_iterations: int = 500
    random_seed: int = 7

    def __post_init__(self) -> None:
        _require_probability("optimization.train_pct", self.train_pct)
        if self.train_pct in {0.0, 1.0}:
            raise ValueError("optimization.train_pct must be between 0 and 1")
        _require_min_int("optimization.walk_forward_train_bars", self.walk_forward_train_bars, 1)
        _require_min_int("optimization.walk_forward_test_bars", self.walk_forward_test_bars, 1)
        _require_min_int("optimization.random_iterations", self.random_iterations, 1)
        _require_min_int("optimization.monte_carlo_iterations", self.monte_carlo_iterations, 1)


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    file_path: str | None = "reports/backtest.log"

    def __post_init__(self) -> None:
        level = self.level.upper()
        _require_choice("logging.level", level, VALID_LOG_LEVELS)
        object.__setattr__(self, "level", level)


@dataclass(frozen=True)
class BacktestConfig:
    data: DataConfig = field(default_factory=DataConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    instruments: dict[str, InstrumentSpec] = field(default_factory=dict)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "BacktestConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BacktestConfig":
        instruments_raw = raw.get("instruments", {})
        instruments = {}
        for symbol, values in instruments_raw.items():
            cleaned = {key: value for key, value in dict(values).items() if key != "symbol"}
            instruments[symbol.upper()] = InstrumentSpec(symbol=symbol.upper(), **cleaned)
        return cls(
            data=_dataclass_from_dict(DataConfig, raw.get("data", {})),
            execution=_dataclass_from_dict(ExecutionConfig, raw.get("execution", {})),
            strategy=_dataclass_from_dict(StrategyConfig, raw.get("strategy", {})),
            risk=_dataclass_from_dict(RiskConfig, raw.get("risk", {})),
            analytics=_dataclass_from_dict(AnalyticsConfig, raw.get("analytics", {})),
            optimization=_dataclass_from_dict(OptimizationConfig, raw.get("optimization", {})),
            instruments=instruments,
            logging=_dataclass_from_dict(LoggingConfig, raw.get("logging", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return _serialize_dataclass(self)

    def instrument_for(self, symbol: str) -> InstrumentSpec:
        return self.instruments.get(symbol.upper(), InstrumentSpec(symbol=symbol.upper()))


def _dataclass_from_dict(cls: type[T], raw: dict[str, Any]) -> T:
    valid = {field.name for field in fields(cls)}
    values = {key: value for key, value in raw.items() if key in valid}
    return cls(**values)  # type: ignore[arg-type]


def _serialize_dataclass(value: Any) -> Any:
    if isinstance(value, InstrumentSpec):
        return asdict(value)
    if isinstance(value, PositionMode):
        return value.value
    if is_dataclass(value):
        return {key: _serialize_dataclass(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _serialize_dataclass(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_dataclass(item) for item in value]
    return value
