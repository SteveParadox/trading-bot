"""Configuration models for the backtesting framework."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

from backtester.models import InstrumentSpec, PositionMode, normalize_timeframe

T = TypeVar("T")


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
        object.__setattr__(self, "base_timeframe", normalize_timeframe(self.base_timeframe))
        object.__setattr__(
            self,
            "timeframes",
            sorted({normalize_timeframe(tf) for tf in self.timeframes + [self.base_timeframe]}),
        )
        object.__setattr__(self, "symbols", [symbol.upper() for symbol in self.symbols])


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
        object.__setattr__(self, "stop_mode", self.stop_mode.lower())


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


@dataclass(frozen=True)
class AnalyticsConfig:
    output_dir: str = "reports/latest"
    export_json: bool = True
    export_csv: bool = True
    risk_free_rate: float = 0.0


@dataclass(frozen=True)
class OptimizationConfig:
    objective: str = "calmar"
    train_pct: float = 0.70
    walk_forward_train_bars: int = 2_000
    walk_forward_test_bars: int = 500
    random_iterations: int = 25
    monte_carlo_iterations: int = 500
    random_seed: int = 7


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    file_path: str | None = "reports/backtest.log"


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
