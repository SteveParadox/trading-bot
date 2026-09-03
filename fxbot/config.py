"""Environment-driven configuration for the FX forward-testing platform."""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

from fxbot.instruments import normalize_instrument_name

load_dotenv()

DEFAULT_INSTRUMENTS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _get_csv(name: str, default: Iterable[str]) -> list[str]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return [item.upper() for item in default]
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def _get_instruments(name: str, default: Iterable[str]) -> list[str]:
    return [normalize_instrument_name(item) for item in _get_csv(name, default)]


def _get_optional_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    return int(raw)


def _get_symbol_map(name: str) -> dict[str, str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return {}
    pairs: dict[str, str] = {}
    for item in raw.split(","):
        chunk = item.strip()
        if not chunk:
            continue
        if "=" in chunk:
            strategy_name, broker_symbol = chunk.split("=", 1)
        elif ":" in chunk:
            strategy_name, broker_symbol = chunk.split(":", 1)
        else:
            raise ValueError(f"{name} entries must look like EUR_USD=EURUSD.a")
        pairs[normalize_instrument_name(strategy_name)] = broker_symbol.strip()
    return pairs


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _news_event_from_dict(payload: dict[str, object]) -> "NewsEvent":
    name = str(payload.get("name") or "").strip()
    currency = str(payload.get("currency") or "").strip().upper()
    impact = str(payload.get("impact") or "").strip().lower()
    starts_at = str(payload.get("starts_at") or payload.get("start") or "").strip()
    ends_at = str(payload.get("ends_at") or payload.get("end") or starts_at).strip()
    if not name or not currency or not starts_at:
        raise ValueError(f"invalid news event payload: {payload!r}")
    return NewsEvent(
        name=name,
        currency=currency,
        impact=impact or "high",
        starts_at=_parse_datetime(starts_at),
        ends_at=_parse_datetime(ends_at),
    )


def load_news_events() -> list["NewsEvent"]:
    """Load optional high-impact news blackout events from env JSON or a file."""

    raw = os.getenv("FX_NEWS_EVENTS_JSON", "").strip()
    path = os.getenv("FX_NEWS_EVENTS_FILE", "").strip()
    if raw:
        payload = json.loads(raw)
    elif path:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    else:
        return []
    if not isinstance(payload, list):
        raise ValueError("FX news events must be a JSON list")
    return [_news_event_from_dict(item) for item in payload if isinstance(item, dict)]


@dataclass(frozen=True)
class BrokerSettings:
    login: int | None = None
    password: str = ""
    server: str = ""
    terminal_path: str = ""
    portable: bool = False
    timeout_ms: int = 60_000
    demo_only: bool = True
    deviation_points: int = 20
    magic_number: int = 260828
    order_filling: str = "RETURN"
    symbol_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timeout_ms <= 0:
            raise ValueError("broker.timeout_ms must be positive")
        if self.deviation_points < 0:
            raise ValueError("broker.deviation_points cannot be negative")
        if self.magic_number < 0:
            raise ValueError("broker.magic_number cannot be negative")
        filling = self.order_filling.upper()
        if filling not in {"RETURN", "IOC", "FOK"}:
            raise ValueError("broker.order_filling must be RETURN, IOC, or FOK")
        object.__setattr__(self, "order_filling", filling)
        object.__setattr__(
            self,
            "symbol_map",
            {normalize_instrument_name(key): value.strip() for key, value in self.symbol_map.items() if value.strip()},
        )

    @property
    def provider(self) -> str:
        return "mt5"

    @property
    def configured(self) -> bool:
        return bool(self.terminal_path or self.login or self.server)

    def broker_symbol_for(self, instrument: str) -> str:
        name = normalize_instrument_name(instrument)
        return self.symbol_map.get(name, name.replace("_", ""))

    def strategy_symbol_for(self, broker_symbol: str) -> str:
        reverse = {value.upper(): key for key, value in self.symbol_map.items()}
        return reverse.get(broker_symbol.upper(), normalize_instrument_name(broker_symbol))


@dataclass(frozen=True)
class StrategySettings:
    entry_timeframe: str = "15m"
    htf_timeframe: str = "1h"
    candle_limit: int = 180
    adx_min: float = 10.0
    htf_adx_min: float = 10.0
    min_risk_reward: float = 1.40
    stop_mode: str = "atr"
    atr_sl_multiplier: float = 1.8
    trailing_atr_multiplier: float = 1.4
    breakeven_buffer_pips: float = 0.2
    partial_tp_enabled: bool = True
    tp1_units_pct: float = 0.50
    tp2_multiplier: float = 2.0
    max_spread_pips: float = 2.2
    max_entry_deviation_pips: float = 1.5
    min_atr_pips: float = 2.0
    max_atr_pips: float = 28.0
    max_entry_extension_atr: float | None = None
    min_stop_pips: float = 4.0
    max_stop_pips: float = 40.0
    require_volume_confirmation: bool = False
    volume_ratio_min: float = 0.8
    htf_require_momentum_candle: bool = False
    score_adx_ceiling: float = 40.0
    score_di_edge_ceiling: float = 25.0
    score_volume_ratio_floor: float = 0.8
    score_volume_ratio_ceiling: float = 2.0
    trade_sessions_utc: tuple[str, ...] = ("london", "new_york", "overlap")
    avoid_rollover_minutes: int = 15
    close_before_weekend_minutes: int = 60
    news_blackout_before_minutes: int = 30
    news_blackout_after_minutes: int = 30

    def __post_init__(self) -> None:
        if self.stop_mode not in {"atr", "ma"}:
            raise ValueError("strategy.stop_mode must be 'atr' or 'ma'")
        if self.entry_timeframe not in {"5m", "15m", "30m", "1h"}:
            raise ValueError("unsupported FX entry timeframe")
        if self.htf_timeframe not in {"1h", "4h", "1d"}:
            raise ValueError("unsupported FX HTF timeframe")
        if self.min_stop_pips > self.max_stop_pips:
            raise ValueError("strategy.min_stop_pips cannot exceed max_stop_pips")
        if self.min_atr_pips > self.max_atr_pips:
            raise ValueError("strategy.min_atr_pips cannot exceed max_atr_pips")
        if self.max_entry_extension_atr is not None and self.max_entry_extension_atr <= 0:
            raise ValueError("strategy.max_entry_extension_atr must be positive when set")
        if not 0 < self.tp1_units_pct < 1:
            raise ValueError("strategy.tp1_units_pct must leave units for TP1 and TP2")
        if self.score_adx_ceiling <= self.adx_min:
            raise ValueError("strategy.score_adx_ceiling must be greater than adx_min")
        if self.score_di_edge_ceiling <= 0:
            raise ValueError("strategy.score_di_edge_ceiling must be positive")
        if self.score_volume_ratio_floor < 0:
            raise ValueError("strategy.score_volume_ratio_floor cannot be negative")
        if self.score_volume_ratio_ceiling <= self.score_volume_ratio_floor:
            raise ValueError("strategy.score_volume_ratio_ceiling must exceed score_volume_ratio_floor")


@dataclass(frozen=True)
class RiskSettings:
    account_currency: str = "USD"
    min_balance: float = 100.0
    risk_per_trade_pct: float = 0.0025
    max_daily_loss_pct: float = 0.02
    max_drawdown_pct: float = 0.10
    max_open_positions: int = 3
    max_portfolio_risk_pct: float = 0.03
    max_pair_exposure_pct: float = 1.50
    max_gross_exposure_pct: float = 4.00
    max_currency_exposure_pct: float = 2.00
    min_free_margin_pct: float = 0.20
    max_units_per_trade: float = 100_000.0
    emergency_close_on_protection_failure: bool = True

    def __post_init__(self) -> None:
        if self.account_currency.upper() != self.account_currency:
            object.__setattr__(self, "account_currency", self.account_currency.upper())
        bounded_percentages = (
            "risk_per_trade_pct",
            "max_daily_loss_pct",
            "max_drawdown_pct",
            "max_portfolio_risk_pct",
            "min_free_margin_pct",
        )
        exposure_multiples = (
            "max_pair_exposure_pct",
            "max_gross_exposure_pct",
            "max_currency_exposure_pct",
        )
        for name in bounded_percentages:
            value = getattr(self, name)
            if value < 0 or value > 1.5:
                raise ValueError(f"risk.{name} is outside a sane percentage range")
        for name in exposure_multiples:
            value = getattr(self, name)
            if value < 0 or value > 50:
                raise ValueError(f"risk.{name} is outside a sane exposure multiple range")


@dataclass(frozen=True)
class RuntimeSettings:
    database_url: str = "sqlite:///./data/fx_forward_test.db"
    loop_interval_seconds: int = 60
    api_key: str = "change-me-demo-key"
    log_jsonl_path: str = "data/fx_journal.jsonl"
    frontend_origin: str = "http://127.0.0.1:5173"
    start_worker_with_api: bool = True


@dataclass(frozen=True)
class NewsEvent:
    name: str
    currency: str
    impact: str
    starts_at: datetime
    ends_at: datetime


@dataclass(frozen=True)
class FxBotSettings:
    instruments: list[str] = field(default_factory=lambda: DEFAULT_INSTRUMENTS.copy())
    broker: BrokerSettings = field(default_factory=BrokerSettings)
    strategy: StrategySettings = field(default_factory=StrategySettings)
    risk: RiskSettings = field(default_factory=RiskSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    news_events: list[NewsEvent] = field(default_factory=list)


def settings_from_env() -> FxBotSettings:
    return FxBotSettings(
        instruments=_get_instruments("FX_INSTRUMENTS", DEFAULT_INSTRUMENTS),
        broker=BrokerSettings(
            login=_get_optional_int("MT5_LOGIN"),
            password=os.getenv("MT5_PASSWORD", ""),
            server=os.getenv("MT5_SERVER", ""),
            terminal_path=os.getenv("MT5_TERMINAL_PATH", ""),
            portable=_get_bool("MT5_PORTABLE", False),
            timeout_ms=_get_int("MT5_TIMEOUT_MS", 60_000),
            demo_only=_get_bool("MT5_DEMO_ONLY", True),
            deviation_points=_get_int("MT5_DEVIATION_POINTS", 20),
            magic_number=_get_int("MT5_MAGIC_NUMBER", 260828),
            order_filling=os.getenv("MT5_ORDER_FILLING", "RETURN"),
            symbol_map=_get_symbol_map("MT5_SYMBOL_MAP"),
        ),
        strategy=StrategySettings(
            entry_timeframe=os.getenv("FX_ENTRY_TIMEFRAME", "15m"),
            htf_timeframe=os.getenv("FX_HTF_TIMEFRAME", "1h"),
            candle_limit=_get_int("FX_CANDLE_LIMIT", 180),
            adx_min=_get_float("FX_ADX_MIN", 10.0),
            htf_adx_min=_get_float("FX_HTF_ADX_MIN", 10.0),
            min_risk_reward=_get_float("FX_MIN_RISK_REWARD", 1.40),
            stop_mode=os.getenv("FX_STOP_MODE", "atr").strip().lower(),
            atr_sl_multiplier=_get_float("FX_ATR_SL_MULTIPLIER", 1.8),
            trailing_atr_multiplier=_get_float("FX_TRAILING_ATR_MULTIPLIER", 1.4),
            breakeven_buffer_pips=_get_float("FX_BREAKEVEN_BUFFER_PIPS", 0.2),
            partial_tp_enabled=_get_bool("FX_PARTIAL_TP_ENABLED", True),
            tp1_units_pct=_get_float("FX_TP1_UNITS_PCT", 0.50),
            tp2_multiplier=_get_float("FX_TP2_MULTIPLIER", 2.0),
            max_spread_pips=_get_float("FX_MAX_SPREAD_PIPS", 2.2),
            max_entry_deviation_pips=_get_float("FX_MAX_ENTRY_DEVIATION_PIPS", 1.5),
            min_atr_pips=_get_float("FX_MIN_ATR_PIPS", 2.0),
            max_atr_pips=_get_float("FX_MAX_ATR_PIPS", 28.0),
            max_entry_extension_atr=(
                _get_float("FX_MAX_ENTRY_EXTENSION_ATR", 0.0)
                if os.getenv("FX_MAX_ENTRY_EXTENSION_ATR", "").strip()
                else None
            ),
            min_stop_pips=_get_float("FX_MIN_STOP_PIPS", 4.0),
            max_stop_pips=_get_float("FX_MAX_STOP_PIPS", 40.0),
            require_volume_confirmation=_get_bool("FX_REQUIRE_VOLUME_CONFIRMATION", False),
            volume_ratio_min=_get_float("FX_VOLUME_RATIO_MIN", 0.80),
            htf_require_momentum_candle=_get_bool("FX_HTF_REQUIRE_MOMENTUM_CANDLE", False),
            score_adx_ceiling=_get_float("FX_SCORE_ADX_CEILING", 40.0),
            score_di_edge_ceiling=_get_float("FX_SCORE_DI_EDGE_CEILING", 25.0),
            score_volume_ratio_floor=_get_float("FX_SCORE_VOLUME_RATIO_FLOOR", 0.80),
            score_volume_ratio_ceiling=_get_float("FX_SCORE_VOLUME_RATIO_CEILING", 2.0),
            trade_sessions_utc=tuple(
                item.lower() for item in _get_csv("FX_TRADE_SESSIONS_UTC", ["london", "new_york", "overlap"])
            ),
            avoid_rollover_minutes=_get_int("FX_AVOID_ROLLOVER_MINUTES", 15),
            close_before_weekend_minutes=_get_int("FX_CLOSE_BEFORE_WEEKEND_MINUTES", 60),
            news_blackout_before_minutes=_get_int("FX_NEWS_BLACKOUT_BEFORE_MINUTES", 30),
            news_blackout_after_minutes=_get_int("FX_NEWS_BLACKOUT_AFTER_MINUTES", 30),
        ),
        risk=RiskSettings(
            account_currency=os.getenv("FX_ACCOUNT_CURRENCY", "USD").upper(),
            min_balance=_get_float("FX_MIN_BALANCE", 100.0),
            risk_per_trade_pct=_get_float("FX_RISK_PER_TRADE_PCT", 0.0025),
            max_daily_loss_pct=_get_float("FX_MAX_DAILY_LOSS_PCT", 0.02),
            max_drawdown_pct=_get_float("FX_MAX_DRAWDOWN_PCT", 0.10),
            max_open_positions=_get_int("FX_MAX_OPEN_POSITIONS", 3),
            max_portfolio_risk_pct=_get_float("FX_MAX_PORTFOLIO_RISK_PCT", 0.03),
            max_pair_exposure_pct=_get_float("FX_MAX_PAIR_EXPOSURE_PCT", 0.35),
            max_gross_exposure_pct=_get_float("FX_MAX_GROSS_EXPOSURE_PCT", 1.20),
            max_currency_exposure_pct=_get_float("FX_MAX_CURRENCY_EXPOSURE_PCT", 0.70),
            min_free_margin_pct=_get_float("FX_MIN_FREE_MARGIN_PCT", 0.20),
            max_units_per_trade=_get_float("FX_MAX_UNITS_PER_TRADE", 100_000.0),
            emergency_close_on_protection_failure=_get_bool("FX_EMERGENCY_CLOSE_ON_PROTECTION_FAILURE", True),
        ),
        runtime=RuntimeSettings(
            database_url=os.getenv("FX_DATABASE_URL", "sqlite:///./data/fx_forward_test.db"),
            loop_interval_seconds=_get_int("FX_LOOP_INTERVAL_SECONDS", 60),
            api_key=os.getenv("FX_API_KEY", "change-me-demo-key"),
            log_jsonl_path=os.getenv("FX_JSONL_JOURNAL", "data/fx_journal.jsonl"),
            frontend_origin=os.getenv("FX_FRONTEND_ORIGIN", "http://127.0.0.1:5173"),
            start_worker_with_api=_get_bool("FX_START_WORKER_WITH_API", True),
        ),
        news_events=load_news_events(),
    )


def ensure_runtime_dirs(settings: FxBotSettings) -> None:
    if settings.runtime.database_url.startswith("sqlite:///"):
        db_path = settings.runtime.database_url.replace("sqlite:///", "", 1)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(settings.runtime.log_jsonl_path).parent.mkdir(parents=True, exist_ok=True)
