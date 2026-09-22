"""Environment-driven configuration for the FX forward-testing platform."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

from fxbot.instruments import normalize_instrument_name
from fxbot.sniper import SniperSettings, settings_from_env as sniper_settings_from_env

load_dotenv()

DEFAULT_INSTRUMENTS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]


def _get_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip()


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid float, got {raw!r}") from exc


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid integer, got {raw!r}") from exc


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
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid integer, got {raw!r}") from exc


def _get_optional_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid float, got {raw!r}") from exc


def _get_optional_float_with_default(name: str, default: float) -> float | None:
    """Read a float with explicit ``off``/``none`` support for safety filters."""

    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    if raw.strip().lower() in {"off", "none", "disabled"}:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid float or 'off', got {raw!r}") from exc


def _get_symbol_map(name: str) -> dict[str, str]:
    raw = _get_str(name, "")
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


def _get_mapping(name: str, default: dict[str, str] | None = None) -> dict[str, str]:
    """Read a small comma-separated ``key=value`` mapping from the environment."""

    raw = _get_str(name, "")
    if not raw:
        return dict(default or {})
    result: dict[str, str] = {}
    for item in raw.split(","):
        chunk = item.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"{name} entries must look like key=value")
        key, value = (part.strip() for part in chunk.split("=", 1))
        if not key or not value:
            raise ValueError(f"{name} entries must contain a non-empty key and value")
        result[key.lower()] = value.lower()
    return result


_DATETIME_12H_FORMATS = ("%Y-%m-%d %I:%M%p", "%Y-%m-%d %I%p")


def _parse_datetime(value: str) -> datetime:
    text = value.replace("Z", "+00:00").strip()
    try:
        return datetime.fromisoformat(text)
    except (ValueError, TypeError):
        pass
    # Forex Factory community feed uses 12-hour timestamps like
    # "2026-09-16 12:30am". Try those before giving up.
    for fmt in _DATETIME_12H_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except (ValueError, TypeError):
            continue
    raise ValueError(f"unrecognized datetime format: {value!r}")


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _optional_float(value: object | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _news_event_from_dict(payload: dict[str, object]) -> "NewsEvent":
    name = str(payload.get("name") or payload.get("event_name") or "").strip()
    currency = str(payload.get("currency") or "").strip().upper()
    impact = str(payload.get("impact") or payload.get("importance") or "").strip().lower()
    country = str(payload.get("country") or "").strip()
    starts_at_raw = str(payload.get("starts_at") or payload.get("start") or "").strip()
    ends_at_raw = str(payload.get("ends_at") or payload.get("end") or starts_at_raw).strip()
    if not name or not currency or not starts_at_raw:
        raise ValueError(f"invalid news event payload: {payload!r}")
    now = datetime.now(timezone.utc)
    starts_at = _as_utc(_parse_datetime(starts_at_raw))
    ends_at = _as_utc(_parse_datetime(ends_at_raw))
    if ends_at < starts_at:
        raise ValueError(f"news event ends before it starts: {payload!r}")
    event_id = str(payload.get("event_id") or "").strip() or None
    impact_score_raw = _optional_float(payload.get("impact_score"))
    impact_score = max(0, min(100, int(round(impact_score_raw)))) if impact_score_raw is not None else 0
    confidence_raw = _optional_float(payload.get("confidence"))
    confidence = None if confidence_raw is None else max(0.0, min(1.0, confidence_raw))
    return NewsEvent(
        name=name,
        currency=currency,
        impact=impact or "low",
        starts_at=starts_at,
        ends_at=ends_at,
        event_id=event_id,
        country=country,
        previous=_optional_float(payload.get("previous")),
        forecast=_optional_float(payload.get("forecast")),
        actual=_optional_float(payload.get("actual")),
        source=str(payload.get("source") or "manual").strip() or "manual",
        created_at=created_at if (created_at := _parse_optional_datetime(payload.get("created_at"))) is not None else now,
        updated_at=_parse_optional_datetime(payload.get("updated_at")) or now,
        impact_score=impact_score,
        description=str(payload.get("description") or payload.get("desc") or "").strip(),
        source_url=str(payload.get("source_url") or payload.get("url") or "").strip(),
        status=str(payload.get("status") or "scheduled").strip().lower() or "scheduled",
        confidence=confidence,
    )


def _parse_optional_datetime(value: object | None) -> datetime | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return _as_utc(_parse_datetime(str(value)))
    except ValueError:
        return None


def load_news_events() -> list["NewsEvent"]:
    """Load optional high-impact news blackout events from env JSON or a file."""

    raw = _get_str("FX_NEWS_EVENTS_JSON", "")
    path = _get_str("FX_NEWS_EVENTS_FILE", "")
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
    adx_min: float = 18.0
    htf_adx_min: float = 20.0
    min_risk_reward: float = 1.50
    stop_mode: str = "atr"
    atr_sl_multiplier: float = 1.8
    trailing_atr_multiplier: float = 1.4
    runner_take_profit_r: float | None = None
    breakeven_buffer_pips: float = 0.2
    partial_tp_enabled: bool = True
    tp1_units_pct: float = 0.50
    max_spread_pips: float = 3.0
    max_spread_atr_ratio: float = 0.35
    max_entry_deviation_pips: float = 1.5
    min_atr_pct: float = 0.00015
    max_atr_pct: float = 0.02
    min_atr_pips: float | None = None
    max_atr_pips: float | None = None
    # A trend can remain valid after price is already extended. This cap avoids
    # entering the late, statistically less attractive portion of that move.
    max_entry_extension_atr: float | None = 0.80
    min_stop_pips: float | None = None
    max_stop_pips: float | None = None
    min_stop_atr_multiple: float = 0.75
    max_stop_atr_multiple: float = 4.0
    require_volume_confirmation: bool = False
    volume_ratio_min: float = 0.8
    htf_require_momentum_candle: bool = True
    min_signal_score: float = 60.0
    min_di_edge: float = 5.0
    require_adx_non_decreasing: bool = True
    require_ma28_slope: bool = True
    ma28_slope_lookback: int = 3
    min_ma28_slope_atr: float = 0.10
    score_adx_ceiling: float = 40.0
    score_di_edge_ceiling: float = 25.0
    score_volume_ratio_floor: float = 0.8
    score_volume_ratio_ceiling: float = 2.0
    # Close-location strength is a directional candle-quality gate. A long
    # entry must close in the upper portion of its completed candle (and a
    # short in the lower portion) when this is enabled.
    min_entry_close_strength: float = 0.0
    # Tiny targets are only viable when their gross reward comfortably exceeds
    # the current bid/ask spread. Zero keeps the legacy behavior.
    min_reward_to_spread_ratio: float = 0.0
    trade_sessions_utc: tuple[str, ...] = ("london", "new_york", "overlap")
    avoid_rollover_minutes: int = 15
    close_before_weekend_minutes: int = 60
    news_blackout_before_minutes: int = 30
    news_blackout_after_minutes: int = 30
    news_blackout_impact_score_min: int = 71
    require_news_data: bool = False
    news_medium_impact_enabled: bool = False
    news_restricted_currencies: tuple[str, ...] = ()
    news_restricted_instruments: tuple[str, ...] = ()
    news_event_overrides: dict[str, str] = field(default_factory=dict)
    news_manual_override: str = "none"
    news_emergency_kill_switch: bool = False
    news_risk_action: str = "protect_and_block"
    news_data_max_age_seconds: int = 3600
    news_sync_interval_seconds: int = 300
    news_events_file: str = ""
    news_api_endpoint: str = ""
    news_api_key: str = ""
    news_http_timeout_seconds: int = 30
    news_use_forex_factory: bool = False

    def __post_init__(self) -> None:
        if self.stop_mode not in {"atr", "ma"}:
            raise ValueError("strategy.stop_mode must be 'atr' or 'ma'")
        if self.entry_timeframe not in {"5m", "15m", "30m", "1h"}:
            raise ValueError("unsupported FX entry timeframe")
        if self.htf_timeframe not in {"1h", "4h", "1d"}:
            raise ValueError("unsupported FX HTF timeframe")
        if self.min_stop_pips is not None and self.max_stop_pips is not None and self.min_stop_pips > self.max_stop_pips:
            raise ValueError("strategy.min_stop_pips cannot exceed max_stop_pips")
        if self.min_atr_pips is not None and self.max_atr_pips is not None and self.min_atr_pips > self.max_atr_pips:
            raise ValueError("strategy.min_atr_pips cannot exceed max_atr_pips")
        if self.min_atr_pct <= 0 or self.max_atr_pct <= self.min_atr_pct:
            raise ValueError("strategy ATR percentage bounds are invalid")
        if self.max_spread_atr_ratio <= 0:
            raise ValueError("strategy.max_spread_atr_ratio must be positive")
        if self.atr_sl_multiplier <= 0 or self.trailing_atr_multiplier <= 0:
            raise ValueError("strategy ATR multipliers must be positive")
        if self.min_stop_atr_multiple <= 0 or self.max_stop_atr_multiple < self.min_stop_atr_multiple:
            raise ValueError("strategy stop ATR bounds are invalid")
        if self.runner_take_profit_r is not None and self.runner_take_profit_r <= 0:
            raise ValueError("strategy.runner_take_profit_r must be positive when set")
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
        if not 0 <= self.min_entry_close_strength <= 1:
            raise ValueError("strategy.min_entry_close_strength must be between 0 and 1")
        if self.min_reward_to_spread_ratio < 0:
            raise ValueError("strategy.min_reward_to_spread_ratio cannot be negative")
        if not 0 <= self.min_signal_score <= 100:
            raise ValueError("strategy.min_signal_score must be between 0 and 100")
        if self.min_di_edge < 0:
            raise ValueError("strategy.min_di_edge cannot be negative")
        if self.ma28_slope_lookback < 1:
            raise ValueError("strategy.ma28_slope_lookback must be at least 1")
        if self.min_ma28_slope_atr < 0:
            raise ValueError("strategy.min_ma28_slope_atr cannot be negative")
        if self.news_risk_action not in {"block_entries", "protect_and_block", "close_positions"}:
            raise ValueError("strategy.news_risk_action must be block_entries, protect_and_block, or close_positions")
        if self.news_data_max_age_seconds <= 0:
            raise ValueError("strategy.news_data_max_age_seconds must be positive")
        if self.news_http_timeout_seconds <= 0:
            raise ValueError("strategy.news_http_timeout_seconds must be positive")
        if self.news_sync_interval_seconds <= 0:
            raise ValueError("strategy.news_sync_interval_seconds must be positive")
        if not 0 <= self.news_blackout_impact_score_min <= 100:
            raise ValueError("strategy.news_blackout_impact_score_min must be between 0 and 100")
        if self.news_manual_override not in {"none", "allow", "block"}:
            raise ValueError("strategy.news_manual_override must be none, allow, or block")
        normalized_currencies = tuple(dict.fromkeys(item.strip().upper() for item in self.news_restricted_currencies if item.strip()))
        normalized_instruments = tuple(
            dict.fromkeys(normalize_instrument_name(item) for item in self.news_restricted_instruments if item.strip())
        )
        overrides: dict[str, str] = {}
        for key, value in self.news_event_overrides.items():
            normalized_key = str(key).strip().lower()
            normalized_value = str(value).strip().lower()
            if not normalized_key or normalized_value not in {"allow", "block"}:
                raise ValueError("strategy.news_event_overrides values must be allow or block")
            overrides[normalized_key] = normalized_value
        object.__setattr__(self, "news_restricted_currencies", normalized_currencies)
        object.__setattr__(self, "news_restricted_instruments", normalized_instruments)
        object.__setattr__(self, "news_event_overrides", overrides)


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
    loop_interval_seconds: int = 10
    api_key: str = ""
    log_jsonl_path: str | None = None
    frontend_origin: str = "http://127.0.0.1:5173"
    cors_origins: tuple[str, ...] = ()
    start_worker_with_api: bool = True
    bind_host: str = "127.0.0.1"
    api_port: int = 8000
    api_rate_limit_per_minute: int = 120
    live_trading_enabled: bool = False
    live_release_ack: str = ""
    max_price_age_seconds: int = 120

    @property
    def live_release_approved(self) -> bool:
        return self.live_trading_enabled and self.live_release_ack == "I_UNDERSTAND_LIVE_TRADING_RISK"


@dataclass(frozen=True)
class AiDeliberationSettings:
    """Configuration for the isolated, optional signal-audit service.

    This object deliberately contains no broker, risk, or order settings.  The
    AI service receives an immutable evidence package and returns untrusted
    audit data only.
    """

    mode: str = "off"
    provider: str = "openai_compatible"
    endpoint: str = ""
    api_key: str = ""
    model: str = ""
    timeout_seconds: float = 8.0
    max_output_tokens: int = 1200
    max_retries: int = 1
    prompt_version: str = "v1"
    fail_policy: str = "fail_closed_if_confirmation_required"
    flag_blocks: bool = False
    reject_blocks: bool = False
    minimum_confidence: float = 0.70
    advisory_require_confirmation: bool = False

    def __post_init__(self) -> None:
        mode = self.mode.lower().strip()
        if mode not in {"off", "shadow", "advisory"}:
            raise ValueError("ai.mode must be off, shadow, or advisory")
        if self.provider.lower().strip() not in {"openai_compatible", "none"}:
            raise ValueError("ai.provider must be openai_compatible or none")
        if self.timeout_seconds <= 0:
            raise ValueError("ai.timeout_seconds must be positive")
        if self.max_output_tokens < 64:
            raise ValueError("ai.max_output_tokens must be at least 64")
        if self.max_retries < 0 or self.max_retries > 3:
            raise ValueError("ai.max_retries must be between 0 and 3")
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("ai.minimum_confidence must be between 0 and 1")
        if self.fail_policy not in {"fail_open", "fail_closed_if_confirmation_required"}:
            raise ValueError("ai.fail_policy is not supported")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "provider", self.provider.lower().strip())


@dataclass(frozen=True)
class NewsEvent:
    """Canonical internal news-event schema.

    Required: name, currency, impact, starts_at, ends_at.
    The remaining fields implement the provider-agnostic ingestion schema
    (https spec: event_id, timestamp_utc, currency, country, event_name,
    importance, previous, forecast, actual, source, created_at, updated_at).
    """

    name: str
    currency: str
    impact: str
    starts_at: datetime
    ends_at: datetime
    event_id: str | None = None
    country: str = ""
    previous: float | None = None
    forecast: float | None = None
    actual: float | None = None
    source: str = "manual"
    created_at: datetime | None = None
    updated_at: datetime | None = None
    impact_score: int = 0
    description: str = ""
    source_url: str = ""
    status: str = "scheduled"
    confidence: float | None = None

    @property
    def event_name(self) -> str:
        """Alias for the spec's event_name field."""
        return self.name

    @property
    def importance(self) -> str:
        """Alias for the spec's importance field."""
        return self.impact


@dataclass(frozen=True)
class FxBotSettings:
    instruments: list[str] = field(default_factory=lambda: DEFAULT_INSTRUMENTS.copy())
    broker: BrokerSettings = field(default_factory=BrokerSettings)
    strategy: StrategySettings = field(default_factory=StrategySettings)
    risk: RiskSettings = field(default_factory=RiskSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    ai: AiDeliberationSettings = field(default_factory=AiDeliberationSettings)
    news_events: list[NewsEvent] = field(default_factory=list)
    sniper: SniperSettings = field(default_factory=SniperSettings)

    def __post_init__(self) -> None:
        # Provider precedence must match build_news_gateway. An opt-in FF flag
        # must not invalidate a separately configured HTTP or manual provider.
        uses_free_feed = (
            self.strategy.news_use_forex_factory
            and not self.strategy.news_api_endpoint
            and not self.strategy.news_events_file
        )
        if uses_free_feed:
            if not self.broker.demo_only or self.runtime.live_trading_enabled:
                raise ValueError("Forex Factory free feed is demo/forward-test only; configure another news provider for live trading")
            if not self.strategy.require_news_data:
                raise ValueError("Forex Factory requires FX_REQUIRE_NEWS_DATA=true")


def settings_from_env() -> FxBotSettings:
    demo_only = _get_bool("MT5_DEMO_ONLY", True)
    return FxBotSettings(
        sniper=sniper_settings_from_env(_get_str, _get_bool, _get_float, _get_int, _get_optional_float),
        instruments=_get_instruments("FX_INSTRUMENTS", DEFAULT_INSTRUMENTS),
        broker=BrokerSettings(
            login=_get_optional_int("MT5_LOGIN"),
            password=_get_str("MT5_PASSWORD", ""),
            server=_get_str("MT5_SERVER", ""),
            terminal_path=_get_str("MT5_TERMINAL_PATH", ""),
            portable=_get_bool("MT5_PORTABLE", False),
            timeout_ms=_get_int("MT5_TIMEOUT_MS", 60_000),
            demo_only=demo_only,
            deviation_points=_get_int("MT5_DEVIATION_POINTS", 20),
            magic_number=_get_int("MT5_MAGIC_NUMBER", 260828),
            order_filling=_get_str("MT5_ORDER_FILLING", "RETURN"),
            symbol_map=_get_symbol_map("MT5_SYMBOL_MAP"),
        ),
        strategy=StrategySettings(
            entry_timeframe=_get_str("FX_ENTRY_TIMEFRAME", "15m"),
            htf_timeframe=_get_str("FX_HTF_TIMEFRAME", "1h"),
            candle_limit=_get_int("FX_CANDLE_LIMIT", 180),
            adx_min=_get_float("FX_ADX_MIN", 18.0),
            htf_adx_min=_get_float("FX_HTF_ADX_MIN", 20.0),
            min_risk_reward=_get_float("FX_MIN_RISK_REWARD", 1.50),
            stop_mode=_get_str("FX_STOP_MODE", "atr").lower(),
            atr_sl_multiplier=_get_float("FX_ATR_SL_MULTIPLIER", 1.8),
            trailing_atr_multiplier=_get_float("FX_TRAILING_ATR_MULTIPLIER", 1.4),
            runner_take_profit_r=_get_optional_float("FX_RUNNER_TAKE_PROFIT_R"),
            breakeven_buffer_pips=_get_float("FX_BREAKEVEN_BUFFER_PIPS", 0.2),
            partial_tp_enabled=_get_bool("FX_PARTIAL_TP_ENABLED", True),
            tp1_units_pct=_get_float("FX_TP1_UNITS_PCT", 0.50),
            max_spread_pips=_get_float("FX_MAX_SPREAD_PIPS", 3.0),
            max_spread_atr_ratio=_get_float("FX_MAX_SPREAD_ATR_RATIO", 0.35),
            max_entry_deviation_pips=_get_float("FX_MAX_ENTRY_DEVIATION_PIPS", 1.5),
            min_atr_pct=_get_float("FX_MIN_ATR_PCT", 0.00015),
            max_atr_pct=_get_float("FX_MAX_ATR_PCT", 0.02),
            min_atr_pips=_get_optional_float("FX_MIN_ATR_PIPS"),
            max_atr_pips=_get_optional_float("FX_MAX_ATR_PIPS"),
            max_entry_extension_atr=_get_optional_float_with_default("FX_MAX_ENTRY_EXTENSION_ATR", 0.80),
            min_stop_pips=_get_optional_float("FX_MIN_STOP_PIPS"),
            max_stop_pips=_get_optional_float("FX_MAX_STOP_PIPS"),
            min_stop_atr_multiple=_get_float("FX_MIN_STOP_ATR_MULTIPLE", 0.75),
            max_stop_atr_multiple=_get_float("FX_MAX_STOP_ATR_MULTIPLE", 4.0),
            require_volume_confirmation=_get_bool("FX_REQUIRE_VOLUME_CONFIRMATION", False),
            volume_ratio_min=_get_float("FX_VOLUME_RATIO_MIN", 0.80),
            htf_require_momentum_candle=_get_bool("FX_HTF_REQUIRE_MOMENTUM_CANDLE", True),
            min_signal_score=_get_float("FX_MIN_SIGNAL_SCORE", 60.0),
            min_di_edge=_get_float("FX_MIN_DI_EDGE", 5.0),
            require_adx_non_decreasing=_get_bool("FX_REQUIRE_ADX_NON_DECREASING", True),
            require_ma28_slope=_get_bool("FX_REQUIRE_MA28_SLOPE", True),
            ma28_slope_lookback=_get_int("FX_MA28_SLOPE_LOOKBACK", 3),
            min_ma28_slope_atr=_get_float("FX_MIN_MA28_SLOPE_ATR", 0.10),
            score_adx_ceiling=_get_float("FX_SCORE_ADX_CEILING", 40.0),
            score_di_edge_ceiling=_get_float("FX_SCORE_DI_EDGE_CEILING", 25.0),
            score_volume_ratio_floor=_get_float("FX_SCORE_VOLUME_RATIO_FLOOR", 0.80),
            score_volume_ratio_ceiling=_get_float("FX_SCORE_VOLUME_RATIO_CEILING", 2.0),
            min_entry_close_strength=_get_float("FX_MIN_ENTRY_CLOSE_STRENGTH", 0.0),
            min_reward_to_spread_ratio=_get_float("FX_MIN_REWARD_TO_SPREAD_RATIO", 0.0),
            trade_sessions_utc=tuple(
                item.lower() for item in _get_csv("FX_TRADE_SESSIONS_UTC", ["london", "new_york", "overlap"])
            ),
            avoid_rollover_minutes=_get_int("FX_AVOID_ROLLOVER_MINUTES", 15),
            close_before_weekend_minutes=_get_int("FX_CLOSE_BEFORE_WEEKEND_MINUTES", 60),
            news_blackout_before_minutes=_get_int("FX_NEWS_BLACKOUT_BEFORE_MINUTES", 30),
            news_blackout_after_minutes=_get_int("FX_NEWS_BLACKOUT_AFTER_MINUTES", 30),
            news_blackout_impact_score_min=_get_int("FX_NEWS_BLACKOUT_IMPACT_SCORE_MIN", 71),
            # A live account fails closed by default. Demo/forward testing keeps
            # the historical permissive default unless the operator opts in.
            require_news_data=_get_bool("FX_REQUIRE_NEWS_DATA", not demo_only),
            news_medium_impact_enabled=_get_bool("FX_MEDIUM_IMPACT_NEWS_ENABLED", False),
            news_restricted_currencies=tuple(_get_csv("FX_NEWS_RESTRICTED_CURRENCIES", [])),
            news_restricted_instruments=tuple(_get_instruments("FX_NEWS_RESTRICTED_INSTRUMENTS", [])),
            news_event_overrides=_get_mapping("FX_NEWS_EVENT_OVERRIDES"),
            news_manual_override=_get_str("FX_NEWS_MANUAL_OVERRIDE", "none").lower(),
            news_emergency_kill_switch=_get_bool("FX_NEWS_EMERGENCY_KILL_SWITCH", False),
            news_risk_action=_get_str("FX_NEWS_RISK_ACTION", "protect_and_block").lower(),
            news_data_max_age_seconds=_get_int("FX_NEWS_DATA_MAX_AGE_SECONDS", 3600),
            news_sync_interval_seconds=_get_int("FX_NEWS_SYNC_INTERVAL_SECONDS", 300),
            news_events_file=_get_str("FX_NEWS_EVENTS_FILE", ""),
            news_api_endpoint=_get_str("FX_NEWS_API_ENDPOINT", ""),
            news_api_key=_get_str("FX_NEWS_API_KEY", ""),
            news_http_timeout_seconds=_get_int("FX_NEWS_HTTP_TIMEOUT_SECONDS", 30),
            news_use_forex_factory=_get_bool("FX_USE_FOREX_FACTORY", False),
        ),
        risk=RiskSettings(
            account_currency=_get_str("FX_ACCOUNT_CURRENCY", "USD").upper(),
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
            database_url=_get_str("FX_DATABASE_URL", "sqlite:///./data/fx_forward_test.db"),
            loop_interval_seconds=_get_int("FX_LOOP_INTERVAL_SECONDS", 10),
            api_key=_get_str("FX_API_KEY", ""),
            log_jsonl_path=_get_str("FX_JSONL_JOURNAL", "") or None,
            frontend_origin=_get_str("FX_FRONTEND_ORIGIN", "http://127.0.0.1:5173"),
            cors_origins=tuple(item.strip() for item in _get_str("FX_CORS_ORIGINS", "").split(",") if item.strip()),
            start_worker_with_api=_get_bool("FX_START_WORKER_WITH_API", True),
            bind_host=_get_str("FX_API_HOST", "127.0.0.1"),
            api_port=_get_int("FX_API_PORT", 8000),
            api_rate_limit_per_minute=_get_int("FX_API_RATE_LIMIT_PER_MINUTE", 120),
            live_trading_enabled=_get_bool("FX_LIVE_TRADING_ENABLED", False),
            live_release_ack=_get_str("FX_LIVE_RELEASE_ACK", ""),
            max_price_age_seconds=_get_int("FX_MAX_PRICE_AGE_SECONDS", 120),
        ),
        ai=AiDeliberationSettings(
            mode=_get_str("FX_AI_DELIBERATION", "off"),
            provider=_get_str("FX_AI_PROVIDER", "openai_compatible"),
            endpoint=_get_str("FX_AI_ENDPOINT", ""),
            api_key=_get_str("FX_AI_API_KEY", ""),
            model=_get_str("FX_AI_MODEL", ""),
            timeout_seconds=_get_float("FX_AI_TIMEOUT_SECONDS", 8.0),
            max_output_tokens=_get_int("FX_AI_MAX_OUTPUT_TOKENS", 1200),
            max_retries=_get_int("FX_AI_MAX_RETRIES", 1),
            prompt_version=_get_str("AI_DELIBERATION_PROMPT_VERSION", "v1"),
            fail_policy=_get_str("FX_AI_FAIL_POLICY", "fail_closed_if_confirmation_required"),
            flag_blocks=_get_bool("FX_AI_FLAG_BLOCKS", False),
            reject_blocks=_get_bool("FX_AI_REJECT_BLOCKS", False),
            minimum_confidence=_get_float("FX_AI_MIN_CONFIDENCE", 0.70),
            advisory_require_confirmation=_get_bool("FX_AI_ADVISORY_REQUIRE_CONFIRMATION", False),
        ),
        news_events=load_news_events(),
    )


def ensure_runtime_dirs(settings: FxBotSettings) -> None:
    if settings.runtime.database_url.startswith("sqlite:///"):
        db_path = settings.runtime.database_url.replace("sqlite:///", "", 1)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    if settings.runtime.log_jsonl_path:
        Path(settings.runtime.log_jsonl_path).parent.mkdir(parents=True, exist_ok=True)
