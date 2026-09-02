"""FX instrument metadata, pip-value, and margin helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Any


def normalize_instrument_name(name: str) -> str:
    raw = name.strip().upper()
    if "_" in raw:
        parts = raw.split("_")
        if len(parts) == 2 and all(parts):
            return f"{parts[0]}_{parts[1]}"
    letters = "".join(ch for ch in raw if ch.isalpha())
    if len(letters) >= 6:
        return f"{letters[:3]}_{letters[3:6]}"
    raise ValueError(f"FX instrument names must look like EUR_USD or EURUSD, got {name!r}")


def split_instrument_name(name: str) -> tuple[str, str]:
    parts = normalize_instrument_name(name).split("_")
    return parts[0], parts[1]


@dataclass(frozen=True)
class FxInstrument:
    name: str
    pip_location: int = -4
    display_precision: int = 5
    trade_units_precision: int = 0
    trade_unit_step: float | None = None
    margin_rate: float = 0.0333333333
    minimum_trade_size: float = 1.0
    maximum_order_units: float = 100_000_000.0
    minimum_stop_distance: float = 0.0
    contract_size: float = 1.0
    volume_min: float | None = None
    volume_max: float | None = None
    volume_step: float | None = None
    broker_symbol: str | None = None
    financing_long_rate: float | None = None
    financing_short_rate: float | None = None
    financing_days: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    @property
    def base_currency(self) -> str:
        return split_instrument_name(self.name)[0]

    @property
    def quote_currency(self) -> str:
        return split_instrument_name(self.name)[1]

    @property
    def pip_size(self) -> float:
        return 10.0 ** self.pip_location

    @property
    def unit_step(self) -> float:
        if self.trade_unit_step and self.trade_unit_step > 0:
            return self.trade_unit_step
        return 10.0 ** (-self.trade_units_precision)

    @classmethod
    def from_mt5(
        cls,
        strategy_name: str,
        symbol_info: Any,
        *,
        account_leverage: float | None = None,
        broker_symbol: str | None = None,
    ) -> "FxInstrument":
        payload = _as_dict(symbol_info)
        name = normalize_instrument_name(strategy_name)
        mt5_symbol = str(payload.get("name") or broker_symbol or strategy_name)
        digits = int(payload.get("digits") or 5)
        point = _positive_float(payload.get("point"), 10.0 ** (-digits))
        contract_size = _positive_float(payload.get("trade_contract_size"), 100_000.0)
        volume_min = _positive_float(payload.get("volume_min"), 0.01)
        volume_max = _positive_float(payload.get("volume_max"), 100.0)
        volume_step = _positive_float(payload.get("volume_step"), volume_min)
        unit_step = max(volume_step * contract_size, 1.0)
        return cls(
            name=name,
            pip_location=_default_pip_location(name),
            display_precision=digits,
            trade_units_precision=0,
            trade_unit_step=unit_step,
            margin_rate=_margin_rate(payload, account_leverage),
            minimum_trade_size=max(volume_min * contract_size, 1.0),
            maximum_order_units=max(volume_max * contract_size, unit_step),
            minimum_stop_distance=max(0.0, float(payload.get("trade_stops_level") or 0.0) * point),
            contract_size=contract_size,
            volume_min=volume_min,
            volume_max=volume_max,
            volume_step=volume_step,
            broker_symbol=mt5_symbol,
        )

    def round_price(self, price: float) -> float:
        return round(float(price), self.display_precision)

    def round_units(self, units: float) -> float:
        if self.trade_unit_step is None and self.trade_units_precision == 0:
            return math.floor(float(units) + 1e-9)
        value = Decimal(str(units))
        step = Decimal(str(self.unit_step))
        rounded = (value / step).to_integral_value(rounding=ROUND_DOWN) * step
        return float(rounded)

    def units_to_volume(self, units: float) -> float:
        if self.contract_size <= 0:
            raise ValueError(f"{self.name} has invalid MT5 contract size")
        raw_volume = Decimal(str(abs(units))) / Decimal(str(self.contract_size))
        if self.volume_step and self.volume_step > 0:
            step = Decimal(str(self.volume_step))
            raw_volume = (raw_volume / step).to_integral_value(rounding=ROUND_DOWN) * step
        return float(raw_volume)


@dataclass(frozen=True)
class PriceSnapshot:
    instrument: str
    bid: float
    ask: float
    time: datetime
    quote_to_home_factor: float | None = None

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    def spread_pips(self, instrument: FxInstrument) -> float:
        return (self.ask - self.bid) / instrument.pip_size

    @classmethod
    def from_mt5(
        cls,
        instrument: str,
        tick: Any,
        *,
        quote_to_home_factor: float | None = None,
    ) -> "PriceSnapshot":
        payload = _as_dict(tick)
        bid = _positive_float(payload.get("bid"), 0.0)
        ask = _positive_float(payload.get("ask"), 0.0)
        if bid <= 0 or ask <= 0:
            raise ValueError(f"MT5 tick has no usable bid/ask: {payload}")
        return cls(
            instrument=normalize_instrument_name(instrument),
            bid=bid,
            ask=ask,
            time=_parse_mt5_time(payload),
            quote_to_home_factor=quote_to_home_factor,
        )


def quote_to_home_factor(
    instrument: FxInstrument,
    account_currency: str,
    price: float,
    conversion_rates: dict[str, float] | None = None,
    snapshot_factor: float | None = None,
) -> float:
    """Return the factor for converting quote-currency P/L into home currency."""

    account = account_currency.upper()
    if snapshot_factor and snapshot_factor > 0:
        return snapshot_factor
    if instrument.quote_currency == account:
        return 1.0
    if instrument.base_currency == account:
        if price <= 0:
            raise ValueError("price must be positive when account currency is the base currency")
        return 1.0 / price
    conversion_rates = conversion_rates or {}
    direct = conversion_rates.get(instrument.quote_currency)
    if direct and direct > 0:
        return direct
    raise ValueError(
        f"missing conversion rate from {instrument.quote_currency} to {account}; "
        "add a direct/reverse account-currency symbol to MT5 or MT5_SYMBOL_MAP"
    )


def pip_value_per_unit_home(
    instrument: FxInstrument,
    account_currency: str,
    price: float,
    conversion_rates: dict[str, float] | None = None,
    snapshot_factor: float | None = None,
) -> float:
    return instrument.pip_size * quote_to_home_factor(
        instrument,
        account_currency,
        price,
        conversion_rates=conversion_rates,
        snapshot_factor=snapshot_factor,
    )


def position_value_home(
    instrument: FxInstrument,
    units: float,
    price: float,
    account_currency: str,
    conversion_rates: dict[str, float] | None = None,
    snapshot_factor: float | None = None,
) -> float:
    quote_value = abs(units) * price
    return quote_value * quote_to_home_factor(
        instrument,
        account_currency,
        price,
        conversion_rates=conversion_rates,
        snapshot_factor=snapshot_factor,
    )


def margin_required_home(
    instrument: FxInstrument,
    units: float,
    price: float,
    account_currency: str,
    conversion_rates: dict[str, float] | None = None,
    snapshot_factor: float | None = None,
) -> float:
    return position_value_home(
        instrument,
        units,
        price,
        account_currency,
        conversion_rates=conversion_rates,
        snapshot_factor=snapshot_factor,
    ) * instrument.margin_rate


def estimated_daily_financing_home(
    instrument: FxInstrument,
    *,
    side: str,
    units: float,
    price: float,
    account_currency: str,
    timestamp: datetime,
    conversion_rates: dict[str, float] | None = None,
    snapshot_factor: float | None = None,
) -> float:
    """Estimate one rollover financing charge in account currency when rates exist."""

    raw_rate = instrument.financing_long_rate if side.upper() == "LONG" else instrument.financing_short_rate
    if raw_rate is None:
        return 0.0
    annual_rate = float(raw_rate)
    notional = position_value_home(
        instrument,
        units,
        price,
        account_currency,
        conversion_rates=conversion_rates,
        snapshot_factor=snapshot_factor,
    )
    return notional * annual_rate * _financing_days_charged(instrument, timestamp) / 365.0


def pips_between(instrument: FxInstrument, first: float, second: float) -> float:
    return abs(first - second) / instrument.pip_size


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _financing_days_charged(instrument: FxInstrument, timestamp: datetime) -> float:
    weekday = timestamp.strftime("%A").upper()
    for item in instrument.financing_days:
        day = str(item.get("dayOfWeek") or item.get("day") or "").upper()
        if day == weekday:
            try:
                return float(item.get("daysCharged") or 1.0)
            except (TypeError, ValueError):
                return 1.0
    return 1.0


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "_asdict"):
        return value._asdict()
    return {
        name: getattr(value, name)
        for name in dir(value)
        if not name.startswith("_") and not callable(getattr(value, name))
    }


def _positive_float(value: Any, default: float) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    return converted if math.isfinite(converted) and converted > 0 else default


def _default_pip_location(name: str) -> int:
    _, quote = split_instrument_name(name)
    return -2 if quote == "JPY" else -4


def _margin_rate(payload: dict[str, Any], account_leverage: float | None) -> float:
    leverage = account_leverage or 0.0
    if leverage > 0:
        return 1.0 / leverage
    margin_initial = _optional_float(payload.get("margin_initial"))
    contract_size = _positive_float(payload.get("trade_contract_size"), 100_000.0)
    if margin_initial and contract_size > 0:
        return max(margin_initial / contract_size, 0.0)
    return 0.0333333333


def _parse_mt5_time(payload: dict[str, Any]) -> datetime:
    timestamp_msc = payload.get("time_msc")
    try:
        if timestamp_msc not in (None, "", 0):
            return datetime.fromtimestamp(float(timestamp_msc) / 1000.0, tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        pass
    timestamp = payload.get("time") or 0
    try:
        return datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return datetime.now(timezone.utc)
