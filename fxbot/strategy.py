"""FX-retuned strategy adapter around the existing indicator calculations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from typing import Any

import pandas as pd

import indicators as legacy_indicators

from fxbot.config import StrategySettings
from fxbot.instruments import FxInstrument, pips_between
from fxbot.models import FxSignalIntent, Side


FX_MA_PERIODS = (7, 14, 28)
FX_MAVOL_FAST = 9
FX_MAVOL_SLOW = 18
FX_ADX_PERIOD = 14
FX_ATR_PERIOD = 14
FX_MIN_SIGNAL_CANDLES = 60
FX_MIN_HTF_CANDLES = 60

TIMEFRAME_DELTAS: dict[str, timedelta] = {
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "1d": timedelta(days=1),
}


@dataclass(frozen=True)
class FxSignalDecision:
    signal: Side | None
    reason: str
    details: dict[str, Any]


@dataclass(frozen=True)
class FxScoreDetails:
    score: float
    adx: float
    di_edge: float
    volume_ma_ratio: float
    current_volume_ratio: float
    atr_pips: float
    entry_extension_atr: float
    distance_ma28_pips: float
    adx_points: float
    di_points: float
    volume_points: float


class CandleFrameError(ValueError):
    def __init__(self, reason: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = details or {}


def prepare_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Return FX-specific indicator data using explicit runtime parameters.

    This keeps the strategy stateless and avoids mutating the global indicator
    module while injecting the live FX configuration.
    """
    normalized = _normalize_market_frame(frame, require_ohlcv=True)
    return legacy_indicators.calculate_indicators(
        normalized,
        ma_periods=FX_MA_PERIODS,
        mavol_fast=FX_MAVOL_FAST,
        mavol_slow=FX_MAVOL_SLOW,
        adx_period=FX_ADX_PERIOD,
        atr_period=FX_ATR_PERIOD,
        volume_ratio_min=1.0,
    ).sort_index()


def evaluate_signal_frame(
    entry_frame: pd.DataFrame,
    htf_frame: pd.DataFrame,
    *,
    instrument: FxInstrument,
    settings: StrategySettings,
    timestamp: datetime | None = None,
) -> FxSignalDecision:
    try:
        entry = _ensure_indicators(entry_frame)
        htf = _ensure_indicators(htf_frame)
        entry_window = last_closed_window(
            entry,
            settings.entry_timeframe,
            timestamp=timestamp,
        )
        htf_window = last_closed_window(
            htf,
            settings.htf_timeframe,
            timestamp=timestamp,
        )
    except CandleFrameError as exc:
        return FxSignalDecision(None, exc.reason, exc.details)

    if len(entry_window) < FX_MIN_SIGNAL_CANDLES or len(htf_window) < FX_MIN_HTF_CANDLES:
        return FxSignalDecision(
            None,
            "insufficient_closed_candles",
            {
                "entry": len(entry_window),
                "htf": len(htf_window),
                "min_entry": FX_MIN_SIGNAL_CANDLES,
                "min_htf": FX_MIN_HTF_CANDLES,
            },
        )

    entry_row = entry_window.iloc[-1]
    htf_row = htf_window.iloc[-1]
    signal = _direction_from_row(entry_row, settings.adx_min, settings.require_volume_confirmation, settings.volume_ratio_min)
    if signal.signal is None:
        return signal

    htf_signal = _htf_direction_from_row(htf_row, settings)
    if htf_signal.signal is None:
        return htf_signal
    if htf_signal.signal is not signal.signal:
        return FxSignalDecision(
            None,
            "htf_conflict",
            {"signal": signal.signal.value, "htf": htf_signal.signal.value, **htf_signal.details},
        )

    atr = float(entry_row.get("atr") or 0.0)
    close = float(entry_row.get("close") or 0.0)
    if atr <= 0 or close <= 0:
        return FxSignalDecision(None, "atr_unavailable", {})
    atr_pips = atr / instrument.pip_size
    if atr_pips < settings.min_atr_pips or atr_pips > settings.max_atr_pips:
        return FxSignalDecision(
            None,
            "atr_pip_filter",
            {"atr_pips": atr_pips, "min": settings.min_atr_pips, "max": settings.max_atr_pips},
        )

    entry_extension_atr = _entry_extension_atr(entry_row, signal.signal)
    ma28 = _safe_float(entry_row.get("ma28"))
    distance_ma28_pips = pips_between(instrument, close, ma28) if close > 0 and ma28 > 0 else 0.0
    if settings.max_entry_extension_atr is not None and entry_extension_atr > settings.max_entry_extension_atr:
        return FxSignalDecision(
            None,
            "entry_extension_filter",
            {
                **signal.details,
                "atr_pips": atr_pips,
                "entry_extension_atr": entry_extension_atr,
                "max_entry_extension_atr": settings.max_entry_extension_atr,
                "signal": signal.signal.value,
            },
        )

    details = {
        **signal.details,
        "htf": htf_signal.details,
        "htf_signal_time": _index_value(htf_row),
        "signal_time": _index_value(entry_row),
        "signal_close": close,
        "atr_price": atr,
        "atr_pips": atr_pips,
        "entry_extension_atr": entry_extension_atr,
        "distance_ma28_pips": distance_ma28_pips,
        "instrument": instrument.name,
    }
    return FxSignalDecision(signal.signal, "signal_confirmed", details)


def build_signal_intent(
    entry_frame: pd.DataFrame,
    htf_frame: pd.DataFrame,
    *,
    instrument: FxInstrument,
    settings: StrategySettings,
    entry_price: float,
    timestamp: datetime | None = None,
    entry_price_source: str = "provided",
) -> FxSignalIntent | None:
    decision = evaluate_signal_frame(
        entry_frame,
        htf_frame,
        instrument=instrument,
        settings=settings,
        timestamp=timestamp,
    )
    if decision.signal is None:
        return None
    try:
        row = last_closed_row(
            _ensure_indicators(entry_frame),
            settings.entry_timeframe,
            timestamp=timestamp,
            min_candles=FX_MIN_SIGNAL_CANDLES,
        )
    except CandleFrameError:
        return None
    if row is None:
        return None
    score_details = _score_signal_details(
        row,
        decision.details,
        settings,
        instrument,
        decision.signal,
    )
    return FxSignalIntent(
        instrument=instrument.name,
        side=decision.signal,
        timestamp=timestamp or datetime.now(timezone.utc),
        entry_price=entry_price,
        signal_row=row.to_dict(),
        score=score_details.score,
        metadata={
            "strategy": "fx_indicator_retune",
            "decision": decision.reason,
            "details": decision.details,
            "score_details": score_details.__dict__,
            "signal_close": decision.details.get("signal_close"),
            "expected_entry_price": entry_price,
            "entry_price_source": entry_price_source,
        },
    )


def _direction_from_row(
    row: pd.Series,
    adx_min: float,
    require_volume: bool,
    volume_ratio_min: float,
) -> FxSignalDecision:
    required = ["ma7", "ma14", "ma28", "adx", "di_plus", "di_minus", "close"]
    if any(column not in row or pd.isna(row[column]) for column in required):
        return FxSignalDecision(None, "indicator_unavailable", {"required": required})

    volume_ma_ratio = _safe_float(row.get("volume_ma_ratio"), _safe_float(row.get("volume_ratio")))
    current_volume_ratio = _safe_float(row.get("current_volume_ratio"))
    details = {
        "ma7": float(row["ma7"]),
        "ma14": float(row["ma14"]),
        "ma28": float(row["ma28"]),
        "adx": float(row["adx"]),
        "di_plus": float(row["di_plus"]),
        "di_minus": float(row["di_minus"]),
        "close": float(row["close"]),
        "volume_ma_ratio": volume_ma_ratio,
        "current_volume_ratio": current_volume_ratio,
        "volume_ratio": volume_ma_ratio,
    }
    if details["adx"] < adx_min:
        return FxSignalDecision(None, "adx_filter", details | {"required_adx": adx_min})

    bull_stack = row["ma7"] > row["ma14"] > row["ma28"]
    bear_stack = row["ma7"] < row["ma14"] < row["ma28"]
    if not bull_stack and not bear_stack:
        return FxSignalDecision(None, "ma_stack_filter", details)

    signal = Side.LONG if bull_stack else Side.SHORT
    if signal is Side.LONG and (row["di_plus"] <= row["di_minus"] or row["close"] <= row["ma7"]):
        return FxSignalDecision(None, "direction_filter", details | {"signal": signal.value})
    if signal is Side.SHORT and (row["di_minus"] <= row["di_plus"] or row["close"] >= row["ma7"]):
        return FxSignalDecision(None, "direction_filter", details | {"signal": signal.value})
    if require_volume and details["volume_ma_ratio"] < volume_ratio_min:
        return FxSignalDecision(None, "volume_filter", details | {"required_volume_ratio": volume_ratio_min})
    return FxSignalDecision(signal, "direction_confirmed", details | {"signal": signal.value})


def _htf_direction_from_row(row: pd.Series, settings: StrategySettings) -> FxSignalDecision:
    decision = _direction_from_row(row, settings.htf_adx_min, False, 0.0)
    if decision.signal is None:
        return FxSignalDecision(None, f"htf_{decision.reason}", decision.details)
    if settings.htf_require_momentum_candle:
        open_price = float(row.get("open") or 0.0)
        close = float(row.get("close") or 0.0)
        if decision.signal is Side.LONG and close <= open_price:
            return FxSignalDecision(None, "htf_momentum_filter", decision.details)
        if decision.signal is Side.SHORT and close >= open_price:
            return FxSignalDecision(None, "htf_momentum_filter", decision.details)
    return FxSignalDecision(decision.signal, "htf_confirmed", decision.details)


def _score_signal_details(
    row: pd.Series,
    details: dict[str, Any],
    settings: StrategySettings,
    instrument: FxInstrument,
    signal: Side,
) -> FxScoreDetails:
    adx = _safe_float(row.get("adx"))
    di_plus = _safe_float(row.get("di_plus"))
    di_minus = _safe_float(row.get("di_minus"))
    di_edge = di_plus - di_minus if signal is Side.LONG else di_minus - di_plus
    volume_ma_ratio = _safe_float(row.get("volume_ma_ratio"), _safe_float(row.get("volume_ratio")))
    current_volume_ratio = _safe_float(row.get("current_volume_ratio"), volume_ma_ratio)
    atr_price = _safe_float(row.get("atr"))
    atr_pips = atr_price / instrument.pip_size if atr_price > 0 and instrument.pip_size > 0 else 0.0
    close = _safe_float(row.get("close"))
    ma28 = _safe_float(row.get("ma28"))
    distance_ma28_pips = pips_between(instrument, close, ma28) if close > 0 and ma28 > 0 else 0.0
    entry_extension_atr = _entry_extension_atr(row, signal)

    adx_points = _normalize_points(adx, settings.adx_min, settings.score_adx_ceiling, 45.0)
    di_points = _normalize_points(di_edge, 0.0, settings.score_di_edge_ceiling, 35.0)
    volume_points = _normalize_points(
        current_volume_ratio,
        settings.score_volume_ratio_floor,
        settings.score_volume_ratio_ceiling,
        20.0,
    )
    score = _bounded_score(adx_points + di_points + volume_points)
    return FxScoreDetails(
        score=score,
        adx=adx,
        di_edge=di_edge,
        volume_ma_ratio=volume_ma_ratio,
        current_volume_ratio=current_volume_ratio,
        atr_pips=atr_pips,
        entry_extension_atr=entry_extension_atr,
        distance_ma28_pips=distance_ma28_pips,
        adx_points=adx_points,
        di_points=di_points,
        volume_points=volume_points,
    )


def _score_signal(
    row: pd.Series,
    details: dict[str, Any],
    settings: StrategySettings,
    instrument: FxInstrument,
    signal: Side,
) -> float:
    return _score_signal_details(row, details, settings, instrument, signal).score


def last_closed_row(
    frame: pd.DataFrame,
    timeframe: str,
    *,
    timestamp: datetime | pd.Timestamp | None = None,
    min_candles: int = 0,
) -> pd.Series | None:
    window = last_closed_window(
        frame,
        timeframe,
        timestamp=timestamp,
        min_candles=min_candles,
    )
    if window.empty:
        return None
    return window.iloc[-1]


def last_closed_window(
    frame: pd.DataFrame,
    timeframe: str,
    *,
    timestamp: datetime | pd.Timestamp | None = None,
    lookback: int | None = None,
    min_candles: int = 0,
) -> pd.DataFrame:
    """Return candles fully closed by the decision time.

    FX frames use candle-open timestamps. MT5 live candles are fetched with
    start position 1, so when no decision timestamp is supplied this adapter
    treats all rows as closed. When a timestamp is supplied, a row is closed
    only when ``open_time + timeframe <= decision_time``.
    """
    if lookback is not None and lookback <= 0:
        return pd.DataFrame(columns=frame.columns)

    normalized = _normalize_market_frame(frame, require_ohlcv=False)
    if normalized.empty:
        return normalized

    if timestamp is not None:
        decision_time = _coerce_utc_timestamp(timestamp)
        future_rows = normalized.index[normalized.index > decision_time]
        if len(future_rows) > 0:
            raise CandleFrameError(
                "future_candle_in_frame",
                {
                    "first_future_time": future_rows[0].isoformat(),
                    "decision_time": decision_time.isoformat(),
                },
            )
        delta = _timeframe_delta(timeframe)
        normalized = normalized.loc[normalized.index + delta <= decision_time]

    if lookback is not None:
        normalized = normalized.tail(lookback) if len(normalized) >= lookback else normalized.iloc[0:0]
    if min_candles > 0 and len(normalized) < min_candles:
        return normalized.iloc[0:0]
    return normalized


def _ensure_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = _normalize_market_frame(frame, require_ohlcv=False)
    if {"ma7", "ma14", "ma28", "adx", "atr"}.issubset(normalized.columns):
        return normalized
    return prepare_indicators(normalized)


def _normalize_market_frame(frame: pd.DataFrame, *, require_ohlcv: bool) -> pd.DataFrame:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise CandleFrameError("datetime_index_required", {"index_type": type(frame.index).__name__})
    result = frame.copy()
    index = result.index
    if index.hasnans:
        raise CandleFrameError("invalid_timestamp", {})
    if index.tz is None:
        result.index = index.tz_localize("UTC")
    else:
        result.index = index.tz_convert("UTC")
    if result.index.has_duplicates:
        duplicated = result.index[result.index.duplicated(keep=False)]
        raise CandleFrameError("duplicate_timestamps", {"timestamps": [item.isoformat() for item in duplicated[:5]]})
    if not result.index.is_monotonic_increasing:
        result = result.sort_index()
    if require_ohlcv:
        missing = [column for column in ("open", "high", "low", "close", "volume") if column not in result.columns]
        if missing:
            raise CandleFrameError("missing_ohlcv", {"missing": missing})
        if result[["open", "high", "low", "close", "volume"]].isna().any().any():
            raise CandleFrameError("nan_ohlcv", {})
    return result


def _coerce_utc_timestamp(value: datetime | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise CandleFrameError("invalid_decision_time", {})
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _timeframe_delta(timeframe: str) -> pd.Timedelta:
    try:
        return pd.Timedelta(TIMEFRAME_DELTAS[timeframe])
    except KeyError as exc:
        raise CandleFrameError("unsupported_timeframe", {"timeframe": timeframe}) from exc


def _entry_extension_atr(row: pd.Series, signal: Side) -> float:
    close = _safe_float(row.get("close"))
    ma7 = _safe_float(row.get("ma7"))
    atr = _safe_float(row.get("atr"))
    if atr <= 0:
        return 0.0
    if signal is Side.LONG:
        return max((close - ma7) / atr, 0.0)
    return max((ma7 - close) / atr, 0.0)


def _normalize_points(value: float, floor: float, ceiling: float, max_points: float) -> float:
    if not math.isfinite(value) or ceiling <= floor or max_points <= 0:
        return 0.0
    normalized = (value - floor) / (ceiling - floor)
    return float(min(max(normalized, 0.0), 1.0) * max_points)


def _bounded_score(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return round(float(min(max(value, 0.0), 100.0)), 2)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    return converted if math.isfinite(converted) else default


def _index_value(row: pd.Series) -> str | None:
    name = getattr(row, "name", None)
    if isinstance(name, pd.Timestamp):
        return _coerce_utc_timestamp(name).isoformat()
    return str(name) if name is not None else None
