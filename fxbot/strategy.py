"""FX-retuned strategy adapter around the existing indicator calculations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

import indicators as legacy_indicators

from fxbot.config import StrategySettings
from fxbot.instruments import FxInstrument, pips_between
from fxbot.models import FxSignalIntent, Side



@dataclass(frozen=True)
class FxSignalDecision:
    signal: Side | None
    reason: str
    details: dict[str, Any]


def prepare_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    original = {
        "MA_PERIODS": legacy_indicators.MA_PERIODS[:],
        "MAVOL_FAST": legacy_indicators.MAVOL_FAST,
        "MAVOL_SLOW": legacy_indicators.MAVOL_SLOW,
        "ADX_PERIOD": legacy_indicators.ADX_PERIOD,
        "ATR_PERIOD": legacy_indicators.ATR_PERIOD,
        "VOLUME_RATIO_MIN": legacy_indicators.VOLUME_RATIO_MIN,
    }
    try:
        legacy_indicators.MA_PERIODS = [7, 14, 28]
        legacy_indicators.MAVOL_FAST = 9
        legacy_indicators.MAVOL_SLOW = 18
        legacy_indicators.ADX_PERIOD = 14
        legacy_indicators.ATR_PERIOD = 14
        legacy_indicators.VOLUME_RATIO_MIN = 1.0
        return legacy_indicators.calculate_indicators(frame.copy()).sort_index()
    finally:
        legacy_indicators.MA_PERIODS = original["MA_PERIODS"]
        legacy_indicators.MAVOL_FAST = original["MAVOL_FAST"]
        legacy_indicators.MAVOL_SLOW = original["MAVOL_SLOW"]
        legacy_indicators.ADX_PERIOD = original["ADX_PERIOD"]
        legacy_indicators.ATR_PERIOD = original["ATR_PERIOD"]
        legacy_indicators.VOLUME_RATIO_MIN = original["VOLUME_RATIO_MIN"]


def evaluate_signal_frame(
    entry_frame: pd.DataFrame,
    htf_frame: pd.DataFrame,
    *,
    instrument: FxInstrument,
    settings: StrategySettings,
    timestamp: datetime | None = None,
) -> FxSignalDecision:
    if len(entry_frame) < 60 or len(htf_frame) < 60:
        return FxSignalDecision(None, "insufficient_candles", {"entry": len(entry_frame), "htf": len(htf_frame)})

    entry = _ensure_indicators(entry_frame)
    htf = _ensure_indicators(htf_frame)
    entry_row = entry.iloc[-1]
    htf_row = htf.iloc[-1]
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

    details = {
        **signal.details,
        "htf": htf_signal.details,
        "atr_pips": atr_pips,
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
    row = _ensure_indicators(entry_frame).iloc[-1]
    score = _score_signal(row, decision.details, settings)
    return FxSignalIntent(
        instrument=instrument.name,
        side=decision.signal,
        timestamp=timestamp or datetime.now(timezone.utc),
        entry_price=entry_price,
        signal_row=row.to_dict(),
        score=score,
        metadata={"strategy": "fx_indicator_retune", "decision": decision.reason, "details": decision.details},
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

    details = {
        "ma7": float(row["ma7"]),
        "ma14": float(row["ma14"]),
        "ma28": float(row["ma28"]),
        "adx": float(row["adx"]),
        "di_plus": float(row["di_plus"]),
        "di_minus": float(row["di_minus"]),
        "close": float(row["close"]),
        "volume_ratio": float(row.get("volume_ratio") or 0.0),
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
    if require_volume and details["volume_ratio"] < volume_ratio_min:
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


def _score_signal(row: pd.Series, details: dict[str, Any], settings: StrategySettings) -> float:
    adx = float(row.get("adx") or 0.0)
    atr_pips = float(details.get("atr_pips") or 0.0)
    ma28 = float(row.get("ma28") or 0.0)
    close = float(row.get("close") or 0.0)
    ma_sep_pips = pips_between(FxInstrument(str(details.get("instrument") or "EUR_USD")), close, ma28) if ma28 > 0 else 0.0
    adx_points = min(max((adx - settings.adx_min) / 20.0, 0.0), 1.0) * 45.0
    atr_points = min(max((atr_pips - settings.min_atr_pips) / max(1.0, settings.max_atr_pips), 0.0), 1.0) * 25.0
    sep_points = min(ma_sep_pips / max(1.0, settings.max_stop_pips), 1.0) * 30.0
    return round(adx_points + atr_points + sep_points, 2)


def _ensure_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    if {"ma7", "ma14", "ma28", "adx", "atr"}.issubset(frame.columns):
        return frame.sort_index()
    return prepare_indicators(frame)

