"""Strategy abstractions and live-bot indicator integration."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Protocol

import pandas as pd

from backtester.config import StrategyConfig
from backtester.data import DataPortal
from backtester.models import Side, SignalIntent, timeframe_to_timedelta

try:
    from indicators import (
        MIN_HTF_CANDLES,
        MIN_SIGNAL_CANDLES,
        SignalDecision,
        TrendDecision,
        calculate_indicators,
        calculate_signal_score_details,
        get_htf_trend,
        get_signal,
    )
except ImportError as exc:  # pragma: no cover - explicit runtime guidance
    raise ImportError(
        "The backtester expects indicators.py from the live bot to be importable "
        "from the project root."
    ) from exc

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StrategyContext:
    timestamp: pd.Timestamp
    bar_index: int
    data: DataPortal
    exchange: object


class Strategy(Protocol):
    def prepare_data(self, data: DataPortal) -> None: ...

    def generate_intents(self, context: StrategyContext) -> list[SignalIntent]: ...


class IndicatorSignalStrategy:
    """Adapter around the existing `indicators.py` signal logic.

    The live indicator module intentionally evaluates `df.iloc[-2]` because a
    live API response often includes a forming candle.  During backtests we add
    a synthetic placeholder candle after the latest closed candle.  The
    placeholder uses only the latest close and zero volume, so the signal row
    remains the real historical candle with no future leakage.
    """

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    def prepare_data(self, data: DataPortal) -> None:
        selected = [self.config.entry_timeframe, self.config.htf_timeframe]
        data.apply_to_frames(calculate_indicators, selected)

    def generate_intents(self, context: StrategyContext) -> list[SignalIntent]:
        intents: list[SignalIntent] = []
        for symbol in context.data.symbols():
            intent = self._intent_for_symbol(symbol, context)
            if intent is not None:
                intents.append(intent)
        if self.config.use_signal_ranking:
            intents.sort(key=lambda item: item.score, reverse=True)
        return intents[: self.config.max_signals_per_bar]

    def _intent_for_symbol(self, symbol: str, context: StrategyContext) -> SignalIntent | None:
        entry_history = context.data.history(
            symbol,
            self.config.entry_timeframe,
            context.timestamp,
            lookback=max(MIN_SIGNAL_CANDLES + 10, 80),
            closed=True,
        )
        if len(entry_history) < MIN_SIGNAL_CANDLES:
            return None

        signal_frame = append_live_placeholder(entry_history, self.config.entry_timeframe)
        decision = get_signal(signal_frame, return_decision=True)
        if not isinstance(decision, SignalDecision) or decision.signal is None:
            return None

        htf_history = context.data.history(
            symbol,
            self.config.htf_timeframe,
            context.timestamp,
            lookback=max(MIN_HTF_CANDLES + 10, 80),
            closed=True,
        )
        if len(htf_history) < MIN_HTF_CANDLES:
            return None
        trend_frame = append_live_placeholder(htf_history, self.config.htf_timeframe)
        trend = get_htf_trend(trend_frame, return_decision=True)
        if not isinstance(trend, TrendDecision) or trend.trend != decision.signal:
            return None

        signal_row = entry_history.iloc[-1].to_dict()
        close = float(signal_row.get("close") or 0.0)
        if close <= 0:
            return None

        side = Side.LONG if decision.signal == "LONG" else Side.SHORT
        risk_distance = self._estimated_risk_distance(side, close, signal_row)
        reward_distance = risk_distance * self.config.min_risk_reward if risk_distance > 0 else 0.0
        score_details = calculate_signal_score_details(
            signal_frame,
            decision.signal,
            reward_distance,
            risk_distance,
        )
        return SignalIntent(
            symbol=symbol,
            side=side,
            timestamp=context.timestamp,
            entry_price_hint=close,
            signal_row=signal_row,
            score=score_details.score,
            metadata={
                "strategy": self.config.name,
                "signal_reason": decision.reason,
                "signal_details": decision.details,
                "htf_reason": trend.reason,
                "htf_details": trend.details,
                "score_details": asdict(score_details),
            },
        )

    def _estimated_risk_distance(self, side: Side, close: float, row: dict) -> float:
        if self.config.stop_mode == "atr":
            atr = float(row.get("atr") or 0.0)
            return atr * self.config.atr_sl_multiplier if atr > 0 else 0.0
        ma28 = float(row.get("ma28") or 0.0)
        if ma28 <= 0:
            return 0.0
        distance = close - ma28 if side is Side.LONG else ma28 - close
        return max(0.0, distance)


def append_live_placeholder(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Append a non-informative forming candle so live signal helpers work."""

    if frame.empty:
        return frame.copy()
    result = frame.copy()
    last = result.iloc[-1].copy()
    next_timestamp = result.index[-1] + pd.Timedelta(timeframe_to_timedelta(timeframe))
    placeholder = last.copy()
    close = float(last["close"])
    placeholder["open"] = close
    placeholder["high"] = close
    placeholder["low"] = close
    placeholder["close"] = close
    placeholder["volume"] = 0.0
    result.loc[next_timestamp] = placeholder
    return result
