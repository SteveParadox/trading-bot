"""Technical indicators and signal logic for the trading bot."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    ADX_MIN,
    ADX_PERIOD,
    ATR_PERIOD,
    MA_PERIODS,
    MAVOL_FAST,
    MAVOL_SLOW,
    VOLUME_RATIO_MIN,
)

MIN_CANDLES = max(max(MA_PERIODS), MAVOL_SLOW, ADX_PERIOD * 2, ATR_PERIOD * 2) + 5


def _wilders_smooth(series: pd.Series, period: int) -> pd.Series:
    """Return Wilder-smoothed values for a positive series."""
    result = pd.Series(np.nan, index=series.index, dtype="float64")
    if len(series) <= period:
        return result

    result.iloc[period] = series.iloc[1 : period + 1].sum()
    for index in range(period + 1, len(series)):
        previous = result.iloc[index - 1]
        result.iloc[index] = previous - (previous / period) + series.iloc[index]
    return result


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add moving averages, volume averages, ADX/DI, ATR, and volume ratio."""
    if df.empty:
        return df.copy()

    df = df.copy()

    for period in MA_PERIODS:
        df[f"ma{period}"] = df["close"].rolling(window=period).mean()

    df["mavol_fast"] = df["volume"].rolling(window=MAVOL_FAST).mean()
    df["mavol_slow"] = df["volume"].rolling(window=MAVOL_SLOW).mean()
    df["volume_ratio"] = df["mavol_fast"] / df["mavol_slow"].replace(0, np.nan)

    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    pos_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    neg_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )

    smooth_tr = _wilders_smooth(true_range, ADX_PERIOD)
    smooth_pos_dm = _wilders_smooth(pos_dm, ADX_PERIOD)
    smooth_neg_dm = _wilders_smooth(neg_dm, ADX_PERIOD)

    safe_tr = smooth_tr.replace(0, np.nan)
    di_plus = 100 * smooth_pos_dm / safe_tr
    di_minus = 100 * smooth_neg_dm / safe_tr
    di_sum = (di_plus + di_minus).replace(0, np.nan)
    dx = 100 * (di_plus - di_minus).abs() / di_sum

    df["di_plus"] = di_plus
    df["di_minus"] = di_minus
    df["adx"] = _wilders_smooth(dx, ADX_PERIOD) / ADX_PERIOD
    df["atr"] = _wilders_smooth(true_range, ATR_PERIOD) / ATR_PERIOD
    df["atr_pct"] = df["atr"] / df["close"].replace(0, np.nan)

    return df


def _last_closed_row(df: pd.DataFrame) -> pd.Series | None:
    if len(df) < MIN_CANDLES:
        return None
    return df.iloc[-2]


def get_signal(df: pd.DataFrame) -> str | None:
    """Return LONG, SHORT, or None from the last closed candle."""
    row = _last_closed_row(df)
    if row is None:
        return None

    required = [
        "ma7",
        "ma14",
        "ma28",
        "mavol_fast",
        "mavol_slow",
        "volume_ratio",
        "adx",
        "di_plus",
        "di_minus",
    ]
    if any(pd.isna(row[column]) for column in required):
        return None

    if row["adx"] < ADX_MIN:
        return None

    bull_trend = row["ma7"] > row["ma14"] > row["ma28"]
    bear_trend = row["ma7"] < row["ma14"] < row["ma28"]
    bull_volume = row["volume_ratio"] >= VOLUME_RATIO_MIN
    bear_volume = row["volume_ratio"] <= (1 / VOLUME_RATIO_MIN)
    bull_direction = row["di_plus"] > row["di_minus"]
    bear_direction = row["di_minus"] > row["di_plus"]
    bull_price = row["close"] > row["ma7"]
    bear_price = row["close"] < row["ma7"]

    if bull_trend and bull_volume and bull_direction and bull_price:
        return "LONG"
    if bear_trend and bear_volume and bear_direction and bear_price:
        return "SHORT"
    return None


def get_htf_trend(df_htf: pd.DataFrame) -> str | None:
    """Return higher-timeframe LONG, SHORT, or None trend confirmation."""
    row = _last_closed_row(df_htf)
    if row is None:
        return None

    required = ["ma7", "ma14", "ma28", "adx", "di_plus", "di_minus"]
    if any(pd.isna(row[column]) for column in required):
        return None

    if row["adx"] < ADX_MIN:
        return None

    if row["ma7"] > row["ma14"] > row["ma28"] and row["di_plus"] > row["di_minus"]:
        return "LONG"
    if row["ma7"] < row["ma14"] < row["ma28"] and row["di_minus"] > row["di_plus"]:
        return "SHORT"
    return None
