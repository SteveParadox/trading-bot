# ============================================================
#  indicators.py — Technical indicator calculations
# ============================================================

import pandas as pd
from config import MA_PERIODS, MAVOL_FAST, MAVOL_SLOW

# Minimum rows needed: MA28 requires 28 rows, plus 2 for iloc[-2] safety
MIN_CANDLES = 30


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds all required indicators to the OHLCV dataframe.

    Indicators added:
      - ma7, ma14, ma28   : Simple Moving Averages of close price
      - mavol_fast (9)    : Moving Average of volume (fast)
      - mavol_slow (18)   : Moving Average of volume (slow)
    """
    df = df.copy()

    # --- Simple Moving Averages (trend direction) ---
    for period in MA_PERIODS:
        df[f"ma{period}"] = df["close"].rolling(window=period).mean()

    # --- Volume Moving Averages (trend volume confirmation) ---
    # Two MAVOLs are compared: if fast > slow → volume momentum is bullish
    # This prevents entries in low-conviction moves.
    df["mavol_fast"] = df["volume"].rolling(window=MAVOL_FAST).mean()
    df["mavol_slow"] = df["volume"].rolling(window=MAVOL_SLOW).mean()

    return df


def get_signal(df: pd.DataFrame) -> str | None:
    """
    Evaluates the last complete candle and returns a trade signal.

    LONG  conditions:
        MA7 > MA14 > MA28        (price uptrend)
        MAVOL_fast > MAVOL_slow  (bullish volume momentum)

    SHORT conditions:
        MA7 < MA14 < MA28        (price downtrend)
        MAVOL_fast < MAVOL_slow  (bearish volume momentum)

    Returns 'LONG', 'SHORT', or None.
    """
    # Guard: need at least MIN_CANDLES rows for valid indicators + safe iloc[-2]
    if len(df) < MIN_CANDLES:
        return None

    # Use the last *closed* candle (index -2); -1 is the still-forming candle
    row = df.iloc[-2]

    # Guard: skip if any indicator is NaN (not enough history yet)
    cols = ["ma7", "ma14", "ma28", "mavol_fast", "mavol_slow"]
    if any(pd.isna(row[c]) for c in cols):
        return None

    ma7, ma14, ma28  = row["ma7"],       row["ma14"],      row["ma28"]
    mavol_f, mavol_s = row["mavol_fast"], row["mavol_slow"]

    bull_trend  = (ma7 > ma14) and (ma14 > ma28)
    bear_trend  = (ma7 < ma14) and (ma14 < ma28)
    bull_volume = mavol_f > mavol_s
    bear_volume = mavol_f < mavol_s

    if bull_trend and bull_volume:
        return "LONG"
    if bear_trend and bear_volume:
        return "SHORT"

    return None


def get_htf_trend(df_htf: pd.DataFrame) -> str | None:
    """
    Evaluate trend direction on a higher timeframe DataFrame.
    Returns 'LONG', 'SHORT', or None if no clear trend.
    Uses the same MA7/MA14/MA28 stack logic as the entry signal.
    """
    # Same guard as get_signal for consistency
    if len(df_htf) < MIN_CANDLES:
        return None

    row = df_htf.iloc[-2]

    cols = ["ma7", "ma14", "ma28"]
    if any(pd.isna(row[c]) for c in cols):
        return None

    if (row["ma7"] > row["ma14"]) and (row["ma14"] > row["ma28"]):
        return "LONG"
    if (row["ma7"] < row["ma14"]) and (row["ma14"] < row["ma28"]):
        return "SHORT"

    return None