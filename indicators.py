```python
"""Technical indicators, filters, and signal scoring for the trading bot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from config import (
    ADX_MIN,
    ADX_PERIOD,
    ATR_PERIOD,
    CROSS_LOOKBACK,
    FRESH_CROSS_FILTER,
    HTF_ADX_MIN,
    HTF_REQUIRE_MOMENTUM_CANDLE,
    MA_PERIODS,
    MAVOL_FAST,
    MAVOL_SLOW,
    RSI_DIVERGENCE_FILTER,
    RSI_DIVERGENCE_LOOKBACK,
    RSI_PERIOD,
    SIGNAL_SCORE_ADX_CEILING,
    SIGNAL_SCORE_ADX_FLOOR,
    SIGNAL_SCORE_MA_SEP_FULL_PCT,
    SIGNAL_SCORE_RR_CEILING,
    SIGNAL_SCORE_RR_FLOOR,
    SIGNAL_SCORE_VOLUME_CEILING,
    SIGNAL_SCORE_VOLUME_FLOOR,
    VOLUME_RATIO_MIN,
    VOLUME_SPIKE_FILTER,
    VOLUME_SPIKE_MULTIPLIER,
)


# ---------------------------------------------------------------------------
# Optional entry-extension guard.
#
# Set to a positive number to reject entries that are too far away from MA7
# in ATR terms. Leave as None until this has been validated with backtesting.
#
# Example:
#     MAX_ENTRY_DISTANCE_ATR = 2.0
# ---------------------------------------------------------------------------

MAX_ENTRY_DISTANCE_ATR: float | None = None


BASE_MIN_CANDLES = (
    max(
        max(MA_PERIODS),
        MAVOL_SLOW,
        ADX_PERIOD * 2,
        ATR_PERIOD * 2,
    )
    + 5
)

SIGNAL_EXTRA_CANDLES = 0

if RSI_DIVERGENCE_FILTER:
    SIGNAL_EXTRA_CANDLES = max(
        SIGNAL_EXTRA_CANDLES,
        RSI_PERIOD * 2,
        RSI_DIVERGENCE_LOOKBACK + 2,
    )

if FRESH_CROSS_FILTER:
    SIGNAL_EXTRA_CANDLES = max(
        SIGNAL_EXTRA_CANDLES,
        CROSS_LOOKBACK + 2,
    )

MIN_SIGNAL_CANDLES = max(
    BASE_MIN_CANDLES,
    SIGNAL_EXTRA_CANDLES + 5,
)

MIN_HTF_CANDLES = BASE_MIN_CANDLES


@dataclass(frozen=True)
class SignalDecision:
    signal: str | None
    reason: str
    details: dict[str, Any]


@dataclass(frozen=True)
class TrendDecision:
    trend: str | None
    reason: str
    details: dict[str, Any]


@dataclass(frozen=True)
class SignalScoreDetails:
    score: float
    adx: float
    volume_ratio: float
    ma_separation_pct: float
    rr_ratio: float
    adx_points: float
    volume_points: float
    ma_points: float
    rr_points: float
    entry_distance_atr: float = 0.0


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a scalar to float, treating None/NaN as default."""
    if value is None:
        return default

    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    return default if not np.isfinite(result) else result


# ---------------------------------------------------------------------------
# Wilder smoothing
# ---------------------------------------------------------------------------

def _wilders_smooth(series: pd.Series, period: int) -> pd.Series:
    """
    Return Wilder-smoothed SUM values for a positive/non-negative series.

    This is appropriate for:
        - RSI gain/loss smoothing
        - ATR true-range smoothing
        - DI component smoothing

    The first valid Wilder sum is seeded at `period` using observations
    1..period because index 0 normally represents the unavailable
    first-difference/previous-close observation.
    """
    result = pd.Series(
        np.nan,
        index=series.index,
        dtype="float64",
    )

    if period <= 0 or len(series) <= period:
        return result

    seed = series.iloc[1 : period + 1]

    if seed.isna().any():
        return result

    result.iloc[period] = seed.sum()

    for index in range(period + 1, len(series)):
        current = series.iloc[index]

        if pd.isna(current) or pd.isna(result.iloc[index - 1]):
            continue

        previous = result.iloc[index - 1]
        result.iloc[index] = (
            previous
            - (previous / period)
            + current
        )

    return result


def _calculate_wilder_adx(
    true_range: pd.Series,
    pos_dm: pd.Series,
    neg_dm: pd.Series,
    period: int,
    index: pd.Index,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate DI+, DI-, and ADX using Wilder's initialization.

    Important:
    DI is first available after `period` observations.
    DX becomes available from that point.
    ADX is then seeded from the average of the first `period` valid DX values.

    This avoids the incorrect early-ADX initialization that occurs when DX
    is fed directly into a generic Wilder smoother.
    """
    smooth_tr = _wilders_smooth(true_range, period)
    smooth_pos_dm = _wilders_smooth(pos_dm, period)
    smooth_neg_dm = _wilders_smooth(neg_dm, period)

    di_plus = pd.Series(np.nan, index=index, dtype="float64")
    di_minus = pd.Series(np.nan, index=index, dtype="float64")

    safe_tr = smooth_tr.replace(0, np.nan)

    di_plus = 100.0 * smooth_pos_dm / safe_tr
    di_minus = 100.0 * smooth_neg_dm / safe_tr

    di_sum = (di_plus + di_minus).replace(0, np.nan)

    dx = (
        100.0
        * (di_plus - di_minus).abs()
        / di_sum
    )

    adx = pd.Series(
        np.nan,
        index=index,
        dtype="float64",
    )

    # First DX is available at `period`.
    # First complete ADX seed therefore requires `period` DX values:
    # [period, ..., 2*period-1].
    first_adx_index = 2 * period - 1

    if len(dx) <= first_adx_index:
        return di_plus, di_minus, adx

    seed_window = dx.iloc[period : first_adx_index + 1]

    if seed_window.isna().any():
        # This should not normally happen with sufficient clean history.
        return di_plus, di_minus, adx

    adx.iloc[first_adx_index] = seed_window.mean()

    for current_index in range(first_adx_index + 1, len(dx)):
        current_dx = dx.iloc[current_index]
        previous_adx = adx.iloc[current_index - 1]

        if pd.isna(current_dx) or pd.isna(previous_adx):
            continue

        adx.iloc[current_index] = (
            (
                previous_adx * (period - 1)
            )
            + current_dx
        ) / period

    return di_plus, di_minus, adx


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def calculate_rsi(
    close: pd.Series,
    period: int = RSI_PERIOD,
) -> pd.Series:
    """Calculate RSI using Wilder's smoothing."""
    if close.empty:
        return pd.Series(
            dtype="float64",
            index=close.index,
        )

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = (
        _wilders_smooth(gain, period)
        / period
    )

    avg_loss = (
        _wilders_smooth(loss, period)
        / period
    )

    rs = (
        avg_gain
        / avg_loss.replace(0, np.nan)
    )

    rsi = 100.0 - (
        100.0 / (1.0 + rs)
    )

    # Handle monotonically rising / flat periods explicitly.
    rsi = rsi.where(
        ~(
            (avg_loss == 0)
            & (avg_gain > 0)
        ),
        100.0,
    )

    rsi = rsi.where(
        ~(
            (avg_loss == 0)
            & (avg_gain == 0)
        ),
        50.0,
    )

    return rsi


# ---------------------------------------------------------------------------
# Indicator calculation
# ---------------------------------------------------------------------------

def calculate_indicators(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add moving averages, volume averages, ADX/DI, ATR, RSI, and ratios.

    Expected input columns:
        open, high, low, close, volume
    """
    if df.empty:
        return df.copy()

    df = df.copy()

    # -----------------------------------------------------------------------
    # Price moving averages
    # -----------------------------------------------------------------------

    for period in MA_PERIODS:
        df[f"ma{period}"] = (
            df["close"]
            .rolling(window=period)
            .mean()
        )

    # -----------------------------------------------------------------------
    # Volume metrics
    # -----------------------------------------------------------------------

    df["mavol_fast"] = (
        df["volume"]
        .rolling(window=MAVOL_FAST)
        .mean()
    )

    df["mavol_slow"] = (
        df["volume"]
        .rolling(window=MAVOL_SLOW)
        .mean()
    )

    # Recent average participation relative to longer-term participation.
    df["volume_ma_ratio"] = (
        df["mavol_fast"]
        / df["mavol_slow"].replace(0, np.nan)
    )

    # Current-bar participation relative to the slow volume average.
    df["current_volume_ratio"] = (
        df["volume"]
        / df["mavol_slow"].replace(0, np.nan)
    )

    # Backward-compatible alias for existing consumers.
    df["volume_ratio"] = df["volume_ma_ratio"]

    # -----------------------------------------------------------------------
    # RSI
    # -----------------------------------------------------------------------

    df["rsi"] = calculate_rsi(
        df["close"],
        RSI_PERIOD,
    )

    # -----------------------------------------------------------------------
    # True Range / Directional Movement
    # -----------------------------------------------------------------------

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
        np.where(
            (up_move > down_move)
            & (up_move > 0),
            up_move,
            0.0,
        ),
        index=df.index,
        dtype="float64",
    )

    neg_dm = pd.Series(
        np.where(
            (down_move > up_move)
            & (down_move > 0),
            down_move,
            0.0,
        ),
        index=df.index,
        dtype="float64",
    )

    # -----------------------------------------------------------------------
    # Correct Wilder ADX / DI calculation
    # -----------------------------------------------------------------------

    (
        di_plus,
        di_minus,
        adx,
    ) = _calculate_wilder_adx(
        true_range=true_range,
        pos_dm=pos_dm,
        neg_dm=neg_dm,
        period=ADX_PERIOD,
        index=df.index,
    )

    df["di_plus"] = di_plus
    df["di_minus"] = di_minus
    df["adx"] = adx

    # -----------------------------------------------------------------------
    # ATR
    # -----------------------------------------------------------------------

    smooth_tr_atr = _wilders_smooth(
        true_range,
        ATR_PERIOD,
    )

    df["atr"] = (
        smooth_tr_atr
        / ATR_PERIOD
    )

    df["atr_pct"] = (
        df["atr"]
        / df["close"].replace(0, np.nan)
    )

    return df


# ---------------------------------------------------------------------------
# Closed-candle helpers
# ---------------------------------------------------------------------------

def _last_closed_row(
    df: pd.DataFrame,
    min_candles: int = BASE_MIN_CANDLES,
) -> pd.Series | None:
    """Return the last fully closed candle when enough history exists."""
    if len(df) < min_candles:
        return None

    # Assumes df.iloc[-1] is the currently forming candle.
    return df.iloc[-2]


def _closed_window(
    df: pd.DataFrame,
    lookback: int,
) -> pd.DataFrame:
    """Return exactly `lookback` closed candles."""
    if lookback <= 0 or len(df) < lookback + 1:
        return pd.DataFrame()

    return df.iloc[-lookback - 1 : -1]


# ---------------------------------------------------------------------------
# Divergence
# ---------------------------------------------------------------------------

def _pivot_indices(
    window: pd.DataFrame,
    column: str,
    pivot_type: str,
) -> list[Any]:
    values = window[column].to_numpy(
        dtype="float64",
    )

    if len(values) < 3 or np.isnan(values).any():
        return []

    indices: list[Any] = []

    for pos in range(1, len(values) - 1):
        value = values[pos]

        if pivot_type == "high":
            previous_ok = value > values[pos - 1]
            next_ok = value >= values[pos + 1]
        else:
            previous_ok = value < values[pos - 1]
            next_ok = value <= values[pos + 1]

        if previous_ok and next_ok:
            indices.append(
                window.index[pos]
            )

    return indices


def detect_divergence(
    df: pd.DataFrame,
    signal: str,
    lookback: int = RSI_DIVERGENCE_LOOKBACK,
) -> bool:
    """
    Return True when clear three-pivot RSI divergence should block the signal.

    LONG:
        price highs rising
        RSI highs falling

    SHORT:
        price lows falling
        RSI lows rising
    """
    window = _closed_window(
        df,
        max(lookback, 3),
    )

    if window.empty or "rsi" not in window:
        return False

    if signal == "LONG":
        required = ["high", "rsi"]

        if any(
            column not in window
            or window[column].isna().any()
            for column in required
        ):
            return False

        pivots = _pivot_indices(
            window,
            "high",
            "high",
        )

        if len(pivots) < 3:
            return False

        last_three = pivots[-3:]

        price_highs = window.loc[
            last_three,
            "high",
        ].to_numpy(dtype="float64")

        rsi_highs = window.loc[
            last_three,
            "rsi",
        ].to_numpy(dtype="float64")

        return bool(
            np.all(np.diff(price_highs) > 0)
            and np.all(np.diff(rsi_highs) < 0)
        )

    if signal == "SHORT":
        required = ["low", "rsi"]

        if any(
            column not in window
            or window[column].isna().any()
            for column in required
        ):
            return False

        pivots = _pivot_indices(
            window,
            "low",
            "low",
        )

        if len(pivots) < 3:
            return False

        last_three = pivots[-3:]

        price_lows = window.loc[
            last_three,
            "low",
        ].to_numpy(dtype="float64")

        rsi_lows = window.loc[
            last_three,
            "rsi",
        ].to_numpy(dtype="float64")

        return bool(
            np.all(np.diff(price_lows) < 0)
            and np.all(np.diff(rsi_lows) > 0)
        )

    return False


# ---------------------------------------------------------------------------
# Fresh MA cross
# ---------------------------------------------------------------------------

def detect_fresh_cross(
    df: pd.DataFrame,
    signal: str,
    lookback: int = CROSS_LOOKBACK,
) -> bool:
    """
    Return True if MA7 crossed MA14 in the signal direction
    within exactly `lookback` closed candles.
    """
    if lookback <= 0:
        return False

    window = _closed_window(
        df,
        lookback,
    )

    if window.empty:
        return False

    if any(
        column not in window
        for column in ("ma7", "ma14")
    ):
        return False

    ma7 = window["ma7"].to_numpy(
        dtype="float64",
    )

    ma14 = window["ma14"].to_numpy(
        dtype="float64",
    )

    if np.isnan(ma7).any() or np.isnan(ma14).any():
        return False

    for index in range(1, len(window)):
        if (
            signal == "LONG"
            and ma7[index - 1] <= ma14[index - 1]
            and ma7[index] > ma14[index]
        ):
            return True

        if (
            signal == "SHORT"
            and ma7[index - 1] >= ma14[index - 1]
            and ma7[index] < ma14[index]
        ):
            return True

    return False


def _fresh_cross_data_available(
    df: pd.DataFrame,
    lookback: int,
) -> bool:
    """Return whether the complete closed cross window is available."""
    window = _closed_window(
        df,
        lookback,
    )

    if window.empty:
        return False

    if any(
        column not in window
        for column in ("ma7", "ma14")
    ):
        return False

    return not bool(
        window[["ma7", "ma14"]]
        .isna()
        .any()
        .any()
    )


# ---------------------------------------------------------------------------
# Entry-location helper
# ---------------------------------------------------------------------------

def _entry_distance_atr(
    row: pd.Series,
    signal: str,
) -> float:
    """
    Return distance from MA7 in ATR units.

    LONG:
        (close - MA7) / ATR

    SHORT:
        (MA7 - close) / ATR

    Positive values therefore represent extension in the signal direction.
    """
    close = _safe_float(row.get("close"))
    ma7 = _safe_float(row.get("ma7"))
    atr = _safe_float(row.get("atr"))

    if atr <= 0:
        return 0.0

    if signal == "LONG":
        return max((close - ma7) / atr, 0.0)

    if signal == "SHORT":
        return max((ma7 - close) / atr, 0.0)

    return 0.0


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def _signal_decision(
    df: pd.DataFrame,
) -> SignalDecision:
    row = _last_closed_row(
        df,
        MIN_SIGNAL_CANDLES,
    )

    if row is None:
        return SignalDecision(
            None,
            "insufficient_candles",
            {
                "min_candles": MIN_SIGNAL_CANDLES,
                "candles": len(df),
            },
        )

    required = [
        "ma7",
        "ma14",
        "ma28",
        "mavol_fast",
        "mavol_slow",
        "volume_ma_ratio",
        "adx",
        "di_plus",
        "di_minus",
    ]

    if any(
        column not in row
        or pd.isna(row[column])
        for column in required
    ):
        return SignalDecision(
            None,
            "indicator_unavailable",
            {
                "required": ",".join(required),
            },
        )

    details = {
        "ma7": _safe_float(row["ma7"]),
        "ma14": _safe_float(row["ma14"]),
        "ma28": _safe_float(row["ma28"]),
        "mavol_fast": _safe_float(row["mavol_fast"]),
        "mavol_slow": _safe_float(row["mavol_slow"]),
        "volume": _safe_float(row["volume"]),
        "volume_ma_ratio": _safe_float(
            row.get("volume_ma_ratio")
        ),
        "current_volume_ratio": _safe_float(
            row.get("current_volume_ratio")
        ),
        # Backward-compatible key.
        "volume_ratio": _safe_float(
            row.get("volume_ma_ratio")
        ),
        "volume_spike_ratio": _safe_float(
            row.get("current_volume_ratio")
        ),
        "adx": _safe_float(row["adx"]),
        "di_plus": _safe_float(row["di_plus"]),
        "di_minus": _safe_float(row["di_minus"]),
        "close": _safe_float(row["close"]),
        "atr": _safe_float(row.get("atr")),
        "atr_pct": _safe_float(row.get("atr_pct")),
    }

    # -----------------------------------------------------------------------
    # ADX
    # -----------------------------------------------------------------------

    if row["adx"] < ADX_MIN:
        details["required_adx"] = ADX_MIN

        return SignalDecision(
            None,
            "adx_filter",
            details,
        )

    # -----------------------------------------------------------------------
    # MA trend structure
    # -----------------------------------------------------------------------

    bull_trend = (
        row["ma7"] > row["ma14"] > row["ma28"]
    )

    bear_trend = (
        row["ma7"] < row["ma14"] < row["ma28"]
    )

    if not bull_trend and not bear_trend:
        return SignalDecision(
            None,
            "ma_stack_filter",
            details,
        )

    signal = "LONG" if bull_trend else "SHORT"

    # -----------------------------------------------------------------------
    # Directional movement confirmation
    # -----------------------------------------------------------------------

    if signal == "LONG":
        if row["di_plus"] <= row["di_minus"]:
            return SignalDecision(
                None,
                "direction_filter",
                details | {"signal": signal},
            )

        if row["close"] <= row["ma7"]:
            return SignalDecision(
                None,
                "price_ma_filter",
                details | {"signal": signal},
            )

    else:
        if row["di_minus"] <= row["di_plus"]:
            return SignalDecision(
                None,
                "direction_filter",
                details | {"signal": signal},
            )

        if row["close"] >= row["ma7"]:
            return SignalDecision(
                None,
                "price_ma_filter",
                details | {"signal": signal},
            )

    # -----------------------------------------------------------------------
    # Entry location in ATR units
    # -----------------------------------------------------------------------

    entry_distance_atr = _entry_distance_atr(
        row,
        signal,
    )

    details["entry_distance_atr"] = (
        entry_distance_atr
    )

    # Optional anti-chasing filter.
    if (
        MAX_ENTRY_DISTANCE_ATR is not None
        and MAX_ENTRY_DISTANCE_ATR > 0
        and entry_distance_atr > MAX_ENTRY_DISTANCE_ATR
    ):
        return SignalDecision(
            None,
            "entry_extension_filter",
            details
            | {
                "signal": signal,
                "required_max_entry_distance_atr": (
                    MAX_ENTRY_DISTANCE_ATR
                ),
            },
        )

    # -----------------------------------------------------------------------
    # Volume participation
    # -----------------------------------------------------------------------

    volume_ma_ratio = _safe_float(
        row.get("volume_ma_ratio")
    )

    if volume_ma_ratio < VOLUME_RATIO_MIN:
        return SignalDecision(
            None,
            "volume_ma_filter",
            details
            | {
                "signal": signal,
                "required_volume_ratio": VOLUME_RATIO_MIN,
            },
        )

    # -----------------------------------------------------------------------
    # Current-bar volume spike
    # -----------------------------------------------------------------------

    if VOLUME_SPIKE_FILTER:
        current_volume_ratio = _safe_float(
            row.get("current_volume_ratio")
        )

        if current_volume_ratio < VOLUME_SPIKE_MULTIPLIER:
            return SignalDecision(
                None,
                "volume_spike_filter",
                details
                | {
                    "signal": signal,
                    "required_spike_ratio": (
                        VOLUME_SPIKE_MULTIPLIER
                    ),
                },
            )

    # -----------------------------------------------------------------------
    # RSI divergence veto
    # -----------------------------------------------------------------------

    if (
        RSI_DIVERGENCE_FILTER
        and detect_divergence(
            df,
            signal,
            RSI_DIVERGENCE_LOOKBACK,
        )
    ):
        return SignalDecision(
            None,
            "rsi_divergence_filter",
            details
            | {
                "signal": signal,
                "lookback": RSI_DIVERGENCE_LOOKBACK,
            },
        )

    # -----------------------------------------------------------------------
    # Fresh MA cross
    # -----------------------------------------------------------------------

    if (
        FRESH_CROSS_FILTER
        and _fresh_cross_data_available(
            df,
            CROSS_LOOKBACK,
        )
        and not detect_fresh_cross(
            df,
            signal,
            CROSS_LOOKBACK,
        )
    ):
        return SignalDecision(
            None,
            "fresh_cross_filter",
            details
            | {
                "signal": signal,
                "lookback": CROSS_LOOKBACK,
            },
        )

    return SignalDecision(
        signal,
        "signal_confirmed",
        details | {"signal": signal},
    )


def get_signal(
    df: pd.DataFrame,
    *,
    return_decision: bool = False,
) -> SignalDecision | str | None:
    """Return LONG/SHORT/None, or a SignalDecision when requested."""
    decision = _signal_decision(df)

    return (
        decision
        if return_decision
        else decision.signal
    )


# ---------------------------------------------------------------------------
# Higher-timeframe trend
# ---------------------------------------------------------------------------

def _trend_decision(
    df_htf: pd.DataFrame,
) -> TrendDecision:
    row = _last_closed_row(
        df_htf,
        MIN_HTF_CANDLES,
    )

    if row is None:
        return TrendDecision(
            None,
            "insufficient_candles",
            {
                "min_candles": MIN_HTF_CANDLES,
                "candles": len(df_htf),
            },
        )

    required = [
        "ma7",
        "ma14",
        "ma28",
        "adx",
        "di_plus",
        "di_minus",
        "open",
        "close",
    ]

    if any(
        column not in row
        or pd.isna(row[column])
        for column in required
    ):
        return TrendDecision(
            None,
            "indicator_unavailable",
            {
                "required": ",".join(required),
            },
        )

    details = {
        "ma7": _safe_float(row["ma7"]),
        "ma14": _safe_float(row["ma14"]),
        "ma28": _safe_float(row["ma28"]),
        "adx": _safe_float(row["adx"]),
        "di_plus": _safe_float(row["di_plus"]),
        "di_minus": _safe_float(row["di_minus"]),
        "open": _safe_float(row["open"]),
        "close": _safe_float(row["close"]),
    }

    # Only the HTF threshold is required here.
    # The previous implementation checked ADX_MIN first, which was redundant
    # whenever HTF_ADX_MIN >= ADX_MIN.
    if row["adx"] < HTF_ADX_MIN:
        return TrendDecision(
            None,
            "htf_adx_filter",
            details
            | {
                "required_adx": HTF_ADX_MIN,
            },
        )

    bull_stack = (
        row["ma7"] > row["ma14"] > row["ma28"]
    )

    bear_stack = (
        row["ma7"] < row["ma14"] < row["ma28"]
    )

    if not bull_stack and not bear_stack:
        return TrendDecision(
            None,
            "htf_ma_stack_filter",
            details,
        )

    trend = "LONG" if bull_stack else "SHORT"

    if (
        trend == "LONG"
        and row["di_plus"] <= row["di_minus"]
    ):
        return TrendDecision(
            None,
            "htf_di_filter",
            details | {"trend": trend},
        )

    if (
        trend == "SHORT"
        and row["di_minus"] <= row["di_plus"]
    ):
        return TrendDecision(
            None,
            "htf_di_filter",
            details | {"trend": trend},
        )

    # Keep this as an explicit optional filter because candle colour is
    # a potentially noisy constraint inside an established trend.
    if HTF_REQUIRE_MOMENTUM_CANDLE:
        bullish_body = row["close"] > row["open"]
        bearish_body = row["close"] < row["open"]

        if (
            trend == "LONG"
            and not bullish_body
        ):
            return TrendDecision(
                None,
                "htf_momentum_candle_filter",
                details | {"trend": trend},
            )

        if (
            trend == "SHORT"
            and not bearish_body
        ):
            return TrendDecision(
                None,
                "htf_momentum_candle_filter",
                details | {"trend": trend},
            )

    return TrendDecision(
        trend,
        "htf_trend_confirmed",
        details | {"trend": trend},
    )


def get_htf_trend(
    df_htf: pd.DataFrame,
    *,
    return_decision: bool = False,
) -> TrendDecision | str | None:
    """Return LONG/SHORT/None, or a TrendDecision when requested."""
    decision = _trend_decision(df_htf)

    return (
        decision
        if return_decision
        else decision.trend
    )


# ---------------------------------------------------------------------------
# Signal scoring
# ---------------------------------------------------------------------------

def _normalize_points(
    value: float,
    floor: float,
    ceiling: float,
    max_points: float,
) -> float:
    """Normalize a value into [0, max_points]."""
    if (
        not np.isfinite(value)
        or ceiling <= floor
        or max_points <= 0
    ):
        return 0.0

    normalized = (
        (value - floor)
        / (ceiling - floor)
    )

    return float(
        np.clip(
            normalized,
            0.0,
            1.0,
        )
        * max_points
    )


def calculate_signal_score_details(
    df: pd.DataFrame,
    signal: str,
    tp_distance: float,
    sl_distance: float,
) -> SignalScoreDetails:
    """Return signal score with components for ranking logs."""
    row = _last_closed_row(
        df,
        MIN_SIGNAL_CANDLES,
    )

    if row is None:
        return SignalScoreDetails(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    adx = _safe_float(
        row.get("adx")
    )

    mavol_slow = _safe_float(
        row.get("mavol_slow")
    )

    current_volume = _safe_float(
        row.get("volume")
    )

    # Scoring deliberately uses CURRENT BAR participation, while the
    # signal eligibility filter uses recent-average participation.
    #
    # These are different concepts and are now named explicitly.
    if mavol_slow > 0:
        current_volume_ratio = (
            current_volume / mavol_slow
        )
    else:
        current_volume_ratio = _safe_float(
            row.get("current_volume_ratio")
        )

    ma7 = _safe_float(row.get("ma7"))
    ma28 = _safe_float(row.get("ma28"))

    if ma28 != 0:
        ma_separation = (
            abs(ma7 - ma28)
            / abs(ma28)
        )
    else:
        ma_separation = 0.0

    rr_ratio = (
        tp_distance / sl_distance
        if sl_distance > 0
        else 0.0
    )

    entry_distance_atr = _entry_distance_atr(
        row,
        signal,
    )

    adx_points = _normalize_points(
        adx,
        SIGNAL_SCORE_ADX_FLOOR,
        SIGNAL_SCORE_ADX_CEILING,
        40.0,
    )

    volume_points = _normalize_points(
        current_volume_ratio,
        SIGNAL_SCORE_VOLUME_FLOOR,
        SIGNAL_SCORE_VOLUME_CEILING,
        30.0,
    )

    ma_points = _normalize_points(
        ma_separation,
        0.0,
        SIGNAL_SCORE_MA_SEP_FULL_PCT / 100.0,
        20.0,
    )

    rr_points = _normalize_points(
        rr_ratio,
        SIGNAL_SCORE_RR_FLOOR,
        SIGNAL_SCORE_RR_CEILING,
        10.0,
    )

    score = float(
        np.clip(
            adx_points
            + volume_points
            + ma_points
            + rr_points,
            0.0,
            100.0,
        )
    )

    return SignalScoreDetails(
        score=score,
        adx=adx,
        volume_ratio=current_volume_ratio,
        ma_separation_pct=ma_separation * 100.0,
        rr_ratio=rr_ratio,
        adx_points=adx_points,
        volume_points=volume_points,
        ma_points=ma_points,
        rr_points=rr_points,
        entry_distance_atr=entry_distance_atr,
    )


def calculate_signal_score(
    df: pd.DataFrame,
    signal: str,
    tp_distance: float,
    sl_distance: float,
) -> float:
    """Return a 0-100 quality score for a valid signal."""
    return calculate_signal_score_details(
        df,
        signal,
        tp_distance,
        sl_distance,
    ).score
```
