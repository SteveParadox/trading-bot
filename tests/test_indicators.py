from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import indicators


def realistic_fx_frame(rows: int = 90) -> pd.DataFrame:
    index = pd.date_range("2026-01-05T00:00:00Z", periods=rows, freq="15min")
    x = np.arange(rows, dtype="float64")
    close = 1.1000 + x * 0.00012 + np.sin(x / 3.0) * 0.0012
    open_ = close - np.cos(x / 5.0) * 0.00015
    spread = 0.00045 + (np.sin(x / 7.0) + 1.0) * 0.00012
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread * 0.85
    volume = 100.0 + (np.sin(x / 4.0) + 1.0) * 25.0 + x * 0.2
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def reference_wilder_sum(values: np.ndarray, period: int) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype="float64")
    if period <= 0 or len(values) <= period:
        return result
    seed = values[1 : period + 1]
    if np.isnan(seed).any():
        return result
    result[period] = seed.sum()
    for idx in range(period + 1, len(values)):
        if np.isnan(values[idx]) or np.isnan(result[idx - 1]):
            continue
        result[idx] = result[idx - 1] - (result[idx - 1] / period) + values[idx]
    return result


def reference_adx_atr(frame: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    high = frame["high"].to_numpy(dtype="float64")
    low = frame["low"].to_numpy(dtype="float64")
    close = frame["close"].to_numpy(dtype="float64")
    true_range = np.full(len(frame), np.nan, dtype="float64")
    pos_dm = np.zeros(len(frame), dtype="float64")
    neg_dm = np.zeros(len(frame), dtype="float64")

    for idx in range(len(frame)):
        if idx == 0:
            true_range[idx] = high[idx] - low[idx]
            continue
        true_range[idx] = max(
            high[idx] - low[idx],
            abs(high[idx] - close[idx - 1]),
            abs(low[idx] - close[idx - 1]),
        )
        up_move = high[idx] - high[idx - 1]
        down_move = low[idx - 1] - low[idx]
        pos_dm[idx] = up_move if up_move > down_move and up_move > 0 else 0.0
        neg_dm[idx] = down_move if down_move > up_move and down_move > 0 else 0.0

    smooth_tr = reference_wilder_sum(true_range, period)
    smooth_pos_dm = reference_wilder_sum(pos_dm, period)
    smooth_neg_dm = reference_wilder_sum(neg_dm, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        di_plus = 100.0 * smooth_pos_dm / smooth_tr
        di_minus = 100.0 * smooth_neg_dm / smooth_tr
        dx = 100.0 * np.abs(di_plus - di_minus) / (di_plus + di_minus)

    adx = np.full(len(frame), np.nan, dtype="float64")
    first_adx = 2 * period - 1
    if len(frame) > first_adx:
        seed = dx[period : first_adx + 1]
        if not np.isnan(seed).any():
            adx[first_adx] = seed.mean()
            for idx in range(first_adx + 1, len(frame)):
                adx[idx] = ((adx[idx - 1] * (period - 1)) + dx[idx]) / period

    return pd.DataFrame(
        {
            "di_plus": di_plus,
            "di_minus": di_minus,
            "adx": adx,
            "atr": smooth_tr / period,
        },
        index=frame.index,
    )


def reference_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    values = close.to_numpy(dtype="float64")
    delta = np.full(len(values), np.nan, dtype="float64")
    delta[1:] = np.diff(values)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    gains[0] = np.nan
    losses[0] = np.nan
    avg_gain = reference_wilder_sum(gains, period) / period
    avg_loss = reference_wilder_sum(losses, period) / period
    result = np.full(len(values), np.nan, dtype="float64")
    for idx in range(len(values)):
        gain = avg_gain[idx]
        loss = avg_loss[idx]
        if np.isnan(gain) or np.isnan(loss):
            continue
        if loss == 0 and gain > 0:
            result[idx] = 100.0
        elif loss == 0 and gain == 0:
            result[idx] = 50.0
        else:
            result[idx] = 100.0 - (100.0 / (1.0 + (gain / loss)))
    return pd.Series(result, index=close.index)


class IndicatorMathTests(unittest.TestCase):
    def test_adx_di_atr_match_independent_wilder_reference(self) -> None:
        frame = realistic_fx_frame()
        result = indicators.calculate_indicators(
            frame,
            ma_periods=(7, 14, 28),
            mavol_fast=9,
            mavol_slow=18,
            adx_period=14,
            atr_period=14,
        )
        reference = reference_adx_atr(frame, period=14)

        for column in ("di_plus", "di_minus", "adx", "atr"):
            valid = reference[column].notna()
            np.testing.assert_allclose(
                result.loc[valid, column].to_numpy(),
                reference.loc[valid, column].to_numpy(),
                rtol=1e-12,
                atol=1e-12,
            )

        self.assertEqual(result["atr"].first_valid_index(), frame.index[14])
        self.assertEqual(result["adx"].first_valid_index(), frame.index[27])

    def test_rsi_matches_independent_wilder_reference_and_flat_market(self) -> None:
        frame = realistic_fx_frame()
        result = indicators.calculate_rsi(frame["close"], period=14)
        reference = reference_rsi(frame["close"], period=14)
        valid = reference.notna()
        np.testing.assert_allclose(result.loc[valid].to_numpy(), reference.loc[valid].to_numpy(), rtol=1e-12, atol=1e-12)
        self.assertEqual(result.first_valid_index(), frame.index[14])

        flat = pd.Series([1.2345] * 30, index=pd.date_range("2026-01-01T00:00:00Z", periods=30, freq="15min"))
        flat_rsi = indicators.calculate_rsi(flat, period=14)
        self.assertTrue((flat_rsi.iloc[14:] == 50.0).all())

    def test_moving_average_and_volume_ratio_columns_are_explicit(self) -> None:
        frame = realistic_fx_frame()
        result = indicators.calculate_indicators(
            frame,
            ma_periods=(7, 14, 28),
            mavol_fast=9,
            mavol_slow=18,
            adx_period=14,
            atr_period=14,
        )

        pd.testing.assert_series_equal(result["ma7"], frame["close"].rolling(7).mean(), check_names=False)
        pd.testing.assert_series_equal(result["mavol_fast"], frame["volume"].rolling(9).mean(), check_names=False)
        pd.testing.assert_series_equal(result["mavol_slow"], frame["volume"].rolling(18).mean(), check_names=False)
        pd.testing.assert_series_equal(
            result["volume_ma_ratio"],
            result["mavol_fast"] / result["mavol_slow"],
            check_names=False,
        )
        pd.testing.assert_series_equal(
            result["current_volume_ratio"],
            frame["volume"] / result["mavol_slow"],
            check_names=False,
        )
        pd.testing.assert_series_equal(result["volume_ratio"], result["volume_ma_ratio"], check_names=False)

    def test_fresh_cross_lookback_inspects_exactly_closed_window(self) -> None:
        index = pd.date_range("2026-01-01T00:00:00Z", periods=5, freq="15min")
        outside_cross = pd.DataFrame(
            {
                "ma7": [0.9, 1.1, 1.2, 1.3, 1.3],
                "ma14": [1.0, 1.0, 1.0, 1.0, 1.0],
            },
            index=index,
        )
        inside_cross = pd.DataFrame(
            {
                "ma7": [1.2, 0.9, 1.1, 1.2, 1.2],
                "ma14": [1.0, 1.0, 1.0, 1.0, 1.0],
            },
            index=index,
        )

        self.assertFalse(indicators.detect_fresh_cross(outside_cross, "LONG", lookback=3))
        self.assertTrue(indicators.detect_fresh_cross(inside_cross, "LONG", lookback=3))

    def test_indicator_calculation_is_stateless_under_concurrent_parameters(self) -> None:
        frame = realistic_fx_frame()

        def calculate(ma_periods: tuple[int, ...]) -> tuple[tuple[str, ...], float]:
            result = indicators.calculate_indicators(
                frame,
                ma_periods=ma_periods,
                mavol_fast=5,
                mavol_slow=10,
                adx_period=14,
                atr_period=14,
            )
            return tuple(sorted(column for column in result.columns if column.startswith("ma") and column[2:].isdigit())), float(
                result[f"ma{ma_periods[0]}"].iloc[-1]
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            first, second = pool.map(calculate, [(5, 10, 20), (7, 14, 28)])

        self.assertEqual(first[0], ("ma10", "ma20", "ma5"))
        self.assertEqual(second[0], ("ma14", "ma28", "ma7"))
        self.assertNotEqual(first[1], second[1])


if __name__ == "__main__":
    unittest.main()
