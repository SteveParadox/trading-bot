from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import indicators
from fxbot.config import StrategySettings
from fxbot.instruments import FxInstrument
from fxbot.models import Side
from fxbot.strategy import evaluate_signal_frame, prepare_indicators


def trending_frame(start: float, step: float, rows: int = 120) -> pd.DataFrame:
    index = pd.date_range("2026-01-06T07:00:00Z", periods=rows, freq="15min")
    close = start + np.arange(rows) * step
    open_ = close - step * 0.5
    high = np.maximum(open_, close) + abs(step) * 2
    low = np.minimum(open_, close) - abs(step) * 2
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.linspace(100, 150, rows),
        },
        index=index,
    )


class FxStrategyTests(unittest.TestCase):
    def test_prepare_indicators_ignore_legacy_global_config(self) -> None:
        original = {
            "ma_periods": indicators.MA_PERIODS[:],
            "mavol_fast": indicators.MAVOL_FAST,
            "mavol_slow": indicators.MAVOL_SLOW,
            "adx_period": indicators.ADX_PERIOD,
        }
        try:
            indicators.MA_PERIODS = [20, 40, 60]
            indicators.MAVOL_FAST = 25
            indicators.MAVOL_SLOW = 50
            indicators.ADX_PERIOD = 7
            frame = prepare_indicators(trending_frame(1.08, 0.00025))
            self.assertIn("ma7", frame.columns)
            self.assertIn("ma14", frame.columns)
            self.assertIn("ma28", frame.columns)
            self.assertIn("adx", frame.columns)
            self.assertIn("volume_ratio", frame.columns)
        finally:
            indicators.MA_PERIODS = original["ma_periods"]
            indicators.MAVOL_FAST = original["mavol_fast"]
            indicators.MAVOL_SLOW = original["mavol_slow"]
            indicators.ADX_PERIOD = original["adx_period"]

    def test_confirms_fx_trend_without_volume_confirmation(self) -> None:
        entry = prepare_indicators(trending_frame(1.08, 0.00025))
        htf = prepare_indicators(trending_frame(1.06, 0.0005))

        decision = evaluate_signal_frame(
            entry,
            htf,
            instrument=FxInstrument("EUR_USD"),
            settings=StrategySettings(
                require_volume_confirmation=False,
                min_atr_pips=0.1,
                max_atr_pips=20,
                adx_min=10,
                htf_adx_min=10,
            ),
        )

        self.assertEqual(decision.signal, Side.LONG)
        self.assertEqual(decision.reason, "signal_confirmed")

    def test_blocks_when_htf_conflicts(self) -> None:
        entry = prepare_indicators(trending_frame(1.08, 0.00025))
        htf = prepare_indicators(trending_frame(1.12, -0.0005))

        decision = evaluate_signal_frame(
            entry,
            htf,
            instrument=FxInstrument("EUR_USD"),
            settings=StrategySettings(min_atr_pips=0.1, max_atr_pips=30, adx_min=10, htf_adx_min=10),
        )

        self.assertIsNone(decision.signal)
        self.assertEqual(decision.reason, "htf_conflict")


if __name__ == "__main__":
    unittest.main()

