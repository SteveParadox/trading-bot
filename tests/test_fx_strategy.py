from __future__ import annotations

import unittest
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import indicators
from fxbot.config import StrategySettings
from fxbot.instruments import FxInstrument
from fxbot.models import Side
from fxbot.strategy import build_signal_intent, evaluate_signal_frame, last_closed_row, prepare_indicators


def trending_frame(
    start: float,
    step: float,
    rows: int = 120,
    *,
    freq: str = "15min",
    start_time: str = "2026-01-06T07:00:00Z",
) -> pd.DataFrame:
    index = pd.date_range(start_time, periods=rows, freq=freq)
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
        self.assertEqual(decision.details["signal_time"], entry.index[-1].isoformat())
        self.assertIn("entry_extension_atr", decision.details)
        self.assertIn("signal_close", decision.details)

    def test_uses_last_closed_candle_for_signal_decision(self) -> None:
        frame = trending_frame(1.08, 0.00025, rows=140)
        decision_time = frame.index[-1] + pd.Timedelta(minutes=10)
        frame.loc[frame.index[-1], "close"] = 1.07
        frame.loc[frame.index[-1], "open"] = 1.07
        frame.loc[frame.index[-1], "high"] = 1.08
        frame.loc[frame.index[-1], "low"] = 1.06
        frame.loc[frame.index[-1], "volume"] = 10

        entry = prepare_indicators(frame)
        htf = prepare_indicators(trending_frame(1.06, 0.0005, rows=140))

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
            timestamp=decision_time.to_pydatetime(),
        )

        self.assertEqual(decision.signal, Side.LONG)
        self.assertEqual(decision.reason, "signal_confirmed")
        self.assertEqual(decision.details["signal_time"], entry.index[-2].isoformat())

    def test_timestamp_exactly_at_close_selects_that_candle(self) -> None:
        frame = prepare_indicators(trending_frame(1.08, 0.00025, rows=80))
        decision_time = frame.index[-1] + pd.Timedelta(minutes=15)

        row = last_closed_row(frame, "15m", timestamp=decision_time)

        self.assertIsNotNone(row)
        self.assertEqual(row.name, frame.index[-1])

    def test_timestamp_immediately_before_close_selects_previous_candle(self) -> None:
        frame = prepare_indicators(trending_frame(1.08, 0.00025, rows=80))
        decision_time = frame.index[-1] + pd.Timedelta(minutes=15) - pd.Timedelta(microseconds=1)

        row = last_closed_row(frame, "15m", timestamp=decision_time)

        self.assertIsNotNone(row)
        self.assertEqual(row.name, frame.index[-2])

    def test_naive_timestamps_are_treated_as_utc(self) -> None:
        frame = trending_frame(1.08, 0.00025, rows=80)
        frame.index = frame.index.tz_localize(None)
        prepared = prepare_indicators(frame)
        decision_time = datetime(2026, 1, 7, 3, 0)

        row = last_closed_row(prepared, "15m", timestamp=decision_time)

        self.assertIsNotNone(row)
        self.assertEqual(str(row.name.tz), "UTC")

    def test_duplicate_timestamps_are_rejected(self) -> None:
        frame = trending_frame(1.08, 0.00025, rows=120)
        frame = pd.concat([frame, frame.iloc[[-1]]])

        decision = evaluate_signal_frame(
            frame,
            trending_frame(1.06, 0.0005, rows=120),
            instrument=FxInstrument("EUR_USD"),
            settings=StrategySettings(min_atr_pips=0.1, max_atr_pips=30, adx_min=10, htf_adx_min=10),
        )

        self.assertIsNone(decision.signal)
        self.assertEqual(decision.reason, "duplicate_timestamps")

    def test_reordering_input_rows_is_normalized_before_indicators(self) -> None:
        entry = trending_frame(1.08, 0.00025, rows=140)
        htf = trending_frame(1.06, 0.0005, rows=140)
        settings = StrategySettings(min_atr_pips=0.1, max_atr_pips=30, adx_min=10, htf_adx_min=10)

        ordered = evaluate_signal_frame(
            entry,
            htf,
            instrument=FxInstrument("EUR_USD"),
            settings=settings,
        )
        shuffled = evaluate_signal_frame(
            entry.sample(frac=1, random_state=7),
            htf.sample(frac=1, random_state=8),
            instrument=FxInstrument("EUR_USD"),
            settings=settings,
        )

        self.assertEqual(shuffled.signal, ordered.signal)
        self.assertEqual(shuffled.reason, ordered.reason)
        self.assertEqual(shuffled.details.get("signal_time"), ordered.details.get("signal_time"))

    def test_appending_forming_candle_does_not_change_previous_signal(self) -> None:
        closed_entry = prepare_indicators(trending_frame(1.08, 0.00025, rows=140))
        closed_htf = prepare_indicators(trending_frame(1.06, 0.0005, rows=140))
        forming_time = closed_entry.index[-1] + pd.Timedelta(minutes=15)
        decision_time = forming_time + pd.Timedelta(minutes=5)
        settings = StrategySettings(min_atr_pips=0.1, max_atr_pips=30, adx_min=10, htf_adx_min=10)
        baseline = evaluate_signal_frame(
            closed_entry,
            closed_htf,
            instrument=FxInstrument("EUR_USD"),
            settings=settings,
            timestamp=forming_time.to_pydatetime(),
        )

        forming = closed_entry.iloc[-1].copy()
        forming.name = forming_time
        forming[["open", "high", "low", "close", "volume"]] = [1.30, 1.40, 1.00, 1.01, 1_000_000.0]
        with_forming = pd.concat([closed_entry, forming.to_frame().T])
        changed_forming = with_forming.copy()
        changed_forming.loc[forming_time, ["open", "high", "low", "close", "volume"]] = [0.90, 1.60, 0.80, 1.55, 1.0]

        first = evaluate_signal_frame(
            with_forming,
            closed_htf,
            instrument=FxInstrument("EUR_USD"),
            settings=settings,
            timestamp=decision_time.to_pydatetime(),
        )
        second = evaluate_signal_frame(
            changed_forming,
            closed_htf,
            instrument=FxInstrument("EUR_USD"),
            settings=settings,
            timestamp=decision_time.to_pydatetime(),
        )

        self.assertEqual(first.signal, baseline.signal)
        self.assertEqual(second.signal, baseline.signal)
        self.assertEqual(first.details.get("signal_time"), baseline.details.get("signal_time"))
        self.assertEqual(second.details.get("signal_time"), baseline.details.get("signal_time"))

    def test_calculate_indicators_accepts_runtime_parameters_without_global_mutation(self) -> None:
        original = {
            "ma_periods": indicators.MA_PERIODS[:],
            "mavol_fast": indicators.MAVOL_FAST,
            "mavol_slow": indicators.MAVOL_SLOW,
            "adx_period": indicators.ADX_PERIOD,
            "atr_period": indicators.ATR_PERIOD,
        }
        try:
            result = indicators.calculate_indicators(
                trending_frame(1.08, 0.00025, rows=140),
                ma_periods=(7, 14, 28),
                mavol_fast=9,
                mavol_slow=18,
                adx_period=14,
                atr_period=14,
            )
            self.assertIn("ma7", result.columns)
            self.assertEqual(indicators.MA_PERIODS, original["ma_periods"])
            self.assertEqual(indicators.MAVOL_FAST, original["mavol_fast"])
            self.assertEqual(indicators.MAVOL_SLOW, original["mavol_slow"])
            self.assertEqual(indicators.ADX_PERIOD, original["adx_period"])
            self.assertEqual(indicators.ATR_PERIOD, original["atr_period"])
        finally:
            indicators.MA_PERIODS = original["ma_periods"]
            indicators.MAVOL_FAST = original["mavol_fast"]
            indicators.MAVOL_SLOW = original["mavol_slow"]
            indicators.ADX_PERIOD = original["adx_period"]
            indicators.ATR_PERIOD = original["atr_period"]

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

    def test_htf_momentum_candle_is_configurable(self) -> None:
        entry = prepare_indicators(trending_frame(1.08, 0.00025, rows=140))
        htf_raw = trending_frame(1.06, 0.0005, rows=140)
        htf_raw.loc[htf_raw.index[-1], "open"] = float(htf_raw.iloc[-1]["close"]) + 0.01
        htf_raw.loc[htf_raw.index[-1], "high"] = float(htf_raw.iloc[-1]["open"])
        htf = prepare_indicators(htf_raw)

        enabled = evaluate_signal_frame(
            entry,
            htf,
            instrument=FxInstrument("EUR_USD"),
            settings=StrategySettings(
                min_atr_pips=0.1,
                max_atr_pips=30,
                adx_min=10,
                htf_adx_min=10,
                htf_require_momentum_candle=True,
            ),
        )
        disabled = evaluate_signal_frame(
            entry,
            htf,
            instrument=FxInstrument("EUR_USD"),
            settings=StrategySettings(
                min_atr_pips=0.1,
                max_atr_pips=30,
                adx_min=10,
                htf_adx_min=10,
                htf_require_momentum_candle=False,
            ),
        )

        self.assertIsNone(enabled.signal)
        self.assertEqual(enabled.reason, "htf_momentum_filter")
        self.assertEqual(disabled.signal, Side.LONG)

    def test_entry_extension_filter_is_optional_and_direction_aware(self) -> None:
        entry = prepare_indicators(trending_frame(1.08, 0.00025, rows=140))
        htf = prepare_indicators(trending_frame(1.06, 0.0005, rows=140))
        disabled = evaluate_signal_frame(
            entry,
            htf,
            instrument=FxInstrument("EUR_USD"),
            settings=StrategySettings(min_atr_pips=0.1, max_atr_pips=30, adx_min=10, htf_adx_min=10),
        )
        enabled = evaluate_signal_frame(
            entry,
            htf,
            instrument=FxInstrument("EUR_USD"),
            settings=StrategySettings(
                min_atr_pips=0.1,
                max_atr_pips=30,
                adx_min=10,
                htf_adx_min=10,
                max_entry_extension_atr=0.5,
            ),
        )

        self.assertEqual(disabled.signal, Side.LONG)
        self.assertGreater(disabled.details["entry_extension_atr"], 0.5)
        self.assertIsNone(enabled.signal)
        self.assertEqual(enabled.reason, "entry_extension_filter")

    def test_jpy_instrument_changes_pip_units_not_price_indicators(self) -> None:
        entry = prepare_indicators(trending_frame(1.08, 0.00025, rows=140))
        htf = prepare_indicators(trending_frame(1.06, 0.0005, rows=140))
        settings = StrategySettings(min_atr_pips=0.001, max_atr_pips=300, adx_min=10, htf_adx_min=10)

        eur = evaluate_signal_frame(
            entry,
            htf,
            instrument=FxInstrument("EUR_USD", pip_location=-4),
            settings=settings,
        )
        jpy = evaluate_signal_frame(
            entry,
            htf,
            instrument=FxInstrument("USD_JPY", pip_location=-2),
            settings=settings,
        )

        self.assertEqual(eur.signal, jpy.signal)
        self.assertAlmostEqual(eur.details["atr_price"], jpy.details["atr_price"])
        self.assertAlmostEqual(eur.details["atr_pips"] / 100.0, jpy.details["atr_pips"])

    def test_signal_score_is_deterministic_and_not_coupled_to_max_stop_pips(self) -> None:
        entry = prepare_indicators(trending_frame(1.08, 0.00025, rows=140))
        htf = prepare_indicators(trending_frame(1.06, 0.0005, rows=140))
        common = dict(min_atr_pips=0.1, max_atr_pips=30, adx_min=10, htf_adx_min=10)

        first = build_signal_intent(
            entry,
            htf,
            instrument=FxInstrument("EUR_USD"),
            settings=StrategySettings(**common, max_stop_pips=20),
            entry_price=1.12,
            timestamp=datetime(2026, 1, 8, 0, 0, tzinfo=timezone.utc),
        )
        second = build_signal_intent(
            entry,
            htf,
            instrument=FxInstrument("EUR_USD"),
            settings=StrategySettings(**common, max_stop_pips=200),
            entry_price=1.12,
            timestamp=datetime(2026, 1, 8, 0, 0, tzinfo=timezone.utc),
        )

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(first.score, second.score)
        self.assertGreaterEqual(first.score, 0)
        self.assertLessEqual(first.score, 100)
        self.assertEqual(first.metadata["signal_close"], first.signal_row["close"])
        self.assertEqual(first.metadata["entry_price_source"], "provided")

    def test_rejects_technically_valid_but_low_quality_signal(self) -> None:
        entry = prepare_indicators(trending_frame(1.08, 0.00025, rows=140))
        htf = prepare_indicators(trending_frame(1.06, 0.0005, rows=140))

        decision = evaluate_signal_frame(
            entry,
            htf,
            instrument=FxInstrument("EUR_USD"),
            settings=StrategySettings(
                min_atr_pips=0.1,
                max_atr_pips=30,
                adx_min=10,
                htf_adx_min=10,
                min_signal_score=90,
            ),
        )

        self.assertIsNone(decision.signal)
        self.assertEqual(decision.reason, "signal_quality_filter")
        self.assertLess(decision.details["quality_score"], 90)

    def test_rejects_weak_di_separation(self) -> None:
        entry = prepare_indicators(trending_frame(1.08, 0.00025, rows=140))
        htf = prepare_indicators(trending_frame(1.06, 0.0005, rows=140))

        decision = evaluate_signal_frame(
            entry,
            htf,
            instrument=FxInstrument("EUR_USD"),
            settings=StrategySettings(
                min_atr_pips=0.1,
                max_atr_pips=30,
                adx_min=10,
                htf_adx_min=10,
                min_di_edge=25,
                min_signal_score=0,
            ),
        )

        self.assertIsNone(decision.signal)
        self.assertEqual(decision.reason, "di_edge_filter")

    def test_rejects_weakening_adx_and_flat_ma28(self) -> None:
        entry = prepare_indicators(trending_frame(1.08, 0.00025, rows=140))
        htf = prepare_indicators(trending_frame(1.06, 0.0005, rows=140))
        weakening = entry.copy()
        weakening.iloc[-1, weakening.columns.get_loc("adx")] = 90.0
        settings = StrategySettings(
            min_atr_pips=0.1,
            max_atr_pips=30,
            adx_min=10,
            htf_adx_min=10,
            min_signal_score=0,
        )

        adx_decision = evaluate_signal_frame(
            weakening,
            htf,
            instrument=FxInstrument("EUR_USD"),
            settings=settings,
        )
        self.assertIsNone(adx_decision.signal)
        self.assertEqual(adx_decision.reason, "adx_weakening_filter")

        flat_slope = entry.copy()
        flat_slope.iloc[-4, flat_slope.columns.get_loc("ma28")] = float(flat_slope.iloc[-1]["ma28"])
        slope_decision = evaluate_signal_frame(
            flat_slope,
            htf,
            instrument=FxInstrument("EUR_USD"),
            settings=settings,
        )
        self.assertIsNone(slope_decision.signal)
        self.assertEqual(slope_decision.reason, "ma28_slope_filter")

    def test_tick_volume_does_not_change_quality_score_when_volume_confirmation_is_off(self) -> None:
        entry = prepare_indicators(trending_frame(1.08, 0.00025, rows=140))
        htf = prepare_indicators(trending_frame(1.06, 0.0005, rows=140))
        no_participation = entry.copy()
        no_participation.iloc[-1, no_participation.columns.get_loc("current_volume_ratio")] = 0.01
        no_participation.iloc[-1, no_participation.columns.get_loc("volume_ma_ratio")] = 0.01
        settings = StrategySettings(
            min_atr_pips=0.1,
            max_atr_pips=30,
            adx_min=10,
            htf_adx_min=10,
            min_signal_score=0,
            require_volume_confirmation=False,
        )

        normal = build_signal_intent(
            entry,
            htf,
            instrument=FxInstrument("EUR_USD"),
            settings=settings,
            entry_price=1.12,
        )
        altered = build_signal_intent(
            no_participation,
            htf,
            instrument=FxInstrument("EUR_USD"),
            settings=settings,
            entry_price=1.12,
        )

        self.assertIsNotNone(normal)
        self.assertIsNotNone(altered)
        self.assertEqual(normal.score, altered.score)


if __name__ == "__main__":
    unittest.main()
