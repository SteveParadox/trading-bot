"""H4 primary trend, H1 structure, M5 execution regression coverage."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import unittest

import numpy as np
import pandas as pd

from fxbot.config import StrategySettings
from fxbot.instruments import FxInstrument
from fxbot.models import Side
from fxbot.strategy import build_signal_intent, evaluate_signal_frame


DECISION_TIME = datetime(2026, 1, 8, 12, 4, tzinfo=timezone.utc)


def trend(freq: str, last_open: str, step: float, start: float = 1.08) -> pd.DataFrame:
    index = pd.date_range(end=pd.Timestamp(last_open), periods=140, freq=freq)
    closes = start + np.arange(140) * step
    opens = closes - step / 2
    return pd.DataFrame(
        {"open": opens, "high": np.maximum(opens, closes) + abs(step) * 2,
         "low": np.minimum(opens, closes) - abs(step) * 2,
         "close": closes, "volume": np.linspace(100, 150, 140)},
        index=index,
    )


class MultiTimeframeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = StrategySettings(
            entry_timeframe="5m", structure_timeframe="1h", htf_timeframe="4h",
            adx_min=10, htf_adx_min=10, structure_adx_min=10,
            min_atr_pips=0.01, max_atr_pips=100,
            min_signal_score=0, min_di_edge=0,
            require_adx_non_decreasing=False, require_ma28_slope=False,
            max_entry_extension_atr=None,
        )
        self.entry = trend("5min", "2026-01-08T11:55:00Z", 0.00008)
        self.structure = trend("1h", "2026-01-08T11:00:00Z", 0.0003)
        self.htf = trend("4h", "2026-01-08T08:00:00Z", 0.0007)

    def decide(self, structure_frame=None, *, supply=True):
        return evaluate_signal_frame(
            self.entry, self.htf, instrument=FxInstrument("EUR_USD"),
            settings=self.settings, timestamp=DECISION_TIME,
            structure_frame=(self.structure if structure_frame is None else structure_frame) if supply else None,
        )

    def test_h4_h1_m5_alignment_confirms_and_records_times(self):
        decision = self.decide()
        self.assertEqual(decision.signal, Side.LONG, decision.reason)
        self.assertEqual(decision.details["htf_timeframe"], "4h")
        self.assertEqual(decision.details["structure_timeframe"], "1h")
        self.assertEqual(decision.details["entry_timeframe"], "5m")
        self.assertEqual(decision.details["structure_signal_time"], self.structure.index[-1].isoformat())
        intent = build_signal_intent(
            self.entry, self.htf, instrument=FxInstrument("EUR_USD"),
            settings=self.settings, entry_price=1.1, timestamp=DECISION_TIME,
            structure_frame=self.structure,
        )
        self.assertIsNotNone(intent)
        self.assertEqual(intent.metadata["details"]["structure_timeframe"], "1h")

    def test_opposite_h1_blocks_h4_trade(self):
        opposing_h1 = trend("1h", "2026-01-08T11:00:00Z", -0.0003, start=1.15)
        decision = self.decide(opposing_h1)
        self.assertIsNone(decision.signal)
        self.assertEqual(decision.reason, "structure_conflict")

    def test_missing_h1_fails_closed(self):
        decision = self.decide(supply=False)
        self.assertIsNone(decision.signal)
        self.assertEqual(decision.reason, "structure_frame_unavailable")

    def test_unfinished_h4_h1_bars_do_not_repaint(self):
        baseline = self.decide()
        self.assertEqual(baseline.signal, Side.LONG)
        last_h4 = self.htf.iloc[-1].copy()
        self.htf.loc[pd.Timestamp("2026-01-08T12:00:00Z")] = last_h4
        self.htf.loc[pd.Timestamp("2026-01-08T12:00:00Z"), "close"] = 0.5
        last_h1 = self.structure.iloc[-1].copy()
        self.structure.loc[pd.Timestamp("2026-01-08T12:00:00Z")] = last_h1
        self.structure.loc[pd.Timestamp("2026-01-08T12:00:00Z"), "close"] = 0.5
        confirmed = self.decide()
        self.assertEqual(confirmed.signal, Side.LONG, confirmed.reason)
        self.assertEqual(baseline.details["htf_signal_time"], confirmed.details["htf_signal_time"])
        self.assertEqual(baseline.details["structure_signal_time"], confirmed.details["structure_signal_time"])

    def test_stale_h1_and_h4_bars_block_replay(self):
        previous_h1 = self.structure
        self.structure = trend("1h", "2026-01-08T09:00:00Z", 0.0003)
        stale_h1 = self.decide()
        self.assertIsNone(stale_h1.signal)
        self.assertEqual(stale_h1.reason, "stale_closed_candles")
        self.assertEqual(stale_h1.details["timeframe"], "1h")
        self.structure = previous_h1
        self.htf = trend("4h", "2026-01-08T00:00:00Z", 0.0007)
        stale_h4 = self.decide()
        self.assertIsNone(stale_h4.signal)
        self.assertEqual(stale_h4.reason, "stale_closed_candles")
        self.assertEqual(stale_h4.details["timeframe"], "4h")

    def test_short_direction_requires_h1_slope_confirmation(self):
        self.entry = trend("5min", "2026-01-08T11:55:00Z", -0.00008, start=1.2)
        self.structure = trend("1h", "2026-01-08T11:00:00Z", -0.0003, start=1.2)
        self.htf = trend("4h", "2026-01-08T08:00:00Z", -0.0007, start=1.25)
        result = self.decide()
        self.assertEqual(result.signal, Side.SHORT, result.reason)
        self.assertGreater(result.details["structure"]["directional_ma28_slope_atr"], 0)

    def test_revalidation_requires_matching_candle_ids_and_signal(self):
        from types import SimpleNamespace
        from fxbot.forward import ForwardTestWorker

        instrument = FxInstrument("EUR_USD")
        intent = build_signal_intent(
            self.entry, self.htf, instrument=instrument,
            settings=self.settings, entry_price=1.1, timestamp=DECISION_TIME,
            structure_frame=self.structure,
        )
        self.assertIsNotNone(intent)
        frames = {"5m": self.entry, "1h": self.structure, "4h": self.htf}
        worker = object.__new__(ForwardTestWorker)
        worker.settings = SimpleNamespace(
            strategy=self.settings, runtime=SimpleNamespace(max_price_age_seconds=30)
        )
        worker.client = SimpleNamespace(candles=lambda _instrument, timeframe, _count: frames[timeframe])
        self.assertTrue(worker._timeframe_evidence_still_current(intent, instrument, DECISION_TIME))
        metadata = dict(intent.metadata)
        metadata["details"] = dict(metadata["details"])
        metadata["details"]["htf_signal_time"] = "2026-01-01T00:00:00+00:00"
        self.assertFalse(worker._timeframe_evidence_still_current(
            replace(intent, metadata=metadata), instrument, DECISION_TIME
        ))
        metadata["details"]["htf_signal_time"] = intent.metadata["details"]["htf_signal_time"]
        metadata["details"]["structure_signal_time"] = "2026-01-01T00:00:00+00:00"
        self.assertFalse(worker._timeframe_evidence_still_current(
            replace(intent, metadata=metadata), instrument, DECISION_TIME
        ))
        frames["1h"] = trend("1h", "2026-01-08T11:00:00Z", -0.0003, start=1.15)
        self.assertFalse(worker._timeframe_evidence_still_current(intent, instrument, DECISION_TIME))

    def test_h4_default_disables_only_momentum_candle_gate(self):
        self.assertFalse(StrategySettings(htf_timeframe="4h").htf_require_momentum_candle)
        self.assertTrue(StrategySettings(htf_timeframe="1h").htf_require_momentum_candle)
        self.assertTrue(StrategySettings(
            htf_timeframe="4h", htf_require_momentum_candle=True
        ).htf_require_momentum_candle)

    def test_timeframe_order_validation_and_legacy_mode(self):
        with self.assertRaisesRegex(ValueError, "entry < structure < HTF"):
            replace(self.settings, structure_timeframe="4h")
        with self.assertRaisesRegex(ValueError, "entry < structure < HTF"):
            replace(self.settings, structure_timeframe="5m")
        legacy = replace(self.settings, structure_timeframe=None)
        result = evaluate_signal_frame(
            self.entry, self.htf, instrument=FxInstrument("EUR_USD"),
            settings=legacy, timestamp=DECISION_TIME,
        )
        self.assertEqual(result.signal, Side.LONG)
