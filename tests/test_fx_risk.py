from __future__ import annotations

import unittest
from datetime import datetime, timezone

from fxbot.config import RiskSettings, StrategySettings
from fxbot.instruments import FxInstrument, pip_value_per_unit_home
from fxbot.models import FxPortfolioState, FxSignalIntent, Side
from fxbot.risk import FxRiskManager


def portfolio() -> FxPortfolioState:
    return FxPortfolioState(
        equity=10_000.0,
        balance=10_000.0,
        margin_used=0.0,
        open_positions=0,
        portfolio_risk=0.0,
        gross_exposure=0.0,
    )


class FxRiskTests(unittest.TestCase):
    def test_eurusd_position_sizing_uses_pip_value_in_account_currency(self) -> None:
        instrument = FxInstrument("EUR_USD", pip_location=-4, margin_rate=0.0333333333)
        risk = RiskSettings(
            risk_per_trade_pct=0.01,
            max_units_per_trade=1_000_000,
            max_pair_exposure_pct=10.0,
            max_gross_exposure_pct=10.0,
            max_currency_exposure_pct=10.0,
        )
        strategy = StrategySettings(min_stop_pips=1, max_stop_pips=100, atr_sl_multiplier=1.0)
        intent = FxSignalIntent(
            instrument="EUR_USD",
            side=Side.LONG,
            timestamp=datetime(2026, 1, 6, 14, 0, tzinfo=timezone.utc),
            entry_price=1.1000,
            signal_row={"atr": 0.0020, "close": 1.1},
        )

        decision = FxRiskManager(risk, strategy).evaluate_intent(intent, instrument, portfolio())

        self.assertTrue(decision.allowed)
        self.assertEqual(decision.units, 50_000)
        self.assertAlmostEqual(decision.risk_amount, 100.0, places=5)

    def test_usdjpy_pip_value_converts_quote_currency_to_usd(self) -> None:
        instrument = FxInstrument("USD_JPY", pip_location=-2, margin_rate=0.04)

        value = pip_value_per_unit_home(instrument, "USD", 150.0)

        self.assertAlmostEqual(value, 0.01 / 150.0)

    def test_currency_exposure_cap_reduces_or_rejects_correlated_basket_risk(self) -> None:
        instrument = FxInstrument("GBP_USD", pip_location=-4, margin_rate=0.0333333333)
        risk = RiskSettings(
            risk_per_trade_pct=0.01,
            max_units_per_trade=1_000_000,
            max_currency_exposure_pct=0.50,
        )
        strategy = StrategySettings(min_stop_pips=1, max_stop_pips=100, atr_sl_multiplier=1.0)
        crowded = FxPortfolioState(
            equity=10_000,
            balance=10_000,
            margin_used=0,
            open_positions=1,
            gross_exposure=4_900,
            currency_exposures={"GBP": 4_900, "USD": -4_900},
        )
        intent = FxSignalIntent(
            instrument="GBP_USD",
            side=Side.LONG,
            timestamp=datetime(2026, 1, 6, 14, 0, tzinfo=timezone.utc),
            entry_price=1.2500,
            signal_row={"atr": 0.0010, "close": 1.25},
        )

        decision = FxRiskManager(risk, strategy).evaluate_intent(intent, instrument, crowded)

        self.assertTrue(decision.allowed)
        self.assertLessEqual(decision.position_value, 100.0)
        self.assertEqual(decision.units, 80)


if __name__ == "__main__":
    unittest.main()
