from __future__ import annotations

import unittest

import pandas as pd

from backtester.config import BacktestConfig
from backtester.models import Side, SignalIntent
from backtester.risk import RiskManager


class ExchangeRiskStub:
    cash = 1_000.0
    realized_pnl = 0.0

    def __init__(
        self,
        *,
        equity: float = 1_000.0,
        gross_exposure: float = 0.0,
        symbol_exposure: float = 0.0,
        margin_used: float = 0.0,
    ) -> None:
        self._equity = equity
        self._gross_exposure = gross_exposure
        self._symbol_exposure = symbol_exposure
        self._margin_used = margin_used

    def equity(self) -> float:
        return self._equity

    def open_position_count(self) -> int:
        return 0

    def has_position(self, symbol: str, side: Side | None = None) -> bool:
        return False

    def gross_exposure(self) -> float:
        return self._gross_exposure

    def symbol_exposure(self, symbol: str) -> float:
        return self._symbol_exposure

    def margin_used(self) -> float:
        return self._margin_used

    def portfolio_heat(self) -> float:
        return 0.0


def risk_config() -> BacktestConfig:
    return BacktestConfig.from_dict(
        {
            "data": {"symbols": ["BTCUSDT"], "base_timeframe": "5m", "timeframes": ["5m"]},
            "risk": {
                "initial_equity": 1_000.0,
                "risk_per_trade_pct": 0.005,
                "max_trade_notional": 1_000.0,
                "max_symbol_positions": 2,
                "max_symbol_exposure_pct": 0.35,
                "max_gross_exposure_pct": 2.0,
                "leverage": 2.0,
            },
            "instruments": {
                "BTCUSDT": {
                    "tick_size": 0.1,
                    "qty_step": 0.001,
                    "min_qty": 0.001,
                    "min_notional": 1.0,
                    "max_leverage": 50.0,
                }
            },
        }
    )


def long_intent() -> SignalIntent:
    return SignalIntent(
        symbol="BTCUSDT",
        side=Side.LONG,
        timestamp=pd.Timestamp("2025-01-01T00:00:00Z"),
        entry_price_hint=100.0,
        signal_row={"close": 100.0, "ma28": 95.0, "atr_pct": 0.01},
    )


class RiskManagerTests(unittest.TestCase):
    def test_sizing_respects_remaining_symbol_exposure(self) -> None:
        manager = RiskManager(risk_config())
        exchange = ExchangeRiskStub(symbol_exposure=300.0)

        decision = manager.evaluate_intent(
            long_intent(),
            exchange,
            pd.Timestamp("2025-01-01T00:00:00Z"),
        )

        self.assertTrue(decision.allowed)
        self.assertAlmostEqual(decision.notional, 50.0)
        self.assertAlmostEqual(decision.qty, 0.5)

    def test_sizing_rejects_when_free_equity_cannot_cover_margin(self) -> None:
        manager = RiskManager(risk_config())
        exchange = ExchangeRiskStub(margin_used=995.0)

        decision = manager.evaluate_intent(
            long_intent(),
            exchange,
            pd.Timestamp("2025-01-01T00:00:00Z"),
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "insufficient_free_equity_for_margin")


if __name__ == "__main__":
    unittest.main()
