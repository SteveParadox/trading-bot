from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from fxbot.config import FxBotSettings, RiskSettings, StrategySettings, settings_from_env
from fxbot.forward import ForwardTestWorker
from fxbot.instruments import FxInstrument
from fxbot.models import FxSignalIntent, Side
from fxbot.risk import FxExitPlan, FxRiskDecision, FxRiskManager
from fxbot.strategy import evaluate_signal_frame, prepare_indicators
from test_fx_strategy import trending_frame


@pytest.mark.parametrize("bound", [{"min_atr_pips": 100}, {"max_atr_pips": 0.1}])
def test_single_atr_bound_rejects_outside_range(bound):
    entry = prepare_indicators(trending_frame(1.08, 0.00025))
    htf = prepare_indicators(trending_frame(1.06, 0.0005))
    instrument = FxInstrument("EUR_USD")
    assert evaluate_signal_frame(entry, htf, instrument=instrument, settings=StrategySettings()).signal
    decision = evaluate_signal_frame(entry, htf, instrument=instrument, settings=StrategySettings(**bound))
    assert decision.signal is None
    assert decision.reason == "atr_volatility_filter"


@pytest.mark.parametrize("side", [Side.LONG, Side.SHORT])
@pytest.mark.parametrize("bound", [{"min_stop_pips": 50}, {"max_stop_pips": 1}])
def test_single_stop_bound_rejects_outside_range(side, bound):
    intent = FxSignalIntent(instrument="EUR_USD", side=side,
                            timestamp=datetime.now(timezone.utc), entry_price=1.1,
                            signal_row={"atr": 0.001})
    instrument = FxInstrument("EUR_USD")
    assert FxRiskManager(RiskSettings(), StrategySettings()).build_exit_plan(intent, instrument)
    assert FxRiskManager(RiskSettings(), StrategySettings(**bound)).build_exit_plan(intent, instrument) is None


def test_additional_cors_origin_keeps_browser_case(monkeypatch):
    monkeypatch.setenv("FX_CORS_ORIGINS", " https://trading-bot-six-steel.vercel.app, https://other.example ")
    assert settings_from_env().runtime.cors_origins == (
        "https://trading-bot-six-steel.vercel.app", "https://other.example")


def test_split_falls_back_when_leg_is_below_broker_minimum():
    worker = SimpleNamespace(settings=FxBotSettings(), _hedging_enabled=True)
    instrument = FxInstrument("EUR_USD", minimum_trade_size=1000, trade_unit_step=100)
    intent = FxSignalIntent(instrument="EUR_USD", side=Side.LONG,
                            timestamp=datetime.now(timezone.utc), entry_price=1.1,
                            signal_row={"atr": 0.001})
    plan = FxExitPlan(1.098, 1.103, 0.002, 0.003, 1.5, 20, 30)
    risk = FxRiskDecision(True, "accepted", units=1500, exit_plan=plan)
    assert ForwardTestWorker._order_legs(worker, intent, risk, instrument) == [("full", 1500, 1.103)]
