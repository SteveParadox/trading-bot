import asyncio
from dataclasses import replace
from datetime import timedelta
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest

from fxbot.config import FxBotSettings, RuntimeSettings, StrategySettings, RiskSettings
from fxbot.config import settings_from_env
from fxbot.forward import ForwardTestWorker, candles_are_current
from fxbot.instruments import FxInstrument, PriceSnapshot
from fxbot.journal import StructuredJournal
from fxbot.models import BotRunState, FxPortfolioState, FxSignalIntent, Side
from fxbot.mt5 import Mt5Error
from fxbot.operations import freshness
from fxbot.risk import FxRiskManager
from test_fx_forward import FakeMt5Client, FIXED_NOW, FixedDatetime, trending_frame


@pytest.fixture
def worker(tmp_path):
    settings = FxBotSettings(instruments=["EUR_USD"], runtime=RuntimeSettings(
        database_url=f"sqlite:///{tmp_path / 'journal.db'}", log_jsonl_path=None))
    journal = StructuredJournal(settings.runtime.database_url)
    client = FakeMt5Client(entry_frame=trending_frame(1.08, .00025))
    result = ForwardTestWorker(settings, client=client, journal=journal)
    yield result
    result.close()


@pytest.mark.parametrize("state", [BotRunState.PAUSED, BotRunState.HALTED])
def test_paused_and_halted_manage_without_new_entries(worker, state):
    worker.journal.set_state(state, "operator")
    worker._sync_open_trades = Mock()
    worker._scan_instrument = Mock()
    with patch("fxbot.forward.datetime", FixedDatetime):
        worker.scan_once()
    worker._sync_open_trades.assert_called_once()
    worker._scan_instrument.assert_not_called()
    assert worker.journal.get_state().state == state.value


def test_stale_other_symbol_does_not_disable_fresh_position_management(worker):
    fresh = worker.client.price
    stale = PriceSnapshot("GBP_USD", 1.2, 1.2001, FIXED_NOW - timedelta(hours=1))
    worker.settings = replace(worker.settings, instruments=["EUR_USD", "GBP_USD"])
    snapshot = worker.client.pricing([])
    snapshot.prices = {"EUR_USD": fresh, "GBP_USD": stale}
    worker.client.pricing = Mock(return_value=snapshot)
    worker._sync_open_trades = Mock()
    worker._scan_instrument = Mock()
    worker.journal.set_state(BotRunState.RUNNING)
    with patch("fxbot.forward.datetime", FixedDatetime):
        worker.scan_once()
    assert worker._sync_open_trades.call_args.args[2] == {"EUR_USD": fresh}
    worker._scan_instrument.assert_not_called()


@pytest.mark.parametrize("state", [BotRunState.RUNNING, BotRunState.PAUSED, BotRunState.HALTED])
def test_reconnect_retries_without_overwriting_operator_state(worker, state):
    worker.journal.set_state(state, "operator")
    worker.scan_once = Mock()
    def scan():
        if worker.scan_once.call_count == 1:
            raise Mt5Error("offline")
        worker.stop()
    worker.scan_once.side_effect = scan
    with patch("fxbot.forward.asyncio.sleep", new_callable=AsyncMock) as sleep:
        asyncio.run(worker.run_forever())
    assert worker.scan_once.call_count == 2
    assert sleep.call_args_list[0].args[0] == worker.settings.runtime.loop_interval_seconds * 2
    assert worker.journal.get_state().state == state.value


@pytest.mark.parametrize("offset,allowed", [(0, True), (-3600, False), (1, False)])
def test_closed_candle_age_uses_actual_utc(offset, allowed):
    index = pd.date_range(end=FIXED_NOW - timedelta(minutes=15) + timedelta(seconds=offset), periods=4, freq="15min")
    frame = pd.DataFrame({"close": [1.1] * 4}, index=index)
    assert candles_are_current(frame, "15m", FIXED_NOW, 120) is allowed


def test_future_tick_is_not_fresh():
    assert not freshness(FIXED_NOW, FIXED_NOW + timedelta(seconds=60), max_age_seconds=120)[0]


def test_entry_path_rejects_stale_history_with_fresh_quote(worker):
    worker.client.entry_frame.index -= pd.Timedelta(days=1)
    worker._skip = Mock()
    worker._scan_instrument(FIXED_NOW, FxInstrument("EUR_USD"), worker.client.price,
                            FxPortfolioState(10000, 10000, 0, 0), {}, None)
    assert worker._skip.call_args.args[2] == "stale_or_invalid_candles"
    assert not worker.client.created_orders


def test_submission_rechecks_pause(worker):
    worker.journal.set_state(BotRunState.PAUSED)
    intent = FxSignalIntent("EUR_USD", Side.LONG, FIXED_NOW, 1.1, {"atr": .001})
    risk = worker.risk.evaluate_intent(intent, FxInstrument("EUR_USD"), FxPortfolioState(10000, 10000, 0, 0))
    assert not worker._submit_idempotent(intent, FxInstrument("EUR_USD"), risk)
    assert not worker.client.created_orders


def test_same_signal_on_later_scan_cannot_submit_twice(worker):
    worker.settings = replace(worker.settings, strategy=StrategySettings(
        partial_tp_enabled=False, adx_min=10, htf_adx_min=10,
        min_atr_pips=.1, max_atr_pips=30, require_volume_confirmation=False))
    worker.risk = FxRiskManager(worker.settings.risk, worker.settings.strategy)
    worker.journal.set_state(BotRunState.RUNNING)
    for now in (FIXED_NOW, FIXED_NOW + timedelta(seconds=10)):
        worker._scan_instrument(now, FxInstrument("EUR_USD"), worker.client.price,
                               FxPortfolioState(10000, 10000, 0, 0), {}, None)
    assert len(worker.client.created_orders) == 1


@pytest.mark.parametrize("hedging", [False, True])
def test_split_requires_confirmed_hedging(worker, hedging):
    worker._hedging_enabled = hedging
    intent = FxSignalIntent("EUR_USD", Side.LONG, FIXED_NOW, 1.1, {"atr": .001})
    instrument = FxInstrument("EUR_USD")
    risk = worker.risk.evaluate_intent(intent, instrument, FxPortfolioState(10000, 10000, 0, 0))
    legs = worker._order_legs(intent, risk, instrument)
    assert len(legs) == (2 if hedging else 1)
    if not hedging:
        assert legs[0][2] == risk.exit_plan.take_profit


def test_small_account_explains_cap_without_rounding_up():
    intent = FxSignalIntent("EUR_USD", Side.LONG, FIXED_NOW, 1.1, {"atr": .001})
    manager = FxRiskManager(RiskSettings(max_pair_exposure_pct=.35), StrategySettings())
    risk = manager.evaluate_intent(intent, FxInstrument("EUR_USD", minimum_trade_size=1000), FxPortfolioState(100, 100, 0, 0))
    assert risk.reason == "units_below_minimum"
    assert risk.metadata["binding_constraint"] == "pair_exposure"
    assert risk.metadata["minimum_units"] == 1000
    assert risk.metadata["raw_units"] < 32


def test_costs_reduce_size_and_reject_unprofitable_target():
    intent = FxSignalIntent("EUR_USD", Side.LONG, FIXED_NOW, 1.1, {"atr": .001})
    instrument = FxInstrument("EUR_USD")
    portfolio = FxPortfolioState(10000, 10000, 0, 0)
    settings = RiskSettings(max_pair_exposure_pct=10, max_gross_exposure_pct=10, max_currency_exposure_pct=10)
    def decision(cost):
        return FxRiskManager(settings, StrategySettings(execution_cost_pips_round_trip=cost)).evaluate_intent(intent, instrument, portfolio)
    assert decision(2).units < decision(0).units
    assert decision(50).reason == "reward_below_execution_cost"


@pytest.mark.parametrize("side", [Side.LONG, Side.SHORT])
def test_breakeven_covers_configured_cost_and_broker_distance(worker, side):
    worker.settings = replace(worker.settings, strategy=StrategySettings(execution_cost_pips_round_trip=2))
    instrument = FxInstrument("EUR_USD", minimum_stop_distance=.0005)
    trade = {"id": "42", "price": 1.1, "currentUnits": 1000 * side.sign,
             "stopLossOrder": {"price": 1.1 - .001 * side.sign}}
    exit_price = 1.1 + .002 * side.sign
    price = PriceSnapshot("EUR_USD", exit_price, exit_price, FIXED_NOW)
    worker.client.set_trade_dependent_orders = Mock(return_value={})
    worker._maybe_move_stop_to_breakeven(trade, instrument, price)
    assert worker.client.set_trade_dependent_orders.call_args.kwargs["stop_loss"] == pytest.approx(1.1 + .00022 * side.sign)
    worker.client.set_trade_dependent_orders.reset_mock()
    trade["stopLossOrder"] = {"price": 1.1 - .001 * side.sign}
    worker._maybe_move_stop_to_breakeven(trade, replace(instrument, minimum_stop_distance=.003), price)
    worker.client.set_trade_dependent_orders.assert_not_called()


@pytest.mark.parametrize("cost", [-1, float("nan"), float("inf")])
def test_invalid_cost_configuration_rejected(cost):
    with pytest.raises(ValueError):
        StrategySettings(execution_cost_pips_round_trip=cost)


def test_risk_defaults_match_environment_loader(monkeypatch):
    for key in ("FX_MAX_PAIR_EXPOSURE_PCT", "FX_MAX_GROSS_EXPOSURE_PCT", "FX_MAX_CURRENCY_EXPOSURE_PCT"):
        monkeypatch.delenv(key, raising=False)
    actual = settings_from_env().risk
    for key in ("max_pair_exposure_pct", "max_gross_exposure_pct", "max_currency_exposure_pct"):
        assert getattr(actual, key) == getattr(RiskSettings(), key)
