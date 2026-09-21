from dataclasses import replace
from datetime import datetime, timedelta, timezone
from contextlib import closing
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from fxbot.config import FxBotSettings, StrategySettings, RiskSettings, RuntimeSettings, settings_from_env
from fxbot.forward import ForwardTestWorker
from fxbot.instruments import FxInstrument, PriceSnapshot
from fxbot.journal import StructuredJournal
from fxbot.models import Side, FxSignalIntent, BotRunState
from fxbot.mt5 import Mt5Client, Mt5Error
from fxbot.risk import FxRiskManager
from fxbot.sniper import SniperSettings, qualify_entry, qualify_execution, exit_reason
from test_fx_forward import FakeMt5Client, trending_frame, FIXED_NOW, FixedDatetime


def frame(side=Side.LONG):
    f = pd.DataFrame({"open": 1.0999, "high": 1.1002, "low": 1.0998,
                      "close": 1.1001, "atr": .001, "ma7": 1.1001,
                      "ma14": 1.0998, "ma28": 1.0995, "adx": 40.,
                      "di_plus": 30., "di_minus": 5.},
                     index=pd.date_range("2026-01-06", periods=65, freq="15min", tz="UTC"))
    f.loc[f.index[-1], ["open", "high", "low", "close"]] = [1.1001, 1.1005, 1.1000, 1.1004]
    if side is Side.SHORT:
        for c in ["open", "high", "low", "close", "ma7", "ma14", "ma28"]:
            f[c] = 2.2 - f[c]
        f["high"], f["low"] = f.low.copy(), f.high.copy()
        f["di_plus"], f["di_minus"] = f.di_minus.copy(), f.di_plus.copy()
    return f


@pytest.mark.parametrize("side", list(Side))
def test_trigger_and_direction_symmetry(side):
    f = frame(side)
    q = qualify_entry(f, side, float(f.iloc[-1].close), SniperSettings(mode="shadow"))
    assert q.allowed
    assert q.snapshot["trigger"] == "pullback"
    assert (q.snapshot["structure_stop"] - f.iloc[-1].close) * side.sign < 0


@pytest.mark.parametrize("change,reason", [
    ({"max_extension_atr": .2}, "POOR_LOCATION"),
    ({"max_candle_atr": .2}, "OVEREXTENDED"),
    ({"max_move_atr": .1}, "OVEREXTENDED"),
    ({"min_atr_ratio": 1.2}, "REGIME"),
    ({"min_close_strength": .99}, "WEAK_TRIGGER"),
])
def test_each_critical_gate_cannot_be_outscored(change, reason):
    q = qualify_entry(frame(), Side.LONG, 1.1004, SniperSettings(mode="enforce", **change))
    assert not q.allowed and q.reason == "SNIPER_REJECT_" + reason


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_invalid_data_fails_closed(value):
    f = frame()
    f.loc[f.index[-1], "atr"] = value
    assert qualify_entry(f, Side.LONG, 1.1004, SniperSettings()).reason == "SNIPER_REJECT_DATA"


def test_trigger_is_event_not_persistent_trend():
    f = frame()
    f.iloc[-1] = f.iloc[-2]
    assert qualify_entry(f, Side.LONG, 1.1001, SniperSettings()).reason == "SNIPER_REJECT_WEAK_TRIGGER"


def test_cost_known_unknown_and_no_double_spread():
    q = qualify_entry(frame(), Side.LONG, 1.1004, SniperSettings())
    unknown = qualify_execution(q.snapshot, reward=.001, spread=.0001, pip_size=.0001, settings=SniperSettings())
    assert unknown.reason == "SNIPER_REJECT_UNKNOWN_COST"
    known = SniperSettings(slippage_pips_per_side=.1, commission_pips_round_trip=.2)
    good = qualify_execution(q.snapshot, reward=.001, spread=.0001, pip_size=.0001, settings=known)
    assert good.allowed
    assert good.snapshot["net_target_after_cost"] == pytest.approx(.00096)
    assert good.snapshot["round_trip_cost"] == pytest.approx(.00014)
    assert not qualify_execution(q.snapshot, reward=.0001, spread=.0001, pip_size=.0001, settings=known).allowed


def test_nearby_obstacle_rejects_even_with_good_trigger():
    f = frame()
    f.loc[f.index[-10], "high"] = 1.1007
    q = qualify_entry(f, Side.LONG, 1.1004, SniperSettings())
    result = qualify_execution(q.snapshot, reward=.001, spread=.0001, pip_size=.0001,
                               settings=SniperSettings(slippage_pips_per_side=0, commission_pips_round_trip=0))
    assert "SNIPER_REJECT_LOW_ROOM_TO_TARGET" in result.snapshot["rejections"]


@pytest.mark.parametrize("side", list(Side))
def test_structure_stop_obeys_original_bounds_and_rr(side):
    risk = FxRiskManager(RiskSettings(), StrategySettings())
    entry = 1.1
    intent = FxSignalIntent("EUR_USD", side, FIXED_NOW, entry, {"atr": .001},
                            metadata={"sniper_structure_stop": entry - side.sign * .001})
    plan = risk.build_exit_plan(intent, FxInstrument("EUR_USD"))
    assert plan is not None and plan.risk_reward >= 1.5
    assert risk.build_exit_plan(replace(intent, metadata={"sniper_structure_stop": entry - side.sign * .01}), FxInstrument("EUR_USD")) is None
    assert risk.build_exit_plan(replace(intent, metadata={"sniper_structure_stop": entry + side.sign * .001}), FxInstrument("EUR_USD")) is None


@pytest.mark.parametrize("side", list(Side))
def test_failure_and_stagnation_exits(side):
    args = dict(side=side, close=1.1 - side.sign * .0003, di_plus=25, di_minus=10,
                entry=1.1, entry_atr=.001, trigger_level=1.1, bars_held=2)
    assert exit_reason(**args, settings=SniperSettings(failure_exit=True)) == "SNIPER_EXIT_THESIS_FAILED"
    assert exit_reason(**args, settings=SniperSettings()) is None
    args.update(close=1.1 + side.sign * .0001, bars_held=4)
    assert exit_reason(**args, settings=SniperSettings(time_exit=True)) == "SNIPER_EXIT_STAGNANT"
    args["close"] = 1.1 + side.sign * .0003
    assert exit_reason(**args, settings=SniperSettings(time_exit=True)) is None


def settings(tmp_path, mode):
    return FxBotSettings(instruments=["EUR_USD"], sniper=SniperSettings(mode=mode),
        strategy=StrategySettings(partial_tp_enabled=False, trade_sessions_utc=(), avoid_rollover_minutes=0,
                                  min_atr_pips=.1, max_atr_pips=30, adx_min=10, htf_adx_min=10),
        risk=RiskSettings(risk_per_trade_pct=.01, max_units_per_trade=1_000_000,
                          max_pair_exposure_pct=10, max_gross_exposure_pct=10, max_currency_exposure_pct=10),
        runtime=RuntimeSettings(database_url=f"sqlite:///{tmp_path / (mode + '.db')}"))


def test_off_and_shadow_have_identical_orders_enforce_suppresses(tmp_path):
    orders = {}
    for mode in ["off", "shadow", "enforce"]:
        config = settings(tmp_path, mode)
        with closing(StructuredJournal(config.runtime.database_url)) as journal:
            client = FakeMt5Client(entry_frame=trending_frame(1.08, .00025), htf_frame=trending_frame(1.06, .0005))
            worker = ForwardTestWorker(config, client=client, journal=journal)
            journal.set_state(BotRunState.RUNNING)
            with patch("fxbot.forward.datetime", FixedDatetime):
                worker.scan_once()
            orders[mode] = client.created_orders
            if mode == "shadow":
                assert journal.recent_signals()[0].payload["intent"]["metadata"]["sniper"]["rejections"]
            if mode == "enforce":
                assert journal.recent_signals()[0].reason.startswith("SNIPER_REJECT_")
    assert len(orders["off"]) == 1
    assert orders["off"] == orders["shadow"]
    assert orders["enforce"] == []


def test_break_even_cannot_be_undone_by_trailing(tmp_path):
    config = settings(tmp_path, "off")
    client = Mock()
    client.set_trade_dependent_orders.return_value = {"ok": True}
    client.candles.return_value = trending_frame(1.08, .00025)
    trade = {"id": "99", "price": 1.1, "currentUnits": 1000, "stopLossOrder": {"price": 1.099}}
    price = PriceSnapshot("EUR_USD", 1.1011, 1.1012, FIXED_NOW)
    with closing(StructuredJournal(config.runtime.database_url)) as journal:
        worker = ForwardTestWorker(config, client=client, journal=journal)
        journal.upsert_trade(broker_trade_id="99", instrument="EUR_USD", side="LONG", units=1000, state="open",
                             payload={"strategy_context": {"initial_stop": 1.099}})
        worker._maybe_move_stop_to_breakeven(trade, FxInstrument("EUR_USD"), price)
        assert trade["stopLossOrder"]["price"] > 1.1
        worker._maybe_update_trailing_stop(trade, FxInstrument("EUR_USD"), price)
        assert client.set_trade_dependent_orders.call_count == 1


@pytest.mark.parametrize("units,quote,order_type", [(1000, 1.1, 1), (-1000, 1.1002, 0)])
def test_close_uses_executable_quote_and_checks_complete_request(units, quote, order_type):
    module = Mock()
    module.ORDER_TYPE_SELL = 1
    module.ORDER_TYPE_BUY = 0
    module.TRADE_ACTION_DEAL = 1
    module.ORDER_TIME_GTC = 0
    client = Mt5Client(settings=FxBotSettings().broker, module=module)
    client._ensure_connected = Mock()
    client._price_snapshot = Mock(return_value=PriceSnapshot("EUR_USD", 1.1, 1.1002, FIXED_NOW))
    client._validated_order_filling = Mock(return_value=1)
    client._checked_result = Mock(return_value={"price": quote, "deal": 10})
    client.close_position(trade_id="99", instrument=FxInstrument("EUR_USD"), signed_units=units)
    checked = client._validated_order_filling.call_args.args[0]
    assert checked["price"] == quote and checked["type"] == order_type
    assert checked["position"] == 99 and checked["volume"] > 0


def test_environment_defaults_and_validation(monkeypatch):
    monkeypatch.setenv("FX_SNIPER_MODE", "shadow")
    monkeypatch.setenv("FX_SNIPER_TIME_EXIT", "true")
    assert settings_from_env().sniper == SniperSettings(mode="shadow", time_exit=True)
    for kw in ({"max_cost_ratio": float("nan")}, {"slippage_pips_per_side": -1},
               {"lookback": 2}, {"mode": "true"}):
        with pytest.raises(ValueError):
            SniperSettings(**kw)


@pytest.mark.parametrize("mode", ["off", "shadow", "enforce"])
def test_exit_context_survives_restart_and_cannot_duplicate_close(tmp_path, mode):
    config = replace(settings(tmp_path, mode), sniper=SniperSettings(mode=mode, failure_exit=True))
    client = Mock()
    client.candles.return_value = trending_frame(1.08, .00025)
    client.close_position.return_value = {"ok": True}
    trade = {"id": "99", "price": 1.11, "currentUnits": 1000,
             "openTime": FIXED_NOW - timedelta(hours=1), "stopLossOrder": {"price": 1.108}}
    price = PriceSnapshot("EUR_USD", 1.1099, 1.11, FIXED_NOW)
    with closing(StructuredJournal(config.runtime.database_url)) as journal:
        journal.upsert_trade(broker_trade_id="99", instrument="EUR_USD", side="LONG", units=1000, state="open",
                             payload={"strategy_context": {"sniper": {"mode": "enforce", "atr": .001,
                                                                        "trigger": "breakout", "trigger_level": 1.1102}}})
        worker = ForwardTestWorker(config, client=client, journal=journal)
        worker._manage_sniper_trade(FIXED_NOW, trade, FxInstrument("EUR_USD"), price)
        restarted = ForwardTestWorker(config, client=client, journal=journal)
        restarted._manage_sniper_trade(FIXED_NOW, trade, FxInstrument("EUR_USD"), price)
        assert client.close_position.call_count == (1 if mode == "enforce" else 0)
        assert journal.find_trade("99").payload["sniper_excursions"]["mae"] > 0


def test_costs_are_budgeted_and_partial_orders_respect_position_slots(tmp_path):
    from fxbot.models import FxPortfolioState
    risk_settings = RiskSettings(risk_per_trade_pct=.01, max_units_per_trade=1_000_000,
                                 max_pair_exposure_pct=50, max_gross_exposure_pct=50,
                                 max_currency_exposure_pct=50, min_free_margin_pct=0,
                                 max_open_positions=1)
    risk = FxRiskManager(risk_settings, StrategySettings())
    instrument = FxInstrument("EUR_USD", margin_rate=.001)
    portfolio = FxPortfolioState(10000, 10000, 0, 0)
    intent = FxSignalIntent("EUR_USD", Side.LONG, FIXED_NOW, 1.1, {"atr": .001})
    baseline = risk.evaluate_intent(intent, instrument, portfolio)
    with_cost = risk.evaluate_intent(replace(intent, metadata={"execution_cost_price": .0002}), instrument, portfolio)
    assert baseline.allowed and with_cost.allowed and with_cost.units < baseline.units
    assert with_cost.risk_amount <= 100 + 1e-9
    with closing(StructuredJournal(settings(tmp_path, "off").runtime.database_url)) as journal:
        worker = ForwardTestWorker(replace(settings(tmp_path, "off"), strategy=StrategySettings(partial_tp_enabled=True)),
                                   client=Mock(), journal=journal)
        assert len(worker._order_legs(intent, with_cost, instrument)) == 1


@pytest.mark.parametrize("ai_mode", ["off", "shadow", "advisory"])
def test_ai_cannot_resurrect_a_sniper_rejection(tmp_path, ai_mode):
    config = settings(tmp_path, "enforce")
    config = replace(config, ai=replace(config.ai, mode=ai_mode))
    with closing(StructuredJournal(config.runtime.database_url)) as journal:
        client = FakeMt5Client(entry_frame=trending_frame(1.08, .00025), htf_frame=trending_frame(1.06, .0005))
        worker = ForwardTestWorker(config, client=client, journal=journal)
        worker.deliberator = Mock()
        journal.set_state(BotRunState.RUNNING)
        with patch("fxbot.forward.datetime", FixedDatetime):
            worker.scan_once()
        assert not client.created_orders
        worker.deliberator.deliberate.assert_not_called()


def test_revalidation_blocks_stale_quote_and_new_news_event(tmp_path):
    from types import SimpleNamespace
    from fxbot.models import FxPortfolioState
    config = replace(settings(tmp_path, "enforce"), sniper=SniperSettings(mode="enforce", cost_filter=False))
    client = FakeMt5Client()
    instrument = FxInstrument("EUR_USD")
    snap = qualify_entry(frame(), Side.LONG, 1.10005, config.sniper).snapshot
    intent = FxSignalIntent("EUR_USD", Side.LONG, FIXED_NOW, client.price.ask, {"atr": .001}, metadata={"sniper": snap})
    risk = FxRiskManager(config.risk, config.strategy).evaluate_intent(intent, instrument, FxPortfolioState(10000, 10000, 0, 0))
    with closing(StructuredJournal(config.runtime.database_url)) as journal:
        worker = ForwardTestWorker(config, client=client, journal=journal)
        journal.set_state(BotRunState.RUNNING)
        client.price = replace(client.price, time=FIXED_NOW - timedelta(minutes=10))
        with patch("fxbot.forward.datetime", FixedDatetime):
            assert not worker._revalidate_sniper_execution(intent, instrument, risk)
        client.price = replace(client.price, time=FIXED_NOW)
        with patch("fxbot.forward.datetime", FixedDatetime), patch("fxbot.forward.can_trade", return_value=SimpleNamespace(allowed=False)) as gate:
            assert not worker._revalidate_sniper_execution(intent, instrument, risk)
            gate.assert_called_once()


def test_recovery_blocks_until_broker_confirms_fill(tmp_path):
    from fxbot.recovery import reconcile_order, RecoveryState
    config = settings(tmp_path, "off")
    with closing(StructuredJournal(config.runtime.database_url)) as journal:
        row, _ = journal.reserve_order(client_order_id="test-recovery", timestamp=FIXED_NOW,
                                        instrument="EUR_USD", side="LONG", units=1000, order_type="MARKET",
                                        risk_amount=1, payload={})
        assert journal.has_unresolved_orders()
        result = reconcile_order(row, lambda _: {"id": "99", "state": "filled", "trade_id": "99"})
        assert result.after == RecoveryState.FILLED
        journal.update_order("test-recovery", status="filled", broker_trade_id="99")
        assert not journal.has_unresolved_orders()


def test_closed_signal_cannot_reenter_on_the_same_trigger(tmp_path):
    from fxbot.models import FxPortfolioState
    config = settings(tmp_path, "off")
    client = FakeMt5Client()
    intent = FxSignalIntent("EUR_USD", Side.LONG, FIXED_NOW, 1.1, {"atr": .001})
    instrument = FxInstrument("EUR_USD")
    risk = FxRiskManager(config.risk, config.strategy).evaluate_intent(intent, instrument, FxPortfolioState(10000, 10000, 0, 0))
    with closing(StructuredJournal(config.runtime.database_url)) as journal:
        worker = ForwardTestWorker(config, client=client, journal=journal)
        assert worker._submit_idempotent(intent, instrument, risk)
        order = journal.recent_orders()[0]
        journal.mark_trade_orders_closed(order.broker_trade_id)
        assert not worker._submit_idempotent(intent, instrument, risk)
        assert len(client.created_orders) == 1
