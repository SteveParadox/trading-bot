"""Pure exit policy invariants before any broker adapter is permitted."""
from datetime import datetime, timezone

import pytest

from fxbot.ai.exit_intelligence import EXIT_ACTIONS, ExitPrediction
from fxbot.ai.exit_policy import ExitPolicySettings, evaluate_exit_policy


def _sample(action="TRAIL_STOP", direction="BUY"):
    timestamp = datetime.now(timezone.utc).isoformat()
    prediction = ExitPrediction(
        prediction_id="p1", position_id="1", timestamp=timestamp,
        status="ok", decision=action, confidence=.8, reason_codes=("TEST",),
        model_version="m", feature_version="exit-v1", prompt_version=None,
        strategy_version="s",
        probabilities={key: (.8 if key == action else .2 / 4) for key in EXIT_ACTIONS},
    )
    snapshot = {
        "position_id": "1", "timestamp": timestamp, "direction": direction,
        "stop_loss": 1.1000 if direction == "BUY" else 1.1100,
        "bid": 1.1060, "ask": 1.1062,
        "atr_pips": 5.0, "estimated_net_pl": None,
        "broker_volume_lots": 0.12,
    }
    return snapshot, prediction


def _evaluate(snapshot, prediction, **kwargs):
    return evaluate_exit_policy(
        snapshot, prediction, ExitPolicySettings(),
        advisory_released=True, quote_age_seconds=1,
        pip_size=.0001, tick_size=.00001, stop_distance=.00015,
        freeze_distance=.0001, **kwargs,
    )


def test_unreleased_execution_always_denied():
    snapshot, prediction = _sample("TAKE_PROFIT_NOW")
    result = evaluate_exit_policy(snapshot, prediction, ExitPolicySettings(),
                                  advisory_released=False, quote_age_seconds=0)
    assert not result.eligible
    assert result.reason == "advisory_execution_not_released"


def test_take_profit_requires_verified_net_profit():
    snapshot, prediction = _sample("TAKE_PROFIT_NOW")
    assert not _evaluate(snapshot, prediction).eligible
    snapshot["estimated_net_pl"] = .50
    assert not _evaluate(snapshot, prediction).eligible
    snapshot["estimated_net_pl"] = 1.35
    assert _evaluate(snapshot, prediction).eligible


def test_hold_does_not_change_position_and_defensive_exit_not_auto_approved():
    snapshot, prediction = _sample("HOLD")
    result = _evaluate(snapshot, prediction)
    assert result.eligible and result.suggested_stop is None and result.reduction_volume is None
    from dataclasses import replace
    defensive = replace(prediction, decision="EXIT")
    assert _evaluate(snapshot, defensive).reason == "defensive_exit_not_validated"


def test_trailing_only_tightens_for_buy_or_sell():
    buy, p = _sample()
    result = _evaluate(buy, p)
    assert result.eligible
    assert result.suggested_stop is not None
    assert buy["stop_loss"] < result.suggested_stop < buy["bid"]
    buy["stop_loss"] = buy["bid"] - .0001
    assert not _evaluate(buy, p).eligible
    sell, p_sell = _sample(direction="SELL")
    outcome = _evaluate(sell, p_sell)
    assert outcome.eligible
    assert sell["ask"] < outcome.suggested_stop < sell["stop_loss"]
    sell["stop_loss"] = sell["ask"] + .0001
    assert not _evaluate(sell, p_sell).eligible


def test_partial_close_uses_broker_lot_step():
    snap, pred = _sample("REDUCE_POSITION")
    outcome = _evaluate(snap, pred)
    assert outcome.eligible
    assert outcome.reduction_volume == pytest.approx(.06)
    snap["broker_volume_lots"] = .01
    assert not _evaluate(snap, pred).eligible
    del snap["broker_volume_lots"]
    assert not _evaluate(snap, pred).eligible


def test_untrusted_inputs_fail_identity_and_freshness():
    snap, pred = _sample("HOLD")
    snap["position_id"] = "unrelated"
    assert _evaluate(snap, pred).reason == "position_identity_mismatch"
    snap["position_id"] = "1"
    assert not evaluate_exit_policy(snap, pred, ExitPolicySettings(),
                                    advisory_released=True, quote_age_seconds=500).eligible
