"""Exit research safety regression tests: no broker side effects."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fxbot.ai.exit_intelligence import (
    EXIT_ACTIONS, EXIT_FEATURE_COLUMNS, EXIT_FEATURE_VERSION,
    ExitPrediction, ExitPredictionService, build_exit_snapshot, exit_features,
)
from fxbot.config import ExitAiSettings


class FakeExitEstimator:
    classes_ = [0, 1, 2, 3, 4]

    def predict_proba(self, frame):
        assert list(frame.columns) == list(EXIT_FEATURE_COLUMNS)
        return [[0.05, 0.75, 0.10, 0.05, 0.05]]


def _inputs(signed_units: float = 1000):
    now = datetime(2026, 10, 9, 10, 0, tzinfo=timezone.utc)
    trade = {
        "id": "400123",
        "instrument": "EUR_USD",
        "currentUnits": signed_units,
        "price": 1.1050,
        "openTime": now - timedelta(minutes=3),
        "stopLossOrder": {"price": 1.1020 if signed_units > 0 else 1.1080},
        "takeProfitOrder": {"price": 1.1100 if signed_units > 0 else 1.1000},
        "unrealizedPL": 1.2,
    }
    quote = SimpleNamespace(bid=1.1060, ask=1.1062, time=now - timedelta(seconds=1))
    instrument = SimpleNamespace(name="EUR_USD", pip_size=.0001)
    return now, trade, quote, instrument


def test_long_exit_uses_bid_and_only_causal_fields():
    now, trade, quote, instrument = _inputs()
    snapshot = build_exit_snapshot(trade, quote, instrument, now,
                                   recorded_payload={"sniper_excursions": {"mfe": .0008, "mae": .0005}},
                                   strategy_version="strategy-123")
    assert snapshot["liquidation_price"] == quote.bid
    assert snapshot["pnl_pips"] == pytest.approx(10.0)
    assert snapshot["mfe_pips"] == pytest.approx(10.0)
    assert snapshot["drawdown_from_peak_pips"] == 0
    assert snapshot["estimated_net_pl"] is None
    assert snapshot["sampled_excursions_only"] is True
    assert snapshot["strategy_version"] == "strategy-123"
    assert tuple(exit_features(snapshot)) == EXIT_FEATURE_COLUMNS


def test_short_exit_uses_ask():
    now, trade, quote, instrument = _inputs(-1000)
    snapshot = build_exit_snapshot(trade, quote, instrument, now)
    assert snapshot["liquidation_price"] == quote.ask
    assert snapshot["pnl_pips"] == pytest.approx(-12.0)


def test_reject_stale_or_future_quotes_and_positions():
    now, trade, quote, instrument = _inputs()
    quote.time = now - timedelta(seconds=121)
    with pytest.raises(ValueError, match="fresh causal"):
        build_exit_snapshot(trade, quote, instrument, now)
    quote.time = now + timedelta(seconds=1)
    with pytest.raises(ValueError, match="fresh causal"):
        build_exit_snapshot(trade, quote, instrument, now)
    quote.time = now
    trade["openTime"] = now + timedelta(seconds=1)
    with pytest.raises(ValueError, match="future"):
        build_exit_snapshot(trade, quote, instrument, now)


def test_no_model_does_not_fabricate_ml_decision():
    now, trade, quote, instrument = _inputs()
    snap = build_exit_snapshot(trade, quote, instrument, now)
    result = ExitPredictionService().predict(snap)
    assert result.status == "unavailable"
    assert result.decision is None
    assert result.prompt_version is None
    assert result.feature_version == EXIT_FEATURE_VERSION


def test_corrupted_hash_fails_closed(tmp_path: Path):
    import joblib
    now, trade, quote, instrument = _inputs()
    snap = build_exit_snapshot(trade, quote, instrument, now)
    model_path = tmp_path / "exit.joblib"
    meta_path = tmp_path / "exit.json"
    joblib.dump(FakeExitEstimator(), model_path)
    meta_path.write_text(json.dumps({
        "target": "EXIT_ACTION", "feature_columns": list(EXIT_FEATURE_COLUMNS),
        "feature_builder_version": EXIT_FEATURE_VERSION,
        "class_labels": list(EXIT_ACTIONS), "model_version": "candidate-a",
        "model_sha256": "0" * 64,
    }), encoding="utf-8")
    result = ExitPredictionService(model_path=str(model_path), metadata_path=str(meta_path)).predict(snap)
    assert result.status == "error"
    assert result.decision is None
    assert result.error == "ValueError"


def test_valid_model_is_prediction_only(tmp_path: Path):
    import joblib
    now, trade, quote, instrument = _inputs()
    snap = build_exit_snapshot(trade, quote, instrument, now)
    model_path = tmp_path / "exit.joblib"
    meta_path = tmp_path / "exit.json"
    joblib.dump(FakeExitEstimator(), model_path)
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    meta_path.write_text(json.dumps({
        "target": "EXIT_ACTION", "feature_columns": list(EXIT_FEATURE_COLUMNS),
        "feature_builder_version": EXIT_FEATURE_VERSION,
        "class_labels": list(EXIT_ACTIONS), "model_version": "candidate-a",
        "model_sha256": digest,
    }), encoding="utf-8")
    result = ExitPredictionService(model_path=str(model_path), metadata_path=str(meta_path)).predict(snap)
    assert result.status == "ok"
    assert result.decision == "TAKE_PROFIT_NOW"
    assert result.confidence == pytest.approx(.75)
    assert result.model_sha256 == digest
    assert result.probabilities is not None


def test_invalid_prediction_and_unreleased_execution_mode_rejected():
    now = datetime.now(timezone.utc).isoformat()
    with pytest.raises(ValueError, match="sum to one"):
        ExitPrediction(
            prediction_id="x", position_id="1", timestamp=now,
            status="ok", decision="HOLD", confidence=.8,
            probabilities={name: .1 for name in EXIT_ACTIONS},
            reason_codes=("TEST",), model_version="m", feature_version="v",
            prompt_version=None, strategy_version=None,
        )
    assert ExitAiSettings().mode == "shadow"
    assert ExitAiSettings(mode="off").mode == "off"
    with pytest.raises(ValueError, match="not approved"):
        ExitAiSettings(mode="advisory")
    with pytest.raises(ValueError, match="integrity"):
        ExitAiSettings(model_path="x", metadata_path="y", verify_hash=False)


def test_exit_model_paths_are_private_in_config_api():
    from fxbot.api import _config_payload
    from fxbot.config import FxBotSettings
    settings = FxBotSettings(exit_ai=ExitAiSettings(
        model_path="/private/exit.joblib",
        metadata_path="/private/exit.metadata.json",
        registry_path="/private/model-registry",
    ))
    result = _config_payload(settings)
    assert result["exit_ai"]["model_configured"] is True
    assert result["exit_ai"]["execution_enabled"] is False
    serialised = json.dumps(result)
    assert "/private/" not in serialised
    assert "exit.metadata.json" not in serialised
