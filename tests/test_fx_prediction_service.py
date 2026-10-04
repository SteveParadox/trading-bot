from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import pandas as pd
import pytest

from fxbot.ai.feature_builder import FEATURE_BUILDER_VERSION, build_prediction_features
from fxbot.ai.model_loader import ModelLoadError, VersionedModelLoader
from fxbot.ai.predictor import PredictionService
from fxbot.ai.schemas import PredictionRequest
from fxbot.config import MlPredictionSettings
from fxbot.training_dataset import FEATURE_COLUMNS


class _ProbModel:
    def __init__(self, probability: float = 0.83) -> None:
        self.probability = probability

    def predict_proba(self, frame):
        assert list(frame.columns) == FEATURE_COLUMNS
        return [[1.0 - self.probability, self.probability]]


class _FakeLoader:
    def __init__(self, probability: float = 0.83) -> None:
        self.calls = 0
        self.artifact = SimpleNamespace(
            model=_ProbModel(probability),
            metadata={},
            model_name="XGBoost",
            model_version="xgb_tp_before_sl_v1",
            model_hash="abc123",
            target="TP_BEFORE_SL",
        )

    def load(self):
        self.calls += 1
        return self.artifact


class _EntryProbModel:
    def predict_proba(self, frame):
        assert list(frame.columns) == FEATURE_COLUMNS
        return [[0.10, 0.15, 0.50, 0.15, 0.10]]


class _EntryLoader:
    def __init__(self) -> None:
        self.calls = 0
        self.artifact = SimpleNamespace(
            model=_EntryProbModel(),
            metadata={
                "class_labels": ["ENTER_NOW", "WAIT_30S", "WAIT_1M", "WAIT_3M", "SKIP"],
            },
            model_name="XGBoost",
            model_version="xgb_entry_timing_v1",
            model_hash="entry123",
            target="ENTRY_ACTION_LABEL",
        )

    def load(self):
        self.calls += 1
        return self.artifact


class _TargetLoader:
    def __init__(self, target: str, probability: float, version: str) -> None:
        self.artifact = SimpleNamespace(
            model=_ProbModel(probability),
            metadata={},
            model_name="XGBoost",
            model_version=version,
            model_hash=f"hash-{target.lower()}",
            target=target,
        )

    def load(self):
        return self.artifact


def _request() -> PredictionRequest:
    snapshot = {
        "version": "v1",
        "candidate_id": "fxsig-1",
        "symbol": "EUR_USD",
        "timestamp": "2026-10-04T14:30:00+00:00",
        "direction": "LONG",
        "bid": 1.1000,
        "ask": 1.1002,
        "spread": 0.0002,
        "spread_pips": 2.0,
        "atr": 0.0010,
        "rsi": 58.0,
        "momentum": 0.7,
        "trend_strength": 27.0,
        "support_distance_pips": 8.0,
        "resistance_distance_pips": 16.0,
        "session": ["london", "overlap"],
        "volatility": 1.1,
        "proposed_entry": 1.1002,
        "stop_loss": 1.0982,
        "take_profit": 1.1032,
        "risk_reward": 1.5,
        "current_exposure": {
            "account_currency": "USD",
            "open_positions": 1,
            "portfolio_risk": 20.0,
            "gross_exposure": 1200.0,
            "pair_exposure": 900.0,
            "free_margin": 9800.0,
        },
        "news_context": {
            "risk_level": "LOW",
            "upcoming_event": {
                "currency": "USD",
                "impact_level": "LOW",
                "minutes_until_event": 48.0,
            },
            "recent_event": {
                "currency": "EUR",
                "minutes_since_event": 120.0,
            },
            "event_just_occurred": False,
            "freshness": {
                "state": "FRESH",
                "stale": False,
                "age_seconds": 20.0,
            },
        },
    }
    return PredictionRequest(
        candidate_id="fxsig-1",
        candidate_trade={
            "symbol": "EUR_USD",
            "direction": "LONG",
            "entry": 1.1002,
            "executed": True,  # Audit-only junk must not become a model feature.
            "rejection_reason": "future_information",
        },
        market_snapshot=snapshot,
        strategy_signal="signal_confirmed",
        strategy_score=74.0,
        execution_cost_pips_round_trip=0.4,
        pip_size=0.0001,
    )


def test_prediction_package_has_no_mt5_or_order_execution_dependency() -> None:
    import fxbot.ai.feature_builder as feature_builder
    import fxbot.ai.model_loader as model_loader
    import fxbot.ai.predictor as predictor

    source = "\n".join(
        inspect.getsource(module)
        for module in (feature_builder, model_loader, predictor)
    )
    assert "fxbot.mt5" not in source
    assert "create_market_order" not in source
    assert "close_position" not in source
    assert "order_send" not in source


def test_feature_builder_matches_training_feature_manifest_exactly() -> None:
    features = build_prediction_features(_request())
    assert list(features) == FEATURE_COLUMNS
    assert set(features) == set(FEATURE_COLUMNS)
    assert "executed" not in features
    assert "rejection_reason" not in features
    assert features["symbol"] == "EUR_USD"
    assert features["direction"] == "LONG"
    assert features["session"] == "london+overlap"
    assert features["hour_utc"] == 14
    assert features["day_of_week"] == 6
    assert features["atr_pips"] == pytest.approx(10.0)
    assert features["spread_relative_to_atr"] == pytest.approx(0.2)
    assert features["execution_cost_pips_round_trip"] == pytest.approx(0.4)


def test_prediction_service_returns_versioned_probability_without_execution_dependency() -> None:
    loader = _FakeLoader(0.83)
    service = PredictionService(
        MlPredictionSettings(
            mode="shadow",
            model_path="ignored.joblib",
            metadata_path="ignored.json",
        ),
        loader=loader,
    )
    result = service.predict(_request())
    assert result.successful is True
    assert result.tp_before_sl_probability == pytest.approx(0.83)
    assert result.model_name == "XGBoost"
    assert result.model_version == "xgb_tp_before_sl_v1"
    assert result.model_hash == "abc123"
    assert result.feature_version == FEATURE_BUILDER_VERSION
    assert result.feature_hash
    assert loader.calls == 1


def test_prediction_service_returns_entry_timing_distribution_as_shadow_evidence() -> None:
    loader = _FakeLoader(0.81)
    entry_loader = _EntryLoader()
    service = PredictionService(
        MlPredictionSettings(
            mode="shadow",
            model_path="ignored.joblib",
            metadata_path="ignored.json",
            entry_model_path="ignored-entry.joblib",
            entry_metadata_path="ignored-entry.json",
        ),
        loader=loader,
        entry_loader=entry_loader,
    )
    result = service.predict(_request())

    assert result.successful is True
    assert result.tp_before_sl_probability == pytest.approx(0.81)
    assert result.entry_action == "WAIT_1M"
    assert result.entry_action_confidence == pytest.approx(0.50)
    assert result.entry_action_probabilities == {
        "ENTER_NOW": pytest.approx(0.10),
        "WAIT_30S": pytest.approx(0.15),
        "WAIT_1M": pytest.approx(0.50),
        "WAIT_3M": pytest.approx(0.15),
        "SKIP": pytest.approx(0.10),
    }
    assert result.entry_model_version == "xgb_entry_timing_v1"
    assert result.entry_model_hash == "entry123"
    assert entry_loader.calls == 1


def test_entry_timing_failure_does_not_destroy_primary_shadow_prediction() -> None:
    class BadEntryLoader:
        def load(self):
            raise ModelLoadError("broken entry model")

    result = PredictionService(
        MlPredictionSettings(mode="shadow"),
        loader=_FakeLoader(0.79),
        entry_loader=BadEntryLoader(),
    ).predict(_request())

    assert result.successful is True
    assert result.tp_before_sl_probability == pytest.approx(0.79)
    assert result.entry_action is None
    assert "ModelLoadError" in result.entry_action_error


def test_prediction_service_combines_auxiliary_entry_quality_probabilities() -> None:
    service = PredictionService(
        MlPredictionSettings(mode="shadow"),
        loader=_FakeLoader(0.82),
        auxiliary_loaders={
            "IMMEDIATE_ADVERSE_MOVEMENT": _TargetLoader(
                "IMMEDIATE_ADVERSE_MOVEMENT", 0.21, "xgb_immediate_adverse_movement_v1"
            ),
            "CONTINUATION": _TargetLoader(
                "CONTINUATION", 0.74, "xgb_continuation_v1"
            ),
            "FAKE_BREAKOUT": _TargetLoader(
                "FAKE_BREAKOUT", 0.18, "xgb_fake_breakout_v1"
            ),
        },
    )
    result = service.predict(_request())

    assert result.immediate_adverse_probability == pytest.approx(0.21)
    assert result.continuation_probability == pytest.approx(0.74)
    assert result.fake_breakout_probability == pytest.approx(0.18)
    assert set(result.auxiliary_models) == {
        "IMMEDIATE_ADVERSE_MOVEMENT",
        "CONTINUATION",
        "FAKE_BREAKOUT",
    }
    assert result.auxiliary_errors is None


def test_auxiliary_classifier_failure_isolated_from_primary_prediction() -> None:
    class BrokenLoader:
        def load(self):
            raise ModelLoadError("bad auxiliary")

    result = PredictionService(
        MlPredictionSettings(mode="shadow"),
        loader=_FakeLoader(0.80),
        auxiliary_loaders={
            "CONTINUATION": BrokenLoader(),
        },
    ).predict(_request())

    assert result.successful is True
    assert result.tp_before_sl_probability == pytest.approx(0.80)
    assert result.continuation_probability is None
    assert "CONTINUATION" in result.auxiliary_errors


def test_prediction_service_off_mode_does_not_load_model() -> None:
    loader = _FakeLoader()
    result = PredictionService(
        MlPredictionSettings(mode="off"),
        loader=loader,
    ).predict(_request())
    assert result.status == "off"
    assert result.successful is False
    assert loader.calls == 0


def test_required_prediction_configuration_needs_artifact_paths() -> None:
    with pytest.raises(ValueError, match="model_path"):
        MlPredictionSettings(mode="required")


def test_model_loader_verifies_hash_target_and_feature_manifest(tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "model.metadata.json"
    joblib.dump(_ProbModel(0.77), model_path)
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps({
        "model_name": "XGBoost",
        "model_version": "xgb_tp_before_sl_v1",
        "target": "TP_BEFORE_SL",
        "problem_type": "binary_classification",
        "feature_columns": FEATURE_COLUMNS,
        "model_sha256": digest,
    }))

    loaded = VersionedModelLoader(
        model_path=model_path,
        metadata_path=metadata_path,
    ).load()
    assert loaded.model_name == "XGBoost"
    assert loaded.model_hash == digest
    assert loaded.model.predict_proba(pd.DataFrame([build_prediction_features(_request())]))[0][1] == pytest.approx(0.77)

    model_path.write_bytes(model_path.read_bytes() + b"tampered")
    with pytest.raises(ModelLoadError, match="SHA-256"):
        VersionedModelLoader(
            model_path=model_path,
            metadata_path=metadata_path,
        ).load()


def test_model_loader_rejects_training_serving_feature_mismatch(tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "model.metadata.json"
    joblib.dump(_ProbModel(), model_path)
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps({
        "model_name": "XGBoost",
        "model_version": "bad",
        "target": "TP_BEFORE_SL",
        "feature_columns": FEATURE_COLUMNS[:-1],
        "model_sha256": digest,
    }))
    with pytest.raises(ModelLoadError, match="feature manifest"):
        VersionedModelLoader(
            model_path=model_path,
            metadata_path=metadata_path,
        ).load()


def test_model_loader_rejects_feature_builder_version_mismatch(tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "model.metadata.json"
    joblib.dump(_ProbModel(), model_path)
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps({
        "model_name": "XGBoost",
        "model_version": "bad-feature-version",
        "target": "TP_BEFORE_SL",
        "feature_columns": FEATURE_COLUMNS,
        "feature_builder_version": "future-incompatible-version",
        "model_sha256": digest,
    }))
    with pytest.raises(ModelLoadError, match="feature-builder version"):
        VersionedModelLoader(
            model_path=model_path,
            metadata_path=metadata_path,
        ).load()
