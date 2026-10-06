"""Numerical candidate-quality prediction service."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import math
from typing import Protocol

import pandas as pd

from fxbot.ai.feature_builder import FEATURE_BUILDER_VERSION, build_prediction_features
from fxbot.ai.model_loader import VersionedModelLoader
from fxbot.ai.schemas import NumericalPrediction, PredictionRequest
from fxbot.config import MlPredictionSettings

log = logging.getLogger(__name__)


class TradePredictor(Protocol):
    def predict(self, request: PredictionRequest) -> NumericalPrediction: ...


class PredictionService:
    """Local, versioned prediction boundary.

    It has no MT5/order dependency. Any load/inference failure is represented as
    structured data for the deliberation layer and never raises into execution.
    """

    def __init__(
        self,
        settings: MlPredictionSettings,
        *,
        loader: VersionedModelLoader | None = None,
        entry_loader: VersionedModelLoader | None = None,
        auxiliary_loaders: dict[str, VersionedModelLoader] | None = None,
    ) -> None:
        self.settings = settings
        self.loader = loader
        self.entry_loader = entry_loader
        self.auxiliary_loaders = dict(auxiliary_loaders or {})
        if self.loader is None and settings.mode != "off" and settings.model_path and settings.metadata_path:
            self.loader = VersionedModelLoader(
                model_path=settings.model_path,
                metadata_path=settings.metadata_path,
                expected_target=settings.target,
                verify_hash=settings.verify_hash,
            )
        if (
            self.entry_loader is None
            and settings.mode != "off"
            and settings.entry_model_path
            and settings.entry_metadata_path
        ):
            self.entry_loader = VersionedModelLoader(
                model_path=settings.entry_model_path,
                metadata_path=settings.entry_metadata_path,
                expected_target="ENTRY_ACTION_LABEL",
                verify_hash=settings.verify_hash,
            )
        if settings.mode != "off":
            configured_auxiliary = {
                "IMMEDIATE_ADVERSE_MOVEMENT": (
                    settings.immediate_adverse_model_path,
                    settings.immediate_adverse_metadata_path,
                ),
                "CONTINUATION": (
                    settings.continuation_model_path,
                    settings.continuation_metadata_path,
                ),
                "FAKE_BREAKOUT": (
                    settings.fake_breakout_model_path,
                    settings.fake_breakout_metadata_path,
                ),
            }
            for target, (model_path, metadata_path) in configured_auxiliary.items():
                if target in self.auxiliary_loaders or not model_path or not metadata_path:
                    continue
                self.auxiliary_loaders[target] = VersionedModelLoader(
                    model_path=model_path,
                    metadata_path=metadata_path,
                    expected_target=target,
                    verify_hash=settings.verify_hash,
                )

    def predict(self, request: PredictionRequest) -> NumericalPrediction:
        started = time.perf_counter()
        if self.settings.mode == "off":
            return NumericalPrediction(
                status="off",
                latency_ms=_elapsed_ms(started),
                error="prediction_service_disabled",
            )
        if self.loader is None:
            return NumericalPrediction(
                status="unavailable",
                latency_ms=_elapsed_ms(started),
                error="prediction_model_not_configured",
            )
        try:
            features = build_prediction_features(request)
            feature_hash = hashlib.sha256(
                json.dumps(features, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            artifact = self.loader.load()
            frame = pd.DataFrame([features])
            probability = _binary_probability(artifact, frame)
            outputs = _classification_outputs(artifact.target, probability)
            auxiliary_outputs = _auxiliary_classification_outputs(
                self.auxiliary_loaders,
                frame,
            )
            entry_outputs = _entry_timing_outputs(self.entry_loader, frame)
            return NumericalPrediction(
                status="ok",
                **outputs,
                **auxiliary_outputs,
                **entry_outputs,
                model_name=artifact.model_name,
                model_version=artifact.model_version,
                model_hash=artifact.model_hash,
                target=artifact.target,
                feature_version=FEATURE_BUILDER_VERSION,
                feature_hash=feature_hash,
                latency_ms=_elapsed_ms(started),
            )
        except Exception as exc:
            log.warning("ML prediction failed: %s", type(exc).__name__)
            return NumericalPrediction(
                status="error",
                latency_ms=_elapsed_ms(started),
                error=type(exc).__name__,
            )


def _elapsed_ms(started: float) -> int:
    return max(0, int(round((time.perf_counter() - started) * 1000)))


def _probabilities(artifact, frame, expected_classes: list[int]) -> list[float]:
    classes = list(getattr(artifact.model, "classes_", []))
    if classes != expected_classes:
        raise ValueError("model probability class order is incompatible")
    values = [float(v) for v in artifact.model.predict_proba(frame)[0]]
    if len(values) != len(expected_classes) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("invalid model probabilities")
    if not math.isclose(sum(values), 1, abs_tol=1e-6):
        raise ValueError("model probabilities do not sum to one")
    return values


def _binary_probability(artifact, frame) -> float:
    return _probabilities(artifact, frame, [0, 1])[1]


def _classification_outputs(target: str, probability: float) -> dict[str, float]:
    mapping = {
        "TP_BEFORE_SL": "tp_before_sl_probability",
        "PROFITABLE_WITHIN_5_MIN": "profitable_5m_probability",
        "PROFITABLE_WITHIN_15_MIN": "profitable_15m_probability",
        "IMMEDIATE_ADVERSE_MOVEMENT": "immediate_adverse_probability",
        "CONTINUATION": "continuation_probability",
        "FAKE_BREAKOUT": "fake_breakout_probability",
    }
    try:
        key = mapping[target]
    except KeyError as exc:
        raise ValueError(f"unsupported classification target {target!r}") from exc
    return {key: probability}


def _entry_timing_outputs(
    loader: VersionedModelLoader | None,
    frame: pd.DataFrame,
) -> dict[str, object]:
    if loader is None:
        return {}
    try:
        artifact = loader.load()
        labels = artifact.metadata.get("class_labels")
        expected = ["ENTER_NOW", "WAIT_30S", "WAIT_1M", "WAIT_3M", "SKIP"]
        if labels != expected:
            raise ValueError("entry timing artifact class labels do not match serving action space")
        probabilities = _probabilities(artifact, frame, list(range(5)))
        if len(probabilities) != len(labels):
            raise ValueError("entry timing probability count does not match class labels")
        distribution = {
            label: probability
            for label, probability in zip(labels, probabilities)
        }
        best_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        return {
            "entry_action": labels[best_index],
            "entry_action_confidence": probabilities[best_index],
            "entry_action_probabilities": distribution,
            "entry_model_name": artifact.model_name,
            "entry_model_version": artifact.model_version,
            "entry_model_hash": artifact.model_hash,
        }
    except Exception as exc:
        log.warning("entry timing prediction failed: %s", type(exc).__name__)
        return {
            "entry_action_error": type(exc).__name__,
        }


def _auxiliary_classification_outputs(
    loaders: dict[str, VersionedModelLoader],
    frame: pd.DataFrame,
) -> dict[str, object]:
    if not loaders:
        return {}
    outputs: dict[str, object] = {}
    model_metadata: dict[str, dict[str, str]] = {}
    errors: dict[str, str] = {}
    for target, loader in loaders.items():
        try:
            artifact = loader.load()
            if artifact.target != target:
                raise ValueError("auxiliary target mismatch")
            probability = _binary_probability(artifact, frame)
            outputs.update(_classification_outputs(target, probability))
            model_metadata[target] = {
                "model_name": artifact.model_name,
                "model_version": artifact.model_version,
                "model_hash": artifact.model_hash,
            }
        except Exception as exc:
            log.warning("auxiliary %s prediction failed: %s", target, type(exc).__name__)
            errors[target] = type(exc).__name__
    if model_metadata:
        outputs["auxiliary_models"] = model_metadata
    if errors:
        outputs["auxiliary_errors"] = errors
    return outputs
