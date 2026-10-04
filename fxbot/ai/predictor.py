"""Numerical candidate-quality prediction service."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Protocol

import pandas as pd

from fxbot.ai.feature_builder import FEATURE_BUILDER_VERSION, build_prediction_features
from fxbot.ai.model_loader import ModelLoadError, VersionedModelLoader
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
    ) -> None:
        self.settings = settings
        self.loader = loader
        if self.loader is None and settings.mode != "off" and settings.model_path and settings.metadata_path:
            self.loader = VersionedModelLoader(
                model_path=settings.model_path,
                metadata_path=settings.metadata_path,
                expected_target=settings.target,
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
            probability = float(artifact.model.predict_proba(frame)[0][1])
            return NumericalPrediction(
                status="ok",
                tp_before_sl_probability=probability,
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
                error=f"{type(exc).__name__}: {exc}",
            )


def _elapsed_ms(started: float) -> int:
    return max(0, int(round((time.perf_counter() - started) * 1000)))
