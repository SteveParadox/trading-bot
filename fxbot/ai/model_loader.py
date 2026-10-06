"""Load and verify versioned local model artifacts for candidate prediction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import io
from pathlib import Path
from typing import Any

from fxbot.ai.feature_builder import FEATURE_BUILDER_VERSION
from fxbot.training_dataset import FEATURE_COLUMNS


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    metadata: dict[str, Any]
    model_name: str
    model_version: str
    model_hash: str
    target: str


class ModelLoadError(RuntimeError):
    pass


class VersionedModelLoader:
    """Lazy model loader with feature-manifest and content-hash verification."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        metadata_path: str | Path,
        expected_target: str = "TP_BEFORE_SL",
        verify_hash: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.expected_target = expected_target
        self.verify_hash = verify_hash
        self._loaded: LoadedModel | None = None

    def load(self) -> LoadedModel:
        if self._loaded is not None:
            return self._loaded
        if not self.model_path.is_file():
            raise ModelLoadError("model artifact not found")
        if not self.metadata_path.is_file():
            raise ModelLoadError("model metadata not found")

        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelLoadError("model metadata is unreadable or invalid JSON") from exc
        if not isinstance(metadata, dict):
            raise ModelLoadError("model metadata must be an object")

        target = str(metadata.get("target") or "")
        if target != self.expected_target:
            raise ModelLoadError(
                f"model target mismatch: expected {self.expected_target}, got {target or 'missing'}"
            )
        feature_columns = metadata.get("feature_columns")
        if feature_columns != FEATURE_COLUMNS:
            raise ModelLoadError("model feature manifest does not match serving feature schema")
        artifact_feature_version = metadata.get("feature_builder_version")
        if artifact_feature_version != FEATURE_BUILDER_VERSION:
            raise ModelLoadError(
                "model feature-builder version does not match serving feature builder"
            )

        artifact_bytes = self.model_path.read_bytes()
        actual_hash = hashlib.sha256(artifact_bytes).hexdigest()
        expected_hash = str(metadata.get("model_sha256") or "")
        if self.verify_hash and (not expected_hash or actual_hash != expected_hash):
            raise ModelLoadError("model SHA-256 does not match metadata")

        try:
            import joblib
            # Deserialize exactly the verified bytes, avoiding a file-swap race.
            model = joblib.load(io.BytesIO(artifact_bytes))
        except Exception as exc:
            raise ModelLoadError(f"model artifact could not be loaded: {type(exc).__name__}") from exc
        if not callable(getattr(model, "predict_proba", None)):
            raise ModelLoadError("loaded baseline model does not expose predict_proba")

        self._loaded = LoadedModel(
            model=model,
            metadata=metadata,
            model_name=str(metadata.get("model_name") or "unknown"),
            model_version=str(metadata.get("model_version") or "unknown"),
            model_hash=actual_hash,
            target=target,
        )
        return self._loaded
