"""Local candidate/champion registry with immutable model identity and explicit approval.

Registry activation is metadata-only until the serving process is configured to
consume that approved artifact. It NEVER switches running MT5 execution or
promotes candidates as a side effect of training.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


class RegistryError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def exclusive_file_lock(path: Path):
    """Fail fast rather than overlap unsafe registry/training modifications.

    An interrupted process can leave a stale lock. The operator must inspect
    and remove it after confirming no previous job is active.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RegistryError(f"lock already exists: {path.name}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        path.unlink(missing_ok=True)


class ModelRegistry:
    """Persist a small, auditable, atomic JSON manifest; never deserialize models."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest = self.root / "registry.json"
        self.lock = self.root / "registry.lock"

    def _read(self) -> dict[str, Any]:
        if not self.manifest.exists():
            return {"schema_version": 1, "models": {}, "active": {}, "history": []}
        value = json.loads(self.manifest.read_text(encoding="utf-8"))
        if (not isinstance(value, dict) or value.get("schema_version") != 1
                or not isinstance(value.get("models"), dict)
                or not isinstance(value.get("active"), dict)
                or not isinstance(value.get("history"), list)):
            raise RegistryError("invalid model registry schema")
        return value

    def _write(self, data: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        descriptor, temp_path = tempfile.mkstemp(prefix=".registry-", suffix=".json", dir=self.root)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(data, handle, sort_keys=True, indent=2, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.manifest)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def register(
        self,
        *, model_path: str | Path, metadata_path: str | Path, model_type: str,
    ) -> dict[str, Any]:
        if model_type not in {"entry_quality", "entry_timing", "exit_management"}:
            raise RegistryError("unsupported model type")
        model, meta = Path(model_path).resolve(), Path(metadata_path).resolve()
        if not model.is_file() or not meta.is_file():
            raise RegistryError("model and metadata files are required")
        data = json.loads(meta.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise RegistryError("model metadata must be an object")
        digest = _digest(model)
        if not data.get("model_sha256") or data["model_sha256"] != digest:
            raise RegistryError("candidate model SHA-256 mismatch")
        if not data.get("feature_columns") or not data.get("feature_builder_version"):
            raise RegistryError("candidate feature manifest is missing")
        version = str(data.get("model_version") or "").strip()
        if not version:
            raise RegistryError("model version is required")
        model_id = f"{model_type}:{version}:{digest[:16]}"
        record = {
            "model_id": model_id,
            "model_type": model_type,
            "model_name": str(data.get("model_name") or "unknown"),
            "model_version": version,
            "target": data.get("target"),
            "model_path": str(model),
            "metadata_path": str(meta),
            "model_sha256": digest,
            "metadata_sha256": _digest(meta),
            "feature_version": data["feature_builder_version"],
            "feature_manifest_sha256": hashlib.sha256(
                json.dumps(data["feature_columns"], sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "dataset_sha256": data.get("dataset_sha256"),
            "label_versions": data.get("label_versions"),
            "registered_at": _utc_now(),
            "status": "candidate",
        }
        with exclusive_file_lock(self.lock):
            store = self._read()
            existing = store["models"].get(model_id)
            if existing is not None:
                if {k: v for k, v in existing.items() if k != "status"} != {
                    k: v for k, v in record.items() if k != "status"
                }:
                    raise RegistryError("immutable model identity is already registered")
                return existing
            if any(row["model_type"] == model_type and row["model_version"] == version
                   for row in store["models"].values()):
                raise RegistryError("model version is already registered with a different artifact")
            store["models"][model_id] = record
            store["history"].append({"at": _utc_now(), "event": "registered", "model_id": model_id})
            self._write(store)
        return record

    def verify(self, model_id: str) -> dict[str, Any]:
        store = self._read()
        record = store["models"].get(model_id)
        if record is None:
            raise RegistryError("unknown model id")
        model, meta = Path(record["model_path"]), Path(record["metadata_path"])
        if (not model.is_file() or not meta.is_file()
                or _digest(model) != record["model_sha256"]
                or _digest(meta) != record["metadata_sha256"]):
            raise RegistryError("registered artifact missing or corrupted")
        data = json.loads(meta.read_text(encoding="utf-8"))
        manifest = hashlib.sha256(
            json.dumps(data["feature_columns"], sort_keys=True).encode("utf-8")
        ).hexdigest()
        if manifest != record["feature_manifest_sha256"]:
            raise RegistryError("registered feature manifest was modified")
        return record

    def approve(self, model_id: str, *, approver: str, evidence: str) -> dict[str, Any]:
        if not approver.strip() or not evidence.strip():
            raise RegistryError("approval identity and review evidence are required")
        with exclusive_file_lock(self.lock):
            self.verify(model_id)
            store = self._read()
            record = store["models"][model_id]
            if record["status"] not in {"candidate", "approved"}:
                raise RegistryError("model cannot be approved from current state")
            record["status"] = "approved"
            store["history"].append({"at": _utc_now(), "event": "approved",
                                     "model_id": model_id, "approver": approver, "evidence": evidence})
            self._write(store)
            return record

    def activate(self, model_id: str) -> dict[str, Any]:
        """Atomically switch registry pointer, not the engine's already loaded model."""
        with exclusive_file_lock(self.lock):
            record = self.verify(model_id)
            store = self._read()
            if store["models"][model_id]["status"] not in {"approved", "active"}:
                raise RegistryError("model has not passed manual approval")
            family = record["model_type"]
            prior = store["active"].get(family)
            if prior and prior != model_id:
                store["models"][prior]["status"] = "retired"
            store["active"][family] = model_id
            store["models"][model_id]["status"] = "active"
            store["history"].append({"at": _utc_now(), "event": "activated", "model_id": model_id, "previous": prior})
            self._write(store)
            return store["models"][model_id]

    def rollback(self, model_type: str, previous_model_id: str) -> dict[str, Any]:
        """Rollback only to previously approved known-good artifact."""
        with exclusive_file_lock(self.lock):
            record = self.verify(previous_model_id)
            store = self._read()
            if record["model_type"] != model_type:
                raise RegistryError("rollback model family mismatch")
            approved_ids = {event["model_id"] for event in store["history"] if event["event"] == "approved"}
            if previous_model_id not in approved_ids:
                raise RegistryError("rollback target was never approved")
            current = store["active"].get(model_type)
            if current and current != previous_model_id:
                store["models"][current]["status"] = "retired"
            store["models"][previous_model_id]["status"] = "active"
            store["active"][model_type] = previous_model_id
            store["history"].append({"at": _utc_now(), "event": "rollback",
                                     "model_id": previous_model_id, "previous": current})
            self._write(store)
            return store["models"][previous_model_id]

    def status(self) -> dict[str, Any]:
        store = self._read()
        return {
            "schema_version": store["schema_version"],
            "active": dict(store["active"]),
            "models": [{key: value for key, value in row.items()
                        if key not in {"model_path", "metadata_path"}}
                       for row in store["models"].values()],
            "history": list(store["history"]),
        }
