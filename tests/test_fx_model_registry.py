"""Registry promotion and integrity regression tests."""
import hashlib
import json
from pathlib import Path

import pytest

from fxbot.ai.model_registry import ModelRegistry, RegistryError, exclusive_file_lock


def _candidate(directory: Path, version: str):
    directory.mkdir(parents=True, exist_ok=True)
    model = directory / "model.joblib"
    metadata = directory / "metadata.json"
    model.write_bytes(f"artifact:{version}".encode())
    metadata.write_text(json.dumps({
        "model_version": version, "model_name": "XGB",
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "feature_columns": ["a", "b"], "feature_builder_version": "v1",
        "target": "TP_BEFORE_SL",
    }), encoding="utf-8")
    return model, metadata


def test_register_verify_and_explicit_approval(tmp_path: Path):
    registry = ModelRegistry(tmp_path / "registry")
    model, metadata = _candidate(tmp_path / "run1", "entry-v1")
    row = registry.register(model_path=model, metadata_path=metadata, model_type="entry_quality")
    assert row["status"] == "candidate"
    assert registry.verify(row["model_id"])["model_sha256"] == row["model_sha256"]
    with pytest.raises(RegistryError, match="manual approval"):
        registry.activate(row["model_id"])
    with pytest.raises(RegistryError, match="required"):
        registry.approve(row["model_id"], approver="", evidence="")
    registry.approve(row["model_id"], approver="reviewer", evidence="demo review")
    active = registry.activate(row["model_id"])
    assert active["status"] == "active"
    assert registry.status()["active"]["entry_quality"] == row["model_id"]


def test_cannot_promote_tampered_artifact(tmp_path: Path):
    registry = ModelRegistry(tmp_path / "registry")
    model, meta = _candidate(tmp_path / "run1", "entry-v1")
    row = registry.register(model_path=model, metadata_path=meta, model_type="entry_quality")
    model.write_bytes(b"tampered")
    with pytest.raises(RegistryError, match="corrupted"):
        registry.approve(row["model_id"], approver="reviewer", evidence="report")
    assert registry.status()["active"] == {}


def test_rollback_requires_approved_model(tmp_path: Path):
    registry = ModelRegistry(tmp_path / "registry")
    a, am = _candidate(tmp_path / "run1", "entry-v1")
    b, bm = _candidate(tmp_path / "run2", "entry-v2")
    first = registry.register(model_path=a, metadata_path=am, model_type="entry_quality")
    second = registry.register(model_path=b, metadata_path=bm, model_type="entry_quality")
    registry.approve(first["model_id"], approver="ops", evidence="validated")
    registry.activate(first["model_id"])
    with pytest.raises(RegistryError, match="never approved"):
        registry.rollback("entry_quality", second["model_id"])
    registry.approve(second["model_id"], approver="ops", evidence="validated")
    registry.activate(second["model_id"])
    result = registry.rollback("entry_quality", first["model_id"])
    assert result["status"] == "active"
    assert registry.status()["active"]["entry_quality"] == first["model_id"]


def test_exclusive_file_lock_blocks_overlap(tmp_path: Path):
    lock = tmp_path / "retraining.lock"
    with exclusive_file_lock(lock):
        with pytest.raises(RegistryError, match="lock already exists"):
            with exclusive_file_lock(lock):
                pass
    assert not lock.exists()


def test_version_collision_and_missing_metadata(tmp_path: Path):
    registry = ModelRegistry(tmp_path / "registry")
    a, am = _candidate(tmp_path / "run1", "entry-v1")
    row = registry.register(model_path=a, metadata_path=am, model_type="entry_quality")
    assert row["model_id"]
    b, bm = _candidate(tmp_path / "run2", "entry-v1")
    b.write_bytes(b"unique")
    details = json.loads(bm.read_text())
    details["model_sha256"] = hashlib.sha256(b.read_bytes()).hexdigest()
    bm.write_text(json.dumps(details))
    with pytest.raises(RegistryError, match="already registered"):
        registry.register(model_path=b, metadata_path=bm, model_type="entry_quality")
