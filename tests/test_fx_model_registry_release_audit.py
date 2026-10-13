"""Regression tests for the registry audit before merging exit research."""

import hashlib
import json
from unittest.mock import patch

import pytest

import fxbot.ai.model_registry as registry_module
from fxbot.ai.model_registry import ModelRegistry, RegistryError
from test_fx_model_registry import _candidate


def test_metadata_swap_between_validation_and_registration_is_rejected(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    model, metadata = _candidate(tmp_path / "run", "first")
    original_validate = registry_module._validate_metadata

    def mutate_after_validation(data, model_type):
        original_validate(data, model_type)
        # Simulates a different process rewriting the file after validation.
        metadata.write_text(json.dumps({"target": "EXIT_ACTION"}), encoding="utf-8")

    with patch.object(registry_module, "_validate_metadata", side_effect=mutate_after_validation):
        with pytest.raises(RegistryError, match="changed during registration"):
            registry.register(model_path=model, metadata_path=metadata, model_type="entry_quality")

    assert not registry.manifest.exists()


def test_registry_rejects_orphaned_active_model(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    model, metadata = _candidate(tmp_path / "run", "first")
    row = registry.register(model_path=model, metadata_path=metadata, model_type="entry_quality")
    registry.approve(row["model_id"], approver="reviewer", evidence="verified")
    registry.activate(row["model_id"])
    payload = json.loads(registry.manifest.read_text(encoding="utf-8"))
    payload["active"] = {}
    registry.manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RegistryError, match="orphaned active"):
        registry.status()


def test_registry_verify_checks_metadata_contract_again(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    model, metadata = _candidate(tmp_path / "run", "first")
    row = registry.register(model_path=model, metadata_path=metadata, model_type="entry_quality")
    data = json.loads(metadata.read_text(encoding="utf-8"))
    data["target"] = "EXIT_ACTION"
    metadata.write_text(json.dumps(data), encoding="utf-8")
    payload = json.loads(registry.manifest.read_text(encoding="utf-8"))
    payload["models"][row["model_id"]]["metadata_sha256"] = hashlib.sha256(metadata.read_bytes()).hexdigest()
    registry.manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RegistryError, match="entry target mismatch"):
        registry.verify(row["model_id"])
