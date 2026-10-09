"""Challenger jobs must leave the champion and sample state intact on failure."""
from datetime import timedelta
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from fxbot.ai.model_registry import ModelRegistry, RegistryError, exclusive_file_lock
from fxbot.retrain import train_challenger, _atomic_json
from test_fx_ml_baseline import _row


def _dataset(path, *, incomplete=False):
    rows = []
    for year, count in [(2024, 12), (2025, 6), (2026, 6)]:
        for index in range(count):
            at = pd.Timestamp(f"{year}-02-01T12:00:00Z") + pd.Timedelta(hours=index)
            row = _row(at.isoformat(), f"{year}-{index}", index % 2, value=index+1.)
            row["label_end_timestamp"] = (at+pd.Timedelta(minutes=30)).isoformat()
            rows.append(row)
    if incomplete:
        rows[0]["TP_BEFORE_SL"] = None
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def _run(tmp_path, **kwargs):
    return train_challenger(tmp_path / "dataset.csv", tmp_path / "models", tmp_path / "registry",
                            min_new_samples=kwargs.pop("min_new_samples", 1), **kwargs)


def test_real_training_registers_only_candidate_and_repeat_is_rejected(tmp_path):
    _dataset(tmp_path / "dataset.csv")
    report = _run(tmp_path)
    assert report["new_records"] == 12
    assert report["champion_comparison"] == "not_evaluated"
    registry = ModelRegistry(tmp_path / "registry")
    assert registry.status()["active"] == {}
    assert registry.verify(report["candidate_model_id"])["status"] == "candidate"
    metadata = json.loads(Path(registry.verify(report["candidate_model_id"])["metadata_path"]).read_text())
    assert metadata["rows"] == {"train": 12, "validation": 6, "test": 6, "forward": 0}
    assert metadata["fit_data"] == "train_only"
    with pytest.raises(ValueError, match="only 0 new"):
        _run(tmp_path)


def test_late_completed_label_is_new_but_forward_and_unknown_labels_are_not(tmp_path):
    frame = _dataset(tmp_path / "dataset.csv", incomplete=True)
    report = _run(tmp_path)
    assert report["new_records"] == 11
    extra = frame.iloc[-1].copy()
    extra["candidate_id"] = "forward"
    extra["timestamp"] = "2026-09-01T00:00:00Z"
    extra["label_end_timestamp"] = "2026-09-01T00:30:00Z"
    frame = pd.concat([frame, pd.DataFrame([extra])], ignore_index=True)
    frame.to_csv(tmp_path / "dataset.csv", index=False)
    with pytest.raises(ValueError, match="only 0 new"):
        _run(tmp_path)
    frame.loc[0, "TP_BEFORE_SL"] = 0
    frame.to_csv(tmp_path / "dataset.csv", index=False)
    assert _run(tmp_path)["new_records"] == 1


@pytest.mark.parametrize("failure", ["training", "registration"])
def test_failed_job_keeps_champion_and_watermark_and_releases_lock(tmp_path, failure):
    frame = _dataset(tmp_path / "dataset.csv")
    report = _run(tmp_path)
    registry = ModelRegistry(tmp_path / "registry")
    registry.approve(report["candidate_model_id"], approver="fixture", evidence="test evidence")
    registry.activate(report["candidate_model_id"])
    original_registry = registry.manifest.read_bytes()
    state = tmp_path / "models" / "retraining_state.json"
    original_state = state.read_bytes()
    frame.loc[0, "TP_BEFORE_SL"] = 1
    frame.to_csv(tmp_path / "dataset.csv", index=False)
    target = "fxbot.baseline_model.train_xgboost_baseline" if failure == "training" else "fxbot.retrain.ModelRegistry.register"
    with patch(target, side_effect=RuntimeError("injected")):
        with pytest.raises(RuntimeError, match="injected"):
            _run(tmp_path)
    assert state.read_bytes() == original_state
    assert registry.manifest.read_bytes() == original_registry
    assert not (tmp_path / "models" / "retraining.lock").exists()
    assert not list((tmp_path / "models").glob(".challenger-*"))


@pytest.mark.parametrize("problem", ["duplicate", "bad_target", "missing_period", "label_unknown", "empty", "missing"])
def test_invalid_inputs_do_not_register_or_advance_state(tmp_path, problem):
    frame = _dataset(tmp_path / "dataset.csv")
    if problem == "duplicate":
        frame.loc[1, "candidate_id"] = frame.loc[0, "candidate_id"]
    elif problem == "bad_target":
        frame.loc[0, "TP_BEFORE_SL"] = 2
    elif problem == "missing_period":
        frame = frame.iloc[:12]
    elif problem == "label_unknown":
        frame["label_end_timestamp"] = None
    elif problem == "empty":
        frame = frame.iloc[:0]
    frame.to_csv(tmp_path / "dataset.csv", index=False)
    if problem == "missing":
        (tmp_path / "dataset.csv").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        _run(tmp_path)
    assert not (tmp_path / "models" / "retraining_state.json").exists()
    assert ModelRegistry(tmp_path / "registry").status()["models"] == []


def test_concurrent_job_and_interrupted_job_lock_fail_closed(tmp_path):
    _dataset(tmp_path / "dataset.csv")
    lock = tmp_path / "models" / "retraining.lock"
    with exclusive_file_lock(lock):
        with pytest.raises(RegistryError):
            _run(tmp_path)
    lock.write_text("operator-must-confirm-stale-lock")
    with pytest.raises(RegistryError):
        _run(tmp_path)
    assert lock.exists()


def test_atomic_json_failure_cleans_temp_and_preserves_original(tmp_path):
    path = tmp_path / "state.json"
    _atomic_json(path, {"valid": True})
    with pytest.raises(ValueError):
        _atomic_json(path, {"bad": float("nan")})
    assert json.loads(path.read_text()) == {"valid": True}
    assert not list(tmp_path.glob(".run-*"))
