"""Repeatable off-engine entry-model challenger retraining.

The command never edits an active model or changes MT5 configuration. Feed it
a freshly exported causal training-dataset-v2 CSV on a scheduler. Exit-model
training is held until decision-level counterfactual labels are trustworthy.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from uuid import uuid4

from fxbot.ai.model_registry import ModelRegistry, RegistryError, exclusive_file_lock


def _atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", prefix=".run-",
                                     encoding="utf-8", dir=path.parent, delete=False) as handle:
        name = Path(handle.name)
        try:
            json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            name.unlink(missing_ok=True)
            raise
    try:
        os.replace(name, path)
    finally:
        name.unlink(missing_ok=True)


def train_challenger(
    dataset: Path,
    artifacts_root: Path,
    registry_root: Path,
    *,
    target: str = "TP_BEFORE_SL",
    min_new_samples: int = 100,
    train_start: str = "2023-01-01",
    validation_start: str = "2025-01-01",
    test_start: str = "2026-01-01",
    forward_start: str = "2026-07-01",
) -> dict:
    """Train exactly one registered candidate; default deployment stays unchanged."""
    if min_new_samples < 1:
        raise ValueError("min_new_samples must be positive")
    # Optional heavy imports stay out of live worker startup.
    import pandas as pd
    from fxbot.baseline_model import XGBoostBaselineConfig, train_xgboost_baseline
    from fxbot.chronological_split import ChronologicalSplitConfig
    from fxbot.training_dataset import FEATURE_COLUMNS

    artifacts_root.mkdir(parents=True, exist_ok=True)
    registry = ModelRegistry(registry_root)
    with exclusive_file_lock(artifacts_root / "retraining.lock"):
        frame = pd.read_csv(dataset)
        required = {"candidate_id", "timestamp", target, *FEATURE_COLUMNS}
        missing = required - set(frame)
        if missing:
            raise ValueError(f"required dataset columns missing: {sorted(missing)}")
        if frame.empty or frame["candidate_id"].isna().any() or frame["candidate_id"].duplicated().any():
            raise ValueError("dataset is empty or contains duplicate/invalid candidate IDs")
        timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
        if timestamps.isna().any():
            raise ValueError("dataset timestamp missing")
        state_path = artifacts_root / "retraining_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
        last_timestamp = state.get("last_timestamp")
        count_new = int((timestamps > pd.Timestamp(last_timestamp)).sum()) if last_timestamp else len(frame)
        if count_new < min_new_samples:
            raise ValueError(f"only {count_new} new records, requires {min_new_samples}")
        now = datetime.now(timezone.utc)
        run_id = f"{now.strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
        stage = Path(tempfile.mkdtemp(prefix=".challenger-", dir=artifacts_root))
        destination = artifacts_root / run_id
        moved = False
        try:
            split = ChronologicalSplitConfig(train_start=train_start,
                                              validation_start=validation_start,
                                              test_start=test_start,
                                              forward_start=forward_start)
            artifacts = train_xgboost_baseline(
                frame, stage,
                split_config=split,
                model_config=XGBoostBaselineConfig(target=target),
            )
            meta = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
            # Baseline research uses a fixed model family version. Challenger
            # runs must instead have unique immutable version identities.
            meta["model_version"] = f"{meta['model_version']}_{run_id}"
            meta["training_run_id"] = run_id
            meta["trained_at"] = now.isoformat()
            meta["promotion_status"] = "candidate_only"
            meta["forward_used_for_fit"] = False
            meta["live_execution_enabled"] = False
            _atomic_json(artifacts.metadata_path, meta)
            os.replace(stage, destination)
            moved = True
            model = destination / artifacts.model_path.name
            metadata = destination / artifacts.metadata_path.name
            record = registry.register(model_path=model, metadata_path=metadata,
                                       model_type="entry_quality")
            report = {
                "run_id": run_id,
                "candidate_model_id": record["model_id"],
                "new_records": count_new,
                "validation_metrics": artifacts.validation_metrics,
                "test_metrics": artifacts.test_metrics,
                "champion_comparison": "not_evaluated",
                "promotion": "manual_review_required",
                "forward_evaluation": "held_out",
            }
            _atomic_json(destination / "retraining_report.json", report)
            # Advance only after a candidate was durably registered.
            _atomic_json(state_path, {
                "last_timestamp": timestamps.max().isoformat(),
                "last_run_id": run_id,
                "dataset_path": str(dataset),
            })
            return report
        finally:
            if not moved:
                import shutil
                shutil.rmtree(stage, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, default=Path("data/models/challengers"))
    parser.add_argument("--registry-root", type=Path, default=Path("data/models/registry"))
    parser.add_argument("--target", default="TP_BEFORE_SL")
    parser.add_argument("--min-new-samples", type=int, default=100)
    parser.add_argument("--train-start", default="2023-01-01")
    parser.add_argument("--validation-start", default="2025-01-01")
    parser.add_argument("--test-start", default="2026-01-01")
    parser.add_argument("--forward-start", default="2026-07-01")
    args = parser.parse_args()
    result = train_challenger(args.dataset, args.artifacts_root, args.registry_root,
                              target=args.target, min_new_samples=args.min_new_samples,
                              train_start=args.train_start,
                              validation_start=args.validation_start,
                              test_start=args.test_start,
                              forward_start=args.forward_start)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
