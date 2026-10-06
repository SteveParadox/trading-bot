"""Train the first five-class sniper entry-timing model with XGBoost.

Target actions:
ENTER_NOW / WAIT_30S / WAIT_1M / WAIT_3M / SKIP

Training remains chronological and research-only. The model is not permitted to
submit, modify, or size orders.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

from fxbot.ai.feature_builder import FEATURE_BUILDER_VERSION
from fxbot.chronological_split import ChronologicalSplitConfig, chronological_split
from fxbot.ai.preprocessing import CATEGORICAL_FEATURES, feature_frame
from fxbot.training_dataset import ENTRY_ACTIONS, FEATURE_COLUMNS


ENTRY_TIMING_MODEL_VERSION = "xgb_entry_timing_v1"
ENTRY_TIMING_TARGET = "ENTRY_ACTION_LABEL"


@dataclass(frozen=True)
class EntryTimingModelConfig:
    n_estimators: int = 400
    max_depth: int = 4
    learning_rate: float = 0.05
    min_child_weight: float = 5.0
    reg_lambda: float = 1.0
    random_state: int = 42
    min_samples_per_class: int = 20

    def validate(self) -> None:
        if self.min_samples_per_class < 2:
            raise ValueError("min_samples_per_class must be at least 2")
        if self.n_estimators <= 0 or self.max_depth <= 0 or self.learning_rate <= 0:
            raise ValueError("invalid entry-timing XGBoost hyperparameters")


@dataclass(frozen=True)
class EntryTimingArtifacts:
    model_path: Path
    metadata_path: Path
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]


def train_entry_timing_model(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    split_config: ChronologicalSplitConfig | None = None,
    model_config: EntryTimingModelConfig | None = None,
) -> EntryTimingArtifacts:
    cfg = model_config or EntryTimingModelConfig()
    cfg.validate()
    split_cfg = split_config or ChronologicalSplitConfig()
    splits = chronological_split(frame, config=split_cfg)

    train = _ready(splits.train)
    validation = _ready(splits.validation)
    test = _ready(splits.test)
    forward = splits.forward.copy()
    _require_periods(train, validation, test)

    observed = set(train[ENTRY_TIMING_TARGET].astype(str))
    required = set(ENTRY_ACTIONS)
    if observed != required:
        missing = sorted(required - observed)
        extra = sorted(observed - required)
        raise ValueError(
            "entry-timing training period must contain the complete action space; "
            f"missing={missing}, extra={extra}"
        )

    label_to_index = {label: index for index, label in enumerate(ENTRY_ACTIONS)}
    counts = train[ENTRY_TIMING_TARGET].value_counts()
    if (counts < cfg.min_samples_per_class).any():
        raise ValueError(f"insufficient per-class training support (minimum {cfg.min_samples_per_class}): {counts.to_dict()}")
    x_train = _feature_frame(train)
    y_train = train[ENTRY_TIMING_TARGET].map(label_to_index).astype(int)
    x_validation = _feature_frame(validation)
    y_validation = validation[ENTRY_TIMING_TARGET].map(label_to_index)
    x_test = _feature_frame(test)
    y_test = test[ENTRY_TIMING_TARGET].map(label_to_index)
    if y_validation.isna().any() or y_test.isna().any():
        raise ValueError("validation/test entry actions contain unsupported labels")

    pipeline = _pipeline(cfg)
    pipeline.fit(x_train, y_train)

    validation_metrics = _multiclass_metrics(
        y_validation.astype(int).to_numpy(),
        pipeline.predict_proba(x_validation),
    )
    test_metrics = _multiclass_metrics(
        y_test.astype(int).to_numpy(),
        pipeline.predict_proba(x_test),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{ENTRY_TIMING_MODEL_VERSION}.joblib"
    joblib.dump(pipeline, model_path)
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()

    metadata_path = output_dir / f"{ENTRY_TIMING_MODEL_VERSION}.metadata.json"
    metadata = {
        "model_name": "XGBoost",
        "model_version": ENTRY_TIMING_MODEL_VERSION,
        "model_role": "entry_timing_candidate",
        "target": ENTRY_TIMING_TARGET,
        "problem_type": "multiclass_classification",
        "class_labels": list(ENTRY_ACTIONS),
        "class_to_index": label_to_index,
        "feature_columns": FEATURE_COLUMNS,
        "feature_builder_version": FEATURE_BUILDER_VERSION,
        "categorical_features": CATEGORICAL_FEATURES,
        "model_config": asdict(cfg),
        "split_config": asdict(split_cfg),
        "fit_data": "train_only",
        "validation_used_for_fit": False,
        "test_used_for_fit": False,
        "forward_used_for_fit": False,
        "random_shuffle": False,
        "model_sha256": model_hash,
        "dataset_sha256": hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest(),
        "training_rows_sha256": hashlib.sha256(train.to_csv(index=False).encode()).hexdigest(),
        "dependency_versions": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "xgboost", "joblib")},
        "dataset_versions": sorted(train["dataset_version"].dropna().astype(str).unique()) if "dataset_version" in train else [],
        "label_versions": sorted(train["label_version"].dropna().astype(str).unique()) if "label_version" in train else [],
        "rows": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "test": int(len(test)),
            "forward": int(len(forward)),
        },
        "class_counts": {
            split: {
                label: int((data[ENTRY_TIMING_TARGET] == label).sum())
                for label in ENTRY_ACTIONS
            }
            for split, data in (
                ("train", train),
                ("validation", validation),
                ("test", test),
            )
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "promotion_status": "candidate_only",
        "live_execution_enabled": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    return EntryTimingArtifacts(
        model_path=model_path,
        metadata_path=metadata_path,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
    )


def _pipeline(config: EntryTimingModelConfig) -> Pipeline:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError(
            "xgboost is required for entry timing; install requirements-fx-research.txt"
        ) from exc

    categorical = [column for column in CATEGORICAL_FEATURES if column in FEATURE_COLUMNS]
    numeric = [column for column in FEATURE_COLUMNS if column not in categorical]
    preprocess = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]),
                numeric,
            ),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN", keep_empty_features=True)),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical,
            ),
        ],
        remainder="drop",
    )
    classifier = XGBClassifier(
        objective="multi:softprob",
        num_class=len(ENTRY_ACTIONS),
        eval_metric="mlogloss",
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        min_child_weight=config.min_child_weight,
        reg_lambda=config.reg_lambda,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=config.random_state,
        n_jobs=1,
        tree_method="hist",
        verbosity=0,
    )
    return Pipeline([
        ("coerce", FunctionTransformer(feature_frame)),
        ("preprocess", preprocess),
        ("model", classifier),
    ])


_feature_frame = feature_frame

def _ready(frame: pd.DataFrame) -> pd.DataFrame:
    if ENTRY_TIMING_TARGET not in frame.columns:
        raise ValueError(f"dataset is missing target column {ENTRY_TIMING_TARGET}")
    labels = frame[ENTRY_TIMING_TARGET].astype("string")
    valid = labels.isin(ENTRY_ACTIONS)
    ready = frame.loc[valid].copy()
    ready[ENTRY_TIMING_TARGET] = labels.loc[valid]
    return ready.reset_index(drop=True)


def _require_periods(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    missing = [
        name
        for name, split in (
            ("train", train),
            ("validation", validation),
            ("test", test),
        )
        if split.empty
    ]
    if missing:
        raise ValueError(
            f"cannot train entry timing: chronological split(s) have no labeled rows: {missing}"
        )


def _multiclass_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    probs = np.asarray(probabilities, dtype=float)
    predictions = np.argmax(probs, axis=1)
    labels = list(range(len(ENTRY_ACTIONS)))
    return {
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, labels=labels, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_true, probs, labels=labels)),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=labels).tolist(),
        "class_labels": list(ENTRY_ACTIONS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-start", default="2023-01-01")
    parser.add_argument("--validation-start", default="2025-01-01")
    parser.add_argument("--test-start", default="2026-01-01")
    parser.add_argument("--forward-start", default="2026-07-01")
    parser.add_argument("--min-samples-per-class", type=int, default=20)
    args = parser.parse_args()

    frame = pd.read_csv(args.dataset)
    artifacts = train_entry_timing_model(
        frame,
        args.output_dir,
        model_config=EntryTimingModelConfig(min_samples_per_class=args.min_samples_per_class),
        split_config=ChronologicalSplitConfig(
            train_start=args.train_start,
            validation_start=args.validation_start,
            test_start=args.test_start,
            forward_start=args.forward_start,
        ),
    )
    print(json.dumps({
        "model": str(artifacts.model_path),
        "metadata": str(artifacts.metadata_path),
        "validation_metrics": artifacts.validation_metrics,
        "test_metrics": artifacts.test_metrics,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
