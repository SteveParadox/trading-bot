"""Train the first tabular FX trade-quality baseline with XGBoost.

The default target is TP_BEFORE_SL. Training is chronological:
- fit on TRAIN only
- evaluate on VALIDATION
- evaluate once on TEST
- never fit on FORWARD data

This module is research-only and does not load models into live execution.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from fxbot.chronological_split import ChronologicalSplitConfig, chronological_split
from fxbot.ai.feature_builder import FEATURE_BUILDER_VERSION
from fxbot.training_dataset import FEATURE_COLUMNS


BASELINE_MODEL_FAMILY_VERSION = "v1"
SUPPORTED_BINARY_TARGETS = {
    "TP_BEFORE_SL",
    "PROFITABLE_WITHIN_5_MIN",
    "PROFITABLE_WITHIN_15_MIN",
}

CATEGORICAL_FEATURES = [
    "symbol",
    "direction",
    "strategy_signal",
    "session",
    "account_currency",
    "news_risk",
    "upcoming_news_currency",
    "upcoming_news_impact",
    "recent_news_currency",
    "news_freshness_state",
]


@dataclass(frozen=True)
class XGBoostBaselineConfig:
    target: str = "TP_BEFORE_SL"
    n_estimators: int = 300
    max_depth: int = 4
    learning_rate: float = 0.05
    min_child_weight: float = 5.0
    reg_lambda: float = 1.0
    random_state: int = 42

    def validate(self) -> None:
        if self.target not in SUPPORTED_BINARY_TARGETS:
            raise ValueError(
                f"unsupported baseline target {self.target!r}; "
                f"supported={sorted(SUPPORTED_BINARY_TARGETS)}"
            )
        if self.n_estimators <= 0 or self.max_depth <= 0 or self.learning_rate <= 0:
            raise ValueError("invalid XGBoost baseline hyperparameters")


@dataclass(frozen=True)
class BaselineArtifacts:
    model_path: Path
    metadata_path: Path
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]


def train_xgboost_baseline(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    split_config: ChronologicalSplitConfig | None = None,
    model_config: XGBoostBaselineConfig | None = None,
) -> BaselineArtifacts:
    """Train a leakage-safe chronological XGBoost classification baseline."""

    cfg = model_config or XGBoostBaselineConfig()
    cfg.validate()
    split_cfg = split_config or ChronologicalSplitConfig()
    splits = chronological_split(frame, config=split_cfg)

    train = _target_ready(splits.train, cfg.target)
    validation = _target_ready(splits.validation, cfg.target)
    test = _target_ready(splits.test, cfg.target)
    forward = splits.forward.copy()

    _require_training_periods(train, validation, test, cfg.target)

    x_train = _feature_frame(train)
    y_train = train[cfg.target].astype(int)
    x_validation = _feature_frame(validation)
    y_validation = validation[cfg.target].astype(int)
    x_test = _feature_frame(test)
    y_test = test[cfg.target].astype(int)

    if y_train.nunique() < 2:
        raise ValueError(f"training target {cfg.target} requires both classes")

    pipeline = _pipeline(cfg)
    pipeline.fit(x_train, y_train)

    validation_metrics = _classification_metrics(
        y_validation,
        pipeline.predict_proba(x_validation)[:, 1],
    )
    test_metrics = _classification_metrics(
        y_test,
        pipeline.predict_proba(x_test)[:, 1],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_version = _model_version(cfg.target)
    model_path = output_dir / f"{model_version}.joblib"
    joblib.dump(pipeline, model_path)
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()

    metadata_path = output_dir / f"{model_version}.metadata.json"
    metadata = {
        "model_name": "XGBoost",
        "model_version": model_version,
        "model_role": "candidate",
        "target": cfg.target,
        "problem_type": "binary_classification",
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
        "rows": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "test": int(len(test)),
            "forward": int(len(forward)),
        },
        "class_balance": {
            "train_positive_rate": float(y_train.mean()),
            "validation_positive_rate": float(y_validation.mean()),
            "test_positive_rate": float(y_test.mean()),
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "promotion_status": "candidate_only",
        "live_execution_enabled": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    return BaselineArtifacts(
        model_path=model_path,
        metadata_path=metadata_path,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
    )


def _model_version(target: str) -> str:
    slug = target.lower()
    return f"xgb_{slug}_{BASELINE_MODEL_FAMILY_VERSION}"


def _pipeline(config: XGBoostBaselineConfig) -> Pipeline:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError(
            "xgboost is required for the baseline; install requirements-fx-research.txt"
        ) from exc

    categorical = [column for column in CATEGORICAL_FEATURES if column in FEATURE_COLUMNS]
    numeric = [column for column in FEATURE_COLUMNS if column not in categorical]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
    )
    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
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
        ("preprocess", preprocess),
        ("model", classifier),
    ])


def _feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"dataset is missing model feature columns: {missing}")
    features = frame.loc[:, FEATURE_COLUMNS].copy()
    for column in CATEGORICAL_FEATURES:
        if column in features:
            features[column] = features[column].astype("string")
    for column in features.columns:
        if column not in CATEGORICAL_FEATURES:
            features[column] = pd.to_numeric(features[column], errors="coerce")
    return features


def _target_ready(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    if target not in frame.columns:
        raise ValueError(f"dataset is missing target column {target}")
    if frame.empty:
        return frame.copy()
    numeric = pd.to_numeric(frame[target], errors="coerce")
    valid = numeric.isin([0, 1])
    ready = frame.loc[valid].copy()
    ready[target] = numeric.loc[valid].astype(int)
    return ready.reset_index(drop=True)


def _require_training_periods(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
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
            f"cannot train {target}: chronological split(s) have no labeled rows: {missing}. "
            "Do not borrow rows from later periods; load the required historical candidate dataset."
        )


def _classification_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    probs = np.asarray(probabilities, dtype=float)
    predictions = (probs >= 0.5).astype(int)
    y = np.asarray(y_true, dtype=int)
    metrics: dict[str, Any] = {
        "rows": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(y, probs)),
    }
    labels = np.unique(y)
    if len(labels) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y, probs))
        metrics["average_precision"] = float(average_precision_score(y, probs))
        metrics["log_loss"] = float(log_loss(y, probs, labels=[0, 1]))
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None
        metrics["log_loss"] = None
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", default="TP_BEFORE_SL", choices=sorted(SUPPORTED_BINARY_TARGETS))
    parser.add_argument("--train-start", default="2023-01-01")
    parser.add_argument("--validation-start", default="2025-01-01")
    parser.add_argument("--test-start", default="2026-01-01")
    parser.add_argument("--forward-start", default="2026-07-01")
    args = parser.parse_args()

    frame = pd.read_csv(args.dataset)
    split_config = ChronologicalSplitConfig(
        train_start=args.train_start,
        validation_start=args.validation_start,
        test_start=args.test_start,
        forward_start=args.forward_start,
    )
    artifacts = train_xgboost_baseline(
        frame,
        args.output_dir,
        split_config=split_config,
        model_config=XGBoostBaselineConfig(target=args.target),
    )
    print(json.dumps({
        "model": str(artifacts.model_path),
        "metadata": str(artifacts.metadata_path),
        "validation_metrics": artifacts.validation_metrics,
        "test_metrics": artifacts.test_metrics,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
