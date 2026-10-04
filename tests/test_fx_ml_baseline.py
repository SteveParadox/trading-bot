from __future__ import annotations

import json

import pandas as pd
import pytest

from fxbot.baseline_model import XGBoostBaselineConfig, train_xgboost_baseline
from fxbot.chronological_split import ChronologicalSplitConfig, chronological_split
from fxbot.training_dataset import FEATURE_COLUMNS


def _row(timestamp: str, candidate_id: str, target: int, value: float = 1.0) -> dict:
    row = {
        "timestamp": timestamp,
        "candidate_id": candidate_id,
        "TP_BEFORE_SL": target,
        "PROFITABLE_WITHIN_5_MIN": target,
        "PROFITABLE_WITHIN_15_MIN": target,
    }
    categorical = {
        "symbol": "EUR_USD",
        "direction": "LONG",
        "strategy_signal": "signal_confirmed",
        "session": "london",
        "account_currency": "USD",
        "news_risk": "LOW",
        "upcoming_news_currency": "USD",
        "upcoming_news_impact": "LOW",
        "recent_news_currency": "EUR",
        "news_freshness_state": "FRESH",
    }
    for column in FEATURE_COLUMNS:
        if column in categorical:
            row[column] = categorical[column]
        elif column in {"news_event_just_occurred", "news_stale"}:
            row[column] = False
        else:
            row[column] = value
    return row


def test_chronological_split_uses_explicit_half_open_boundaries() -> None:
    frame = pd.DataFrame([
        _row("2023-01-01T00:00:00+00:00", "train-start", 0),
        _row("2024-12-31T23:59:59+00:00", "train-end", 1),
        _row("2025-01-01T00:00:00+00:00", "validation-start", 0),
        _row("2025-12-31T23:59:59+00:00", "validation-end", 1),
        _row("2026-01-01T00:00:00+00:00", "test-start", 0),
        _row("2026-06-30T23:59:59+00:00", "test-end", 1),
        _row("2026-07-01T00:00:00+00:00", "forward-start", 1),
        _row("2026-10-04T11:00:00+00:00", "forward-current", 0),
    ])

    splits = chronological_split(frame)

    assert splits.train["candidate_id"].tolist() == ["train-start", "train-end"]
    assert splits.validation["candidate_id"].tolist() == ["validation-start", "validation-end"]
    assert splits.test["candidate_id"].tolist() == ["test-start", "test-end"]
    assert splits.forward["candidate_id"].tolist() == ["forward-start", "forward-current"]


def test_chronological_split_sorts_each_period_without_random_shuffle() -> None:
    frame = pd.DataFrame([
        _row("2024-06-01T00:00:00+00:00", "later", 1),
        _row("2023-06-01T00:00:00+00:00", "earlier", 0),
    ])
    splits = chronological_split(frame)
    assert splits.train["candidate_id"].tolist() == ["earlier", "later"]


def test_baseline_refuses_to_borrow_future_rows_when_history_is_missing(tmp_path) -> None:
    frame = pd.DataFrame([
        _row("2026-08-01T00:00:00+00:00", "forward-only", 1),
    ])
    with pytest.raises(ValueError, match="Do not borrow rows from later periods"):
        train_xgboost_baseline(
            frame,
            tmp_path,
            model_config=XGBoostBaselineConfig(n_estimators=5, max_depth=2),
        )


def test_xgboost_baseline_fits_train_only_and_reports_validation_and_test(tmp_path) -> None:
    rows: list[dict] = []

    for index in range(24):
        year = 2023 if index < 12 else 2024
        month = (index % 12) + 1
        rows.append(
            _row(
                f"{year}-{month:02d}-15T12:00:00+00:00",
                f"train-{index}",
                index % 2,
                value=float(index + 1),
            )
        )

    for index in range(8):
        rows.append(
            _row(
                f"2025-{index + 1:02d}-15T12:00:00+00:00",
                f"validation-{index}",
                index % 2,
                value=float(index + 30),
            )
        )

    for index in range(8):
        rows.append(
            _row(
                f"2026-{index % 6 + 1:02d}-15T12:00:00+00:00",
                f"test-{index}",
                index % 2,
                value=float(index + 50),
            )
        )

    for index in range(4):
        rows.append(
            _row(
                f"2026-{index + 7:02d}-15T12:00:00+00:00",
                f"forward-{index}",
                index % 2,
                value=float(index + 70),
            )
        )

    frame = pd.DataFrame(rows)
    artifacts = train_xgboost_baseline(
        frame,
        tmp_path,
        split_config=ChronologicalSplitConfig(),
        model_config=XGBoostBaselineConfig(
            n_estimators=10,
            max_depth=2,
            learning_rate=0.1,
            min_child_weight=1.0,
        ),
    )

    assert artifacts.model_path.exists()
    assert artifacts.metadata_path.exists()
    metadata = json.loads(artifacts.metadata_path.read_text())
    assert metadata["model_name"] == "XGBoost"
    assert metadata["target"] == "TP_BEFORE_SL"
    assert metadata["fit_data"] == "train_only"
    assert metadata["validation_used_for_fit"] is False
    assert metadata["test_used_for_fit"] is False
    assert metadata["forward_used_for_fit"] is False
    assert metadata["random_shuffle"] is False
    assert metadata["rows"] == {
        "train": 24,
        "validation": 8,
        "test": 8,
        "forward": 4,
    }
    assert artifacts.validation_metrics["rows"] == 8
    assert artifacts.test_metrics["rows"] == 8
    assert 0.0 <= artifacts.validation_metrics["brier_score"] <= 1.0
    assert 0.0 <= artifacts.test_metrics["brier_score"] <= 1.0
