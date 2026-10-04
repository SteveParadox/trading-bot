from __future__ import annotations

import json
from contextlib import closing
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from fxbot.baseline_model import XGBoostBaselineConfig, train_xgboost_baseline
from fxbot.chronological_split import ChronologicalSplitConfig, chronological_split
from fxbot.config import BrokerSettings, FxBotSettings, RiskSettings, StrategySettings
from fxbot.historical_reconstruction import HistoricalReconstructionConfig, reconstruct_historical_candidates
from fxbot.instruments import FxInstrument
from fxbot.journal import StructuredJournal
from fxbot.training_dataset import FEATURE_COLUMNS, build_training_dataset


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


def _trending_history(
    start: str,
    *,
    rows: int,
    freq: str,
    price: float,
    step: float,
) -> pd.DataFrame:
    index = pd.date_range(start, periods=rows, freq=freq)
    close = price + np.arange(rows) * step
    open_ = close - step * 0.5
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + abs(step) * 2,
            "low": np.minimum(open_, close) - abs(step) * 2,
            "close": close,
            "volume": np.linspace(100, 150, rows),
            "spread_points": 10,
        },
        index=index,
    )


class _HistoricalSource:
    def __init__(self) -> None:
        self.entry = _trending_history(
            "2023-12-25T00:00:00Z",
            rows=900,
            freq="15min",
            price=1.0800,
            step=0.00005,
        )
        self.htf = _trending_history(
            "2023-12-20T00:00:00Z",
            rows=400,
            freq="1h",
            price=1.0600,
            step=0.00010,
        )

    def instruments(self, names):
        return {name: FxInstrument(name) for name in names}

    def historical_candles(self, instrument, timeframe, start, end):
        frame = self.entry if timeframe == "15m" else self.htf
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        return frame.loc[(frame.index >= start_ts) & (frame.index <= end_ts)].copy()

    def historical_ticks(self, instrument, start, end):
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        closed = self.entry.loc[self.entry.index + pd.Timedelta(minutes=15) <= start_ts]
        if closed.empty:
            return pd.DataFrame(columns=["timestamp", "instrument", "bid", "ask"])
        base = float(closed.iloc[-1]["close"])
        index = pd.date_range(start_ts, end_ts, freq="10s")
        bid = base + np.arange(len(index)) * 0.00003
        return pd.DataFrame({
            "timestamp": index,
            "instrument": instrument,
            "bid": bid,
            "ask": bid + 0.00010,
        })


def test_historical_reconstruction_builds_2024_candidate_outcome_and_training_row(tmp_path) -> None:
    settings = FxBotSettings(
        instruments=["EUR_USD"],
        broker=BrokerSettings(),
        strategy=StrategySettings(
            require_volume_confirmation=False,
            min_atr_pips=0.1,
            max_atr_pips=100.0,
            adx_min=10.0,
            htf_adx_min=10.0,
            trade_sessions_utc=(),
            avoid_rollover_minutes=0,
            close_before_weekend_minutes=0,
            partial_tp_enabled=False,
        ),
        risk=RiskSettings(),
    )
    start = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
    with closing(StructuredJournal(f"sqlite:///{tmp_path / 'historical.db'}")) as journal:
        report = reconstruct_historical_candidates(
            source=_HistoricalSource(),
            settings=settings,
            journal=journal,
            config=HistoricalReconstructionConfig(
                start=start,
                end=start + timedelta(minutes=1),
                outcome_lag_tolerance_seconds=30,
            ),
            news_authoritative=False,
        )

        assert report["totals"]["decision_points"] == 1
        assert report["totals"]["strategy_candidates"] == 1
        assert report["totals"]["candidates_with_entry_quotes"] == 1
        assert report["totals"]["complete_outcomes"] == 1

        candidates = journal.recent_candidates()
        assert len(candidates) == 1
        candidate = candidates[0]
        assert candidate.timestamp.replace(tzinfo=timezone.utc).year == 2024
        assert candidate.payload["historical_reconstruction"] is True
        assert candidate.executed is False

        outcome = journal.find_candidate_outcome(candidate.candidate_id)
        assert outcome.status == "complete"
        assert outcome.observation_count > 100
        assert outcome.tp_hit is True
        assert outcome.tp_before_sl is True

        dataset = build_training_dataset(journal)
        assert len(dataset) == 1
        assert dataset.iloc[0]["candidate_id"] == candidate.candidate_id
        assert pd.Timestamp(dataset.iloc[0]["timestamp"]).year == 2024
        assert dataset.iloc[0]["TP_BEFORE_SL"] == 1


def test_historical_reconstruction_requires_authoritative_news_when_production_requires_it(tmp_path) -> None:
    settings = FxBotSettings(
        instruments=["EUR_USD"],
        strategy=StrategySettings(require_news_data=True),
    )
    with closing(StructuredJournal(f"sqlite:///{tmp_path / 'historical.db'}")) as journal:
        with pytest.raises(ValueError, match="authoritative historical news"):
            reconstruct_historical_candidates(
                source=_HistoricalSource(),
                settings=settings,
                journal=journal,
                config=HistoricalReconstructionConfig(
                    start=datetime(2024, 1, 2, tzinfo=timezone.utc),
                    end=datetime(2024, 1, 3, tzinfo=timezone.utc),
                ),
                news_events=[],
                news_authoritative=False,
            )


def test_authoritative_empty_historical_news_window_is_not_treated_as_missing(tmp_path) -> None:
    settings = FxBotSettings(
        instruments=["EUR_USD"],
        strategy=StrategySettings(
            require_news_data=True,
            require_volume_confirmation=False,
            min_atr_pips=0.1,
            max_atr_pips=100.0,
            adx_min=10.0,
            htf_adx_min=10.0,
            trade_sessions_utc=(),
            avoid_rollover_minutes=0,
            close_before_weekend_minutes=0,
            partial_tp_enabled=False,
        ),
    )
    start = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
    with closing(StructuredJournal(f"sqlite:///{tmp_path / 'historical.db'}")) as journal:
        report = reconstruct_historical_candidates(
            source=_HistoricalSource(),
            settings=settings,
            journal=journal,
            config=HistoricalReconstructionConfig(
                start=start,
                end=start + timedelta(minutes=1),
            ),
            news_events=[],
            news_authoritative=True,
        )
        assert report["totals"]["permission_blocks"] == 0
        assert report["totals"]["strategy_candidates"] == 1


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
