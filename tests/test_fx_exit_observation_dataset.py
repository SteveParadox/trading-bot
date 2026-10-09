"""Observation horizon samples must never invent an exit policy label."""
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fxbot.ai.exit_intelligence import build_exit_snapshot
from fxbot.ai.exit_observation_dataset import (
    TARGET_COLUMNS, build_observed_exit_rows, export_observed_exit_rows,
)


BASE = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)


def _observation(seconds=0, *, direction="BUY", offset_pips=0.0,
                 quote_lag=0, opened="2026-10-09T11:00:00+00:00"):
    now = BASE + timedelta(seconds=seconds)
    signed_units = 1000 if direction == "BUY" else -1000
    bid = 1.1060 + offset_pips * .0001
    quote = SimpleNamespace(
        bid=bid, ask=bid+.0002, time=now-timedelta(seconds=quote_lag),
    )
    trade = {
        "id": "1234", "instrument": "EUR_USD", "currentUnits": signed_units,
        "price": 1.1050, "openTime": opened, "unrealizedPL": 1.0,
        "stopLossOrder": {"price": 1.1000},
        "takeProfitOrder": {"price": 1.1100},
    }
    instrument = SimpleNamespace(name="EUR_USD", pip_size=.0001)
    return {
        "snapshot": build_exit_snapshot(trade, quote, instrument, now),
        "prediction": {"prediction_id": str(seconds), "model_version": "none"},
    }


def test_future_marks_at_horizon_are_labels_not_features():
    source = [_observation(), _observation(60, offset_pips=4)]
    rows = build_observed_exit_rows(source)
    assert len(rows) == 2
    assert rows[0]["observed_mark_return_60s_pips"] == pytest.approx(4)
    assert rows[0]["observed_mark_return_180s_pips"] is None
    assert rows[0]["label_end_timestamp"] == (BASE+timedelta(seconds=60)).isoformat()
    assert rows[1]["observed_mark_return_60s_pips"] is None
    assert not any(name.startswith("observed_mark") for name in
                   ("direction", "spread_pips", "pnl_pips"))


def test_future_quote_not_yet_at_horizon_is_unknown():
    rows = build_observed_exit_rows([_observation(), _observation(60, offset_pips=4, quote_lag=1)])
    assert rows[0]["observed_mark_return_60s_pips"] is None
    assert rows[0]["label_end_timestamp"] is None


def test_late_sample_and_reopened_ticket_have_no_fabricated_label():
    rows = build_observed_exit_rows([_observation(), _observation(100, offset_pips=5)])
    assert rows[0]["observed_mark_return_60s_pips"] is None
    rows = build_observed_exit_rows([
        _observation(), _observation(60, offset_pips=5, opened="2026-10-09T11:30:00+00:00"),
    ])
    assert rows[0]["observed_mark_return_60s_pips"] is None


def test_sell_mark_return_uses_ask_in_correct_direction():
    rows = build_observed_exit_rows([
        _observation(direction="SELL"),
        _observation(60, direction="SELL", offset_pips=3),
    ])
    assert rows[0]["observed_mark_return_60s_pips"] == pytest.approx(-3)


def test_duplicates_and_bad_future_data_never_expose_future_as_feature():
    now = _observation()
    next_sample = _observation(60, offset_pips=2)
    rows = build_observed_exit_rows([now, now, next_sample])
    assert len(rows) == 2
    bad = _observation()
    bad["snapshot"]["timestamp"] = (BASE - timedelta(seconds=5)).isoformat()
    assert len(build_observed_exit_rows([bad, next_sample])) == 1


def test_export_separates_feature_target_and_audit_manifests(tmp_path: Path):
    rows = build_observed_exit_rows([_observation(), _observation(60, offset_pips=3)])
    output = tmp_path / "observed.csv"
    metadata = export_observed_exit_rows(rows, output)
    assert metadata["exit_action_labels_available"] is False
    assert metadata["broker_execution_counterfactuals"] is False
    assert len(set(metadata["feature_columns"]) & set(metadata["target_columns"])) == 0
    assert len(set(metadata["audit_columns"]) & set(metadata["target_columns"])) == 0
    with output.open(newline="") as handle:
        exported = list(csv.DictReader(handle))
    assert len(exported) == 2
    assert exported[0][TARGET_COLUMNS[0]]
    assert exported[1][TARGET_COLUMNS[0]] == ""
    assert json.loads(output.with_suffix(".csv.metadata.json").read_text()) == metadata
