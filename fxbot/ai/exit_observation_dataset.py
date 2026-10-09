"""Causal, observation-only exit horizon dataset, NOT exit action labels.

Reads shadow exit snapshots from the existing event journal. Future mark-to-mark
movement is observable when a sufficiently timely later sample exists. This is
NOT an estimate of what trailing/closing/partial MT5 execution would have done,
and is intentionally unusable as a five-action classifier target.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Iterable

from fxbot.ai.exit_intelligence import EXIT_FEATURE_COLUMNS, EXIT_FEATURE_VERSION, exit_features


HORIZONS_SECONDS = (60, 180, 300)
TARGET_COLUMNS = tuple(f"observed_mark_return_{sec}s_pips" for sec in HORIZONS_SECONDS)
AUDIT_COLUMNS = (
    "position_id", "candidate_id", "timestamp", "quote_timestamp",
    "prediction_id", "model_version", "feature_version",
    "strategy_version", "prompt_version", "label_end_timestamp",
    "observation_quality",
)


def _utc(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("missing timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def build_observed_exit_rows(
    events: Iterable[dict[str, Any]],
    *,
    max_lag_seconds: int = 20,
) -> list[dict[str, Any]]:
    """Mark changes require later observed executable quotes, never future features."""
    if max_lag_seconds < 0 or max_lag_seconds >= min(HORIZONS_SECONDS):
        raise ValueError("max lag must be shorter than the smallest horizon")
    grouped: dict[str, list[tuple[datetime, datetime, dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for event in events:
        if not isinstance(event, dict):
            continue
        snapshot = event.get("snapshot")
        prediction = event.get("prediction") or {}
        if not isinstance(snapshot, dict) or not isinstance(prediction, dict):
            continue
        try:
            if snapshot.get("schema_version") != EXIT_FEATURE_VERSION:
                continue
            captured_at = _utc(snapshot["timestamp"])
            quote_at = _utc(snapshot["quote_timestamp"])
            if quote_at > captured_at or (captured_at - quote_at).total_seconds() > 120:
                continue
            pip = float(snapshot["pip_size"])
            mark = float(snapshot["liquidation_price"])
            if (snapshot.get("direction") not in {"BUY", "SELL"}
                    or not math.isfinite(pip) or pip <= 0
                    or not math.isfinite(mark) or mark <= 0):
                continue
            exit_features(snapshot)  # Reject malformed decision-time feature inputs.
            ticket = str(snapshot.get("position_id") or "")
            if not ticket:
                continue
            key = (ticket, captured_at.isoformat())
            if key in seen:
                continue
            seen.add(key)
            grouped[ticket].append((captured_at, quote_at, snapshot, prediction))
        except (TypeError, ValueError, OverflowError, KeyError):
            continue

    rows: list[dict[str, Any]] = []
    for ticket, observations in grouped.items():
        observations.sort(key=lambda entry: entry[0])
        for index, (at, quote_at, snapshot, prediction) in enumerate(observations):
            features = exit_features(snapshot)
            record: dict[str, Any] = {
                **features,
                "position_id": ticket,
                "candidate_id": snapshot.get("candidate_id"),
                "timestamp": at.isoformat(),
                "quote_timestamp": quote_at.isoformat(),
                "prediction_id": prediction.get("prediction_id"),
                "model_version": prediction.get("model_version"),
                "feature_version": snapshot["schema_version"],
                "strategy_version": snapshot.get("strategy_version"),
                "prompt_version": prediction.get("prompt_version"),
                "label_end_timestamp": None,
                "observation_quality": "scan_sampled_not_tick_complete",
            }
            last_target_at: datetime | None = None
            for horizon, target in zip(HORIZONS_SECONDS, TARGET_COLUMNS):
                record[target] = None
                required = at + timedelta(seconds=horizon)
                # Do not look at quotes observed before the requested horizon.
                # Never consider a different instrument/entry/direction/ticket.
                for future_at, future_quote_at, future, _ in observations[index + 1:]:
                    if future_at < required:
                        continue
                    if (future_at - required).total_seconds() > max_lag_seconds:
                        break
                    # The actual quote, not merely scan time, must have been
                    # observed at or after the horizon and within its lag budget.
                    if (future_quote_at < required
                            or (future_quote_at - required).total_seconds() > max_lag_seconds
                            or future_quote_at <= quote_at):
                        continue
                    if (future.get("direction") != snapshot.get("direction")
                            or future.get("entry_timestamp") != snapshot.get("entry_timestamp")
                            or future.get("symbol") != snapshot.get("symbol")
                            or float(future.get("pip_size") or 0) != float(snapshot["pip_size"])):
                        continue
                    base_mark = float(snapshot["liquidation_price"])
                    future_mark = float(future["liquidation_price"])
                    sign = 1 if snapshot["direction"] == "BUY" else -1
                    record[target] = sign * (future_mark - base_mark) / float(snapshot["pip_size"])
                    last_target_at = future_quote_at if last_target_at is None else max(last_target_at, future_quote_at)
                    break
            record["label_end_timestamp"] = last_target_at.isoformat() if last_target_at else None
            rows.append(record)
    return sorted(rows, key=lambda item: (item["timestamp"], item["position_id"]))


def _atomic_write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".exit-dataset-", delete=False) as stream:
        temp = Path(stream.name)
        stream.write(contents)
        stream.flush()
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def export_observed_exit_rows(rows: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    """Export features, future-only observed marks, and attribution separately."""
    from io import StringIO
    columns = list(AUDIT_COLUMNS) + list(EXIT_FEATURE_COLUMNS) + list(TARGET_COLUMNS)
    if len(columns) != len(set(columns)):
        raise ValueError("feature/target/audit manifest overlap")
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    serialized = buffer.getvalue()
    _atomic_write(output, serialized)
    metadata = {
        "dataset_version": "exit-observed-v1",
        "feature_version": EXIT_FEATURE_VERSION,
        "feature_columns": list(EXIT_FEATURE_COLUMNS),
        "target_columns": list(TARGET_COLUMNS),
        "audit_columns": list(AUDIT_COLUMNS),
        "rows": len(rows),
        "complete_horizons": {
            str(sec): sum(1 for row in rows if row[f"observed_mark_return_{sec}s_pips"] is not None)
            for sec in HORIZONS_SECONDS
        },
        "sampled_quotes_only": True,
        "broker_execution_counterfactuals": False,
        "exit_action_labels_available": False,
        "sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    }
    _atomic_write(output.with_suffix(output.suffix + ".metadata.json"),
                  json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default="sqlite:///./data/fx_forward_test.db")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=50_000)
    parser.add_argument("--max-lag-seconds", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.limit <= 1_000_000:
        raise ValueError("limit must be between 1 and 1,000,000")
    from fxbot.journal import StructuredJournal
    journal = StructuredJournal(args.database_url)
    try:
        events = [
            event.payload
            for event in journal.recent_events(limit=args.limit)
            if event.event_type == "exit_ai_observation"
        ]
    finally:
        journal.close()
    rows = build_observed_exit_rows(events, max_lag_seconds=args.max_lag_seconds)
    print(json.dumps(export_observed_exit_rows(rows, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
