"""Chronological dataset splitting for FX candidate model research.

The split is intentionally date-driven and never shuffles rows.

Default policy:
- TRAIN:       2023-01-01 <= timestamp < 2025-01-01
- VALIDATION:  2025-01-01 <= timestamp < 2026-01-01
- TEST:        2026-01-01 <= timestamp < 2026-07-01
- FORWARD:     timestamp >= 2026-07-01

These defaults make "early 2026" explicit as January through June 2026.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


SPLIT_VERSION = "v1"


@dataclass(frozen=True)
class ChronologicalSplitConfig:
    train_start: str = "2023-01-01"
    validation_start: str = "2025-01-01"
    test_start: str = "2026-01-01"
    forward_start: str = "2026-07-01"

    def validate(self) -> None:
        points = [
            pd.Timestamp(self.train_start, tz="UTC"),
            pd.Timestamp(self.validation_start, tz="UTC"),
            pd.Timestamp(self.test_start, tz="UTC"),
            pd.Timestamp(self.forward_start, tz="UTC"),
        ]
        if points != sorted(points) or len(set(points)) != len(points):
            raise ValueError("chronological split boundaries must be strictly increasing")


@dataclass(frozen=True)
class SplitFrames:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    forward: pd.DataFrame


def chronological_split(
    frame: pd.DataFrame,
    *,
    config: ChronologicalSplitConfig | None = None,
) -> SplitFrames:
    """Split a training dataset by timestamp without randomization."""

    cfg = config or ChronologicalSplitConfig()
    cfg.validate()
    required = {"timestamp", "candidate_id"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"dataset is missing required split columns: {missing}")

    working = frame.copy()
    timestamps = pd.to_datetime(working["timestamp"], utc=True, errors="raise")
    working = working.assign(_split_timestamp=timestamps)
    working = working.sort_values(["_split_timestamp", "candidate_id"], kind="stable").reset_index(drop=True)

    train_start = pd.Timestamp(cfg.train_start, tz="UTC")
    validation_start = pd.Timestamp(cfg.validation_start, tz="UTC")
    test_start = pd.Timestamp(cfg.test_start, tz="UTC")
    forward_start = pd.Timestamp(cfg.forward_start, tz="UTC")

    train = _slice(working, train_start, validation_start)
    validation = _slice(working, validation_start, test_start)
    test = _slice(working, test_start, forward_start)
    forward = working.loc[working["_split_timestamp"] >= forward_start].copy()

    result = SplitFrames(
        train=_finalize(train),
        validation=_finalize(validation),
        test=_finalize(test),
        forward=_finalize(forward),
    )
    _validate_split_integrity(result, cfg)
    return result


def export_chronological_splits(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    config: ChronologicalSplitConfig | None = None,
) -> tuple[SplitFrames, Path]:
    """Export split CSVs and a reproducibility manifest."""

    cfg = config or ChronologicalSplitConfig()
    splits = chronological_split(frame, config=cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, dict[str, Any]] = {}
    for name, split_frame in (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
        ("forward", splits.forward),
    ):
        path = output_dir / f"{name}.csv"
        split_frame.to_csv(path, index=False)
        files[name] = {
            "path": str(path),
            "rows": int(len(split_frame)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "start": _boundary_value(split_frame, first=True),
            "end": _boundary_value(split_frame, first=False),
        }

    manifest_path = output_dir / "split_manifest.json"
    manifest = {
        "split_version": SPLIT_VERSION,
        "config": asdict(cfg),
        "chronological": True,
        "random_shuffle": False,
        "interval_convention": "[start, end)",
        "files": files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return splits, manifest_path


def _slice(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return frame.loc[
        (frame["_split_timestamp"] >= start)
        & (frame["_split_timestamp"] < end)
    ].copy()


def _finalize(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=["_split_timestamp"]).reset_index(drop=True)


def _validate_split_integrity(splits: SplitFrames, cfg: ChronologicalSplitConfig) -> None:
    seen: set[str] = set()
    boundaries = {
        "train": (pd.Timestamp(cfg.train_start, tz="UTC"), pd.Timestamp(cfg.validation_start, tz="UTC")),
        "validation": (pd.Timestamp(cfg.validation_start, tz="UTC"), pd.Timestamp(cfg.test_start, tz="UTC")),
        "test": (pd.Timestamp(cfg.test_start, tz="UTC"), pd.Timestamp(cfg.forward_start, tz="UTC")),
        "forward": (pd.Timestamp(cfg.forward_start, tz="UTC"), None),
    }
    for name, split_frame in (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
        ("forward", splits.forward),
    ):
        if split_frame.empty:
            continue
        timestamps = pd.to_datetime(split_frame["timestamp"], utc=True)
        if not timestamps.is_monotonic_increasing:
            raise ValueError(f"{name} split is not chronological")
        start, end = boundaries[name]
        if (timestamps < start).any() or (end is not None and (timestamps >= end).any()):
            raise ValueError(f"{name} split contains rows outside configured date range")
        ids = set(split_frame["candidate_id"].astype(str))
        overlap = seen.intersection(ids)
        if overlap:
            raise ValueError(f"candidate IDs overlap across chronological splits: {sorted(overlap)[:3]}")
        seen.update(ids)


def _boundary_value(frame: pd.DataFrame, *, first: bool) -> str | None:
    if frame.empty:
        return None
    value = frame.iloc[0 if first else -1]["timestamp"]
    return pd.Timestamp(value).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-start", default="2023-01-01")
    parser.add_argument("--validation-start", default="2025-01-01")
    parser.add_argument("--test-start", default="2026-01-01")
    parser.add_argument("--forward-start", default="2026-07-01")
    args = parser.parse_args()

    frame = pd.read_csv(args.dataset)
    config = ChronologicalSplitConfig(
        train_start=args.train_start,
        validation_start=args.validation_start,
        test_start=args.test_start,
        forward_start=args.forward_start,
    )
    _, manifest = export_chronological_splits(
        frame,
        args.output_dir,
        config=config,
    )
    print(json.dumps({"manifest": str(manifest)}, sort_keys=True))


if __name__ == "__main__":
    main()
