"""Build leakage-safe tabular training data from candidate snapshots and outcomes.

Features come exclusively from the candidate's first-observation market snapshot.
Targets come exclusively from the later candidate_outcomes row. The module does
not shuffle rows or call MT5, and its action label is research-only.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select

from fxbot.database import CandidateOutcomeRow, TradeCandidateRow
from fxbot.instruments import FxInstrument
from fxbot.journal import StructuredJournal


TRAINING_DATASET_VERSION = "v1"
ACTION_LABEL_VERSION = "v1"

IDENTIFIER_COLUMNS = [
    "dataset_version", "candidate_id", "timestamp", "feature_version",
    "label_version", "strategy_hash", "code_version",
]

FEATURE_COLUMNS = [
    "symbol", "direction", "strategy_signal", "strategy_score",
    "hour_utc", "day_of_week", "session",
    "bid", "ask", "spread_pips", "spread_relative_to_atr",
    "atr_price", "atr_pips", "rsi", "momentum", "trend_strength", "volatility",
    "support_distance_pips", "resistance_distance_pips", "risk_reward",
    "proposed_entry", "stop_loss", "take_profit",
    "open_positions", "portfolio_risk", "gross_exposure", "pair_exposure",
    "free_margin", "account_currency",
    "news_risk", "news_risk_score", "upcoming_news_currency",
    "upcoming_news_impact", "minutes_until_news", "recent_news_currency",
    "minutes_since_news", "news_event_just_occurred",
    "news_freshness_state", "news_stale", "news_age_seconds",
]

TARGET_COLUMNS = [
    "TP_BEFORE_SL", "PROFITABLE_WITHIN_5_MIN", "PROFITABLE_WITHIN_15_MIN",
    "EXPECTED_MFE", "EXPECTED_MAE", "EXPECTED_RETURN",
    "ENTRY_NOW", "WAIT", "SKIP", "ACTION_LABEL",
]

AUXILIARY_OUTCOME_COLUMNS = [
    "tp_hit", "sl_hit", "return_1m_pips", "return_5m_pips",
    "return_15m_pips", "return_30m_pips", "mfe_pips", "mae_pips",
    "time_to_profit_seconds", "time_to_loss_seconds",
    "time_to_tp_seconds", "time_to_sl_seconds",
    "final_net_pnl", "final_net_pnl_currency",
    "outcome_data_quality", "max_observation_gap_seconds",
]

AUDIT_COLUMNS = ["audit_executed", "audit_rejection_reason"]


@dataclass(frozen=True)
class ActionLabelConfig:
    """Conservative v1 research label for ENTRY_NOW / WAIT / SKIP.

    This label is not an execution policy. It is a supervised-learning target
    derived from future outcomes and must never be computed in the live path.
    """

    min_expected_return_pips: float = 0.0
    min_wait_improvement_pips: float = 1.0
    require_mfe_gt_mae: bool = True


def build_training_dataset(
    journal: StructuredJournal,
    *,
    include_degraded: bool = False,
    action_config: ActionLabelConfig | None = None,
) -> pd.DataFrame:
    """Join immutable candidate-time features to future-only outcome labels."""

    config = action_config or ActionLabelConfig()
    rows: list[dict[str, Any]] = []
    with journal.sessions() as session:
        query = (
            select(TradeCandidateRow, CandidateOutcomeRow)
            .join(
                CandidateOutcomeRow,
                CandidateOutcomeRow.candidate_id == TradeCandidateRow.candidate_id,
            )
            .where(CandidateOutcomeRow.status == "complete")
            .order_by(TradeCandidateRow.timestamp.asc())
        )
        if not include_degraded:
            query = query.where(CandidateOutcomeRow.data_quality == "good")
        pairs = list(session.execute(query).all())

    for candidate, outcome in pairs:
        feature_row = _feature_row(candidate, outcome)
        target_row = _target_row(outcome, config=config)
        if target_row is None:
            continue
        rows.append({**feature_row, **target_row})

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values(["timestamp", "candidate_id"], kind="stable").reset_index(drop=True)
    if frame["candidate_id"].duplicated().any():
        raise ValueError("training dataset contains duplicate candidate_id values")
    missing_features = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    missing_targets = [column for column in TARGET_COLUMNS if column not in frame.columns]
    if missing_features or missing_targets:
        raise ValueError(
            f"training dataset schema mismatch; missing_features={missing_features}, missing_targets={missing_targets}"
        )
    if set(FEATURE_COLUMNS).intersection(TARGET_COLUMNS + AUXILIARY_OUTCOME_COLUMNS + AUDIT_COLUMNS):
        raise ValueError("training feature manifest contains leakage-prone outcome/audit columns")
    if not pd.to_datetime(frame["timestamp"], utc=True).is_monotonic_increasing:
        raise ValueError("training dataset must remain chronological")
    return frame


def export_training_dataset(
    journal: StructuredJournal,
    output: Path,
    *,
    include_degraded: bool = False,
    action_config: ActionLabelConfig | None = None,
) -> tuple[Path, Path]:
    """Write CSV plus a versioned metadata sidecar with a content hash."""

    config = action_config or ActionLabelConfig()
    frame = build_training_dataset(
        journal,
        include_degraded=include_degraded,
        action_config=config,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    metadata = {
        "dataset_version": TRAINING_DATASET_VERSION,
        "action_label_version": ACTION_LABEL_VERSION,
        "action_label_config": asdict(config),
        "include_degraded": include_degraded,
        "rows": int(len(frame)),
        "sha256": digest,
        "chronological": True,
        "random_shuffle": False,
        "feature_source": "trade_candidates.first_observation.market_snapshot",
        "target_source": "candidate_outcomes.future_observations",
        "expected_return_definition": "30-minute executable liquidation return in pips",
        "final_net_pnl_definition": "executed trades only: MT5 realized_pl + financing in account currency",
        "identifier_columns": IDENTIFIER_COLUMNS,
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "auxiliary_outcome_columns": AUXILIARY_OUTCOME_COLUMNS,
        "audit_columns": AUDIT_COLUMNS,
        "leakage_guard": "audit and outcome columns are not model features",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return output, metadata_path


def _feature_row(candidate: TradeCandidateRow, outcome: CandidateOutcomeRow) -> dict[str, Any]:
    payload = candidate.payload or {}
    snapshot = payload.get("market_snapshot") or {}
    exposure = snapshot.get("current_exposure") or {}
    news = candidate.news_risk or {}
    upcoming = news.get("upcoming_event") or {}
    recent = news.get("recent_event") or {}
    freshness = news.get("freshness") or {}
    sessions = snapshot.get("session") or []

    pip_size = _finite_or_none((outcome.payload or {}).get("pip_size"))
    if pip_size is None or pip_size <= 0:
        pip_size = FxInstrument(candidate.symbol).pip_size
    atr_price = _first_finite(snapshot.get("atr"), candidate.atr)
    atr_pips = atr_price / pip_size if atr_price is not None and pip_size > 0 else None
    spread_pips = _first_finite(snapshot.get("spread_pips"))
    if spread_pips is None:
        spread_pips = float(candidate.spread) / pip_size
    risk_reward = _first_finite(snapshot.get("risk_reward"))
    if risk_reward is None:
        risk_reward = _risk_reward(candidate)
    timestamp = _utc(candidate.timestamp)
    news_risk = str(news.get("risk_level") or "UNKNOWN").upper()

    return {
        "dataset_version": TRAINING_DATASET_VERSION,
        "candidate_id": candidate.candidate_id,
        "timestamp": timestamp.isoformat(),
        "symbol": candidate.symbol,
        "direction": candidate.direction,
        "feature_version": str(snapshot.get("version") or "unknown"),
        "strategy_hash": candidate.strategy_hash,
        "code_version": candidate.code_version,
        "strategy_signal": candidate.strategy_signal,
        "strategy_score": _finite_or_none(payload.get("signal_score")),
        "hour_utc": timestamp.hour,
        "day_of_week": timestamp.weekday(),
        "session": "+".join(str(item) for item in sessions) if sessions else "none",
        "bid": _finite_or_none(snapshot.get("bid")),
        "ask": _finite_or_none(snapshot.get("ask")),
        "spread_pips": spread_pips,
        "spread_relative_to_atr": (
            float(candidate.spread) / atr_price
            if atr_price is not None and atr_price > 0
            else None
        ),
        "atr_price": atr_price,
        "atr_pips": atr_pips,
        "rsi": _finite_or_none(snapshot.get("rsi")),
        "momentum": _first_finite(snapshot.get("momentum"), candidate.momentum),
        "trend_strength": _first_finite(snapshot.get("trend_strength"), candidate.trend_strength),
        "volatility": _finite_or_none(snapshot.get("volatility")),
        "support_distance_pips": _finite_or_none(snapshot.get("support_distance_pips")),
        "resistance_distance_pips": _finite_or_none(snapshot.get("resistance_distance_pips")),
        "risk_reward": risk_reward,
        "proposed_entry": float(candidate.entry),
        "stop_loss": _finite_or_none(candidate.stop_loss),
        "take_profit": _finite_or_none(candidate.take_profit),
        "open_positions": _finite_or_none(exposure.get("open_positions")),
        "portfolio_risk": _finite_or_none(exposure.get("portfolio_risk")),
        "gross_exposure": _finite_or_none(exposure.get("gross_exposure")),
        "pair_exposure": _finite_or_none(exposure.get("pair_exposure")),
        "free_margin": _finite_or_none(exposure.get("free_margin")),
        "account_currency": exposure.get("account_currency"),
        "news_risk": news_risk,
        "news_risk_score": _news_risk_score(news_risk),
        "upcoming_news_currency": upcoming.get("currency"),
        "upcoming_news_impact": upcoming.get("impact_level"),
        "minutes_until_news": _finite_or_none(upcoming.get("minutes_until_event")),
        "recent_news_currency": recent.get("currency"),
        "minutes_since_news": _finite_or_none(recent.get("minutes_since_event")),
        "news_event_just_occurred": bool(news.get("event_just_occurred", False)),
        "news_freshness_state": freshness.get("state"),
        "news_stale": bool(freshness.get("stale", False)),
        "news_age_seconds": _finite_or_none(freshness.get("age_seconds")),
        "audit_executed": bool(candidate.executed),
        "audit_rejection_reason": candidate.rejection_reason,
    }


def _target_row(
    outcome: CandidateOutcomeRow,
    *,
    config: ActionLabelConfig,
) -> dict[str, Any] | None:
    required = (
        outcome.return_5m_pips,
        outcome.return_15m_pips,
        outcome.return_30m_pips,
    )
    if any(value is None or not math.isfinite(float(value)) for value in required):
        return None

    first_touch_reliable = bool((outcome.payload or {}).get("first_touch_reliable", True))
    tp_before_sl = outcome.tp_before_sl if first_touch_reliable else None
    expected_return = float(outcome.return_30m_pips)
    action = _action_label(outcome, config=config)

    return {
        "label_version": str((outcome.payload or {}).get("label_version") or "unknown"),
        "TP_BEFORE_SL": _bool_int(tp_before_sl),
        "PROFITABLE_WITHIN_5_MIN": int(float(outcome.return_5m_pips) > 0.0),
        "PROFITABLE_WITHIN_15_MIN": int(float(outcome.return_15m_pips) > 0.0),
        "EXPECTED_MFE": float(outcome.mfe_pips),
        "EXPECTED_MAE": float(outcome.mae_pips),
        "EXPECTED_RETURN": expected_return,
        "ENTRY_NOW": int(action == "ENTRY_NOW"),
        "WAIT": int(action == "WAIT"),
        "SKIP": int(action == "SKIP"),
        "ACTION_LABEL": action,
        "tp_hit": int(bool(outcome.tp_hit)),
        "sl_hit": int(bool(outcome.sl_hit)),
        "return_1m_pips": _finite_or_none(outcome.return_1m_pips),
        "return_5m_pips": float(outcome.return_5m_pips),
        "return_15m_pips": float(outcome.return_15m_pips),
        "return_30m_pips": expected_return,
        "mfe_pips": float(outcome.mfe_pips),
        "mae_pips": float(outcome.mae_pips),
        "time_to_profit_seconds": _finite_or_none(outcome.time_to_profit_seconds),
        "time_to_loss_seconds": _finite_or_none(outcome.time_to_loss_seconds),
        "time_to_tp_seconds": _finite_or_none(outcome.time_to_tp_seconds),
        "time_to_sl_seconds": _finite_or_none(outcome.time_to_sl_seconds),
        "final_net_pnl": _finite_or_none(outcome.final_net_pnl),
        "final_net_pnl_currency": outcome.final_net_pnl_currency,
        "outcome_data_quality": outcome.data_quality,
        "max_observation_gap_seconds": float(outcome.max_observation_gap_seconds or 0.0),
    }


def _action_label(outcome: CandidateOutcomeRow, *, config: ActionLabelConfig) -> str:
    expected_return = float(outcome.return_30m_pips)
    mfe = float(outcome.mfe_pips)
    mae = float(outcome.mae_pips)
    positive_setup = expected_return > config.min_expected_return_pips
    if config.require_mfe_gt_mae:
        positive_setup = positive_setup and mfe > mae
    if not positive_setup:
        return "SKIP"

    wait_values = [
        outcome.wait_30s_improvement_pips,
        outcome.wait_1m_improvement_pips,
        outcome.wait_3m_improvement_pips,
        outcome.wait_5m_improvement_pips,
    ]
    valid_waits = [float(value) for value in wait_values if value is not None and math.isfinite(float(value))]
    if valid_waits and max(valid_waits) >= config.min_wait_improvement_pips:
        return "WAIT"
    return "ENTRY_NOW"


def _risk_reward(candidate: TradeCandidateRow) -> float | None:
    entry = float(candidate.entry)
    stop = _finite_or_none(candidate.stop_loss)
    target = _finite_or_none(candidate.take_profit)
    if stop is None or target is None:
        return None
    risk = abs(entry - stop)
    reward = abs(target - entry)
    if risk <= 0:
        return None
    value = reward / risk
    return value if math.isfinite(value) else None


def _news_risk_score(level: str) -> int:
    return {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "UNKNOWN": -1}.get(level.upper(), -1)


def _bool_int(value: bool | None) -> int | None:
    if value is None:
        return None
    return int(bool(value))


def _first_finite(*values: Any) -> float | None:
    for value in values:
        parsed = _finite_or_none(value)
        if parsed is not None:
            return parsed
    return None


def _finite_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-degraded", action="store_true")
    parser.add_argument("--min-wait-improvement-pips", type=float, default=1.0)
    parser.add_argument("--min-expected-return-pips", type=float, default=0.0)
    args = parser.parse_args()

    config = ActionLabelConfig(
        min_expected_return_pips=args.min_expected_return_pips,
        min_wait_improvement_pips=args.min_wait_improvement_pips,
    )
    journal = StructuredJournal(args.database_url)
    try:
        output, metadata = export_training_dataset(
            journal,
            args.output,
            include_degraded=args.include_degraded,
            action_config=config,
        )
        print(json.dumps({"dataset": str(output), "metadata": str(metadata)}, sort_keys=True))
    finally:
        journal.close()


if __name__ == "__main__":
    main()
