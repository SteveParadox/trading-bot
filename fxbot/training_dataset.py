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


TRAINING_DATASET_VERSION = "v2"
ACTION_LABEL_VERSION = "v2"
ENTRY_ACTIONS = ("ENTER_NOW", "WAIT_30S", "WAIT_1M", "WAIT_3M", "SKIP")
ENTRY_DELAY_SECONDS = {
    "ENTER_NOW": 0,
    "WAIT_30S": 30,
    "WAIT_1M": 60,
    "WAIT_3M": 180,
    "SKIP": None,
}

IDENTIFIER_COLUMNS = [
    "dataset_version", "candidate_id", "timestamp", "feature_version",
    "label_version", "strategy_hash", "code_version",
]

FEATURE_COLUMNS = [
    "symbol", "direction", "strategy_signal", "strategy_score",
    "hour_utc", "day_of_week", "session",
    "bid", "ask", "spread_pips", "spread_relative_to_atr",
    "execution_cost_pips_round_trip",
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
    "IMMEDIATE_ADVERSE_MOVEMENT", "EXPECTED_PULLBACK",
    "BEST_ENTRY_DELAY_SECONDS", "CONTINUATION", "FAKE_BREAKOUT",
    "ENTER_NOW", "WAIT_30S", "WAIT_1M", "WAIT_3M", "SKIP",
    "ENTRY_ACTION_LABEL",
    # Compatibility targets retained for older notebooks/reports.
    "ENTRY_NOW", "WAIT", "ACTION_LABEL",
]

AUXILIARY_OUTCOME_COLUMNS = [
    "tp_hit", "sl_hit", "return_1m_pips", "return_5m_pips",
    "return_15m_pips", "return_30m_pips",
    "net_return_5m_pips", "net_return_15m_pips", "net_return_30m_pips",
    "mfe_pips", "mae_pips",
    "wait_30s_improvement_pips", "wait_1m_improvement_pips",
    "wait_3m_improvement_pips", "wait_5m_improvement_pips",
    "time_to_profit_seconds", "time_to_loss_seconds",
    "time_to_tp_seconds", "time_to_sl_seconds",
    "final_net_pnl", "final_net_pnl_currency",
    "outcome_data_quality", "max_observation_gap_seconds",
]

AUDIT_COLUMNS = ["audit_executed", "audit_rejection_reason"]


@dataclass(frozen=True)
class ActionLabelConfig:
    """Conservative v2 research labels for entry quality and timing.

    All values are targets derived from observations after candidate creation.
    They must never be computed in the live feature path.
    """

    min_expected_return_pips: float = 0.0
    min_wait_improvement_pips: float = 1.0
    immediate_adverse_window_seconds: float = 60.0
    immediate_adverse_min_pips: float = 1.0
    require_mfe_gt_mae: bool = True
    breakout_lookback: int = 3

    def __post_init__(self) -> None:
        if self.min_wait_improvement_pips < 0:
            raise ValueError("min_wait_improvement_pips cannot be negative")
        if self.immediate_adverse_window_seconds <= 0:
            raise ValueError("immediate_adverse_window_seconds must be positive")
        if self.immediate_adverse_min_pips < 0:
            raise ValueError("immediate_adverse_min_pips cannot be negative")
        if self.breakout_lookback < 2:
            raise ValueError("breakout_lookback must be at least 2")


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
        target_row = _target_row(
            candidate,
            outcome,
            execution_cost_pips=candidate_execution_cost_pips(candidate, outcome),
            config=config,
        )
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
        "entry_action_space": list(ENTRY_ACTIONS),
        "action_label_config": asdict(config),
        "include_degraded": include_degraded,
        "rows": int(len(frame)),
        "sha256": digest,
        "chronological": True,
        "random_shuffle": False,
        "feature_source": "trade_candidates.first_observation.market_snapshot",
        "target_source": "candidate_outcomes.future_observations",
        "expected_return_definition": "30-minute executable liquidation return in pips minus configured non-spread commission/slippage allowance",
        "profitability_label_definition": "executable liquidation return at horizon minus configured non-spread commission/slippage allowance > 0",
        "entry_quality_label_definitions": {
            "IMMEDIATE_ADVERSE_MOVEMENT": "1 when loss begins within configured immediate window and 30m MAE reaches the configured minimum adverse pips",
            "EXPECTED_PULLBACK": "maximum positive executable entry improvement observed at 30s, 1m, or 3m",
            "BEST_ENTRY_DELAY_SECONDS": "0/30/60/180 from ENTRY_ACTION_LABEL; null for SKIP",
            "CONTINUATION": "1 when cost-adjusted 15m return is positive, MFE exceeds MAE, and reliable TP-before-SL is not false",
            "FAKE_BREAKOUT": "nullable; defined only for snapshot breakouts. 1 when continuation fails and immediate adverse movement or non-positive 15m return follows",
            "ENTRY_ACTION_LABEL": "ENTER_NOW, WAIT_30S, WAIT_1M, WAIT_3M, or SKIP; waits are excluded if the original candidate hit TP before that delay",
        },
        "raw_return_columns": ["return_1m_pips", "return_5m_pips", "return_15m_pips", "return_30m_pips"],
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
    risk_payload = payload.get("risk") or {}
    risk_metadata = risk_payload.get("metadata") or {}
    execution_cost_price = _finite_or_none(risk_metadata.get("execution_cost_price"))
    execution_cost_pips = (
        execution_cost_price / pip_size
        if execution_cost_price is not None and pip_size > 0
        else _finite_or_none(payload.get("execution_cost_pips_round_trip"))
    )

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
        "execution_cost_pips_round_trip": execution_cost_pips,
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
    candidate: TradeCandidateRow,
    outcome: CandidateOutcomeRow,
    *,
    execution_cost_pips: float | None,
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
    if execution_cost_pips is None or not math.isfinite(float(execution_cost_pips)):
        return None

    net_return_5m = float(outcome.return_5m_pips) - float(execution_cost_pips)
    net_return_15m = float(outcome.return_15m_pips) - float(execution_cost_pips)
    expected_return = float(outcome.return_30m_pips) - float(execution_cost_pips)
    immediate_adverse = _immediate_adverse_label(outcome, config=config)
    expected_pullback = _expected_pullback_pips(outcome)
    continuation = _continuation_label(
        outcome,
        net_return_15m=net_return_15m,
        tp_before_sl=tp_before_sl,
    )
    breakout_candidate = _snapshot_breakout_candidate(candidate, lookback=config.breakout_lookback)
    fake_breakout = (
        int(
            continuation == 0
            and (
                immediate_adverse == 1
                or net_return_15m <= 0.0
            )
        )
        if breakout_candidate
        else None
    )
    entry_action = _entry_action_label(
        outcome,
        expected_return=expected_return,
        config=config,
    )
    compatibility_action = (
        "SKIP"
        if entry_action == "SKIP"
        else "WAIT"
        if entry_action.startswith("WAIT_")
        else "ENTRY_NOW"
    )

    return {
        "label_version": str((outcome.payload or {}).get("label_version") or "unknown"),
        "TP_BEFORE_SL": _bool_int(tp_before_sl),
        "PROFITABLE_WITHIN_5_MIN": int(net_return_5m > 0.0),
        "PROFITABLE_WITHIN_15_MIN": int(net_return_15m > 0.0),
        "EXPECTED_MFE": float(outcome.mfe_pips),
        "EXPECTED_MAE": float(outcome.mae_pips),
        "EXPECTED_RETURN": expected_return,
        "IMMEDIATE_ADVERSE_MOVEMENT": immediate_adverse,
        "EXPECTED_PULLBACK": expected_pullback,
        "BEST_ENTRY_DELAY_SECONDS": ENTRY_DELAY_SECONDS[entry_action],
        "CONTINUATION": continuation,
        "FAKE_BREAKOUT": fake_breakout,
        "ENTER_NOW": int(entry_action == "ENTER_NOW"),
        "WAIT_30S": int(entry_action == "WAIT_30S"),
        "WAIT_1M": int(entry_action == "WAIT_1M"),
        "WAIT_3M": int(entry_action == "WAIT_3M"),
        "SKIP": int(entry_action == "SKIP"),
        "ENTRY_ACTION_LABEL": entry_action,
        # v1 compatibility.
        "ENTRY_NOW": int(compatibility_action == "ENTRY_NOW"),
        "WAIT": int(compatibility_action == "WAIT"),
        "ACTION_LABEL": compatibility_action,
        "tp_hit": int(bool(outcome.tp_hit)),
        "sl_hit": int(bool(outcome.sl_hit)),
        "return_1m_pips": _finite_or_none(outcome.return_1m_pips),
        "return_5m_pips": float(outcome.return_5m_pips),
        "return_15m_pips": float(outcome.return_15m_pips),
        "return_30m_pips": float(outcome.return_30m_pips),
        "net_return_5m_pips": net_return_5m,
        "net_return_15m_pips": net_return_15m,
        "net_return_30m_pips": expected_return,
        "mfe_pips": float(outcome.mfe_pips),
        "mae_pips": float(outcome.mae_pips),
        "wait_30s_improvement_pips": _finite_or_none(outcome.wait_30s_improvement_pips),
        "wait_1m_improvement_pips": _finite_or_none(outcome.wait_1m_improvement_pips),
        "wait_3m_improvement_pips": _finite_or_none(outcome.wait_3m_improvement_pips),
        "wait_5m_improvement_pips": _finite_or_none(outcome.wait_5m_improvement_pips),
        "time_to_profit_seconds": _finite_or_none(outcome.time_to_profit_seconds),
        "time_to_loss_seconds": _finite_or_none(outcome.time_to_loss_seconds),
        "time_to_tp_seconds": _finite_or_none(outcome.time_to_tp_seconds),
        "time_to_sl_seconds": _finite_or_none(outcome.time_to_sl_seconds),
        "final_net_pnl": _finite_or_none(outcome.final_net_pnl),
        "final_net_pnl_currency": outcome.final_net_pnl_currency,
        "outcome_data_quality": outcome.data_quality,
        "max_observation_gap_seconds": float(outcome.max_observation_gap_seconds or 0.0),
    }


def _entry_action_label(
    outcome: CandidateOutcomeRow,
    *,
    expected_return: float,
    config: ActionLabelConfig,
) -> str:
    mfe = float(outcome.mfe_pips)
    mae = float(outcome.mae_pips)
    positive_setup = expected_return > config.min_expected_return_pips
    if config.require_mfe_gt_mae:
        positive_setup = positive_setup and mfe > mae
    if not positive_setup:
        return "SKIP"

    candidates = [
        ("WAIT_30S", 30, outcome.wait_30s_improvement_pips),
        ("WAIT_1M", 60, outcome.wait_1m_improvement_pips),
        ("WAIT_3M", 180, outcome.wait_3m_improvement_pips),
    ]
    feasible: list[tuple[str, float]] = []
    tp_time = _finite_or_none(outcome.time_to_tp_seconds)
    for action, delay, value in candidates:
        improvement = _finite_or_none(value)
        if improvement is None:
            continue
        # Do not label a wait that begins after the original setup already
        # reached its target. That would teach the model to miss the trade.
        if tp_time is not None and tp_time <= delay:
            continue
        feasible.append((action, improvement))

    if not feasible:
        return "ENTER_NOW"
    best_action, best_improvement = max(feasible, key=lambda item: item[1])
    if best_improvement >= config.min_wait_improvement_pips:
        return best_action
    return "ENTER_NOW"


def _immediate_adverse_label(
    outcome: CandidateOutcomeRow,
    *,
    config: ActionLabelConfig,
) -> int:
    time_to_loss = _finite_or_none(outcome.time_to_loss_seconds)
    mae = _finite_or_none(outcome.mae_pips) or 0.0
    return int(
        time_to_loss is not None
        and time_to_loss <= config.immediate_adverse_window_seconds
        and mae >= config.immediate_adverse_min_pips
    )


def _expected_pullback_pips(outcome: CandidateOutcomeRow) -> float:
    values = [
        _finite_or_none(outcome.wait_30s_improvement_pips),
        _finite_or_none(outcome.wait_1m_improvement_pips),
        _finite_or_none(outcome.wait_3m_improvement_pips),
    ]
    valid = [value for value in values if value is not None]
    return max(0.0, max(valid)) if valid else 0.0


def _continuation_label(
    outcome: CandidateOutcomeRow,
    *,
    net_return_15m: float,
    tp_before_sl: bool | None,
) -> int:
    mfe = float(outcome.mfe_pips or 0.0)
    mae = float(outcome.mae_pips or 0.0)
    return int(
        net_return_15m > 0.0
        and mfe > mae
        and tp_before_sl is not False
    )


def _snapshot_breakout_candidate(candidate: TradeCandidateRow, *, lookback: int) -> bool:
    snapshot = (candidate.payload or {}).get("market_snapshot") or {}
    candles = snapshot.get("recent_candles") or []
    if len(candles) < lookback + 1:
        return False
    current = candles[-1]
    prior = candles[-1 - lookback:-1]
    try:
        close = float(current["close"])
        if candidate.direction == "LONG":
            return close > max(float(row["high"]) for row in prior)
        if candidate.direction == "SHORT":
            return close < min(float(row["low"]) for row in prior)
    except (KeyError, TypeError, ValueError):
        return False
    return False


def candidate_execution_cost_pips(
    candidate: TradeCandidateRow,
    outcome: CandidateOutcomeRow,
) -> float | None:
    payload = candidate.payload or {}
    risk_payload = payload.get("risk") or {}
    risk_metadata = risk_payload.get("metadata") or {}
    pip_size = _finite_or_none((outcome.payload or {}).get("pip_size"))
    if pip_size is None or pip_size <= 0:
        pip_size = FxInstrument(candidate.symbol).pip_size
    execution_cost_price = _finite_or_none(risk_metadata.get("execution_cost_price"))
    if execution_cost_price is not None and pip_size > 0:
        return execution_cost_price / pip_size
    return _finite_or_none(payload.get("execution_cost_pips_round_trip"))


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
    parser.add_argument("--immediate-adverse-window-seconds", type=float, default=60.0)
    parser.add_argument("--immediate-adverse-min-pips", type=float, default=1.0)
    args = parser.parse_args()

    config = ActionLabelConfig(
        min_expected_return_pips=args.min_expected_return_pips,
        min_wait_improvement_pips=args.min_wait_improvement_pips,
        immediate_adverse_window_seconds=args.immediate_adverse_window_seconds,
        immediate_adverse_min_pips=args.immediate_adverse_min_pips,
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
