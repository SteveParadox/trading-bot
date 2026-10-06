"""Evaluate shadow AI decisions against forward-observed candidate outcomes.

This module is research-only. Shadow AI has zero execution authority, so the
AI comparison is a counterfactual candidate filter, not a realized portfolio
backtest. Returns use executable bid/ask outcomes and subtract configured
non-spread commission/slippage allowances when available.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import pandas as pd
from sqlalchemy import select

from fxbot.ai_deliberation import AiResponseValidationError, validate_ai_audit_response
from fxbot.database import AiDeliberationRow, CandidateOutcomeRow, TradeCandidateRow
from fxbot.journal import StructuredJournal
from fxbot.training_dataset import candidate_execution_cost_pips


AI_EVALUATION_VERSION = "v1"


@dataclass(frozen=True)
class AiEvaluationConfig:
    min_wait_improvement_pips: float = 1.0
    include_degraded: bool = False
    max_candidates: int = 10000

    def __post_init__(self) -> None:
        if self.min_wait_improvement_pips < 0:
            raise ValueError("min_wait_improvement_pips cannot be negative")
        if not 1 <= self.max_candidates <= 100000:
            raise ValueError("max_candidates must be between 1 and 100000")


def build_ai_outcome_frame(
    journal: StructuredJournal,
    *,
    config: AiEvaluationConfig | None = None,
) -> pd.DataFrame:
    """Join AI decisions to future-only candidate outcomes."""

    cfg = config or AiEvaluationConfig()
    with journal.sessions() as session:
        query = (
            select(TradeCandidateRow, CandidateOutcomeRow, AiDeliberationRow)
            .join(
                CandidateOutcomeRow,
                CandidateOutcomeRow.candidate_id == TradeCandidateRow.candidate_id,
            )
            .join(
                AiDeliberationRow,
                AiDeliberationRow.signal_id == TradeCandidateRow.candidate_id,
            )
            .where(CandidateOutcomeRow.status == "complete")
            .where(AiDeliberationRow.mode == "shadow")
            .order_by(TradeCandidateRow.timestamp.desc(), TradeCandidateRow.candidate_id)
            .limit(cfg.max_candidates)
        )
        if not cfg.include_degraded:
            query = query.where(CandidateOutcomeRow.data_quality == "good")
        joined = list(session.execute(query).all())

    rows: list[dict[str, Any]] = []
    for candidate, outcome, ai in joined:
        decision = _decision_from_row(ai)
        if decision is None:
            continue
        if outcome.return_30m_pips is None:
            continue
        cost = candidate_execution_cost_pips(candidate, outcome)
        if cost is None or not math.isfinite(float(cost)):
            continue
        if not math.isfinite(float(outcome.return_30m_pips)):
            continue
        net_return = float(outcome.return_30m_pips) - float(cost)
        wait_action, wait_improvement = _best_wait(outcome)
        wait_improved = (
            wait_action is not None
            and wait_improvement is not None
            and wait_improvement >= cfg.min_wait_improvement_pips
        )
        result_class = _result_classification(
            decision=decision,
            net_return=net_return,
            wait_improved=wait_improved,
        )
        rows.append({
            "evaluation_version": AI_EVALUATION_VERSION,
            "candidate_id": candidate.candidate_id,
            "timestamp": _iso(candidate.timestamp),
            "symbol": candidate.symbol,
            "direction": candidate.direction,
            "ai_mode": ai.mode,
            "ai_model": ai.model,
            "prompt_version": ai.prompt_version,
            "ai_decision": decision,
            "ai_confidence": ai.confidence,
            "reason_codes": _reason_codes(ai),
            "executed": bool(candidate.executed),
            "outcome_data_quality": outcome.data_quality,
            "return_30m_pips": float(outcome.return_30m_pips),
            "execution_cost_pips": float(cost),
            "net_return_after_costs_pips": net_return,
            "profitable_after_costs": net_return > 0.0,
            "tp_before_sl": outcome.tp_before_sl,
            "mfe_pips": float(outcome.mfe_pips or 0.0),
            "mae_pips": float(outcome.mae_pips or 0.0),
            "best_wait_action": wait_action,
            "best_wait_improvement_pips": wait_improvement,
            "wait_improved_entry": bool(wait_improved),
            "result_classification": result_class,
            "final_net_pnl": outcome.final_net_pnl,
            "final_net_pnl_currency": outcome.final_net_pnl_currency,
        })

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["timestamp", "candidate_id"], kind="stable").reset_index(drop=True)


def ai_value_report(
    journal: StructuredJournal,
    *,
    config: AiEvaluationConfig | None = None,
) -> dict[str, Any]:
    """Compare deterministic baseline candidates with shadow AI TAKE filtering."""

    cfg = config or AiEvaluationConfig()
    frame = build_ai_outcome_frame(journal, config=cfg)
    if frame.empty:
        return {
            "version": AI_EVALUATION_VERSION,
            "sample_size": 0,
            "decision_outcomes": _empty_decision_outcomes(),
            "baseline": _candidate_metrics([]),
            "ai_take_filter": _candidate_metrics([]),
            "value_add": {},
            "limitations": _limitations(),
        }

    returns = frame["net_return_after_costs_pips"].astype(float)
    take = frame.loc[frame["ai_decision"] == "TAKE"]
    baseline_metrics = _candidate_metrics(returns.tolist())
    ai_metrics = _candidate_metrics(
        take["net_return_after_costs_pips"].astype(float).tolist(),
        baseline_count=len(frame),
    )

    counts = frame["result_classification"].value_counts().to_dict()
    decision_outcomes = {
        **_empty_decision_outcomes(),
        **{str(key): int(value) for key, value in counts.items()},
    }
    decision_outcomes["AI_WAIT_TOTAL"] = int((frame["ai_decision"] == "WAIT").sum())
    decision_outcomes["AI_SKIP_TOTAL"] = int((frame["ai_decision"] == "SKIP").sum())
    decision_outcomes["AI_TAKE_TOTAL"] = int((frame["ai_decision"] == "TAKE").sum())

    return {
        "version": AI_EVALUATION_VERSION,
        "sample_size": int(len(frame)),
        "sample_window": "most recent completed shadow candidates with valid decisions and known costs",
        "query_limit": cfg.max_candidates,
        "config": asdict(cfg),
        "decision_outcomes": decision_outcomes,
        "baseline": baseline_metrics,
        "ai_take_filter": ai_metrics,
        "value_add": _metric_deltas(baseline_metrics, ai_metrics),
        "realized_shadow_execution": _realized_shadow_metrics(frame),
        "cost_basis": (
            "return_30m_pips already uses executable liquidation bid/ask, so spread is included; "
            "configured non-spread execution cost pips are subtracted separately"
        ),
        "limitations": _limitations(),
    }


def export_ai_value_report(
    journal: StructuredJournal,
    output: Path,
    *,
    config: AiEvaluationConfig | None = None,
    rows_output: Path | None = None,
) -> tuple[Path, Path | None]:
    report = ai_value_report(journal, config=config)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    row_path: Path | None = None
    if rows_output is not None:
        rows = build_ai_outcome_frame(journal, config=config)
        rows_output.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(rows_output, index=False)
        row_path = rows_output
    return output, row_path


def _decision_from_row(row: AiDeliberationRow) -> str | None:
    payload = row.response or {}
    try:
        if payload:
            return validate_ai_audit_response(
                payload,
                allow_legacy_stored_response=True,
            ).decision
    except AiResponseValidationError:
        return None
    decision = str(row.decision or "").upper()
    return decision if decision in {"TAKE", "WAIT", "SKIP"} else None


def _reason_codes(row: AiDeliberationRow) -> list[str]:
    payload = row.response or {}
    codes = payload.get("reason_codes") if isinstance(payload, dict) else None
    if isinstance(codes, list):
        return [str(code) for code in codes]
    return []


def _best_wait(outcome: CandidateOutcomeRow) -> tuple[str | None, float | None]:
    options = [
        ("WAIT_30S", 30.0, outcome.wait_30s_improvement_pips),
        ("WAIT_1M", 60.0, outcome.wait_1m_improvement_pips),
        ("WAIT_3M", 180.0, outcome.wait_3m_improvement_pips),
    ]
    tp_time = _finite(outcome.time_to_tp_seconds)
    valid: list[tuple[str, float]] = []
    for action, delay, raw in options:
        value = _finite(raw)
        if value is None:
            continue
        if tp_time is not None and tp_time <= delay:
            continue
        valid.append((action, value))
    if not valid:
        return None, None
    return max(valid, key=lambda item: item[1])


def _result_classification(
    *,
    decision: str,
    net_return: float,
    wait_improved: bool,
) -> str:
    if decision == "TAKE":
        if net_return > 0:
            return "AI_TAKE_PROFITABLE"
        if net_return < 0:
            return "AI_TAKE_LOSS"
        return "AI_TAKE_FLAT"
    if decision == "SKIP":
        if net_return > 0:
            return "AI_SKIP_WOULD_HAVE_WON"
        if net_return < 0:
            return "AI_SKIP_WOULD_HAVE_LOST"
        return "AI_SKIP_FLAT"
    if decision == "WAIT":
        if wait_improved:
            return "AI_WAIT_IMPROVED_ENTRY"
        if net_return > 0:
            return "AI_WAIT_MISSED_TRADE"
        if net_return < 0:
            return "AI_WAIT_AVOIDED_LOSS"
        return "AI_WAIT_FLAT"
    raise ValueError(f"unsupported AI decision {decision!r}")


def _candidate_metrics(
    returns: list[float],
    *,
    baseline_count: int | None = None,
) -> dict[str, Any]:
    values = [float(value) for value in returns if math.isfinite(float(value))]
    wins = [value for value in values if value > 0]
    losses = [value for value in values if value < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    count = len(values)
    comparison_count = baseline_count if baseline_count is not None else count
    return {
        "trade_count": count,
        "trade_frequency_vs_baseline": (
            count / comparison_count if comparison_count else 0.0
        ),
        "expectancy_pips": mean(values) if values else 0.0,
        "win_rate": len(wins) / count if count else 0.0,
        "average_win_pips": mean(wins) if wins else 0.0,
        "average_loss_pips": mean(losses) if losses else 0.0,
        "profit_factor": (
            gross_profit / gross_loss
            if gross_loss > 0
            else None
        ),
        "net_profit_after_costs_pips": sum(values),
        "candidate_max_drawdown_pips_proxy": _max_drawdown(values),
        "candidate_sharpe_proxy": _sharpe(values),
        "gross_profit_pips": gross_profit,
        "gross_loss_pips": gross_loss,
    }


def _metric_deltas(
    baseline: dict[str, Any],
    ai: dict[str, Any],
) -> dict[str, Any]:
    keys = (
        "expectancy_pips",
        "win_rate",
        "average_win_pips",
        "average_loss_pips",
        "net_profit_after_costs_pips",
        "candidate_max_drawdown_pips_proxy",
        "candidate_sharpe_proxy",
        "trade_frequency_vs_baseline",
    )
    result: dict[str, Any] = {}
    for key in keys:
        result[f"{key}_delta"] = float(ai.get(key) or 0.0) - float(baseline.get(key) or 0.0)
    base_pf = baseline.get("profit_factor")
    ai_pf = ai.get("profit_factor")
    result["profit_factor_delta"] = (
        float(ai_pf) - float(base_pf)
        if ai_pf is not None and base_pf is not None
        else None
    )
    return result


def _realized_shadow_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    executed = frame.loc[
        frame["executed"].astype(bool)
        & frame["final_net_pnl"].notna()
    ]
    values = [float(value) for value in executed["final_net_pnl"].tolist()]
    currencies = sorted({
        str(value)
        for value in executed["final_net_pnl_currency"].dropna().tolist()
    })
    if len(currencies) > 1 or executed["final_net_pnl_currency"].isna().any():
        return {
            "trade_count": len(values),
            "comparable": False,
            "reason": "multiple or unknown final_net_pnl currencies",
            "currencies": currencies,
        }
    metrics = _money_metrics(values)
    return {
        **metrics,
        "comparable": True,
        "currency": currencies[0] if currencies else None,
        "note": "realized baseline execution only; shadow AI did not control these trades",
    }


def _money_metrics(values: list[float]) -> dict[str, Any]:
    wins = [value for value in values if value > 0]
    losses = [value for value in values if value < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    return {
        "trade_count": len(values),
        "total_net_pnl": sum(values),
        "expectancy": mean(values) if values else 0.0,
        "win_rate": len(wins) / len(values) if values else 0.0,
        "average_win": mean(wins) if wins else 0.0,
        "average_loss": mean(losses) if losses else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else None,
    }


def _max_drawdown(returns: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    maximum = 0.0
    for value in returns:
        equity += value
        peak = max(peak, equity)
        maximum = max(maximum, peak - equity)
    return maximum


def _sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    volatility = pstdev(returns)
    if volatility <= 0:
        return 0.0
    return mean(returns) / volatility * math.sqrt(len(returns))


def _empty_decision_outcomes() -> dict[str, int]:
    return {
        "AI_TAKE_PROFITABLE": 0,
        "AI_TAKE_LOSS": 0,
        "AI_TAKE_FLAT": 0,
        "AI_SKIP_WOULD_HAVE_LOST": 0,
        "AI_SKIP_WOULD_HAVE_WON": 0,
        "AI_SKIP_FLAT": 0,
        "AI_WAIT_IMPROVED_ENTRY": 0,
        "AI_WAIT_MISSED_TRADE": 0,
        "AI_WAIT_AVOIDED_LOSS": 0,
        "AI_WAIT_FLAT": 0,
        "AI_TAKE_TOTAL": 0,
        "AI_WAIT_TOTAL": 0,
        "AI_SKIP_TOTAL": 0,
    }


def _limitations() -> list[str]:
    return [
        "Shadow AI never controlled execution, so AI metrics are counterfactual candidate-filter metrics.",
        "Candidate observations can overlap in time; candidate drawdown and Sharpe are sequence proxies, not portfolio statistics.",
        "WAIT currently has no live duration; improved-entry attribution uses the best observed 30s/1m/3m executable entry only.",
        "Realized MT5 P&L is reported separately and cannot be assigned counterfactually to skipped/unexecuted candidates.",
    ]


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _iso(value: datetime) -> str:
    return value.isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path)
    parser.add_argument("--include-degraded", action="store_true")
    parser.add_argument("--min-wait-improvement-pips", type=float, default=1.0)
    args = parser.parse_args()

    journal = StructuredJournal(args.database_url)
    try:
        report_path, rows_path = export_ai_value_report(
            journal,
            args.output,
            rows_output=args.rows_output,
            config=AiEvaluationConfig(
                min_wait_improvement_pips=args.min_wait_improvement_pips,
                include_degraded=args.include_degraded,
            ),
        )
        print(json.dumps({
            "report": str(report_path),
            "rows": str(rows_path) if rows_path else None,
        }, sort_keys=True))
    finally:
        journal.close()


if __name__ == "__main__":
    main()
