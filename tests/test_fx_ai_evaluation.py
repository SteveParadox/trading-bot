from __future__ import annotations

from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from fxbot.ai_evaluation import AiEvaluationConfig, ai_value_report, build_ai_outcome_frame
from fxbot.journal import StructuredJournal


START = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


def _add_case(
    journal: StructuredJournal,
    *,
    index: int,
    decision: str,
    net_return_pips: float,
    wait_30s: float = 0.0,
    wait_1m: float = 0.0,
    wait_3m: float = 0.0,
    executed: bool = False,
    final_net_pnl: float | None = None,
) -> str:
    candidate_id = f"fxsig-ai-eval-{index}"
    timestamp = START + timedelta(minutes=index)
    execution_cost_pips = 1.0
    candidate, _ = journal.record_candidate(
        candidate_id=candidate_id,
        timestamp=timestamp,
        symbol="EUR_USD",
        direction="LONG",
        entry=1.1002,
        stop_loss=1.0992,
        take_profit=1.1020,
        spread=0.0002,
        strategy_signal="signal_confirmed",
        executed=executed,
        payload={
            "execution_cost_pips_round_trip": execution_cost_pips,
            "market_snapshot": {"version": "v1"},
        },
    )
    journal.ensure_candidate_outcome(
        candidate_id=candidate_id,
        started_at=timestamp,
        payload={
            "label_version": "v2",
            "pip_size": 0.0001,
            "first_touch_reliable": True,
        },
    )
    journal.update_candidate_outcome(
        candidate_id,
        values={
            "status": "complete",
            "completed_at": timestamp + timedelta(minutes=30),
            "data_quality": "good",
            "tp_before_sl": net_return_pips > 0,
            "mfe_pips": max(0.0, net_return_pips + 4.0),
            "mae_pips": max(0.0, -net_return_pips + 1.0),
            "return_5m_pips": net_return_pips + execution_cost_pips,
            "return_15m_pips": net_return_pips + execution_cost_pips,
            "return_30m_pips": net_return_pips + execution_cost_pips,
            "wait_30s_improvement_pips": wait_30s,
            "wait_1m_improvement_pips": wait_1m,
            "wait_3m_improvement_pips": wait_3m,
            "time_to_tp_seconds": 600.0 if net_return_pips > 0 else None,
            "final_net_pnl": final_net_pnl,
            "final_net_pnl_currency": "USD" if final_net_pnl is not None else None,
        },
    )
    response = {
        "decision": decision,
        "confidence": 0.9,
        "reason_codes": {
            "TAKE": ["trend_alignment"],
            "WAIT": ["pullback_risk"],
            "SKIP": ["weak_tp_probability"],
        }[decision],
    }
    journal.record_ai_deliberation(
        payload={
            "signal_id": candidate_id,
            "timestamp": timestamp,
            "instrument": "EUR_USD",
            "side": "LONG",
            "model": "test-llm",
            "prompt_version": "v3",
            "mode": "shadow",
            "status": "completed",
            "decision": decision,
            "confidence": 0.9,
            "reasoning_audit_status": "STRUCTURED_V2",
            "reasoning_issues": [],
            "reasoning_supporting_factors": response["reason_codes"],
            "market_context_status": "EMBEDDED_IN_EVIDENCE",
            "market_context_issues": [],
            "market_context_supporting_factors": [],
            "contradictions": [],
            "recommended_action": decision,
            "summary": ",".join(response["reason_codes"]),
            "evidence_hash": f"evidence-{index}",
            "output_hash": f"output-{index}",
            "latency_ms": 2,
            "failure_reason": None,
            "evidence": {"candidate_id": candidate_id},
            "response": response,
        }
    )
    return candidate_id


def test_ai_outcome_attribution_tracks_requested_decision_result_buckets(tmp_path: Path) -> None:
    with closing(StructuredJournal(f"sqlite:///{tmp_path / 'journal.db'}")) as journal:
        _add_case(journal, index=1, decision="TAKE", net_return_pips=5.0, executed=True, final_net_pnl=4.0)
        _add_case(journal, index=2, decision="TAKE", net_return_pips=-1.0, executed=True, final_net_pnl=-1.0)
        _add_case(journal, index=3, decision="SKIP", net_return_pips=-4.0)
        _add_case(journal, index=4, decision="SKIP", net_return_pips=1.0)
        _add_case(journal, index=5, decision="WAIT", net_return_pips=2.0, wait_1m=2.5)
        _add_case(journal, index=6, decision="WAIT", net_return_pips=1.0, wait_30s=0.1, wait_1m=0.2, wait_3m=0.3)
        _add_case(journal, index=7, decision="WAIT", net_return_pips=-2.0)

        frame = build_ai_outcome_frame(
            journal,
            config=AiEvaluationConfig(min_wait_improvement_pips=1.0),
        )
        assert frame["result_classification"].tolist() == [
            "AI_TAKE_PROFITABLE",
            "AI_TAKE_LOSS",
            "AI_SKIP_WOULD_HAVE_LOST",
            "AI_SKIP_WOULD_HAVE_WON",
            "AI_WAIT_IMPROVED_ENTRY",
            "AI_WAIT_MISSED_TRADE",
            "AI_WAIT_AVOIDED_LOSS",
        ]

        report = ai_value_report(
            journal,
            config=AiEvaluationConfig(min_wait_improvement_pips=1.0),
        )
        outcomes = report["decision_outcomes"]
        assert outcomes["AI_TAKE_PROFITABLE"] == 1
        assert outcomes["AI_TAKE_LOSS"] == 1
        assert outcomes["AI_SKIP_WOULD_HAVE_LOST"] == 1
        assert outcomes["AI_SKIP_WOULD_HAVE_WON"] == 1
        assert outcomes["AI_WAIT_IMPROVED_ENTRY"] == 1
        assert outcomes["AI_WAIT_MISSED_TRADE"] == 1
        assert outcomes["AI_WAIT_AVOIDED_LOSS"] == 1


def test_ai_value_report_compares_baseline_and_take_filter_after_costs(tmp_path: Path) -> None:
    with closing(StructuredJournal(f"sqlite:///{tmp_path / 'journal.db'}")) as journal:
        _add_case(journal, index=1, decision="TAKE", net_return_pips=5.0, executed=True, final_net_pnl=4.0)
        _add_case(journal, index=2, decision="TAKE", net_return_pips=-1.0, executed=True, final_net_pnl=-1.0)
        _add_case(journal, index=3, decision="SKIP", net_return_pips=-4.0)
        _add_case(journal, index=4, decision="SKIP", net_return_pips=1.0)
        _add_case(journal, index=5, decision="WAIT", net_return_pips=2.0, wait_1m=2.0)

        report = ai_value_report(journal)
        baseline = report["baseline"]
        ai = report["ai_take_filter"]

        assert baseline["trade_count"] == 5
        assert baseline["expectancy_pips"] == pytest.approx(0.6)
        assert baseline["net_profit_after_costs_pips"] == pytest.approx(3.0)
        assert ai["trade_count"] == 2
        assert ai["trade_frequency_vs_baseline"] == pytest.approx(0.4)
        assert ai["expectancy_pips"] == pytest.approx(2.0)
        assert ai["win_rate"] == pytest.approx(0.5)
        assert ai["average_win_pips"] == pytest.approx(5.0)
        assert ai["average_loss_pips"] == pytest.approx(-1.0)
        assert ai["profit_factor"] == pytest.approx(5.0)
        assert report["value_add"]["expectancy_pips_delta"] == pytest.approx(1.4)
        assert report["realized_shadow_execution"]["total_net_pnl"] == pytest.approx(3.0)
        assert report["realized_shadow_execution"]["note"].startswith("realized baseline")
        assert any("counterfactual" in item.lower() for item in report["limitations"])
