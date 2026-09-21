from __future__ import annotations

import tempfile
import inspect
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fxbot.ai_deliberation import (
    AiDeliberationService,
    AiDeliberationResult,
    apply_ai_execution_policy,
    build_signal_evidence,
    deterministic_reasoning_audit,
    validate_ai_audit_response,
)
from fxbot.config import AiDeliberationSettings, NewsEvent, StrategySettings
from fxbot.instruments import FxInstrument, PriceSnapshot
from fxbot.journal import StructuredJournal
from fxbot.models import FxPortfolioState, FxSignalIntent, Side
from fxbot.risk import FxExitPlan, FxRiskDecision


NOW = datetime(2026, 9, 15, 14, 0, tzinfo=timezone.utc)


def valid_response() -> dict:
    return {
        "decision": "FLAG",
        "confidence": 0.9,
        "reasoning_audit": {"status": "CONSISTENT", "issues": [], "supporting_factors": ["ADX is 25"]},
        "market_context": {"status": "MATERIAL_CONTRADICTION", "issues": ["USD event in 12 minutes"], "supporting_factors": []},
        "contradictions": ["USD event in 12 minutes"],
        "recommended_action": "FLAG",
        "summary": "The supplied event is imminent.",
    }


def intent() -> FxSignalIntent:
    return FxSignalIntent(
        instrument="EUR_USD",
        side=Side.LONG,
        timestamp=NOW,
        entry_price=1.1010,
        score=70.0,
        signal_row={"close": 1.1008, "ma7": 1.1004, "ma28": 1.098, "adx": 25, "di_plus": 22, "di_minus": 9, "atr": 0.001},
        metadata={
            "decision": "signal_confirmed",
            "strategy_reasoning": "strong bullish trend and not extended",
            "entry_price_source": "broker_executable_bid_ask",
            "details": {"di_edge": 13, "directional_ma28_slope_atr": 0.2, "entry_extension_atr": 0.4, "htf": {"signal": "LONG", "ma7": 1.2, "ma14": 1.1, "ma28": 1.0, "di_plus": 20, "di_minus": 5, "open": 1.1, "close": 1.2}},
            "score_details": {"score": 70.0, "adx_points": 20, "di_points": 30, "volume_points": 0},
        },
    )


def risk() -> FxRiskDecision:
    return FxRiskDecision(True, "accepted", units=1_000, risk_amount=10.0, exit_plan=FxExitPlan(1.099, 1.105, 0.002, 0.004, 2, 20, 40))


def test_off_policy_does_not_require_or_consult_ai() -> None:
    policy = apply_ai_execution_policy(
        hard_safety_allowed=True,
        settings=AiDeliberationSettings(mode="off"),
        result=None,
    )
    assert policy.allowed is True
    assert policy.reason == "deterministic_execution_authoritative"


def test_hard_safety_gate_always_wins_over_ai_confirm() -> None:
    service = AiDeliberationService(AiDeliberationSettings(mode="advisory"), provider=lambda *_: valid_response() | {"decision": "CONFIRM", "recommended_action": "ALLOW"})
    result = service.deliberate(_evidence())
    policy = apply_ai_execution_policy(hard_safety_allowed=False, settings=AiDeliberationSettings(mode="advisory", reject_blocks=True), result=result)
    assert result.successful is True
    assert policy.allowed is False
    assert policy.reason == "hard_safety_gate_blocked"


def test_shadow_provider_failure_is_recordable_and_nonblocking() -> None:
    service = AiDeliberationService(AiDeliberationSettings(mode="shadow"), provider=lambda *_: {"not": "the schema"})
    result = service.deliberate(_evidence())
    policy = apply_ai_execution_policy(hard_safety_allowed=True, settings=AiDeliberationSettings(mode="shadow"), result=result)
    assert result.successful is False
    assert result.failure_reason and "AiResponseValidationError" in result.failure_reason
    assert policy.allowed is True


def test_advisory_reject_requires_explicit_opt_in_and_confidence() -> None:
    service = AiDeliberationService(AiDeliberationSettings(mode="advisory"), provider=lambda *_: valid_response() | {"decision": "REJECT", "recommended_action": "REJECT"})
    result = service.deliberate(_evidence())
    default = apply_ai_execution_policy(hard_safety_allowed=True, settings=AiDeliberationSettings(mode="advisory"), result=result)
    opt_in = apply_ai_execution_policy(hard_safety_allowed=True, settings=AiDeliberationSettings(mode="advisory", reject_blocks=True), result=result)
    assert default.allowed is True
    assert opt_in.allowed is False


def test_inconsistent_decision_and_action_is_rejected() -> None:
    malformed = valid_response() | {"decision": "CONFIRM", "recommended_action": "REJECT"}
    try:
        validate_ai_audit_response(malformed)
    except Exception as exc:
        assert type(exc).__name__ == "AiResponseValidationError"
    else:
        raise AssertionError("inconsistent AI output was accepted")


def test_deterministic_audit_catches_numeric_reasoning_contradictions() -> None:
    contradictory = intent()
    contradictory = FxSignalIntent(**{**contradictory.__dict__, "signal_row": {**contradictory.signal_row, "adx": 10}, "metadata": {**contradictory.metadata, "details": {**contradictory.metadata["details"], "entry_extension_atr": 1.2}}})
    findings = {item["check"]: item for item in deterministic_reasoning_audit(contradictory, StrategySettings())}
    assert findings["adx_min"]["passed"] is False
    assert findings["strong_trend_claim_has_adx"]["passed"] is False
    assert findings["entry_extension_max"]["passed"] is False
    assert findings["not_extended_claim"]["passed"] is False


def test_evidence_uses_executable_ask_and_authoritative_calendar_metadata() -> None:
    evidence = _evidence()
    payload = evidence.to_dict()
    assert payload["observed_facts"]["executable_entry_price"] == 1.1010
    assert payload["observed_facts"]["ask"] == 1.1010
    assert payload["observed_facts"]["bid"] == 1.1008
    event = payload["external_context"]["upcoming_or_recent_events"][0]
    assert event["currency"] == "USD"
    assert event["time_until_event_seconds"] == 720
    assert "description" not in event


def test_persistence_is_idempotent_per_parent_signal() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        with closing(StructuredJournal(f"sqlite:///{Path(tmp) / 'journal.db'}")) as journal:
            payload = {
                "signal_id": "fxsig-EURUSD-abc", "timestamp": NOW, "instrument": "EUR_USD", "side": "LONG", "model": "test",
                "prompt_version": "v1", "mode": "shadow", "status": "completed", "decision": "FLAG", "confidence": 0.9,
                "reasoning_audit_status": "CONSISTENT", "reasoning_issues": [], "reasoning_supporting_factors": [],
                "market_context_status": "MATERIAL_CONTRADICTION", "market_context_issues": [], "market_context_supporting_factors": [],
                "contradictions": [], "recommended_action": "FLAG", "summary": "test", "evidence_hash": "a", "output_hash": "b",
                "latency_ms": 5, "failure_reason": None, "evidence": {"parent": True}, "response": valid_response(),
            }
            first, created_first = journal.record_ai_deliberation(payload=payload)
            second, created_second = journal.record_ai_deliberation(payload=payload)
            assert created_first is True
            assert created_second is False
            assert first.id == second.id
            assert len(journal.recent_ai_deliberations()) == 1


def test_ai_auditor_has_no_mt5_or_execution_dependency() -> None:
    import fxbot.ai_deliberation as auditor

    source = inspect.getsource(auditor)
    assert "fxbot.mt5" not in source
    assert "create_market_order" not in source
    assert "close_position" not in source


def _evidence():
    event_time = NOW + timedelta(minutes=12)
    return build_signal_evidence(
        signal_id="fxsig-EURUSD-abc", intent=intent(), instrument=FxInstrument("EUR_USD"),
        price=PriceSnapshot("EUR_USD", bid=1.1008, ask=1.1010, time=NOW),
        portfolio=FxPortfolioState(10_000, 10_000, 0, 0), risk=risk(), strategy=StrategySettings(),
        news_events=[NewsEvent("FOMC", "USD", "high", event_time, event_time + timedelta(minutes=15), impact_score=95)],
        news_stale=False, active_sessions={"new_york"}, now=NOW, demo_only=True,
    )
