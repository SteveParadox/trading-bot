"""Typed schemas for point-in-time trade-quality prediction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


PREDICTION_SCHEMA_VERSION = "v2"
PREDICTION_STATUSES = {"ok", "off", "unavailable", "error"}


@dataclass(frozen=True)
class PredictionRequest:
    """Causal input to the numerical prediction service.

    `market_snapshot` must be the snapshot captured for this candidate at
    decision time. No future outcome or execution result belongs in this object.
    """

    candidate_id: str
    candidate_trade: dict[str, Any]
    market_snapshot: dict[str, Any]
    strategy_signal: str
    strategy_score: float
    execution_cost_pips_round_trip: float
    pip_size: float

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id is required")
        if not isinstance(self.candidate_trade, dict):
            raise ValueError("candidate_trade must be an object")
        if not isinstance(self.market_snapshot, dict) or not self.market_snapshot:
            raise ValueError("market_snapshot is required")
        if not math.isfinite(float(self.strategy_score)):
            raise ValueError("strategy_score must be finite")
        if not math.isfinite(float(self.execution_cost_pips_round_trip)) or self.execution_cost_pips_round_trip < 0:
            raise ValueError("execution_cost_pips_round_trip must be finite and nonnegative")
        if not math.isfinite(float(self.pip_size)) or self.pip_size <= 0:
            raise ValueError("pip_size must be finite and positive")


@dataclass(frozen=True)
class NumericalPrediction:
    """Structured numerical-model response consumed by the LLM deliberator."""

    status: str
    schema_version: str = PREDICTION_SCHEMA_VERSION
    tp_before_sl_probability: float | None = None
    profitable_5m_probability: float | None = None
    profitable_15m_probability: float | None = None
    immediate_adverse_probability: float | None = None
    continuation_probability: float | None = None
    fake_breakout_probability: float | None = None
    expected_mfe_pips: float | None = None
    expected_mae_pips: float | None = None
    expected_return_pips: float | None = None
    pullback_probability: float | None = None
    expected_pullback_pips: float | None = None
    entry_action: str | None = None
    entry_action_confidence: float | None = None
    entry_action_probabilities: dict[str, float] | None = None
    entry_model_name: str | None = None
    entry_model_version: str | None = None
    entry_model_hash: str | None = None
    entry_action_error: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    model_hash: str | None = None
    target: str | None = None
    feature_version: str | None = None
    feature_hash: str | None = None
    latency_ms: int = 0
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status not in PREDICTION_STATUSES:
            raise ValueError(f"invalid prediction status {self.status!r}")
        for name in (
            "tp_before_sl_probability",
            "profitable_5m_probability",
            "profitable_15m_probability",
            "immediate_adverse_probability",
            "continuation_probability",
            "fake_breakout_probability",
            "pullback_probability",
            "entry_action_confidence",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0):
                raise ValueError(f"{name} must be between 0 and 1")
        if self.entry_action is not None and self.entry_action not in {
            "ENTER_NOW", "WAIT_30S", "WAIT_1M", "WAIT_3M", "SKIP"
        }:
            raise ValueError("entry_action is not supported")
        if self.entry_action_probabilities is not None:
            expected = {"ENTER_NOW", "WAIT_30S", "WAIT_1M", "WAIT_3M", "SKIP"}
            if set(self.entry_action_probabilities) != expected:
                raise ValueError("entry_action_probabilities must contain the complete action space")
            probabilities = [float(value) for value in self.entry_action_probabilities.values()]
            if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in probabilities):
                raise ValueError("entry_action_probabilities must be valid probabilities")
            if not math.isclose(sum(probabilities), 1.0, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError("entry_action_probabilities must sum to 1")
        for name in (
            "expected_mfe_pips",
            "expected_mae_pips",
            "expected_return_pips",
            "expected_pullback_pips",
        ):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.latency_ms < 0:
            raise ValueError("latency_ms cannot be negative")
        if self.status == "ok" and all(
            value is None
            for value in (
                self.tp_before_sl_probability,
                self.profitable_5m_probability,
                self.profitable_15m_probability,
                self.immediate_adverse_probability,
                self.continuation_probability,
                self.fake_breakout_probability,
                self.expected_mfe_pips,
                self.expected_mae_pips,
                self.expected_return_pips,
                self.pullback_probability,
                self.expected_pullback_pips,
                self.entry_action_confidence,
            )
        ):
            raise ValueError("successful prediction requires at least one numerical output")

    @property
    def successful(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
