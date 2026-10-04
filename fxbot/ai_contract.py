"""Typed contract for AI evaluation of strategy-generated FX candidates.

The AI is deliberately constrained to evaluating an existing candidate. It is
not a signal generator, position sizer, risk authority, or broker interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any


class AiTradeDecision(str, Enum):
    """The only decisions the candidate-evaluation AI may return."""

    TAKE = "TAKE"
    WAIT = "WAIT"
    SKIP = "SKIP"


@dataclass(frozen=True)
class AiTradeRecommendation:
    """Structured, non-executing recommendation for one strategy candidate."""

    decision: AiTradeDecision
    confidence: float
    reason_codes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and between 0 and 1")
        for name, values in (("reason_codes", self.reason_codes), ("warnings", self.warnings)):
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"{name} must contain non-empty strings")

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "confidence": self.confidence,
            "reason_codes": list(self.reason_codes),
            "warnings": list(self.warnings),
        }


def validate_ai_trade_recommendation(payload: dict[str, Any]) -> AiTradeRecommendation:
    """Validate an LLM response against the constrained TAKE/WAIT/SKIP schema.

    This function intentionally accepts no order, sizing, stop, target, or risk
    fields. Those concerns stay with deterministic trading code.
    """

    if not isinstance(payload, dict):
        raise ValueError("AI trade recommendation must be a JSON object")
    required = {"decision", "confidence", "reason_codes", "warnings"}
    missing = required.difference(payload)
    extra = set(payload).difference(required)
    if missing:
        raise ValueError(f"AI trade recommendation is missing fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"AI trade recommendation has unsupported fields: {sorted(extra)}")
    try:
        decision = AiTradeDecision(str(payload["decision"]).upper())
    except ValueError as exc:
        raise ValueError("decision must be TAKE, WAIT, or SKIP") from exc
    confidence = payload["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be numeric")
    reason_codes = payload["reason_codes"]
    warnings = payload["warnings"]
    if not isinstance(reason_codes, list) or not isinstance(warnings, list):
        raise ValueError("reason_codes and warnings must be arrays")
    return AiTradeRecommendation(
        decision=decision,
        confidence=float(confidence),
        reason_codes=tuple(reason_codes),
        warnings=tuple(warnings),
    )
