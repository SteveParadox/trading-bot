"""Explicit order recovery state machine.

An API timeout after submission is not a rejection.  The order is kept in an
unknown state until a broker lookup proves whether it exists.  This module
keeps that distinction explicit and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable


class RecoveryState(str, Enum):
    PENDING = "pending"
    UNKNOWN = "unknown"
    SUBMITTED = "submitted"
    FILLED = "filled"
    CLOSED = "closed"
    REJECTED = "rejected"
    OPERATOR_REVIEW = "operator_review"


class RecoveryEvent(str, Enum):
    BROKER_FOUND = "broker_found"
    BROKER_NOT_FOUND = "broker_not_found"
    SUBMISSION_FAILED = "submission_failed"
    SUBMISSION_REJECTED = "submission_rejected"
    FILL_CONFIRMED = "fill_confirmed"
    RECONCILIATION_FAILED = "reconciliation_failed"


@dataclass(frozen=True)
class RecoveryTransition:
    before: RecoveryState
    event: RecoveryEvent
    after: RecoveryState
    requires_alert: bool = False


def transition(state: str | RecoveryState, event: str | RecoveryEvent) -> RecoveryTransition:
    before = RecoveryState(state)
    signal = RecoveryEvent(event)
    if signal is RecoveryEvent.BROKER_FOUND:
        after = RecoveryState.SUBMITTED
        alert = False
    elif signal is RecoveryEvent.FILL_CONFIRMED:
        after = RecoveryState.FILLED
        alert = False
    elif signal is RecoveryEvent.SUBMISSION_FAILED:
        after = RecoveryState.UNKNOWN
        alert = True
    elif signal is RecoveryEvent.SUBMISSION_REJECTED:
        after = RecoveryState.REJECTED
        alert = True
    elif signal is RecoveryEvent.RECONCILIATION_FAILED:
        after = RecoveryState.OPERATOR_REVIEW
        alert = True
    elif signal is RecoveryEvent.BROKER_NOT_FOUND:
        after = RecoveryState.UNKNOWN if before in {RecoveryState.PENDING, RecoveryState.UNKNOWN} else before
        alert = before in {RecoveryState.PENDING, RecoveryState.UNKNOWN}
    else:  # pragma: no cover - exhaustive enum guard
        raise ValueError(f"unsupported recovery event: {signal}")
    return RecoveryTransition(before, signal, after, alert)


def reconcile_order(
    row: Any,
    broker_lookup: Callable[[str], dict[str, Any] | None],
) -> RecoveryTransition:
    """Resolve one journal row using a broker idempotency lookup."""

    try:
        broker_order = broker_lookup(str(row.client_order_id))
    except Exception:
        return transition(row.status, RecoveryEvent.RECONCILIATION_FAILED)
    if broker_order:
        return transition(row.status, RecoveryEvent.FILL_CONFIRMED if broker_order.get("state") == "filled" else RecoveryEvent.BROKER_FOUND)
    return transition(row.status, RecoveryEvent.BROKER_NOT_FOUND)
