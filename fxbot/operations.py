"""Operational health calculations used by the worker and monitoring API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class ClockHealth:
    status: str
    skew_seconds: float
    observed_at: str
    reason: str


def clock_health(host_time: datetime, broker_time: datetime | None, *, max_skew_seconds: float = 300.0) -> ClockHealth:
    host = _utc(host_time)
    if broker_time is None:
        return ClockHealth("unknown", 0.0, "", "broker_time_unavailable")
    broker = _utc(broker_time)
    skew = (broker - host).total_seconds()
    status = "healthy" if abs(skew) <= max_skew_seconds else "unhealthy"
    reason = "within_tolerance" if status == "healthy" else "broker_host_clock_skew"
    return ClockHealth(status, skew, broker.isoformat(), reason)


def freshness(now: datetime, observed_at: datetime | None, *, max_age_seconds: float) -> tuple[bool, float | None]:
    if observed_at is None:
        return False, None
    age = max(0.0, (_utc(now) - _utc(observed_at)).total_seconds())
    return age <= max_age_seconds, age


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)

