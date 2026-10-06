"""Forward-only candidate outcome tracking from observed executable FX quotes.

This module labels what happened *after* a strategy candidate was created.
It never feeds future observations back into live candidate features.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
from typing import Any, Mapping

from fxbot.instruments import FxInstrument, PriceSnapshot
from fxbot.models import Side


RETURN_HORIZONS_SECONDS: dict[str, int] = {
    "return_1m_pips": 60,
    "return_3m_pips": 180,
    "return_5m_pips": 300,
    "return_15m_pips": 900,
    "return_30m_pips": 1800,
}

WAIT_HORIZONS_SECONDS: dict[str, int] = {
    "wait_30s_improvement_pips": 30,
    "wait_1m_improvement_pips": 60,
    "wait_3m_improvement_pips": 180,
    "wait_5m_improvement_pips": 300,
}

MAX_OUTCOME_HORIZON_SECONDS = max(RETURN_HORIZONS_SECONDS.values())
OUTCOME_LABEL_VERSION = "v2"


class CandidateOutcomeTracker:
    """Incrementally label candidates using only quotes observed after creation."""

    def __init__(
        self,
        journal: Any,
        *,
        observation_lag_tolerance_seconds: float = 30.0,
        candidate_scan_limit: int = 2000,
    ) -> None:
        if observation_lag_tolerance_seconds <= 0:
            raise ValueError("observation lag tolerance must be positive")
        self.journal = journal
        self.observation_lag_tolerance_seconds = float(observation_lag_tolerance_seconds)
        self.candidate_scan_limit = max(100, int(candidate_scan_limit))
        self._reconciled_history = False

    def seed(
        self,
        *,
        candidate: Any,
        price: PriceSnapshot,
        instrument: FxInstrument,
        observed_at: datetime,
    ) -> None:
        """Create the outcome row and record the decision-time executable quote."""

        started_at = _utc(candidate.timestamp)
        self.journal.ensure_candidate_outcome(
            candidate_id=candidate.candidate_id,
            started_at=started_at,
            payload={
                "sampling_method": "forward_scan_executable_quotes",
                "label_version": OUTCOME_LABEL_VERSION,
                "lag_tolerance_seconds": self.observation_lag_tolerance_seconds,
                "entry": float(candidate.entry),
                "stop_loss": candidate.stop_loss,
                "take_profit": candidate.take_profit,
                "direction": candidate.direction,
                "pip_size": instrument.pip_size,
                "spread_included_in_returns": True,
                "commission_and_slippage_included": False,
                "missed_horizons": [],
            },
        )
        self.observe(
            candidate=candidate,
            price=price,
            instrument=instrument,
            observed_at=max(_utc(observed_at), started_at),
        )

    def observe_active(
        self,
        *,
        prices: Mapping[str, PriceSnapshot],
        instruments: Mapping[str, FxInstrument],
        observed_at: datetime,
    ) -> None:
        """Update all recent candidates with currently observed executable quotes."""

        now = _utc(observed_at)
        candidates = self.journal.pending_outcome_candidates(limit=self.candidate_scan_limit)
        for candidate in candidates:
            started = _utc(candidate.timestamp)
            age = (now - started).total_seconds()
            if age < 0:
                continue
            outcome = self.journal.find_candidate_outcome(candidate.candidate_id)
            if outcome is not None and outcome.status in {"complete", "incomplete"}:
                continue
            if age > MAX_OUTCOME_HORIZON_SECONDS + self.observation_lag_tolerance_seconds:
                # Never use a quote arriving well after the label horizon to
                # fabricate missing 30-minute/path labels.
                if outcome is None:
                    self.journal.ensure_candidate_outcome(
                        candidate_id=candidate.candidate_id,
                        started_at=started,
                        payload={
                            "sampling_method": "forward_scan_executable_quotes",
                            "label_version": OUTCOME_LABEL_VERSION,
                            "lag_tolerance_seconds": self.observation_lag_tolerance_seconds,
                            "missed_horizons": sorted(
                                [*RETURN_HORIZONS_SECONDS.keys(), *WAIT_HORIZONS_SECONDS.keys()]
                            ),
                        },
                    )
                    reason = "tracker_started_after_horizon"
                else:
                    reason = "tracking_gap_exceeded_horizon"
                self.journal.update_candidate_outcome(
                    candidate.candidate_id,
                    values={
                        "status": "incomplete",
                        "completed_at": now,
                        "data_quality": "degraded",
                    },
                    payload_update={"incomplete_reason": reason},
                )
                continue

            price = prices.get(candidate.symbol)
            instrument = instruments.get(candidate.symbol)
            if price is None or instrument is None:
                continue
            if outcome is None:
                self.journal.ensure_candidate_outcome(
                    candidate_id=candidate.candidate_id,
                    started_at=started,
                    payload={
                        "sampling_method": "forward_scan_executable_quotes",
                        "label_version": OUTCOME_LABEL_VERSION,
                        "lag_tolerance_seconds": self.observation_lag_tolerance_seconds,
                        "entry": float(candidate.entry),
                        "stop_loss": candidate.stop_loss,
                        "take_profit": candidate.take_profit,
                        "direction": candidate.direction,
                        "pip_size": instrument.pip_size,
                        "spread_included_in_returns": True,
                        "commission_and_slippage_included": False,
                        "missed_horizons": [],
                    },
                )
            self.observe(
                candidate=candidate,
                price=price,
                instrument=instrument,
                observed_at=_utc(price.time),
            )

    def observe(
        self,
        *,
        candidate: Any,
        price: PriceSnapshot,
        instrument: FxInstrument,
        observed_at: datetime,
    ) -> None:
        outcome = self.journal.find_candidate_outcome(candidate.candidate_id)
        if outcome is None or outcome.status in {"complete", "incomplete"}:
            return

        started = _utc(candidate.timestamp)
        requested_observed = _utc(observed_at)
        quote_observed = _utc(price.time)
        # The first sample is the decision-time baseline: the quote was known
        # then even if its broker tick timestamp is a few seconds older. Every
        # subsequent sample must advance on an actually newer broker tick.
        observed = (
            max(requested_observed, started)
            if outcome.observation_count == 0
            else quote_observed
        )
        elapsed = (observed - started).total_seconds()
        if elapsed < 0:
            return
        if not _valid_quote(price):
            return
        if elapsed > MAX_OUTCOME_HORIZON_SECONDS + self.observation_lag_tolerance_seconds:
            self.journal.update_candidate_outcome(candidate.candidate_id, values={
                "status": "incomplete", "completed_at": observed, "data_quality": "degraded",
            }, payload_update={"incomplete_reason": "late_observation_after_horizon"})
            return
        if outcome.last_observed_at is not None and observed <= _utc(outcome.last_observed_at):
            return

        side = Side(candidate.direction)
        pip_size = instrument.pip_size
        if pip_size <= 0:
            raise ValueError("instrument pip size must be positive")

        liquidation = float(price.bid if side is Side.LONG else price.ask)
        executable_wait_entry = float(price.ask if side is Side.LONG else price.bid)
        entry = float(candidate.entry)
        move_pips = (liquidation - entry) * side.sign / pip_size
        wait_improvement_pips = (entry - executable_wait_entry) * side.sign / pip_size

        updates: dict[str, Any] = {
            "last_observed_at": observed,
            "observation_count": int(outcome.observation_count) + 1,
        }
        payload = dict(outcome.payload or {})
        missed = set(payload.get("missed_horizons") or [])

        gap = max(0.0, elapsed) if outcome.observation_count == 0 else 0.0
        if outcome.last_observed_at is not None:
            gap = max(0.0, (observed - _utc(outcome.last_observed_at)).total_seconds())
        max_gap = max(float(outcome.max_observation_gap_seconds or 0.0), gap)
        updates["max_observation_gap_seconds"] = max_gap
        degraded = outcome.data_quality == "degraded" or max_gap > self.observation_lag_tolerance_seconds

        if move_pips > 0 and outcome.time_to_profit_seconds is None:
            updates["time_to_profit_seconds"] = elapsed
        if move_pips < 0 and outcome.time_to_loss_seconds is None:
            updates["time_to_loss_seconds"] = elapsed

        favorable = max(0.0, move_pips)
        adverse = max(0.0, -move_pips)
        if favorable > float(outcome.mfe_pips or 0.0):
            updates["mfe_pips"] = favorable
            updates["time_to_mfe_seconds"] = elapsed
        if adverse > float(outcome.mae_pips or 0.0):
            updates["mae_pips"] = adverse
            updates["time_to_mae_seconds"] = elapsed

        tp_now, sl_now = _observed_level_hits(
            side=side,
            liquidation=liquidation,
            stop_loss=payload.get("stop_loss", candidate.stop_loss),
            take_profit=payload.get("take_profit", candidate.take_profit),
        )
        if tp_now and not outcome.tp_hit:
            updates["tp_hit"] = True
            updates["time_to_tp_seconds"] = elapsed
        if sl_now and not outcome.sl_hit:
            updates["sl_hit"] = True
            updates["time_to_sl_seconds"] = elapsed

        if outcome.first_touch is None and (tp_now or sl_now):
            if tp_now and sl_now:
                updates["first_touch"] = "AMBIGUOUS"
                updates["tp_before_sl"] = None
                degraded = True
            else:
                updates["first_touch"] = "TP" if tp_now else "SL"
                updates["tp_before_sl"] = tp_now
            updates["first_touch_at"] = observed
            payload["first_touch_sampling_gap_seconds"] = gap
            payload["first_touch_reliable"] = (
                not (tp_now and sl_now)
                and (gap <= self.observation_lag_tolerance_seconds or outcome.observation_count == 0)
            )

        for field, horizon in RETURN_HORIZONS_SECONDS.items():
            if getattr(outcome, field) is not None:
                continue
            _capture_horizon(
                field=field,
                horizon_seconds=horizon,
                elapsed_seconds=elapsed,
                value=move_pips,
                tolerance_seconds=self.observation_lag_tolerance_seconds,
                updates=updates,
                missed=missed,
            )

        for field, horizon in WAIT_HORIZONS_SECONDS.items():
            if getattr(outcome, field) is not None:
                continue
            _capture_horizon(
                field=field,
                horizon_seconds=horizon,
                elapsed_seconds=elapsed,
                value=wait_improvement_pips,
                tolerance_seconds=self.observation_lag_tolerance_seconds,
                updates=updates,
                missed=missed,
            )

        if missed:
            degraded = True
        updates["data_quality"] = "degraded" if degraded else "good"
        payload["missed_horizons"] = sorted(missed)
        payload["latest_quote_time"] = _utc(price.time).isoformat()
        payload["latest_bid"] = float(price.bid)
        payload["latest_ask"] = float(price.ask)
        sampled = dict(payload.get("horizon_observed_seconds") or {})
        for field in (*RETURN_HORIZONS_SECONDS, *WAIT_HORIZONS_SECONDS):
            if field in updates:
                sampled[field] = elapsed
        payload["horizon_observed_seconds"] = sampled

        if elapsed >= MAX_OUTCOME_HORIZON_SECONDS:
            return_30m = updates.get("return_30m_pips", outcome.return_30m_pips)
            if return_30m is not None:
                updates["status"] = "complete"
                updates["completed_at"] = observed
            elif elapsed > MAX_OUTCOME_HORIZON_SECONDS + self.observation_lag_tolerance_seconds:
                updates["status"] = "incomplete"
                updates["completed_at"] = observed
                updates["data_quality"] = "degraded"
                payload["incomplete_reason"] = "missed_30m_horizon"

        self.journal.update_candidate_outcome(
            candidate.candidate_id,
            values=updates,
            payload_update=payload,
        )


def _capture_horizon(
    *,
    field: str,
    horizon_seconds: int,
    elapsed_seconds: float,
    value: float,
    tolerance_seconds: float,
    updates: dict[str, Any],
    missed: set[str],
) -> None:
    if elapsed_seconds < horizon_seconds:
        return
    lateness = elapsed_seconds - horizon_seconds
    if lateness <= tolerance_seconds:
        updates[field] = float(value)
    else:
        missed.add(field)


def _observed_level_hits(
    *,
    side: Side,
    liquidation: float,
    stop_loss: float | None,
    take_profit: float | None,
) -> tuple[bool, bool]:
    stop = _finite_or_none(stop_loss)
    target = _finite_or_none(take_profit)
    if side is Side.LONG:
        tp_hit = target is not None and liquidation >= target
        sl_hit = stop is not None and liquidation <= stop
    else:
        tp_hit = target is not None and liquidation <= target
        sl_hit = stop is not None and liquidation >= stop
    return tp_hit, sl_hit


def _valid_quote(price: PriceSnapshot) -> bool:
    return (
        math.isfinite(float(price.bid))
        and math.isfinite(float(price.ask))
        and price.bid > 0
        and price.ask > price.bid
    )


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
