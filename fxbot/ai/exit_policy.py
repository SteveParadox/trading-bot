"""Pure deterministic validation of proposed exit actions.

No MT5 dependency, no broker execution. A future advisory controller must
re-fetch/lock the live broker position and repeat these checks immediately
before sending anything. Deliberately not wired to the running position manager.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite
from typing import Any

from fxbot.ai.exit_intelligence import ExitAction, ExitPrediction


@dataclass(frozen=True)
class ExitPolicySettings:
    minimum_net_profit: float = 1.0
    trailing_atr_multiplier: float = 1.4
    minimum_stop_change_pips: float = 1.0
    reduction_ratio: float = 0.5
    min_volume: float = 0.01
    volume_step: float = 0.01
    maximum_quote_age_seconds: float = 15.0

    def __post_init__(self) -> None:
        if (not isfinite(self.minimum_net_profit) or self.minimum_net_profit < 0
                or self.trailing_atr_multiplier <= 0 or self.minimum_stop_change_pips <= 0
                or not 0 < self.reduction_ratio <= 1 or self.min_volume <= 0
                or self.volume_step <= 0 or self.maximum_quote_age_seconds <= 0):
            raise ValueError("invalid deterministic exit policy configuration")


@dataclass(frozen=True)
class ExitPolicyResult:
    eligible: bool
    action: str
    reason: str
    suggested_stop: float | None = None
    reduction_volume: float | None = None


def evaluate_exit_policy(
    snapshot: dict[str, Any], prediction: ExitPrediction, settings: ExitPolicySettings,
    *,
    advisory_released: bool = False,
    quote_age_seconds: float | None = None,
    pip_size: float | None = None,
    tick_size: float | None = None,
    stop_distance: float | None = None,
    freeze_distance: float | None = None,
) -> ExitPolicyResult:
    """Validate only; never treat a planned action as a confirmed broker fill."""
    action = prediction.decision or "NONE"
    reject = lambda reason: ExitPolicyResult(False, action, reason)
    if prediction.status != "ok" or prediction.decision not in {a.value for a in ExitAction}:
        return reject("prediction_unavailable")
    if prediction.position_id != str(snapshot.get("position_id") or ""):
        return reject("position_identity_mismatch")
    if prediction.timestamp != snapshot.get("timestamp"):
        return reject("stale_prediction_timestamp")
    if not advisory_released:
        return reject("advisory_execution_not_released")
    if (quote_age_seconds is None or not isfinite(quote_age_seconds)
            or quote_age_seconds < 0 or quote_age_seconds > settings.maximum_quote_age_seconds):
        return reject("stale_or_unknown_quote")
    if snapshot.get("direction") not in {"BUY", "SELL"}:
        return reject("invalid_direction")
    if not snapshot.get("stop_loss"):
        return reject("protective_stop_missing")
    if action == ExitAction.HOLD.value:
        return ExitPolicyResult(True, action, "hold_existing_protection")
    if action == ExitAction.TAKE_PROFIT_NOW.value:
        net = snapshot.get("estimated_net_pl")
        if net is None or not isfinite(float(net)) or float(net) < settings.minimum_net_profit:
            return reject("positive_executable_net_profit_unverified")
        return ExitPolicyResult(True, action, "profitability_gate_passed")
    if action == ExitAction.EXIT.value:
        # A future advisory release must verify this was justified by an
        # approved, forward-tested defensive-exit policy.
        return reject("defensive_exit_not_validated")
    if action == ExitAction.REDUCE_POSITION.value:
        # Broker "units" are not lots: the execution layer must supply MT5
        # current volume in lots from a fresh position read.
        volume = snapshot.get("broker_volume_lots")
        if volume is None or not isfinite(float(volume)) or volume <= 0:
            return reject("broker_lot_volume_unavailable")
        raw = float(volume) * settings.reduction_ratio
        steps = floor((raw + 1e-12) / settings.volume_step)
        reduction = round(steps * settings.volume_step, 8)
        remaining = round(float(volume) - reduction, 8)
        if reduction < settings.min_volume or remaining < settings.min_volume:
            return reject("partial_close_broker_volume_constraints")
        return ExitPolicyResult(True, action, "partial_close_volume_feasible",
                                reduction_volume=reduction)
    if action == ExitAction.TRAIL_STOP.value:
        if (pip_size is None or pip_size <= 0 or tick_size is None or tick_size <= 0
                or stop_distance is None or stop_distance < 0
                or freeze_distance is None or freeze_distance < 0):
            return reject("broker_stop_constraints_unavailable")
        atr = snapshot.get("atr_pips")
        old_stop = snapshot.get("stop_loss")
        quote = snapshot.get("bid") if snapshot["direction"] == "BUY" else snapshot.get("ask")
        if (atr is None or not isfinite(float(atr)) or float(atr) <= 0
                or old_stop is None or quote is None):
            return reject("trailing_indicators_unavailable")
        direction = 1 if snapshot["direction"] == "BUY" else -1
        distance = float(atr) * settings.trailing_atr_multiplier * pip_size
        distance = max(distance, stop_distance, freeze_distance)
        candidate = float(quote) - direction * distance
        # Round AWAY from executable price to avoid violating broker stops.
        if direction > 0:
            candidate = floor(candidate / tick_size + 1e-9) * tick_size
        else:
            from math import ceil
            candidate = ceil(candidate / tick_size - 1e-9) * tick_size
        candidate = round(candidate, 10)
        improvement = direction * (candidate - float(old_stop))
        if improvement < settings.minimum_stop_change_pips * pip_size:
            return reject("trailing_stop_would_not_tighten_enough")
        if direction * (float(quote) - candidate) <= max(stop_distance, freeze_distance):
            return reject("trailing_stop_broker_distance")
        return ExitPolicyResult(True, action, "tighter_stop_only", suggested_stop=candidate)
    return reject("unsupported_action")
