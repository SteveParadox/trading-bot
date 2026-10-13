"""Pure deterministic validation of proposed exit actions.

No MT5 dependency, no broker execution. A future advisory controller must
re-fetch/lock the live broker position and repeat these checks immediately
before sending anything. Deliberately not wired to the running position manager.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite
from typing import Any

from fxbot.ai.exit_intelligence import ExitAction, ExitPrediction, _finite, _time


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
        if (any(not isinstance(value, (int, float)) or _finite(value) is None for value in vars(self).values())
                or self.minimum_net_profit < 0
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
    quote_age_seconds = _finite(quote_age_seconds)
    pip_size, tick_size = _finite(pip_size), _finite(tick_size)
    stop_distance, freeze_distance = _finite(stop_distance), _finite(freeze_distance)
    reject = lambda reason: ExitPolicyResult(False, action, reason)
    if prediction.status != "ok" or prediction.decision not in {a.value for a in ExitAction}:
        return reject("prediction_unavailable")
    if prediction.position_id != str(snapshot.get("position_id") or ""):
        return reject("position_identity_mismatch")
    if prediction.timestamp != snapshot.get("timestamp"):
        return reject("stale_prediction_timestamp")
    if not advisory_released:
        return reject("advisory_execution_not_released")
    if (_finite(quote_age_seconds) is None
            or quote_age_seconds < 0 or quote_age_seconds > settings.maximum_quote_age_seconds):
        return reject("stale_or_unknown_quote")
    if snapshot.get("direction") not in {"BUY", "SELL"}:
        return reject("invalid_direction")
    stop = _finite(snapshot.get("stop_loss"))
    bid, ask = _finite(snapshot.get("bid")), _finite(snapshot.get("ask"))
    if stop is None or stop <= 0:
        return reject("protective_stop_missing")
    if bid is None or ask is None or bid <= 0 or ask < bid:
        return reject("invalid_executable_quote")
    if action == ExitAction.HOLD.value:
        return ExitPolicyResult(True, action, "hold_existing_protection")
    if action == ExitAction.TAKE_PROFIT_NOW.value:
        evidence = snapshot.get("net_profit_evidence")
        if not isinstance(evidence, dict):
            return reject("positive_executable_net_profit_unverified")
        components = [_finite(evidence.get(key)) for key in
                      ("gross_liquidation_pl", "swap", "paid_commission", "closing_commission", "slippage_cost")]
        if (any(value is None for value in components)
                or any(value < 0 for value in components[2:])
                or evidence.get("costs_in_account_currency") is not True
                or not evidence.get("account_currency")
                or not evidence.get("quote_timestamp")
                or evidence.get("quote_timestamp") != snapshot.get("quote_timestamp")
                or _finite(evidence.get("liquidation_price")) != (bid if snapshot["direction"] == "BUY" else ask)):
            return reject("positive_executable_net_profit_unverified")
        net = components[0] + components[1] - sum(components[2:])
        if not isfinite(net) or net <= 0 or net < settings.minimum_net_profit:
            return reject("positive_executable_net_profit_unverified")
        return ExitPolicyResult(True, action, "profitability_gate_passed")
    if action == ExitAction.EXIT.value:
        # A future advisory release must verify this was justified by an
        # approved, forward-tested defensive-exit policy.
        return reject("defensive_exit_not_validated")
    if action == ExitAction.REDUCE_POSITION.value:
        # Broker "units" are not lots: the execution layer must supply MT5
        # current volume in lots from a fresh position read.
        volume = _finite(snapshot.get("broker_volume_lots"))
        if volume is None or volume <= 0:
            return reject("broker_lot_volume_unavailable")
        if (snapshot.get("partial_close_state_verified") is not True
                or snapshot.get("account_mode") not in {"hedging", "netting"}
                or snapshot.get("pending_partial_close") is not False):
            return reject("partial_close_state_unverified")
        minimum = _finite(snapshot.get("broker_volume_min"))
        step = _finite(snapshot.get("broker_volume_step"))
        if minimum is None or step is None or minimum <= 0 or step <= 0:
            return reject("partial_close_broker_volume_constraints")
        if not isfinite(volume / step):
            return reject("partial_close_broker_volume_constraints")
        if not abs(volume / step - round(volume / step)) < 1e-8:
            return reject("partial_close_broker_volume_constraints")
        raw = float(volume) * settings.reduction_ratio
        if not isfinite(raw / step) or not isfinite(volume / step):
            return reject("partial_close_broker_volume_constraints")
        steps = floor((raw + 1e-12) / step)
        reduction = round(steps * step, 8)
        remaining = round(float(volume) - reduction, 8)
        if reduction < minimum or remaining < minimum:
            return reject("partial_close_broker_volume_constraints")
        return ExitPolicyResult(True, action, "partial_close_volume_feasible",
                                reduction_volume=reduction)
    if action == ExitAction.TRAIL_STOP.value:
        if (any(_finite(value) is None for value in (pip_size, tick_size, stop_distance, freeze_distance))
                or pip_size <= 0 or tick_size <= 0
                or stop_distance is None or stop_distance < 0
                or freeze_distance is None or freeze_distance < 0):
            return reject("broker_stop_constraints_unavailable")
        atr = _finite(snapshot.get("atr_pips"))
        old_stop = stop
        quote = snapshot.get("bid") if snapshot["direction"] == "BUY" else snapshot.get("ask")
        if (atr is None or not isfinite(float(atr)) or float(atr) <= 0
                or old_stop is None or quote is None):
            return reject("trailing_indicators_unavailable")
        direction = 1 if snapshot["direction"] == "BUY" else -1
        try:
            if _time(snapshot.get("atr_timestamp")) > _time(snapshot.get("timestamp")):
                return reject("noncausal_trailing_indicator")
        except (ValueError, TypeError):
            return reject("noncausal_trailing_indicator")
        if direction * (float(quote) - old_stop) <= freeze_distance:
            return reject("existing_stop_in_freeze_zone")
        distance = float(atr) * settings.trailing_atr_multiplier * pip_size
        if not isfinite(distance):
            return reject("trailing_indicators_unavailable")
        distance = max(distance, stop_distance, freeze_distance)
        candidate = float(quote) - direction * distance
        if not isfinite(candidate / tick_size):
            return reject("broker_stop_constraints_unavailable")
        # Round AWAY from executable price to avoid violating broker stops.
        if direction > 0:
            candidate = floor(candidate / tick_size + 1e-9) * tick_size
        else:
            from math import ceil
            candidate = ceil(candidate / tick_size - 1e-9) * tick_size
        candidate = round(candidate, 10)
        improvement = direction * (candidate - float(old_stop))
        if candidate <= 0:
            return reject("invalid_stop_price")
        if improvement < settings.minimum_stop_change_pips * pip_size:
            return reject("trailing_stop_would_not_tighten_enough")
        if direction * (float(quote) - candidate) <= max(stop_distance, freeze_distance):
            return reject("trailing_stop_broker_distance")
        return ExitPolicyResult(True, action, "tighter_stop_only", suggested_stop=candidate)
    return reject("unsupported_action")
