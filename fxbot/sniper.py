"""Optional deterministic qualification. No broker calls and no probability claims."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from typing import Any

import pandas as pd

from fxbot.models import Side


@dataclass(frozen=True)
class SniperSettings:
    mode: str = "off"
    regime_filter: bool = True
    location_filter: bool = True
    trigger_required: bool = True
    anti_chase: bool = True
    cost_filter: bool = True
    structure_stop: bool = False
    failure_exit: bool = False
    time_exit: bool = False
    lookback: int = 20
    trigger_lookback: int = 3
    trigger_mode: str = "either"
    min_atr_ratio: float = 0.7
    max_atr_ratio: float = 2.0
    max_candle_atr: float = 1.8
    max_move_atr: float = 2.0
    max_extension_atr: float = 0.6
    min_close_strength: float = 0.65
    structure_buffer_atr: float = 0.1
    max_cost_ratio: float = 0.20
    # None means unknown. Enforcement fails closed when the cost gate is on.
    slippage_pips_per_side: float | None = None
    commission_pips_round_trip: float | None = None
    time_exit_bars: int = 4
    min_progress_atr: float = 0.25
    failure_buffer_atr: float = 0.1

    def __post_init__(self) -> None:
        if self.mode not in {"off", "shadow", "enforce"}:
            raise ValueError("FX_SNIPER_MODE must be off, shadow or enforce")
        if self.trigger_mode not in {"either", "breakout", "pullback"}:
            raise ValueError("invalid sniper trigger_mode")
        for name in ("lookback", "trigger_lookback", "time_exit_bars"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"sniper.{name} must be a positive integer")
        if self.lookback < max(3, self.trigger_lookback + 1):
            raise ValueError("sniper.lookback must cover the trigger and volatility history")
        for name in ("min_atr_ratio", "max_atr_ratio", "max_candle_atr", "max_move_atr",
                     "max_extension_atr", "structure_buffer_atr", "min_progress_atr", "failure_buffer_atr"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"sniper.{name} must be finite and positive")
        if self.min_atr_ratio >= self.max_atr_ratio:
            raise ValueError("invalid sniper ATR ratio bounds")
        if not 0 < self.max_cost_ratio < 1 or not 0 < self.min_close_strength <= 1:
            raise ValueError("invalid sniper cost/close ratio")
        for name in ("slippage_pips_per_side", "commission_pips_round_trip"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"sniper.{name} must be finite and nonnegative, or None")


def settings_from_env(get_str, get_bool, get_float, get_int, get_optional_float) -> SniperSettings:
    defaults = SniperSettings()
    values = {}
    for item in fields(defaults):
        default = getattr(defaults, item.name)
        name = "FX_SNIPER_" + item.name.upper()
        if default is None:
            values[item.name] = get_optional_float(name)
        elif isinstance(default, bool):
            values[item.name] = get_bool(name, default)
        elif isinstance(default, int):
            values[item.name] = get_int(name, default)
        elif isinstance(default, float):
            values[item.name] = get_float(name, default)
        else:
            values[item.name] = get_str(name, default).lower()
    return SniperSettings(**values)


@dataclass(frozen=True)
class Qualification:
    allowed: bool
    reason: str
    snapshot: dict[str, Any]


def qualify_entry(window: pd.DataFrame, side: Side, entry: float, settings: SniperSettings) -> Qualification:
    """Window must contain completed candles only, with causal indicators.

    Every stage is recorded, including when another stage fails. Structural
    levels exclude the trigger candle and only use already-confirmed extrema.
    """
    snapshot: dict[str, Any] = {"version": 1, "mode": settings.mode, "side": side.value,
                                "entry": entry, "parameters": asdict(settings)}
    columns = ["open", "high", "low", "close", "atr", "ma7", "ma14", "ma28", "adx", "di_plus", "di_minus"]
    if len(window) < settings.lookback + 1 or any(c not in window for c in columns):
        return Qualification(False, "SNIPER_REJECT_DATA", snapshot)
    recent = window.iloc[-settings.lookback - 1:]
    try:
        valid = all(math.isfinite(float(v)) for v in recent[columns].to_numpy().flat)
    except (TypeError, ValueError):
        valid = False
    if not valid or not math.isfinite(entry) or entry <= 0 or (recent["atr"] <= 0).any():
        return Qualification(False, "SNIPER_REJECT_DATA", snapshot)
    if ((recent["high"] < recent[["open", "close", "low"]].max(axis=1)) |
        (recent["low"] > recent[["open", "close", "high"]].min(axis=1)) |
        (recent[["open", "high", "low", "close"]] <= 0).any(axis=1)).any():
        return Qualification(False, "SNIPER_REJECT_DATA", snapshot)
    row, previous = recent.iloc[-1], recent.iloc[-2]
    history = recent.iloc[:-1]
    atr = float(row.atr)
    sign = side.sign
    atr_ratio = atr / float(history.atr.median())
    candle_atr = float(row.high - row.low) / atr
    extension = max(0.0, (entry - float(row.ma7)) * sign / atr)
    move = (entry - float(recent.iloc[-1 - settings.trigger_lookback].close)) * sign / atr
    close_strength = ((float(row.close - row.low) if sign > 0 else float(row.high - row.close)) /
                      float(row.high - row.low)) if row.high > row.low else 0.0
    levels = history.iloc[-settings.trigger_lookback:]
    breakout_level = float(levels.high.max() if sign > 0 else levels.low.min())
    breakout = (float(row.close) - breakout_level) * sign > 0
    touched = previous.low <= previous.ma7 if sign > 0 else previous.high >= previous.ma7
    reclaim_level = float(previous.high if sign > 0 else previous.low)
    pullback = bool(touched and (float(row.close) - reclaim_level) * sign > 0)
    directional = (float(row.close) - float(row.open)) * sign > 0
    breakout = bool(breakout and directional and close_strength >= settings.min_close_strength)
    pullback = bool(pullback and directional and close_strength >= settings.min_close_strength)
    trigger = ("pullback" if pullback and settings.trigger_mode != "breakout" else
               "breakout" if breakout and settings.trigger_mode != "pullback" else "none")
    trigger_level = reclaim_level if trigger == "pullback" else breakout_level
    # One neighbour on both sides confirms a swing, wholly within prior history.
    candidates = history.high if sign > 0 else history.low
    pivots = candidates[(candidates > candidates.shift(1)) & (candidates >= candidates.shift(-1))] if sign > 0 else candidates[(candidates < candidates.shift(1)) & (candidates <= candidates.shift(-1))]
    prior_extreme = float(history.high.max() if sign > 0 else history.low.min())
    obstacles = [float(v) for v in [*pivots, prior_extreme] if (float(v) - entry) * sign > 0]
    obstacle = min(obstacles, key=lambda v: abs(v - entry)) if obstacles else None
    invalidation = float(recent.iloc[-settings.trigger_lookback:].low.min() if sign > 0 else recent.iloc[-settings.trigger_lookback:].high.max())
    structure_stop = invalidation - sign * settings.structure_buffer_atr * atr
    regime = "compression" if atr_ratio < settings.min_atr_ratio else "unstable_expansion" if atr_ratio > settings.max_atr_ratio else "trend_expansion" if atr_ratio > 1 else "trend"
    stages = {
        "regime": not settings.regime_filter or settings.min_atr_ratio <= atr_ratio <= settings.max_atr_ratio,
        "location": not settings.location_filter or extension <= settings.max_extension_atr,
        "trigger": not settings.trigger_required or trigger != "none",
        "anti_chase": not settings.anti_chase or (candle_atr <= settings.max_candle_atr and move <= settings.max_move_atr),
    }
    snapshot.update({"signal_time": row.name.isoformat(), "atr": atr, "atr_ratio": atr_ratio,
                     "regime": regime, "candle_atr": candle_atr, "extension_atr": extension,
                     "move_atr": move, "close_strength": close_strength, "trigger": trigger,
                     "trigger_level": trigger_level, "structure_stop": structure_stop,
                     "obstacle": obstacle, "stages": stages,
                     "features": {c: float(row[c]) for c in columns}})
    reasons = {"regime": "REGIME", "location": "POOR_LOCATION", "trigger": "WEAK_TRIGGER", "anti_chase": "OVEREXTENDED"}
    failures = ["SNIPER_REJECT_" + reasons[k] for k, passed in stages.items() if not passed]
    snapshot["rejections"] = failures
    return Qualification(not failures, failures[0] if failures else "SNIPER_QUALIFIED", snapshot)


def qualify_execution(snapshot: dict[str, Any], *, reward: float, spread: float,
                      pip_size: float, settings: SniperSettings) -> Qualification:
    """Reward is liquidation-quote target minus executable entry (signed).

    That reward already includes the entry spread. Add spread back to obtain
    the underlying gross move; never subtract spread twice from projected P/L.
    Commission is explicitly specified in round-trip pip equivalents.
    """
    result = {**snapshot, "stages": dict(snapshot.get("stages", {})),
              "rejections": list(snapshot.get("rejections", []))}
    valid = all(math.isfinite(x) and x > 0 for x in (reward, spread, pip_size))
    known = settings.slippage_pips_per_side is not None and settings.commission_pips_round_trip is not None
    extra = ((2 * settings.slippage_pips_per_side + settings.commission_pips_round_trip) * pip_size) if known else None
    gross = reward + spread if valid else None
    cost = spread + extra if valid and known else None
    ratio = cost / gross if cost is not None else None
    net = reward - extra if valid and known else None
    result.update({"spread": spread, "gross_move": gross, "round_trip_cost": cost,
                   "net_target_after_cost": net, "cost_ratio": ratio, "reward": reward})
    if not valid:
        result["rejections"].append("SNIPER_REJECT_DATA")
    cost_ok = valid and (not settings.cost_filter or (known and net > 0 and ratio <= settings.max_cost_ratio))
    obstacle = snapshot.get("obstacle")
    side = Side(snapshot["side"])
    # Short historical bars are bid prices; leave one current spread buffer.
    room = (obstacle - snapshot["entry"]) * side.sign - (spread if side is Side.SHORT else 0) if obstacle is not None else None
    room_ok = not settings.location_filter or room is None or room >= reward
    result.update({"room_to_target": room, "unbounded_room": obstacle is None})
    result["stages"].update({"cost": bool(cost_ok), "room": bool(room_ok)})
    if not cost_ok:
        result["rejections"].append("SNIPER_REJECT_HIGH_COST" if known else "SNIPER_REJECT_UNKNOWN_COST")
    if not room_ok:
        result["rejections"].append("SNIPER_REJECT_LOW_ROOM_TO_TARGET")
    failures = result["rejections"]
    return Qualification(not failures, failures[0] if failures else "SNIPER_QUALIFIED", result)


def exit_reason(*, side: Side, close: float, di_plus: float, di_minus: float,
                entry: float, entry_atr: float, trigger_level: float, bars_held: int,
                settings: SniperSettings) -> str | None:
    if not all(math.isfinite(v) for v in (close, di_plus, di_minus, entry, entry_atr, trigger_level)) or entry_atr <= 0:
        return None
    if settings.failure_exit and (close - trigger_level) * side.sign < -settings.failure_buffer_atr * entry_atr:
        return "SNIPER_EXIT_THESIS_FAILED"
    if settings.time_exit and bars_held >= settings.time_exit_bars and (close - entry) * side.sign < settings.min_progress_atr * entry_atr:
        return "SNIPER_EXIT_STAGNANT"
    return None
