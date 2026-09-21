"""Offline, quote-path diagnostics for journalled FX candidates.

This is a conditional candidate study, NOT a portfolio backtest. Candidates
can overlap. It has no MT5 client, no AI request and no order submission path.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

import pandas as pd

from fxbot.instruments import FxInstrument
from fxbot.models import Side


@dataclass(frozen=True)
class Outcome:
    net_r: float
    gross_r: float
    mae_r: float
    mfe_r: float
    holding_seconds: float
    time_to_mfe_seconds: float
    stop_seconds: float | None
    reason: str
    spread_r: float
    slippage_r: float


def utc(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("Research timestamps must have explicit UTC offsets")
    return stamp.tz_convert("UTC")


def validate_quotes(quotes: pd.DataFrame) -> pd.DataFrame:
    if not {"timestamp", "instrument", "bid", "ask"}.issubset(quotes.columns):
        raise ValueError("Quote CSV requires timestamp,instrument,bid,ask")
    q = quotes.copy()
    q["timestamp"] = q.timestamp.map(utc)
    for _, group in q.groupby("instrument"):
        if not group.timestamp.is_monotonic_increasing or group.timestamp.duplicated().any():
            raise ValueError("Quote timestamps must be strictly increasing per instrument")
    for name in q.instrument.unique():
        FxInstrument(name)  # Reject malformed/non-FX symbols.
    if not all(math.isfinite(float(v)) and float(v) > 0 for v in q[["bid", "ask"]].to_numpy().flat):
        raise ValueError("Invalid executable quote")
    if (q.ask <= q.bid).any():
        raise ValueError("Crossed or zero-spread quote")
    return q


def quote_outcome(candidate: dict, quotes: pd.DataFrame, *, target_r: float,
                  horizon_seconds: int, cost_multiplier: float = 1.0,
                  max_gap_seconds: int = 120) -> Outcome | None:
    """First-touch quote replay; missing coverage returns None, not a win/loss.

    No OHLC path ordering is invented. Stops fill at the first observed quote
    beyond the level (including gaps); targets receive no price improvement.
    Commission and adverse slippage apply once each. Tick timestamps must be
    observation times consistent with the candidate journal.
    """
    if target_r <= 0 or not math.isfinite(target_r) or cost_multiplier < 1 or not math.isfinite(cost_multiplier):
        raise ValueError("Invalid target or cost multiplier")
    if horizon_seconds <= 0 or max_gap_seconds <= 0:
        raise ValueError("Horizon and maximum gap must be positive")
    snap = candidate["snapshot"]
    pars = snap["parameters"]
    if pars["slippage_pips_per_side"] is None or pars["commission_pips_round_trip"] is None:
        raise ValueError("Explicit slippage and commission assumptions required")
    start = utc(snap["observed_at"])
    end = start + pd.Timedelta(seconds=horizon_seconds)
    side = Side(snap["side"])
    pip = FxInstrument(candidate["instrument"]).pip_size
    slip = pars["slippage_pips_per_side"] * pip * cost_multiplier
    commission = pars["commission_pips_round_trip"] * pip * cost_multiplier
    spread = (snap["ask"] - snap["bid"]) * cost_multiplier
    mid = (snap["ask"] + snap["bid"]) / 2
    entry = mid + side.sign * (spread / 2 + slip)
    stop = candidate["baseline_risk"]["exit_plan"]["stop_loss"]
    risk = (entry - stop) * side.sign
    if not all(math.isfinite(float(v)) for v in (entry, risk, slip, commission)) or risk <= 0 or min(slip, commission) < 0:
        raise ValueError("Invalid stop/cost assumptions")
    target = entry + side.sign * risk * target_r
    path = quotes[(quotes.instrument == candidate["instrument"]) & (quotes.timestamp > start)]
    last_time = start
    mae, mfe, mfe_time = spread + slip, 0.0, 0.0
    for row in path.itertuples():
        stamp = row.timestamp
        gap = (stamp - last_time).total_seconds()
        if gap > max_gap_seconds or stamp > end + pd.Timedelta(seconds=max_gap_seconds):
            return None
        stressed_spread = (row.ask - row.bid) * cost_multiplier
        liquidation = (row.ask + row.bid) / 2 - side.sign * stressed_spread / 2
        move = (liquidation - entry) * side.sign
        held = (stamp - start).total_seconds()
        mae = max(mae, -move)
        if move > mfe:
            mfe, mfe_time = move, held
        reason = "stop" if (liquidation - stop) * side.sign <= 0 else "target" if (liquidation - target) * side.sign >= 0 else "horizon" if stamp >= end else None
        if reason:
            gross = min(move, target_r * risk) if reason == "target" else move
            return Outcome((gross - slip - commission) / risk, gross / risk, mae / risk, mfe / risk,
                           held, mfe_time, held if reason == "stop" else None, reason, spread / risk, 2 * slip / risk)
        last_time = stamp
    return None


def metrics(outcomes: list[Outcome]) -> dict[str, Any]:
    if not outcomes:
        return {"trades": 0, "expectancy_r": None, "profit_factor": None}
    values = [v.net_r for v in outcomes]
    wins, losses = [v for v in values if v > 0], [v for v in values if v < 0]
    gross_profit, gross_loss = sum(wins), -sum(losses)
    avg_win, avg_loss = mean(wins) if wins else None, mean(losses) if losses else None
    streak, max_streak = 0, 0
    for value in values:
        streak = streak + 1 if value < 0 else 0
        max_streak = max(max_streak, streak)
    return {"trades": len(values), "win_rate": len(wins) / len(values), "loss_rate": len(losses) / len(values),
            "breakeven_rate": values.count(0) / len(values),
            "profit_factor": gross_profit / gross_loss if gross_loss else None,
            "profit_factor_unbounded": bool(gross_profit and not gross_loss),
            "expectancy_r": mean(values), "median_r": median(values), "average_winner_r": avg_win,
            "average_loser_r": avg_loss, "payoff_ratio": avg_win / -avg_loss if wins and losses else None,
            "max_consecutive_losses_in_candidate_order": max_streak,
            "mean_holding_seconds": mean(o.holding_seconds for o in outcomes),
            "median_holding_seconds": median(o.holding_seconds for o in outcomes),
            "mean_mae_r": mean(o.mae_r for o in outcomes), "mean_mfe_r": mean(o.mfe_r for o in outcomes),
            "mean_time_to_mfe_seconds": mean(o.time_to_mfe_seconds for o in outcomes),
            "mean_time_to_stop_seconds": mean(o.stop_seconds for o in outcomes if o.stop_seconds is not None) if any(o.stop_seconds is not None for o in outcomes) else None,
            "mean_spread_r": mean(o.spread_r for o in outcomes), "mean_slippage_r": mean(o.slippage_r for o in outcomes),
            "winners_mae_r": [o.mae_r for o in outcomes if o.net_r > 0],
            "losers_mae_r": [o.mae_r for o in outcomes if o.net_r < 0],
            "winners_mfe_r": [o.mfe_r for o in outcomes if o.net_r > 0],
            "losers_mfe_r": [o.mfe_r for o in outcomes if o.net_r < 0]}


def load_candidates(path: Path) -> list[dict]:
    unique = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        event = row.get("payload", row)
        if event.get("event_type") != "sniper_candidate":
            continue
        candidate = event["payload"]
        snap = candidate["snapshot"]
        key = (candidate["instrument"], candidate["decision_time"], snap["side"])
        # Keep earliest observation, not the last rescan with future context.
        if key not in unique or utc(snap["observed_at"]) < utc(unique[key]["snapshot"]["observed_at"]):
            unique[key] = candidate
    return sorted(unique.values(), key=lambda c: utc(c["snapshot"]["observed_at"]))


def study(candidates: list[dict], quotes: pd.DataFrame, *, train_end, validation_end,
          horizon_seconds=3600, max_gap_seconds=120) -> dict:
    train_end, validation_end = utc(train_end), utc(validation_end)
    if train_end >= validation_end:
        raise ValueError("Chronological split boundaries must increase")
    quotes = validate_quotes(quotes)
    records, purged = [], 0
    for candidate in candidates:
        start = utc(candidate["snapshot"]["observed_at"])
        end = start + pd.Timedelta(seconds=horizon_seconds + max_gap_seconds)
        if any(start < boundary <= end for boundary in (train_end, validation_end)):
            purged += 1
            continue
        split = "train" if start < train_end else "validation" if start < validation_end else "out_of_sample"
        records.append((split, candidate))
    report = {"kind": "conditional_candidate_diagnostics", "not_portfolio_backtest": True,
              "candidates": len(candidates), "purged_boundary_candidates": purged,
              "splits": {"train_end": train_end.isoformat(), "validation_end": validation_end.isoformat()},
              "horizon_seconds": horizon_seconds, "max_gap_seconds": max_gap_seconds,
              "eligibility": "Entry gate membership is frozen at observed settings across target/cost scenarios. Scenarios are not deployable configurations.",
              "portfolio_metrics": {"net_profit": None, "max_drawdown": None, "average_drawdown": None,
                                    "recovery_factor": None, "sharpe": None, "sortino": None},
              "unavailable_reason": "Overlapping hypothetical candidates do not define a portfolio equity curve or independent trades.",
              "experiments": []}
    for split in ["train", "validation", "out_of_sample"]:
        for target in [.25, .5, .75, 1., 1.5]:
            for stress in [1., 1.25, 1.5, 2.]:
                groups: dict[str, list] = {"baseline_candidates": [], "all_gates": [], "rejected": []}
                missing = 0
                for part, candidate in records:
                    if part != split:
                        continue
                    outcome = quote_outcome(candidate, quotes, target_r=target, horizon_seconds=horizon_seconds,
                                            cost_multiplier=stress, max_gap_seconds=max_gap_seconds)
                    if outcome is None:
                        missing += 1
                        continue
                    snap = candidate["snapshot"]
                    stages = snap["stages"]
                    passes = bool(candidate["would_allow"])
                    groups["baseline_candidates"].append(outcome)
                    groups["all_gates" if passes else "rejected"].append(outcome)
                    for stage, passed in stages.items():
                        if passed:
                            groups.setdefault("only_" + stage, []).append(outcome)
                    for cap in [.4, .6, .8]:
                        if snap.get("extension_atr", math.inf) <= cap and all(v for k, v in stages.items() if k != "location") and not snap.get("stale_frames"):
                            groups.setdefault("extension_cap_" + str(cap), []).append(outcome)
                    for name, value in [("instrument", candidate["instrument"]), ("regime", snap.get("regime", "unknown")),
                                        ("session", "+".join(snap.get("sessions", [])) or "none")]:
                        groups.setdefault(name + ":" + value, []).append(outcome)
                report["experiments"].append({"split": split, "target_r": target, "cost_multiplier": stress,
                                               "missing_quote_coverage": missing,
                                               "groups": {k: metrics(v) for k, v in groups.items()}})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--quotes", type=Path, required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--validation-end", required=True)
    parser.add_argument("--horizon-seconds", type=int, default=3600)
    parser.add_argument("--max-gap-seconds", type=int, default=120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = load_candidates(args.journal)
    if not candidates:
        parser.error("No FX sniper_candidate events found; no report fabricated")
    report = study(candidates, pd.read_csv(args.quotes), train_end=args.train_end, validation_end=args.validation_end,
                   horizon_seconds=args.horizon_seconds, max_gap_seconds=args.max_gap_seconds)
    report["source_hashes"] = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
                               [("journal", args.journal), ("quotes", args.quotes)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
