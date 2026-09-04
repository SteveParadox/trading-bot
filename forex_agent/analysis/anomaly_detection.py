from __future__ import annotations

from forex_agent.data.schemas import TradeRecord
from forex_agent.data.ingestion import compute_r_multiple, extract_session


def detect_losing_streaks(
    trades: list[TradeRecord], min_streak: int = 3
) -> list[dict]:
    closed = [t for t in trades if t.exit_price is not None]
    streaks: list[dict] = []
    current_streak = 0
    streak_start_idx = 0

    for i, t in enumerate(closed):
        if not t.is_winner:
            if current_streak == 0:
                streak_start_idx = i
            current_streak += 1
        else:
            if current_streak >= min_streak:
                streaks.append({
                    "start_index": streak_start_idx,
                    "end_index": i - 1,
                    "length": current_streak,
                    "total_loss": sum(
                        t.pnl for t in closed[streak_start_idx:i]
                    ),
                })
            current_streak = 0

    if current_streak >= min_streak:
        streaks.append({
            "start_index": streak_start_idx,
            "end_index": len(closed) - 1,
            "length": current_streak,
            "total_loss": sum(
                t.pnl for t in closed[streak_start_idx:]
            ),
        })

    return streaks


def detect_drawdown_anomalies(
    trades: list[TradeRecord], threshold_pct: float = 0.10
) -> list[dict]:
    closed = [t for t in trades if t.exit_price is not None]
    if not closed:
        return []

    balance = closed[0].account_balance
    peak = balance
    anomalies: list[dict] = []

    for i, t in enumerate(closed):
        balance += t.pnl
        if balance > peak:
            peak = balance
        dd_pct = (peak - balance) / peak if peak > 0 else 0.0
        if dd_pct > threshold_pct:
            anomalies.append({
                "trade_index": i,
                "trade_id": t.trade_id,
                "drawdown_pct": dd_pct,
                "peak_balance": peak,
                "current_balance": balance,
            })

    return anomalies


def detect_expectancy_shift(
    trades: list[TradeRecord], window: int = 20, threshold: float = 0.5
) -> list[dict]:
    closed = [t for t in trades if t.exit_price is not None]
    if len(closed) < window * 2:
        return []

    shifts: list[dict] = []
    prev_exp = None

    for i in range(window, len(closed)):
        start = i - window
        subset = closed[start:i]
        wins = [t for t in subset if t.is_winner]
        losses = [t for t in subset if not t.is_winner]
        win_rate = len(wins) / len(subset)
        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0.0
        avg_loss = abs(sum(t.pnl for t in losses) / len(losses)) if losses else 0.0
        exp = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        if prev_exp is not None:
            shift = exp - prev_exp
            if abs(shift) > threshold:
                shifts.append({
                    "trade_index": i,
                    "trade_id": closed[i].trade_id,
                    "previous_expectancy": prev_exp,
                    "current_expectancy": exp,
                    "shift": shift,
                    "direction": "improvement" if shift > 0 else "degradation",
                })

        prev_exp = exp

    return shifts


def detect_distribution_shift(
    trades: list[TradeRecord], window: int = 30
) -> list[dict]:
    """Detect shifts in the R-multiple distribution between halves of data."""
    closed = [t for t in trades if t.exit_price is not None]
    r_values = []
    for t in closed:
        r = compute_r_multiple(t)
        if r is not None:
            r_values.append(r)
    if len(r_values) < window * 2:
        return []

    first = r_values[: len(r_values) // 2]
    second = r_values[len(r_values) // 2 :]

    import math

    mean1 = sum(first) / len(first)
    mean2 = sum(second) / len(second)
    var1 = sum((x - mean1) ** 2 for x in first) / (len(first) - 1) if len(first) > 1 else 0.0
    var2 = sum((x - mean2) ** 2 for x in second) / (len(second) - 1) if len(second) > 1 else 0.0
    std1 = math.sqrt(var1)
    std2 = math.sqrt(var2)

    return [{
        "type": "distribution_shift",
        "first_half_mean_r": mean1,
        "second_half_mean_r": mean2,
        "first_half_std": std1,
        "second_half_std": std2,
        "mean_difference": mean2 - mean1,
        "description": (
            f"R-distribution shifted from {mean1:.2f}R to {mean2:.2f}R "
            f"(std {std1:.2f} -> {std2:.2f})"
        ),
        "severity": "high" if abs(mean2 - mean1) > 0.5 else "medium",
    }]


def detect_trade_duration_anomalies(
    trades: list[TradeRecord], z_threshold: float = 2.5
) -> list[dict]:
    """Detect trades with anomalous durations (holding time)."""
    closed = [t for t in trades
              if t.exit_price is not None and t.exit_time is not None]
    if len(closed) < 10:
        return []

    durations = []
    for t in closed:
        dur = (t.exit_time - t.entry_time).total_seconds() / 3600.0
        durations.append(dur)

    mean = sum(durations) / len(durations)
    import math

    variance = sum((d - mean) ** 2 for d in durations) / (len(durations) - 1) if len(durations) > 1 else 0.0
    std = math.sqrt(variance)
    if std == 0:
        return []

    anomalies = []
    for i, (t, dur) in enumerate(zip(closed, durations)):
        z = (dur - mean) / std
        if abs(z) > z_threshold:
            anomalies.append({
                "trade_index": i,
                "trade_id": t.trade_id,
                "duration_hours": dur,
                "mean_duration_hours": mean,
                "std_duration_hours": std,
                "z_score": z,
                "severity": "high" if abs(z) > 3.0 else "medium",
            })
    return anomalies


def detect_loss_clustering(
    trades: list[TradeRecord], window: int = 10, ratio_threshold: float = 0.8
) -> list[dict]:
    """Detect windows where losses cluster unusually densely.

    Only reports the most severe non-overlapping windows to avoid alert spam.
    """
    closed = [t for t in trades if t.exit_price is not None]
    if len(closed) < window:
        return []

    # Evaluate each window, collect those above threshold.
    candidates = []
    for i in range(len(closed) - window + 1):
        subset = closed[i : i + window]
        losses = sum(1 for t in subset if not t.is_winner)
        ratio = losses / window
        if ratio >= ratio_threshold:
            candidates.append((i, losses, ratio))

    if not candidates:
        return []

    # Merge overlapping windows and keep only the most severe from each
    # contiguous run of high-loss windows.
    anomalies = []
    current_run: list[tuple[int, int, float]] = []
    for cand in candidates:
        if current_run and cand[0] > current_run[-1][0] + 1:
            # End of a run; pick the worst window in it.
            worst = max(current_run, key=lambda c: (c[2], c[1]))
            anomalies.append(_loss_cluster_anomaly(worst, window))
            current_run = []
        current_run.append(cand)
    if current_run:
        worst = max(current_run, key=lambda c: (c[2], c[1]))
        anomalies.append(_loss_cluster_anomaly(worst, window))

    return anomalies


def _loss_cluster_anomaly(
    cand: tuple[int, int, float], window: int
) -> dict:
    start_idx, losses, ratio = cand
    return {
        "start_index": start_idx,
        "end_index": start_idx + window - 1,
        "window_size": window,
        "loss_count": losses,
        "loss_ratio": ratio,
        "description": (
            f"{losses}/{window} trades in a {window}-trade window were losses "
            f"({ratio:.0%})"
        ),
        "severity": "high" if ratio >= 0.9 else "medium",
    }


def detect_pair_deterioration(
    trades: list[TradeRecord], min_trades: int = 10
) -> list[dict]:
    """Detect pairs whose recent performance degrades vs historical."""
    from collections import defaultdict

    by_pair: dict[str, list[TradeRecord]] = defaultdict(list)
    for t in trades:
        by_pair[t.symbol].append(t)

    anomalies = []
    for pair, pair_trades in by_pair.items():
        closed = [t for t in pair_trades if t.exit_price is not None]
        if len(closed) < min_trades:
            continue
        split = len(closed) // 2
        first = closed[:split]
        second = closed[split:]
        r_first = [compute_r_multiple(t) for t in first if compute_r_multiple(t) is not None]
        r_second = [compute_r_multiple(t) for t in second if compute_r_multiple(t) is not None]
        if not r_first or not r_second:
            continue
        mean_first = sum(r_first) / len(r_first)
        mean_second = sum(r_second) / len(r_second)
        if mean_first > 0.1 and mean_second < mean_first - 0.3:
            anomalies.append({
                "type": "pair_deterioration",
                "pair": pair,
                "historical_mean_r": mean_first,
                "recent_mean_r": mean_second,
                "first_n": len(r_first),
                "second_n": len(r_second),
                "description": (
                    f"{pair} recent expectancy {mean_second:.2f}R vs "
                    f"historical {mean_first:.2f}R"
                ),
                "severity": "medium",
            })
    return anomalies


def detect_session_deterioration(
    trades: list[TradeRecord], min_trades: int = 10
) -> list[dict]:
    """Detect sessions whose performance degrades vs historical."""
    from collections import defaultdict

    by_session: dict[str, list[TradeRecord]] = defaultdict(list)
    for t in trades:
        session = extract_session(t.entry_time)
        by_session[session].append(t)

    anomalies = []
    for session, sess_trades in by_session.items():
        closed = [t for t in sess_trades if t.exit_price is not None]
        if len(closed) < min_trades:
            continue
        split = len(closed) // 2
        first = closed[:split]
        second = closed[split:]
        r_first = [compute_r_multiple(t) for t in first if compute_r_multiple(t) is not None]
        r_second = [compute_r_multiple(t) for t in second if compute_r_multiple(t) is not None]
        if not r_first or not r_second:
            continue
        mean_first = sum(r_first) / len(r_first)
        mean_second = sum(r_second) / len(r_second)
        if mean_first > 0.1 and mean_second < mean_first - 0.3:
            anomalies.append({
                "type": "session_deterioration",
                "session": session,
                "historical_mean_r": mean_first,
                "recent_mean_r": mean_second,
                "first_n": len(r_first),
                "second_n": len(r_second),
                "description": (
                    f"{session} session recent expectancy {mean_second:.2f}R vs "
                    f"historical {mean_first:.2f}R"
                ),
                "severity": "medium",
            })
    return anomalies


def detect_spread_anomalies(
    trades: list[TradeRecord], z_threshold: float = 2.5
) -> list[dict]:
    """Detect unusually wide spreads."""
    closed = [t for t in trades if t.exit_price is not None]
    spreads = [t.spread_at_entry for t in closed if t.spread_at_entry > 0]
    if len(spreads) < 10:
        return []

    import math

    mean = sum(spreads) / len(spreads)
    variance = sum((s - mean) ** 2 for s in spreads) / (len(spreads) - 1) if len(spreads) > 1 else 0.0
    std = math.sqrt(variance)
    if std == 0:
        return []

    anomalies = []
    for i, t in enumerate(closed):
        if t.spread_at_entry <= 0:
            continue
        z = (t.spread_at_entry - mean) / std
        if z > z_threshold:
            anomalies.append({
                "trade_index": i,
                "trade_id": t.trade_id,
                "spread": t.spread_at_entry,
                "mean_spread": mean,
                "std_spread": std,
                "z_score": z,
                "severity": "low",
            })
            break  # only report the worst outlier to avoid alert spam
    return anomalies


def detect_mae_mfe_shift(trades: list[TradeRecord]) -> list[dict]:
    """Detect deterioration in MAE/MFE profile (e.g., larger adverse moves)."""
    closed = [t for t in trades if t.exit_price is not None and t.mae is not None]
    if len(closed) < 20:
        return []

    split = len(closed) // 2
    first_mae = [t.mae for t in closed[:split] if t.mae is not None]
    second_mae = [t.mae for t in closed[split:] if t.mae is not None]
    if not first_mae or not second_mae:
        return []

    mean1 = sum(first_mae) / len(first_mae)
    mean2 = sum(second_mae) / len(second_mae)
    if mean2 > mean1 * 1.3:
        return [{
            "type": "mae_mfe_shift",
            "first_half_avg_mae": mean1,
            "second_half_avg_mae": mean2,
            "description": (
                f"Average MAE increased from {mean1:.5f} to {mean2:.5f} "
                f"({(mean2/mean1 - 1)*100:.0f}% increase)"
            ),
            "severity": "medium",
        }]
    return []


def detect_all(trades: list[TradeRecord]) -> list[dict]:
    """Run all anomaly detectors and return combined results."""
    results = []
    results.extend(detect_losing_streaks(trades))
    results.extend(detect_drawdown_anomalies(trades))
    results.extend(detect_expectancy_shift(trades))
    results.extend(detect_distribution_shift(trades))
    results.extend(detect_trade_duration_anomalies(trades))
    results.extend(detect_loss_clustering(trades))
    results.extend(detect_pair_deterioration(trades))
    results.extend(detect_session_deterioration(trades))
    results.extend(detect_spread_anomalies(trades))
    results.extend(detect_mae_mfe_shift(trades))
    return results
