from __future__ import annotations

from typing import Any

import numpy as np

from forex_agent.data.schemas import SimilarTradeResult, TradeRecord
from forex_agent.data.ingestion import compute_r_multiple, extract_session


def find_similar_trades(
    target: TradeRecord,
    all_trades: list[TradeRecord],
    min_matches: int = 5,
) -> SimilarTradeResult:
    """Find historically comparable trades using transparent rule-based matching.

    Similarity dimensions (in priority order):
      - symbol (mandatory)
      - direction (mandatory)
      - session
      - regime
      - signal strength band
      - stop distance band (normalized)
      - timeframe
    """
    closed = [t for t in all_trades if t.exit_price is not None and t.trade_id != target.trade_id]
    if not closed:
        return SimilarTradeResult(trade_id=target.trade_id, definition_of_similar="no closed trades available")

    # --- Tier 1: mandatory filters (symbol + direction) ---
    mandatory = [
        t for t in closed
        if t.symbol == target.symbol and t.direction == target.direction
    ]
    if len(mandatory) < min_matches:
        # Widen: drop direction requirement
        mandatory = [t for t in closed if t.symbol == target.symbol]

    definition_parts = [f"symbol={target.symbol}", f"direction={target.direction}"]

    # --- Tier 2: soft filters (narrowing where data allows) ---
    candidates = mandatory

    if target.session or target.entry_time is not None:
        target_session = target.session or extract_session(target.entry_time)
        session_matches = [t for t in candidates if _trade_session(t) == target_session]
        if len(session_matches) >= min_matches:
            candidates = session_matches
            definition_parts.append(f"session={target_session}")

    if target.regime is not None:
        regime_matches = [t for t in candidates if t.regime == target.regime]
        if len(regime_matches) >= min_matches:
            candidates = regime_matches
            definition_parts.append(f"regime={target.regime.value}")

    if target.signal_strength is not None:
        band = _signal_band(target.signal_strength)
        band_matches = [t for t in candidates if t.signal_strength is not None and _signal_band(t.signal_strength) == band]
        if len(band_matches) >= min_matches:
            candidates = band_matches
            definition_parts.append(f"signal_band={band}")

    if target.timeframe:
        tf_matches = [t for t in candidates if t.timeframe == target.timeframe]
        if len(tf_matches) >= min_matches:
            candidates = tf_matches
            definition_parts.append(f"timeframe={target.timeframe}")

    stop_band = _stop_distance_band(target)
    if stop_band is not None:
        band_matches = [t for t in candidates if _stop_distance_band(t) == stop_band]
        if len(band_matches) >= min_matches:
            candidates = band_matches
            definition_parts.append(f"stop_band={stop_band}")

    definition = " and ".join(definition_parts)

    # --- Compute stats on matched set ---
    n = len(candidates)
    if n == 0:
        return SimilarTradeResult(
            trade_id=target.trade_id,
            definition_of_similar=definition,
            n_matches=0,
            sample_size_warning="No comparable trades found.",
        )

    r_values = [r for r in (compute_r_multiple(t) for t in candidates) if r is not None]
    winners = [t for t in candidates if t.is_winner]
    win_rate = len(winners) / n

    expectancy = float(np.mean(r_values)) if r_values else 0.0

    mae_values = [t.mae for t in candidates if t.mae is not None]
    mfe_values = [t.mfe for t in candidates if t.mfe is not None]

    outcome_dist: dict[str, int] = {}
    for t in candidates:
        r = compute_r_multiple(t)
        if r is None:
            continue
        if r > 0:
            outcome_dist["win"] = outcome_dist.get("win", 0) + 1
        elif r == 0:
            outcome_dist["breakeven"] = outcome_dist.get("breakeven", 0) + 1
        elif r > -1:
            outcome_dist["small_loss"] = outcome_dist.get("small_loss", 0) + 1
        else:
            outcome_dist["large_loss"] = outcome_dist.get("large_loss", 0) + 1

    warning = ""
    if n < 10:
        warning = f"Small sample ({n} trades). Results should be treated with caution."
    elif n < 30:
        warning = f"Moderate sample ({n} trades). Results are indicative but not conclusive."

    return SimilarTradeResult(
        trade_id=target.trade_id,
        definition_of_similar=definition,
        n_matches=n,
        win_rate=win_rate,
        expectancy_r=expectancy,
        median_mae=float(np.median(mae_values)) if mae_values else 0.0,
        median_mfe=float(np.median(mfe_values)) if mfe_values else 0.0,
        outcome_distribution=outcome_dist,
        matching_trade_ids=[t.trade_id for t in candidates[:20]],
        sample_size_warning=warning,
    )


def _trade_session(t: TradeRecord) -> str:
    if t.session:
        return t.session
    if t.entry_time is not None:
        return extract_session(t.entry_time)
    return "unknown"


def _signal_band(strength: float) -> str:
    if strength >= 0.8:
        return "strong"
    if strength >= 0.5:
        return "moderate"
    return "weak"


def _stop_distance_band(t: TradeRecord) -> str | None:
    if t.entry_price <= 0 or not t.stop_loss:
        return None
    dist_pct = abs(t.entry_price - t.stop_loss) / t.entry_price
    if dist_pct < 0.002:
        return "tight"
    if dist_pct < 0.005:
        return "normal"
    return "wide"
