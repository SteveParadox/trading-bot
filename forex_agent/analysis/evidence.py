from __future__ import annotations

from typing import Any

import numpy as np

from forex_agent.data.schemas import (
    CounterfactualResult,
    EvidencePackage,
    TradeRecord,
)
from forex_agent.data.ingestion import compute_r_multiple, compute_r_values, extract_session
from forex_agent.analysis.similarity import find_similar_trades


def build_evidence_package(
    trade: TradeRecord,
    all_trades: list[TradeRecord],
) -> EvidencePackage:
    """Construct a structured Evidence Package for a single trade.

    This is the factual foundation that feeds into the LLM explanation layer.
    It must never invent data.
    """
    closed = [t for t in all_trades if t.exit_price is not None]
    closed_same_symbol = [t for t in closed if t.symbol == trade.symbol]

    # --- Baseline metrics (portfolio-wide) ---
    baseline: dict[str, float] = {}
    if closed:
        r_all = compute_r_values(closed)
        pnls_all = np.array([t.pnl for t in closed])
        baseline = {
            "win_rate": float(np.mean(pnls_all > 0)),
            "expectancy": float(np.mean(r_all)) if r_all else 0.0,
            "median_r": float(np.median(r_all)) if r_all else 0.0,
            "total_trades": float(len(closed)),
        }

    # --- Similar trades ---
    similar = find_similar_trades(trade, all_trades)
    similar_dict: dict[str, Any] = {
        "n": similar.n_matches,
        "win_rate": similar.win_rate,
        "expectancy": similar.expectancy_r,
        "median_mae": similar.median_mae,
        "median_mfe": similar.median_mfe,
        "outcome_distribution": similar.outcome_distribution,
        "definition": similar.definition_of_similar,
        "warning": similar.sample_size_warning,
    }

    # --- Regime context ---
    regime_dict: dict[str, Any] = {}
    if trade.regime is not None:
        regime_trades = [t for t in closed_same_symbol if t.regime == trade.regime]
        if regime_trades:
            r_regime = compute_r_values(regime_trades)
            regime_dict = {
                "regime": trade.regime.value,
                "n_trades_in_regime": len(regime_trades),
                "win_rate": float(np.mean([t.pnl > 0 for t in regime_trades])),
                "expectancy_r": float(np.mean(r_regime)) if r_regime else 0.0,
            }

    # --- Execution context ---
    execution_dict: dict[str, Any] = {}
    spreads = [t.spread_at_entry for t in closed_same_symbol if t.spread_at_entry > 0]
    if spreads:
        med_spread = float(np.median(spreads))
        execution_dict = {
            "spread_at_entry": trade.spread_at_entry,
            "median_spread": med_spread,
            "spread_ratio": trade.spread_at_entry / med_spread if med_spread > 0 else 0.0,
            "slippage_pips": trade.slippage_pips,
        }

    # --- Timing context ---
    timing_dict: dict[str, Any] = {}
    if trade.entry_time is not None:
        session = extract_session(trade.entry_time)
        session_trades = [t for t in closed_same_symbol if extract_session(t.entry_time) == session]
        if session_trades:
            r_session = compute_r_values(session_trades)
            timing_dict = {
                "session": session,
                "hour": trade.entry_time.hour,
                "day_of_week": trade.entry_time.strftime("%A"),
                "n_trades_in_session": len(session_trades),
                "session_win_rate": float(np.mean([t.pnl > 0 for t in session_trades])),
                "session_expectancy": float(np.mean(r_session)) if r_session else 0.0,
            }

    # --- Risk context ---
    risk_dict: dict[str, Any] = {}
    if trade.risk_amount > 0 and trade.account_balance > 0:
        risk_pct = trade.risk_amount / trade.account_balance
        risk_dict = {
            "risk_amount": trade.risk_amount,
            "risk_pct": risk_pct,
            "avg_risk_pct": _avg_risk_pct(closed),
        }

    # --- Anomalies for this trade ---
    anomalies: list[dict[str, Any]] = []
    r_trade = compute_r_multiple(trade)
    if r_trade is not None and closed:
        r_all = np.array(compute_r_values(closed))
        if len(r_all) >= 10:
            mean_r = float(np.mean(r_all))
            std_r = float(np.std(r_all, ddof=1))
            if std_r > 0:
                z = (r_trade - mean_r) / std_r
                if abs(z) > 2.0:
                    anomalies.append({
                        "type": "r_outlier",
                        "r_multiple": r_trade,
                        "z_score": float(z),
                        "description": f"R-multiple of {r_trade:.2f} is {abs(z):.1f} std devs from mean",
                    })

    # --- Statistical tests ---
    stat_tests: list[dict[str, Any]] = []
    if similar.n_matches >= 10:
        r_similar = [r for r in (compute_r_multiple(t) for t in all_trades
                     if t.trade_id != trade.trade_id
                     and t.symbol == trade.symbol
                     and t.direction == trade.direction
                     and t.exit_price is not None) if r is not None]
        if r_trade is not None and len(r_similar) >= 10:
            from forex_agent.analysis.statistics_enhanced import permutation_test
            perm = permutation_test(
                np.array([r_trade]),
                np.array(r_similar),
                n_permutations=1000,
            )
            stat_tests.append({
                "test": "permutation_test",
                "description": "Is this trade's R-multiple unusual for its peer group?",
                "p_value": perm["p_value"],
                "effect_size": perm["effect_size"],
            })

    # --- Counterfactuals ---
    counterfactuals = _build_counterfactuals(trade)

    # --- Confidence ---
    data_points = sum([
        1 if baseline else 0,
        1 if similar_dict["n"] >= 5 else 0,
        1 if regime_dict else 0,
        1 if execution_dict else 0,
        1 if timing_dict else 0,
    ])
    confidence = min(1.0, data_points / 4.0)

    return EvidencePackage(
        trade_id=trade.trade_id,
        trade=trade.to_dict(),
        baseline=baseline,
        similar_trades=similar_dict,
        regime=regime_dict,
        execution=execution_dict,
        timing=timing_dict,
        risk=risk_dict,
        anomalies=anomalies,
        counterfactuals=[c.to_dict() for c in counterfactuals],
        statistical_tests=stat_tests,
        confidence=confidence,
    )


def _build_counterfactuals(trade: TradeRecord) -> list[CounterfactualResult]:
    """Build structured counterfactual scenarios."""
    results: list[CounterfactualResult] = []
    r = compute_r_multiple(trade)

    if trade.stop_loss and trade.entry_price > 0:
        stop_dist = abs(trade.entry_price - trade.stop_loss)
        # Wider stop counterfactual
        wider_stop = trade.stop_loss + (-stop_dist * 0.5 if trade.direction == "LONG" else stop_dist * 0.5)
        results.append(CounterfactualResult(
            scenario=f"Wider stop (1.5x current distance of {stop_dist:.5f})",
            estimated_outcome_r=None,
            methodology="hypothetical",
            confidence=0.3,
            data_available=False,
            notes="Would require tick-level data to estimate precisely.",
        ))

        # Tighter stop
        results.append(CounterfactualResult(
            scenario=f"Tighter stop (0.75x current distance)",
            estimated_outcome_r=None,
            methodology="hypothetical",
            confidence=0.3,
            data_available=False,
            notes="Would require tick-level data.",
        ))

    # No-trade counterfactual
    if r is not None and r < 0:
        results.append(CounterfactualResult(
            scenario="Avoid trade (no entry)",
            estimated_outcome_r=0.0,
            methodology="hypothetical",
            confidence=0.5,
            data_available=True,
            notes="Avoiding the trade would have resulted in 0R.",
        ))

    # Session alternative
    if trade.entry_time is not None:
        session = extract_session(trade.entry_time)
        results.append(CounterfactualResult(
            scenario=f"Different session (current: {session})",
            estimated_outcome_r=None,
            methodology="estimated",
            confidence=0.4,
            data_available=True,
            notes="Would need to compare expectancy across sessions for this pair.",
        ))

    return results


def _avg_risk_pct(closed: list[TradeRecord]) -> float:
    pcts = [t.risk_amount / t.account_balance for t in closed if t.account_balance > 0 and t.risk_amount > 0]
    return float(np.mean(pcts)) if pcts else 0.0
