from __future__ import annotations

from forex_agent.data.schemas import (
    ConfidenceLevel,
    FailureCategory,
    TradeFailureAnalysis,
    TradeRecord,
)
from forex_agent.data.ingestion import compute_r_multiple


def analyze_trade_failure(
    trade: TradeRecord,
    context: dict | None = None,
) -> TradeFailureAnalysis:
    if trade.exit_price is None:
        return TradeFailureAnalysis(
            trade_id=trade.trade_id,
            failure_category=FailureCategory.UNKNOWN,
            confidence=ConfidenceLevel.LOW,
            description="Trade has no exit price; cannot analyze failure.",
            verdict="Uncertain",
        )

    if trade.is_winner:
        return TradeFailureAnalysis(
            trade_id=trade.trade_id,
            failure_category=FailureCategory.UNKNOWN,
            confidence=ConfidenceLevel.LOW,
            description="This trade was a winner, not a failure.",
            verdict="Valid loss",
        )

    category = classify_failure(trade, context)
    counterfactuals = compute_counterfactuals(trade, context)
    factors = _get_contributing_factors(trade, context)
    evidence = _gather_evidence(trade, context)

    verdict = _map_verdict(category)

    return TradeFailureAnalysis(
        trade_id=trade.trade_id,
        failure_category=category,
        confidence=_determine_confidence(trade, context),
        description=_describe_failure(trade, category, context),
        contributing_factors=factors,
        counterfactual=counterfactuals,
        historical_comparison=_historical_comparison(trade, context),
        recommended_action=_recommend_action(category),
        verdict=verdict,
        research_implication=_research_implication(category),
        evidence=evidence,
    )


def classify_failure(
    trade: TradeRecord, context: dict | None = None
) -> FailureCategory:
    """Classify a losing trade into a failure category using deterministic rules.

    Order of precedence (most specific / most actionable first):
      1. Data-quality problems
      2. Rule violations (failed to follow documented process)
      3. Execution errors (spread / slippage)
      4. Risk-management failures (stop handled poorly, sizing, R:R)
      5. Regime mismatch (entry regime historically poor)
      6. Poor entry (extended, late, bad location)
      7. Signal failure (setup conditions met but edge invalidated)
      8. Otherwise: valid strategy loss / market condition
    """
    ctx = context or {}

    # 1. Data-quality failure
    if _has_data_quality_issue(trade):
        return FailureCategory.DATA_QUALITY_FAILURE

    # 2. Rule violation (documented rules not followed)
    if ctx.get("rule_violation", False):
        return FailureCategory.RULE_VIOLATION

    # 3. Execution error
    spread = ctx.get("avg_spread", trade.spread_at_entry)
    slippage = ctx.get("avg_slippage", trade.slippage_pips)
    if spread > 5.0 or slippage > 3.0:
        return FailureCategory.EXECUTION_ERROR

    # 4. Risk-management failure
    if trade.follow_stop_loss is False and trade.follow_take_profit is False:
        return FailureCategory.RISK_MISMANAGEMENT

    r = compute_r_multiple(trade)
    if r is not None and r < -1.5:
        return FailureCategory.RISK_MISMANAGEMENT

    # 5. Regime mismatch
    regime = ctx.get("regime")
    if regime and regime != "ranging":
        entry_regime = trade.regime
        if entry_regime and entry_regime.value != regime:
            return FailureCategory.REGIME_MISMATCH

    # 6. Poor entry
    if _is_poor_entry(trade, ctx):
        return FailureCategory.POOR_ENTRY

    # 7. Rule / trend violation first (more specific than valid loss)
    if ctx.get("traded_against_trend", False):
        return FailureCategory.INVALID_SETUP

    # 8. Signal failure (M turned against, stop hit after valid setup)
    if ctx.get("signal_failure", False):
        return FailureCategory.SIGNAL_FAILURE

    # 9. Valid strategy loss
    if r is not None and -1.0 <= r <= 0.0:
        # Stop was hit at (or close to) the planned distance with no
        # identifiable defect -> normal statistical loss.
        return FailureCategory.VALID_STRATEGY_LOSS

    return FailureCategory.MARKET_CONDITION


def compute_counterfactuals(
    trade: TradeRecord, context: dict | None = None
) -> str:
    ctx = context or {}
    parts = []

    # MFE-based: did price reach TP at some point?
    if trade.mfe is not None and trade.take_profit is not None:
        if trade.direction == "LONG":
            tp_reached = (trade.entry_price + trade.mfe) >= trade.take_profit
        else:
            tp_reached = (trade.entry_price - trade.mfe) <= trade.take_profit
        if tp_reached:
            parts.append(
                f"Price reached max favorable excursion of {trade.mfe:.5f}. "
                "Take profit was touched at {trade.take_profit}. A trailing or "
                "partial close would have retained some profit."
            )

    # Was entry extended / did price come back to a better entry?
    if trade.mae is not None and trade.exit_price is not None:
        if trade.direction == "LONG":
            came_back = (trade.exit_price - max(trade.entry_price - trade.mae,
                                                trade.entry_price)) > 0
        else:
            came_back = (min(trade.entry_price + trade.mae, trade.entry_price)
                         - trade.exit_price) > 0
        if came_back:
            parts.append(
                f"Price recovered from adverse excursion of {trade.mae:.5f} "
                f"back to exit at {trade.exit_price:.5f}. "
                "A wider stop or delayed entry may have improved the outcome."
            )

    # Stop-variation counterfactual
    if trade.direction == "LONG" and trade.stop_loss:
        stop_distance = abs(trade.entry_price - trade.stop_loss)
        if stop_distance > 0:
            parts.append(
                f"Stop was {stop_distance:.5f} from entry. With a 1.5x wider "
                "stop, the loss would have been larger but survival chances "
                "increase (requires rebalancing position size)."
            )

    if not trade.follow_stop_loss:
        parts.append(
            "Did not follow stop loss. If SL was honored, loss would have been "
            f"limited to {abs(trade.entry_price - trade.stop_loss):.5f}."
        )

    if trade.spread_at_entry > 3.0:
        parts.append(
            f"High spread of {trade.spread_at_entry} pips ate into potential profit."
        )

    if not parts:
        parts.append("No obvious counterfactual improvements identified.")
    return " ".join(parts)


def _get_contributing_factors(
    trade: TradeRecord, context: dict | None = None
) -> list[str]:
    factors = []
    if trade.spread_at_entry > 3.0:
        factors.append(f"High spread: {trade.spread_at_entry} pips")
    if trade.slippage_pips > 1.0:
        factors.append(f"Slippage: {trade.slippage_pips} pips")
    if not trade.follow_stop_loss:
        factors.append("Did not follow stop loss")
    if not trade.follow_take_profit:
        factors.append("Did not follow take profit")
    ctx = context or {}
    if ctx.get("traded_against_trend"):
        factors.append("Traded against the trend")
    if trade.mae is not None and trade.mfe is not None:
        if trade.mfe < abs(trade.mae) * 0.5:
            factors.append(
                f"Trade showed little favorable movement (MFE {trade.mfe:.5f} "
                f"vs MAE {trade.mae:.5f})"
            )
    return factors


def _determine_confidence(
    trade: TradeRecord, context: dict | None = None
) -> ConfidenceLevel:
    factors = _get_contributing_factors(trade, context)
    if len(factors) >= 3:
        return ConfidenceLevel.HIGH
    elif len(factors) >= 1:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


def _describe_failure(
    trade: TradeRecord, category: FailureCategory, context: dict | None = None
) -> str:
    descs = {
        FailureCategory.VALID_STRATEGY_LOSS: (
            "Setup was followed correctly; market moved against the position. "
            "This is an expected statistical loss."
        ),
        FailureCategory.SIGNAL_FAILURE: (
            "Entry conditions were technically satisfied, but subsequent price "
            "action invalidated the expected edge."
        ),
        FailureCategory.REGIME_MISMATCH: (
            "Trade was taken in a market regime that did not suit the strategy."
        ),
        FailureCategory.POOR_ENTRY: (
            "Entry was technically valid but occurred at an unfavorable "
            "location (excessive extension, late entry, or poor risk/reward)."
        ),
        FailureCategory.EXECUTION_ERROR: (
            "Trade suffered from execution issues (spread/slippage)."
        ),
        FailureCategory.RISK_MISMANAGEMENT: (
            "Risk was not properly managed on this trade."
        ),
        FailureCategory.RULE_VIOLATION: (
            "Trade did not satisfy the documented strategy/process rules."
        ),
        FailureCategory.DATA_QUALITY_FAILURE: (
            "The record contains missing, corrupted, or suspicious data; "
            "result is unreliable."
        ),
        FailureCategory.EMOTIONAL_TRADE: "Trade appears to be emotionally driven.",
        FailureCategory.MARKET_CONDITION: "Market conditions caused the loss.",
        FailureCategory.INVALID_SETUP: "Trade was entered with an invalid setup.",
        FailureCategory.UNKNOWN: "Unable to determine the cause of loss.",
    }
    return descs.get(category, "Unknown failure.")


def _historical_comparison(
    trade: TradeRecord, context: dict | None = None
) -> str:
    ctx = context or {}
    avg_r = ctx.get("historical_avg_r", 0.0)
    r = compute_r_multiple(trade)
    if r is not None and avg_r != 0:
        diff = r - avg_r
        if diff < -0.5:
            return f"R-multiple of {r:.2f} is below historical avg of {avg_r:.2f}."
    return "Insufficient historical data for comparison."


def _recommend_action(category: FailureCategory) -> str:
    actions = {
        FailureCategory.VALID_STRATEGY_LOSS: (
            "Normal variance; continue running the strategy. Do not change "
            "parameters based on a single trade."
        ),
        FailureCategory.SIGNAL_FAILURE: (
            "Review the signal definition; consider whether additional "
            "confirmation would improve reliability without overfitting."
        ),
        FailureCategory.REGIME_MISMATCH: (
            "Add regime filter to entry conditions after gathering sufficient "
            "out-of-sample evidence."
        ),
        FailureCategory.POOR_ENTRY: (
            "Review entry criteria for excessive extension; consider entry "
            "timing (e.g., wait for pullback)."
        ),
        FailureCategory.EXECUTION_ERROR: (
            "Consider using limit orders and avoid high-spread sessions."
        ),
        FailureCategory.RISK_MISMANAGEMENT: (
            "Always honor stop losses and position sizing rules."
        ),
        FailureCategory.RULE_VIOLATION: (
            "Enforce checklist compliance; investigate process failure."
        ),
        FailureCategory.DATA_QUALITY_FAILURE: (
            "Investigate the data source; do not draw conclusions from "
            "unreliable records."
        ),
        FailureCategory.EMOTIONAL_TRADE: (
            "Implement a cooling-off period before placing trades."
        ),
        FailureCategory.MARKET_CONDITION: (
            "Reduce position size during volatile conditions."
        ),
        FailureCategory.INVALID_SETUP: (
            "Review entry criteria and ensure all conditions are met."
        ),
        FailureCategory.UNKNOWN: "Review trade journal for additional context.",
    }
    return actions.get(category, "Review trade journal.")


def _map_verdict(category: FailureCategory) -> str:
    mapping = {
        FailureCategory.VALID_STRATEGY_LOSS: "Valid loss",
        FailureCategory.SIGNAL_FAILURE: "Strategy failure",
        FailureCategory.REGIME_MISMATCH: "Regime mismatch",
        FailureCategory.POOR_ENTRY: "Strategy failure",
        FailureCategory.EXECUTION_ERROR: "Execution failure",
        FailureCategory.RISK_MISMANAGEMENT: "Risk-management failure",
        FailureCategory.RULE_VIOLATION: "Rule violation",
        FailureCategory.DATA_QUALITY_FAILURE: "Data issue",
        FailureCategory.EMOTIONAL_TRADE: "Rule violation",
        FailureCategory.MARKET_CONDITION: "Valid loss",
        FailureCategory.INVALID_SETUP: "Strategy failure",
        FailureCategory.UNKNOWN: "Uncertain",
    }
    return mapping.get(category, "Uncertain")


def _research_implication(category: FailureCategory) -> str:
    mapping = {
        FailureCategory.VALID_STRATEGY_LOSS: "Change nothing.",
        FailureCategory.SIGNAL_FAILURE: "Monitor; consider whether signal definition needs review.",
        FailureCategory.REGIME_MISMATCH: "Contribute to broader hypothesis; do not act on single trade.",
        FailureCategory.POOR_ENTRY: "Trigger deeper investigation into entry timing.",
        FailureCategory.EXECUTION_ERROR: "Monitor execution costs; investigate broker.",
        FailureCategory.RISK_MISMANAGEMENT: "Trigger deeper investigation into process compliance.",
        FailureCategory.RULE_VIOLATION: "Trigger deeper investigation; process failure, not strategy.",
        FailureCategory.DATA_QUALITY_FAILURE: "Investigate data pipeline.",
        FailureCategory.EMOTIONAL_TRADE: "Trigger investigation into trader process.",
        FailureCategory.MARKET_CONDITION: "Change nothing.",
        FailureCategory.INVALID_SETUP: "Contribute to broader hypothesis.",
        FailureCategory.UNKNOWN: "Monitor; insufficient evidence.",
    }
    return mapping.get(category, "Monitor.")


def _gather_evidence(trade: TradeRecord, context: dict | None = None) -> list[str]:
    evidence = []
    r = compute_r_multiple(trade)
    if r is not None:
        evidence.append(f"R-multiple: {r:.2f}")
    if trade.mfe is not None:
        evidence.append(f"MFE: {trade.mfe:.5f}")
    if trade.mae is not None:
        evidence.append(f"MAE: {trade.mae:.5f}")
    if trade.spread_at_entry > 0:
        evidence.append(f"Spread at entry: {trade.spread_at_entry} pips")
    if trade.slippage_pips != 0:
        evidence.append(f"Slippage: {trade.slippage_pips} pips")
    if trade.exit_reason:
        evidence.append(f"Exit reason: {trade.exit_reason}")
    if trade.regime:
        evidence.append(f"Entry regime: {trade.regime.value}")
    ctx = context or {}
    if ctx.get("historical_avg_r") is not None:
        evidence.append(f"Historical avg R: {ctx['historical_avg_r']:.2f}")
    return evidence


def _has_data_quality_issue(trade: TradeRecord) -> bool:
    if trade.entry_price <= 0:
        return True
    if trade.stop_loss and trade.stop_loss <= 0:
        return True
    if trade.exit_price is not None and trade.exit_price <= 0:
        return True
    if trade.entry_time and trade.exit_time and trade.exit_time < trade.entry_time:
        return True
    return False


def _is_poor_entry(trade: TradeRecord, ctx: dict) -> bool:
    if ctx.get("poor_entry", False):
        return True
    rr = ctx.get("entry_rr")
    if rr is not None and rr < 1.0:
        return True
    return False
