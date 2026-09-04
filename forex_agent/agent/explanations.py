from __future__ import annotations

from typing import Any

from forex_agent.data.schemas import (
    ConfidenceLevel,
    FailureCategory,
    TradeFailureAnalysis,
    TradeRecord,
)


def generate_trade_explanation(
    trade: TradeRecord,
    analysis: TradeFailureAnalysis,
) -> str:
    sections: list[str] = []

    sections.append(f"TRADE SUMMARY: {trade.trade_id}")
    sections.append(f"  Pair: {trade.symbol} | Direction: {trade.direction}")
    sections.append(
        f"  Entry: {trade.entry_price:.5f} | SL: {trade.stop_loss:.5f} | "
        f"TP: {trade.take_profit:.5f}" if trade.take_profit is not None else
        f"  Entry: {trade.entry_price:.5f} | SL: {trade.stop_loss:.5f}"
    )
    if trade.exit_price is not None:
        sections.append(f"  Exit: {trade.exit_price:.5f} | P&L: ${trade.pnl:.2f}")
    sections.append(f"  Result: {'WIN' if trade.pnl > 0 else 'LOSS'}")

    sections.append("")
    sections.append("WHAT THE STRATEGY EXPECTED:")
    direction_text = "bullish" if trade.direction == "LONG" else "bearish"
    sections.append(
        f"  The entry targeted a {direction_text} move in {trade.symbol}."
    )
    if trade.take_profit is not None and trade.stop_loss is not None:
        if trade.direction == "LONG":
            rr = abs(trade.take_profit - trade.entry_price) / max(abs(trade.entry_price - trade.stop_loss), 0.0001)
        else:
            rr = abs(trade.entry_price - trade.take_profit) / max(abs(trade.stop_loss - trade.entry_price), 0.0001)
        sections.append(f"  Planned R:R = 1:{rr:.1f}")
    if trade.notes:
        sections.append(f"  Notes: {trade.notes}")

    sections.append("")
    sections.append("WHAT ACTUALLY HAPPENED:")
    if trade.exit_price is not None:
        if trade.direction == "LONG":
            move = trade.exit_price - trade.entry_price
        else:
            move = trade.entry_price - trade.exit_price
        if move < 0:
            sections.append(f"  Price moved against the position by {abs(move):.5f}")
        else:
            sections.append(f"  Price moved in favor by {move:.5f}")

    sections.append("")
    sections.append("FAILURE CLASSIFICATION:")
    sections.append(f"  Category: {analysis.failure_category.value.upper()}")
    sections.append(f"  Confidence: {analysis.confidence.value.upper()}")
    sections.append(f"  {analysis.description}")

    if analysis.contributing_factors:
        sections.append("")
        sections.append("EVIDENCE:")
        for factor in analysis.contributing_factors:
            sections.append(f"  - {factor}")

    if analysis.counterfactual:
        sections.append("")
        sections.append("COUNTERFACTUALS:")
        sections.append(f"  {analysis.counterfactual}")

    if analysis.historical_comparison:
        sections.append("")
        sections.append("HISTORICAL COMPARISON:")
        sections.append(f"  {analysis.historical_comparison}")

    verdict = _determine_verdict(analysis)
    sections.append("")
    sections.append(f"VERDICT: {verdict}")

    sections.append("")
    sections.append("RESEARCH IMPLICATION:")
    sections.append(f"  {analysis.recommended_action}")

    return "\n".join(sections)


def generate_failure_summary(failures: list[TradeFailureAnalysis]) -> str:
    if not failures:
        return "No failures to analyze."

    total = len(failures)
    category_counts: dict[str, int] = {}
    confidence_counts: dict[str, int] = {}

    for f in failures:
        cat = f.failure_category.value
        category_counts[cat] = category_counts.get(cat, 0) + 1
        conf = f.confidence.value
        confidence_counts[conf] = confidence_counts.get(conf, 0) + 1

    lines: list[str] = []
    lines.append(f"FAILURE SUMMARY: {total} losing trades analyzed")
    lines.append("")
    lines.append("BY CATEGORY:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        pct = count / total
        lines.append(f"  {cat:.<30} {count:>3} ({pct:.0%})")

    lines.append("")
    lines.append("BY CONFIDENCE:")
    for conf, count in sorted(confidence_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {conf:.<30} {count:>3}")

    high_conf = [f for f in failures if f.confidence == ConfidenceLevel.HIGH]
    if high_conf:
        lines.append("")
        lines.append("HIGH-CONFIDENCE FINDINGS:")
        for f in high_conf[:5]:
            lines.append(f"  [{f.trade_id}] {f.failure_category.value}: {f.description}")

    action_counts: dict[str, int] = {}
    for f in failures:
        action_counts[f.recommended_action] = action_counts.get(f.recommended_action, 0) + 1

    lines.append("")
    lines.append("RECOMMENDED ACTIONS:")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1])[:5]:
        lines.append(f"  [{count}x] {action}")

    return "\n".join(lines)


def generate_research_hypotheses(patterns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hypotheses: list[dict[str, Any]] = []

    for pattern in patterns:
        pattern_text = pattern.get("pattern", "")
        count = pattern.get("count", 0)
        confidence = pattern.get("confidence", "low")

        if "session" in pattern_text.lower():
            session = "unknown"
            for word in ["asian", "london", "new_york", "off_hours"]:
                if word in pattern_text.lower():
                    session = word
                    break
            hypotheses.append({
                "hypothesis": f"Strategy edge is weaker during {session} session",
                "reason": f"High loss rate ({pattern.get('loss_rate', 0):.0%}) observed over {count} trades",
                "dataset": "All closed trades with session classification",
                "control_group": f"Trades outside {session} session",
                "treatment_group": f"Trades during {session} session",
                "metric": "Expectancy (R-multiple)",
                "min_sample_size": 30,
                "statistical_test": "Welch t-test on R-multiples",
                "acceptance_criteria": "p < 0.05 and effect size > 0.3R difference",
                "overfitting_risk": "low - session is pre-defined, not fitted",
                "out_of_sample_plan": "Test on next 30 trades; validate with regime-aware split",
            })

        elif "symbol" in pattern_text.lower() or any(
            pair in pattern_text for pair in ["EUR", "GBP", "USD", "JPY", "AUD"]
        ):
            symbol = "unknown"
            for part in pattern_text.split():
                if any(pair in part for pair in ["EUR", "GBP", "USD", "JPY", "AUD"]):
                    symbol = part
                    break
            hypotheses.append({
                "hypothesis": f"Strategy edge is weaker on {symbol}",
                "reason": f"High loss rate ({pattern.get('loss_rate', 0):.0%}) over {count} trades",
                "dataset": f"All {symbol} trades",
                "control_group": f"All non-{symbol} trades",
                "treatment_group": f"All {symbol} trades",
                "metric": "Expectancy (R-multiple)",
                "min_sample_size": 20,
                "statistical_test": "Welch t-test on R-multiples",
                "acceptance_criteria": "p < 0.05 and effect size > 0.3R difference",
                "overfitting_risk": "medium - pair selection may be data-mined",
                "out_of_sample_plan": "Validate on rolling 6-month windows",
            })

        elif "regime" in pattern_text.lower() or "execution" in pattern_text.lower():
            hypotheses.append({
                "hypothesis": f"Pattern '{pattern_text}' represents a systematic edge decay",
                "reason": f"Observed {count} times with {confidence} confidence",
                "dataset": "All closed trades",
                "control_group": "All trades outside the pattern condition",
                "treatment_group": "All trades matching the pattern condition",
                "metric": "Expectancy and win rate",
                "min_sample_size": 25,
                "statistical_test": "Welch t-test and chi-squared test",
                "acceptance_criteria": "p < 0.05 on both metrics",
                "overfitting_risk": "medium",
                "out_of_sample_plan": "Walk-forward validation on non-overlapping windows",
            })

    default_hypothesis = {
        "hypothesis": "Recent performance degradation is within normal variance",
        "reason": f"Analyzed {len(patterns)} patterns; need to rule out random variation",
        "dataset": "All closed trades split into halves",
        "control_group": "First half of trade history",
        "treatment_group": "Second half of trade history",
        "metric": "Expectancy (R-multiple)",
        "min_sample_size": 20,
        "statistical_test": "Welch t-test on R-multiples",
        "acceptance_criteria": "p < 0.05 with meaningful effect size",
        "overfitting_risk": "low - split is time-based, not fitted",
        "out_of_sample_plan": "Monitor next 20 trades for continuation",
    }
    if not hypotheses:
        hypotheses.append(default_hypothesis)

    return hypotheses


def _determine_verdict(analysis: TradeFailureAnalysis) -> str:
    mapping = {
        FailureCategory.UNKNOWN: "Uncertain",
        FailureCategory.EXECUTION_ERROR: "Execution failure",
        FailureCategory.REGIME_MISMATCH: "Regime mismatch",
        FailureCategory.RISK_MISMANAGEMENT: "Risk-management failure",
        FailureCategory.EMOTIONAL_TRADE: "Rule violation",
        FailureCategory.INVALID_SETUP: "Strategy failure",
        FailureCategory.MARKET_CONDITION: "Valid loss",
    }
    return mapping.get(analysis.failure_category, "Uncertain")
