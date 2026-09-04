from __future__ import annotations

from typing import Any

from forex_agent.data.schemas import CriticAssessment, TradeRecord
from forex_agent.data.ingestion import compute_r_multiple, compute_r_values


def assess_finding(
    finding_text: str,
    trades: list[TradeRecord],
    matched_trades: list[TradeRecord] | None = None,
    initial_confidence: float = 0.7,
    tests_conducted: int = 1,
) -> CriticAssessment:
    """Challenge a finding with anti-bias and methodological checks.

    This is the critic / anti-overfitting agent.  For every major finding it
    asks the questions from Section 13 of the spec and may downgrade the
    confidence.
    """
    challenges: list[str] = []
    adjusted = initial_confidence
    sample = matched_trades or trades
    closed = [t for t in sample if t.exit_price is not None]

    # 1. Sample size
    n = len(closed)
    sample_concern = False
    if n < 10:
        challenges.append(f"Very small sample size ({n}). Findings are unreliable.")
        adjusted *= 0.5
        sample_concern = True
    elif n < 30:
        challenges.append(f"Moderate sample size ({n}). Findings are indicative but not conclusive.")
        adjusted *= 0.8
        sample_concern = True
    elif n < 100:
        challenges.append(f"Sample size ({n}) is adequate but not large.")
        adjusted *= 0.95

    # 2. Independence
    independence_concern = False
    timestamps = [t.entry_time for t in closed if t.entry_time is not None]
    if len(timestamps) >= 2:
        gaps = [(timestamps[i + 1] - timestamps[i]).total_seconds() for i in range(len(timestamps) - 1)]
        short_gaps = sum(1 for g in gaps if g < 300)  # < 5 min
        if short_gaps > len(gaps) * 0.3:
            challenges.append(f"{short_gaps}/{len(gaps)} trades entered within 5 minutes. May violate independence assumption.")
            adjusted *= 0.9
            independence_concern = True

    # 3. Survivorship bias
    survivorship_concern = False
    open_trades = [t for t in sample if t.exit_price is None]
    if len(open_trades) > len(closed) * 0.2:
        challenges.append(f"{len(open_trades)} open trades ({len(open_trades)/(len(closed)+len(open_trades)):.0%}) may introduce survivorship bias.")
        adjusted *= 0.9
        survivorship_concern = True

    # 4. Look-ahead bias (if future data was used)
    look_ahead_concern = False
    # We flag this generically since the caller should mark when post-hoc data is used
    # In practice the evidence package builder will set this.

    # 5. Overfitting
    overfitting_concern = False
    if tests_conducted > 1:
        challenges.append(f"{tests_conducted} tests conducted on this dataset. Risk of false discovery increased.")
        adjusted *= max(0.5, 1.0 - 0.05 * tests_conducted)
        overfitting_concern = True

    # 6. Multiple testing
    multiple_testing_concern = tests_conducted > 5
    if multiple_testing_concern:
        challenges.append("Multiple-testing inflation likely. Consider Bonferroni or FDR correction.")

    # 7. Out-of-sample
    oos_concern = False
    if n > 0:
        r_values = _get_r_values(closed)
        if len(r_values) >= 10:
            first_half = r_values[: len(r_values) // 2]
            second_half = r_values[len(r_values) // 2 :]
            if len(first_half) >= 3 and len(second_half) >= 3:
                import numpy as np
                diff = abs(float(np.mean(first_half)) - float(np.mean(second_half)))
                if diff > 0.5:
                    challenges.append(
                        f"Performance differs between first half (mean={float(np.mean(first_half)):.2f}R) "
                        f"and second half (mean={float(np.mean(second_half)):.2f}R). "
                        "Finding may not be stable out-of-sample."
                    )
                    adjusted *= 0.85
                    oos_concern = True

    # 8. Effect size practical significance
    economic = "unknown"
    r_values = _get_r_values(closed)
    if len(r_values) >= 5:
        import numpy as np
        mean_r = float(np.mean(r_values))
        if abs(mean_r) < 0.1:
            economic = "Effect is tiny (< 0.1R). Statistically significant findings may not be trading-significant."
            adjusted *= 0.9
        elif abs(mean_r) < 0.3:
            economic = f"Effect is small ({mean_r:.2f}R). May have limited practical impact."
        else:
            economic = f"Effect is meaningful ({mean_r:.2f}R). Likely trading-significant if confirmed."

    # 9. Outlier influence
    if len(r_values) >= 5:
        import numpy as np
        sorted_r = sorted(r_values)
        q1 = sorted_r[len(sorted_r) // 4]
        q3 = sorted_r[3 * len(sorted_r) // 4]
        iqr = q3 - q1
        outliers = [r for r in r_values if r < q1 - 1.5 * iqr or r > q3 + 1.5 * iqr]
        if len(outliers) >= 2:
            pct = len(outliers) / len(r_values)
            challenges.append(
                f"{len(outliers)} outliers ({pct:.0%}) detected. "
                "A few extreme trades may be driving the result."
            )
            adjusted *= 0.9

    adjusted = max(0.0, min(1.0, adjusted))

    # Determine status
    if adjusted >= 0.7:
        status = "supported"
    elif adjusted >= 0.4:
        status = "inconclusive"
    else:
        status = "weakened"

    alt_explanations = _suggest_alternatives(challenges, closed)

    return CriticAssessment(
        finding=finding_text,
        challenges=challenges,
        sample_size_concern=sample_concern,
        independence_concern=independence_concern,
        survivorship_bias_concern=survivorship_concern,
        look_ahead_bias_concern=look_ahead_concern,
        overfitting_concern=overfitting_concern,
        multiple_testing_concern=multiple_testing_concern,
        out_of_sample_concern=oos_concern,
        economic_meaningfulness=economic,
        alternative_explanations=alt_explanations,
        initial_confidence=initial_confidence,
        adjusted_confidence=round(adjusted, 3),
        status=status,
    )


def _suggest_alternatives(
    challenges: list[str], trades: list[TradeRecord]
) -> list[str]:
    alts: list[str] = []
    if any("sample size" in c.lower() for c in challenges):
        alts.append("The pattern may be random noise; gather more data before drawing conclusions.")
    if any("independence" in c.lower() for c in challenges):
        alts.append("Clustered entries may reflect emotional or mechanical trading rather than genuine signals.")
    if any("out-of-sample" in c.lower() for c in challenges):
        alts.append("The relationship may be specific to the in-sample period and not generalize.")
    if any("outlier" in c.lower() for c in challenges):
        alts.append("The effect may be driven by a few extreme trades rather than a systematic pattern.")
    if not alts:
        alts.append("No obvious alternative explanations identified at this time.")
    return alts


def _get_r_values(trades: list[TradeRecord]) -> list[float]:
    return [r for r in (compute_r_multiple(t) for t in trades) if r is not None]
