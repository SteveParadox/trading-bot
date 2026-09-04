from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from forex_agent.data.ingestion import compute_r_values
from forex_agent.data.schemas import TradeRecord


def _get_r_values(trades: list[TradeRecord]) -> np.ndarray:
    """R-multiples for a batch of trades via the canonical helper.

    Skips open trades (no exit) and trades with no risk denominator.
    """
    return np.array(compute_r_values(trades))


@dataclass
class Experiment:
    hypothesis: str
    reason: str
    dataset_description: str
    control_group: str
    treatment_group: str
    metric: str
    min_sample_size: int
    statistical_test: str
    acceptance_criteria: str
    rejection_criteria: str
    overfitting_risk: str
    out_of_sample_plan: str


@dataclass
class ExperimentResult:
    hypothesis: str
    supported: bool
    evidence: str
    sample_size: int
    p_value: float
    effect_size: float
    confidence: str
    warning: str


def _welch_t_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if len(a) < 2 or len(b) < 2:
        return 0.0, 1.0
    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    n_a = len(a)
    n_b = len(b)

    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se == 0:
        return 0.0, 1.0

    t_stat = (mean_a - mean_b) / se
    num = (var_a / n_a + var_b / n_b) ** 2
    denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = num / denom if denom > 0 else 1.0

    p_value = 2.0 * (1.0 - _t_cdf(abs(t_stat), df))
    return t_stat, p_value


def _t_cdf(t: float, df: float) -> float:
    x = df / (df + t * t)
    if t >= 0:
        return 1.0 - 0.5 * _beta_inc(df / 2.0, 0.5, x)
    return 0.5 * _beta_inc(df / 2.0, 0.5, x)


def _beta_inc(a: float, b: float, x: float) -> float:
    if x < 0 or x > 1:
        return 0.0
    if x == 0 or x == 1:
        return x

    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )

    if x < (a + 1) / (a + b + 2):
        return bt * _beta_cf(a, b, x) / a
    return 1.0 - bt * _beta_cf(b, a, 1.0 - x) / b


def _beta_cf(a: float, b: float, x: float) -> float:
    max_iter = 200
    eps = 1e-10

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0

    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < eps:
        d = eps
    d = 1.0 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < eps:
            d = eps
        c = 1.0 + aa / c
        if abs(c) < eps:
            c = eps
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < eps:
            d = eps
        c = 1.0 + aa / c
        if abs(c) < eps:
            c = eps
        d = 1.0 / d
        delta = d * c
        h *= delta

        if abs(delta - 1.0) < eps:
            break

    return h


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled_std = math.sqrt(
        ((len(a) - 1) * float(np.var(a, ddof=1)) + (len(b) - 1) * float(np.var(b, ddof=1)))
        / (len(a) + len(b) - 2)
    )
    if pooled_std == 0:
        return 0.0
    return (float(np.mean(a)) - float(np.mean(b))) / pooled_std


def propose_experiments(analysis_report: Any) -> list[Experiment]:
    experiments: list[Experiment] = []

    patterns = getattr(analysis_report, "recurring_patterns", [])
    for pattern in patterns:
        pattern_text = pattern.get("pattern", "")
        confidence = pattern.get("confidence", "low")
        count = pattern.get("count", 0)

        if "session" in pattern_text.lower():
            session = "unknown"
            for word in ["asian", "london", "new_york", "off_hours"]:
                if word in pattern_text.lower():
                    session = word
                    break
            experiments.append(Experiment(
                hypothesis=f"Strategy edge is weaker during {session} session",
                reason=f"Pattern '{pattern_text}' observed {count} times with {confidence} confidence",
                dataset_description="All closed trades with session classification",
                control_group=f"Trades outside {session} session",
                treatment_group=f"Trades during {session} session",
                metric="Expectancy (R-multiple)",
                min_sample_size=30,
                statistical_test="Welch t-test",
                acceptance_criteria="p < 0.05 and Cohen's d > 0.3",
                rejection_criteria="p >= 0.10 or effect size < 0.1",
                overfitting_risk="low - session is pre-defined",
                out_of_sample_plan="Validate on next 30 trades; retest quarterly",
            ))

        elif "symbol" in pattern_text.lower() or any(
            pair in pattern_text for pair in ["EUR", "GBP", "USD", "JPY", "AUD"]
        ):
            symbol = "unknown"
            for part in pattern_text.split():
                if any(pair in part for pair in ["EUR", "GBP", "USD", "JPY", "AUD"]):
                    symbol = part
                    break
            experiments.append(Experiment(
                hypothesis=f"Strategy lacks edge on {symbol}",
                reason=f"High loss rate observed on {symbol} over {count} trades",
                dataset_description=f"All {symbol} trades",
                control_group=f"Trades on other symbols",
                treatment_group=f"Trades on {symbol}",
                metric="Expectancy (R-multiple)",
                min_sample_size=20,
                statistical_test="Welch t-test",
                acceptance_criteria="p < 0.05 and mean difference > 0.2R",
                rejection_criteria="p >= 0.10",
                overfitting_risk="medium - symbol selection may be data-mined",
                out_of_sample_plan="Walk-forward on non-overlapping 2-month windows",
            ))

    if not experiments:
        metrics = getattr(analysis_report, "metrics", None)
        if metrics is not None:
            wr = getattr(metrics, "win_rate", 0.0)
            exp_val = getattr(metrics, "expectancy", 0.0)
            if exp_val < 0 or wr < 0.4:
                experiments.append(Experiment(
                    hypothesis="Strategy edge has degraded compared to historical baseline",
                    reason=f"Current expectancy ({exp_val:.2f}R) and win rate ({wr:.0%}) suggest edge decay",
                    dataset_description="All closed trades, split by time",
                    control_group="First half of trade history",
                    treatment_group="Second half of trade history",
                    metric="Expectancy (R-multiple)",
                    min_sample_size=20,
                    statistical_test="Welch t-test on R-multiples",
                    acceptance_criteria="p < 0.05 with Cohen's d > 0.3",
                    rejection_criteria="p >= 0.10 or effect size < 0.1",
                    overfitting_risk="low - time-based split is non-fitted",
                    out_of_sample_plan="Monitor next 20 trades for trend continuation",
                ))

    return experiments


def evaluate_experiment_hypothesis(
    hypothesis: str,
    trades: list[TradeRecord],
    regime_fn: Callable[[TradeRecord], str] | None = None,
) -> ExperimentResult:
    closed = [t for t in trades if t.exit_price is not None]
    if len(closed) < 10:
        return ExperimentResult(
            hypothesis=hypothesis,
            supported=False,
            evidence="Insufficient sample size for evaluation",
            sample_size=len(closed),
            p_value=1.0,
            effect_size=0.0,
            confidence="unknown",
            warning=f"Only {len(closed)} closed trades available; need at least 10",
        )

    r_values = _get_r_values(closed)
    if len(r_values) < 10:
        return ExperimentResult(
            hypothesis=hypothesis,
            supported=False,
            evidence="Insufficient R-multiple data",
            sample_size=len(r_values),
            p_value=1.0,
            effect_size=0.0,
            confidence="unknown",
            warning="Could not compute sufficient R-multiples",
        )

    mid = len(closed) // 2
    group_a = _get_r_values(closed[:mid])
    group_b = _get_r_values(closed[mid:])

    if len(group_a) < 3 or len(group_b) < 3:
        return ExperimentResult(
            hypothesis=hypothesis,
            supported=False,
            evidence="Groups too small for statistical comparison",
            sample_size=len(r_values),
            p_value=1.0,
            effect_size=0.0,
            confidence="unknown",
            warning="Split groups too small",
        )

    t_stat, p_value = _welch_t_test(group_a, group_b)
    effect = _cohens_d(group_a, group_b)

    mean_a = float(np.mean(group_a))
    mean_b = float(np.mean(group_b))

    if p_value < 0.05 and abs(effect) > 0.3:
        supported = True
        confidence = "high"
    elif p_value < 0.10 and abs(effect) > 0.2:
        supported = True
        confidence = "medium"
    else:
        supported = False
        confidence = "high" if p_value >= 0.10 else "medium"

    evidence = (
        f"Group A (n={len(group_a)}): mean={mean_a:.2f}R, "
        f"Group B (n={len(group_b)}): mean={mean_b:.2f}R, "
        f"t={t_stat:.2f}, Cohen's d={effect:.2f}"
    )

    warning = ""
    total = len(r_values)
    if total < 30:
        warning = f"Sample size ({total}) is small; results may not be robust"

    return ExperimentResult(
        hypothesis=hypothesis,
        supported=supported,
        evidence=evidence,
        sample_size=total,
        p_value=p_value,
        effect_size=effect,
        confidence=confidence,
        warning=warning,
    )
