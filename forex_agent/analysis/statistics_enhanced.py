from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from forex_agent.data.schemas import TradeRecord
from forex_agent.data.ingestion import compute_r_values


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------

def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic_fn: Any = None,
    n_bootstrap: int = 5000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Compute a bootstrap confidence interval for any statistic.

    Parameters
    ----------
    data : array of observed values
    statistic_fn : callable that takes a 1-D array and returns a scalar
        (default: np.mean)
    n_bootstrap : number of bootstrap resamples
    confidence_level : CI level (e.g. 0.95 for 95%)
    seed : RNG seed for reproducibility

    Returns
    -------
    dict with keys: statistic, ci_lower, ci_upper, se
    """
    if statistic_fn is None:
        statistic_fn = np.mean
    if len(data) < 2:
        val = float(statistic_fn(data)) if len(data) > 0 else 0.0
        return {"statistic": val, "ci_lower": val, "ci_upper": val, "se": 0.0}

    rng = np.random.default_rng(seed)
    observed = float(statistic_fn(data))

    boot_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_stats[i] = statistic_fn(sample)

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    se = float(np.std(boot_stats, ddof=1))

    return {
        "statistic": observed,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "se": se,
    }


def bootstrap_expectancy_ci(
    trades: list[TradeRecord],
    n_bootstrap: int = 5000,
    confidence_level: float = 0.95,
) -> dict[str, float]:
    """Bootstrap CI for mean R-multiple (expectancy)."""
    r_vals = np.array(compute_r_values(trades))
    if len(r_vals) < 2:
        val = float(np.mean(r_vals)) if len(r_vals) > 0 else 0.0
        return {"statistic": val, "ci_lower": val, "ci_upper": val, "se": 0.0}
    return bootstrap_confidence_interval(r_vals, np.mean, n_bootstrap, confidence_level)


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Two-sample permutation test for difference in means.

    Returns
    -------
    dict with keys: observed_diff, p_value, effect_size (Cohen's d)
    """
    if len(a) < 2 or len(b) < 2:
        return {"observed_diff": 0.0, "p_value": 1.0, "effect_size": 0.0}

    rng = np.random.default_rng(seed)
    observed_diff = float(np.mean(a) - np.mean(b))
    combined = np.concatenate([a, b])
    n_a = len(a)
    count = 0

    for _ in range(n_permutations):
        perm = rng.permutation(combined)
        perm_diff = float(np.mean(perm[:n_a]) - np.mean(perm[n_a:]))
        if abs(perm_diff) >= abs(observed_diff):
            count += 1

    p_value = count / n_permutations
    effect = _cohens_d(a, b)

    return {
        "observed_diff": observed_diff,
        "p_value": p_value,
        "effect_size": effect,
    }


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------

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


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """Hedges' g - bias-corrected Cohen's d for small samples."""
    d = _cohens_d(a, b)
    n = len(a) + len(b)
    if n <= 2:
        return d
    correction = 1.0 - (3.0 / (4.0 * (n - 2) - 1.0))
    return d * correction


def rank_biserial_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Rank-biserial r for non-parametric effect size."""
    if len(a) < 1 or len(b) < 1:
        return 0.0
    combined = np.concatenate([a, b])
    ranks = np.empty_like(combined)
    sorted_idx = np.argsort(combined)
    ranks[sorted_idx] = np.arange(1, len(combined) + 1, dtype=float)
    mean_rank_a = float(np.mean(ranks[:len(a)]))
    mean_rank_b = float(np.mean(ranks[len(a):]))
    n = len(combined)
    return 2.0 * (mean_rank_a - mean_rank_b) / n


# ---------------------------------------------------------------------------
# Multiple-testing correction
# ---------------------------------------------------------------------------

def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[dict[str, Any]]:
    """Benjamini-Hochberg FDR correction.

    Returns a list of dicts with original index, p_value, adjusted_p, and
    whether the test is significant at the given alpha level.
    """
    n = len(p_values)
    if n == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    adjusted[indexed[-1][0]] = indexed[-1][1]

    for i in range(n - 2, -1, -1):
        orig_idx, p = indexed[i]
        rank = i + 1
        adj = min(p * n / rank, 1.0)
        next_orig_idx = indexed[i + 1][0]
        adjusted[orig_idx] = min(adj, adjusted[next_orig_idx])

    return [
        {
            "index": orig_idx,
            "p_value": p,
            "adjusted_p": adjusted[orig_idx],
            "significant": adjusted[orig_idx] < alpha,
        }
        for orig_idx, p in indexed
    ]


def bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> list[dict[str, Any]]:
    """Bonferroni correction for multiple comparisons."""
    n = len(p_values)
    if n == 0:
        return []
    return [
        {
            "index": i,
            "p_value": p,
            "adjusted_p": min(p * n, 1.0),
            "significant": (p * n) < alpha,
        }
        for i, p in enumerate(p_values)
    ]


def minimum_sample_size(
    expected_effect: float,
    std_dev: float,
    power: float = 0.80,
    alpha: float = 0.05,
) -> int:
    """Minimum sample size for detecting an effect with given power.

    Uses the normal approximation (two-sided z-test).
    """
    if expected_effect == 0 or std_dev == 0:
        return 0
    from scipy import stats  # type: ignore[import-untyped]
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n = ((z_alpha + z_beta) * std_dev / expected_effect) ** 2
    return int(math.ceil(n))
