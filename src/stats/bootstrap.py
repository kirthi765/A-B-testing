"""Percentile-CI bootstrap on the difference of an arbitrary statistic.

Default statistic is the mean — same hypothesis as Welch's t-test, just
non-parametric. Use this when the metric is heavy-tailed (latency, revenue
per user) or has bounded support that makes the normal-approx CI suspect.

The percentile p-value reported here is `2 * min(P(boot_diff <= 0),
P(boot_diff >= 0))` — i.e. the smallest two-sided level at which the
percentile CI would exclude zero. Strictly speaking this is *not* a
permutation p-value; it's a "duality with the CI" p-value. Reasonable for
A/B reporting; for high-stakes inference, prefer a permutation test or a
studentized bootstrap.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ._result import TestResult


def bootstrap_diff_of_means(
    df: pd.DataFrame,
    *,
    variant_col: str,
    metric_col: str,
    control: str = "control",
    treatment: str = "treatment",
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> TestResult:
    """Bootstrap CI for `mean(treatment) - mean(control)`.

    Vectorized: allocates an `(n_resamples, n_arm)` int32 index matrix per arm.
    At n=10k users x n_resamples=10k that's ~800MB in float64-of-values terms;
    we keep memory down by indexing into the original float array rather than
    materializing per-resample copies.
    """
    return _bootstrap_diff(
        df,
        variant_col=variant_col,
        metric_col=metric_col,
        control=control,
        treatment=treatment,
        statistic=np.mean,
        n_resamples=n_resamples,
        alpha=alpha,
        seed=seed,
        method_name="bootstrap_diff_of_means",
    )


def bootstrap_diff(
    df: pd.DataFrame,
    *,
    variant_col: str,
    metric_col: str,
    statistic: Callable[[np.ndarray], float],
    control: str = "control",
    treatment: str = "treatment",
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> TestResult:
    """Bootstrap CI for an arbitrary `statistic(arr) -> float` per arm.

    The statistic must accept a 1-D numpy array and return a scalar. Examples:
    `np.median`, `np.std`, or a custom lambda for quantile-based metrics.
    """
    return _bootstrap_diff(
        df,
        variant_col=variant_col,
        metric_col=metric_col,
        control=control,
        treatment=treatment,
        statistic=statistic,
        n_resamples=n_resamples,
        alpha=alpha,
        seed=seed,
        method_name=f"bootstrap_diff[{getattr(statistic, '__name__', 'statistic')}]",
    )


def _bootstrap_diff(
    df: pd.DataFrame,
    *,
    variant_col: str,
    metric_col: str,
    control: str,
    treatment: str,
    statistic: Callable[[np.ndarray], float],
    n_resamples: int,
    alpha: float,
    seed: int,
    method_name: str,
) -> TestResult:
    rng = np.random.default_rng(seed)
    x_c = df.loc[df[variant_col] == control, metric_col].dropna().to_numpy()
    x_t = df.loc[df[variant_col] == treatment, metric_col].dropna().to_numpy()
    n_c, n_t = len(x_c), len(x_t)
    if n_c < 2 or n_t < 2:
        raise ValueError(f"need at least 2 obs per arm; got n_c={n_c}, n_t={n_t}")

    obs_diff = float(statistic(x_t)) - float(statistic(x_c))

    # If the statistic is np.mean, we can use the fully vectorized form for ~30x speedup.
    if statistic is np.mean:
        idx_c = rng.integers(0, n_c, size=(n_resamples, n_c))
        idx_t = rng.integers(0, n_t, size=(n_resamples, n_t))
        boot_diffs = x_t[idx_t].mean(axis=1) - x_c[idx_c].mean(axis=1)
    else:
        boot_diffs = np.empty(n_resamples)
        for i in range(n_resamples):
            c_sample = rng.choice(x_c, size=n_c, replace=True)
            t_sample = rng.choice(x_t, size=n_t, replace=True)
            boot_diffs[i] = statistic(t_sample) - statistic(c_sample)

    lo = float(np.quantile(boot_diffs, alpha / 2))
    hi = float(np.quantile(boot_diffs, 1 - alpha / 2))
    # Two-sided CI-dual p-value, lower-bounded by 1/n_resamples to avoid p=0.
    p_below = float((boot_diffs <= 0).mean())
    p_above = float((boot_diffs >= 0).mean())
    p_value = max(2 * min(p_below, p_above), 1 / n_resamples)

    return TestResult(
        point_estimate=obs_diff,
        ci_low=lo,
        ci_high=hi,
        p_value=p_value,
        method_name=method_name,
        metadata={
            "n_control": n_c,
            "n_treatment": n_t,
            "n_resamples": n_resamples,
            "boot_se": float(boot_diffs.std(ddof=1)),
        },
    )
