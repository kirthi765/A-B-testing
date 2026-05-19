"""Welch's t-test (continuous metrics) and the two-proportion z-test (binary).

Both functions take `(df, variant_col, metric_col)` and return a `TestResult`
for `treatment - control` on the absolute scale. NaNs in the metric column
are dropped silently; users with no events upstream show up as NaN in
`fct_experiment_metrics.conversion_rate` and would otherwise blow up the
variance calc.

The z-test uses *pooled* variance for the p-value (the textbook test
statistic under H0: p_T = p_C) and *unpooled* variance for the CI (which
estimates the SE *given* the observed proportions). This split is standard
and avoids the well-known mismatch where the test rejects but the CI covers
zero, or vice versa.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from ._result import TestResult


def welch_t_test(
    df: pd.DataFrame,
    *,
    variant_col: str,
    metric_col: str,
    control: str = "control",
    treatment: str = "treatment",
    alpha: float = 0.05,
) -> TestResult:
    """Welch's two-sample t-test for unequal variances."""
    x_c = df.loc[df[variant_col] == control, metric_col].dropna().to_numpy()
    x_t = df.loc[df[variant_col] == treatment, metric_col].dropna().to_numpy()
    n_c, n_t = len(x_c), len(x_t)
    if n_c < 2 or n_t < 2:
        raise ValueError(f"need at least 2 obs per arm; got n_c={n_c}, n_t={n_t}")

    mean_c, mean_t = float(x_c.mean()), float(x_t.mean())
    var_c, var_t = float(x_c.var(ddof=1)), float(x_t.var(ddof=1))
    diff = mean_t - mean_c
    se = math.sqrt(var_t / n_t + var_c / n_c)

    # Welch–Satterthwaite degrees of freedom.
    df_w = (var_t / n_t + var_c / n_c) ** 2 / (
        (var_t / n_t) ** 2 / (n_t - 1) + (var_c / n_c) ** 2 / (n_c - 1)
    )
    t_stat = diff / se if se > 0 else 0.0
    p_value = float(2 * stats.t.sf(abs(t_stat), df_w))
    t_crit = float(stats.t.ppf(1 - alpha / 2, df_w))

    return TestResult(
        point_estimate=diff,
        ci_low=diff - t_crit * se,
        ci_high=diff + t_crit * se,
        p_value=p_value,
        method_name="welch_t_test",
        metadata={
            "se": se,
            "df": float(df_w),
            "t_stat": float(t_stat),
            "n_control": n_c,
            "n_treatment": n_t,
            "mean_control": mean_c,
            "mean_treatment": mean_t,
        },
    )


def two_proportion_z_test(
    df: pd.DataFrame,
    *,
    variant_col: str,
    metric_col: str,
    control: str = "control",
    treatment: str = "treatment",
    alpha: float = 0.05,
) -> TestResult:
    """Two-sample z-test for proportions. `metric_col` must be 0/1 (or bool)."""
    c = df.loc[df[variant_col] == control, metric_col].dropna().astype(int).to_numpy()
    t = df.loc[df[variant_col] == treatment, metric_col].dropna().astype(int).to_numpy()
    n_c, n_t = len(c), len(t)
    if n_c < 1 or n_t < 1:
        raise ValueError(f"need at least 1 obs per arm; got n_c={n_c}, n_t={n_t}")

    k_c, k_t = int(c.sum()), int(t.sum())
    p_c, p_t = k_c / n_c, k_t / n_t
    p_pool = (k_c + k_t) / (n_c + n_t)
    diff = p_t - p_c

    se_pool = math.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
    se_unpool = math.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)

    z_stat = diff / se_pool if se_pool > 0 else 0.0
    p_value = float(2 * stats.norm.sf(abs(z_stat)))
    z_crit = float(stats.norm.ppf(1 - alpha / 2))

    return TestResult(
        point_estimate=diff,
        ci_low=diff - z_crit * se_unpool,
        ci_high=diff + z_crit * se_unpool,
        p_value=p_value,
        method_name="two_proportion_z_test",
        metadata={
            "se_unpooled": se_unpool,
            "se_pooled": se_pool,
            "z_stat": float(z_stat),
            "n_control": n_c,
            "n_treatment": n_t,
            "p_control": p_c,
            "p_treatment": p_t,
            "p_pooled": p_pool,
        },
    )
