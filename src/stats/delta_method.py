"""Delta-method CI for ratio metrics where unit of randomization ≠ unit of analysis.

Classic example: click-through rate measured as `total_clicks / total_impressions`
across all users in a variant. The user is the randomization unit, but the metric
is computed at the impression unit. Naively running a z-test on per-impression
rows over-counts heavy users and shrinks the SE — leading to inflated false
positive rates. The delta method corrects this by treating each user's
`(clicks_u, impressions_u)` pair as one bivariate observation and computing the
ratio's variance via linearization.

For a single arm with per-user `(y_u, x_u)`:
    R = mean(y) / mean(x)
    Var(R) ≈ (1/n) * (Var(y) - 2*R*Cov(x,y) + R²*Var(x)) / mean(x)²

(Deng, Knoblich, Lu 2017; "Applying the Delta Method in Metric Analytics".)
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from ._result import TestResult


def _ratio_variance(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Returns (ratio, var(ratio)) via the delta method for ratio of means."""
    n = len(y)
    if n < 2:
        raise ValueError("need at least 2 observations for the delta method")
    y_bar = float(y.mean())
    x_bar = float(x.mean())
    if x_bar == 0:
        return float("nan"), float("nan")
    r = y_bar / x_bar
    var_y = float(y.var(ddof=1))
    var_x = float(x.var(ddof=1))
    cov_xy = float(np.cov(x, y, ddof=1)[0, 1])
    var_r = (1.0 / n) * (var_y - 2.0 * r * cov_xy + r * r * var_x) / (x_bar * x_bar)
    return r, var_r


def delta_method_ratio(
    df: pd.DataFrame,
    *,
    variant_col: str,
    numerator_col: str,
    denominator_col: str,
    control: str = "control",
    treatment: str = "treatment",
    alpha: float = 0.05,
) -> TestResult:
    """Delta-method test for `mean(numerator)/mean(denominator)` between arms.

    `df` must be user-level: one row per randomization unit, with summed
    numerator and denominator across that user's events. (For impression-level
    rows, aggregate up to user first.)
    """
    sub_c = df[df[variant_col] == control]
    sub_t = df[df[variant_col] == treatment]
    y_c = sub_c[numerator_col].to_numpy(dtype=float)
    x_c = sub_c[denominator_col].to_numpy(dtype=float)
    y_t = sub_t[numerator_col].to_numpy(dtype=float)
    x_t = sub_t[denominator_col].to_numpy(dtype=float)

    r_c, var_r_c = _ratio_variance(y_c, x_c)
    r_t, var_r_t = _ratio_variance(y_t, x_t)
    diff = r_t - r_c
    var_diff = var_r_c + var_r_t  # independent arms
    se = math.sqrt(var_diff) if var_diff > 0 else 0.0
    z = diff / se if se > 0 else 0.0
    p_value = float(2 * stats.norm.sf(abs(z)))
    z_crit = float(stats.norm.ppf(1 - alpha / 2))

    return TestResult(
        point_estimate=diff,
        ci_low=diff - z_crit * se,
        ci_high=diff + z_crit * se,
        p_value=p_value,
        method_name="delta_method_ratio",
        metadata={
            "se": se,
            "z_stat": float(z),
            "n_control": len(y_c),
            "n_treatment": len(y_t),
            "ratio_control": r_c,
            "ratio_treatment": r_t,
            "var_ratio_control": var_r_c,
            "var_ratio_treatment": var_r_t,
        },
    )
