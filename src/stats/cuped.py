"""CUPED — variance reduction via a pre-experiment covariate.

(Deng, Xu, Kohavi, Walker 2013 — "Improving the Sensitivity of Online
Controlled Experiments by Utilizing Pre-Experiment Data".)

Given a per-user post-period metric `y` and a pre-experiment covariate `x`
(typically the same metric measured during the pre-period), CUPED constructs
an adjusted metric

    y' = y - θ · (x - mean(x))     with   θ = Cov(y, x) / Var(x)

and runs the standard t-test on `y'` instead of `y`. The treatment effect
estimate is unchanged in expectation (because `x` is balanced between arms
by random assignment), but `Var(y') = Var(y) · (1 - ρ²)` where `ρ = corr(y, x)`.
A pre-period covariate with `ρ = 0.7` shrinks the SE by ~30% — the
moral equivalent of 2x'ing the sample size for free.

We fit θ on the *pooled* (control + treatment) data to avoid using the
treatment indicator in the adjustment, which would bias the effect estimate
toward zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._result import TestResult
from .frequentist import welch_t_test


def cuped_theta(y: np.ndarray, x: np.ndarray) -> float:
    """OLS slope of `y ~ x`. Fit on pooled (both-arm) data."""
    var_x = float(x.var(ddof=1))
    if var_x == 0:
        return 0.0
    return float(np.cov(x, y, ddof=1)[0, 1]) / var_x


def cuped_adjust(
    df: pd.DataFrame,
    *,
    metric_col: str,
    covariate_col: str,
) -> pd.Series:
    """Return the CUPED-adjusted metric `y' = y - θ·(x - x̄)`."""
    y = df[metric_col].to_numpy(dtype=float)
    x = df[covariate_col].to_numpy(dtype=float)
    theta = cuped_theta(y, x)
    return pd.Series(y - theta * (x - x.mean()), index=df.index, name=f"{metric_col}_cuped")


def cuped_t_test(
    df: pd.DataFrame,
    *,
    variant_col: str,
    metric_col: str,
    covariate_col: str,
    control: str = "control",
    treatment: str = "treatment",
    alpha: float = 0.05,
) -> TestResult:
    """CUPED-adjusted Welch's t-test, with the variance reduction reported in metadata."""
    y = df[metric_col].to_numpy(dtype=float)
    x = df[covariate_col].to_numpy(dtype=float)
    theta = cuped_theta(y, x)

    adjusted = pd.Series(y - theta * (x - x.mean()), index=df.index)
    df2 = df.assign(_cuped_y=adjusted)
    result = welch_t_test(
        df2,
        variant_col=variant_col,
        metric_col="_cuped_y",
        control=control,
        treatment=treatment,
        alpha=alpha,
    )

    var_y = float(y.var(ddof=1))
    var_y_adj = float(adjusted.to_numpy().var(ddof=1))
    var_reduction = 1.0 - var_y_adj / var_y if var_y > 0 else 0.0

    return TestResult(
        point_estimate=result.point_estimate,
        ci_low=result.ci_low,
        ci_high=result.ci_high,
        p_value=result.p_value,
        method_name="cuped_t_test",
        metadata={
            **result.metadata,
            "theta": theta,
            "variance_reduction": var_reduction,
            "var_y": var_y,
            "var_y_adjusted": var_y_adj,
        },
    )
