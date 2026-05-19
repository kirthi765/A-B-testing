"""CUPED tests.

CUPED is verified two ways:
  1. **Theory.** When `y = α·x + ε` with `Var(ε) = σ²ε`, the adjustment
     `y' = y - θ(x - x̄)` with `θ = α` gives `Var(y') = σ²ε`. So the variance
     reduction should equal `1 - σ²ε / Var(y) = α²·Var(x) / Var(y)`.
  2. **Power.** On the same simulated experiment, CUPED's p-value should be
     systematically smaller (more power) than plain Welch's t-test when `ρ`
     between `y` and `x` is non-trivial.

A correctness regression (e.g. fitting θ separately per arm, which is wrong)
would either bias the point estimate or fail the variance-reduction check.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.stats import cuped_adjust, cuped_t_test, cuped_theta, welch_t_test


def _simulate_pre_post(
    *,
    n: int,
    alpha_link: float,        # slope of y on x
    sigma_eps: float,         # residual SD
    sigma_x: float,           # pre-period SD
    true_lift: float,         # additive treatment effect on y
    seed: int,
) -> pd.DataFrame:
    """User-level dataset with pre-period covariate `x_pre` and post-period metric `y`."""
    rng = np.random.default_rng(seed)
    variant = np.array(["control"] * n + ["treatment"] * n)
    x_pre = rng.normal(0.0, sigma_x, size=2 * n)
    eps = rng.normal(0.0, sigma_eps, size=2 * n)
    y = alpha_link * x_pre + eps
    # Treatment shift on y.
    y = y + (variant == "treatment") * true_lift
    return pd.DataFrame({"variant": variant, "y": y, "x_pre": x_pre})


# --- theta recovery ----------------------------------------------------------


def test_cuped_theta_recovers_slope():
    df = _simulate_pre_post(
        n=5_000, alpha_link=0.7, sigma_eps=1.0, sigma_x=1.0, true_lift=0.0, seed=101
    )
    theta = cuped_theta(df["y"].to_numpy(), df["x_pre"].to_numpy())
    assert math.isclose(theta, 0.7, abs_tol=0.03)


# --- variance reduction ------------------------------------------------------


def test_cuped_variance_reduction_matches_correlation_squared():
    """ρ²(y, x_pre) ≈ α²·Var(x)/Var(y); the variance reduction should match within MC noise."""
    df = _simulate_pre_post(
        n=10_000, alpha_link=0.7, sigma_eps=1.0, sigma_x=1.0, true_lift=0.0, seed=102
    )
    y = df["y"].to_numpy()
    x = df["x_pre"].to_numpy()
    rho = np.corrcoef(y, x)[0, 1]
    expected = rho * rho

    r = cuped_t_test(df, variant_col="variant", metric_col="y", covariate_col="x_pre")
    assert math.isclose(r.metadata["variance_reduction"], expected, abs_tol=0.02)


def test_cuped_variance_reduction_is_zero_when_covariate_is_noise():
    """Pre-period uncorrelated with post → CUPED reduces variance ≈ 0."""
    rng = np.random.default_rng(103)
    n = 5_000
    df = pd.DataFrame(
        {
            "variant": ["control"] * n + ["treatment"] * n,
            "y": rng.normal(0.0, 1.0, 2 * n),
            "x_pre": rng.normal(0.0, 1.0, 2 * n),  # independent
        }
    )
    r = cuped_t_test(df, variant_col="variant", metric_col="y", covariate_col="x_pre")
    # Reduction can be slightly negative due to sample correlation noise (~|rho|<0.03 → ~0.001 reduction or so).
    assert abs(r.metadata["variance_reduction"]) < 0.02


# --- point estimate is unchanged in expectation -----------------------------


def test_cuped_point_estimate_close_to_welch():
    """CUPED should *not* shift the effect estimate — only shrink its SE."""
    df = _simulate_pre_post(
        n=5_000, alpha_link=0.7, sigma_eps=1.0, sigma_x=1.0, true_lift=0.10, seed=104
    )
    welch = welch_t_test(df, variant_col="variant", metric_col="y")
    cuped = cuped_t_test(df, variant_col="variant", metric_col="y", covariate_col="x_pre")
    # The two estimates differ slightly because CUPED subtracts θ·(x̄_T - x̄_C),
    # which is approximately zero under random assignment but not exactly.
    assert math.isclose(welch.point_estimate, cuped.point_estimate, abs_tol=0.02)


# --- power gain --------------------------------------------------------------


def test_cuped_has_narrower_ci_than_welch_when_covariate_is_predictive():
    df = _simulate_pre_post(
        n=2_000, alpha_link=0.7, sigma_eps=1.0, sigma_x=1.0, true_lift=0.05, seed=105
    )
    welch = welch_t_test(df, variant_col="variant", metric_col="y")
    cuped = cuped_t_test(df, variant_col="variant", metric_col="y", covariate_col="x_pre")
    welch_width = welch.ci_high - welch.ci_low
    cuped_width = cuped.ci_high - cuped.ci_low
    assert cuped_width < welch_width
    # Expected ratio of widths ≈ sqrt(1 - ρ²) ≈ sqrt(1 - 0.49·1/(0.49+1)) ≈ 0.812.
    assert 0.70 < cuped_width / welch_width < 0.90


def test_cuped_adjust_produces_series_aligned_to_df():
    df = _simulate_pre_post(
        n=500, alpha_link=0.5, sigma_eps=1.0, sigma_x=1.0, true_lift=0.0, seed=106
    )
    adj = cuped_adjust(df, metric_col="y", covariate_col="x_pre")
    assert len(adj) == len(df)
    assert (adj.index == df.index).all()
    assert adj.name.endswith("_cuped")
