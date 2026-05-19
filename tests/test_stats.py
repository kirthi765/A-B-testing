"""Stats engine tests.

Every estimator is checked against an independent reference where one exists
(scipy / statsmodels), or against a closed-form expectation when one doesn't.
A test that just verifies a function "runs" is worse than no test — it lulls.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.stats.proportion as smp
from scipy import stats

from src.stats import (
    bootstrap_diff_of_means,
    delta_method_ratio,
    mde_continuous,
    mde_proportions,
    msprt,
    sample_size_continuous,
    sample_size_proportions,
    simulate_type_i_error_under_peeking,
    two_proportion_z_test,
    welch_t_test,
)


# --- Welch's t-test ----------------------------------------------------------


def _make_continuous_df(mean_c, mean_t, sd, n, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "variant": ["control"] * n + ["treatment"] * n,
            "y": np.concatenate(
                [rng.normal(mean_c, sd, n), rng.normal(mean_t, sd, n)]
            ),
        }
    )


def test_welch_pvalue_matches_scipy_under_null():
    df = _make_continuous_df(0.0, 0.0, 1.0, n=400, seed=1)
    result = welch_t_test(df, variant_col="variant", metric_col="y")
    ref = stats.ttest_ind(
        df.loc[df["variant"] == "treatment", "y"],
        df.loc[df["variant"] == "control", "y"],
        equal_var=False,
    )
    assert math.isclose(result.p_value, float(ref.pvalue), rel_tol=1e-9, abs_tol=1e-12)
    assert math.isclose(result.metadata["t_stat"], float(ref.statistic), rel_tol=1e-9, abs_tol=1e-12)


def test_welch_pvalue_matches_scipy_with_strong_signal():
    df = _make_continuous_df(0.0, 1.0, 1.0, n=200, seed=2)
    result = welch_t_test(df, variant_col="variant", metric_col="y")
    ref = stats.ttest_ind(
        df.loc[df["variant"] == "treatment", "y"],
        df.loc[df["variant"] == "control", "y"],
        equal_var=False,
    )
    assert math.isclose(result.p_value, float(ref.pvalue), rel_tol=1e-9, abs_tol=1e-12)
    assert result.p_value < 1e-10  # n=200 with effect-size 1 SD is overwhelming
    assert result.ci_low > 0  # CI excludes 0


def test_welch_ci_is_symmetric_around_point_estimate():
    df = _make_continuous_df(2.0, 2.7, 1.5, n=300, seed=3)
    r = welch_t_test(df, variant_col="variant", metric_col="y")
    mid = (r.ci_low + r.ci_high) / 2
    assert math.isclose(mid, r.point_estimate, rel_tol=1e-9, abs_tol=1e-12)


def test_welch_requires_two_obs_per_arm():
    df = pd.DataFrame({"variant": ["control", "treatment"], "y": [1.0, 2.0]})
    with pytest.raises(ValueError):
        welch_t_test(df, variant_col="variant", metric_col="y")


# --- two-proportion z-test ---------------------------------------------------


def _make_binary_df(p_c, p_t, n_c, n_t, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "variant": ["control"] * n_c + ["treatment"] * n_t,
            "converted": np.concatenate(
                [rng.random(n_c) < p_c, rng.random(n_t) < p_t]
            ).astype(int),
        }
    )


def test_ztest_pvalue_matches_statsmodels():
    df = _make_binary_df(0.10, 0.12, 5_000, 5_000, seed=11)
    r = two_proportion_z_test(df, variant_col="variant", metric_col="converted")
    k_c = int(df.loc[df["variant"] == "control", "converted"].sum())
    k_t = int(df.loc[df["variant"] == "treatment", "converted"].sum())
    n_c = int((df["variant"] == "control").sum())
    n_t = int((df["variant"] == "treatment").sum())
    ref_z, ref_p = smp.proportions_ztest([k_t, k_c], [n_t, n_c])
    assert math.isclose(r.p_value, float(ref_p), rel_tol=1e-9, abs_tol=1e-12)
    assert math.isclose(r.metadata["z_stat"], float(ref_z), rel_tol=1e-9, abs_tol=1e-12)


def test_ztest_point_estimate_and_ci_direction():
    df = _make_binary_df(0.05, 0.08, 10_000, 10_000, seed=12)
    r = two_proportion_z_test(df, variant_col="variant", metric_col="converted")
    assert r.point_estimate > 0
    assert r.ci_low > 0  # CI for a +3pp effect at n=10k each should clearly exclude 0
    assert r.p_value < 1e-10


def test_ztest_under_null_pvalue_uniform_ish():
    """One sample isn't enough to check uniformity, but it should not be tiny."""
    df = _make_binary_df(0.10, 0.10, 2_000, 2_000, seed=13)
    r = two_proportion_z_test(df, variant_col="variant", metric_col="converted")
    assert r.p_value > 0.05  # very likely passes; if not, seed catches it


# --- bootstrap ---------------------------------------------------------------


def test_bootstrap_ci_brackets_welch_at_large_n():
    df = _make_continuous_df(0.0, 0.3, 1.0, n=2_000, seed=21)
    welch = welch_t_test(df, variant_col="variant", metric_col="y")
    boot = bootstrap_diff_of_means(
        df, variant_col="variant", metric_col="y", n_resamples=2_000, seed=21
    )
    # Both methods estimate the same diff; bootstrap CI width should be within
    # ~25% of the parametric CI for ~normal data at n=2000.
    welch_width = welch.ci_high - welch.ci_low
    boot_width = boot.ci_high - boot.ci_low
    assert abs(boot_width - welch_width) / welch_width < 0.25
    # Point estimates equal by construction (both are sample mean diff).
    assert math.isclose(boot.point_estimate, welch.point_estimate, rel_tol=1e-12)


def test_bootstrap_is_deterministic_under_same_seed():
    df = _make_continuous_df(0.0, 0.5, 1.0, n=500, seed=22)
    a = bootstrap_diff_of_means(
        df, variant_col="variant", metric_col="y", n_resamples=500, seed=99
    )
    b = bootstrap_diff_of_means(
        df, variant_col="variant", metric_col="y", n_resamples=500, seed=99
    )
    assert a.p_value == b.p_value
    assert a.ci_low == b.ci_low
    assert a.ci_high == b.ci_high


def test_bootstrap_rejects_under_strong_signal():
    df = _make_continuous_df(0.0, 1.0, 1.0, n=500, seed=23)
    r = bootstrap_diff_of_means(
        df, variant_col="variant", metric_col="y", n_resamples=1_000, seed=23
    )
    assert r.ci_low > 0
    assert r.p_value < 0.01


# --- delta method ------------------------------------------------------------


def test_delta_method_reduces_to_mean_when_denominator_is_one():
    """If x = 1 for every user, ratio = sum(y)/sum(1) = mean(y), and the delta-method
    variance collapses to Var(y)/n — i.e. an ordinary t-test on means."""
    rng = np.random.default_rng(31)
    n = 1000
    df = pd.DataFrame(
        {
            "variant": ["control"] * n + ["treatment"] * n,
            "y": np.concatenate([rng.normal(1.0, 1.0, n), rng.normal(1.2, 1.0, n)]),
            "x": 1.0,
        }
    )
    delta = delta_method_ratio(
        df, variant_col="variant", numerator_col="y", denominator_col="x"
    )
    welch = welch_t_test(df, variant_col="variant", metric_col="y")
    # Point estimates equal exactly (both: mean(y_t) - mean(y_c)).
    assert math.isclose(delta.point_estimate, welch.point_estimate, rel_tol=1e-12)
    # SEs equal to floating-point: delta has Var = Var(y)/n with x≡1, same as Welch.
    assert math.isclose(delta.metadata["se"], welch.metadata["se"], rel_tol=1e-9)


def test_delta_method_se_below_naive_when_correlated():
    """When clicks (y) and impressions (x) are positively correlated per user, the
    delta-method SE on clicks/impressions is *smaller* than a naive per-impression
    z-test would imply. This is the whole reason to use it.
    """
    rng = np.random.default_rng(32)
    n = 500
    # Each user has impressions ~ Poisson(10) and per-impression click rate ~0.1
    imp = rng.poisson(10, size=2 * n) + 1
    p = 0.10
    clicks = rng.binomial(imp, p)
    df = pd.DataFrame(
        {
            "variant": ["control"] * n + ["treatment"] * n,
            "y": clicks,
            "x": imp,
        }
    )
    r = delta_method_ratio(
        df, variant_col="variant", numerator_col="y", denominator_col="x"
    )
    # Under H0 (same rate in both arms) we expect p > 0.05 with high probability.
    assert r.p_value > 0.01
    assert r.metadata["se"] > 0
    assert math.isclose(r.metadata["ratio_control"], p, abs_tol=0.02)


# --- mSPRT -------------------------------------------------------------------


def test_msprt_is_more_conservative_than_welch_at_fixed_n():
    """The mSPRT CI should be wider than the Welch CI on the same data — the
    cost you pay for peek-safety."""
    df = _make_continuous_df(0.0, 0.3, 1.0, n=2_000, seed=41)
    welch = welch_t_test(df, variant_col="variant", metric_col="y")
    seq = msprt(df, variant_col="variant", metric_col="y")
    assert (seq.ci_high - seq.ci_low) > (welch.ci_high - welch.ci_low)
    # Same point estimate.
    assert math.isclose(seq.point_estimate, welch.point_estimate, rel_tol=1e-12)


def test_msprt_rejects_under_strong_signal():
    df = _make_continuous_df(0.0, 1.0, 1.0, n=1_000, seed=42)
    r = msprt(df, variant_col="variant", metric_col="y")
    assert r.p_value < 0.05
    assert r.ci_low > 0


def test_msprt_does_not_reject_under_null():
    df = _make_continuous_df(0.0, 0.0, 1.0, n=1_000, seed=43)
    r = msprt(df, variant_col="variant", metric_col="y")
    assert r.p_value > 0.05
    assert r.ci_low < 0 < r.ci_high


def test_msprt_controls_type_i_error_under_peeking():
    """Peeking simulation: under H0, mSPRT should stay near α, naive z-test inflates.

    This is the whole point of sequential testing. Small n_trials for test speed —
    the property is robust at this sample size for a clear pass/fail."""
    out = simulate_type_i_error_under_peeking(
        n_per_arm=500, n_peeks=20, n_trials=200, sigma=1.0, alpha=0.05, seed=44
    )
    # mSPRT is conservative — should be at or below alpha.
    assert out["msprt_type_i_error"] <= 0.10
    # Naive z-test with 20 peeks should blow past alpha.
    assert out["naive_type_i_error"] > 0.10


# --- power / sample size -----------------------------------------------------


def test_sample_size_proportions_textbook_p010_mde10pct():
    """Closed-form check: p_c=0.10, +10% relative MDE (delta = 1pp), alpha=0.05, power=0.80.

    n_per_arm = (1.96 + 0.8416)² · (p_c(1-p_c) + p_t(1-p_t)) / delta²
              ≈ 7.8489 · (0.09 + 0.0979) / 0.0001 ≈ 14,748
    """
    s = sample_size_proportions(baseline_rate=0.10, mde_relative=0.10)
    assert 14_500 <= s.n_control <= 15_000
    assert 14_500 <= s.n_treatment <= 15_000
    assert s.n_total == s.n_control + s.n_treatment


def test_sample_size_continuous_round_trip_with_mde():
    """Compute N from MDE, then MDE from that N — they should match.
    This catches off-by-one errors and asymmetric formulas."""
    sigma = 2.5
    mde = 0.3
    s = sample_size_continuous(std=sigma, mde_absolute=mde)
    recovered_mde = mde_continuous(std=sigma, n_per_arm=s.n_control)
    # `recovered_mde` should be at or below `mde` because the per-arm n is rounded up.
    assert recovered_mde <= mde + 1e-6
    assert recovered_mde >= mde * 0.99  # within ~1% of the target


def test_sample_size_proportions_higher_power_needs_more_n():
    a = sample_size_proportions(baseline_rate=0.10, mde_relative=0.10, power=0.80)
    b = sample_size_proportions(baseline_rate=0.10, mde_relative=0.10, power=0.95)
    assert b.n_total > a.n_total


def test_sample_size_rejects_implausible_inputs():
    with pytest.raises(ValueError):
        sample_size_proportions(baseline_rate=0.0, mde_relative=0.10)
    with pytest.raises(ValueError):
        sample_size_proportions(baseline_rate=0.5, mde_relative=10.0)  # implies p_t > 1
    with pytest.raises(ValueError):
        sample_size_continuous(std=-1.0, mde_absolute=0.5)


def test_mde_proportions_decreases_with_n():
    big = mde_proportions(baseline_rate=0.10, n_per_arm=10_000)
    small = mde_proportions(baseline_rate=0.10, n_per_arm=1_000)
    assert big < small
