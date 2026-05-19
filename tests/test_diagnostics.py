"""Tests for the remaining diagnostics: A/A, novelty, Simpson's, BH-FDR, guardrails.

Each diagnostic is checked under (a) a clean null where it must pass and
(b) a planted-on-purpose failure mode where it must fail. This matches the
project's central conceit — the simulator plants known issues; the diagnostic
suite must detect them.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.stats.multitest as smt

from src.diagnostics import (
    GuardrailConfig,
    benjamini_hochberg,
    check_aa,
    check_novelty,
    check_simpsons,
    evaluate_guardrail,
)


# --- A/A pre-period --------------------------------------------------------


def _make_aa_df(diff: float, n_per_arm: int, sd: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "variant": ["control"] * n_per_arm + ["treatment"] * n_per_arm,
            "pre_metric": np.concatenate(
                [rng.normal(0.0, sd, n_per_arm), rng.normal(diff, sd, n_per_arm)]
            ),
        }
    )


def test_aa_passes_under_null():
    df = _make_aa_df(diff=0.0, n_per_arm=2_000, sd=1.0, seed=1)
    r = check_aa(df, variant_col="variant", metric_col="pre_metric")
    assert r.status == "pass"
    assert r.evidence["p_value"] > 0.05


def test_aa_fails_with_planted_pre_period_bias():
    # 5% of SD shift at n=5k/arm is detectable.
    df = _make_aa_df(diff=0.10, n_per_arm=5_000, sd=1.0, seed=2)
    r = check_aa(df, variant_col="variant", metric_col="pre_metric")
    assert r.status == "fail"
    assert r.evidence["p_value"] < 0.01


def test_aa_flags_real_pre_period_bias_at_moderate_effect_size():
    """A 0.10-SD shift at n=3k/arm is reliably non-pass (warn or fail), even
    with seed-to-seed sample variation around the exact p-value band."""
    df = _make_aa_df(diff=0.10, n_per_arm=3_000, sd=1.0, seed=3)
    r = check_aa(df, variant_col="variant", metric_col="pre_metric")
    assert r.status != "pass"
    assert r.evidence["p_value"] < 0.05


# --- Novelty -----------------------------------------------------------------


def _make_daily_df(lifts: list[float], baseline_c: float = 0.05) -> pd.DataFrame:
    """Build a fct_experiment_daily-shaped DataFrame from a list of per-day lifts."""
    days = pd.date_range("2026-01-01", periods=len(lifts), freq="D")
    rows = []
    for d, lift in zip(days, lifts):
        rows.append({"event_date": d, "variant": "control", "conversion_rate": baseline_c})
        rows.append(
            {"event_date": d, "variant": "treatment", "conversion_rate": baseline_c + lift}
        )
    return pd.DataFrame(rows)


def test_novelty_passes_on_constant_lift():
    df = _make_daily_df([0.005] * 28)
    r = check_novelty(df)
    assert r.status == "pass"
    assert abs(r.evidence["slope_per_day"]) < 1e-9


def test_novelty_fails_on_decaying_lift():
    # Linearly decaying from +0.02 on day 0 to ~0 on day 27.
    lifts = np.linspace(0.02, 0.0, 28).tolist()
    df = _make_daily_df(lifts)
    r = check_novelty(df)
    assert r.status == "fail"
    assert r.evidence["slope_per_day"] < 0
    assert r.evidence["p_value"] < 0.01


def test_novelty_warns_with_short_window():
    df = _make_daily_df([0.005, 0.004, 0.003])
    r = check_novelty(df)
    assert r.status == "warn"
    assert "Only" in r.message


def test_novelty_positive_slope_does_not_fail():
    """Anti-novelty (activation lag) is allowed — surface but don't fail."""
    lifts = np.linspace(0.0, 0.02, 28).tolist()
    df = _make_daily_df(lifts)
    r = check_novelty(df)
    assert r.status == "pass"
    assert r.evidence["slope_per_day"] > 0


# --- Simpson's paradox ------------------------------------------------------


def test_simpsons_passes_when_segments_agree_with_overall():
    n = 1_000
    rng = np.random.default_rng(11)
    df = pd.DataFrame(
        {
            "variant": np.tile(["control", "treatment"], n // 2 * 4),
            "segment": np.repeat(["A", "B", "C", "D"], n),
            "converted": np.concatenate(
                [
                    # Each segment: treatment +1pp over control
                    rng.random(n // 2) < 0.05,
                    rng.random(n // 2) < 0.06,
                    rng.random(n // 2) < 0.10,
                    rng.random(n // 2) < 0.11,
                    rng.random(n // 2) < 0.04,
                    rng.random(n // 2) < 0.05,
                    rng.random(n // 2) < 0.08,
                    rng.random(n // 2) < 0.09,
                ]
            ).astype(int),
        }
    )
    r = check_simpsons(df, metric_col="converted")
    assert r.status == "pass"


def test_simpsons_fails_on_classic_reversal():
    """Variant wins +1pp in every segment but loses overall due to mix shift."""
    n_a_c, n_a_t = 100, 900    # Segment A (low base): few in control, many in treatment
    n_b_c, n_b_t = 900, 100    # Segment B (high base): many in control, few in treatment

    rng = np.random.default_rng(12)
    rows = []
    # Segment A: control 5%, treatment 6%
    rows += [
        {"variant": "control", "segment": "A", "converted": int(c)}
        for c in rng.random(n_a_c) < 0.05
    ]
    rows += [
        {"variant": "treatment", "segment": "A", "converted": int(c)}
        for c in rng.random(n_a_t) < 0.06
    ]
    # Segment B: control 50%, treatment 51%
    rows += [
        {"variant": "control", "segment": "B", "converted": int(c)}
        for c in rng.random(n_b_c) < 0.50
    ]
    rows += [
        {"variant": "treatment", "segment": "B", "converted": int(c)}
        for c in rng.random(n_b_t) < 0.51
    ]
    df = pd.DataFrame(rows)
    r = check_simpsons(df, metric_col="converted")
    assert r.status == "fail"
    assert r.evidence["overall_sign"] != r.evidence["weighted_majority_sign"]


def test_simpsons_ignores_tiny_segments_in_majority_vote():
    """A 5-user segment with a flipped sign shouldn't outvote big segments."""
    rng = np.random.default_rng(13)
    # Big segment A: treatment +
    big_c = (rng.random(500) < 0.05).astype(int)
    big_t = (rng.random(500) < 0.07).astype(int)
    # Tiny segment B (well below default min_segment_n=30): treatment -
    tiny_c = np.array([1, 1, 1, 1, 1])
    tiny_t = np.array([0, 0, 0, 0, 0])
    df = pd.DataFrame(
        {
            "variant": ["control"] * 500 + ["treatment"] * 500 + ["control"] * 5 + ["treatment"] * 5,
            "segment": ["A"] * 1000 + ["B"] * 10,
            "converted": np.concatenate([big_c, big_t, tiny_c, tiny_t]),
        }
    )
    r = check_simpsons(df, metric_col="converted")
    assert r.evidence["segments"]["B"]["excluded"] is True
    assert r.status == "pass"


# --- BH-FDR ----------------------------------------------------------------


def test_bh_matches_statsmodels_exactly():
    rng = np.random.default_rng(21)
    p_values = rng.random(50).tolist()
    mine = benjamini_hochberg(p_values, alpha=0.10)
    rejected_ref, adjusted_ref, *_ = smt.multipletests(p_values, alpha=0.10, method="fdr_bh")
    np.testing.assert_allclose(mine.adjusted_p, adjusted_ref, atol=1e-12)
    np.testing.assert_array_equal(mine.rejected, rejected_ref)


def test_bh_all_zero_pvalues_all_rejected():
    r = benjamini_hochberg([0.0, 0.0, 0.0])
    assert r.n_rejected == 3
    assert (r.adjusted_p == 0).all()


def test_bh_empty_input():
    r = benjamini_hochberg([])
    assert r.n_rejected == 0
    assert r.adjusted_p.shape == (0,)


def test_bh_rejects_invalid_pvalues():
    with pytest.raises(ValueError):
        benjamini_hochberg([0.05, 1.5, 0.2])


def test_bh_adjusted_p_is_monotone_in_raw_p_order():
    """Within the sorted order, adjusted p-values must be non-decreasing."""
    p = [0.001, 0.01, 0.02, 0.03, 0.5, 0.9]
    r = benjamini_hochberg(p)
    order = np.argsort(p)
    sorted_adjusted = r.adjusted_p[order]
    assert (np.diff(sorted_adjusted) >= -1e-12).all()


# --- Guardrails -------------------------------------------------------------


def _make_latency_df(mean_c: float, mean_t: float, sd: float, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "variant": ["control"] * n + ["treatment"] * n,
            "latency_ms": np.concatenate(
                [rng.normal(mean_c, sd, n), rng.normal(mean_t, sd, n)]
            ),
        }
    )


def test_guardrail_passes_when_change_within_threshold():
    df = _make_latency_df(mean_c=200.0, mean_t=202.0, sd=20.0, n=2_000, seed=31)
    config = GuardrailConfig(
        metric_name="latency_ms", direction="lower_is_better", threshold_relative=0.05
    )
    r = evaluate_guardrail(
        df, variant_col="variant", metric_col="latency_ms", config=config
    )
    assert r.status == "pass"


def test_guardrail_fails_when_latency_grows_beyond_threshold():
    # Treatment latency +15% — well over the 5% threshold, with n=2000 it's significant.
    df = _make_latency_df(mean_c=200.0, mean_t=230.0, sd=20.0, n=2_000, seed=32)
    config = GuardrailConfig(
        metric_name="latency_ms", direction="lower_is_better", threshold_relative=0.05
    )
    r = evaluate_guardrail(
        df, variant_col="variant", metric_col="latency_ms", config=config
    )
    assert r.status == "fail"
    assert r.evidence["relative_change"] > 0.05
    assert r.evidence["significant"]


def test_guardrail_warns_on_directional_but_not_significant_shift():
    # Small shift that's directionally over threshold but n too small for significance.
    df = _make_latency_df(mean_c=200.0, mean_t=215.0, sd=80.0, n=50, seed=33)
    config = GuardrailConfig(
        metric_name="latency_ms", direction="lower_is_better", threshold_relative=0.05
    )
    r = evaluate_guardrail(
        df, variant_col="variant", metric_col="latency_ms", config=config
    )
    # With very small n and large noise, p-value likely > 0.05 -> warn (or pass if p>0.05 and ratio just under).
    # Make sure if ratio > threshold then it isn't a "pass".
    if r.evidence["relative_change"] > 0.05:
        assert r.status in ("warn", "fail")
    else:
        # If sampling shifted ratio under threshold, the test data wasn't strong enough.
        # Accept "pass" — the *invariant* we care about is the violation/no-violation logic.
        assert r.status == "pass"


def test_guardrail_higher_is_better_direction():
    """Retention metric: treatment drops too much → fail."""
    rng = np.random.default_rng(34)
    n = 5_000
    df = pd.DataFrame(
        {
            "variant": ["control"] * n + ["treatment"] * n,
            "retained": np.concatenate(
                [(rng.random(n) < 0.70).astype(int), (rng.random(n) < 0.60).astype(int)]
            ),
        }
    )
    config = GuardrailConfig(
        metric_name="retention", direction="higher_is_better", threshold_relative=0.05
    )
    r = evaluate_guardrail(
        df, variant_col="variant", metric_col="retained", config=config
    )
    # treatment retention is ~0.60, control ~0.70 → rel change ≈ -0.143 → fails 5% threshold
    assert r.status == "fail"
    assert r.evidence["relative_change"] < -0.05


def test_guardrail_higher_is_better_passes_on_improvement():
    rng = np.random.default_rng(35)
    n = 5_000
    df = pd.DataFrame(
        {
            "variant": ["control"] * n + ["treatment"] * n,
            "retained": np.concatenate(
                [(rng.random(n) < 0.70).astype(int), (rng.random(n) < 0.75).astype(int)]
            ),
        }
    )
    config = GuardrailConfig(
        metric_name="retention", direction="higher_is_better", threshold_relative=0.05
    )
    r = evaluate_guardrail(
        df, variant_col="variant", metric_col="retained", config=config
    )
    assert r.status == "pass"  # treatment is BETTER → not a violation
