"""SRM check tests.

The SRM diagnostic is the lowest-cost / highest-impact check in the suite —
a regression here would silently invalidate every downstream effect estimate.
Tests cover: pass on a clean 50/50 split, fail on a planted bug, 3-way splits,
off-balance allocations, and exact agreement with scipy.stats.chisquare.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest
from scipy import stats

from src.diagnostics import check_srm


def _split_df(n_control: int, n_treatment: int) -> pd.DataFrame:
    return pd.DataFrame(
        {"variant": ["control"] * n_control + ["treatment"] * n_treatment}
    )


def test_clean_50_50_passes():
    df = _split_df(5_000, 5_000)
    result = check_srm(df)
    assert result.status == "pass"
    assert result.evidence["p_value"] > 0.5  # 5000/5000 should give very high p


def test_planted_55_45_fails():
    df = _split_df(5_500, 4_500)
    result = check_srm(df)
    assert result.status == "fail"
    assert result.evidence["p_value"] < 1e-10  # 1000-user gap at n=10k is overwhelmingly significant


def test_subtle_5050_with_small_skew_still_passes():
    # 4970/5030 — within ordinary sampling variation for a deterministic bucketer.
    df = _split_df(4_970, 5_030)
    result = check_srm(df)
    assert result.status == "pass"
    assert result.evidence["p_value"] > 0.3


def test_borderline_skew_triggers_warn():
    # Just inside the [0.001, 0.01) p-value band. Need a skew that yields p ~0.005.
    # chi2 ~7.88 at df=1 → p ~0.005. Solve (n_c - 5000)² * 2 / 5000 = 7.88
    # → diff² = 7.88 * 2500 = 19,700 → diff ≈ 140. Try 4860/5140.
    df = _split_df(4_860, 5_140)
    result = check_srm(df)
    assert result.status == "warn"


def test_chi_square_p_value_matches_scipy_directly():
    df = _split_df(4_900, 5_100)
    result = check_srm(df)
    chi2_ref, p_ref = stats.chisquare(f_obs=[4_900, 5_100], f_exp=[5_000, 5_000])
    assert math.isclose(result.evidence["chi2"], float(chi2_ref), rel_tol=1e-9)
    assert math.isclose(result.evidence["p_value"], float(p_ref), rel_tol=1e-9)


def test_three_way_balanced_passes():
    df = pd.DataFrame(
        {"variant": ["control"] * 3_333 + ["treatment_a"] * 3_333 + ["treatment_b"] * 3_334}
    )
    result = check_srm(
        df,
        expected_weights={"control": 1 / 3, "treatment_a": 1 / 3, "treatment_b": 1 / 3},
    )
    assert result.status == "pass"


def test_three_way_with_one_arm_dropped_fails():
    df = pd.DataFrame(
        {"variant": ["control"] * 4_000 + ["treatment_a"] * 3_000 + ["treatment_b"] * 3_000}
    )
    result = check_srm(
        df,
        expected_weights={"control": 1 / 3, "treatment_a": 1 / 3, "treatment_b": 1 / 3},
    )
    assert result.status == "fail"


def test_off_balance_70_30_with_correct_expectation_passes():
    df = _split_df(7_000, 3_000)
    result = check_srm(df, expected_weights={"control": 0.7, "treatment": 0.3})
    assert result.status == "pass"


def test_off_balance_70_30_against_5050_expectation_fails():
    df = _split_df(7_000, 3_000)
    result = check_srm(df)  # default 50/50 expectation
    assert result.status == "fail"


def test_weights_not_summing_to_one_raises():
    df = _split_df(5_000, 5_000)
    with pytest.raises(ValueError):
        check_srm(df, expected_weights={"control": 0.4, "treatment": 0.4})


def test_evidence_includes_observed_and_expected_counts():
    df = _split_df(5_000, 5_000)
    result = check_srm(df)
    assert result.evidence["observed_counts"] == {"control": 5_000, "treatment": 5_000}
    assert result.evidence["expected_counts"]["control"] == pytest.approx(5_000)
