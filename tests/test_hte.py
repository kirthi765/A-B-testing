"""HTE module tests.

The expensive test (`test_causal_forest_recovers_per_segment_ordering`) is the
load-bearing one — it fits a CausalForestDML on the heterogeneous scenario
and checks that the per-segment CATE rank order matches the planted lifts.
Marked tolerant; the model is stochastic and the planted effects on `casual`
and `enterprise` are intentionally small.

The utility tests (`cate_by_decile`, `cate_by_segment`, `build_user_level_dataset`,
`one_hot_segments`, the heterogeneous-scenario plant) are fast and isolated —
they catch most regressions without paying the model-fit cost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.hte import (
    build_user_level_dataset,
    cate_by_decile,
    cate_by_segment,
    fit_causal_forest,
    one_hot_segments,
    predict_cate,
)
from src.simulator.scenarios import clean_lift, heterogeneous


# --- heterogeneous scenario plant -------------------------------------------


def test_heterogeneous_scenario_plants_distinct_per_segment_lifts():
    s = heterogeneous(
        n_users=4_000,
        experiment_days=14,
        seed=1,
        segment_lifts={
            "power_user": 0.20,
            "casual": -0.05,
            "new_signup": 0.10,
            "enterprise": -0.02,
        },
    )
    # Ground truth is recorded in the Scenario.
    assert s.ground_truth["scenario"] == "heterogeneous"
    gt = s.ground_truth["true_lift_relative_by_segment"]
    assert gt["power_user"] == 0.20
    assert gt["casual"] == -0.05

    # Reproducibility: same seed → identical exposures (deterministic hash-mod).
    s2 = heterogeneous(n_users=4_000, experiment_days=14, seed=1)
    pd.testing.assert_frame_equal(s.exposures, s2.exposures)


def test_heterogeneous_recovers_per_segment_direction_at_aggregate_level():
    """Sanity: empirical per-segment lifts should at least rank like the planted ones,
    without any HTE machinery. If this fails, the scenario itself is broken."""
    s = heterogeneous(n_users=8_000, experiment_days=14, seed=2)
    df = build_user_level_dataset(s.users, s.events, s.exposures)
    by = (
        df.groupby(["segment", "variant"])["n_conversions"]
        .mean()
        .unstack("variant")
    )
    lifts = (by["treatment"] - by["control"]).to_dict()
    # Power_user has the largest planted relative lift on the highest base — its
    # absolute count lift should dominate.
    assert lifts["power_user"] == max(lifts.values())


# --- utility tests ----------------------------------------------------------


def test_cate_by_decile_shapes_and_monotonicity():
    rng = np.random.default_rng(11)
    cate = rng.normal(0.0, 1.0, size=1_000)
    out = cate_by_decile(cate, n_buckets=10)
    assert list(out.columns) == ["bucket", "n", "cate_mean", "cate_p05", "cate_p95"]
    assert len(out) == 10
    assert (out["n"].sum()) == 1_000
    # Decile means must be monotonically increasing (rank-based binning).
    assert (np.diff(out["cate_mean"]) >= 0).all()


def test_cate_by_decile_empty_input():
    out = cate_by_decile(np.array([]), n_buckets=10)
    assert len(out) == 0


def test_cate_by_segment_orders_descending_by_mean():
    seg = pd.Series(["a"] * 100 + ["b"] * 100 + ["c"] * 100)
    cate = np.concatenate(
        [np.full(100, 0.5), np.full(100, -0.2), np.full(100, 1.0)]
    )
    out = cate_by_segment(cate, seg)
    assert out.iloc[0]["segment"] == "c"
    assert out.iloc[-1]["segment"] == "b"
    assert list(out["cate_mean"]) == [1.0, 0.5, -0.2]


def test_build_user_level_dataset_has_one_row_per_user():
    s = clean_lift(n_users=1_000, experiment_days=7, seed=3)
    df = build_user_level_dataset(s.users, s.events, s.exposures)
    assert df["user_id"].is_unique
    assert len(df) == 1_000
    assert {"segment", "variant", "treatment", "n_sessions", "n_conversions", "converted"} <= set(df.columns)
    # treatment is 0/1
    assert set(df["treatment"].unique()) <= {0, 1}
    # converted matches n_conversions > 0
    assert (df["converted"] == (df["n_conversions"] > 0).astype(int)).all()


def test_one_hot_segments_returns_correct_columns():
    df = pd.DataFrame({"segment": ["a", "b", "a", "c"]})
    out, cols = one_hot_segments(df)
    assert set(cols) == {"seg_a", "seg_b", "seg_c"}
    assert (out["seg_a"].to_numpy() == [1, 0, 1, 0]).all()
    assert (out["seg_b"].to_numpy() == [0, 1, 0, 0]).all()
    assert (out["seg_c"].to_numpy() == [0, 0, 0, 1]).all()


# --- the load-bearing test: CATE recovery on heterogeneous ------------------


@pytest.mark.slow
def test_causal_forest_recovers_per_segment_ordering():
    """Fit CausalForestDML on the heterogeneous scenario; verify that the
    estimated per-segment mean CATE ranks `power_user` at the top and one of
    `{casual, enterprise}` at the bottom — matching the planted +20% / -5%
    / -2% structure."""
    s = heterogeneous(n_users=4_000, experiment_days=14, seed=4)
    df = build_user_level_dataset(s.users, s.events, s.exposures)
    df, feature_cols = one_hot_segments(df)

    # CausalForestDML requires n_estimators % subforest_size == 0; default is 4.
    # 32 is plenty for a rank-recovery test — production would use 200+.
    model = fit_causal_forest(
        df,
        treatment_col="treatment",
        outcome_col="n_conversions",
        feature_cols=feature_cols,
        n_estimators=32,
        random_state=4,
    )
    cate = predict_cate(model, df, feature_cols)
    by_seg = cate_by_segment(cate, df["segment"]).set_index("segment")

    # 1. Power_user has the highest mean CATE — the planted +20% on base rate
    #    8% with ~70 sessions gives ~+1.1 conversion delta per user, dominates.
    assert by_seg["cate_mean"].idxmax() == "power_user"

    # 2. The lowest-CATE segment is one of the planted-negative segments.
    assert by_seg["cate_mean"].idxmin() in {"casual", "enterprise"}

    # 3. Power_user CATE is clearly positive and substantively larger than 0.
    assert by_seg.loc["power_user", "cate_mean"] > 0.4
