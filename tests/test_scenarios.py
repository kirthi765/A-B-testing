"""Each planted scenario must be detected by its matching diagnostic.

This is the central conceit of the project: the simulator plants known
failure modes; the diagnostic suite catches them. If any scenario stops
firing its matching check, that's a regression in either the simulator
or the diagnostic.

We build a small per-test dataset (n ≈ 5k, ~14 days) and run the relevant
diagnostic directly — no dbt mart construction here; that lives in test_ui.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.diagnostics import (
    GuardrailConfig,
    check_novelty,
    check_simpsons,
    check_srm,
    evaluate_guardrail,
)
from src.hte.causal_trees import build_user_level_dataset
from src.simulator.scenarios import (
    SCENARIOS,
    aa_drift,
    guardrail_violation,
    novelty_effect,
    simpsons,
    srm_bug,
)
from src.stats import two_proportion_z_test, welch_t_test


def _daily_rollup(scenario) -> pd.DataFrame:
    """Build a fct_experiment_daily-shaped DataFrame from raw scenario events."""
    e = scenario.events.merge(
        scenario.exposures[["user_id", "variant"]], on="user_id"
    )
    e["event_date"] = pd.to_datetime(e["ts"]).dt.normalize()
    rollup = (
        e.groupby(["variant", "event_date"])
        .agg(
            n_sessions=("event_type", lambda x: (x == "session").sum()),
            n_conversions=("event_type", lambda x: (x == "conversion").sum()),
        )
        .reset_index()
    )
    rollup["conversion_rate"] = rollup["n_conversions"] / rollup["n_sessions"].clip(
        lower=1
    )
    return rollup


# --- registry --------------------------------------------------------------


def test_scenario_registry_lists_all_seven():
    assert set(SCENARIOS.keys()) == {
        "clean_lift",
        "heterogeneous",
        "srm_bug",
        "novelty_effect",
        "simpsons",
        "guardrail_violation",
        "aa_drift",
    }


# --- srm_bug ---------------------------------------------------------------


def test_srm_bug_triggers_srm_check():
    s = srm_bug(n_users=5_000, experiment_days=14, actual_treatment_share=0.55, seed=1)
    r = check_srm(s.exposures)
    assert r.status == "fail"
    assert r.evidence["observed_share"]["treatment"] > 0.52
    assert s.ground_truth["true_assignment_ratio"] == 0.55


def test_srm_bug_ground_truth_records_planted_ratio():
    s = srm_bug(actual_treatment_share=0.60, seed=2)
    assert s.ground_truth["true_assignment_ratio"] == 0.60
    assert s.ground_truth["expected_assignment_ratio"] == 0.5


# --- novelty_effect --------------------------------------------------------


def test_novelty_effect_triggers_novelty_check():
    # n=15k is needed to make the daily slope signal clearly distinguishable
    # from noise on a +10% planted decay; defaults to n=10k for case-study runs.
    s = novelty_effect(n_users=15_000, experiment_days=28, seed=3)
    daily = _daily_rollup(s)
    r = check_novelty(daily, metric_col="conversion_rate")
    assert r.status == "fail"
    assert r.evidence["slope_per_day"] < 0
    assert r.evidence["p_value"] < 0.01


def test_novelty_effect_lift_is_higher_in_week1_than_week4():
    """Empirical check on the planted decay: week-1 lift should be clearly
    larger than week-4 lift (close to zero by construction)."""
    s = novelty_effect(n_users=8_000, experiment_days=28, seed=4)
    daily = _daily_rollup(s)
    pivot = daily.pivot_table(
        index="event_date", columns="variant", values="conversion_rate"
    )
    pivot["lift"] = pivot["treatment"] - pivot["control"]
    pivot = pivot.sort_index()
    week1_mean = pivot["lift"].iloc[:7].mean()
    week4_mean = pivot["lift"].iloc[-7:].mean()
    assert week1_mean > week4_mean
    # Week-1 lift should be clearly positive on a +10% planted initial effect.
    assert week1_mean > 0.001


# --- simpsons --------------------------------------------------------------


def test_simpsons_triggers_simpsons_check():
    s = simpsons(n_users=10_000, experiment_days=21, seed=5)
    df = build_user_level_dataset(s.users, s.events, s.exposures)
    r = check_simpsons(df, metric_col="converted")
    assert r.status == "fail"
    # Overall sign and weighted-segment majority must disagree.
    assert r.evidence["overall_sign"] != r.evidence["weighted_majority_sign"]


def test_simpsons_overall_loses_but_each_segment_wins():
    """Direct verification of the planted reversal at the data level."""
    s = simpsons(n_users=10_000, experiment_days=21, seed=6)
    df = build_user_level_dataset(s.users, s.events, s.exposures)
    overall_c = df.loc[df["variant"] == "control", "converted"].mean()
    overall_t = df.loc[df["variant"] == "treatment", "converted"].mean()
    assert overall_t < overall_c  # variant loses overall

    # And wins in every segment.
    for seg in df["segment"].unique():
        sub = df[df["segment"] == seg]
        c = sub.loc[sub["variant"] == "control", "converted"].mean()
        t = sub.loc[sub["variant"] == "treatment", "converted"].mean()
        if len(sub) < 100:
            continue
        # Allow rare ties on small / saturated segments.
        assert t >= c - 0.01, f"segment {seg}: treatment {t:.4f} should beat control {c:.4f}"


# --- guardrail_violation ---------------------------------------------------


def test_guardrail_violation_triggers_latency_check():
    s = guardrail_violation(
        n_users=5_000, experiment_days=14, latency_lift_ms=30.0, seed=7
    )
    df = build_user_level_dataset(s.users, s.events, s.exposures)
    config = GuardrailConfig(
        metric_name="latency",
        direction="lower_is_better",
        threshold_relative=0.05,
    )
    r = evaluate_guardrail(
        df.dropna(subset=["mean_latency_ms"]),
        variant_col="variant",
        metric_col="mean_latency_ms",
        config=config,
    )
    assert r.status == "fail"
    assert r.evidence["relative_change"] > 0.05


def test_guardrail_violation_still_has_real_conversion_lift():
    """The whole point: primary metric looks like a win, so without the
    guardrail you'd ship — but the guardrail blocks. We use per-user
    session-level conversion_rate (not binary `converted`) because the
    binary metric saturates on heavy-session segments before the planted
    +5% lift shows up clearly."""
    s = guardrail_violation(n_users=10_000, experiment_days=28, seed=8)
    df = build_user_level_dataset(s.users, s.events, s.exposures)
    primary = welch_t_test(
        df.dropna(subset=["conversion_rate"]),
        variant_col="variant",
        metric_col="conversion_rate",
    )
    assert primary.point_estimate > 0
    assert primary.p_value < 0.05


# --- aa_drift --------------------------------------------------------------


def test_aa_drift_creates_false_positive_in_primary_test():
    """No real lift, but segment-biased assignment over-represents high-converting
    power_users in the treatment arm. A naive primary-metric test rejects."""
    s = aa_drift(n_users=10_000, experiment_days=14, seed=9)
    df = build_user_level_dataset(s.users, s.events, s.exposures)
    test = welch_t_test(
        df.dropna(subset=["conversion_rate"]),
        variant_col="variant",
        metric_col="conversion_rate",
    )
    assert test.point_estimate > 0  # spurious win for treatment
    assert test.p_value < 0.05  # ...and "significant"
    assert s.ground_truth["true_lift_relative"] == 0.0  # but no real effect


def test_aa_drift_keeps_overall_assignment_near_5050():
    """aa_drift is supposed to NOT trigger the SRM check (the bias is on
    *segment-conditional* shares, not the overall ratio). If this stops
    being true, SRM would fire first and mask the drift demo."""
    s = aa_drift(n_users=15_000, experiment_days=7, seed=10)
    share_t = (s.exposures["variant"] == "treatment").mean()
    assert abs(share_t - 0.5) < 0.02  # within 2 pp of 50/50


# --- backward-compat regression check --------------------------------------


def test_existing_clean_lift_unchanged_by_daily_multiplier_addition():
    """The TreatmentEffect refactor (adding daily_lift_multiplier + restructured
    p_conv_per_session computation) must not change events for scenarios that
    don't use the new field. We pin a hash on the events DataFrame."""
    from src.simulator.scenarios import clean_lift

    s = clean_lift(n_users=500, experiment_days=7, seed=999)
    # The number of events at a fixed seed is the strongest cheap signature.
    n_sessions = int((s.events["event_type"] == "session").sum())
    n_conversions = int((s.events["event_type"] == "conversion").sum())
    # Exact counts from a baseline run at seed=999 — pin them so any
    # accidental change to random-state consumption order fails loudly here.
    assert n_sessions == 6_058
    assert n_conversions == 370
