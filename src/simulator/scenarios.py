"""Named scenarios — each function returns a `Scenario` with known ground truth.

Phase 1 ships `clean_lift` only. Subsequent phases add novelty / SRM / Simpson's /
guardrail / heterogeneous / aa_drift, each planting a specific failure mode that
the downstream diagnostics are then required to detect.

Assignment is deterministic hash-mod bucketing via `src.assignment.bucketing`
(see Phase 2). Scenarios that plant assignment-side bugs (e.g. SRM) can pass
non-uniform `weights` to bias the split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from src.assignment.bucketing import make_exposures

from .events import TreatmentEffect, generate_events
from .users import generate_users


DEFAULT_DB_PATH = Path("data") / "warehouse.duckdb"


@dataclass
class Scenario:
    name: str
    users: pd.DataFrame
    events: pd.DataFrame
    exposures: pd.DataFrame
    ground_truth: dict[str, Any] = field(default_factory=dict)


def clean_lift(
    n_users: int = 10_000,
    experiment_days: int = 28,
    true_lift: float = 0.05,
    experiment_start: pd.Timestamp = pd.Timestamp("2026-01-01"),
    seed: int = 42,
) -> Scenario:
    """Variant truly +`true_lift` relative conversion, uniform across segments.

    No SRM, no novelty, no segment heterogeneity, no guardrail violation. This
    is the baseline scenario that every downstream method must handle cleanly.
    """
    experiment_id = "exp_clean_lift"
    users = generate_users(n_users, experiment_start=experiment_start, seed=seed)
    exposures = make_exposures(
        user_ids=users["user_id"].tolist(),
        experiment_id=experiment_id,
        variants=("control", "treatment"),
        weights=(0.5, 0.5),
        exposed_at=experiment_start,
    )
    effects = {
        "control": TreatmentEffect(),
        "treatment": TreatmentEffect(conversion_lift=true_lift),
    }
    events = generate_events(
        users, exposures, experiment_start, experiment_days, effects, seed=seed + 2
    )
    ground_truth = {
        "scenario": "clean_lift",
        "experiment_id": experiment_id,
        "true_lift_relative": true_lift,
        "true_assignment_ratio": 0.5,
        "primary_metric": "conversion_rate",
        "experiment_start": pd.Timestamp(experiment_start),
        "experiment_days": experiment_days,
    }
    return Scenario(
        name="clean_lift",
        users=users,
        events=events,
        exposures=exposures,
        ground_truth=ground_truth,
    )


def heterogeneous(
    n_users: int = 10_000,
    experiment_days: int = 28,
    experiment_start: pd.Timestamp = pd.Timestamp("2026-01-01"),
    segment_lifts: dict[str, float] | None = None,
    seed: int = 42,
) -> Scenario:
    """Variant lift varies sharply by segment — designed to fool an aggregate test.

    Default lifts: power_user +20%, casual -5%, new_signup +10%, enterprise -2%.
    The aggregate effect is muddy (a mix of wins and losses weighted by segment
    population), but the HTE estimator should still recover the per-segment
    direction and rank-order.
    """
    if segment_lifts is None:
        segment_lifts = {
            "power_user": 0.20,
            "casual": -0.05,
            "new_signup": 0.10,
            "enterprise": -0.02,
        }
    experiment_id = "exp_heterogeneous"
    users = generate_users(n_users, experiment_start=experiment_start, seed=seed)
    exposures = make_exposures(
        user_ids=users["user_id"].tolist(),
        experiment_id=experiment_id,
        variants=("control", "treatment"),
        weights=(0.5, 0.5),
        exposed_at=experiment_start,
    )
    effects = {
        "control": TreatmentEffect(),
        "treatment": TreatmentEffect(
            conversion_lift=0.0,
            conversion_lift_by_segment=segment_lifts,
        ),
    }
    events = generate_events(
        users, exposures, experiment_start, experiment_days, effects, seed=seed + 2
    )
    ground_truth = {
        "scenario": "heterogeneous",
        "experiment_id": experiment_id,
        "true_lift_relative_by_segment": dict(segment_lifts),
        "true_assignment_ratio": 0.5,
        "primary_metric": "conversion_rate",
        "experiment_start": pd.Timestamp(experiment_start),
        "experiment_days": experiment_days,
    }
    return Scenario(
        name="heterogeneous",
        users=users,
        events=events,
        exposures=exposures,
        ground_truth=ground_truth,
    )


def _biased_assignment(
    users: pd.DataFrame,
    *,
    experiment_id: str,
    p_treatment_by_segment: dict[str, float],
    experiment_start: pd.Timestamp,
    seed: int,
) -> pd.DataFrame:
    """Per-segment Bernoulli assignment. Bypasses deterministic hash-mod —
    used to plant scenarios whose pathology *is* the assignment bias
    (Simpson's, aa_drift). Real-world analogue: a bucketer that takes a
    segment-correlated input (e.g. signup_date) and so produces an
    assignment that's not independent of segment."""
    rng = np.random.default_rng(seed)
    n = len(users)
    p_vec = users["segment"].map(p_treatment_by_segment).to_numpy(dtype=float)
    if np.isnan(p_vec).any():
        missing = users["segment"][np.isnan(p_vec)].unique().tolist()
        raise ValueError(f"p_treatment_by_segment missing keys: {missing}")
    is_treatment = rng.random(n) < p_vec
    variants = np.where(is_treatment, "treatment", "control")
    return pd.DataFrame(
        {
            "user_id": users["user_id"].to_numpy(),
            "experiment_id": experiment_id,
            "variant": variants,
            "exposed_at": pd.Timestamp(experiment_start),
        }
    )


def srm_bug(
    n_users: int = 10_000,
    experiment_days: int = 28,
    actual_treatment_share: float = 0.55,
    true_lift: float = 0.05,
    experiment_start: pd.Timestamp = pd.Timestamp("2026-01-01"),
    seed: int = 42,
) -> Scenario:
    """Bucketer mis-configured to a 55/45 split. Conversion lift is real (+5%),
    but the SRM check must fire and block any read of the primary metric —
    you can't trust an effect estimate from a biased assignment.
    """
    experiment_id = "exp_srm_bug"
    users = generate_users(n_users, experiment_start=experiment_start, seed=seed)
    exposures = make_exposures(
        user_ids=users["user_id"].tolist(),
        experiment_id=experiment_id,
        variants=("control", "treatment"),
        weights=(1.0 - actual_treatment_share, actual_treatment_share),
        exposed_at=experiment_start,
    )
    effects = {
        "control": TreatmentEffect(),
        "treatment": TreatmentEffect(conversion_lift=true_lift),
    }
    events = generate_events(
        users, exposures, experiment_start, experiment_days, effects, seed=seed + 2
    )
    ground_truth = {
        "scenario": "srm_bug",
        "experiment_id": experiment_id,
        "true_lift_relative": true_lift,
        "true_assignment_ratio": actual_treatment_share,
        "expected_assignment_ratio": 0.5,
        "primary_metric": "conversion_rate",
        "experiment_start": pd.Timestamp(experiment_start),
        "experiment_days": experiment_days,
    }
    return Scenario(
        name="srm_bug",
        users=users,
        events=events,
        exposures=exposures,
        ground_truth=ground_truth,
    )


def novelty_effect(
    n_users: int = 10_000,
    experiment_days: int = 28,
    initial_lift: float = 0.10,
    final_lift: float = 0.0,
    experiment_start: pd.Timestamp = pd.Timestamp("2026-01-01"),
    seed: int = 42,
) -> Scenario:
    """Variant +10% on day 0, linearly decaying to +0% by the last day.

    Naive analyses that average over the full window will see something
    like +5% and ship — but the user-experience effect on day-30 visitors
    is zero. The novelty detector picks this up from the daily slope.
    """
    if initial_lift == 0:
        multipliers = [0.0] * experiment_days
    else:
        end_mult = final_lift / initial_lift
        multipliers = list(np.linspace(1.0, end_mult, experiment_days))
    experiment_id = "exp_novelty"
    users = generate_users(n_users, experiment_start=experiment_start, seed=seed)
    exposures = make_exposures(
        user_ids=users["user_id"].tolist(),
        experiment_id=experiment_id,
        variants=("control", "treatment"),
        weights=(0.5, 0.5),
        exposed_at=experiment_start,
    )
    effects = {
        "control": TreatmentEffect(),
        "treatment": TreatmentEffect(
            conversion_lift=initial_lift,
            daily_lift_multiplier=multipliers,
        ),
    }
    events = generate_events(
        users, exposures, experiment_start, experiment_days, effects, seed=seed + 2
    )
    ground_truth = {
        "scenario": "novelty_effect",
        "experiment_id": experiment_id,
        "initial_lift_relative": initial_lift,
        "final_lift_relative": final_lift,
        "true_assignment_ratio": 0.5,
        "primary_metric": "conversion_rate",
        "experiment_start": pd.Timestamp(experiment_start),
        "experiment_days": experiment_days,
    }
    return Scenario(
        name="novelty_effect",
        users=users,
        events=events,
        exposures=exposures,
        ground_truth=ground_truth,
    )


def simpsons(
    n_users: int = 10_000,
    experiment_days: int = 28,
    within_segment_lift: float = 0.05,
    p_treatment_by_segment: dict[str, float] | None = None,
    experiment_start: pd.Timestamp = pd.Timestamp("2026-01-01"),
    seed: int = 42,
) -> Scenario:
    """Variant wins +5% in every segment, but loses overall — Simpson's reversal.

    Plant: segment-correlated assignment over-represents low-converting
    new_signups in treatment and high-converting power_users in control.
    The overall ratio stays near 50/50 (so SRM doesn't fire), but the
    per-segment population shift means the aggregate test sees treatment
    losing despite winning every subgroup.
    """
    if p_treatment_by_segment is None:
        # Calibrated so overall ratio ≈ 0.50 (no SRM trigger) but per-segment
        # mix shifts toward low-converting new_signups in the treatment arm.
        p_treatment_by_segment = {
            "power_user": 0.20,
            "casual": 0.50,
            "new_signup": 0.70,
            "enterprise": 0.55,
        }
    experiment_id = "exp_simpsons"
    users = generate_users(n_users, experiment_start=experiment_start, seed=seed)
    exposures = _biased_assignment(
        users,
        experiment_id=experiment_id,
        p_treatment_by_segment=p_treatment_by_segment,
        experiment_start=experiment_start,
        seed=seed + 1,
    )
    effects = {
        "control": TreatmentEffect(),
        "treatment": TreatmentEffect(conversion_lift=within_segment_lift),
    }
    events = generate_events(
        users, exposures, experiment_start, experiment_days, effects, seed=seed + 2
    )
    ground_truth = {
        "scenario": "simpsons",
        "experiment_id": experiment_id,
        "within_segment_lift_relative": within_segment_lift,
        "p_treatment_by_segment": dict(p_treatment_by_segment),
        "primary_metric": "conversion_rate",
        "experiment_start": pd.Timestamp(experiment_start),
        "experiment_days": experiment_days,
    }
    return Scenario(
        name="simpsons",
        users=users,
        events=events,
        exposures=exposures,
        ground_truth=ground_truth,
    )


def guardrail_violation(
    n_users: int = 10_000,
    experiment_days: int = 28,
    conversion_lift: float = 0.05,
    latency_lift_ms: float = 30.0,
    experiment_start: pd.Timestamp = pd.Timestamp("2026-01-01"),
    seed: int = 42,
) -> Scenario:
    """Treatment delivers a real conversion lift but tanks latency.

    The primary-metric read says "ship" — and a platform without a
    guardrail check would. The latency guardrail catches the regression
    (+30ms on a ~180ms baseline ≈ +17% relative, well past the 5% threshold).
    """
    experiment_id = "exp_guardrail"
    users = generate_users(n_users, experiment_start=experiment_start, seed=seed)
    exposures = make_exposures(
        user_ids=users["user_id"].tolist(),
        experiment_id=experiment_id,
        variants=("control", "treatment"),
        weights=(0.5, 0.5),
        exposed_at=experiment_start,
    )
    effects = {
        "control": TreatmentEffect(),
        "treatment": TreatmentEffect(
            conversion_lift=conversion_lift,
            latency_lift_ms=latency_lift_ms,
        ),
    }
    events = generate_events(
        users, exposures, experiment_start, experiment_days, effects, seed=seed + 2
    )
    ground_truth = {
        "scenario": "guardrail_violation",
        "experiment_id": experiment_id,
        "true_lift_relative": conversion_lift,
        "true_latency_lift_ms": latency_lift_ms,
        "true_assignment_ratio": 0.5,
        "primary_metric": "conversion_rate",
        "guardrail_metric": "mean_latency_ms",
        "experiment_start": pd.Timestamp(experiment_start),
        "experiment_days": experiment_days,
    }
    return Scenario(
        name="guardrail_violation",
        users=users,
        events=events,
        exposures=exposures,
        ground_truth=ground_truth,
    )


def aa_drift(
    n_users: int = 10_000,
    experiment_days: int = 28,
    p_treatment_by_segment: dict[str, float] | None = None,
    experiment_start: pd.Timestamp = pd.Timestamp("2026-01-01"),
    seed: int = 42,
) -> Scenario:
    """No real treatment effect — but assignment is biased so high-conversion
    power_users disproportionately land in treatment. The primary-metric test
    will spuriously reject H0 (false positive). An A/A check on pre-period
    data would have caught the imbalance before it ever turned into a wrong
    "ship" decision.
    """
    if p_treatment_by_segment is None:
        # Tuned so overall ratio is ~50/50 (no direct SRM), but the bias on
        # power_user vs new_signup is enough to drive a spurious primary read.
        p_treatment_by_segment = {
            "power_user": 0.80,
            "casual": 0.523,
            "new_signup": 0.20,
            "enterprise": 0.523,
        }
    experiment_id = "exp_aa_drift"
    users = generate_users(n_users, experiment_start=experiment_start, seed=seed)
    exposures = _biased_assignment(
        users,
        experiment_id=experiment_id,
        p_treatment_by_segment=p_treatment_by_segment,
        experiment_start=experiment_start,
        seed=seed + 1,
    )
    effects = {
        "control": TreatmentEffect(),
        "treatment": TreatmentEffect(),  # zero — no real lift
    }
    events = generate_events(
        users, exposures, experiment_start, experiment_days, effects, seed=seed + 2
    )
    ground_truth = {
        "scenario": "aa_drift",
        "experiment_id": experiment_id,
        "true_lift_relative": 0.0,
        "p_treatment_by_segment": dict(p_treatment_by_segment),
        "primary_metric": "conversion_rate",
        "experiment_start": pd.Timestamp(experiment_start),
        "experiment_days": experiment_days,
    }
    return Scenario(
        name="aa_drift",
        users=users,
        events=events,
        exposures=exposures,
        ground_truth=ground_truth,
    )


SCENARIOS: dict[str, Any] = {
    "clean_lift": clean_lift,
    "heterogeneous": heterogeneous,
    "srm_bug": srm_bug,
    "novelty_effect": novelty_effect,
    "simpsons": simpsons,
    "guardrail_violation": guardrail_violation,
    "aa_drift": aa_drift,
}


def write_to_duckdb(
    scenario: Scenario,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> Path:
    """Replace `users`, `events`, `exposures` tables in DuckDB with scenario data.

    One scenario lives in the warehouse at a time. To swap scenarios, re-run
    this module with a different name and rebuild the dbt marts.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    users_df = scenario.users  # noqa: F841 — referenced by DuckDB replacement scan
    events_df = scenario.events  # noqa: F841
    exposures_df = scenario.exposures  # noqa: F841
    con = duckdb.connect(str(db_path))
    try:
        con.execute("CREATE OR REPLACE TABLE users AS SELECT * FROM users_df")
        con.execute("CREATE OR REPLACE TABLE events AS SELECT * FROM events_df")
        con.execute("CREATE OR REPLACE TABLE exposures AS SELECT * FROM exposures_df")
    finally:
        con.close()
    return db_path


def _summary(scenario: Scenario) -> str:
    e = scenario.events
    n_sessions = int((e["event_type"] == "session").sum())
    n_conv = int((e["event_type"] == "conversion").sum())
    by_var = (
        scenario.exposures["variant"].value_counts().rename_axis("variant").to_dict()
    )
    return (
        f"scenario={scenario.name} users={len(scenario.users):,} "
        f"sessions={n_sessions:,} conversions={n_conv:,} "
        f"variants={by_var}"
    )


if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "clean_lift"
    if name not in SCENARIOS:
        print(
            f"Unknown scenario {name!r}. Available: {', '.join(SCENARIOS.keys())}"
        )
        sys.exit(1)
    s = SCENARIOS[name]()
    path = write_to_duckdb(s)
    print(_summary(s))
    print(f"wrote -> {path}")
