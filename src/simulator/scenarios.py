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


def write_to_duckdb(
    scenario: Scenario,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> Path:
    """Replace `users`, `events`, `exposures` tables in DuckDB with scenario data.

    Phase 1 keeps one scenario in the warehouse at a time. When scenarios
    multiply (Phase 1.x), this will be extended to namespace by scenario.
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
    s = clean_lift()
    path = write_to_duckdb(s)
    print(_summary(s))
    print(f"wrote -> {path}")
