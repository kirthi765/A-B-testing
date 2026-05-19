"""Read-only loaders that pull one experiment's data from the DuckDB warehouse.

The UI never recomputes the marts — it assumes `fct_experiment_metrics` and
`fct_experiment_daily` exist and are fresh. If you re-run the simulator with
a different scenario, rebuild the marts (`cd dbt_project && dbt build`)
before opening the UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd


DEFAULT_DB_PATH = Path("data") / "warehouse.duckdb"


@dataclass(frozen=True)
class ExperimentData:
    experiment_id: str
    metrics: pd.DataFrame   # one row per user — from fct_experiment_metrics
    daily: pd.DataFrame     # one row per (variant, day) — from fct_experiment_daily
    exposures: pd.DataFrame # raw exposures (used by the SRM check)
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    n_total_events: int


def list_experiments(db_path: str | Path = DEFAULT_DB_PATH) -> list[str]:
    """All experiment_ids present in the `exposures` table, sorted."""
    db_path = Path(db_path)
    if not db_path.exists():
        return []
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute(
            "SELECT DISTINCT experiment_id FROM exposures ORDER BY 1"
        ).fetchall()
    finally:
        con.close()
    return [r[0] for r in rows]


def load_experiment(
    experiment_id: str,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> ExperimentData:
    """Pull all per-experiment data the UI needs in a single read-only session."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        metrics = con.execute(
            "SELECT * FROM fct_experiment_metrics WHERE experiment_id = ?",
            [experiment_id],
        ).df()
        daily = con.execute(
            "SELECT * FROM fct_experiment_daily WHERE experiment_id = ? "
            "ORDER BY event_date, variant",
            [experiment_id],
        ).df()
        exposures = con.execute(
            "SELECT * FROM exposures WHERE experiment_id = ?",
            [experiment_id],
        ).df()
        n_events = con.execute(
            "SELECT COUNT(*) FROM stg_events WHERE experiment_id = ?",
            [experiment_id],
        ).fetchone()[0]
    finally:
        con.close()

    if exposures.empty:
        raise ValueError(f"experiment_id={experiment_id!r} not found in warehouse")

    daily["event_date"] = pd.to_datetime(daily["event_date"])
    exposures["exposed_at"] = pd.to_datetime(exposures["exposed_at"])

    return ExperimentData(
        experiment_id=experiment_id,
        metrics=metrics,
        daily=daily,
        exposures=exposures,
        start_date=pd.Timestamp(exposures["exposed_at"].min()),
        end_date=pd.Timestamp(daily["event_date"].max()),
        n_total_events=int(n_events),
    )
