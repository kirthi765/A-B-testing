"""Tests for the UI's data + analysis layers. The Streamlit page itself is
display-only and not unit-tested — these tests cover everything that involves
a decision (which checks to run, which recommendation to make).

The fixture builds a fresh DuckDB warehouse + dbt-shaped mart tables inside
a tmp_path so the tests don't depend on the developer's local warehouse
state (which may have been rebuilt for a different scenario).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src.simulator.scenarios import clean_lift, write_to_duckdb
from src.ui.analysis import analyze_experiment
from src.ui.data import list_experiments, load_experiment


def _build_marts(db_path: Path) -> None:
    """Hand-roll the dbt-shaped tables instead of running dbt — keeps the test
    fast and independent of the dbt CLI."""
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE OR REPLACE VIEW stg_exposures AS
            SELECT user_id, experiment_id, variant, exposed_at FROM exposures
            """
        )
        con.execute(
            """
            CREATE OR REPLACE VIEW stg_events AS
            SELECT
                e.event_id, e.user_id, x.experiment_id, x.variant,
                e.event_type, e.ts, e.value AS latency_ms
            FROM events e
            JOIN exposures x USING (user_id)
            WHERE e.ts >= x.exposed_at
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE fct_experiment_metrics AS
            WITH exposures_x AS (
                SELECT x.experiment_id, x.variant, x.user_id, u.segment
                FROM stg_exposures x
                LEFT JOIN users u USING (user_id)
            ),
            rolled AS (
                SELECT
                    x.experiment_id, x.variant, x.user_id, x.segment,
                    SUM(CASE WHEN e.event_type='session'    THEN 1 ELSE 0 END) AS n_sessions,
                    SUM(CASE WHEN e.event_type='conversion' THEN 1 ELSE 0 END) AS n_conversions,
                    AVG(CASE WHEN e.event_type='session' THEN e.latency_ms END) AS mean_latency_ms,
                    MAX(CASE WHEN e.event_type='session' THEN e.latency_ms END) AS max_latency_ms
                FROM exposures_x x
                LEFT JOIN stg_events e USING (experiment_id, user_id)
                GROUP BY 1,2,3,4
            )
            SELECT
                experiment_id, variant, user_id, segment,
                n_sessions, n_conversions,
                CAST(n_conversions AS DOUBLE) / NULLIF(n_sessions, 0) AS conversion_rate,
                (n_conversions > 0) AS converted,
                mean_latency_ms, max_latency_ms
            FROM rolled
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE fct_experiment_daily AS
            SELECT
                experiment_id, variant,
                CAST(date_trunc('day', ts) AS DATE) AS event_date,
                COUNT(DISTINCT user_id) AS n_active_users,
                SUM(CASE WHEN event_type='session'    THEN 1 ELSE 0 END) AS n_sessions,
                SUM(CASE WHEN event_type='conversion' THEN 1 ELSE 0 END) AS n_conversions,
                CAST(SUM(CASE WHEN event_type='conversion' THEN 1 ELSE 0 END) AS DOUBLE)
                    / NULLIF(SUM(CASE WHEN event_type='session' THEN 1 ELSE 0 END), 0) AS conversion_rate,
                AVG(CASE WHEN event_type='session' THEN latency_ms END) AS mean_latency_ms
            FROM stg_events
            GROUP BY 1,2,3
            ORDER BY 1,2,3
            """
        )
    finally:
        con.close()


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory) -> Path:
    db_path = tmp_path_factory.mktemp("ui_test") / "warehouse.duckdb"
    s = clean_lift(n_users=3_000, experiment_days=14, true_lift=0.05, seed=42)
    write_to_duckdb(s, db_path=db_path)
    _build_marts(db_path)
    return db_path


# --- data loaders -----------------------------------------------------------


def test_list_experiments_returns_known_id(warehouse):
    ids = list_experiments(warehouse)
    assert ids == ["exp_clean_lift"]


def test_list_experiments_on_missing_warehouse_returns_empty(tmp_path):
    assert list_experiments(tmp_path / "nope.duckdb") == []


def test_load_experiment_shapes_match_warehouse(warehouse):
    data = load_experiment("exp_clean_lift", warehouse)
    assert len(data.metrics) == 3_000
    assert {"experiment_id", "variant", "user_id", "segment", "converted"} <= set(
        data.metrics.columns
    )
    assert data.n_total_events > 0
    assert data.start_date <= data.end_date
    # Daily frame: one row per (variant, day) across the experiment window.
    assert len(data.daily) > 0
    assert set(data.daily["variant"].unique()) == {"control", "treatment"}
    assert pd.api.types.is_datetime64_any_dtype(data.daily["event_date"])


def test_load_experiment_raises_on_missing_id(warehouse):
    with pytest.raises(ValueError):
        load_experiment("exp_doesnt_exist", warehouse)


# --- analysis pipeline ------------------------------------------------------


def test_analyze_experiment_runs_all_checks(warehouse):
    data = load_experiment("exp_clean_lift", warehouse)
    report = analyze_experiment(data)
    # All headline fields populated.
    assert report.experiment_id == "exp_clean_lift"
    assert report.srm.name == "srm"
    assert report.primary.method_name == "two_proportion_z_test"
    assert report.primary_sequential.method_name == "msprt"
    assert len(report.guardrails) == 1
    assert report.guardrails[0].name == "guardrail:mean_latency_ms"
    assert report.novelty.name == "novelty"
    assert report.simpsons.name == "simpsons"
    # Segments — 4 (one per planted simulator segment), all >= min_n.
    assert {s.segment for s in report.segments} == {
        "power_user", "casual", "new_signup", "enterprise",
    }


def test_clean_lift_is_recommended_to_ship(warehouse):
    """`clean_lift` plants a real +5% conversion lift, no SRM, no novelty,
    no latency change. With n=3k users (1.5k/arm), this should clear all
    guardrails and the recommendation should be 'Ship' — most of the time.
    If sampling noise puts the CI on zero, accept 'Iterate' as a soft pass."""
    data = load_experiment("exp_clean_lift", warehouse)
    report = analyze_experiment(data)
    assert report.recommendation in {"Ship", "Iterate"}
    assert report.srm.status == "pass"
    # Latency wasn't planted to move.
    assert report.guardrails[0].status in {"pass", "warn"}


def test_recommendation_reasons_explain_the_call(warehouse):
    data = load_experiment("exp_clean_lift", warehouse)
    report = analyze_experiment(data)
    assert len(report.recommendation_reasons) > 0
    # The first reason should reference the primary metric.
    assert "primary metric" in report.recommendation_reasons[0].lower()


# --- Streamlit page smoke test ---------------------------------------------


def test_streamlit_page_renders_without_exception():
    """Render `src/ui/app.py` in-process via Streamlit's AppTest harness. If
    the page crashes (template error, missing import, bad attribute access)
    this surfaces it as `at.exception` — the HTTP-level probe in the boot
    smoke check can't see Python exceptions inside the render loop.

    The test deliberately does *not* depend on a populated warehouse: if the
    warehouse is missing or empty, the app uses `st.error()` + `st.stop()`,
    which AppTest treats as a clean stop, not an exception.
    """
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("src/ui/app.py", default_timeout=60)
    at.run()
    assert not at.exception, f"app raised: {at.exception}"
