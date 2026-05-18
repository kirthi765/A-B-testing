"""End-to-end smoke tests on the clean_lift scenario.

These tests pin the *shape* of the simulator output (schema, ratios, segments
present) and the *direction* of the planted effect — they should fail loudly
if a refactor changes the data contract that downstream phases depend on.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.simulator.scenarios import clean_lift, write_to_duckdb
from src.simulator.users import SEGMENTS


@pytest.fixture(scope="module")
def scenario():
    return clean_lift(n_users=5_000, experiment_days=14, true_lift=0.05, seed=42)


def test_users_schema(scenario):
    cols = set(scenario.users.columns)
    assert {"user_id", "segment", "signup_date"} <= cols
    assert len(scenario.users) == 5_000
    assert scenario.users["user_id"].is_unique


def test_all_four_segments_present(scenario):
    segs = set(scenario.users["segment"].unique())
    assert segs == set(SEGMENTS.keys())


def test_exposures_schema_and_ratio(scenario):
    e = scenario.exposures
    assert {"user_id", "experiment_id", "variant", "exposed_at"} <= set(e.columns)
    assert set(e["variant"].unique()) == {"control", "treatment"}
    share_treat = (e["variant"] == "treatment").mean()
    assert math.isclose(share_treat, 0.5, abs_tol=0.03)


def test_events_have_sessions_and_conversions(scenario):
    types = scenario.events["event_type"].value_counts()
    assert types.get("session", 0) > 0
    assert types.get("conversion", 0) > 0
    assert scenario.events["event_id"].is_unique


def test_session_value_is_positive_latency(scenario):
    sess = scenario.events[scenario.events["event_type"] == "session"]
    assert (sess["value"] > 0).all()


def test_variant_conversion_rate_is_higher(scenario):
    e = scenario.events
    n_sess = e[e["event_type"] == "session"].groupby("user_id").size().rename("n_sess")
    n_conv = e[e["event_type"] == "conversion"].groupby("user_id").size().rename("n_conv")
    per_user = pd.concat([n_sess, n_conv], axis=1).fillna(0).reset_index()
    per_user = per_user.merge(scenario.exposures[["user_id", "variant"]], on="user_id")
    per_user["cvr"] = per_user["n_conv"] / per_user["n_sess"].clip(lower=1)
    by_var = per_user.groupby("variant")["cvr"].mean()
    assert by_var["treatment"] > by_var["control"]
    # And the lift should be roughly in the right neighborhood (loose bound — sampling noise).
    rel = (by_var["treatment"] - by_var["control"]) / by_var["control"]
    assert 0.0 < rel < 0.20


def test_event_timestamps_in_experiment_window(scenario):
    start = scenario.ground_truth["experiment_start"]
    days = scenario.ground_truth["experiment_days"]
    end = start + pd.Timedelta(days=days)
    ts = scenario.events["ts"]
    assert ts.min() >= start
    assert ts.max() < end


def test_reproducible_with_same_seed():
    a = clean_lift(n_users=500, experiment_days=7, seed=123)
    b = clean_lift(n_users=500, experiment_days=7, seed=123)
    pd.testing.assert_frame_equal(a.users, b.users)
    pd.testing.assert_frame_equal(a.exposures, b.exposures)
    pd.testing.assert_frame_equal(a.events, b.events)


def test_round_trip_to_duckdb(tmp_path, scenario):
    db = tmp_path / "warehouse.duckdb"
    write_to_duckdb(scenario, db_path=db)
    import duckdb

    con = duckdb.connect(str(db))
    try:
        nu = con.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        ne = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        nx = con.execute("SELECT COUNT(*) FROM exposures").fetchone()[0]
    finally:
        con.close()
    assert nu == len(scenario.users)
    assert ne == len(scenario.events)
    assert nx == len(scenario.exposures)
