"""Tests for `src.assignment.bucketing` — the hash-mod assignment primitives.

The plan calls these out explicitly: determinism, salt-namespacing, and
near-uniform large-N behavior. If any of these regress, every downstream
phase (SRM diagnostics, t-tests, CUPED) is built on sand.
"""

from __future__ import annotations

import math

import pytest

from src.assignment.bucketing import (
    DEFAULT_N_BUCKETS,
    _variant_thresholds,
    assign,
    make_exposures,
    user_bucket,
)


# --- determinism -------------------------------------------------------------


def test_user_bucket_is_deterministic_within_process():
    a = user_bucket("u_0000001", "exp_x")
    b = user_bucket("u_0000001", "exp_x")
    assert a == b
    assert 0 <= a < DEFAULT_N_BUCKETS


def test_user_bucket_is_stable_across_calls_with_known_value():
    # Pin a single known hash so a refactor that "fixes" the digest endianness
    # or salt format would fail loudly instead of silently shifting buckets.
    assert user_bucket("u_0000001", "exp_clean_lift") == 6130


def test_assign_is_deterministic():
    a = assign("u_0000042", "exp_x")
    b = assign("u_0000042", "exp_x")
    assert a == b


# --- salt namespacing --------------------------------------------------------


def test_same_user_diverges_across_salts():
    # Across 1000 users + two salts, the joint distribution should not be
    # perfectly collinear; if it is, the salt isn't entering the hash.
    uids = [f"u_{i:07d}" for i in range(1000)]
    matches = sum(
        1
        for u in uids
        if assign(u, "exp_a", salt="salt_a") == assign(u, "exp_a", salt="salt_b")
    )
    # With independent salts and 50/50 split, ~500/1000 should match by chance.
    # We just check it isn't 0 or 1000 (which would mean salt is ignored or
    # is the *entire* signal).
    assert 350 < matches < 650


def test_salt_defaults_to_experiment_id():
    assert assign("u_0000001", "exp_x") == assign("u_0000001", "exp_x", salt="exp_x")


# --- uniformity --------------------------------------------------------------


def test_50_50_split_is_within_one_percent_at_100k():
    uids = [f"u_{i:07d}" for i in range(100_000)]
    variants = [assign(u, "exp_uniform") for u in uids]
    share_treatment = sum(v == "treatment" for v in variants) / len(variants)
    assert math.isclose(share_treatment, 0.5, abs_tol=0.01)


def test_70_30_split_is_within_one_percent_at_100k():
    uids = [f"u_{i:07d}" for i in range(100_000)]
    variants = [
        assign(u, "exp_skewed", variants=("control", "treatment"), weights=(0.7, 0.3))
        for u in uids
    ]
    share_treatment = sum(v == "treatment" for v in variants) / len(variants)
    assert math.isclose(share_treatment, 0.3, abs_tol=0.01)


def test_three_way_split_each_within_one_percent_at_60k():
    uids = [f"u_{i:07d}" for i in range(60_000)]
    variants = [
        assign(
            u,
            "exp_3way",
            variants=("a", "b", "c"),
            weights=(1 / 3, 1 / 3, 1 / 3),
        )
        for u in uids
    ]
    for name in ("a", "b", "c"):
        share = sum(v == name for v in variants) / len(variants)
        assert math.isclose(share, 1 / 3, abs_tol=0.01)


# --- API validation ----------------------------------------------------------


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        _variant_thresholds([0.4, 0.4], n_buckets=10_000)


def test_variants_and_weights_must_match_length():
    with pytest.raises(ValueError):
        assign("u_0000001", "exp_x", variants=("a", "b"), weights=(1.0,))


# --- exposures helper --------------------------------------------------------


def test_make_exposures_schema_and_determinism():
    uids = [f"u_{i:07d}" for i in range(100)]
    exp1 = make_exposures(uids, "exp_y")
    exp2 = make_exposures(uids, "exp_y")
    assert {"user_id", "experiment_id", "variant", "exposed_at"} <= set(exp1.columns)
    assert len(exp1) == 100
    assert (exp1["variant"] == exp2["variant"]).all()
    assert set(exp1["variant"].unique()) <= {"control", "treatment"}
