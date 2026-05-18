"""Tests for `src.assignment.layers` — mutually-exclusive experiment layers.

The plan calls out "layers don't bleed" — i.e. a user enrolled in experiment A
within layer L can never also appear in experiment B within the same layer.
We also confirm that cross-layer assignments are statistically independent.
"""

from __future__ import annotations

import math

import pytest

from src.assignment.layers import (
    ExperimentInLayer,
    Layer,
    assign_in_layer,
    make_layered_exposures,
)


def _uids(n: int) -> list[str]:
    return [f"u_{i:07d}" for i in range(n)]


# --- traffic allocation ------------------------------------------------------


def test_full_layer_50_50_two_experiments():
    layer = Layer(
        layer_id="layer_growth",
        experiments=[
            ExperimentInLayer("exp_a", traffic_pct=0.5),
            ExperimentInLayer("exp_b", traffic_pct=0.5),
        ],
    )
    results = [assign_in_layer(u, layer) for u in _uids(20_000)]
    in_a = sum(1 for r in results if r and r[0] == "exp_a")
    in_b = sum(1 for r in results if r and r[0] == "exp_b")
    in_holdout = sum(1 for r in results if r is None)
    assert in_holdout == 0  # full layer → nobody in holdout
    assert math.isclose(in_a / 20_000, 0.5, abs_tol=0.01)
    assert math.isclose(in_b / 20_000, 0.5, abs_tol=0.01)


def test_partial_layer_leaves_holdout():
    layer = Layer(
        layer_id="layer_partial",
        experiments=[
            ExperimentInLayer("exp_only", traffic_pct=0.3),
        ],
    )
    results = [assign_in_layer(u, layer) for u in _uids(20_000)]
    in_exp = sum(1 for r in results if r is not None)
    in_holdout = sum(1 for r in results if r is None)
    assert math.isclose(in_exp / 20_000, 0.3, abs_tol=0.01)
    assert math.isclose(in_holdout / 20_000, 0.7, abs_tol=0.01)


def test_traffic_overflow_rejected():
    with pytest.raises(ValueError):
        Layer(
            layer_id="layer_oops",
            experiments=[
                ExperimentInLayer("exp_a", traffic_pct=0.6),
                ExperimentInLayer("exp_b", traffic_pct=0.6),
            ],
        )


# --- mutual exclusion --------------------------------------------------------


def test_layers_dont_bleed_no_user_in_two_experiments():
    """Plan's literal test: a user assigned to A in layer L is *never* in B."""
    layer = Layer(
        layer_id="layer_mx",
        experiments=[
            ExperimentInLayer("exp_a", traffic_pct=0.4),
            ExperimentInLayer("exp_b", traffic_pct=0.4),
            ExperimentInLayer("exp_c", traffic_pct=0.2),
        ],
    )
    seen: dict[str, str] = {}
    for u in _uids(20_000):
        result = assign_in_layer(u, layer)
        if result is None:
            continue
        exp_id, _ = result
        if u in seen:
            assert seen[u] == exp_id, "user assigned to two experiments in the same layer"
        seen[u] = exp_id


def test_within_experiment_50_50_variant_split():
    layer = Layer(
        layer_id="layer_var",
        experiments=[
            ExperimentInLayer(
                "exp_x",
                traffic_pct=0.5,
                variants=("control", "treatment"),
                variant_weights=(0.5, 0.5),
            ),
        ],
    )
    variants = [assign_in_layer(u, layer) for u in _uids(20_000)]
    in_exp = [v for v in variants if v is not None]
    n_treat = sum(1 for _, v in in_exp if v == "treatment")
    n_ctrl = sum(1 for _, v in in_exp if v == "control")
    assert math.isclose(n_treat / len(in_exp), 0.5, abs_tol=0.02)
    assert math.isclose(n_ctrl / len(in_exp), 0.5, abs_tol=0.02)


def test_within_experiment_three_variants():
    layer = Layer(
        layer_id="layer_3var",
        experiments=[
            ExperimentInLayer(
                "exp_y",
                traffic_pct=1.0,
                variants=("control", "treatment_a", "treatment_b"),
                variant_weights=(1 / 3, 1 / 3, 1 / 3),
            ),
        ],
    )
    variants = [assign_in_layer(u, layer) for u in _uids(30_000)]
    for name in ("control", "treatment_a", "treatment_b"):
        share = sum(1 for r in variants if r and r[1] == name) / 30_000
        assert math.isclose(share, 1 / 3, abs_tol=0.02)


# --- cross-layer independence -----------------------------------------------


def test_cross_layer_independence():
    """A 2x2 contingency of variants across two layers should look ~independent."""
    layer1 = Layer(
        layer_id="layer_ui",
        experiments=[ExperimentInLayer("exp_ui", traffic_pct=1.0)],
    )
    layer2 = Layer(
        layer_id="layer_pricing",
        experiments=[ExperimentInLayer("exp_pricing", traffic_pct=1.0)],
    )
    uids = _uids(20_000)
    contingency = {
        ("control", "control"): 0,
        ("control", "treatment"): 0,
        ("treatment", "control"): 0,
        ("treatment", "treatment"): 0,
    }
    for u in uids:
        v1 = assign_in_layer(u, layer1)[1]
        v2 = assign_in_layer(u, layer2)[1]
        contingency[(v1, v2)] += 1
    expected_per_cell = 20_000 / 4  # 5,000
    for cell, count in contingency.items():
        assert math.isclose(count, expected_per_cell, rel_tol=0.05), (
            f"cell {cell} count {count} far from expected {expected_per_cell}"
        )


def test_same_user_different_layer_can_differ():
    layer1 = Layer(
        layer_id="layer_a",
        experiments=[ExperimentInLayer("e1", traffic_pct=1.0)],
    )
    layer2 = Layer(
        layer_id="layer_b",
        experiments=[ExperimentInLayer("e2", traffic_pct=1.0)],
    )
    # Aggregate: for 1000 users, we expect ~500 to land on the same variant
    # across layers and ~500 to differ. If always-equal or always-different,
    # layers aren't isolated.
    same = sum(
        1
        for u in _uids(1000)
        if assign_in_layer(u, layer1)[1] == assign_in_layer(u, layer2)[1]
    )
    assert 350 < same < 650


# --- exposures helper --------------------------------------------------------


def test_make_layered_exposures_excludes_holdout():
    layer = Layer(
        layer_id="layer_holdout",
        experiments=[ExperimentInLayer("exp_only", traffic_pct=0.5)],
    )
    df = make_layered_exposures(_uids(2000), layer)
    assert set(df.columns) >= {"user_id", "experiment_id", "variant", "exposed_at"}
    assert math.isclose(len(df) / 2000, 0.5, abs_tol=0.03)
    assert set(df["experiment_id"].unique()) == {"exp_only"}


def test_make_layered_exposures_with_multiple_experiments():
    layer = Layer(
        layer_id="layer_multi",
        experiments=[
            ExperimentInLayer("exp_a", traffic_pct=0.5),
            ExperimentInLayer("exp_b", traffic_pct=0.5),
        ],
    )
    df = make_layered_exposures(_uids(2000), layer)
    # Mutual exclusion at the user level even when we shape it into a DataFrame.
    assert df["user_id"].is_unique
    assert set(df["experiment_id"].unique()) == {"exp_a", "exp_b"}
