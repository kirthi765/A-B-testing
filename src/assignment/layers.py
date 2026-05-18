"""Mutually-exclusive experiment layers.

A `Layer` is a salt namespace that owns a contiguous `[0, n_buckets)` bucket
space. Experiments inside the layer claim non-overlapping bucket ranges
proportional to their `traffic_pct`. A user's layer bucket — `user_bucket(uid,
layer_id)` — uniquely determines which experiment (if any) they're enrolled in
*and* their variant within that experiment.

This gives two guarantees the diagnostics will rely on:
  1. *Mutual exclusion within a layer.* A user cannot be assigned to two
     experiments in the same layer — by construction of disjoint ranges.
  2. *Independence across layers.* Different layer ids produce uncorrelated
     md5 hashes, so a user's assignment in layer A is independent of their
     assignment in layer B.

`assign_in_layer` returns `None` when the user falls into the layer's unclaimed
("holdout") bucket range, which is realistic — not every layer is 100%
allocated to running experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd

from .bucketing import DEFAULT_N_BUCKETS, _variant_thresholds, user_bucket


@dataclass(frozen=True)
class ExperimentInLayer:
    """One experiment's claim on a slice of its enclosing layer."""

    experiment_id: str
    traffic_pct: float
    variants: tuple[str, ...] = ("control", "treatment")
    variant_weights: tuple[float, ...] | None = None  # None = uniform across variants


@dataclass
class Layer:
    layer_id: str
    experiments: list[ExperimentInLayer]
    n_buckets: int = DEFAULT_N_BUCKETS
    _ranges: list[tuple[int, int]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        total = sum(e.traffic_pct for e in self.experiments)
        if total > 1.0 + 1e-9:
            raise ValueError(
                f"experiments in layer {self.layer_id!r} claim {total:.4f}"
                f" of traffic, which exceeds 1.0"
            )
        cursor = 0
        ranges: list[tuple[int, int]] = []
        for e in self.experiments:
            width = int(round(e.traffic_pct * self.n_buckets))
            ranges.append((cursor, cursor + width))
            cursor += width
        if cursor > self.n_buckets:
            # Rounding can push past by 1 — clamp the last range.
            start, _ = ranges[-1]
            ranges[-1] = (start, self.n_buckets)
        self._ranges = ranges


def assign_in_layer(user_id: str, layer: Layer) -> tuple[str, str] | None:
    """Return `(experiment_id, variant)` or `None` if the user is in holdout."""
    bucket = user_bucket(user_id, layer.layer_id, layer.n_buckets)
    for exp, (start, end) in zip(layer.experiments, layer._ranges):
        if start <= bucket < end:
            width = end - start
            weights: Sequence[float] = (
                exp.variant_weights
                if exp.variant_weights is not None
                else [1.0 / len(exp.variants)] * len(exp.variants)
            )
            cuts = _variant_thresholds(weights, width)
            sub = bucket - start
            for i, cut in enumerate(cuts):
                if sub < cut:
                    return (exp.experiment_id, exp.variants[i])
            return (exp.experiment_id, exp.variants[-1])
    return None


def make_layered_exposures(
    user_ids: Sequence[str],
    layer: Layer,
    exposed_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build an exposures DataFrame across every experiment in `layer`.

    Users in the layer's holdout (unclaimed bucket range) are omitted — they
    aren't exposed to any experiment in this layer.
    """
    rows = []
    for uid in user_ids:
        result = assign_in_layer(uid, layer)
        if result is None:
            continue
        exp_id, var = result
        rows.append((uid, exp_id, var))
    df = pd.DataFrame(rows, columns=["user_id", "experiment_id", "variant"])
    df["exposed_at"] = pd.Timestamp(exposed_at) if exposed_at is not None else pd.NaT
    return df
