"""Deterministic hash-mod variant assignment.

`user_bucket(user_id, salt)` produces a stable integer in [0, n_buckets) from
md5(salt || user_id). Same `(user_id, salt)` always lands in the same bucket,
across processes, machines, and restarts — that's the whole point. The `salt`
is the experiment_id (standalone) or the layer_id (when experiments share a
layer for mutual exclusion).

Bucket ranges within `[0, n_buckets)` are mapped to variants by cumulative
weight, so for a 50/50 split with n_buckets=10_000, buckets 0–4999 are control
and 5000–9999 are treatment. This deterministic-range approach (vs. a second
hash for the variant) keeps everything reproducible from `(user_id, salt)`
alone and composes cleanly with layered experiments in `layers.py`.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_N_BUCKETS: int = 10_000


def user_bucket(user_id: str, salt: str, n_buckets: int = DEFAULT_N_BUCKETS) -> int:
    """Hash `salt:user_id` with md5 and reduce mod `n_buckets`.

    Uses the leading 8 bytes of the digest as an unsigned int. Different salts
    hash the same user to (effectively) independent buckets.
    """
    raw = f"{salt}:{user_id}".encode("utf-8")
    digest = hashlib.md5(raw).digest()
    val = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return val % n_buckets


def _variant_thresholds(
    weights: Sequence[float],
    n_buckets: int,
) -> list[int]:
    """Cumulative bucket cut-points for the given weights."""
    total = sum(weights)
    if not (1.0 - 1e-9) <= total <= (1.0 + 1e-9):
        raise ValueError(f"variant weights must sum to 1.0, got {total}")
    cum = 0.0
    cuts: list[int] = []
    for w in weights:
        cum += w
        cuts.append(int(round(cum * n_buckets)))
    cuts[-1] = n_buckets
    return cuts


def assign(
    user_id: str,
    experiment_id: str,
    salt: str | None = None,
    variants: Sequence[str] = ("control", "treatment"),
    weights: Sequence[float] | None = None,
    n_buckets: int = DEFAULT_N_BUCKETS,
) -> str:
    """Return the variant for `user_id` in `experiment_id`.

    `salt` defaults to `experiment_id` for standalone experiments. When several
    experiments share a layer, pass the layer_id as `salt` so they all hash
    the same user identically and bucket-range allocation gives mutual
    exclusion (see `layers.assign_in_layer`).
    """
    if weights is None:
        weights = [1.0 / len(variants)] * len(variants)
    if len(weights) != len(variants):
        raise ValueError("variants and weights must be the same length")
    cuts = _variant_thresholds(weights, n_buckets)
    bucket = user_bucket(user_id, salt or experiment_id, n_buckets)
    for i, cut in enumerate(cuts):
        if bucket < cut:
            return variants[i]
    return variants[-1]


def make_exposures(
    user_ids: Sequence[str],
    experiment_id: str,
    salt: str | None = None,
    variants: Sequence[str] = ("control", "treatment"),
    weights: Sequence[float] | None = None,
    n_buckets: int = DEFAULT_N_BUCKETS,
    exposed_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build an exposures DataFrame `(user_id, experiment_id, variant, exposed_at)`.

    All users get an exposure row. `exposed_at` is the experiment start by
    convention; later phases will refine this to first-activity time.
    """
    user_ids = list(user_ids)
    variants_seq = [
        assign(uid, experiment_id, salt=salt, variants=variants, weights=weights, n_buckets=n_buckets)
        for uid in user_ids
    ]
    return pd.DataFrame(
        {
            "user_id": user_ids,
            "experiment_id": experiment_id,
            "variant": variants_seq,
            "exposed_at": pd.Timestamp(exposed_at) if exposed_at is not None else pd.NaT,
        }
    )
