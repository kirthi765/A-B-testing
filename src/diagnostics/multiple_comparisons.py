"""Benjamini–Hochberg FDR control for multiple-hypothesis reporting.

When an experiment dashboard reports N segments × M metrics, looking at 20
p-values and picking the smallest one is a classic Type-I machine. BH is
the standard correction:

  - Order the p-values ascending: p_(1) ≤ p_(2) ≤ ... ≤ p_(N).
  - Find the largest k such that p_(k) ≤ (k/N) · α.
  - Reject H0 for ranks 1..k.
  - The adjusted p-value (q-value) for rank i is min over j ≥ i of (p_(j) · N / j).

The adjusted p-values are monotone non-decreasing in the rank order — we
enforce this with a right-to-left cumulative min, which is the standard
implementation. Verified against `statsmodels.stats.multitest.multipletests`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FDRResult:
    raw_p: np.ndarray
    adjusted_p: np.ndarray
    rejected: np.ndarray
    alpha: float
    n_rejected: int


def benjamini_hochberg(
    p_values: list[float] | np.ndarray,
    alpha: float = 0.05,
) -> FDRResult:
    """BH-FDR. Returns adjusted p-values aligned to the input order."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return FDRResult(
            raw_p=p,
            adjusted_p=np.empty(0, dtype=float),
            rejected=np.empty(0, dtype=bool),
            alpha=alpha,
            n_rejected=0,
        )
    if ((p < 0) | (p > 1)).any():
        raise ValueError("p-values must be in [0, 1]")

    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    ranks = np.arange(1, n + 1, dtype=float)

    # Adjusted p-values (q-values) per BH.
    adjusted_sorted = sorted_p * n / ranks
    # Enforce monotonicity from the right and cap at 1.
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    adjusted = np.empty(n, dtype=float)
    adjusted[order] = adjusted_sorted
    rejected = adjusted <= alpha

    return FDRResult(
        raw_p=p,
        adjusted_p=adjusted,
        rejected=rejected,
        alpha=alpha,
        n_rejected=int(rejected.sum()),
    )
