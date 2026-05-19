"""Sample Ratio Mismatch — the cheapest, highest-leverage diagnostic.

If 50/50 traffic split is configured but you observe 51.5/48.5, *every*
downstream effect estimate is suspect. The most common causes are:
  - Bucketer bug (Phase 2's responsibility — but the simulator can plant it).
  - Exposure-logging dropout that's correlated with variant (e.g. a bug that
    crashes the treatment client more often than control).
  - Eligibility filtering applied post-randomization.

The test: chi-square goodness-of-fit on observed vs expected variant counts.
Threshold of `p < 0.001` follows Microsoft / Booking convention — the SRM
diagnostic should rarely fire under H0 (false-positive once per ~1000 clean
experiments), because a "warn"-level false positive on every dashboard kills
trust in the framework.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ._result import DiagnosticResult


def check_srm(
    df: pd.DataFrame,
    *,
    variant_col: str = "variant",
    expected_weights: dict[str, float] | None = None,
    fail_threshold: float = 0.001,
    warn_threshold: float = 0.01,
) -> DiagnosticResult:
    """Chi-square goodness-of-fit on observed variant counts vs. expected weights.

    `expected_weights` defaults to equal allocation across the observed variants
    (`{control: 0.5, treatment: 0.5}` for a two-arm test). Pass explicit weights
    if the experiment is run on an off-balance split (e.g. 90/10 for a risky
    launch).
    """
    counts = df[variant_col].value_counts().sort_index()
    n_total = int(counts.sum())

    if expected_weights is None:
        n_variants = len(counts)
        expected_weights = {v: 1.0 / n_variants for v in counts.index}

    # Re-order observed to match expected variant ordering.
    variants = list(expected_weights.keys())
    obs = np.array([int(counts.get(v, 0)) for v in variants], dtype=float)
    exp = np.array([expected_weights[v] * n_total for v in variants], dtype=float)

    if (exp <= 0).any():
        raise ValueError(f"expected counts must be > 0; got {dict(zip(variants, exp))}")
    if not np.isclose(sum(expected_weights.values()), 1.0, atol=1e-9):
        raise ValueError(
            f"expected_weights must sum to 1.0, got {sum(expected_weights.values())}"
        )

    chi2, p_value = stats.chisquare(f_obs=obs, f_exp=exp)
    chi2 = float(chi2)
    p_value = float(p_value)

    observed_share = {v: float(obs[i] / n_total) for i, v in enumerate(variants)}
    expected_share = {v: float(expected_weights[v]) for v in variants}

    if p_value < fail_threshold:
        status: str = "fail"
        message = (
            f"SRM detected: chi-square p={p_value:.2e} < {fail_threshold}. "
            f"Observed split {observed_share} vs expected {expected_share}."
        )
    elif p_value < warn_threshold:
        status = "warn"
        message = (
            f"SRM borderline: chi-square p={p_value:.2e} below warn threshold "
            f"{warn_threshold}. Observed {observed_share}; investigate bucketer."
        )
    else:
        status = "pass"
        message = (
            f"No SRM: chi-square p={p_value:.3f}. "
            f"Observed split {observed_share} consistent with expected {expected_share}."
        )

    return DiagnosticResult(
        name="srm",
        status=status,  # type: ignore[arg-type]
        message=message,
        evidence={
            "chi2": chi2,
            "p_value": p_value,
            "n_total": n_total,
            "observed_counts": {v: int(obs[i]) for i, v in enumerate(variants)},
            "expected_counts": {v: float(exp[i]) for i, v in enumerate(variants)},
            "observed_share": observed_share,
            "expected_share": expected_share,
            "fail_threshold": fail_threshold,
            "warn_threshold": warn_threshold,
        },
    )
