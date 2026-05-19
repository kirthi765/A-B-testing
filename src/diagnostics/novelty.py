"""Novelty-effect detector — fit a line to the daily treatment effect over time.

A novelty effect is when the treatment looks great in week 1, then decays
toward zero (or even reverses) by week 4 — users initially try the new
feature out of curiosity, then revert. The pure effect estimate computed
on the full experiment window averages over both regimes and ships a
feature that won't actually retain users.

Detection: fit `daily_lift ~ days_since_start`. A significant *negative*
slope is the signature. We use a simple OLS fit; the `daily_lift` series
is short (typically 14–28 days) so heavy machinery like state-space models
would be overkill.

A *positive* slope is also interesting (anti-novelty / activation lag)
but is rarely a ship-blocker — we surface it as info, not a failure.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from ._result import DiagnosticResult


def check_novelty(
    daily_df: pd.DataFrame,
    *,
    variant_col: str = "variant",
    date_col: str = "event_date",
    metric_col: str = "conversion_rate",
    control: str = "control",
    treatment: str = "treatment",
    fail_threshold: float = 0.01,
    warn_threshold: float = 0.05,
    min_days: int = 5,
) -> DiagnosticResult:
    """Fit `daily_lift ~ day_index`. Negative slope at `p < threshold` → flag.

    Expects a daily-rollup dataframe (`fct_experiment_daily`-shaped) with one
    row per `(date, variant)`. Pivots wide, computes the per-day treatment
    minus control lift, and regresses on day index.
    """
    pivot = (
        daily_df.pivot_table(index=date_col, columns=variant_col, values=metric_col)
        .sort_index()
    )
    if control not in pivot.columns or treatment not in pivot.columns:
        raise ValueError(
            f"daily_df must contain both {control!r} and {treatment!r} variants"
        )
    pivot = pivot[[control, treatment]].dropna()
    pivot["lift"] = pivot[treatment] - pivot[control]
    n = len(pivot)

    if n < min_days:
        return DiagnosticResult(
            name="novelty",
            status="warn",
            message=f"Only {n} days of data — need at least {min_days} to assess novelty decay.",
            evidence={"n_days": n, "min_days": min_days},
        )

    days = np.arange(n, dtype=float)
    y = pivot["lift"].to_numpy()
    slope, intercept, r_value, p_value, std_err = stats.linregress(days, y)
    slope = float(slope)
    p_value = float(p_value)

    if slope < 0 and p_value < fail_threshold:
        status: str = "fail"
        message = (
            f"Novelty decay detected: lift drops {abs(slope):.4g}/day, "
            f"p={p_value:.2e}. Holdout the early days or extend the experiment."
        )
    elif slope < 0 and p_value < warn_threshold:
        status = "warn"
        message = (
            f"Possible novelty decay: slope={slope:.4g}/day, p={p_value:.3f}. "
            f"Monitor; extend window if effect continues to shrink."
        )
    else:
        status = "pass"
        message = (
            f"No novelty decay detected (slope={slope:+.4g}/day, p={p_value:.3f})."
        )

    # Week-over-week summary for the evidence dict — useful for the UI plot.
    pivot["week"] = pivot.index.to_series().reset_index(drop=True).index // 7
    weekly_lift = pivot.groupby("week")["lift"].mean().to_dict()

    return DiagnosticResult(
        name="novelty",
        status=status,  # type: ignore[arg-type]
        message=message,
        evidence={
            "slope_per_day": slope,
            "intercept": float(intercept),
            "p_value": p_value,
            "r_squared": float(r_value * r_value) if not math.isnan(r_value) else 0.0,
            "std_err": float(std_err),
            "n_days": n,
            "week_means": {int(k): float(v) for k, v in weekly_lift.items()},
            "fail_threshold": fail_threshold,
            "warn_threshold": warn_threshold,
        },
    )
