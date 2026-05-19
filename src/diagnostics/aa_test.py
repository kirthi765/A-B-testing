"""Pre-period A/A check — run the primary test on data from before exposure.

If the assignment is truly random and the pre-period covariate is unaffected
by treatment (because treatment hasn't been delivered yet), the pre-period
A/A should fail to reject H0 at the configured level. A rejection here means
something about the *cohort definition* or *assignment* is biased — the
two arms are not exchangeable, and any post-period effect is suspect.

Common causes when A/A fails:
  - Variant assigned by an attribute that is itself correlated with outcome
    (e.g. account-creation timestamp leaking into the hash).
  - Cohort filter applied post-randomization that drops users non-uniformly.
  - The bucketer is run before signup and the signup funnel itself differs
    between arms (selection bias).
"""

from __future__ import annotations

import pandas as pd

from src.stats import welch_t_test

from ._result import DiagnosticResult


def check_aa(
    df: pd.DataFrame,
    *,
    variant_col: str,
    metric_col: str,
    control: str = "control",
    treatment: str = "treatment",
    fail_threshold: float = 0.01,
    warn_threshold: float = 0.05,
) -> DiagnosticResult:
    """Welch's t-test on a pre-period metric. Pre-period data is the user's
    responsibility — typically the same metric measured during a pre-experiment
    window (which is exactly the CUPED covariate, conveniently).
    """
    result = welch_t_test(
        df,
        variant_col=variant_col,
        metric_col=metric_col,
        control=control,
        treatment=treatment,
    )
    p = result.p_value
    diff = result.point_estimate

    if p < fail_threshold:
        status: str = "fail"
        message = (
            f"A/A test rejects on pre-period {metric_col}: diff={diff:.4g}, p={p:.2e}. "
            f"Assignment may not be exchangeable — block on this before reporting effects."
        )
    elif p < warn_threshold:
        status = "warn"
        message = (
            f"A/A test borderline: diff={diff:.4g}, p={p:.3f}. "
            f"Review whether {metric_col} is balanced across arms."
        )
    else:
        status = "pass"
        message = (
            f"A/A test passes on pre-period {metric_col}: diff={diff:.4g}, p={p:.3f}."
        )

    return DiagnosticResult(
        name="aa_pre_period",
        status=status,  # type: ignore[arg-type]
        message=message,
        evidence={
            "pre_period_diff": diff,
            "p_value": p,
            "ci_low": result.ci_low,
            "ci_high": result.ci_high,
            "mean_control": result.metadata.get("mean_control"),
            "mean_treatment": result.metadata.get("mean_treatment"),
            "n_control": result.metadata.get("n_control"),
            "n_treatment": result.metadata.get("n_treatment"),
            "fail_threshold": fail_threshold,
            "warn_threshold": warn_threshold,
        },
    )
