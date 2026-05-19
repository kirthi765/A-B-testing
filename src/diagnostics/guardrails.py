"""Guardrail evaluator — keeps the primary-metric "win" from shipping if a
guardrail metric (latency, error rate, retention) regresses.

Every experiment registers a `GuardrailConfig` per guardrail metric:
  - `direction`: lower_is_better (latency, error rate) or higher_is_better
    (retention, satisfaction).
  - `threshold_relative`: max acceptable *worsening* of the metric as a
    fraction of control mean. E.g. `0.05` for latency = "treatment latency
    can't exceed control's by more than 5%". This is the relative-MDE
    formulation that ops teams actually negotiate.

Verdict: a guardrail *fails* when the treatment moves the metric in the
wrong direction *and* the change is statistically significant. A directional
but-non-significant change is a "warn" — surface it, but don't block.

The test is a Welch t-test on the underlying metric column. Variants of
this for percentile metrics (latency_p95) require bootstrap; we'd swap the
test method at the call site, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from src.stats import welch_t_test

from ._result import DiagnosticResult


@dataclass(frozen=True)
class GuardrailConfig:
    """Spec for a single guardrail.

    `direction` says which way is *good*. A treatment that *worsens* the metric
    by more than `threshold_relative` (as a fraction of the control mean) is
    a violation.
    """

    metric_name: str
    direction: Literal["lower_is_better", "higher_is_better"]
    threshold_relative: float


def evaluate_guardrail(
    df: pd.DataFrame,
    *,
    variant_col: str,
    metric_col: str,
    config: GuardrailConfig,
    control: str = "control",
    treatment: str = "treatment",
    alpha: float = 0.05,
) -> DiagnosticResult:
    """Welch t-test on `metric_col`, then verdict against the guardrail config."""
    test = welch_t_test(
        df,
        variant_col=variant_col,
        metric_col=metric_col,
        control=control,
        treatment=treatment,
    )

    mean_c = float(test.metadata["mean_control"])
    mean_t = float(test.metadata["mean_treatment"])
    abs_change = test.point_estimate  # treatment - control
    rel_change = abs_change / mean_c if mean_c != 0 else float("nan")

    if config.direction == "lower_is_better":
        # Worse when treatment > control. Violation when relative increase exceeds threshold.
        directional_violation = rel_change > config.threshold_relative
    else:  # higher_is_better
        # Worse when treatment < control. Violation when relative decrease exceeds threshold.
        directional_violation = rel_change < -config.threshold_relative

    significant = test.p_value < alpha

    if directional_violation and significant:
        status: str = "fail"
        message = (
            f"Guardrail violation [{config.metric_name}]: relative change "
            f"{rel_change:+.2%} (threshold ±{config.threshold_relative:.2%}, "
            f"direction={config.direction}), p={test.p_value:.2e}."
        )
    elif directional_violation:
        status = "warn"
        message = (
            f"Guardrail directional shift [{config.metric_name}]: "
            f"{rel_change:+.2%} but not significant (p={test.p_value:.3f})."
        )
    else:
        status = "pass"
        message = (
            f"Guardrail OK [{config.metric_name}]: change {rel_change:+.2%} "
            f"within threshold ±{config.threshold_relative:.2%}."
        )

    return DiagnosticResult(
        name=f"guardrail:{config.metric_name}",
        status=status,  # type: ignore[arg-type]
        message=message,
        evidence={
            "absolute_change": abs_change,
            "relative_change": rel_change,
            "mean_control": mean_c,
            "mean_treatment": mean_t,
            "p_value": test.p_value,
            "ci_low": test.ci_low,
            "ci_high": test.ci_high,
            "threshold_relative": config.threshold_relative,
            "direction": config.direction,
            "alpha": alpha,
            "directional_violation": directional_violation,
            "significant": significant,
        },
    )
