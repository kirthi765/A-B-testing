"""Glue layer: ExperimentData → diagnostics + stats → AnalysisReport.

The Streamlit page should be dumb — render whatever this returns. Anything
that involves a decision (which test to run, whether to ship) lives here so
it's covered by tests rather than buried in UI code.

Primary metric is the binary `converted` (per-user did they convert at least
once?). Guardrail metric is `mean_latency_ms` per user. Both are columns in
`fct_experiment_metrics` materialized by dbt in Phase 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from src.diagnostics import (
    DiagnosticResult,
    GuardrailConfig,
    check_novelty,
    check_simpsons,
    check_srm,
    evaluate_guardrail,
)
from src.stats import TestResult, msprt, two_proportion_z_test

from .data import ExperimentData


Recommendation = Literal["Ship", "Don't ship", "Iterate"]


@dataclass(frozen=True)
class SegmentRow:
    segment: str
    n_control: int
    n_treatment: int
    conv_rate_control: float
    conv_rate_treatment: float
    lift_absolute: float
    p_value: float


@dataclass(frozen=True)
class AnalysisReport:
    experiment_id: str
    srm: DiagnosticResult
    primary: TestResult           # two-proportion z-test on `converted`
    primary_sequential: TestResult  # mSPRT on `converted`
    guardrails: list[DiagnosticResult]
    novelty: DiagnosticResult
    simpsons: DiagnosticResult
    segments: list[SegmentRow]
    recommendation: Recommendation
    recommendation_reasons: list[str] = field(default_factory=list)


def _per_segment_breakdown(metrics) -> list[SegmentRow]:
    rows: list[SegmentRow] = []
    for seg in sorted(metrics["segment"].dropna().unique().tolist()):
        sub = metrics[metrics["segment"] == seg]
        c = sub[sub["variant"] == "control"]
        t = sub[sub["variant"] == "treatment"]
        if len(c) < 20 or len(t) < 20:
            continue
        try:
            r = two_proportion_z_test(
                sub, variant_col="variant", metric_col="converted"
            )
        except ValueError:
            continue
        rows.append(
            SegmentRow(
                segment=str(seg),
                n_control=int(len(c)),
                n_treatment=int(len(t)),
                conv_rate_control=float(c["converted"].mean()),
                conv_rate_treatment=float(t["converted"].mean()),
                lift_absolute=float(r.point_estimate),
                p_value=float(r.p_value),
            )
        )
    return rows


def _recommend(
    *,
    srm: DiagnosticResult,
    guardrails: list[DiagnosticResult],
    primary: TestResult,
    novelty: DiagnosticResult,
    simpsons: DiagnosticResult,
) -> tuple[Recommendation, list[str]]:
    reasons: list[str] = []

    # Ship-blockers, in order of severity.
    if srm.status == "fail":
        reasons.append(f"SRM detected — assignment is biased. {srm.message}")
        return "Don't ship", reasons

    failed_guardrails = [g for g in guardrails if g.status == "fail"]
    if failed_guardrails:
        for g in failed_guardrails:
            reasons.append(f"Guardrail violation: {g.message}")
        return "Don't ship", reasons

    # Primary metric decision.
    if primary.ci_low > 0:
        reasons.append(
            f"Primary metric is positive and significant: "
            f"{primary.point_estimate:+.4f} "
            f"(95% CI [{primary.ci_low:+.4f}, {primary.ci_high:+.4f}], "
            f"p={primary.p_value:.4f})."
        )
        if novelty.status == "fail":
            reasons.append(
                f"Caveat — novelty effect: {novelty.message}"
            )
        if simpsons.status == "fail":
            reasons.append(
                f"Caveat — Simpson's reversal: {simpsons.message}"
            )
        return "Ship", reasons

    if primary.ci_high < 0:
        reasons.append(
            f"Primary metric is negative and significant: "
            f"{primary.point_estimate:+.4f} "
            f"(95% CI [{primary.ci_low:+.4f}, {primary.ci_high:+.4f}], "
            f"p={primary.p_value:.4f})."
        )
        return "Don't ship", reasons

    reasons.append(
        f"Primary metric is inconclusive — CI includes zero: "
        f"{primary.point_estimate:+.4f} "
        f"(95% CI [{primary.ci_low:+.4f}, {primary.ci_high:+.4f}], "
        f"p={primary.p_value:.4f})."
    )
    return "Iterate", reasons


def analyze_experiment(data: ExperimentData) -> AnalysisReport:
    """Run every check that doesn't require user input. HTE is deferred to the UI
    (it's expensive and the user opts in)."""
    m = data.metrics

    primary = two_proportion_z_test(
        m, variant_col="variant", metric_col="converted"
    )
    primary_seq = msprt(
        m, variant_col="variant", metric_col="converted"
    )

    srm = check_srm(data.exposures)

    latency_config = GuardrailConfig(
        metric_name="mean_latency_ms",
        direction="lower_is_better",
        threshold_relative=0.05,
    )
    latency = evaluate_guardrail(
        m.dropna(subset=["mean_latency_ms"]),
        variant_col="variant",
        metric_col="mean_latency_ms",
        config=latency_config,
    )
    guardrails = [latency]

    novelty = check_novelty(data.daily, metric_col="conversion_rate")
    simpsons = check_simpsons(m, metric_col="converted")
    segments = _per_segment_breakdown(m)

    rec, reasons = _recommend(
        srm=srm,
        guardrails=guardrails,
        primary=primary,
        novelty=novelty,
        simpsons=simpsons,
    )

    return AnalysisReport(
        experiment_id=data.experiment_id,
        srm=srm,
        primary=primary,
        primary_sequential=primary_seq,
        guardrails=guardrails,
        novelty=novelty,
        simpsons=simpsons,
        segments=segments,
        recommendation=rec,
        recommendation_reasons=reasons,
    )
