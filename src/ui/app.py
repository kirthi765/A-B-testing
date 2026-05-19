"""Streamlit experiment-review page — one experiment per render.

Launch with:
    uv run streamlit run src/ui/app.py

The page is intentionally screenshot-friendly: a fixed header, three-column
health-check pills, a single primary-metric block with sequential and
fixed-α p-values side-by-side, then guardrails / segments / HTE.

Styling is minimal HTML+CSS via `st.markdown(unsafe_allow_html=True)` for the
colored status pills. Everything else uses native Streamlit components so the
defaults take care of dark/light theme switching.
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.ui.analysis import AnalysisReport, analyze_experiment
from src.ui.data import DEFAULT_DB_PATH, ExperimentData, list_experiments, load_experiment


STATUS_COLOR = {"pass": "#16a34a", "warn": "#f59e0b", "fail": "#dc2626"}
STATUS_ICON = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}
RECOMMENDATION_COLOR = {
    "Ship": "#16a34a",
    "Don't ship": "#dc2626",
    "Iterate": "#f59e0b",
}


def _pill(name: str, status: str, message: str) -> None:
    color = STATUS_COLOR.get(status, "#64748b")
    icon = STATUS_ICON.get(status, "----")
    st.markdown(
        f"""
        <div style="
            background: rgba(148,163,184,0.08);
            padding: 14px 16px;
            border-left: 6px solid {color};
            border-radius: 4px;
            margin-bottom: 8px;
        ">
            <div style="font-size:0.75rem;color:{color};font-weight:600;letter-spacing:0.06em;">
                {icon}
            </div>
            <div style="font-size:1rem;font-weight:600;margin-top:2px;">{name}</div>
            <div style="font-size:0.85rem;color:#94a3b8;margin-top:4px;">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_header(data: ExperimentData) -> None:
    st.title(f"Experiment review — `{data.experiment_id}`")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Dates",
        f"{data.start_date.date()} → {data.end_date.date()}",
    )
    col2.metric("Users", f"{len(data.metrics):,}")
    col3.metric("Events", f"{data.n_total_events:,}")
    duration_days = (data.end_date - data.start_date).days
    col4.metric("Duration", f"{duration_days} days")


def _render_health_checks(report: AnalysisReport, data: ExperimentData) -> None:
    st.header("Health checks")
    c1, c2, c3 = st.columns(3)
    with c1:
        _pill(report.srm.name, report.srm.status, report.srm.message)
    with c2:
        # A/A pre-period would live here — simulator doesn't currently emit pre-period
        # events, so we surface the gap rather than fake a passing check.
        _pill(
            "aa_pre_period",
            "warn",
            "Pre-period data not available in this scenario; A/A check skipped.",
        )
    with c3:
        n_users = len(data.metrics)
        per_arm = n_users // 2
        # 1,000/arm is a defensible lower bound for portfolio-scale demos.
        ss_status = "pass" if per_arm >= 1_000 else "warn"
        _pill(
            "sample_size",
            ss_status,
            f"{per_arm:,} users per arm (total {n_users:,}).",
        )


def _render_primary_metric(report: AnalysisReport, data: ExperimentData) -> None:
    st.header("Primary metric — converted (binary, per user)")
    p = report.primary
    s = report.primary_sequential
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Effect (Δ rate)", f"{p.point_estimate:+.4f}")
    c2.metric("95% CI", f"[{p.ci_low:+.4f}, {p.ci_high:+.4f}]")
    c3.metric("p-value (fixed α)", f"{p.p_value:.4f}")
    c4.metric("p-value (mSPRT)", f"{s.p_value:.4f}")

    st.caption(
        "mSPRT p-value is peek-safe: it stays valid no matter how many times "
        "you check the experiment during its run."
    )

    st.subheader("Daily conversion rate by variant")
    line = (
        alt.Chart(data.daily)
        .mark_line(point=True)
        .encode(
            x=alt.X("event_date:T", title="Date"),
            y=alt.Y(
                "conversion_rate:Q",
                title="Conversion rate (sessions)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color("variant:N", title="Variant"),
            tooltip=[
                "event_date:T",
                "variant:N",
                alt.Tooltip("conversion_rate:Q", format=".4f"),
                "n_sessions:Q",
            ],
        )
        .properties(height=300)
    )
    st.altair_chart(line, use_container_width=True)


def _render_guardrails(report: AnalysisReport) -> None:
    st.header("Guardrails")
    if not report.guardrails:
        st.info("No guardrails configured for this experiment.")
        return
    rows = []
    for g in report.guardrails:
        ev = g.evidence
        rows.append(
            {
                "name": g.name,
                "status": g.status.upper(),
                "Δ relative": f"{ev.get('relative_change', float('nan')):+.2%}",
                "control": f"{ev.get('mean_control', float('nan')):.3f}",
                "treatment": f"{ev.get('mean_treatment', float('nan')):.3f}",
                "p-value": f"{ev.get('p_value', float('nan')):.4f}",
                "threshold": f"±{ev.get('threshold_relative', float('nan')):.2%}",
                "direction": str(ev.get("direction", "")),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    for g in report.guardrails:
        if g.status != "pass":
            _pill(g.name, g.status, g.message)


def _render_segments(report: AnalysisReport) -> None:
    st.header("Segment breakdown")
    if not report.segments:
        st.info("Not enough per-segment data to break out.")
    else:
        rows = [
            {
                "segment": s.segment,
                "n control": s.n_control,
                "n treatment": s.n_treatment,
                "control rate": f"{s.conv_rate_control:.3%}",
                "treatment rate": f"{s.conv_rate_treatment:.3%}",
                "lift (Δ)": f"{s.lift_absolute:+.4f}",
                "p-value": f"{s.p_value:.4f}",
            }
            for s in report.segments
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if report.simpsons.status == "fail":
        _pill("simpsons", "fail", report.simpsons.message)
    else:
        st.caption(f"Simpson's check: {report.simpsons.message}")


def _render_novelty(report: AnalysisReport) -> None:
    st.header("Novelty / time-effect")
    _pill(report.novelty.name, report.novelty.status, report.novelty.message)


def _render_hte_section(data: ExperimentData) -> None:
    st.header("Heterogeneous treatment effects (CATE)")
    st.caption(
        "Fits a causal forest (econml CausalForestDML) on the per-user metric "
        "with segment one-hots as features. Click to compute — takes 30–60s."
    )
    if not st.button("Compute CATE", type="primary"):
        return

    from src.hte import (
        cate_by_decile,
        cate_by_segment,
        fit_causal_forest,
        one_hot_segments,
        predict_cate,
    )

    with st.spinner("Fitting causal forest…"):
        users = data.metrics.copy()
        users["treatment"] = (users["variant"] == "treatment").astype(int)
        users, feat_cols = one_hot_segments(users)
        model = fit_causal_forest(
            users,
            treatment_col="treatment",
            outcome_col="converted",
            feature_cols=feat_cols,
            n_estimators=32,
            random_state=42,
        )
        cate = predict_cate(model, users, feat_cols)

    by_decile = cate_by_decile(cate, n_buckets=10)
    by_segment = cate_by_segment(cate, users["segment"])

    st.subheader("CATE by decile")
    bar = (
        alt.Chart(by_decile)
        .mark_bar()
        .encode(
            x=alt.X("bucket:O", title="CATE decile (low → high)"),
            y=alt.Y("cate_mean:Q", title="Mean CATE"),
            color=alt.condition(
                alt.datum.cate_mean > 0, alt.value("#16a34a"), alt.value("#dc2626")
            ),
            tooltip=[
                "bucket:O",
                "n:Q",
                alt.Tooltip("cate_mean:Q", format=".4f"),
                alt.Tooltip("cate_p05:Q", format=".4f"),
                alt.Tooltip("cate_p95:Q", format=".4f"),
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(bar, use_container_width=True)

    st.subheader("CATE by known segment")
    seg_df = by_segment.assign(
        cate_mean=lambda d: d["cate_mean"].map("{:+.4f}".format),
        cate_std=lambda d: d["cate_std"].map("{:.4f}".format),
    )
    st.dataframe(seg_df, use_container_width=True, hide_index=True)


def _render_recommendation(report: AnalysisReport) -> None:
    st.header("Recommendation")
    color = RECOMMENDATION_COLOR.get(report.recommendation, "#64748b")
    st.markdown(
        f"""
        <div style="
            background: rgba(148,163,184,0.10);
            padding: 20px 24px;
            border-left: 8px solid {color};
            border-radius: 4px;
        ">
            <div style="font-size:1.75rem;font-weight:700;color:{color};">
                {report.recommendation}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")  # spacer
    for reason in report.recommendation_reasons:
        st.markdown(f"- {reason}")


def main() -> None:
    st.set_page_config(
        page_title="Experiment Review", layout="wide", initial_sidebar_state="expanded"
    )

    db_path = Path(DEFAULT_DB_PATH)
    if not db_path.exists():
        st.error(
            "No warehouse found at `data/warehouse.duckdb`. Run "
            "`uv run python -m src.simulator.scenarios` and then "
            "`cd dbt_project && uv run dbt build --profiles-dir .` first."
        )
        st.stop()

    experiments = list_experiments(db_path)
    if not experiments:
        st.error("Warehouse contains no experiments.")
        st.stop()

    with st.sidebar:
        st.markdown("### Experiment selector")
        experiment_id = st.selectbox("Experiment", experiments)
        st.markdown("---")
        st.caption(
            "Marts are built once by dbt and read-only here. If you swap the "
            "scenario in the warehouse, re-run `dbt build` before reloading."
        )

    data = load_experiment(experiment_id, db_path)
    report = analyze_experiment(data)

    _render_header(data)
    _render_health_checks(report, data)
    _render_primary_metric(report, data)
    _render_guardrails(report)
    _render_segments(report)
    _render_novelty(report)
    _render_hte_section(data)
    _render_recommendation(report)


if __name__ == "__main__":
    main()
