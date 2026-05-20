"""Case-study harness — runs every named scenario in-process and prints the
diagnostic verdict + headline numbers. The README cites these numbers directly,
so this script is the reproducibility artifact: rerun it and the README should
still tell the truth.

Usage:
    uv run python scripts/run_case_study.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow the script to be run directly from the repo root without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.diagnostics import (  # noqa: E402
    GuardrailConfig,
    check_novelty,
    check_simpsons,
    check_srm,
    evaluate_guardrail,
)
from src.hte.causal_trees import build_user_level_dataset  # noqa: E402
from src.simulator.scenarios import SCENARIOS  # noqa: E402
from src.stats import msprt, two_proportion_z_test, welch_t_test  # noqa: E402


SCENARIO_ORDER = [
    "clean_lift",
    "srm_bug",
    "novelty_effect",
    "simpsons",
    "guardrail_violation",
    "aa_drift",
    "heterogeneous",
]


def _daily_rollup(scenario) -> pd.DataFrame:
    e = scenario.events.merge(
        scenario.exposures[["user_id", "variant"]], on="user_id"
    )
    e["event_date"] = pd.to_datetime(e["ts"]).dt.normalize()
    rollup = (
        e.groupby(["variant", "event_date"])
        .agg(
            n_sessions=("event_type", lambda x: (x == "session").sum()),
            n_conversions=("event_type", lambda x: (x == "conversion").sum()),
        )
        .reset_index()
    )
    rollup["conversion_rate"] = rollup["n_conversions"] / rollup["n_sessions"].clip(
        lower=1
    )
    return rollup


def _run_scenario(name: str, **kwargs) -> dict:
    print(f"\n{'=' * 78}")
    print(f"SCENARIO: {name}")
    print("=" * 78)

    s = SCENARIOS[name](**kwargs)
    df = build_user_level_dataset(s.users, s.events, s.exposures)
    daily = _daily_rollup(s)

    srm = check_srm(s.exposures)
    print(f"\n  SRM check:        {srm.status.upper():4}  {srm.message}")

    primary_bin = two_proportion_z_test(
        df, variant_col="variant", metric_col="converted"
    )
    primary_rate = welch_t_test(
        df.dropna(subset=["conversion_rate"]),
        variant_col="variant",
        metric_col="conversion_rate",
    )
    seq = msprt(df, variant_col="variant", metric_col="converted")
    print(
        f"\n  Primary z-test:   delta={primary_bin.point_estimate:+.4f}  "
        f"CI=[{primary_bin.ci_low:+.4f}, {primary_bin.ci_high:+.4f}]  "
        f"p={primary_bin.p_value:.4f}"
    )
    print(
        f"  Primary t-test:   delta={primary_rate.point_estimate:+.4f}  "
        f"CI=[{primary_rate.ci_low:+.4f}, {primary_rate.ci_high:+.4f}]  "
        f"p={primary_rate.p_value:.4f}"
    )
    print(f"  mSPRT (peek-safe): p_sequential={seq.p_value:.4f}")

    config = GuardrailConfig(
        metric_name="latency",
        direction="lower_is_better",
        threshold_relative=0.05,
    )
    lat = evaluate_guardrail(
        df.dropna(subset=["mean_latency_ms"]),
        variant_col="variant",
        metric_col="mean_latency_ms",
        config=config,
    )
    print(
        f"\n  Latency guardrail: {lat.status.upper():4}  "
        f"delta={lat.evidence['relative_change']:+.2%}  p={lat.evidence['p_value']:.4f}"
    )

    nov = check_novelty(daily, metric_col="conversion_rate")
    print(
        f"  Novelty check:     {nov.status.upper():4}  "
        f"slope={nov.evidence.get('slope_per_day', float('nan')):+.6f}/day  "
        f"p={nov.evidence.get('p_value', float('nan')):.4f}"
    )

    sim = check_simpsons(df, metric_col="converted")
    print(
        f"  Simpson's check:   {sim.status.upper():4}  "
        f"overall_sign={sim.evidence['overall_sign']}  "
        f"majority_sign={sim.evidence['weighted_majority_sign']}"
    )

    gt = s.ground_truth
    print("\n  Ground truth:")
    for k, v in gt.items():
        if isinstance(v, dict):
            v = {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}
        print(f"    {k}: {v}")

    return {
        "scenario": name,
        "srm": srm.status,
        "primary_pe": primary_bin.point_estimate,
        "primary_p": primary_bin.p_value,
        "primary_rate_pe": primary_rate.point_estimate,
        "primary_rate_p": primary_rate.p_value,
        "latency": lat.status,
        "latency_rel": lat.evidence["relative_change"],
        "novelty": nov.status,
        "novelty_slope": nov.evidence.get("slope_per_day"),
        "simpsons": sim.status,
        "overall_sign": sim.evidence["overall_sign"],
        "majority_sign": sim.evidence["weighted_majority_sign"],
    }


def main() -> None:
    results = []
    for name in SCENARIO_ORDER:
        # novelty needs more power, otherwise daily noise drowns the slope
        kwargs = {"n_users": 15_000 if name == "novelty_effect" else 10_000}
        results.append(_run_scenario(name, **kwargs))

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    df = pd.DataFrame(results)
    df["latency_rel"] = df["latency_rel"].map("{:+.2%}".format)
    df["primary_pe"] = df["primary_pe"].map("{:+.4f}".format)
    df["primary_p"] = df["primary_p"].map("{:.4f}".format)
    df["primary_rate_pe"] = df["primary_rate_pe"].map("{:+.4f}".format)
    df["primary_rate_p"] = df["primary_rate_p"].map("{:.4f}".format)
    df["novelty_slope"] = df["novelty_slope"].map("{:+.6f}".format)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
