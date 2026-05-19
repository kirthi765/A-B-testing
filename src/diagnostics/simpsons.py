"""Simpson's paradox detector — overall effect direction disagrees with segments.

The textbook case: variant loses overall but wins in every segment (or
vice versa). Within an A/B test this most commonly arises when:
  - Assignment is non-uniform across segments (small SRM-by-segment).
  - Treatment shifts the segment mix (e.g. treatment causes some users
    to drop out, leaving a skewed remainder).
  - The "segment" is itself a post-treatment variable (don't do this).

Detection: compute per-segment lift, weight each segment's sign by N, and
compare the weighted majority sign against the overall sign. If they disagree
*and* the overall absolute lift is non-trivial, flag.

The "majority" is weighted by segment size because a 5-user segment with
a flipped sign shouldn't outvote four 500-user segments — that's noise, not
a paradox.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._result import DiagnosticResult


def _signed_lift(series_c: pd.Series, series_t: pd.Series) -> tuple[float, int]:
    if len(series_c) == 0 or len(series_t) == 0:
        return float("nan"), 0
    diff = float(series_t.mean() - series_c.mean())
    if diff > 0:
        sign = 1
    elif diff < 0:
        sign = -1
    else:
        sign = 0
    return diff, sign


def check_simpsons(
    df: pd.DataFrame,
    *,
    variant_col: str = "variant",
    segment_col: str = "segment",
    metric_col: str = "converted",
    control: str = "control",
    treatment: str = "treatment",
    min_segment_n: int = 30,
) -> DiagnosticResult:
    """Flag if overall lift sign disagrees with the size-weighted segment majority.

    `min_segment_n` ignores tiny segments (per-arm count below this) from the
    majority vote — their lift signs are dominated by noise.
    """
    c_mask = df[variant_col] == control
    t_mask = df[variant_col] == treatment
    overall_diff, overall_sign = _signed_lift(df.loc[c_mask, metric_col], df.loc[t_mask, metric_col])

    segments = df[segment_col].unique().tolist()
    segment_evidence: dict[str, dict] = {}
    pos_weight = 0
    neg_weight = 0
    for seg in segments:
        seg_mask = df[segment_col] == seg
        c = df.loc[seg_mask & c_mask, metric_col]
        t = df.loc[seg_mask & t_mask, metric_col]
        if len(c) < min_segment_n or len(t) < min_segment_n:
            segment_evidence[str(seg)] = {
                "lift": float("nan"),
                "sign": 0,
                "n_control": int(len(c)),
                "n_treatment": int(len(t)),
                "excluded": True,
            }
            continue
        diff, sign = _signed_lift(c, t)
        weight = int(len(c) + len(t))
        if sign > 0:
            pos_weight += weight
        elif sign < 0:
            neg_weight += weight
        segment_evidence[str(seg)] = {
            "lift": diff,
            "sign": sign,
            "n_control": int(len(c)),
            "n_treatment": int(len(t)),
            "excluded": False,
        }

    if pos_weight > neg_weight:
        majority_sign = 1
    elif neg_weight > pos_weight:
        majority_sign = -1
    else:
        majority_sign = 0

    flipped = (
        overall_sign != 0
        and majority_sign != 0
        and overall_sign != majority_sign
    )

    if flipped:
        status: str = "fail"
        direction_overall = "positive" if overall_sign > 0 else "negative"
        direction_segments = "positive" if majority_sign > 0 else "negative"
        message = (
            f"Simpson's reversal: overall lift is {direction_overall} ({overall_diff:+.4g}) "
            f"but the size-weighted segment majority is {direction_segments}. "
            f"Inspect segment mix and assignment per segment."
        )
    else:
        status = "pass"
        message = (
            f"No Simpson's reversal: overall sign={overall_sign}, weighted majority={majority_sign}."
        )

    return DiagnosticResult(
        name="simpsons",
        status=status,  # type: ignore[arg-type]
        message=message,
        evidence={
            "overall_lift": overall_diff,
            "overall_sign": int(overall_sign),
            "weighted_majority_sign": int(majority_sign),
            "positive_segment_weight": int(pos_weight),
            "negative_segment_weight": int(neg_weight),
            "segments": segment_evidence,
            "min_segment_n": min_segment_n,
        },
    )
