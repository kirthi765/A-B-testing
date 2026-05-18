"""User generation with realistic SaaS segments.

Each segment drives downstream event generation (session frequency, conversion
propensity, latency distribution). Segment mix weights make the population
intentionally non-uniform so that segment-heterogeneity scenarios (Simpson's,
HTE) have something to find.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SegmentConfig:
    name: str
    base_conversion_rate: float
    sessions_per_day: float
    latency_log_mean: float
    latency_log_sigma: float
    weight: float


SEGMENTS: dict[str, SegmentConfig] = {
    "power_user": SegmentConfig(
        name="power_user",
        base_conversion_rate=0.08,
        sessions_per_day=5.0,
        latency_log_mean=4.9,
        latency_log_sigma=0.35,
        weight=0.15,
    ),
    "casual": SegmentConfig(
        name="casual",
        base_conversion_rate=0.03,
        sessions_per_day=1.0,
        latency_log_mean=5.2,
        latency_log_sigma=0.50,
        weight=0.55,
    ),
    "new_signup": SegmentConfig(
        name="new_signup",
        base_conversion_rate=0.02,
        sessions_per_day=1.5,
        latency_log_mean=5.4,
        latency_log_sigma=0.55,
        weight=0.20,
    ),
    "enterprise": SegmentConfig(
        name="enterprise",
        base_conversion_rate=0.06,
        sessions_per_day=3.0,
        latency_log_mean=5.0,
        latency_log_sigma=0.30,
        weight=0.10,
    ),
}


def generate_users(
    n: int,
    experiment_start: pd.Timestamp,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate `n` users with segment mix from `SEGMENTS` and randomized signup dates.

    Signup is uniformly drawn 0–365 days before `experiment_start`, giving every
    user a non-trivial pre-period for CUPED / A-A diagnostics later on.
    """
    rng = np.random.default_rng(seed)
    seg_names = list(SEGMENTS.keys())
    weights = np.array([SEGMENTS[s].weight for s in seg_names], dtype=float)
    weights /= weights.sum()
    segs = rng.choice(seg_names, size=n, p=weights)
    days_before = rng.integers(0, 365, size=n)
    signup_dates = pd.to_datetime(experiment_start) - pd.to_timedelta(days_before, unit="D")
    return pd.DataFrame(
        {
            "user_id": [f"u_{i:07d}" for i in range(n)],
            "segment": segs,
            "signup_date": signup_dates.normalize(),
        }
    )
