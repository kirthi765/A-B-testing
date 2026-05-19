"""Sample size and minimum-detectable-effect (MDE) calculators.

Two flavors, both for two-sided tests at level `alpha` with target `power`:
  - Proportions (binary outcome — conversion-rate experiments).
  - Continuous (Welch t-test — latency, time-on-page, revenue per user).

The formulas use the normal approximation. For continuous metrics with
small samples (~n < 30/arm) you'd want the non-central t exact formula —
but A/B tests live in the n >> 30 regime where this is a non-issue.

Allocation is the *control* arm's share of total traffic, default 0.5 (50/50).
Off-balance splits (e.g. 90/10 for a risky launch) increase total N for
the same MDE — this is what the formula recovers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats


@dataclass(frozen=True)
class SampleSize:
    n_control: int
    n_treatment: int
    n_total: int
    alpha: float
    power: float


def _z_alpha_beta(alpha: float, power: float) -> tuple[float, float]:
    z_a = float(stats.norm.ppf(1 - alpha / 2))
    z_b = float(stats.norm.ppf(power))
    return z_a, z_b


def sample_size_proportions(
    *,
    baseline_rate: float,
    mde_relative: float,
    alpha: float = 0.05,
    power: float = 0.8,
    allocation: float = 0.5,
) -> SampleSize:
    """Required sample size for a two-proportion z-test.

    `mde_relative` is the relative lift, e.g. `0.05` for "+5% conversion".
    Set `mde_relative < 0` to size for a relative drop.
    """
    if not 0 < baseline_rate < 1:
        raise ValueError("baseline_rate must be in (0, 1)")
    if not 0 < allocation < 1:
        raise ValueError("allocation must be in (0, 1)")

    p_c = baseline_rate
    p_t = p_c * (1.0 + mde_relative)
    if not 0 < p_t < 1:
        raise ValueError(f"implied treatment rate {p_t} not in (0, 1)")
    delta = p_t - p_c
    z_a, z_b = _z_alpha_beta(alpha, power)

    target_var = delta * delta / (z_a + z_b) ** 2
    n_total = (p_c * (1 - p_c) / allocation + p_t * (1 - p_t) / (1 - allocation)) / target_var
    n_c = math.ceil(n_total * allocation)
    n_t = math.ceil(n_total * (1 - allocation))
    return SampleSize(
        n_control=n_c, n_treatment=n_t, n_total=n_c + n_t, alpha=alpha, power=power
    )


def sample_size_continuous(
    *,
    std: float,
    mde_absolute: float,
    alpha: float = 0.05,
    power: float = 0.8,
    allocation: float = 0.5,
) -> SampleSize:
    """Required sample size for a Welch t-test, assuming equal variance `std²` in both arms.

    `mde_absolute` is the absolute mean shift to detect (same units as `std`).
    """
    if std <= 0:
        raise ValueError("std must be > 0")
    if not 0 < allocation < 1:
        raise ValueError("allocation must be in (0, 1)")
    z_a, z_b = _z_alpha_beta(alpha, power)
    target_var = mde_absolute * mde_absolute / (z_a + z_b) ** 2
    n_total = (std * std) * (1 / allocation + 1 / (1 - allocation)) / target_var
    n_c = math.ceil(n_total * allocation)
    n_t = math.ceil(n_total * (1 - allocation))
    return SampleSize(
        n_control=n_c, n_treatment=n_t, n_total=n_c + n_t, alpha=alpha, power=power
    )


def mde_proportions(
    *,
    baseline_rate: float,
    n_per_arm: int,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Absolute MDE detectable with `n_per_arm` users per arm at 50/50."""
    z_a, z_b = _z_alpha_beta(alpha, power)
    se = math.sqrt(2.0 * baseline_rate * (1.0 - baseline_rate) / n_per_arm)
    return (z_a + z_b) * se


def mde_continuous(
    *,
    std: float,
    n_per_arm: int,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Absolute MDE for a Welch t-test at 50/50 with per-arm `n_per_arm`."""
    z_a, z_b = _z_alpha_beta(alpha, power)
    return (z_a + z_b) * std * math.sqrt(2.0 / n_per_arm)
