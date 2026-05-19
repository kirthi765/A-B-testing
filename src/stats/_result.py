"""The shared return type for every stats method.

Every test in this package — Welch's t, two-proportion z, bootstrap, delta,
CUPED, mSPRT — returns a `TestResult`. That uniformity is what lets the
review UI render any method identically (point estimate, CI, p-value) and
makes the diagnostics framework agnostic to which estimator was used.
Method-specific extras (sample sizes, standard errors, mSPRT prior scale,
CUPED variance reduction, etc.) live in `metadata`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TestResult:
    point_estimate: float
    ci_low: float
    ci_high: float
    p_value: float
    method_name: str
    metadata: dict[str, Any] = field(default_factory=dict)
