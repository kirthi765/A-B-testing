"""mSPRT — mixture Sequential Probability Ratio Test for peek-safe A/B tests.

(Johari, Pekelis, Walsh 2017 — "Peeking at A/B Tests: Why It Matters, and What
to Do About It". Robbins 1970 first introduced the mixture sequential test.)

Under a fixed-α test, peeking at the data N times inflates the false-positive
rate from α toward 1 as N grows. The mSPRT replaces the fixed test with a
*likelihood ratio against a mixture prior*: under H0 the ratio is a
martingale, so by the optional-stopping theorem the test maintains type-I
error α no matter how many times you peek — including continuously.

For a normal mean test on the difference `D = mean_T - mean_C`:
    H0:   D = 0
    H1:   D ~ N(0, τ²)         (mixture prior over alternatives)

Treating the observed `D` as `~ N(θ, SE²)` with known SE (estimated from the
sample), the Bayes factor is

    BF = sqrt(SE²/(SE² + τ²)) · exp( 0.5 · D² · τ² / (SE² · (SE² + τ²)) )

The always-valid p-value is `min(1, 1/BF)`. The always-valid CI is the set
of θ₀ for which the same BF (computed with `D - θ₀` in place of `D`) is
below `1/α`. Inverting gives a closed-form margin around `D`.

Default τ is set to the pooled sample SD — a "reasonable effect size" prior
that's standard practice when you don't have a strong informative prior.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._result import TestResult


def msprt(
    df: pd.DataFrame,
    *,
    variant_col: str,
    metric_col: str,
    control: str = "control",
    treatment: str = "treatment",
    tau: float | None = None,
    alpha: float = 0.05,
) -> TestResult:
    """Always-valid p-value + CI for `mean(treatment) - mean(control)`.

    `tau` is the mixture-prior standard deviation. If `None`, defaults to the
    pooled sample SD (a common "reasonable effect" choice). Smaller `tau`
    concentrates the prior near zero, giving more power against small effects
    at the cost of less power against large ones; larger `tau` is the reverse.
    """
    x_c = df.loc[df[variant_col] == control, metric_col].dropna().to_numpy()
    x_t = df.loc[df[variant_col] == treatment, metric_col].dropna().to_numpy()
    n_c, n_t = len(x_c), len(x_t)
    if n_c < 2 or n_t < 2:
        raise ValueError(f"need at least 2 obs per arm; got n_c={n_c}, n_t={n_t}")

    var_c = float(x_c.var(ddof=1))
    var_t = float(x_t.var(ddof=1))
    se2 = var_c / n_c + var_t / n_t  # variance of the difference of means
    diff = float(x_t.mean()) - float(x_c.mean())

    if tau is None:
        var_pooled = ((n_c - 1) * var_c + (n_t - 1) * var_t) / (n_c + n_t - 2)
        tau = math.sqrt(var_pooled)
    tau2 = tau * tau

    # log Bayes factor: log[ N(diff; 0, se² + τ²) / N(diff; 0, se²) ]
    log_bf = 0.5 * math.log(se2 / (se2 + tau2)) + 0.5 * diff * diff * tau2 / (se2 * (se2 + tau2))
    p_value = min(1.0, math.exp(-log_bf))

    # Always-valid CI: solve log_bf(θ₀) = -log α for θ₀.
    #   log_bf(θ₀) = 0.5·log(se²/(se²+τ²)) + 0.5·(diff-θ₀)²·τ²/(se²·(se²+τ²))
    rhs = -math.log(alpha) - 0.5 * math.log(se2 / (se2 + tau2))
    rhs = max(rhs, 0.0)  # numerical guard
    margin_sq = rhs * 2.0 * se2 * (se2 + tau2) / tau2
    margin = math.sqrt(margin_sq) if margin_sq > 0 else float("inf")

    return TestResult(
        point_estimate=diff,
        ci_low=diff - margin,
        ci_high=diff + margin,
        p_value=p_value,
        method_name="msprt",
        metadata={
            "se": math.sqrt(se2),
            "tau": tau,
            "log_bf": log_bf,
            "n_control": n_c,
            "n_treatment": n_t,
        },
    )


def simulate_type_i_error_under_peeking(
    n_per_arm: int,
    n_peeks: int,
    n_trials: int = 1_000,
    sigma: float = 1.0,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Simulate the false-positive rate under repeated peeking, H0 true.

    Returns the type-I error of (a) a fixed-α Welch test peeked `n_peeks` times
    and (b) the mSPRT peeked the same way. The mSPRT rate should be near `alpha`
    regardless of `n_peeks`; the fixed-α rate inflates with `n_peeks`. This is
    the canonical demo of why naive peeking is dangerous.
    """
    rng = np.random.default_rng(seed)
    peek_at = np.linspace(20, n_per_arm, n_peeks, dtype=int)

    n_msprt_reject = 0
    n_naive_reject = 0
    for _ in range(n_trials):
        # Generate the full stream of n_per_arm observations per arm under H0.
        c = rng.normal(0.0, sigma, size=n_per_arm)
        t = rng.normal(0.0, sigma, size=n_per_arm)
        msprt_rejected = False
        naive_rejected = False
        for k in peek_at:
            mc = c[:k].mean()
            mt = t[:k].mean()
            sc = c[:k].var(ddof=1)
            st = t[:k].var(ddof=1)
            se2 = sc / k + st / k
            diff = mt - mc

            # Fixed-α z-test.
            z = diff / math.sqrt(se2) if se2 > 0 else 0.0
            if 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))) < alpha:
                naive_rejected = True

            # mSPRT with τ = σ (default).
            tau2 = sigma * sigma
            log_bf = 0.5 * math.log(se2 / (se2 + tau2)) + 0.5 * diff * diff * tau2 / (se2 * (se2 + tau2))
            if math.exp(-log_bf) < alpha:
                msprt_rejected = True
        if naive_rejected:
            n_naive_reject += 1
        if msprt_rejected:
            n_msprt_reject += 1

    return {
        "msprt_type_i_error": n_msprt_reject / n_trials,
        "naive_type_i_error": n_naive_reject / n_trials,
        "n_peeks": n_peeks,
        "alpha": alpha,
    }
