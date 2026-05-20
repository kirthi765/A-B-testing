# Experimentation Platform

A mini A/B testing review system that goes beyond *"ran a t-test, p<0.05, ship it."* It catches the failure modes that actually kill experiments in production: sample ratio mismatch, novelty effects, Simpson's reversals, guardrail regressions, and segment heterogeneity.

![Experiment review UI](docs/screenshots/review_page.png)

*Streamlit review page rendering one experiment — health checks, primary metric, guardrails, segment breakdown, recommendation. See [`docs/screenshots/`](docs/screenshots/) for per-scenario captures.*

---

## TL;DR

Most A/B test writeups stop when the variant wins. This platform catches the cases where the variant "won" but shouldn't ship — and that's what a senior product analyst does on day one of the job.

Built end-to-end:

- A **synthetic SaaS simulator** with realistic user segments, time-of-day and weekly seasonality, and seven named failure-mode scenarios (each plants a specific bug on purpose).
- A **deterministic hash-mod assignment service** with mutually-exclusive layers — SRM-safe by construction; verified by uniformity tests at large N.
- **dbt models** that roll raw exposures and events into per-user and daily fact tables (`fct_experiment_metrics`, `fct_experiment_daily`).
- A **stats engine** with six methods returning a uniform `TestResult`: Welch's t-test, two-proportion z-test, percentile bootstrap, delta method for ratio metrics, CUPED variance reduction, and mSPRT for peek-safe sequential testing.
- A **diagnostic suite** that returns a uniform `DiagnosticResult` per check: SRM (chi-square), A/A pre-period bias, novelty decay (linear fit on daily lift), Simpson's reversal, Benjamini–Hochberg FDR, and configurable guardrails.
- **Heterogeneous treatment effects** via `econml.dml.CausalForestDML` with decile and per-segment CATE summaries.
- A **Streamlit review page** that renders all of the above for one experiment, with a Ship / Don't ship / Iterate recommendation generated from the analysis.

Verified by **116 passing tests** — including end-to-end *"this scenario must trigger this diagnostic"* tests for each planted failure mode, and parity checks against `scipy.stats` and `statsmodels` for the stats methods.

---

## The problem

A naive A/B-testing pipeline looks like this:

1. Pick a metric.
2. Run a t-test.
3. If p < 0.05, ship.

In production, this pipeline systematically ships the wrong things. Five of the most common ways it fails:

| Failure mode | What goes wrong | What the naive pipeline does |
|---|---|---|
| Sample Ratio Mismatch (SRM) | Bucketer is broken; arms aren't a random partition of users | Reports a biased effect estimate |
| Novelty effect | Lift in week 1 evaporates by week 4 | Averages the two and ships a feature that won't retain |
| Simpson's paradox | Treatment wins in every segment but loses overall, or vice versa | Reads the aggregate and gets the sign wrong |
| Guardrail violation | Treatment improves conversion but tanks latency | Sees the conversion win and ships a net-negative change |
| Heterogeneous effects | +20% for power users, −5% for casual; aggregate ≈ 0 | Reads the null and kills a feature that would help power users |

This platform plants each of these failure modes in synthetic data — where the ground truth is known by construction — then verifies that the diagnostic suite catches them.

---

## Architecture

```
┌─────────────────────┐
│ Simulator           │  users, events, exposures with planted ground-truth bugs
│ (synthetic SaaS)    │
└──────────┬──────────┘
           │  DuckDB warehouse
           ▼
┌─────────────────────┐
│ Assignment Service  │  hashlib.md5 → bucket; mutually-exclusive layers
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ dbt models          │  stg_exposures, stg_events
│                     │  → fct_experiment_metrics (per-user)
│                     │  → fct_experiment_daily   (per day × variant)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐       ┌─────────────────────┐
│ Stats Engine        │       │ Diagnostics         │
│ Welch's t, z-test,  │       │ SRM, novelty,       │
│ bootstrap, CUPED,   │       │ Simpson's, BH-FDR,  │
│ delta, mSPRT        │       │ guardrails          │
└──────────┬──────────┘       └──────────┬──────────┘
           │                             │
           └──────────────┬──────────────┘
                          ▼
                ┌─────────────────────┐
                │ HTE                 │  econml CausalForestDML →
                │                     │  per-decile + per-segment CATE
                └──────────┬──────────┘
                           ▼
                ┌─────────────────────┐
                │ Streamlit UI        │  per-experiment review page
                │                     │  with Ship/Iterate/Don't ship
                └─────────────────────┘
```

### Stack

- **Python 3.11**, **DuckDB** (file-based OLAP, no infra), **dbt-duckdb** for transforms
- **pandas**, **numpy**, **scipy**, **statsmodels** for stats
- **econml** (CausalForestDML) for HTE
- **Streamlit** + **Altair** for the review UI
- **pytest** for verification
- **uv** as the package manager

---

## Five failure modes, caught

Each scenario lives in [`src/simulator/scenarios.py`](src/simulator/scenarios.py); the numbers below come from running [`scripts/run_case_study.py`](scripts/run_case_study.py) at n=10,000 users × 28 days (15,000 for `novelty_effect`).

### 1. Sample Ratio Mismatch — [`srm_bug`](src/simulator/scenarios.py)

**What's planted.** Bucketer mis-configured to a 55/45 split instead of 50/50. A real +5% conversion lift exists, but the assignment is biased.

**What a naive analysis says.** Primary z-test on `converted`: Δ = +1.8 pp, CI [-0.10 pp, +3.6 pp], p = 0.063. Conclusion: "not significant, but trending positive — let it run another week."

**What the platform catches.** SRM chi-square **p = 2 × 10⁻²²**. Observed 0.451 / 0.549 vs expected 0.500 / 0.500. The recommendation engine refuses to read the primary metric until the bucketer is fixed — because every effect estimate from a non-random assignment is biased.

### 2. Novelty effect — [`novelty_effect`](src/simulator/scenarios.py)

**What's planted.** Variant lift decays linearly from +10% on day 1 to +0% on day 28. Aggregated over the full window, the average lift looks like +5%.

**What a naive analysis says.** Primary z-test: Δ = +1.7 pp, p = 0.026 — significant win, ship it. Per-user rate t-test: Δ = +0.22 pp, p = 2 × 10⁻⁴.

**What the platform catches.** Novelty detector fits `daily_lift ~ day_index` via OLS: slope = **−0.000237/day**, p = **0.0041**. Verdict: novelty decay detected, hold out the early days or extend the experiment before shipping.

### 3. Simpson's paradox — [`simpsons`](src/simulator/scenarios.py)

**What's planted.** Variant is +5% within every segment, but assignment is segment-correlated: low-converting new_signups disproportionately land in treatment (70% → treatment) and high-converting power_users in control (20% → treatment). Overall ratio stays near 50/50 so SRM doesn't fire.

**What a naive analysis says.** Primary z-test: Δ = **−5.2 pp**, CI [−7.1 pp, −3.4 pp], p ≈ 0. Latency also looks worse (+10%). Conclusion: "treatment loses everywhere, kill it."

**What the platform catches.** Simpson's check: overall sign = **−1**, size-weighted majority of segment signs = **+1** → reversal detected. The platform surfaces a per-segment table showing treatment wins in every segment, and the recommendation points the analyst at the mix-shift, not the (real) +5% lift.

### 4. Guardrail violation — [`guardrail_violation`](src/simulator/scenarios.py)

**What's planted.** Variant +5% conversion **and** +30 ms latency (≈ +17% on the ~180 ms baseline).

**What a naive analysis says.** Primary z-test: Δ = +3.1 pp, p = 0.001. Per-user rate: Δ = +0.30 pp, p < 10⁻⁴. Both metrics significant: "clear winner, ship."

**What the platform catches.** Latency guardrail: **+15.0% relative**, p < 10⁻⁴, threshold ±5%. Verdict: **FAIL** — directional violation and statistically significant. The recommendation engine treats this as ship-blocking: a primary-metric win that breaks a guardrail is not a win.

### 5. Heterogeneous treatment effects — [`heterogeneous`](src/simulator/scenarios.py)

**What's planted.** Per-segment lifts: power_user +20%, casual −5%, new_signup +10%, enterprise −2%. The aggregate effect is close to zero.

**What a naive analysis says.** Primary z-test: Δ = +0.16 pp, p = 0.86. Per-user rate: Δ = +0.16 pp, p = 0.03. Mixed signal — analyst concludes "no effect" and kills the feature.

**What the platform catches.** The Simpson's check fires here too (overall sign +1, weighted majority −1) — a signal that the aggregate is hiding segment-level disagreement. The HTE module (econml `CausalForestDML` over segment one-hots) recovers the per-segment ranking: power_user has the largest CATE by a wide margin; casual / enterprise sit at the bottom. The recommendation: don't kill the feature, ship it to power_users.

---

### A note on the sixth scenario: [`aa_drift`](src/simulator/scenarios.py)

The simulator also plants a sixth bug — assignment-conditional bias that creates a **false positive** on the primary metric even with zero real lift. On the current pipeline this scenario reports Δ = +8.95 pp, p < 10⁻⁴ — a slam-dunk "ship" — and **no diagnostic catches it**, because catching it requires pre-period A/A data the simulator doesn't currently emit. This is the next gap to close; see *What I'd build next* below.

---

## Stats methods: when to use which

Every method returns the same `TestResult(point_estimate, ci_low, ci_high, p_value, method_name, metadata)`, so the diagnostics and the UI don't care which one ran.

| Method | When to reach for it | Limitation |
|---|---|---|
| **Welch's t-test** | Continuous metric (latency, revenue per user), unequal variance between arms | Sensitive to heavy tails / outliers |
| **Two-proportion z-test** | Binary metric (converted, retained), large N | Approximation breaks at very small n; saturates on per-user binary at long horizons |
| **Bootstrap (percentile CI)** | Heavy-tailed metric, or any statistic other than the mean (median, p95) | O(n × n_resamples) memory; CI-dual p-value rather than a permutation p-value |
| **Delta method** | Ratio metric where unit of randomization ≠ unit of analysis (CTR per user-impression) | Requires aggregating to the user level first |
| **CUPED** | A pre-experiment covariate strongly correlated with the metric is available | Needs pre-period data; θ must be fit on pooled data, not per arm |
| **mSPRT (always-valid)** | You expect to peek at the experiment multiple times during its run | Wider CIs than fixed-α; the default τ = pooled SD is conservative for small effects |

The Welch and z-test p-values match `scipy.stats.ttest_ind` and `statsmodels.proportions_ztest` to ~10⁻¹². CUPED's variance reduction matches `ρ²(y, x)` within MC noise. The mSPRT peeking simulation verifies type-I error stays at or below α under repeated peeking, while a fixed-α z-test inflates past 10% with 20 peeks.

---

## What I'd build next

The senior signal here is knowing what the current platform *can't* do.

- **Pre-period A/A harness.** Extend the simulator to emit pre-period events and wire the existing `check_aa` into the review UI. This is what would close the `aa_drift` gap above — currently the most embarrassing miss in the suite.
- **Bayesian A/B testing** via `pymc` — posterior over the effect, probability-of-being-best for portfolio review. Better story for stakeholders than "p = 0.04."
- **Switchback / interference detection** for marketplace experiments where treatment and control affect each other (rideshare, dating, two-sided markets). Standard A/B is mis-specified for these.
- **Postgres-backed feature flag service** with a FastAPI front, replacing the in-process assignment — the realistic deployment shape.
- **Cost-aware experimentation.** Roll ops cost into the recommendation engine: a +1% conversion lift that requires a 2× infra spend isn't a ship.
- **MDE × allocation × duration calculator** in the UI — let the analyst negotiate the trade-offs interactively before the experiment starts, not after.

---

## Running it

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+ (uv will install Python for you).

```bash
# Install dependencies
uv sync

# Run all tests (~30s; the causal-forest test is the slow one)
uv run pytest

# Generate a scenario into the warehouse
uv run python -m src.simulator.scenarios clean_lift
# or: srm_bug | novelty_effect | simpsons | guardrail_violation | aa_drift | heterogeneous

# Build the dbt marts
cd dbt_project && uv run dbt build --profiles-dir . && cd ..

# Launch the review UI
uv run streamlit run src/ui/app.py
```

Reproducibility artifact: [`scripts/run_case_study.py`](scripts/run_case_study.py) runs every scenario in-process and prints the diagnostic verdict + headline numbers that this README cites.

---

## Repo layout

```
experiment-platform/
├── README.md                         # this file
├── pyproject.toml
├── plan.md                           # the original build plan
├── src/
│   ├── simulator/                    # users, events, scenarios + planted ground-truth bugs
│   ├── assignment/                   # hash-mod bucketing + mutually-exclusive layers
│   ├── stats/                        # Welch's t, z-test, bootstrap, delta, CUPED, mSPRT, power
│   ├── diagnostics/                  # SRM, A/A, novelty, Simpson's, BH-FDR, guardrails
│   ├── hte/                          # econml CausalForestDML wrapper + decile / segment summaries
│   └── ui/                           # data loader, analysis pipeline, Streamlit page
├── dbt_project/
│   └── models/
│       ├── staging/                  # stg_exposures, stg_events, sources.yml, schema.yml
│       └── marts/                    # fct_experiment_metrics, fct_experiment_daily
├── scripts/
│   ├── inspect_warehouse.py          # quick warehouse inspection
│   └── run_case_study.py             # in-process scenario harness, prints README's numbers
├── tests/                            # 116 tests covering every module
└── docs/
    └── screenshots/                  # UI captures referenced from this README
```
