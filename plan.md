# Experimentation Platform — Portfolio Project Plan

## North Star

Build a **mini experimentation platform** that goes beyond "ran a t-test, p<0.05, ship it." The point is to demonstrate the judgment of a senior product analyst: catching the failure modes that kill real-world experiments — peeking, multiple comparisons, novelty effects, segment heterogeneity, and guardrail violations.

Final deliverable: a working repo + Streamlit app + a written case study where you ship one fake feature, run the full experiment review pipeline on it, and walk through what you caught.

---

## What makes this stand out

The 90th-percentile candidate version of this project includes things 99% of portfolio projects skip:

1. **A simulated SaaS product with realistic user behavior** — heterogeneous user segments, time-of-day effects, novelty bias, weekly seasonality — not iid coin flips. This is the foundation; everything else is only as good as the data-generating process.
2. **A feature flag / assignment service** that does deterministic bucketing (hash-mod on user_id), supports mutually exclusive experiment layers, and handles sample ratio mismatch (SRM) detection.
3. **A statistics engine** with multiple methods side-by-side: Welch's t-test, bootstrap, CUPED, delta method for ratio metrics, sequential testing (mSPRT or always-valid CIs).
4. **A guardrail framework** — every experiment registers a primary metric *and* a set of guardrails (latency, retention, error rate). "Wins" that break guardrails get flagged, not shipped.
5. **Heterogeneous treatment effects** via causal trees / uplift modeling — find the segments where the effect is real vs. noise.
6. **A diagnostic suite that runs automatically**: SRM check, novelty effect detection (effect decay over time), Simpson's paradox detection across segments, pre-experiment A/A bias check.
7. **A Streamlit "experiment review" UI** that mimics what an analyst would actually use at work — a single page per experiment with all checks rendered.

The framing for interviews: *"Most A/B test write-ups stop at 'the variant won.' This platform catches the cases where the variant 'won' but shouldn't ship — and that's what I'd actually be doing on day one."*

---

## Stack

- **Python 3.11+** — core language
- **DuckDB** — local OLAP, fast, no infra. (BigQuery is optional v2.)
- **dbt-duckdb** — transforms experiment exposures + events → metric tables
- **Polars or pandas** — pick one for analytics; Polars if you want a slight modernity flex
- **scipy + statsmodels** — base stats
- **econml or causalml** — HTE / uplift
- **Streamlit** — review UI
- **pytest** — yes, write tests. Most candidates skip this. Don't.
- **uv** — package manager. Faster than pip and shows you keep up.

Keep the dependency list lean. Recruiters skim `pyproject.toml`.

---

## Architecture (high level)

```
┌─────────────────────┐
│ Simulator           │  generates users, events, exposures into DuckDB
│ (synthetic SaaS)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Assignment Service  │  deterministic bucketing, layers, SRM check
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ dbt models          │  exposures → metric rollups → analysis tables
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Stats Engine        │  t-test, CUPED, delta method, mSPRT, HTE
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Diagnostics         │  SRM, novelty, Simpson's, A/A bias, guardrails
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Streamlit UI        │  per-experiment review page
└─────────────────────┘
```

---

## Repo layout

```
experiment-platform/
├── README.md                      # the case study + screenshots live here
├── pyproject.toml
├── .gitignore
├── data/
│   └── warehouse.duckdb           # gitignored
├── src/
│   ├── simulator/
│   │   ├── users.py               # user generation w/ segments
│   │   ├── events.py              # event simulation
│   │   └── scenarios.py           # named scenarios (novelty, SRM, etc.)
│   ├── assignment/
│   │   ├── bucketing.py           # hash-mod deterministic assignment
│   │   └── layers.py              # mutually exclusive layers
│   ├── stats/
│   │   ├── frequentist.py         # t-test, z-test, proportions
│   │   ├── bootstrap.py
│   │   ├── cuped.py               # variance reduction
│   │   ├── delta_method.py        # ratio metrics
│   │   ├── sequential.py          # mSPRT / always-valid CIs
│   │   └── power.py               # sample size + MDE calculators
│   ├── diagnostics/
│   │   ├── srm.py                 # sample ratio mismatch
│   │   ├── novelty.py             # effect-over-time decay
│   │   ├── simpsons.py            # segment reversal detection
│   │   ├── aa_test.py             # pre-launch bias check
│   │   └── guardrails.py          # guardrail evaluation framework
│   ├── hte/
│   │   └── causal_trees.py        # heterogeneous effects via econml
│   └── ui/
│       └── app.py                 # Streamlit review UI
├── dbt_project/
│   ├── dbt_project.yml
│   ├── models/
│   │   ├── staging/
│   │   │   ├── stg_exposures.sql
│   │   │   └── stg_events.sql
│   │   └── marts/
│   │       ├── fct_experiment_metrics.sql
│   │       └── fct_experiment_daily.sql
│   └── profiles.yml
├── notebooks/
│   ├── 01_simulator_demo.ipynb
│   ├── 02_one_experiment_end_to_end.ipynb
│   └── 03_failure_mode_gallery.ipynb   # the money notebook
└── tests/
    ├── test_bucketing.py
    ├── test_stats.py
    ├── test_srm.py
    └── test_cuped.py
```

---

## Phased build plan

Each phase is roughly one focused session. Do not skip ahead — earlier phases catch bugs that would otherwise compound.

### Phase 0 — Setup (30 min)

- Initialize repo with `uv init`, add `duckdb`, `dbt-duckdb`, `scipy`, `statsmodels`, `pandas`, `numpy`, `streamlit`, `pytest`, `econml`.
- Wire up `.gitignore` (exclude `data/*.duckdb`, `.venv`, `__pycache__`).
- Stub the directory structure above. Empty `__init__.py` files.
- One smoke test: `pytest` runs and finds zero tests cleanly.

### Phase 1 — Simulator (the foundation; spend real time here)

The simulator is the unsexy core. Every interesting failure mode you'll demonstrate later has to be **planted here on purpose** so you can detect it downstream. If the simulator is bad, the whole project is bad.

**Build:**
- `User` generation with 3–4 segments (e.g., `power_user`, `casual`, `new_signup`, `enterprise`) with different baseline conversion rates and event frequencies.
- Event generation: sessions per day drawn from a per-segment Poisson, conversion events with per-segment base rates, latency events with a lognormal distribution.
- Time-of-day and day-of-week seasonality (sinusoidal multipliers on event rates).
- Write everything to DuckDB tables: `users`, `events`, `exposures`.

**Scenario presets** (each is a function that produces a dataset with a known ground truth):
- `clean_lift` — variant truly +5% conversion, uniform across segments.
- `novelty_effect` — variant +10% week 1, decays to +0% by week 4.
- `srm_bug` — assignment is 55/45 instead of 50/50.
- `simpsons` — variant loses overall but wins in every segment (or vice versa) due to mix shift.
- `guardrail_violation` — variant +5% conversion but +30ms latency.
- `heterogeneous` — variant +20% for one segment, -5% for another, ~0 average.
- `aa_drift` — no real effect, but pre-period imbalance creates a false positive.

These presets become the test fixtures for everything downstream.

### Phase 2 — Assignment & exposure logging

- `assign(user_id, experiment_id, salt) -> variant` using `hashlib.md5` mod N. Deterministic, salt-namespaced.
- Layer system: an experiment belongs to a layer; users in layer L are bucketed once per layer, ensuring mutual exclusion.
- Write exposures to DuckDB as `(user_id, experiment_id, variant, exposed_at)`.
- Unit test: same `(user_id, salt)` always returns same variant; large-N assignment is ~uniform; layers don't bleed.

### Phase 3 — dbt models

- `stg_exposures` — clean exposures, dedupe to first exposure per user per experiment.
- `stg_events` — clean events, join to first-exposure timestamp so only post-exposure events count.
- `fct_experiment_metrics` — one row per `(experiment, variant, user)` with primary metric + guardrails.
- `fct_experiment_daily` — same but daily, for novelty / time-effect plots.
- Run `dbt build` and verify. Add at least one dbt test (e.g., `unique` on user+experiment).

### Phase 4 — Stats engine

Implement each method as a pure function: takes a dataframe + column names, returns a dataclass with `point_estimate`, `ci_low`, `ci_high`, `p_value`, `method_name`.

- **Welch's t-test** — continuous metrics, unequal variances.
- **Two-proportion z-test** — conversion metrics.
- **Bootstrap** — percentile CI, n=10,000 resamples, for any metric.
- **Delta method** — for ratio metrics like clicks/impressions where the unit of randomization (user) ≠ unit of analysis (impression).
- **CUPED** — needs pre-experiment covariate; uses pre-period metric to reduce variance. Show the variance reduction explicitly.
- **mSPRT or always-valid CIs** — sequential testing so peeking is safe. Even a basic implementation here is a big differentiator.
- **Power & sample size calculator** — given baseline rate, MDE, alpha, power → required sample size. Plus the inverse: given current N, what's the MDE you can detect?

**Test every one** against a known scipy result or a textbook example.

### Phase 5 — Diagnostics

Each diagnostic returns a `DiagnosticResult(status: pass|warn|fail, message, evidence)`.

- **SRM check** — chi-square test on observed vs. expected bucket sizes. Fail if p < 0.001.
- **A/A pre-period check** — run the same test on the pre-exposure window. Should be null. If not, flag bias.
- **Novelty detection** — fit a linear model of daily effect vs. days-since-exposure. Significant negative slope = decaying effect.
- **Simpson's paradox** — compute the effect overall and per segment. If overall sign ≠ majority of segment signs (weighted), flag.
- **Multiple comparisons** — when reporting on N segments or N metrics, apply Benjamini-Hochberg FDR control. Show both raw and adjusted p-values.
- **Guardrail evaluator** — given a config like `{metric: latency_p95, direction: lower_is_better, threshold: +5%}`, compute and verdict.

### Phase 6 — HTE

- Use `econml` `CausalForestDML` or `causalml` uplift trees on the simulated data.
- Input: user features (segment, tenure, pre-period activity), treatment indicator, outcome.
- Output: estimated CATE per user → bucketed into deciles → plot effect by decile.
- On the `heterogeneous` scenario this should clearly recover the per-segment effect.

### Phase 7 — Streamlit review UI

One page per experiment. From top to bottom:
1. **Header** — experiment name, dates, hypothesis, owner.
2. **Health checks panel** — SRM, A/A, sample size achieved vs. planned. Each as a colored pill.
3. **Primary metric** — point estimate, CI, p-value, sequential boundary (if applicable). Time-series of cumulative effect.
4. **Guardrails table** — each guardrail with status.
5. **Segment breakdown** — table of effect by segment with Simpson's flag.
6. **HTE plot** — CATE deciles.
7. **Recommendation** — auto-generated string: "Ship / Don't ship / Iterate" with reasoning.

Keep the styling minimal and clean. Recruiters will screenshot this — make it screenshot-able.

### Phase 8 — The case study (README)

This is the deliverable that actually gets you the interview. Structure:

1. **TL;DR** — one paragraph + one screenshot.
2. **The problem** — why naive A/B testing fails in production.
3. **What I built** — architecture diagram, stack.
4. **Five failure modes, caught** — for each of SRM / novelty / Simpson's / guardrail / HTE, show: synthetic scenario → what a naive analysis would conclude → what the platform catches → screenshot.
5. **Stats methods, when to use which** — short table of t-test vs. CUPED vs. delta vs. sequential, with one-line rationale for each.
6. **What I'd build next** — Bayesian methods, switchback experiments for marketplace, interference detection. (Showing you know the limits of what you built is itself a senior signal.)

Pin the repo on your GitHub. Link it from the top of your resume.

---

## Anti-patterns to avoid

- **Don't over-engineer the simulator before the stats work.** Get a clean_lift scenario producing real data, build the stats end-to-end on that, then add scenarios.
- **Don't use real data.** Real datasets bring confounders you can't explain. Synthetic data with known ground truth is the entire point — you can prove your method works.
- **Don't build a UI before the stats are correct.** Streamlit is dessert.
- **Don't skip tests.** A failing `test_srm.py` that catches your own bucketing bug is the kind of story that wins interviews.
- **Don't write the README last as an afterthought.** Draft the case study outline in week 1 — it forces you to know what story you're telling.

---

## Stretch goals (only after Phase 8 ships)

- Bayesian A/B testing with `pymc` — posterior over the effect, probability of being best.
- Switchback / interference: marketplace experiments where treatment and control affect each other.
- A simple Postgres-backed feature flag service with a FastAPI front, instead of just in-process.
- Cost-aware experimentation — incorporate ops cost into the ship decision.
- Write a blog post on one of the failure modes; link it from the README.

---

## Suggested order of operations for Claude Code

1. Read this plan.
2. Confirm the stack and ask before deviating.
3. Execute Phase 0 (setup) in one commit.
4. Execute Phase 1 (simulator) — get `clean_lift` working end-to-end before adding other scenarios.
5. Pause and let the user inspect the simulated data before moving on.
6. Proceed phase by phase, one commit per phase minimum.
7. Tests written alongside code, not after.
8. UI last.
9. README and case study written iteratively from Phase 4 onward, polished at the end.
