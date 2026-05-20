"""Heterogeneous Treatment Effects via `econml.dml.CausalForestDML`.

The wrapper here is intentionally thin — econml has a stable, well-documented
API, and reimplementing it serves no one. What this module *adds* is the
user-level pipeline glue: feature engineering from `users + events`,
deterministic random_state plumbing, and decile / segment summaries that
the review UI plots directly.

Why a causal forest (vs a simpler T-learner)?
  - Built-in cross-fitting (double ML) — separates the outcome and treatment
    nuisance models from the effect estimator, which is the standard way to
    keep CATE estimates from being contaminated by overfitting on Y or T.
  - Honest splits — separate sub-samples for tree construction and leaf-mean
    estimation, which corrects the well-known overfitting bias of regression
    trees on causal targets.
  - Confidence intervals around CATE come for free.

For the toy demo case here (4 segments, ~5k users), a stratified mean would
recover the same answer faster. The forest scales gracefully to high-d
features (tenure, pre-period activity, etc.) that we'd add in a real platform.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def fit_causal_forest(
    df: pd.DataFrame,
    *,
    treatment_col: str,
    outcome_col: str,
    feature_cols: list[str],
    n_estimators: int = 100,
    random_state: int = 42,
    discrete_treatment: bool = True,
) -> CausalForestDML:
    """Fit a CausalForestDML on user-level data.

    `treatment_col` must be 0/1 when `discrete_treatment=True`. `outcome_col`
    can be continuous or binary; the underlying RF models handle both. We
    pin the random_state on the model *and* the nuisance estimators so two
    runs on the same data produce the same CATE.
    """
    # econml requires `n_estimators % subforest_size == 0` (subforest_size defaults
    # to 4); surface that here rather than wait for the cryptic traceback inside fit.
    if n_estimators % 4 != 0:
        lower = n_estimators - (n_estimators % 4)
        upper = lower + 4
        raise ValueError(
            f"n_estimators must be divisible by 4 (econml subforest_size); "
            f"got {n_estimators}. Try {lower} or {upper}."
        )

    X = df[feature_cols].to_numpy(dtype=float)
    T = df[treatment_col].to_numpy()
    Y = df[outcome_col].to_numpy(dtype=float)

    model = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=n_estimators, random_state=random_state),
        model_t=RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        if discrete_treatment
        else RandomForestRegressor(n_estimators=n_estimators, random_state=random_state),
        n_estimators=n_estimators,
        discrete_treatment=discrete_treatment,
        random_state=random_state,
    )
    model.fit(Y, T, X=X)
    return model


def predict_cate(model: CausalForestDML, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Return per-row CATE estimates aligned to `df.index`."""
    return np.asarray(model.effect(df[feature_cols].to_numpy(dtype=float)))


def cate_by_decile(cate: np.ndarray, n_buckets: int = 10) -> pd.DataFrame:
    """Group CATE estimates into equal-sized rank buckets and summarize.

    Rank-based bucketing (not value-cut) avoids degenerate behavior when
    many rows share an identical estimate (e.g. when features are coarse).
    """
    if len(cate) == 0:
        return pd.DataFrame(columns=["bucket", "n", "cate_mean", "cate_p05", "cate_p95"])
    s = pd.Series(cate, name="cate")
    bucket = pd.qcut(s.rank(method="first"), q=n_buckets, labels=False) + 1
    out = (
        pd.DataFrame({"cate": s, "bucket": bucket})
        .groupby("bucket")
        .agg(
            n=("cate", "size"),
            cate_mean=("cate", "mean"),
            cate_p05=("cate", lambda v: float(np.quantile(v, 0.05))),
            cate_p95=("cate", lambda v: float(np.quantile(v, 0.95))),
        )
        .reset_index()
        .sort_values("bucket")
        .reset_index(drop=True)
    )
    return out


def cate_by_segment(cate: np.ndarray, segments: pd.Series) -> pd.DataFrame:
    """Mean CATE per known segment label. Used to validate that the CATE
    ranking aligns with the planted per-segment lifts in the heterogeneous
    scenario."""
    df = pd.DataFrame({"cate": np.asarray(cate), "segment": segments.to_numpy()})
    return (
        df.groupby("segment")
        .agg(
            n=("cate", "size"),
            cate_mean=("cate", "mean"),
            cate_std=("cate", "std"),
        )
        .reset_index()
        .sort_values("cate_mean", ascending=False)
        .reset_index(drop=True)
    )


def build_user_level_dataset(
    users: pd.DataFrame,
    events: pd.DataFrame,
    exposures: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate the raw simulator output to one-row-per-user with:
    `segment, variant, treatment (0/1), n_sessions, n_conversions, converted`.

    Designed for the HTE pipeline — the caller picks which outcome to use
    (`n_conversions` for count-scale CATE, `converted` for binary CATE).
    """
    sessions_only = events[events["event_type"] == "session"]
    n_sessions = sessions_only.groupby("user_id").size().rename("n_sessions")
    mean_latency = (
        sessions_only.groupby("user_id")["value"].mean().rename("mean_latency_ms")
    )
    n_conversions = (
        events[events["event_type"] == "conversion"]
        .groupby("user_id")
        .size()
        .rename("n_conversions")
    )
    df = (
        users.merge(exposures[["user_id", "variant"]], on="user_id", how="inner")
        .merge(n_sessions, left_on="user_id", right_index=True, how="left")
        .merge(n_conversions, left_on="user_id", right_index=True, how="left")
        .merge(mean_latency, left_on="user_id", right_index=True, how="left")
        .fillna({"n_sessions": 0, "n_conversions": 0})
    )
    df["treatment"] = (df["variant"] == "treatment").astype(int)
    df["converted"] = (df["n_conversions"] > 0).astype(int)
    # Per-user session-level conversion rate. NaN for users with 0 sessions —
    # downstream tests that need it should `.dropna(subset=["conversion_rate"])`.
    df["conversion_rate"] = df["n_conversions"] / df["n_sessions"].where(
        df["n_sessions"] > 0
    )
    df["n_sessions"] = df["n_sessions"].astype(int)
    df["n_conversions"] = df["n_conversions"].astype(int)
    return df


def one_hot_segments(df: pd.DataFrame, segment_col: str = "segment") -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode `segment_col` in place; returns the augmented df and the
    list of new feature column names. Convenience for the standard pipeline."""
    dummies = pd.get_dummies(df[segment_col], prefix="seg").astype(int)
    out = pd.concat([df, dummies], axis=1)
    return out, dummies.columns.tolist()
