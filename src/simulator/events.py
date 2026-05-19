"""Event simulation with time-of-day / day-of-week seasonality and treatment effects.

Schema (long format, one row per event):
    event_id, user_id, event_type, ts, value
where event_type in {'session', 'conversion'} and `value` is latency_ms on
'session' rows and NULL on 'conversion' rows. Conversions are emitted as
separate events (rather than a column on the session row) so the downstream
dbt models can roll up arbitrary event-type metrics uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .users import SEGMENTS


@dataclass
class TreatmentEffect:
    """Per-variant effects applied during event generation.

    `conversion_lift` is multiplicative on the per-segment base rate (0.05 = +5%).
    `latency_lift_ms` is additive on each sampled latency (in milliseconds).

    `conversion_lift_by_segment` / `latency_lift_by_segment_ms` optionally
    override the scalar values for individual segments — required for the
    heterogeneous scenario the HTE module is built to recover.
    """

    conversion_lift: float = 0.0
    latency_lift_ms: float = 0.0
    conversion_lift_by_segment: dict[str, float] | None = None
    latency_lift_by_segment_ms: dict[str, float] | None = None


def _day_of_week_multiplier(dates: pd.DatetimeIndex) -> np.ndarray:
    """Weekends run ~30% quieter than weekdays — a plausible SaaS shape."""
    dow = dates.dayofweek.to_numpy()
    return np.where(dow >= 5, 0.7, 1.0)


def _hour_density(n_hours: int = 24) -> np.ndarray:
    """Sinusoidal hour-of-day density, peaking ~14:00 local."""
    hours = np.arange(n_hours)
    raw = 1.0 + 0.5 * np.sin(2 * np.pi * (hours - 8) / 24)
    return raw / raw.sum()


def _resolve_lift_arrays(
    u: pd.DataFrame,
    treatment_effects: dict[str, TreatmentEffect],
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized per-row lookup of (conversion_lift, latency_lift_ms) by (variant, segment).

    Default to the variant's scalar lift; if the variant defines a
    `*_by_segment` override and the user's segment is in it, that wins.
    """
    n = len(u)
    conv_lift = np.zeros(n, dtype=float)
    lat_lift = np.zeros(n, dtype=float)
    variants = u["variant"].to_numpy()
    segments = u["segment"].to_numpy()
    for variant, effect in treatment_effects.items():
        var_mask = variants == variant
        if not var_mask.any():
            continue
        conv_lift[var_mask] = effect.conversion_lift
        lat_lift[var_mask] = effect.latency_lift_ms
        if effect.conversion_lift_by_segment:
            for seg, lift in effect.conversion_lift_by_segment.items():
                conv_lift[var_mask & (segments == seg)] = lift
        if effect.latency_lift_by_segment_ms:
            for seg, lift in effect.latency_lift_by_segment_ms.items():
                lat_lift[var_mask & (segments == seg)] = lift
    return conv_lift, lat_lift


def generate_events(
    users: pd.DataFrame,
    exposures: pd.DataFrame,
    experiment_start: pd.Timestamp,
    experiment_days: int,
    treatment_effects: dict[str, TreatmentEffect],
    seed: int = 43,
) -> pd.DataFrame:
    """Generate `session` and `conversion` events for the experiment window.

    Sessions per (user, day) ~ Poisson(seg.sessions_per_day * dow_multiplier).
    Within a day, sessions are time-stamped by sampling hours from a sinusoidal
    density and uniform minutes/seconds. Each session emits one row with the
    latency in `value`; a session converts with probability
    `seg.base_conversion_rate * (1 + variant.conversion_lift)`, in which case
    a second 'conversion' row is emitted at the same timestamp.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(experiment_start).normalize()
    dates = pd.date_range(start, periods=experiment_days, freq="D")
    dow_mult = _day_of_week_multiplier(dates)
    hour_probs = _hour_density()

    u = users.merge(exposures[["user_id", "variant"]], on="user_id", how="inner")
    if u.empty:
        return pd.DataFrame(
            columns=["event_id", "user_id", "event_type", "ts", "value"]
        )

    base_conv = u["segment"].map(lambda s: SEGMENTS[s].base_conversion_rate).to_numpy()
    base_lambda = u["segment"].map(lambda s: SEGMENTS[s].sessions_per_day).to_numpy()
    log_mean = u["segment"].map(lambda s: SEGMENTS[s].latency_log_mean).to_numpy()
    log_sigma = u["segment"].map(lambda s: SEGMENTS[s].latency_log_sigma).to_numpy()
    conv_lift, lat_lift = _resolve_lift_arrays(u, treatment_effects)
    user_ids_arr = u["user_id"].to_numpy()

    p_conv_per_user = np.clip(base_conv * (1.0 + conv_lift), 0.0, 0.99)

    lambdas = base_lambda[:, None] * dow_mult[None, :]  # (n_users, n_days)
    counts = rng.poisson(lambdas)  # (n_users, n_days)
    flat_counts = counts.flatten()
    total_sessions = int(flat_counts.sum())
    if total_sessions == 0:
        return pd.DataFrame(
            columns=["event_id", "user_id", "event_type", "ts", "value"]
        )

    cell_idx = np.repeat(np.arange(counts.size), flat_counts)
    user_idx = cell_idx // counts.shape[1]
    day_idx = cell_idx % counts.shape[1]

    hours = rng.choice(24, size=total_sessions, p=hour_probs)
    mins = rng.integers(0, 60, size=total_sessions)
    secs = rng.integers(0, 60, size=total_sessions)
    session_dates = dates[day_idx]
    session_ts = (
        session_dates
        + pd.to_timedelta(hours, unit="h")
        + pd.to_timedelta(mins, unit="m")
        + pd.to_timedelta(secs, unit="s")
    )

    session_user_ids = user_ids_arr[user_idx]
    converted = rng.random(total_sessions) < p_conv_per_user[user_idx]
    latencies = rng.lognormal(log_mean[user_idx], log_sigma[user_idx]) + lat_lift[user_idx]

    session_df = pd.DataFrame(
        {
            "user_id": session_user_ids,
            "event_type": "session",
            "ts": session_ts,
            "value": latencies,
        }
    )
    conv_df = pd.DataFrame(
        {
            "user_id": session_user_ids[converted],
            "event_type": "conversion",
            "ts": session_ts[converted],
            "value": np.nan,
        }
    )

    events = (
        pd.concat([session_df, conv_df], ignore_index=True)
        .sort_values("ts", kind="stable")
        .reset_index(drop=True)
    )
    events.insert(0, "event_id", np.arange(len(events), dtype=np.int64))
    return events
