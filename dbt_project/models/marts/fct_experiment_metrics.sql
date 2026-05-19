-- One row per (experiment_id, variant, user_id). Includes the primary metric
-- (conversion) plus the guardrail metric (latency). LEFT JOIN from exposures
-- so every exposed user appears even with zero post-exposure activity —
-- otherwise SRM / activation rates downstream would be biased.

with exposures as (
    select
        experiment_id,
        variant,
        user_id,
        u.segment
    from {{ ref('stg_exposures') }}
    left join {{ source('raw', 'users') }} u using (user_id)
),
events as (
    select
        experiment_id,
        user_id,
        event_type,
        latency_ms
    from {{ ref('stg_events') }}
),
rolled as (
    select
        x.experiment_id,
        x.variant,
        x.user_id,
        x.segment,
        sum(case when e.event_type = 'session'    then 1 else 0 end) as n_sessions,
        sum(case when e.event_type = 'conversion' then 1 else 0 end) as n_conversions,
        avg(case when e.event_type = 'session' then e.latency_ms end) as mean_latency_ms,
        max(case when e.event_type = 'session' then e.latency_ms end) as max_latency_ms
    from exposures x
    left join events e
        on x.experiment_id = e.experiment_id
        and x.user_id       = e.user_id
    group by x.experiment_id, x.variant, x.user_id, x.segment
)
select
    experiment_id,
    variant,
    user_id,
    segment,
    n_sessions,
    n_conversions,
    cast(n_conversions as double) / nullif(n_sessions, 0) as conversion_rate,
    (n_conversions > 0)                                    as converted,
    mean_latency_ms,
    max_latency_ms
from rolled
