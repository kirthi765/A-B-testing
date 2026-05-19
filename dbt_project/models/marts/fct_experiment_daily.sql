-- Daily roll-up per (experiment, variant). Powers the novelty-effect plot
-- (treatment lift over time) and time-series guardrail checks downstream.
-- Note: `n_active_users` is exact post-exposure activity, not the cumulative
-- exposed cohort — the diagnostic suite re-derives the cohort if needed.

select
    experiment_id,
    variant,
    cast(date_trunc('day', ts) as date)                                            as event_date,
    count(distinct user_id)                                                        as n_active_users,
    sum(case when event_type = 'session'    then 1 else 0 end)                     as n_sessions,
    sum(case when event_type = 'conversion' then 1 else 0 end)                     as n_conversions,
    cast(sum(case when event_type = 'conversion' then 1 else 0 end) as double)
        / nullif(sum(case when event_type = 'session' then 1 else 0 end), 0)        as conversion_rate,
    avg(case when event_type = 'session' then latency_ms end)                       as mean_latency_ms
from {{ ref('stg_events') }}
group by experiment_id, variant, event_date
order by experiment_id, variant, event_date
