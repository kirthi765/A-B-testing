-- Events stamped with the user's variant. We *inner* join to stg_exposures
-- (so events from un-exposed users are dropped) and filter to ts >= exposed_at
-- so pre-exposure activity never contributes to the treatment effect.
--
-- For users enrolled in multiple experiments (Phase 2's layered design), this
-- join intentionally fans out: one event row becomes N rows, one per
-- experiment the user is in. Downstream metrics group by experiment_id, so
-- the fan-out is the correct semantics.

with events as (
    select
        event_id,
        user_id,
        event_type,
        ts,
        value
    from {{ source('raw', 'events') }}
),
exposures as (
    select
        user_id,
        experiment_id,
        variant,
        exposed_at
    from {{ ref('stg_exposures') }}
)
select
    e.event_id,
    e.user_id,
    x.experiment_id,
    x.variant,
    e.event_type,
    e.ts,
    e.value as latency_ms
from events e
inner join exposures x using (user_id)
where e.ts >= x.exposed_at
