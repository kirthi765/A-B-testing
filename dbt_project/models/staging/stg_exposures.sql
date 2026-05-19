-- First exposure per (user_id, experiment_id). If the assignment service ever
-- writes a duplicate (re-bucketing on re-exposure, late-arriving event, etc.),
-- the analysis must lock onto the *first* assignment — otherwise a user could
-- silently switch variants mid-experiment and contaminate every downstream
-- metric.

with ranked as (
    select
        user_id,
        experiment_id,
        variant,
        exposed_at,
        row_number() over (
            partition by user_id, experiment_id
            order by exposed_at asc
        ) as rn
    from {{ source('raw', 'exposures') }}
)
select
    user_id,
    experiment_id,
    variant,
    exposed_at
from ranked
where rn = 1
