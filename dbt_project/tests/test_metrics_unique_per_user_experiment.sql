-- The fact table must have exactly one row per (experiment_id, user_id). If
-- the LEFT JOIN to events ever explodes — say someone adds a new model that
-- forgets to GROUP BY user — this catches it before the t-test silently
-- double-counts a user.

select
    experiment_id,
    user_id,
    count(*) as row_count
from {{ ref('fct_experiment_metrics') }}
group by experiment_id, user_id
having count(*) > 1
