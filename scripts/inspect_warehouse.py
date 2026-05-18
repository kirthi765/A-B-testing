"""Quick inspection of data/warehouse.duckdb after running a scenario.

Run with:  uv run python scripts/inspect_warehouse.py
"""

from __future__ import annotations

from pathlib import Path

import duckdb


DB = Path("data") / "warehouse.duckdb"


def main() -> None:
    con = duckdb.connect(str(DB), read_only=True)
    try:
        print("== tables ==")
        print(con.execute("SHOW TABLES").df().to_string(index=False))

        print("\n== row counts ==")
        for t in ("users", "events", "exposures"):
            n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"  {t}: {n:,}")

        print("\n== exposure split ==")
        print(
            con.execute(
                "SELECT variant, COUNT(*) AS n FROM exposures GROUP BY variant ORDER BY variant"
            ).df().to_string(index=False)
        )

        print("\n== per-user conversion rate by variant ==")
        print(
            con.execute(
                """
                WITH per_user AS (
                    SELECT
                        e.user_id,
                        x.variant,
                        SUM(CASE WHEN event_type = 'session'    THEN 1 ELSE 0 END) AS n_sess,
                        SUM(CASE WHEN event_type = 'conversion' THEN 1 ELSE 0 END) AS n_conv
                    FROM events e
                    JOIN exposures x USING (user_id)
                    GROUP BY 1, 2
                )
                SELECT
                    variant,
                    COUNT(*)              AS users,
                    SUM(n_sess)           AS sessions,
                    SUM(n_conv)           AS conversions,
                    SUM(n_conv)::DOUBLE / NULLIF(SUM(n_sess), 0) AS conv_rate_per_session,
                    AVG(n_conv::DOUBLE / NULLIF(n_sess, 0))      AS conv_rate_per_user_mean
                FROM per_user
                GROUP BY variant
                ORDER BY variant
                """
            ).df().to_string(index=False)
        )

        print("\n== conversion rate by segment x variant ==")
        print(
            con.execute(
                """
                SELECT
                    u.segment,
                    x.variant,
                    SUM(CASE WHEN event_type = 'conversion' THEN 1 ELSE 0 END)::DOUBLE /
                        NULLIF(SUM(CASE WHEN event_type = 'session' THEN 1 ELSE 0 END), 0)
                        AS conv_rate
                FROM events e
                JOIN exposures x USING (user_id)
                JOIN users u     USING (user_id)
                GROUP BY u.segment, x.variant
                ORDER BY u.segment, x.variant
                """
            ).df().to_string(index=False)
        )

        print("\n== latency summary by variant (session rows only) ==")
        print(
            con.execute(
                """
                SELECT
                    x.variant,
                    AVG(value)                    AS mean_ms,
                    quantile_cont(value, 0.5)     AS p50_ms,
                    quantile_cont(value, 0.95)    AS p95_ms
                FROM events e
                JOIN exposures x USING (user_id)
                WHERE event_type = 'session'
                GROUP BY x.variant
                ORDER BY x.variant
                """
            ).df().to_string(index=False)
        )
    finally:
        con.close()


if __name__ == "__main__":
    main()
