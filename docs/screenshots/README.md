# Screenshots

The top-level [README.md](../../README.md) references images from this directory. The Streamlit page renders one experiment at a time, so each capture is one scenario.

## How to capture

```bash
# 1. Load a scenario into the warehouse
uv run python -m src.simulator.scenarios <scenario>

# 2. Rebuild the dbt marts
cd dbt_project && uv run dbt build --profiles-dir . && cd ..

# 3. Launch the UI
uv run streamlit run src/ui/app.py

# 4. Screenshot the page and save it here
```

## Expected files

The README references these paths; capture them in order:

| File | Scenario | What it should show |
|---|---|---|
| `review_page.png` | `clean_lift` | The hero shot at the top of the README — full review page, all health checks green, Ship recommendation |
| `srm_bug.png` | `srm_bug` | SRM check failing red; recommendation = Don't ship |
| `novelty_effect.png` | `novelty_effect` | Novelty check failing; the daily conversion-rate chart showing the decay; recommendation = Iterate |
| `simpsons.png` | `simpsons` | Segment-breakdown table with treatment winning in every segment, Simpson's check failing, recommendation = Don't ship |
| `guardrail_violation.png` | `guardrail_violation` | Primary metric green; latency guardrail red; recommendation = Don't ship |
| `heterogeneous.png` | `heterogeneous` | The HTE block: per-segment CATE table showing power_user >> casual ≈ enterprise |

To keep file sizes reasonable, export at 1600 px wide. The default Streamlit dark theme works well against the colored status pills.
