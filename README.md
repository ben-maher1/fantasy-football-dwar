# DWAR: Dynasty Wins Above Replacement

Values dynasty fantasy football assets as expected future fantasy wins above
replacement, with full Bayesian uncertainty. See `CLAUDE.md` for the design
rules, model, and build order.

## Setup

```bash
uv sync
uv run pytest                 # offline unit tests
uv run pytest -m network      # live nflverse checks (downloads data)
```

## A DWAR run

```python
from datetime import date
from dwar.pipeline import run

run(date(2026, 10, 26), "superflex_ppr")  # fits all positions (~1 hour), then values them
```

This writes `outputs/{as_of}/{format}/dwar.parquet` and `diagnostics.json`. If
any fit fails its checks, only `diagnostics.json` is written (marked failed).
The step 6 CLI will wrap this.

## Status

| Step | Description | State |
| --- | --- | --- |
| 1 | nflverse ingest + caching; configurable fantasy scoring | done |
| 2 | Player-season table with as-of filtering + no-lookahead test | done |
| 3 | WR talent model: fit, diagnostics, parameter recovery | done |
| 4a | Talent fits for QB, RB, WR, TE | done |
| 4b | Replacement level, points to wins, projection, DWAR output | in review |
| 5–7 | Validation, CLI, market backtest | not started |
