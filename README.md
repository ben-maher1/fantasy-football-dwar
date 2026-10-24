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

## Status

| Step | Description | State |
| --- | --- | --- |
| 1 | nflverse ingest + caching; configurable fantasy scoring | in review |
| 2 | Player-season table with as-of filtering + no-lookahead test | not started |
| 3–7 | Model, valuation, validation, CLI, market backtest | not started |
