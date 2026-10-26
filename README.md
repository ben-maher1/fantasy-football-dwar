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
| 1 | nflverse ingest + caching; configurable fantasy scoring | done |
| 2 | Player-season table with as-of filtering + no-lookahead test | done |
| 3 | WR talent model: fit, diagnostics, parameter recovery | done |
| 4a | Talent fits for QB, RB, WR, TE | in review |
| 4b | Replacement level, points to wins, projection, DWAR output | not started |
| 5–7 | Validation, CLI, market backtest | not started |
