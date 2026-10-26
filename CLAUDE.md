# DWAR — Dynasty Wins Above Replacement

DWAR values dynasty fantasy football assets (players and rookie picks) as the
expected future fantasy wins they add over a replacement-level player, with full
Bayesian uncertainty. It feeds a weekly content business built on one idea:
**the gap between DWAR (fundamentals) and market values (KeepTradeCut / FantasyCalc
sentiment) is a buy/sell signal.** Everything here serves producing that signal
reliably, weekly, and unattended.

## Non-negotiable design rules

1. **Point-in-time correctness.** Every fit takes an `as_of` date and uses only data
   available on that date. No lookahead, ever: no future stats, rosters, injuries,
   or draft results leaking into past fits. The backtest depends on this, and so does
   the business. Write a test for it.
2. **Uncertainty is a first-class output.** Every value ships with posterior
   intervals (at least 50% and 90%). Never collapse to a point estimate upstream of
   the output layer.
3. **League settings are parameters, not constants.** Scoring (PPR / half / TE
   premium), superflex vs 1QB, team count, and lineup slots all come from config.
   Replacement level is derived from these settings.
4. **Reproducible and unattended.** One CLI command runs the whole thing. Fixed
   random seeds, cached raw data, logged model diagnostics. No notebooks in the
   production path (notebooks are fine in `explore/`).
5. **Fail loudly.** If a fit has divergences, R-hat > 1.01, or low ESS, the run is
   marked failed and nothing is published.

## Data

- Primary source: nflverse via the `nflreadpy` package. This covers weekly player
  stats, snap counts, rosters, depth charts, draft picks, combine, and player IDs.
  Check current package docs before use; don't assume function names.
- Do NOT scrape Pro Football Reference. nflverse covers what we need.
- Market values (KeepTradeCut, FantasyCalc): read each source's terms before
  collecting. Store snapshots with the date collected. Historical market data is the
  hardest input for the backtest, so flag gaps early.
- Raw pulls are cached as parquet in `data/raw/{source}/{date}/`. Derived tables go
  in `data/processed/`. Neither is committed to git.

## Model (v1: keep it simple, then earn complexity)

Observation level: weekly fantasy points for player i in season s, week w, computed
from raw stats under the configured scoring. Model points per game played (handle
missed games separately, below). Weekly points are right-skewed with a point
mass at zero (about a quarter of WR weeks), so the likelihood is a hurdle on the
square-root scale (`sqrt_hurdle`): a zero-week probability that falls with talent,
and otherwise points = Z^2 with Z Student-t (heavy tails for blowup weeks).
`talent` is the implied expected points per game. A symmetric Student-t on points
was the original spec; on real WRs it put talent about 1.3 PPG below observed
scoring and is kept only as a baseline (`likelihood: studentt`).

Latent structure:
- `talent[i, s]`: player-season true scoring rate. Evolves as a random walk across
  seasons, with position-specific innovation scale.
- Age curve: position-specific, a quadratic in age. v1 fits each position separately
  (thousands of player-seasons each pin their curves down, and separate fits are
  faster and fail independently); pooling across positions or a spline can be added
  if step 5 validation shows noisy or misshapen curves. This drives the forward
  projection.
- Prior for players with little NFL data: centered on draft capital (pick number,
  transformed) plus combine/size where available, by position. Rookies and
  second-year players lean on this. Veterans are dominated by their data.
- Usage covariates (snap share, target share, carry share, team pass rate) are a v2
  addition. Add them only if they improve out-of-sample forecasts.

Availability: model games played per season separately (by position and age, from
history) and career survival (the probability a player is still fantasy-relevant in
each future year). This is empirical and can start as a simple hazard model.

From talent to DWAR:
1. Replacement level per position = the scoring rate of the (N+1)th player at that
   position, where N = starters across the league given lineup settings (flex and
   superflex allocated by optimal lineups).
2. Points above replacement (PAR) per week → wins, using the standard deviation of
   weekly team scores in the configured league format (points-per-win conversion).
3. Project forward 5 seasons via posterior predictive draws (age curve +
   random-walk talent + availability + survival). Apply a discount factor per future
   season (configurable, default 0.85) to reflect contention timelines and
   uncertainty.
4. DWAR = discounted sum of expected future wins above replacement. Report the
   posterior distribution, not just the mean.

Rookie draft picks: value = expected DWAR of a player drafted at that slot, using
historical rookie-draft ADP to map pick → typical NFL draft capital and position mix.
This is v2. Note it in output as unavailable in v1.

## Validation (required before anything is published)

- Out-of-sample: fit as of the start of season s, forecast season s ppg. Compare
  against baselines: last-season ppg, a 3-year weighted average, and a simple age
  adjustment. DWAR must beat all of them on MAE / log score to ship.
- Calibration: 50% and 90% intervals should cover about 50% and 90% of outcomes.
  Plot it.
- Backtest the business claim: when DWAR and market value disagree, does market value
  move toward DWAR over the next 2 to 6 months? This needs point-in-time DWAR plus
  historical market snapshots. It is the most important result in the project.

## Repo layout

```
dwar/
  config/          # league formats (yaml)
  src/dwar/
    ingest/        # nflverse + market pulls, caching
    features/      # fantasy points under scoring config, player-season tables
    model/         # PyMC model, fitting, diagnostics
    value/         # replacement level, PAR→wins, projection, DWAR
    validate/      # out-of-sample, calibration, backtest
    cli.py         # `dwar run --as-of YYYY-MM-DD --format superflex_ppr`
  tests/
  explore/         # notebooks, not imported by src
  data/            # gitignored
  outputs/         # dated DWAR tables + diagnostics, gitignored
```

## Stack and conventions

- Python 3.12, managed with `uv`. PyMC + ArviZ. Use the `nutpie` sampler if it
  installs cleanly. polars or pandas for data. pytest for tests.
- Type hints throughout. Small pure functions in `features/` and `value/` with unit
  tests. The model module gets a smoke test on a tiny synthetic dataset where the
  true parameters are known (parameter recovery).
- Each run writes `outputs/{as_of}/{format}/dwar.parquet` with columns: player_id,
  name, position, age, team, dwar_mean, dwar_p05, dwar_p25, dwar_p50, dwar_p75,
  dwar_p95. It also writes `diagnostics.json` (R-hat, ESS, divergences, runtime).

## Build order

1. Ingest + caching for nflverse; a fantasy-points function under configurable
   scoring, with tests against hand-computed examples.
2. Player-season table with as-of filtering + a no-lookahead test.
3. Model v1 on one position (WR) end to end: fit, diagnostics, parameter recovery
   test.
4. Extend to QB/RB/TE; replacement level; PAR→wins; projection; DWAR output.
5. Out-of-sample validation vs baselines + calibration plots.
6. CLI + one-command weekly run.
7. Market data collection + the model-vs-market backtest.

Work one step at a time. After each step: run the tests, show results, and stop for
review before moving on.
