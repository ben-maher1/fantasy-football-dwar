# Prospect profiles for young WRs (2026-11-09)

Plan: "Plan: prospect profiles for young WRs" (approved by ben 2026-11-08).
Result: **not kept**. Neither all eight college metrics nor a lean pair beat
the current WR model (`young5`) in any of the four backtest periods. The
code stays in, off by default (`config/model.yaml` has no `prospect` entry).

## Data

- collegefootballdata.com (CFBD), key in `CFBD_API_KEY`, cached in
  `data/raw/cfbd/` (never committed; CFBD's terms bar republishing raw data).
  The one-time pull took about 515 calls (the key's tier allows 3,000 a month).
- CFBD's season-stats endpoint leaves players out (Will Fuller's 2015 at Notre
  Dame is missing; 2004-2011 are thin), so FBS seasons are summed from game
  box scores. FCS teams come from the season endpoint where it has them.
- 93% of WRs drafted since 2008 matched to college stats (name + college +
  draft year). Most misses played below FBS. Recruiting rating found for ~73%.

## Metrics (`features/prospect.py`)

breakout age (20% dominator), best-season dominator, team SP+ in that season,
best yards per team pass attempt, age-adjusted yards per team pass attempt,
final-season reception share, early declare (< 4 seasons with stats),
recruiting rating. Only college seasons before a player's draft are used; the
age reference uses only players drafted by the fit date. Missing = flag.

## Results (variant minus `young5`; positive = better)

Discounted per-season score and the rookie (no NFL games yet) group, with
bootstrap 90% intervals over players. All 16 fits passed the checks.

| Period | All 8: MAE / log score | All 8, rookies: log score | Lean 2: MAE / log score | Lean 2, rookies: log score |
|---|---|---|---|---|
| 2014-16 | -0.006 / -0.000 | +0.010 (-0.014, +0.037) | -0.003 / -0.002 | -0.006 (-0.031, +0.017) |
| 2017-19 | -0.001 / +0.001 | -0.015 (-0.037, +0.005) | -0.003 / +0.001 | +0.003 (-0.014, +0.018) |
| 2020-22 | +0.002 / +0.000 | -0.013 (-0.030, +0.007) | -0.009 / -0.002 | +0.005 (-0.009, +0.018) |
| 2023-25 holdout | +0.002 / +0.001 | +0.009 (-0.019, +0.032) | -0.002 / -0.001 | +0.001 (-0.020, +0.019) |

Lean 2 = early declare + age-adjusted yards per team pass attempt, picked
because the current model's rookie forecast errors lean on them most steadily
(+0.3 to +0.8 PPG per SD in 2014-22). Putting them in the rookie prior
removed about half of that lean, but also raised rookie forecasts on average
and moved veterans slightly, so overall scores tied. Leave-one-out of the
eight was skipped (the plan runs it only if all eight help).

Re-run: `uv run python -m dwar.ingest.cfbd 2025`, then
`uv run python -m dwar.validate.career run prospect <period>` and
`... compare prospect young5 <period>` (periods dev, p14, p17, holdout).
