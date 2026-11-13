# Few-game players vs draft capital (2026-11-12)

ben: a few games still move players past draft capital too much (Theo Wease Jr.,
UDFA, 5 games, ahead of Chris Bell, round 3, 8 games).

History 1999-2022 (young players, next-3-season points per team game, zeros
for seasons not played): with 6 or fewer career games each early PPG point
adds 0.08 (WR) / 0.12 (RB) / 0.05 (TE); with 7-16 games 0.56 / 0.41 / 0.52.

Screen, 2020-22 step-through (validate/career.py), each vs the current model
(WR young5, RB rb_usage), discounted per-season MAE / log score gain:

| Variant | WR | RB |
|---|---|---|
| early_noise: first games get extra weekly noise (lam_noise) | +0.004 / +0.000 | +0.001 / -0.001 (fit R-hat 1.0108) |
| pick_sd: rookie prior spread by draft capital | -0.007 / -0.003 | -0.001 / +0.001 |
| no young steps | -0.017 / -0.005 | -0.031 / -0.011 (fit failed) |

All tie or worse; none kept (switches `early_noise`, `pick_sd` off). Not run on
2017-19 / 2023-25.

Why: among players who keep playing, the current WR model's forecasts for
few-game players are already right-sized. Regressing next 1-2 seasons' PPG on
the forecast within draft tier gives slope 1.03 +- 0.05 at 6 or fewer career
games (1.02 at 7-16, 0.99 at 17-40). The weak historical signal comes from
playing time: few-game players often never get another chance, which the
games and survival models carry (survival for few-game young players is close
to calibrated by draft tier, validate/inactive.py rows).

Separate finding: RB forecasts are too spread at every career stage (same
slope 0.58 / 0.66 / 0.70), so differences between RBs' forecasts run about
30-40% too large.
