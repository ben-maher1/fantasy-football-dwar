# Fill-in games for RBs (2026-11-11): tested, not kept

ben: a backup's big game when the starter sits (Monangai, 21 points in week 9
of 2025 with Swift out) says less about his talent than his other games.

**Flag** (`features/fill_in.py`): an RB's game in which his team's lead back
(40%+ of the team's RB carries over at least two earlier games that season)
did not play. Only earlier games are used, so it is known at kickoff.
3,951 RB games flagged in 2002-2026.

**History check** (player seasons with 6+ games, 1+ fill-in game and 3+ other
games; next season 4+ games). Backups scored 7.2 half-PPR PPG in fill-in games
vs 4.2 in their other games. Regressing next-season PPG on the season's points
per game, split by game type:

| Seasons | n | per point, other games | per point, fill-in games |
|---|---|---|---|
| 2002-25 | 912 | 0.83 | 0.46 |
| 2002-13 | 430 | 0.82 | 0.26 |
| 2014-25 | 482 | 0.83 | 0.70 |

**Model test**: `rb_fill` = `rb_usage` + a third usage column (1 in fill-in
games) shifting that game's latent scoring (`fill_in` in ModelConfig;
forecasts assume the lead back plays). All three fits passed. The shift is
clearly there (gamma 0.64-0.65 on the square-root scale, sd 0.04), but
forecasts did not improve:

| Period | discounted MAE gain | discounted log score gain | next season MAE gain |
|---|---|---|---|
| 2017-19 (p17) | -0.003 | -0.008 | +0.011 |
| 2020-22 (dev) | +0.000 | -0.005 | +0.026 |
| 2023-25 (holdout) | -0.008 | +0.000 | +0.006 |

Next-season forecasts got slightly better in all three periods, but the
current season and later seasons slightly worse, so it is a tie overall. The
RB model is fit from 2013, where the history check shows the smaller gap.
Not kept: `fill_in` stays empty in config/model.yaml; the flag and the
variant stay for later use.

## Retest on five-season value (2026-11-25, ben: Emmett Johnson)

ben asked whether Emmett Johnson (KC, pick 161) is overrated: 6.7 PPG in the
8 games Kenneth Walker played, then 32.1 and 22.3 with Walker out. The usage
term only sees earlier games' carry share, so the first fill-in game is read
as talent, and the rest of the season holds his fill-in workload.

Retest scored on five-season value (rest of season plus 5 seasons at 0.85,
`horizon -1`), overall and for backups with fill-in games earlier that season.
`rb_fill_hold` = rb_fill's fit, but rest-of-season forecasts hold carry share
from games the lead back played only (`usage_design_hold`). Gain = rb_usage
error minus variant error (positive = variant better); bias = forecast minus
actual for rb_usage -> variant.

| Period | Variant | All: MAE gain / log gain | 1+ fill-in games: MAE gain / log gain | Fill-in backups' bias |
|---|---|---|---|---|
| 2014-16 | rb_fill | -1.18 / -0.026 | -0.68 / -0.030 | -13.5 -> -16.6 |
| 2014-16 | rb_fill_hold | -1.25 / -0.025 | -1.00 / -0.031 | -13.5 -> -19.8 |
| 2017-19 | rb_fill | -0.31 / -0.010 | -0.85 / -0.020 | -5.1 -> -12.4 |
| 2017-19 | rb_fill_hold | -0.27 / -0.014 | -1.24 / -0.039 | -5.1 -> -15.9 |
| 2020 | rb_fill | +0.07 / -0.018 | +1.04 / -0.014 | +5.2 -> -1.6 |
| 2020 | rb_fill_hold | +0.29 / -0.020 | +1.48 / -0.025 | +5.2 -> -4.6 |

Log score is worse for both in every period, and the 2014-16 and 2017-19
periods are worse on MAE too. The current model does not overrate backups
with fill-in games over five seasons: it slightly underrated them in 2014-16
and 2017-19. Not kept. Caveat: the 2014-16 fits (both) and the 2017-19 rb_fill
fit missed sampler checks (form terms R-hat 1.03-1.5); rb_usage 2017-19 and
both 2020-22 fits passed.
