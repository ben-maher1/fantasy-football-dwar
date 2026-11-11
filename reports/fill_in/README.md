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
