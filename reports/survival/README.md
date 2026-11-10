# Career survival check (2026-11-10)

`uv run python -m dwar.validate.survival --scoring half_ppr` writes `summary.json`.

Survival = does a player who played in season t - 1 play in t. Each season t is
forecast from models trained on earlier seasons only, with last season's PPG
standing in for talent.

| Log loss (lower is better) | recent (production window) | long (1999 on) | long + decline past 30 | long + decline past 28 |
|---|---|---|---|---|
| 2013-22, all | 0.524 | 0.500 | 0.501 | 0.500 |
| 2013-22, 8+ PPG | 0.322 | 0.272 | 0.272 | 0.272 |
| 2013-22, 8+ PPG, age 29+ | 0.383 | 0.348 | 0.350 | 0.349 |
| 2023-25, all | 0.461 | 0.460 | 0.459 | 0.459 |
| 2023-25, 8+ PPG | 0.154 | 0.151 | 0.152 | 0.151 |

Kept: `long` (config/value.yaml `survival_first_season: 1999`). The age hinges
add nothing. A 30-year-old star RB's chance of playing at 34 went from 62% to
53% on the Nov 10 board (history for RBs with 12+ PPG at 30: 22%; most of the
remaining gap is how fast such players' scoring falls, not survival itself).
