# Draft capital in games played and survival (2026-11-10)

ben: low-drafted RBs lose work more easily even when they play well. Step-through
RB forecasts (2014-25, career years 2-4, forecast 7+ PPG) ran high for rounds 3-4
(-1.0 PPG), 5-7 (-1.4) and undrafted (-1.0), +0.2 for rounds 1-2. History since
1999 (young RBs 8+ PPG in 8+ games): played next season 99% / 88% / 89% (rounds
1-2 / 5-7 / undrafted), 13.2 / 11.4 / 11.2 games.

`uv run python -m dwar.validate.draft_avail` (summary.json) adds two terms for
career years 1-4 (a flag, and flag x log(pick / 64), undrafted = pick 260) to the
survival and games models, trained on earlier seasons only. Kept where both
periods improved or tied:

| | 2013-22 | 2023-25 |
|---|---|---|
| Survival, all players (log loss) | 0.5003 -> 0.4966 | 0.4599 -> 0.4565 |
| RB survival, relevant | 0.2770 -> 0.2745 | 0.1871 -> 0.1826 |
| RB games, relevant (log loss / game) | 0.5578 -> 0.5579 | 0.4898 -> 0.4856 |
| QB games, relevant | 0.5667 -> 0.5621 | 0.6498 -> 0.6448 |

Config: `draft_survival: [RB]`, `draft_games: [RB, QB]` (WR/TE mixed, left out).

Draft-scaled growth in RB scoring (career.py `rb_growth`: growth steps at career
years 2-3 + first-round rookie step) tied in all three periods (discounted log
+.005 / -.005 / +.001; draft slope near zero). Not kept.
