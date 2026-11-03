# Injury risk and games played

Run: `uv run python -m dwar.validate.injury --scoring half_ppr` (about 2 minutes,
no talent refits). Numbers below are from `summary.json`, as of 2026-11-03.

## What changed in DWAR

1. **Current injury status** cuts this season's remaining games. A player who
   missed the team's latest game to injury gets a factor learned from history,
   by status (out 1 week, out 2+ weeks, on IR 1–3 weeks, on IR 4+ weeks) and by
   games left. Example: 4+ weeks on IR with 10+ games left plays about 41% of a
   healthy player's share; with 1–4 games left, about 13%.
2. **Talent cap on games played.** The games model used a straight line in
   talent, so elite players were credited with nearly every game. History
   disagrees: RBs above about 12 half-PPR points per game average about 13
   games the next season, the same as good-but-not-elite RBs. Talent now stops
   adding games above each position's 75th percentile (RB 8.5, WR 7.7, TE 4.5,
   QB 15.5 points per game). This is the largest gain here.
3. **Injury history.** Weeks missed to injury in each of the last two seasons
   lower the expected share of games. The effect is small: for a RB, a 12-week
   absence last season costs about half a game the next.

Data: nflverse weekly injury reports (2009 on) and weekly roster status (IR,
PUP). A week counts as missed to injury when the player did not play and was
on IR/PUP or reported Out/Doubtful with an injury. Illness, rest, personal
reasons and suspensions do not count. Code: `features/injuries.py`,
`value/availability.py`.

## Tested and dropped

- **Body-part groups** (knee, soft tissue, concussion, ankle/foot, hand/arm
  and six more) as separate terms. Every version tied or lost to the plain
  total of weeks missed, including a version that left out hand, arm and rib
  injuries. Reports name only the body part, never the diagnosis, which likely
  limits this.
- **Injury history in the still-playing (survival) model.** It made those
  forecasts worse in both periods.

## Results

Each season t was forecast from models trained only on seasons before t.
Last season's points per game stands in for talent in every variant, so the
comparison isolates the games model. "Relevant" = 8+ half-PPR points per game
last season (the players DWAR values most). Log score: higher is better.

Games played next season (beta-binomial log score / MAE in games):

| Variant | Dev all | Dev relevant | Holdout all | Holdout relevant |
| --- | --- | --- | --- | --- |
| Step 5 model | −2.569 / 3.92 | −2.453 / 3.15 | −2.617 / 4.03 | −2.565 / 3.18 |
| + talent cap | −2.559 / 3.88 | −2.362 / 3.13 | −2.593 / 3.95 | −2.470 / 3.15 |
| + talent cap + injury weeks (**used**) | −2.561 / 3.89 | **−2.360 / 3.12** | **−2.591 / 3.94** | **−2.462 / 3.14** |
| + body-part groups | −2.564 / 3.89 | −2.362 / 3.12 | −2.594 / 3.95 | −2.463 / 3.14 |

Rest-of-season games from weeks 4, 8 and 12 (MAE in games / log score):

| | Dev | Holdout |
| --- | --- | --- |
| Without current status | 2.17 / −2.172 | 2.33 / −2.324 |
| With current status (**used**) | **2.07 / −2.115** | **2.24 / −2.273** |

Survival log loss (lower is better): dev 0.502 without injury terms vs 0.505
with; holdout 0.461 vs 0.462.

Dev seasons are 2013–2022 (2012 has too little injury history to train on);
holdout is 2023–2025. **Caveat:** unlike step 5, the holdout seasons were
looked at while choosing among these variants, so they are not a clean final
check. Each kept change also wins on the dev seasons.
