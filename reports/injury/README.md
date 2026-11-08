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
   adding games above each position's 75th percentile (RB 9.5, WR 8.6, TE 5.2,
   QB 16.5 points per game).
3. **Games played are learned from last season's talent.** The games model
   used to pair each season's games with talent estimated from that same
   season. Talent is partly measured from those very games, so a player who
   stayed healthy looked both good and durable, and good players were
   forecast about 1 game too many per season (1.1 too many on dev, 1.0 on
   holdout). Pairing each season's games with the season before, which is
   what a forecast actually knows, cuts that to about 0.3. This and the
   talent cap are the largest gains here.
4. **Injury history.** Weeks missed to injury in each of the last two seasons
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

Games played next season (beta-binomial log score / MAE in games / bias,
the average forecast minus actual games):

| Variant | Dev all | Dev relevant | Holdout all | Holdout relevant |
| --- | --- | --- | --- | --- |
| Step 5 model | −2.569 / 3.92 / −0.2 | −2.453 / 3.15 / +1.3 | −2.617 / 4.03 / −0.2 | −2.565 / 3.18 / +1.2 |
| + talent cap | −2.559 / 3.88 / −0.2 | −2.362 / 3.13 / +1.1 | −2.593 / 3.95 / −0.2 | −2.470 / 3.15 / +1.0 |
| + injury weeks | −2.561 / 3.89 / −0.2 | −2.360 / 3.12 / +1.1 | −2.591 / 3.94 / −0.2 | −2.462 / 3.14 / +1.0 |
| + body-part groups | −2.564 / 3.89 / −0.2 | −2.362 / 3.12 / +1.1 | −2.594 / 3.95 / −0.2 | −2.463 / 3.14 / +1.0 |
| cap + injury weeks + last season's talent (**used**) | **−2.500** / 3.89 / **−0.1** | **−2.301** / 3.29 / **+0.3** | **−2.549** / 3.99 / **−0.1** | **−2.419** / 3.29 / **+0.2** |

The used model has a worse MAE for relevant players (3.29 vs 3.12). MAE
rewards forecasting the typical outcome, a near-full season, and the
earlier models scored well on it partly by forecasting too many games.
DWAR adds up expected wins, so the average forecast must be right, and the
log score (which scores the whole spread of outcomes) agrees.

Rest-of-season games from weeks 4, 8 and 12 (MAE in games / log score):

| | Dev | Holdout |
| --- | --- | --- |
| Without current status | 2.11 / −2.201 | 2.27 / −2.361 |
| With current status (**used**) | **2.02 / −2.156** | **2.19 / −2.320** |

Survival log loss (lower is better): dev 0.502 without injury terms vs 0.505
with; holdout 0.461 vs 0.462.

Dev seasons are 2013–2022 (2012 has too little injury history to train on);
holdout is 2023–2025. **Caveat:** unlike step 5, the holdout seasons were
looked at while choosing among these variants, so they are not a clean final
check. Each kept change also wins on the dev seasons.
