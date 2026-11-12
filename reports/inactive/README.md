# Healthy non-playing weeks in games played and survival (2026-11-12)

ben: Theo Wease Jr. (undrafted, 5 career games) ranked over Chris Bell (2026
round 3) partly because a thin record leaves room for upside. But Wease's
missed games were not injuries: he sat healthy, on the practice squad or off a
roster. That should count against him.

Healthy non-playing share of a season = team games the player neither played
nor missed to injury (reported Out/Doubtful, or on a reserve list), over the
season's team games (`availability.inactive_seasons`; injury reports start in
2009). Wease 2025: 3 games, 14 healthy non-playing weeks (82%).

`uv run python -m dwar.validate.inactive` (summary.json): the value layer's
survival and games designs, trained on earlier seasons only, with and without
last season's share (plus a flag for seasons before 2009 in survival rows).

| | 2013-22 | 2023-25 |
|---|---|---|
| Survival, all (log loss) | 0.5006 -> 0.4702 | 0.4584 -> 0.4242 |
| Survival, 8+ PPG | 0.2711 -> 0.2648 | 0.1498 -> 0.1381 |
| Games, all (log loss / game) | 0.5850 -> 0.5701 | 0.5806 -> 0.5624 |
| Games, all (MAE, games) | 3.89 -> 3.68 | 3.98 -> 3.70 |

Better at every position in both periods (TE 8+ PPG holdout, n = 33, ties).
Survival by share, 2013-22: under 10% plays next season 88% (base said 81%,
new 88%); 60%+ plays 48% (base 61%, new 46%).

Config: `inactive_survival` and `inactive_games`: [QB, RB, WR, TE]. In the
projection, last season's share is actual; the current season's is weeks so
far plus projected rest-of-season games missed not to injury; later seasons'
are projected the same way.

Board (as of 2026-11-09, ben's league tab): total DWAR 453 -> 452; with the
same seed and code, Wease 0.46 -> 0.42 (2027 survival 0.80 -> 0.74), Bell 0.33
either way (2027 survival 0.79 -> 0.88). Wease stays ahead (WR80 vs WR88):
his edge is a higher talent estimate (7.6 PPG in 3 games in 2025), and the two
are within Monte Carlo noise of each other.
