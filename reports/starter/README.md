# Full-game share for QB backups (2026-11-13)

ben: Justin Fields (Mahomes' backup, about 16 PPG as a starter) ranked QB19-20.
"He should be nerfed incredibly hard by being a backup - his predicted games
played should be super low."

Full game = a game with 15+ plays (config/model.yaml `min_plays`). Starter
share of a season = full games / the season's team games
(`availability.starter_seasons`). Backups who started under 15% of games and
scored 12+ per full game played 2-3 full games the next season; the board
gave Fields about 7.

`uv run python -m dwar.validate.starter` (summary.json): QB games and
survival models trained on earlier seasons only, value layer's design (healthy
non-playing share, draft terms) with and without `starter_terms` (share, and
share x (talent - 15) / 5).

| | 2013-22 | 2023-25 |
|---|---|---|
| Games played (log loss / game) | 0.5548 -> 0.5506 | 0.6168 -> 0.6165 |
| Games played (MAE, games) | 3.24 -> 2.99 | 3.53 -> 3.39 |
| Full games (MAE, games) | 3.13 -> 2.91 | 3.46 -> 3.35 |
| Survival (log loss) | 0.4784 -> 0.4721 | 0.4141 -> 0.4190 |
| Backups 12+ PPG: full games pred (actual) | 3.55 -> 2.45 (2.21) | 3.77 -> 2.47 (3.40) |

Kept in the games model (`starter_games: [QB]`); survival is worse on
2023-25, so left off (`starter_survival: []`).

First board build: Fields only fell 17%. Three more changes, each checked:

- Full games valued directly (`full_games: [QB]`): a second games model,
  same terms, fit to full games among QBs who played; projected games played
  are scaled by the ratio of the two models' means. Before, a backup's
  cameos counted as full games unless his preseason-role short rate said
  otherwise (Fields' was 29%).
- Rest of this season: this season's full-game share so far replaces last
  season's within 4 team games, and the full-game ratio moves toward this
  season's (half weight at 2 games played). `rest_of_season` in summary.json:
  after week 9, backups with no full game so far and 40%+ last season played
  1.1 full games the rest of the way; last season's share said 4.8, so far 1.2.
- Later seasons: chained freely, the share drifted back to a starter's. Now
  it is capped at this season's share of the games he was not hurt for.
  `horizons` in summary.json, full games 2 and 3 seasons out:

| | actual | chained | capped (healthy games) |
|---|---|---|---|
| Backups 14+ PPG (n 41), 2 out | 2.20 | 3.35 | 2.48 |
| Backups, 3 out | 2.37 | 4.19 | 2.45 |
| Starters 50%+ (n 353), 2 out | 9.59 | 10.35 | 10.32 |
| Starters, 3 out | 8.41 | 8.87 | 8.82 |

  Capping at the raw share scored about the same (all QBs MAE 3.41 vs 3.46
  at 2 out) but would hold a starter on IR now (Dart) to a backup's share.

Board (as of 2026-11-09, same fits): Superflex Fields 0.69 -> 0.35 (QB19 ->
QB28), Hurts 0.68 -> 1.31 (QB20 -> QB14), Lamar 0.99 -> 1.45 (QB17 -> QB11),
Dart 2.16 -> 1.62 (QB6 -> QB7; on IR since week 3). Other positions move by
Monte Carlo noise only (at most 0.17). Odd one: Taysom Hill 0.11 -> 0.29
(QB34), who had a near-certain short-game rate before.

Also fixed: next season's QB role came from the latest depth chart, which
lists a hurt starter's backup as QB1 (Nov 9: Joe Milton for PHI, Jacoby
Brissett for BAL), so Hurts and Lamar were valued as backups for five
seasons, with 58% and 51% short-game chances. A player who was QB1 this
preseason and is out injured now keeps the role (`dwar._next_roles`).
