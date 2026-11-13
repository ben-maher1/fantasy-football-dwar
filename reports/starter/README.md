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
2023-25, so left off (`starter_survival: []`). In the projection, last
season's share is actual; this season's is full games so far plus projected
games played times one minus the short-game chance; later seasons' the same.

Also fixed: next season's QB role came from the latest depth chart, which
lists a hurt starter's backup as QB1 (Nov 9: Joe Milton for PHI, Jacoby
Brissett for BAL), so Hurts and Lamar were valued as backups for five
seasons, with 58% and 51% short-game chances. A player who was QB1 this
preseason and is out injured now keeps the role (`dwar._next_roles`).
