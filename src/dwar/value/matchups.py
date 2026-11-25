"""NFL matchups and bye weeks for the rest of the current season.

The league title-odds simulator (value/league_odds.py) can scale each
player's expected points in each remaining NFL week by his matchup, and
take his bye week away. Only the current season: later schedules are not
known, so five-season DWAR is unchanged.

Matchup test (ben, 2026-11-23; /mnt/project-files/dwar-wip/matchups, doc
"Plan: do matchups predict weekly scoring?"): learned on 2013-19 weeks,
scored on 2020-25. With the opponent's rating frozen mid-season (as it is
for the rest of a season), QBs passed clearly (log score +0.011 +- 0.004 a
week) and RBs passed for fantasy starters (+0.007 +- 0.003); WR and TE did
not, so they are left at 1. ben chose byes + QB/RB matchups for the league
sim only (2026-11-23).

  expectation  a player's pre-game expected points: a weighted average of
               his earlier games (recent games and this season count more),
               shrunk toward his position's average
  rating       a defense's points allowed to a position this season over
               those players' expectations (ratio - 1), shrunk toward half of
               last season's, then centered within the position (only
               relative information; a league-wide level shift is not a
               matchup)
  multiplier   1 + dvp * rating(opponent, position) + home * (+0.5 home,
               -0.5 away, 0 neutral); 0 on a bye
"""

from __future__ import annotations

import numpy as np
import polars as pl

DECAY = 0.92           # weight per game back
PRIOR_SEASON_W = 0.75  # weight per season back
SHRINK_GAMES = 4.0     # pseudo-games of the position average in a player's expectation
DEFENSE_GAMES = 4.0    # pseudo-games of last season's rating in a defense's rating


def expectations(weeks: pl.DataFrame, pos_mean: dict[str, float]) -> pl.DataFrame:
    """`weeks` (player_id, position, season, week, fpts; one row per game)
    with `expected`: points expected before each game from his earlier
    games only, and the same after his last game as `next_expected` (a
    separate frame: player_id, position, next_expected)."""
    w = weeks.sort("player_id", "season", "week")
    pid, y, ssn = w["player_id"].to_numpy(), w["fpts"].to_numpy().astype(float), w["season"].to_numpy()
    m0 = np.array([pos_mean.get(p, 0.0) for p in w["position"].to_list()])
    exp = np.empty(len(w))
    bounds = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1], True]) if len(w) else np.array([0])
    for a, b in zip(bounds[:-1], bounds[1:]):
        for i in range(a, b):
            j = np.arange(a, i)
            wt = DECAY ** (i - 1 - j) * PRIOR_SEASON_W ** (ssn[i] - ssn[j])
            exp[i] = (np.sum(wt * y[j]) + SHRINK_GAMES * m0[i]) / (np.sum(wt) + SHRINK_GAMES)
    return w.with_columns(expected=pl.Series(exp, dtype=pl.Float64))


def defense_ratings(weeks: pl.DataFrame, season: int) -> pl.DataFrame:
    """(opponent_team, position, rating) for `season` from the games in
    `weeks` (with `expected`; the caller keeps only games known as of the
    run date), centered within position."""
    per = weeks.group_by("opponent_team", "position", "season").agg(
        Y=pl.col("fpts").sum(), EX=pl.col("expected").sum(), n=pl.col("game_id").n_unique())
    cur = per.filter(pl.col("season") == season)
    last = (per.filter(pl.col("season") == season - 1)
            .select("opponent_team", "position", last=pl.col("Y") / pl.col("EX") - 1))
    r = (cur.join(last, on=["opponent_team", "position"], how="full", coalesce=True)
         .with_columns(n=pl.col("n").fill_null(0).cast(pl.Float64),
                       now=pl.when(pl.col("EX") > 0).then(pl.col("Y") / pl.col("EX") - 1).otherwise(0.0).fill_null(0.0),
                       prior=0.5 * pl.col("last").fill_null(0.0))
         .with_columns(rating=(pl.col("n") * pl.col("now") + DEFENSE_GAMES * pl.col("prior")) / (pl.col("n") + DEFENSE_GAMES)))
    return (r.with_columns(rating=pl.col("rating") - pl.col("rating").mean().over("position"))
            .select("opponent_team", "position", "rating"))


def team_weeks(schedules: pl.DataFrame, season: int, weeks: list[int]) -> pl.DataFrame:
    """(team, week, opponent, home) for every regular-season game in those
    weeks; a team with no row in a week is on its bye."""
    s = schedules.filter((pl.col("season") == season) & pl.col("week").is_in(weeks))
    if "game_type" in s.columns:            # as-of snapshots hold regular-season games only
        s = s.filter(pl.col("game_type") == "REG")
    neutral = (pl.col("location") == "Neutral").fill_null(False) if "location" in s.columns else pl.lit(False)
    return pl.concat([
        s.select("week", team=pl.col("home_team"), opponent=pl.col("away_team"),
                 home=pl.when(neutral).then(0.0).otherwise(0.5)),
        s.select("week", team=pl.col("away_team"), opponent=pl.col("home_team"),
                 home=pl.when(neutral).then(0.0).otherwise(-0.5)),
    ]).with_columns(pl.col("week").cast(pl.Int64))


def multipliers(teams: list[str | None], positions: list[str], games: pl.DataFrame, ratings: pl.DataFrame,
                effects: dict[str, dict[str, float]], weeks: list[int], byes: bool) -> np.ndarray:
    """(players, weeks) factor on each player's expected points each week.
    Positions without `effects` keep 1 in game weeks; a bye is 0 when
    `byes`; a player with no known team stays at 1."""
    P = len(teams)
    out = np.ones((P, len(weeks)))
    col = {w: k for k, w in enumerate(weeks)}
    rate = {(o, p): r for o, p, r in ratings.iter_rows()}
    g = {(t, int(w)): (o, h) for w, t, o, h in games.select("week", "team", "opponent", "home").iter_rows()}
    scheduled = {t for t, _ in g}
    for i, (t, p) in enumerate(zip(teams, positions)):
        if t is None or t not in scheduled:   # unknown team (or one missing from the schedule): no byes, no matchups
            continue
        e = effects.get(p)
        for w in weeks:
            hit = g.get((t, w))
            if hit is None:
                if byes:
                    out[i, col[w]] = 0.0
                continue
            if e:
                o, h = hit
                out[i, col[w]] = max(1.0 + e.get("dvp", 0.0) * rate.get((o, p), 0.0) + e.get("home", 0.0) * h, 0.05)
    return out
