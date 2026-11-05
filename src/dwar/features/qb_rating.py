"""The quality of the QB a player played with, game by game, point in time.

A game's rating is its team's main passer (most attempts that game) rated by
his passing EPA per attempt over all his earlier regular-season games,
shrunk toward 0 with `SHRINK_ATTEMPTS` average attempts and divided by
`SCALE` (so 1 = 0.1 EPA per attempt). Only games before the rated one are
used, so a rating never sees the game it rates or anything later.
"""

from __future__ import annotations

import polars as pl

SHRINK_ATTEMPTS = 200.0
SCALE = 0.1
# Below this rating the QB-drift term's extra slope applies (about the
# bottom quarter of WR games, 2013-2019).
LOW_KNOT = -0.15
COLUMNS = ("attempts", "passing_epa")


def team_qb_ratings(stats: pl.DataFrame) -> pl.DataFrame:
    """One row per (season, week, team): `qb_rating` of that game's main passer.
    Empty when the stats lack attempts or passing EPA (small test fixtures)."""
    schema = {"season": pl.Int64, "week": pl.Int64, "team": pl.String, "qb_rating": pl.Float64}
    if not set(COLUMNS) <= set(stats.columns):
        return pl.DataFrame(schema=schema)
    games = stats
    if "season_type" in games.columns:
        games = games.filter(pl.col("season_type") == "REG")
    qb = (
        games.filter(pl.col("attempts").fill_null(0) > 0)
        .select("player_id", "season", "week", "team", "attempts", epa=pl.col("passing_epa").fill_null(0.0))
        .sort("player_id", "season", "week")
        .with_columns(
            prior_epa=pl.col("epa").cum_sum().shift(1).fill_null(0.0).over("player_id"),
            prior_att=pl.col("attempts").cum_sum().shift(1).fill_null(0).over("player_id"),
        )
        .with_columns(qb_rating=pl.col("prior_epa") / (pl.col("prior_att") + SHRINK_ATTEMPTS) / SCALE)
    )
    return (
        qb.sort("attempts", "player_id", descending=[True, False])
        .group_by("season", "week", "team", maintain_order=True)
        .first()
        .select(pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64), "team", "qb_rating")
    )
