"""The quality of the QB a player played with, game by game, point in time.

A game's rating is its team's main passer (most attempts that game) rated by
his passing EPA per attempt over all his earlier regular-season games,
shrunk toward 0 with `SHRINK_ATTEMPTS` average attempts and divided by
`SCALE` (so 1 = 0.1 EPA per attempt). Only games before the rated one are
used, so a rating never sees the game it rates or anything later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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


@dataclass(frozen=True)
class QBFade:
    """How a player's QB rating carries into later seasons: `keep[k-1]` is
    the slope of the rating k seasons later on today's, and `mean` the
    rating they fade toward (QBs and teams change, so it is far from 1)."""
    mean: float
    keep: tuple[float, ...]

    def expected(self, q, offseason: int):
        """Expected rating behind the drift at the `offseason`-th projected
        offseason (1 = the next one, which uses the current rating as is)."""
        keep = np.array((1.0, *self.keep))
        w = keep[np.clip(np.asarray(offseason) - 1, 0, len(self.keep))]
        return self.mean + w * (np.asarray(q) - self.mean)


def qb_fade(player: np.ndarray, season: np.ndarray, q: np.ndarray, max_ahead: int = 3) -> QBFade:
    """Fit QBFade from player-seasons' end-of-season ratings (point in time:
    pass only fitted seasons). A slope needs 30 pairs, otherwise 0."""
    df = pl.DataFrame({"p": player, "s": season, "q": q})
    keep = []
    for k in range(1, max_ahead + 1):
        j = df.join(df.with_columns(s=pl.col("s") - k), on=["p", "s"], suffix="_k")
        x, y = j["q"].to_numpy(), j["q_k"].to_numpy()
        keep.append(float(np.polyfit(x, y, 1)[0]) if len(x) >= 30 and x.std() > 0 else 0.0)
    return QBFade(mean=float(df["q"].mean()), keep=tuple(keep))
