"""A receiver's own catch efficiency, game by game, point in time.

Plan "game-by-game TE model with usage and efficiency" (ben, 2026-11-14),
step 3: TEs with many targets but few yards per target were over-forecast
about 1 PPG the next season. This rating stands in for the QB rating in the
offseason drift term (`qb_drift`), so the same machinery (fit, step-through,
projection, fade into later seasons) carries it.

A game's rating is the player's season-to-date yards per target through
that game, shrunk toward the position's yards per target over earlier
seasons with `SHRINK_TARGETS` targets, minus that prior-season average, in
units of `SCALE` yards. Only that season's games up to the rated one and
earlier seasons are used, so it never sees a later game.
"""

from __future__ import annotations

import polars as pl

SHRINK_TARGETS = 20.0
SCALE = 1.4            # yards per target; about one SD between TE seasons with 30+ targets
DEFAULT_YPT = 7.4      # used before any earlier season is available


def efficiency_ratings(weeks: pl.DataFrame, position: str) -> pl.DataFrame:
    """(player_id, season, week, eff_rating) for `position`'s player-weeks.
    Needs `targets` and `receiving_yards` (nulls count as zero)."""
    w = (
        weeks.filter(pl.col("position") == position)
        .select("player_id", "season", "week",
                tg=pl.col("targets").fill_null(0).cast(pl.Float64),
                yds=pl.col("receiving_yards").fill_null(0).cast(pl.Float64))
        .sort("player_id", "season", "week")
    )
    # Seasons without target data (some early nflverse seasons) are left
    # out of the average and get no rating.
    by_season = w.group_by("season").agg(pl.col("yds").sum(), pl.col("tg").sum()).sort("season")
    has = by_season.filter(pl.col("tg") > 0)
    prior = has.with_columns(
        prior_ypt=(pl.col("yds").cum_sum().shift(1) / pl.col("tg").cum_sum().shift(1)).fill_null(DEFAULT_YPT)
    ).select("season", "prior_ypt")
    return (
        w.join(prior, on="season", how="inner")
        .with_columns(cy=pl.col("yds").cum_sum().over("player_id", "season"),
                      ct=pl.col("tg").cum_sum().over("player_id", "season"))
        .with_columns(eff_rating=((pl.col("cy") + SHRINK_TARGETS * pl.col("prior_ypt")) / (pl.col("ct") + SHRINK_TARGETS)
                                  - pl.col("prior_ypt")) / SCALE)
        .select("player_id", "season", "week", "eff_rating")
    )


def with_efficiency_as_qb(weeks: pl.DataFrame, position: str) -> pl.DataFrame:
    """`weeks` with `position`'s `qb_rating` replaced by its efficiency rating."""
    e = efficiency_ratings(weeks, position)
    out = weeks.join(e, on=["player_id", "season", "week"], how="left")
    return out.with_columns(
        qb_rating=pl.when(pl.col("position") == position).then(pl.col("eff_rating")).otherwise(pl.col("qb_rating"))
    ).drop("eff_rating")
