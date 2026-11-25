"""Counts of 40+ yard plays per player-game, from play-by-play, for scoring
formats with long-play bonuses (Sleeper's *_40p settings)."""

from __future__ import annotations

import polars as pl

from dwar.features.scoring import LONG_PLAY_COLUMNS

LONG = 40


def long_play_counts(pbp: pl.DataFrame) -> pl.DataFrame:
    """One row per (game_id, player_id) with the LONG_PLAY_COLUMNS counts.

    Pure. Two-point tries are not plays from scrimmage and are left out. A
    completion credits the passer and the receiver; a run credits the rusher
    (scrambles included, as nflverse counts them as carries).
    """
    base = pbp.filter(
        (pl.col("two_point_attempt").fill_null(0) == 0) & (pl.col("yards_gained").fill_null(0) >= LONG)
    )
    catch = base.filter(pl.col("complete_pass").fill_null(0) == 1)
    td_pass = pl.col("pass_touchdown").fill_null(0) == 1
    passer = catch.filter(pl.col("passer_player_id").is_not_null()).group_by(
        "game_id", player_id="passer_player_id"
    ).agg(passing_cmp_40p=pl.len(), passing_td_40p=td_pass.sum())
    receiver = catch.filter(pl.col("receiver_player_id").is_not_null()).group_by(
        "game_id", player_id="receiver_player_id"
    ).agg(receiving_40p=pl.len(), receiving_td_40p=td_pass.sum())
    rusher = base.filter(
        (pl.col("rush_attempt").fill_null(0) == 1) & pl.col("rusher_player_id").is_not_null()
    ).group_by("game_id", player_id="rusher_player_id").agg(
        rushing_40p=pl.len(), rushing_td_40p=(pl.col("rush_touchdown").fill_null(0) == 1).sum()
    )
    out = pl.concat([passer, receiver, rusher], how="diagonal_relaxed")
    return out.group_by("game_id", "player_id").agg(
        pl.col(c).fill_null(0).sum().cast(pl.Int64) for c in LONG_PLAY_COLUMNS
    )


def with_long_plays(player_stats: pl.DataFrame, pbp: pl.DataFrame) -> pl.DataFrame:
    """`player_stats` plus LONG_PLAY_COLUMNS (zero when the player had none)."""
    counts = long_play_counts(pbp)
    return player_stats.drop(LONG_PLAY_COLUMNS, strict=False).join(
        counts, on=["game_id", "player_id"], how="left"
    ).with_columns(pl.col(c).fill_null(0) for c in LONG_PLAY_COLUMNS)
