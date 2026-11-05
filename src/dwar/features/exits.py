"""Injury exits: games a player left early hurt.

A game is an injury exit when the player's share of his team's offense
snaps is under `SHARE` of his usual (the median of his previous
`LOOKBACK` games with snaps, at least `MIN_PRIOR` of them) and he then misses his team's next
game that season. Only games already in the tables are used, so a game
whose team has not played its next game yet is never flagged (point in
time). Seasons without snap counts (before 2012) have no flags.
"""

from __future__ import annotations

import polars as pl

SHARE = 0.6
LOOKBACK = 8
MIN_PRIOR = 3


def flag_injury_exits(weeks: pl.DataFrame) -> pl.DataFrame:
    """`weeks` with a boolean `injury_exit` (needs player_id, season, week,
    team, offense_pct; null shares never flag; no flags without the column)."""
    if "offense_pct" not in weeks.columns:
        return weeks.with_columns(injury_exit=pl.lit(False))
    snaps = pl.col("offense_pct").cast(pl.Float64)
    w = weeks.with_row_index("_r").sort("player_id", "season", "week")
    # Usual snaps: median of the previous games with snaps.
    hist = (
        w.filter(snaps > 0).select("_r", "player_id", s=snaps)
        .with_columns(usual=pl.col("s").shift(1).rolling_median(LOOKBACK, min_samples=MIN_PRIOR).over("player_id"))
        .select("_r", "usual")
    )
    w = w.join(hist, on="_r", how="left")
    team_weeks = weeks.select("team", "season", "week").unique().sort("team", "season", "week") \
        .with_columns(next_week=pl.col("week").shift(-1).over("team", "season"))
    played = weeks.select("player_id", "season", next_week="week").unique().with_columns(played_next=pl.lit(True))
    w = (
        w.join(team_weeks, on=["team", "season", "week"], how="left")
        .join(played, on=["player_id", "season", "next_week"], how="left")
        .with_columns(
            injury_exit=(snaps < SHARE * pl.col("usual")).fill_null(False)
            & pl.col("next_week").is_not_null() & pl.col("played_next").is_null()
        )
    )
    return w.sort("_r").drop("_r", "usual", "next_week", "played_next")
