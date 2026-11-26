"""Fantasy points from nflverse weekly player stats under a scoring config."""

from __future__ import annotations

import polars as pl

from dwar.config import ScoringConfig

# ScoringConfig field -> nflverse `load_player_stats` columns it multiplies.
# Fumbles lost sum the three sources so the total matches nflverse's own
# `fantasy_points` / `fantasy_points_ppr` exactly.
STAT_COLUMNS: dict[str, tuple[str, ...]] = {
    "pass_yd": ("passing_yards",),
    "pass_td": ("passing_tds",),
    "pass_int": ("passing_interceptions",),
    "pass_2pt": ("passing_2pt_conversions",),
    "rush_yd": ("rushing_yards",),
    "rush_td": ("rushing_tds",),
    "rush_2pt": ("rushing_2pt_conversions",),
    "rec": ("receptions",),
    "rec_yd": ("receiving_yards",),
    "rec_td": ("receiving_tds",),
    "rec_2pt": ("receiving_2pt_conversions",),
    "fumble_lost": ("sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost"),
    "st_td": ("special_teams_tds",),
    "pass_fd": ("passing_first_downs",),
    "rush_fd": ("rushing_first_downs",),
    "rec_fd": ("receiving_first_downs",),
    # Long plays, counted from play-by-play (features/long_plays.py).
    "pass_cmp_40p": ("passing_cmp_40p",),
    "pass_td_40p": ("passing_td_40p",),
    "rec_40p": ("receiving_40p",),
    "rec_td_40p": ("receiving_td_40p",),
    "rush_40p": ("rushing_40p",),
    "rush_td_40p": ("rushing_td_40p",),
}
# Columns that come from play-by-play rather than nflverse weekly stats.
LONG_PLAY_COLUMNS = tuple(c for f, cols in STAT_COLUMNS.items() if f.endswith("_40p") for c in cols)


def _stat(col: str) -> pl.Expr:
    return pl.col(col).fill_null(0).cast(pl.Float64)


def fantasy_points(weekly: pl.DataFrame, scoring: ScoringConfig) -> pl.DataFrame:
    """Return `weekly` with an added `fpts` column.

    Pure: no I/O, no dates. Null stats count as zero. A position bonus on
    receptions (e.g. TE premium) uses the row's `position` column.
    """
    used = {
        field: cols
        for field, cols in STAT_COLUMNS.items()
        if getattr(scoring, field) != 0
    }
    needed = {c for cols in used.values() for c in cols}
    if scoring.rec_bonus_by_position:
        needed |= {"receptions", "position"}
    needed |= set(scoring.yardage_bonus)
    missing = sorted(needed - set(weekly.columns))
    if missing:
        raise ValueError(f"weekly stats missing columns for scoring '{scoring.name}': {missing}")

    total = pl.lit(0.0)
    for field, cols in used.items():
        total = total + getattr(scoring, field) * pl.sum_horizontal(_stat(c) for c in cols)

    if scoring.rec_bonus_by_position:
        bonus = pl.col("position").replace_strict(
            scoring.rec_bonus_by_position, default=0.0, return_dtype=pl.Float64
        )
        total = total + bonus * _stat("receptions")

    for col, tiers in scoring.yardage_bonus.items():
        bonus = pl.lit(0.0)
        for threshold, points in sorted(tiers):
            bonus = pl.when(_stat(col) >= threshold).then(pl.lit(float(points))).otherwise(bonus)
        total = total + bonus

    return weekly.with_columns(fpts=total.round(4))


def needs_long_plays(scoring: ScoringConfig) -> bool:
    """True when the scoring pays for 40+ yard plays (counts from play-by-play)."""
    return any(getattr(scoring, f) != 0 for f in STAT_COLUMNS if f.endswith("_40p"))
