"""Reach another league's scoring by scaling each player's points (ben,
2026-11-26: redraft league "Length" adds 0.5 per rushing or receiving first
down; chose scaling the half-PPR fits over a refit).

A player's factor is his own ratio of points under the league's scoring to
points under the fits' scoring over the last `years` seasons (as of the
run date), shrunk toward his position's ratio by `prior_games` games of
position-average scoring. Players with no games get the position ratio.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from dwar.config import load_scoring

YEARS = 3
PRIOR_GAMES = 8.0


def scale_factors(weeks: pl.DataFrame, base_col: str, new_col: str, ids: list[str], positions: list[str],
                  season: int, years: int = YEARS, prior_games: float = PRIOR_GAMES) -> np.ndarray:
    """(len(ids),) factor new / base per player from `weeks` (one row per
    game played: player_id, position, season, base_col, new_col)."""
    w = weeks.filter(pl.col("season").is_between(season - years + 1, season))
    by_pos = w.group_by("position").agg(pl.col(base_col).sum().alias("H"), pl.col(new_col).sum().alias("F"))
    r_pos = {p: (f / h if h > 0 else 1.0) for p, h, f in by_pos.select("position", "H", "F").iter_rows()}
    per = w.group_by("player_id").agg(pl.col(base_col).sum().alias("H"), pl.col(new_col).sum().alias("F"),
                                      pl.len().alias("n"))
    stats = {i: (h, f, n) for i, h, f, n in per.iter_rows()}
    out = np.ones(len(ids))
    for k, (i, p) in enumerate(zip(ids, positions)):
        rp = r_pos.get(p, 1.0)
        h, f, n = stats.get(i, (0.0, 0.0, 0))
        hbar = max(h / n, 1.0) if n else 1.0       # per-game base points, floored so tiny scorers shrink hard
        if n == 0 or h <= 0:
            out[k] = rp
        else:
            out[k] = (f + prior_games * rp * hbar) / (h + prior_games * hbar)
    return out


def factors_for(as_of: date, base_scoring: str, new_scoring: str, ids: list[str], positions: list[str],
                season: int) -> np.ndarray:
    """Factors from the cached weekly stats visible on `as_of` (games played only)."""
    from dwar.features.asof import RawTables, asof_snapshot

    snap = asof_snapshot(RawTables.from_cache(), as_of)
    weeks = weekly_both(snap, base_scoring, new_scoring)
    return scale_factors(weeks, "base", "new", ids, positions, season)


def weekly_both(snap, base_scoring: str, new_scoring: str) -> pl.DataFrame:
    """player_id, position, season, week, base and new points per game played."""
    from dwar.features.player_tables import player_weeks

    keys = ["player_id", "season", "week"]
    b = player_weeks(snap, load_scoring(base_scoring)).select(*keys, "position", pl.col("fpts").alias("base"))
    n = player_weeks(snap, load_scoring(new_scoring)).select(*keys, pl.col("fpts").alias("new"))
    return b.join(n, on=keys, how="inner")
