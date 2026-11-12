"""Pull-back for old star players' future seasons (RBs, 2026-11-10).

The talent model keeps an aging star too high: for RBs 29+ coming off a
12+ PPG season, its forecasts one to three seasons out missed by 3.5 PPG on
average in 2014-25, while a bare rule, last season's PPG times the share
such players kept in history, missed by 2.8 (better in three of four test
periods; validate notes in reports/old_star/README.md). The model's career
structure barely pulls a star back toward normal, and old backs show that
pull-back most.

So for such a player, each future season's talent is moved `weight` of the
way from the model's mean toward `ppg * keep[h]`, where `ppg` is his latest
season with at least `min_games` games, h is seasons after it (capped at the
last keep-rate), and keep[h] is measured from history as of the run date:
total PPG in season s + h (players who played `outcome_games`+) over their
PPG in s, among player-seasons at `min_age`+ with `min_ppg`+. Each draw is
scaled by the same factor, so the spread stays proportional.
"""

from __future__ import annotations

import numpy as np
import polars as pl

HORIZONS = 3
OUTCOME_GAMES = 4


def keep_rates(
    player_seasons: pl.DataFrame, position: str, min_age: float, min_ppg: float, min_games: int,
    complete: tuple[int, ...],
) -> np.ndarray:
    """keep[h - 1] for h = 1..HORIZONS, from seasons in `complete` only."""
    ps = player_seasons.filter((pl.col("position") == position) & pl.col("season").is_in(list(complete)))
    origin = ps.filter(
        (pl.col("games_played") >= min_games) & (pl.col("age_sept1") >= min_age) & (pl.col("fpts_per_game") >= min_ppg)
    ).select("player_id", "season", x="fpts_per_game")
    out = []
    for h in range(1, HORIZONS + 1):
        later = ps.filter(pl.col("games_played") >= OUTCOME_GAMES).select(
            "player_id", pl.col("season") - h, y="fpts_per_game"
        )
        j = origin.join(later, on=["player_id", "season"])
        out.append(float(j["y"].sum() / j["x"].sum()) if j.height else np.nan)
    keep = np.array(out)
    # A horizon with no history keeps the one before it.
    for k in range(1, HORIZONS):
        if np.isnan(keep[k]):
            keep[k] = keep[k - 1]
    return keep


def latest_full_season(
    player_seasons: pl.DataFrame, player_ids: list[str], min_games: int,
) -> pl.DataFrame:
    """Per player (in `player_ids` order): season, PPG and age of his latest
    season with `min_games`+ games (nulls if none)."""
    last = (
        player_seasons.filter(pl.col("player_id").is_in(player_ids) & (pl.col("games_played") >= min_games))
        .sort("season").group_by("player_id", maintain_order=True).last()
        .select("player_id", "season", ppg="fpts_per_game", age="age_sept1")
    )
    return pl.DataFrame({"player_id": player_ids}).join(last, on="player_id", how="left", maintain_order="left")


def pull_back(
    talent: np.ndarray, seasons: np.ndarray, current: int, last: pl.DataFrame, keep: np.ndarray,
    min_age: float, min_ppg: float, weight: float,
) -> np.ndarray:
    """talent (draws, players, seasons) with future seasons (after `current`)
    of qualifying players moved toward ppg * keep[h]."""
    out = talent.copy()
    season = last["season"].to_numpy(allow_copy=True).astype(float)
    ppg = last["ppg"].to_numpy(allow_copy=True).astype(float)
    age = last["age"].to_numpy(allow_copy=True).astype(float)
    hit = np.nan_to_num(age, nan=0.0) >= min_age
    hit &= np.nan_to_num(ppg, nan=0.0) >= min_ppg
    for j, s in enumerate(seasons):
        if s <= current:
            continue
        idx = np.flatnonzero(hit & (s > season))
        if not len(idx):
            continue
        h = np.clip((s - season[idx]).astype(int), 1, HORIZONS)
        target = ppg[idx] * keep[h - 1]
        mean = talent[:, idx, j].mean(axis=0)
        factor = np.where(mean > 0, ((1 - weight) * mean + weight * target) / np.maximum(mean, 1e-9), 1.0)
        out[:, idx, j] = talent[:, idx, j] * factor[None, :]
    return out
