"""Points above replacement -> wins.

A week's result is a comparison of two team scores. If each team's weekly
score has standard deviation `sigma` around its mean, a player who adds
`par` points a week lifts the team's win probability that week from 1/2 to
Phi(par / (sqrt(2) * sigma)).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.special import ndtr

from dwar.value.replacement import starters_used


def wins_per_game(par: np.ndarray, sigma: float) -> np.ndarray:
    """Expected wins added in one game by `par` points above replacement."""
    return ndtr(np.asarray(par) / (np.sqrt(2.0) * sigma)) - 0.5


def team_score_sd(
    player_weeks: pl.DataFrame,
    player_seasons: pl.DataFrame,
    seasons: list[int],
    lineup: dict[str, int],
    teams: int,
    min_games: int = 4,
) -> float:
    """Standard deviation of a full starting lineup's weekly score.

    For each season, starters are allocated by points per game (players with
    `min_games` or more) exactly as for replacement level. The weekly-score
    variance of starter-level players at each position is measured from
    their actual weeks, and a lineup's variance is the sum over its starters
    (`starters_used / teams` per position), treating starters' weeks as
    independent. Averaged over `seasons`.
    """
    variances = []
    for season in seasons:
        ps = player_seasons.filter((pl.col("season") == season) & (pl.col("games_played") >= min_games))
        if ps.is_empty():
            continue
        positions = ps["position"].to_numpy()
        used = starters_used(ps["fpts_per_game"].to_numpy()[None, :], positions, lineup, teams)
        var = 0.0
        for p, n in used.items():
            n = int(n[0])
            if n == 0:
                continue
            starters = ps.filter(pl.col("position") == p).sort("fpts_per_game", descending=True).head(n)
            weeks = player_weeks.filter(pl.col("season") == season).join(
                starters.select("player_id"), on="player_id", how="semi"
            )
            var += n / teams * float(weeks["fpts"].var())
        variances.append(var)
    if not variances:
        raise ValueError(f"no player-seasons with {min_games}+ games in seasons {seasons}")
    return float(np.sqrt(np.mean(variances)))
