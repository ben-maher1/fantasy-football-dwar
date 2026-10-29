"""Short appearances: games a player entered but barely played (an injury
exit, mop-up relief), set per position by `min_plays` in the model config.

Talent fits leave these weeks out, because they say little about a scoring
rate. Forecasts and DWAR add them back: each game is short with the player's
own chance of a short game (their history, shrunk toward the position rate),
and a short game scores like the position's observed short weeks. Without
this, a backup QB, whose games are mostly short, is forecast as if every
game were a full start.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

PRIOR_GAMES = 8.0  # games of position-average evidence each player's rate starts from


@dataclass(frozen=True)
class ShortGames:
    rate: dict[str, float]  # player -> chance a game is short
    base_rate: float        # position rate, for players with no history
    points: np.ndarray      # the position's observed short-week points

    def rates(self, player_ids: list[str]) -> np.ndarray:
        return np.array([self.rate.get(p, self.base_rate) for p in player_ids])

    @property
    def mean_points(self) -> float:
        return float(self.points.mean()) if len(self.points) else 0.0

    def expected(self, full: np.ndarray, rates: np.ndarray) -> np.ndarray:
        """Expected points per game from full-game expected points."""
        return (1.0 - rates) * full + rates * self.mean_points

    def simulate(self, full_weeks: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
        """Replace each simulated full week with a short one at `rate`."""
        if not len(self.points):
            return full_weeks
        short = rng.random(full_weeks.shape) < rate
        return np.where(short, rng.choice(self.points, size=full_weeks.shape), full_weeks)


def short_games(player_weeks: pl.DataFrame, position: str, min_plays: int, first_season: int) -> ShortGames | None:
    """Short-game rates from `player_weeks` (as-of tables, with `plays`), or
    None when the position has no play threshold."""
    if not min_plays:
        return None
    w = player_weeks.filter((pl.col("position") == position) & (pl.col("season") >= first_season)).with_columns(
        short=pl.col("plays") < min_plays
    )
    if w.is_empty():
        return ShortGames(rate={}, base_rate=0.0, points=np.array([]))
    base = float(w["short"].mean())
    per = w.group_by("player_id").agg(n=pl.len(), k=pl.col("short").sum())
    rate = (per["k"] + PRIOR_GAMES * base) / (per["n"] + PRIOR_GAMES)
    return ShortGames(
        rate=dict(zip(per["player_id"].to_list(), rate.to_list())),
        base_rate=base,
        points=w.filter(pl.col("short"))["fpts"].to_numpy().astype(float),
    )
