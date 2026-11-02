"""Short appearances: games a player entered but barely played (an injury
exit, mop-up relief), set per position by `min_plays` in the model config.

Talent fits leave these weeks out, because they say little about a scoring
rate. Forecasts and DWAR add them back: each game is short with the player's
own chance of a short game (their history, shrunk toward the position rate),
and a short game scores like the position's observed short weeks. Without
this, a backup QB, whose games are mostly short, is forecast as if every
game were a full start.

With situation factors, the chance also depends on the player's preseason
role (for QBs, whether he is QB1 on the depth chart): history is counted
within each role and shrunk toward that role's rate. A starter benched
over the offseason then gets a backup's chance, not his starter history.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

PRIOR_GAMES = 8.0  # games of position-average evidence each player's rate starts from
ROLE_FACTOR = {"QB": "qb_starter"}  # the situation factor that sets a player's role


@dataclass(frozen=True)
class ShortGames:
    rate: dict[str, float]  # player -> chance a game is short
    base_rate: float        # position rate, for players with no history
    points: np.ndarray      # the position's observed short-week points
    role_rate: dict[tuple[str, float], float] = field(default_factory=dict)  # (player, role) -> chance
    role_base: dict[float, float] = field(default_factory=dict)               # role -> chance

    def rates(self, player_ids: list[str], roles: list[float] | None = None) -> np.ndarray:
        """Each player's chance of a short game, in the given roles when
        known (role rates exist and the player's role is not None)."""
        if roles is None or not self.role_base:
            return np.array([self.rate.get(p, self.base_rate) for p in player_ids])
        return np.array([
            self.rate.get(p, self.base_rate) if r is None
            else self.role_rate.get((p, r), self.role_base.get(r, self.base_rate))
            for p, r in zip(player_ids, roles)
        ])

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


def short_games(
    player_weeks: pl.DataFrame, position: str, min_plays: int, first_season: int,
    roles: pl.DataFrame | None = None,
) -> ShortGames | None:
    """Short-game rates from `player_weeks` (as-of tables, with `plays`), or
    None when the position has no play threshold. `roles` (player_id,
    season, role) adds rates by preseason role."""
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
    role_rate, role_base = {}, {}
    if roles is not None and not roles.is_empty():
        rw = w.join(roles.select("player_id", "season", "role"), on=["player_id", "season"], how="inner")
        role_base = {r: float(v) for r, v in rw.group_by("role").agg(pl.col("short").mean()).iter_rows()}
        by = rw.group_by("player_id", "role").agg(n=pl.len(), k=pl.col("short").sum())
        for pid, r, n, k in by.iter_rows():
            role_rate[(pid, r)] = (k + PRIOR_GAMES * role_base[r]) / (n + PRIOR_GAMES)
    return ShortGames(
        rate=dict(zip(per["player_id"].to_list(), rate.to_list())),
        base_rate=base,
        points=w.filter(pl.col("short"))["fpts"].to_numpy().astype(float),
        role_rate=role_rate,
        role_base=role_base,
    )
