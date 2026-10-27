"""Games played and career survival, estimated from history.

Both are binomial logistic regressions, one per position, on age (centered
at 26, with a square term) and talent (expected points per game):

- games: games played out of the season's games, among players who played
  at least once. Overdispersion (injury seasons cluster) is kept: a
  projected season's share of games is drawn from a Beta whose spread
  matches the observed Pearson dispersion.
- survival: whether a player who played in season s plays in season s + 1.

Training rows come only from seasons whose regular season ended before
`as_of`, so a fit never sees an outcome that was not yet known.
Coefficient uncertainty is carried by drawing from the normal
approximation to each fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

AGE_CENTER = 26.0


def _design(age: np.ndarray, talent: np.ndarray) -> np.ndarray:
    a = np.asarray(age, float) - AGE_CENTER
    return np.stack([np.ones_like(a), a, a**2, np.asarray(talent, float)], axis=-1)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


@dataclass(frozen=True)
class Logistic:
    coef: np.ndarray        # (k,)
    cov: np.ndarray         # (k, k)
    dispersion: float = 1.0  # Pearson chi2 / df; 1 = binomial

    def draw_coefs(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.multivariate_normal(self.coef, self.cov, size=n)


def fit_logistic(x: np.ndarray, successes: np.ndarray, trials: np.ndarray, ridge: float = 0.01) -> Logistic:
    """Binomial logistic regression by Newton's method, with a weak Normal(0, 1/sqrt(ridge))
    prior on each coefficient so a position where everyone plays every game (no
    variation) still gives a finite fit."""
    successes, trials = np.asarray(successes, float), np.asarray(trials, float)
    beta = np.zeros(x.shape[1])
    for _ in range(200):
        p = _sigmoid(x @ beta)
        w = trials * p * (1 - p)
        hess = x.T @ (w[:, None] * x) + ridge * np.eye(x.shape[1])
        step = np.linalg.solve(hess, x.T @ (successes - trials * p) - ridge * beta)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    else:
        raise RuntimeError("logistic regression did not converge")
    p = _sigmoid(x @ beta)
    w = trials * p * (1 - p)
    cov = np.linalg.inv(x.T @ (w[:, None] * x) + ridge * np.eye(x.shape[1]))
    resid = (successes - trials * p) / np.sqrt(np.maximum(w, 1e-12))
    dispersion = float(np.sum(resid**2) / max(len(successes) - x.shape[1], 1))
    return Logistic(coef=beta, cov=cov * max(dispersion, 1.0), dispersion=dispersion)


@dataclass(frozen=True)
class AvailabilityModel:
    games: dict[str, Logistic]
    survival: dict[str, Logistic]
    season_games: int  # games per team in the latest complete season

    def draw(self, n_draws: int, rng: np.random.Generator) -> AvailabilityDraws:
        return AvailabilityDraws(
            # Sorted, so the random stream does not depend on dict order.
            games={p: self.games[p].draw_coefs(n_draws, rng) for p in sorted(self.games)},
            games_dispersion={p: m.dispersion for p, m in self.games.items()},
            survival={p: self.survival[p].draw_coefs(n_draws, rng) for p in sorted(self.survival)},
            season_games=self.season_games,
        )


@dataclass(frozen=True)
class AvailabilityDraws:
    """Coefficient draws, one row per posterior draw."""

    games: dict[str, np.ndarray]
    games_dispersion: dict[str, float]
    survival: dict[str, np.ndarray]
    season_games: int

    def survival_prob(self, position: str, age: np.ndarray, talent: np.ndarray) -> np.ndarray:
        """P(plays next season); `age` and `talent` are (draws, players)."""
        return _sigmoid(np.einsum("dpk,dk->dp", _design(age, talent), self.survival[position]))

    def games_share(
        self, position: str, age: np.ndarray, talent: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Share of a season's games played, drawn with season-to-season spread."""
        p = np.clip(_sigmoid(np.einsum("dpk,dk->dp", _design(age, talent), self.games[position])), 1e-6, 1 - 1e-6)
        n, phi = self.season_games, self.games_dispersion[position]
        if phi <= 1.0 or n <= 1:
            return p
        rho = min((phi - 1.0) / (n - 1.0), 0.99)
        kappa = 1.0 / rho - 1.0
        return rng.beta(p * kappa, (1 - p) * kappa)


def season_games(schedules: pl.DataFrame) -> pl.DataFrame:
    """Regular-season games per team and last game date, by season."""
    per_team = pl.concat([
        schedules.select("season", "gameday", team="home_team"),
        schedules.select("season", "gameday", team="away_team"),
    ])
    return per_team.group_by("season").agg(
        games=pl.len().truediv(pl.col("team").n_unique()).round().cast(pl.Int64),
        last_gameday=pl.col("gameday").max(),
    )


def availability_rows(
    talent: pl.DataFrame, player_seasons: pl.DataFrame, schedules: pl.DataFrame, as_of
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Training rows for the games and survival models.

    `talent` has player_id, season, position, talent_mean (one fitted talent
    per player-season). Returns (games_rows, survival_rows), each with
    position, age, talent, successes, trials."""
    seasons = season_games(schedules).filter(pl.col("last_gameday") < as_of)
    complete = seasons["season"]
    base = talent.join(player_seasons.select("player_id", "season", "age_sept1"), on=["player_id", "season"]).filter(
        pl.col("age_sept1").is_not_null()
    )
    games = (
        base.join(seasons, on="season")
        .filter(pl.col("games_played") > 0)
        .select(
            "position", age="age_sept1", talent="talent_mean",
            successes=pl.min_horizontal("games_played", "games"), trials="games",
        )
    )
    played = talent.filter(pl.col("games_played") > 0).select("player_id", "season").unique()
    survival = (
        base.filter((pl.col("games_played") > 0) & (pl.col("season") + 1).is_in(complete.implode()))
        .join(played.with_columns(pl.col("season") - 1, next_played=pl.lit(1)), on=["player_id", "season"], how="left")
        .select(
            "position", age="age_sept1", talent="talent_mean",
            successes=pl.col("next_played").fill_null(0), trials=pl.lit(1),
        )
    )
    return games, survival


def fit_availability(
    talent: pl.DataFrame, player_seasons: pl.DataFrame, schedules: pl.DataFrame, as_of
) -> AvailabilityModel:
    games_rows, survival_rows = availability_rows(talent, player_seasons, schedules, as_of)
    out: dict[str, dict[str, Logistic]] = {"games": {}, "survival": {}}
    for kind, rows in (("games", games_rows), ("survival", survival_rows)):
        for (position,), r in rows.sort("position").group_by("position", maintain_order=True):
            out[kind][position] = fit_logistic(
                _design(r["age"].to_numpy(), r["talent"].to_numpy()),
                r["successes"].to_numpy(), r["trials"].to_numpy(),
            )
    complete = season_games(schedules).filter(pl.col("last_gameday") < as_of).sort("season")
    return AvailabilityModel(games=out["games"], survival=out["survival"], season_games=int(complete["games"][-1]))
