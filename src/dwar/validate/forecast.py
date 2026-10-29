"""DWAR's out-of-sample forecast of one season's points per game.

For each player, every posterior draw of latent talent is projected one
season ahead (the same projection DWAR uses), then that season's weekly
points are simulated for the number of games the player actually played.
The resulting predictive distribution of points per game is what gets
scored: its mean for MAE, its density for the log score, and where the
actual value falls in it (PIT) for calibration.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.stats import gaussian_kde

from dwar.features.player_tables import PlayerTables
from dwar.model.fit import FitResult
from dwar.model.talent import for_position, link, simulate_weeks
from dwar.value.dwar import draw_indices
from dwar.value.projection import PositionDraws, project_latent, situation_effects
from dwar.value.short_games import ShortGames, short_games
from dwar.value.spread import widen

MIN_GAMES = 4  # player-seasons scored


@dataclass(frozen=True)
class Situation:
    """One position's situation factors for the forecast season, keyed by player_id."""

    level: pl.DataFrame
    range: pl.DataFrame

    @staticmethod
    def aligned(df: pl.DataFrame, player_ids: pl.Series, names: tuple[str, ...]) -> np.ndarray | None:
        """(players, len(names)) in `player_ids` order; players without a row get 0."""
        if not names:
            return None
        rows = pl.DataFrame({"player_id": player_ids}).join(df, on="player_id", how="left")
        return rows.select([pl.col(n).fill_null(0.0) for n in names]).to_numpy().astype(float)


def pit(samples: np.ndarray, actual: float) -> float:
    """Share of the predictive distribution below `actual` (ties split)."""
    return float((samples < actual).mean() + 0.5 * (samples == actual).mean())


def kde_log_density(samples: np.ndarray, actual: float) -> float:
    """Log predictive density at `actual` from simulated samples."""
    if np.ptp(samples) == 0:
        return 0.0 if samples[0] == actual else -np.inf
    return float(np.log(max(gaussian_kde(samples)(actual)[0], 1e-300)))


def forecast_position(
    r: FitResult, likelihood: str, targets: pl.DataFrame, season: int, n_draws: int, rng: np.random.Generator,
    short: ShortGames | None = None, spread: float = 1.0, situation: Situation | None = None,
) -> pl.DataFrame:
    """Forecasts for `targets` (player_id, games_played, fpts_per_game) at
    one position. Players the fit does not know get no row. With `short`,
    some simulated games are short appearances (see value/short_games.py).
    `spread` widens each forecast about its mean (see value/spread.py);
    `pred_sd_raw` is the spread before widening. `situation` holds each
    player's factors for `season`, used when the fit has factor slots."""
    total = r.posterior["posterior"].sizes["chain"] * r.posterior["posterior"].sizes["draw"]
    draws = PositionDraws.from_posterior(r.data, r.posterior, likelihood, draw_indices(total, n_draws))
    latest = (
        r.data.ps_keys.with_row_index("ps_idx")
        .with_columns(age_c=pl.Series(r.data.ps_age_c))
        .filter(pl.col("season") <= season)
        .sort("season").group_by("player_id", maintain_order=True).last()
        .join(targets, on="player_id", how="inner", suffix="_actual")
        .sort("player_id")
    )
    if latest.is_empty():
        return pl.DataFrame()
    last_season = latest["season"].to_numpy()
    ps_idx = latest["ps_idx"].to_numpy()
    level_next = range_next = None
    if situation is not None:
        level_next = situation.aligned(situation.level, latest["player_id"], r.data.season_x_names)
        range_next = situation.aligned(situation.range, latest["player_id"], r.data.season_vol_x_names)
    shift, vol = situation_effects(draws, r.data, ps_idx, last_season, season, level_next, range_next)
    latent = project_latent(
        draws, ps_idx, last_season, latest["age_c"].to_numpy() + 26.0, np.array([season]), rng,
        None if shift is None else shift[:, :, None], None if vol is None else vol[:, :, None],
    )[:, :, 0]
    center = link(latent, likelihood, lib=np)
    params = {n: v[:, None] for n, v in draws.params.items()} | {"zero_value": draws.zero_value}
    talent = draws.expected_points(latent)
    rates = short.rates(latest["player_id"].to_list()) if short else np.zeros(latest.height)
    if short:
        talent = short.expected(talent, rates[None, :])
    rows = []
    for k, row in enumerate(latest.iter_rows(named=True)):
        g = int(row["games_played_actual"])
        weeks = simulate_weeks(likelihood, np.repeat(center[:, k : k + 1], g, axis=1), params, rng)
        if short:
            weeks = short.simulate(weeks, rates[k], rng)
        raw = weeks.mean(axis=1)
        ppg = widen(raw, spread)
        actual = float(row["fpts_per_game"])
        rows.append({
            "player_id": row["player_id"],
            "talent_mean": float(talent[:, k].mean()),
            "pred_mean": float(ppg.mean()),
            "pred_sd": float(ppg.std()),
            "pred_sd_raw": float(raw.std()),
            "spread": spread,
            "pit": pit(ppg, actual),
            "log_score": kde_log_density(ppg, actual),
            "seasons_since_data": season - int(row["season"]),
        })
    return pl.DataFrame(rows)


def forecast_season(
    fits: dict[str, FitResult], likelihood: str | Mapping[str, str], actual: pl.DataFrame, season: int,
    n_draws: int, seed: int, tables: PlayerTables | None = None, min_plays: Mapping[str, int] | None = None,
    spread: Mapping[str, float] | None = None, situation: Mapping[str, Situation] | None = None,
) -> pl.DataFrame:
    """DWAR forecasts for every scored player-season in `actual` (player_id,
    position, games_played, fpts_per_game for `season`), with a
    `has_forecast` flag for players no fit knew (counted, not dropped).
    `tables` (as of the forecast) and `min_plays` add short appearances;
    `spread` widens a position's forecasts."""
    rng = np.random.default_rng(seed + season)
    scored = actual.filter((pl.col("season") == season) & (pl.col("games_played") >= MIN_GAMES))
    frames = []
    for position, r in fits.items():
        targets = scored.filter(pl.col("position") == position).select("player_id", "games_played", "fpts_per_game")
        short = (
            short_games(tables.player_weeks, position, (min_plays or {}).get(position, 0), int(r.data.ps_keys["season"].min()))
            if tables is not None else None
        )
        k = (spread or {}).get(position, 1.0)
        f = forecast_position(
            r, for_position(likelihood, position), targets, season, n_draws, rng, short, k,
            (situation or {}).get(position),
        )
        if not f.is_empty():
            frames.append(f.with_columns(position=pl.lit(position)))
    forecasts = pl.concat(frames) if frames else pl.DataFrame(schema={"player_id": pl.String, "position": pl.String})
    return scored.join(forecasts, on=["player_id", "position"], how="left").with_columns(
        has_forecast=pl.col("talent_mean").is_not_null()
    )
