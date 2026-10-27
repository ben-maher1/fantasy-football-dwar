"""Out-of-sample check of the availability models DWAR uses.

For every player who played in season s - 1, using fits and availability
models as of the start of season s:

- survival: the forecast chance the player plays in season s (at least one game)
  against whether they did;
- games: for those who played, the forecast distribution of games played
  against the actual count.

Each is compared to a no-skill baseline: the position's base rate (share
surviving, mean share of games) from the same training rows.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from dwar.features.player_tables import PlayerTables
from dwar.model.fit import FitResult
from dwar.validate.forecast import pit
from dwar.value.availability import availability_rows, fit_availability, season_games
from dwar.value.dwar import draw_indices
from dwar.value.projection import PositionDraws, project_talent

CALIBRATION_BINS = (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0)


def availability_forecast(
    fits: dict[str, FitResult], tables: PlayerTables, schedules: pl.DataFrame, likelihood: str,
    actual: pl.DataFrame, season: int, n_draws: int, seed: int,
) -> pl.DataFrame:
    """One row per player who played in `season - 1`: p_survive and
    survived; for those who played in `season`, games forecast and actual.
    `actual` is the truth player-season table (player_id, season,
    games_played)."""
    rng = np.random.default_rng(seed + season)
    talent = pl.concat([r.talent.with_columns(position=pl.lit(p)) for p, r in fits.items()])
    model = fit_availability(talent, tables.player_seasons, schedules, tables.as_of)
    av = model.draw(n_draws, rng)
    games_rows, survival_rows = availability_rows(talent, tables.player_seasons, schedules, tables.as_of)
    n_games = int(season_games(schedules).filter(pl.col("season") == season)["games"][0])
    played = actual.filter((pl.col("season") == season) & (pl.col("games_played") > 0)).select(
        "player_id", games_actual="games_played"
    )

    frames = []
    for position, r in fits.items():
        total = r.posterior["posterior"].sizes["chain"] * r.posterior["posterior"].sizes["draw"]
        draws = PositionDraws.from_posterior(r.data, r.posterior, likelihood, draw_indices(total, n_draws))
        prev = (
            r.data.ps_keys.with_row_index("ps_idx")
            .with_columns(age_c=pl.Series(r.data.ps_age_c))
            .filter((pl.col("season") == season - 1) & (pl.col("games_played") > 0))
            .sort("player_id")
        )
        if prev.is_empty():
            continue
        age = prev["age_c"].to_numpy() + 26.0
        t = project_talent(
            draws, prev["ps_idx"].to_numpy(), prev["season"].to_numpy(), age, np.array([season - 1, season]), rng
        )
        ages = np.broadcast_to(np.stack([age, age + 1], axis=-1)[None], t.shape)
        p_survive = av.survival_prob(position, ages[:, :, 0], t[:, :, 0])
        share = av.games_share(position, ages[:, :, 1], t[:, :, 1], rng)
        games = rng.binomial(n_games, share)

        g_rows = games_rows.filter(pl.col("position") == position)
        base_share = float(g_rows["successes"].sum() / g_rows["trials"].sum())
        base_survive = float(survival_rows.filter(pl.col("position") == position)["successes"].mean())
        f = prev.select("player_id").with_columns(
            position=pl.lit(position),
            p_survive=pl.Series(p_survive.mean(axis=0)),
            base_survive=pl.lit(base_survive),
            games_mean=pl.Series(games.mean(axis=0)),
            base_games=pl.lit(base_share * n_games),
        ).join(played, on="player_id", how="left")
        actual_g = f["games_actual"].to_list()
        f = f.with_columns(
            survived=pl.col("games_actual").is_not_null(),
            games_pit=pl.Series(
                [pit(games[:, k], actual_g[k]) if actual_g[k] is not None else None for k in range(len(actual_g))],
                dtype=pl.Float64,
            ),
        )
        frames.append(f)
    return pl.concat(frames).with_columns(season=pl.lit(season), season_games=pl.lit(n_games))


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _group_summary(rows: pl.DataFrame) -> dict:
    y = rows["survived"].to_numpy().astype(float)
    p, base = rows["p_survive"].to_numpy(), rows["base_survive"].to_numpy()
    bins = np.digitize(p, CALIBRATION_BINS[1:-1])
    calibration = [
        {"bin": f"{CALIBRATION_BINS[b]:.2f}-{CALIBRATION_BINS[b + 1]:.2f}", "n": int((bins == b).sum()),
         "forecast": float(p[bins == b].mean()), "observed": float(y[bins == b].mean())}
        for b in range(len(CALIBRATION_BINS) - 1) if (bins == b).any()
    ]
    g = rows.filter(pl.col("survived"))
    ga = g["games_actual"].to_numpy().astype(float)
    centered = np.abs(g["games_pit"].to_numpy() - 0.5)
    return {
        "survival": {
            "n": rows.height, "observed_rate": float(y.mean()), "forecast_rate": float(p.mean()),
            "brier": _brier(p, y), "base_brier": _brier(base, y), "calibration": calibration,
        },
        "games": {
            "n": g.height, "mae": float(np.mean(np.abs(g["games_mean"].to_numpy() - ga))),
            "base_mae": float(np.mean(np.abs(g["base_games"].to_numpy() - ga))),
            "coverage_50": float((centered <= 0.25).mean()), "coverage_90": float((centered <= 0.45).mean()),
        },
    }


def summarize_availability(rows: pl.DataFrame, positions: tuple[str, ...]) -> dict:
    """Survival Brier and calibration, games MAE and coverage, vs base rates."""
    groups = {"all": rows} | {p: rows.filter(pl.col("position") == p) for p in positions}
    return {name: _group_summary(g) for name, g in groups.items() if not g.is_empty()}
