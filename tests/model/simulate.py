"""Simulate receivers from the talent model's own generative process, with
known parameters, packaged as step 2 `PlayerTables` so the real data path is
exercised end to end."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl

from dwar.features.player_tables import PlayerTables
from dwar.model.data import ModelData, build_model_data

TRUTH = {
    "alpha0": 8.0,
    "alpha_pick": -2.0,
    "alpha_undrafted": -1.0,
    "sigma_rookie": 2.0,
    "mu_vet": 9.0,
    "sigma_vet": 3.0,
    "tau": 1.5,
    "beta1": 0.1,
    "beta2": -0.05,
    "sigma": 5.0,
    "nu": 5.0,
}
# Prospect covariates: faster 40 helps; the rest have no effect.
TRUE_GAMMA = {"forty_z": -0.8}
FIRST_SEASON = 2006


@dataclass(frozen=True)
class Simulated:
    tables: PlayerTables
    data: ModelData
    talent: np.ndarray  # true talent per player-season, aligned with data.ps_keys
    realized: dict      # sample statistics of the simulated draws (see below)


def _tables(players: pl.DataFrame, seasons: pl.DataFrame, weeks: pl.DataFrame) -> PlayerTables:
    return PlayerTables(
        as_of=date(2030, 1, 1),
        player_weeks=weeks.with_columns(position=pl.lit("WR")),
        player_seasons=seasons.select("player_id", "season", "age_sept1"),
        player_static=players,
    )


def simulate(
    n_players: int = 150,
    weeks_per_season: int = 12,
    seed: int = 7,
    season_effect: float | None = None,
) -> Simulated:
    """`season_effect`, if given, adds one season covariate with that true
    coefficient, to test the covariate slot."""
    rng = np.random.default_rng(seed)
    player_rows, season_rows = [], []
    for k in range(n_players):
        pid = f"SIM-{k:04d}"
        vet = rng.random() < 0.2
        start = int(rng.integers(2008, 2019))
        rookie_age = rng.normal(22.5, 1.0) + (rng.uniform(3, 7) if vet else 0.0)
        n_seasons = int(rng.integers(2, 7))
        drafted = rng.random() < 0.7
        season, played = start, []
        for _ in range(n_seasons):
            played.append(season)
            season += 2 if rng.random() < 0.1 else 1
        player_rows.append({
            "player_id": pid,
            "draft_pick": int(rng.integers(1, 257)) if drafted else None,
            "first_season": 2003 if vet else start,
            "forty": None if rng.random() < 0.2 else float(rng.normal(4.5, 0.1)),
            "ht_in": int(rng.integers(69, 77)),
            "wt": float(rng.normal(200, 12)),
        })
        for s in played:
            season_rows.append({"player_id": pid, "season": s, "age_sept1": rookie_age + (s - start)})
    players = pl.DataFrame(player_rows, schema_overrides={"draft_pick": pl.Int64, "forty": pl.Float64})
    seasons = pl.DataFrame(season_rows)
    if season_effect is not None:
        seasons = seasons.with_columns(boost=pl.Series(rng.normal(0, 1, seasons.height)))

    # Build once with placeholder weeks to get the exact feature arrays the model sees.
    skeleton = seasons.select("player_id", "season", fpts=pl.lit(0.0))
    covs = seasons.select("player_id", "season", "boost") if season_effect is not None else None
    d0 = build_model_data(_tables(players, seasons, skeleton), first_season=FIRST_SEASON, season_covariates=covs)

    t = TRUTH
    gamma = np.array([TRUE_GAMMA.get(n, 0.0) for n in d0.prospect_x_names])
    rookie_mean = t["alpha0"] + t["alpha_pick"] * d0.log_pick_z + t["alpha_undrafted"] * d0.undrafted + d0.prospect_x @ gamma
    prior_mean = np.where(d0.pre_window > 0, t["mu_vet"], rookie_mean)
    prior_sd = np.where(d0.pre_window > 0, t["sigma_vet"], t["sigma_rookie"])
    age_effect = t["beta1"] * d0.ps_age_c + t["beta2"] * d0.ps_age_c**2
    z = rng.normal(size=d0.n_ps)
    steps = np.where(
        d0.ps_is_first,
        prior_mean[d0.ps_player] + prior_sd[d0.ps_player] * z - age_effect,
        t["tau"] * np.sqrt(np.maximum(d0.ps_gap, 1.0)) * z,
    )
    csum = np.cumsum(steps)
    level = csum - csum[d0.ps_start] + steps[d0.ps_start]
    talent = level + age_effect
    if season_effect is not None:
        talent = talent + season_effect * d0.season_x[:, 0]

    keys = d0.ps_keys.select("player_id", "season")
    n = weeks_per_season
    weeks = pl.DataFrame({
        "player_id": np.repeat(keys["player_id"].to_numpy(), n),
        "season": np.repeat(keys["season"].to_numpy(), n),
        "fpts": np.repeat(talent, n) + t["sigma"] * rng.standard_t(t["nu"], size=d0.n_ps * n),
    })
    tables = _tables(players, seasons, weeks)
    data = build_model_data(tables, first_season=FIRST_SEASON, season_covariates=covs)
    # With ~50 veterans, the sample mean and sd of their starting levels can sit
    # well away from mu_vet / sigma_vet by chance; the data can only reveal the
    # realized values, so the recovery test checks against these.
    vet_first = d0.ps_is_first & (d0.pre_window[d0.ps_player] > 0)
    vet_start = (level + age_effect)[vet_first]
    realized = (
        {"mu_vet": float(vet_start.mean()), "sigma_vet": float(vet_start.std(ddof=1))}
        if len(vet_start) > 1 else {}
    )
    return Simulated(tables=tables, data=data, talent=talent, realized=realized)
