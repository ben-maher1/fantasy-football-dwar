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
from dwar.config import load_model_config
from dwar.model.talent import LIKELIHOOD_PARAMS, link, scale_of, simulate_weeks, talent_from_center

# Latent truth on each likelihood's latent scale (points, or sqrt(points)
# for sqrt_hurdle, log(loc) for sqrt_hurdle_log, where 3.0 is about 9 ppg). The sqrt values sit
# near the real-WR posterior medians.
LATENT_TRUTH = {
    "points": {
        "alpha0": 8.0, "alpha_pick": -2.0, "alpha_undrafted": -1.0, "sigma_rookie": 2.0,
        "mu_vet": 9.0, "sigma_vet": 3.0, "tau": 1.5, "beta1": 0.1, "beta2": -0.05,
    },
    "sqrt": {
        "alpha0": 2.6, "alpha_pick": -0.5, "alpha_undrafted": -0.2, "sigma_rookie": 0.5,
        "mu_vet": 3.0, "sigma_vet": 0.7, "tau": 0.4, "beta1": 0.03, "beta2": -0.02,
    },
    # log(loc): about the sqrt values above, relative to a loc near 3.
    "log": {
        "alpha0": 0.95, "alpha_pick": -0.17, "alpha_undrafted": -0.07, "sigma_rookie": 0.18,
        "mu_vet": 1.1, "sigma_vet": 0.25, "tau": 0.13, "beta1": 0.01, "beta2": -0.007,
    },
}
LIKELIHOOD_TRUTH = {
    "sqrt_hurdle": {"sigma": 1.0, "nu": 20.0, "h0": 2.5, "h1": 2.0},
    "sqrt_hurdle_log": {"sigma": 1.0, "nu": 20.0, "h0": 2.5, "h1": 2.0},
    "studentt": {"sigma": 5.0, "nu": 5.0},
}
# Prospect covariates: faster 40 helps; the rest have no effect.
TRUE_GAMMA = {"points": {"forty_z": -0.8}, "sqrt": {"forty_z": -0.2}, "log": {"forty_z": -0.07}}
# A season covariate's true coefficient, for the covariate-slot test.
SEASON_EFFECT = {"points": 1.5, "sqrt": 0.4, "log": 0.13}
FIRST_SEASON = 2006


@dataclass(frozen=True)
class Simulated:
    tables: PlayerTables
    data: ModelData
    likelihood: str
    talent: np.ndarray  # true talent (expected ppg) per player-season, aligned with data.ps_keys
    realized: dict      # sample statistics of the simulated draws (see below)

    @property
    def scale(self) -> str:
        return scale_of(self.likelihood)

    @property
    def gamma(self) -> dict[str, float]:
        return TRUE_GAMMA[self.scale]

    @property
    def truth(self) -> dict[str, float]:
        params = {k: v for k, v in LIKELIHOOD_TRUTH[self.likelihood].items() if k in LIKELIHOOD_PARAMS[self.likelihood]}
        return LATENT_TRUTH[self.scale] | params | self.realized


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
    likelihood: str | None = None,
    range_effect: float | None = None,
) -> Simulated:
    """`season_effect`, if given, adds one season covariate with that true
    coefficient, to test the covariate slot. `range_effect`, if given, adds
    a 0/1 flag (a third of seasons) that multiplies that season's step by
    exp(range_effect). `likelihood` defaults to the configured one."""
    likelihood = likelihood or load_model_config().likelihood
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
    if range_effect is not None:
        seasons = seasons.with_columns(moved=pl.Series((rng.random(seasons.height) < 1 / 3).astype(float)))

    # Build once with placeholder weeks to get the exact feature arrays the model sees.
    skeleton = seasons.select("player_id", "season", fpts=pl.lit(0.0))
    covs = seasons.select("player_id", "season", "boost") if season_effect is not None else None
    vols = seasons.select("player_id", "season", "moved") if range_effect is not None else None
    d0 = build_model_data(
        _tables(players, seasons, skeleton), first_season=FIRST_SEASON, season_covariates=covs,
        season_range_covariates=vols,
    )

    t = LATENT_TRUTH[scale_of(likelihood)]
    gamma = np.array([TRUE_GAMMA[scale_of(likelihood)].get(n, 0.0) for n in d0.prospect_x_names])
    rookie_mean = t["alpha0"] + t["alpha_pick"] * d0.log_pick_z + t["alpha_undrafted"] * d0.undrafted + d0.prospect_x @ gamma
    prior_mean = np.where(d0.pre_window > 0, t["mu_vet"], rookie_mean)
    prior_sd = np.where(d0.pre_window > 0, t["sigma_vet"], t["sigma_rookie"])
    age_effect = t["beta1"] * d0.ps_age_c + t["beta2"] * d0.ps_age_c**2
    z = rng.normal(size=d0.n_ps)
    steps = np.where(
        d0.ps_is_first,
        prior_mean[d0.ps_player] + prior_sd[d0.ps_player] * z - age_effect,
        t["tau"] * np.sqrt(np.maximum(d0.ps_gap, 1.0)) * z
        * (np.exp(range_effect * d0.season_vol_x[:, 0]) if range_effect is not None else 1.0),
    )
    csum = np.cumsum(steps)
    level = csum - csum[d0.ps_start] + steps[d0.ps_start]
    latent = level + age_effect
    if season_effect is not None:
        latent = latent + season_effect * d0.season_x[:, 0]
    center = link(latent, likelihood, lib=np)
    talent = talent_from_center(likelihood, center, LIKELIHOOD_TRUTH[likelihood])

    keys = d0.ps_keys.select("player_id", "season")
    n = weeks_per_season
    weeks = pl.DataFrame({
        "player_id": np.repeat(keys["player_id"].to_numpy(), n),
        "season": np.repeat(keys["season"].to_numpy(), n),
        "fpts": simulate_weeks(likelihood, np.repeat(center, n), LIKELIHOOD_TRUTH[likelihood], rng),
    })
    tables = _tables(players, seasons, weeks)
    data = build_model_data(tables, first_season=FIRST_SEASON, season_covariates=covs, season_range_covariates=vols)
    # With ~50 veterans, the sample mean and sd of their starting levels can sit
    # well away from mu_vet / sigma_vet by chance; the data can only reveal the
    # realized values, so the recovery test checks against these.
    vet_first = d0.ps_is_first & (d0.pre_window[d0.ps_player] > 0)
    vet_start = (level + age_effect)[vet_first]
    realized = (
        {"mu_vet": float(vet_start.mean()), "sigma_vet": float(vet_start.std(ddof=1))}
        if len(vet_start) > 1 else {}
    )
    return Simulated(tables=tables, data=data, likelihood=likelihood, talent=talent, realized=realized)
