"""Carry each posterior draw of a player's talent forward season by season.

Projection follows the fitted model: each season the latent talent moves
along the position's age curve and takes a random-walk step of the fitted
size; the weekly-points distribution then turns it into expected points
per game. Season covariates (none in v1) are held at zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from xarray import DataTree

from dwar.model.data import AGE_CENTER, ModelData
from dwar.model.talent import LIKELIHOOD_PARAMS, link, scale_of, talent_from_center


def inverse_softplus(y: np.ndarray) -> np.ndarray:
    y = np.maximum(y, 1e-12)
    return y + np.log(-np.expm1(-y))


@dataclass(frozen=True)
class PositionDraws:
    """Posterior draws needed to project one position, thinned to `n` draws."""

    likelihood: str
    latent: np.ndarray         # (draws, player_seasons) latent talent
    talent: np.ndarray         # (draws, player_seasons) expected points per game
    beta1: np.ndarray          # (draws,)
    beta2: np.ndarray
    tau: np.ndarray
    params: dict[str, np.ndarray]  # likelihood parameters, (draws,) each
    zero_value: float
    nu_step: np.ndarray | None = None  # (draws,) Student-t steps when fitted with heavy tails

    @classmethod
    def from_posterior(
        cls, d: ModelData, posterior: DataTree, likelihood: str, draw_idx: np.ndarray
    ) -> PositionDraws:
        def take(name: str) -> np.ndarray:
            v = np.asarray(posterior["posterior"][name].values)
            return v.reshape(-1, *v.shape[2:])[draw_idx]

        if scale_of(likelihood) == "sqrt":
            latent = inverse_softplus(take("loc"))
        elif scale_of(likelihood) == "log":
            latent = np.log(take("loc"))
        else:
            latent = take("talent")
        return cls(
            likelihood=likelihood,
            latent=latent,
            talent=take("talent"),
            beta1=take("beta1"),
            beta2=take("beta2"),
            tau=take("tau"),
            params={n: take(n) for n in LIKELIHOOD_PARAMS[likelihood]},
            zero_value=d.zero_week_mean,
            nu_step=take("nu_step") if "nu_step" in posterior["posterior"] else None,
        )

    def age_effect(self, age: np.ndarray) -> np.ndarray:
        a = np.asarray(age) - AGE_CENTER
        return self.beta1[:, None] * a + self.beta2[:, None] * a**2

    def expected_points(self, latent: np.ndarray) -> np.ndarray:
        center = link(latent, self.likelihood, lib=np)
        params = {n: v[:, None] for n, v in self.params.items()} | {"zero_value": self.zero_value}
        return talent_from_center(self.likelihood, center, params, np)


def project_latent(
    draws: PositionDraws,
    ps_idx: np.ndarray,
    last_season: np.ndarray,
    last_age: np.ndarray,
    seasons: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Latent talent for each player in each of `seasons`, (draws, players,
    seasons). `ps_idx`, `last_season`, `last_age` describe each player's
    latest fitted season; a season at or before it keeps that season's
    latent talent (the projection never looks backward)."""
    latent = draws.latent[:, ps_idx]
    out = np.empty((latent.shape[0], len(ps_idx), len(seasons)))
    prev = last_season.astype(float)
    for j, season in enumerate(seasons):
        ahead = np.maximum(season - prev, 0.0)  # seasons since the last projected one
        age_prev = last_age + (prev - last_season)
        age_next = last_age + np.maximum(season - last_season, 0)
        z = (
            rng.standard_t(draws.nu_step[:, None], latent.shape) if draws.nu_step is not None
            else rng.standard_normal(latent.shape)
        )
        step = draws.tau[:, None] * np.sqrt(ahead) * z
        latent = latent + draws.age_effect(age_next) - draws.age_effect(age_prev) + step
        prev = np.maximum(prev, season)
        out[:, :, j] = latent
    return out


def project_talent(
    draws: PositionDraws,
    ps_idx: np.ndarray,
    last_season: np.ndarray,
    last_age: np.ndarray,
    seasons: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Expected points per game for each player in each of `seasons`,
    (draws, players, seasons); see `project_latent`."""
    latent = project_latent(draws, ps_idx, last_season, last_age, seasons, rng)
    out = draws.expected_points(latent.reshape(latent.shape[0], -1)).reshape(latent.shape)
    at_or_before = seasons[None, :] <= last_season[:, None]
    return np.where(at_or_before[None], draws.talent[:, ps_idx][:, :, None], out)
