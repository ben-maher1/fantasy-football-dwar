"""Carry each posterior draw of a player's talent forward season by season.

Projection follows the fitted model: each season the latent talent moves
along the position's age curve and takes a random-walk step of the fitted
size; the weekly-points distribution then turns it into expected points
per game. Situation factors enter as `shift` (the change in the level
factors' effect, latent scale) and `vol` (each step's size multiplier from
the range factors); without them a player's situation is held as it was.
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
    eta: np.ndarray | None = None    # (draws, k) level-factor effects
    kappa: np.ndarray | None = None  # (draws, j) range-factor effects on log step size

    @classmethod
    def from_posterior(
        cls, d: ModelData, posterior: DataTree, likelihood: str, draw_idx: np.ndarray
    ) -> PositionDraws:
        def take(name: str) -> np.ndarray:
            v = np.asarray(posterior["posterior"][name].values)
            return v.reshape(-1, *v.shape[2:])[draw_idx]

        def maybe(name: str) -> np.ndarray | None:
            return take(name) if name in posterior["posterior"] else None

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
            eta=maybe("eta"),
            kappa=maybe("kappa"),
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
    shift: np.ndarray | None = None,
    vol: np.ndarray | None = None,
) -> np.ndarray:
    """Latent talent for each player in each of `seasons`, (draws, players,
    seasons). `ps_idx`, `last_season`, `last_age` describe each player's
    latest fitted season; a season at or before it keeps that season's
    latent talent (the projection never looks backward). `shift` and `vol`
    are (draws, players, seasons): `shift[..., j]` is season j's level-factor
    effect minus the latest fitted season's (added to that season only,
    never carried into the walk), and `vol[..., j]` scales season j's step."""
    latent = draws.latent[:, ps_idx]
    out = np.empty((latent.shape[0], len(ps_idx), len(seasons)))
    prev = last_season.astype(float)
    for j, season in enumerate(seasons):
        ahead = np.maximum(season - prev, 0.0)  # seasons since the last projected one
        age_prev = last_age + (prev - last_season)
        age_next = last_age + np.maximum(season - last_season, 0)
        step = draws.tau[:, None] * np.sqrt(ahead) * rng.standard_normal(latent.shape)
        if vol is not None:
            step = step * vol[:, :, j]
        latent = latent + draws.age_effect(age_next) - draws.age_effect(age_prev) + step
        prev = np.maximum(prev, season)
        out[:, :, j] = latent if shift is None else latent + shift[:, :, j]
    return out


def project_talent(
    draws: PositionDraws,
    ps_idx: np.ndarray,
    last_season: np.ndarray,
    last_age: np.ndarray,
    seasons: np.ndarray,
    rng: np.random.Generator,
    shift: np.ndarray | None = None,
    vol: np.ndarray | None = None,
) -> np.ndarray:
    """Expected points per game for each player in each of `seasons`,
    (draws, players, seasons); see `project_latent`."""
    latent = project_latent(draws, ps_idx, last_season, last_age, seasons, rng, shift, vol)
    out = draws.expected_points(latent.reshape(latent.shape[0], -1)).reshape(latent.shape)
    at_or_before = seasons[None, :] <= last_season[:, None]
    return np.where(at_or_before[None], draws.talent[:, ps_idx][:, :, None], out)


def situation_effects(
    draws: PositionDraws, d: ModelData, ps_idx: np.ndarray, last_season: np.ndarray, season: int,
    level_next: np.ndarray | None, range_next: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """`shift` and `vol` (draws, players) for projecting one season ahead,
    from each player's factors in `season` (rows aligned with `ps_idx`, in
    the fit's column order). Players whose latest fitted season is `season`
    already carry its factors: shift 0, vol 1."""
    ahead = last_season < season
    shift = vol = None
    if draws.eta is not None and level_next is not None:
        delta = np.where(ahead[:, None], level_next - d.season_x[ps_idx], 0.0)
        shift = delta @ draws.eta.T  # (players, draws)
        shift = shift.T
    if draws.kappa is not None and range_next is not None:
        v = np.where(ahead[:, None], range_next, 0.0)
        vol = np.exp(v @ draws.kappa.T).T
    return shift, vol


def persistence(d: ModelData) -> np.ndarray:
    """Year-to-year correlation of each level factor within a player
    (consecutive seasons), floored at 0: how far a known situation carries
    into the following season."""
    k = d.season_x.shape[1]
    pairs = np.flatnonzero(~d.ps_is_first & (d.ps_gap == 1))
    if not k or len(pairs) < 3:
        return np.zeros(k)
    now, before = d.season_x[pairs], d.season_x[pairs - 1]
    rho = np.array([np.corrcoef(now[:, j], before[:, j])[0, 1] if now[:, j].std() and before[:, j].std() else 0.0
                    for j in range(k)])
    return np.clip(np.nan_to_num(rho), 0.0, 1.0)


def future_situation(
    draws: PositionDraws, d: ModelData, ps_idx: np.ndarray, last_season: np.ndarray, seasons: np.ndarray,
    level_next: np.ndarray | None, range_next: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """`shift` and `vol` (draws, players, seasons) for multi-season projection.
    The season after a player's latest fitted one uses his known situation
    (`level_next`, `range_next`); later seasons let level factors fade toward
    the position average at their historical persistence, and scale steps by
    the average range effect over the fit's later seasons (moves happen at
    the historical rate without assuming any one)."""
    n_draws, n_players = draws.latent.shape[0], len(ps_idx)
    ahead = seasons[None, :] - last_season[:, None]  # (players, seasons); 1 = next season
    shift = vol = None
    if draws.eta is not None and level_next is not None:
        rho = persistence(d)
        k = np.maximum(ahead - 1, 0)[:, :, None]
        level = level_next[:, None, :] * rho[None, None, :] ** k  # (players, seasons, factors)
        delta = np.where((ahead > 0)[:, :, None], level - d.season_x[ps_idx][:, None, :], 0.0)
        shift = np.einsum("psf,df->dps", delta, draws.eta)
    if draws.kappa is not None and range_next is not None:
        later = ~d.ps_is_first
        typical = np.exp(d.season_vol_x[later] @ draws.kappa.T).mean(axis=0) if later.any() else np.ones(n_draws)
        known = np.exp(range_next @ draws.kappa.T).T  # (draws, players)
        vol = np.where(ahead[None] == 1, known[:, :, None], typical[:, None, None])
        vol = np.where(ahead[None] <= 0, 1.0, vol)
        vol = np.broadcast_to(vol, (n_draws, n_players, len(seasons)))
    return shift, vol
