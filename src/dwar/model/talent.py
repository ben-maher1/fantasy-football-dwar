"""PyMC model of player-season talent: expected fantasy points per game played.

    latent[i,s]    = level[i,s] + age_effect(age[i,s]) + season_x @ eta
    age_effect(a)  = beta1*(a-26) + beta2*(a-26)^2
    level[i,s]     = level[i,s-1] + tau*sqrt(gap)*z[i,s]
    latent[i,first] ~ Normal(prior_mean_i, prior_sd_i)   (before season covariates)
    center[i,s]    = link(latent[i,s])
    fpts[i,s,w]    ~ weekly distribution around center[i,s]
    talent[i,s]    = expected weekly points under that distribution

The prior is placed on the first season's latent talent, not its level, so
the prior means (rookie and veteran) are scoring rates at the age the player
entered and do not trade off against the age curve. The rookie prior mean is
draft capital plus prospect covariates; careers that began before the fitting
window get a separate veteran prior.

Weekly distributions (config `likelihood`):

- sqrt_hurdle (default): a week scores at or below zero with probability
  1 - p, where logit(p) = h0 + h1*(loc - 3). Otherwise points = Z^2 with
  Z ~ StudentT(nu, loc, sigma), so the latent scale is sqrt(points) and
  loc = softplus(latent). Matches the quarter of WR weeks that are zero and
  the right skew of the rest. A week at or below zero is worth `zero_value`,
  the position's observed mean over such weeks (about -0.6 for QBs, whose
  bad weeks go negative; about 0 elsewhere). Expected points:
  talent = p * (loc^2 + sigma^2 * nu / (nu - 2)) + (1 - p) * zero_value.
- studentt: symmetric heavy tails on the points scale, as first specified;
  center = talent. Kept as the baseline it was compared against.

Hurdle-gamma, ExGaussian, and a Student-t on sqrt(points + 3) without a zero
part were also tried on real WRs (2006-2024 as of 2025-09-01); see PR #3 for
the comparison.
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from dwar.model.data import ModelData

LIKELIHOODS = ("sqrt_hurdle", "studentt")
SQRT_PIVOT = 3.0  # sqrt-scale location (about 9 ppg) at which the zero-week log-odds are h0

# Global parameters checked by the parameter-recovery test.
LATENT_PARAMS = (
    "alpha0", "alpha_pick", "alpha_undrafted", "sigma_rookie",
    "mu_vet", "sigma_vet", "tau", "beta1", "beta2",
)
LIKELIHOOD_PARAMS = {
    "sqrt_hurdle": ("sigma", "nu", "h0", "h1"),
    "studentt": ("sigma", "nu"),
}

# Prior locations and scales for the latent talent model, on each
# likelihood's latent scale (8 ppg is about 2.8 on the sqrt scale).
PRIORS = {
    "points": {
        "alpha0": (8.0, 5.0), "alpha_pick": 3.0, "alpha_undrafted": 3.0, "sigma_rookie": 5.0,
        "mu_vet": (8.0, 5.0), "sigma_vet": 8.0, "tau": 3.0, "beta1": 1.0, "beta2": 0.2, "covariate": 1.0,
    },
    "sqrt": {
        "alpha0": (3.0, 1.5), "alpha_pick": 1.0, "alpha_undrafted": 1.0, "sigma_rookie": 1.5,
        "mu_vet": (3.0, 1.5), "sigma_vet": 2.0, "tau": 1.0, "beta1": 0.3, "beta2": 0.06, "covariate": 0.3,
    },
}


def _check(likelihood: str) -> None:
    if likelihood not in LIKELIHOODS:
        raise ValueError(f"unknown likelihood {likelihood!r}; choose from {LIKELIHOODS}")


def scale_of(likelihood: str) -> str:
    _check(likelihood)
    return "sqrt" if likelihood == "sqrt_hurdle" else "points"


def softplus(x, lib=pt):
    return lib.log1p(lib.exp(-lib.abs(x))) + lib.maximum(x, 0)


def link(latent, likelihood: str, lib=pt):
    """Latent scale -> the weekly distribution's center. Softplus keeps the
    sqrt-scale location positive while staying close to identity for any
    useful player."""
    return softplus(latent, lib) if scale_of(likelihood) == "sqrt" else latent


def nonzero_prob(loc, h0, h1, lib=pt):
    """sqrt_hurdle: probability a week scores above zero."""
    return 1.0 / (1.0 + lib.exp(-(h0 + h1 * (loc - SQRT_PIVOT))))


def talent_from_center(likelihood: str, center, params: dict, lib=np):
    """Expected weekly points given the distribution's center."""
    if scale_of(likelihood) == "sqrt":
        nu = params["nu"]
        p = nonzero_prob(center, params["h0"], params["h1"], lib)
        return p * (center**2 + params["sigma"] ** 2 * nu / (nu - 2.0)) + (1 - p) * params.get("zero_value", 0.0)
    return center


def simulate_weeks(likelihood: str, center: np.ndarray, params: dict, rng: np.random.Generator) -> np.ndarray:
    """Draw weekly points given each week's distribution center and the
    likelihood's parameters. Used by the posterior predictive check and the
    tests."""
    _check(likelihood)
    t = params["sigma"] * rng.standard_t(params["nu"], center.shape)
    if likelihood == "sqrt_hurdle":
        p = nonzero_prob(center, params["h0"], params["h1"], np)
        return np.where(rng.random(center.shape) < p, (center + t) ** 2, params.get("zero_value", 0.0))
    return center + t


def _shrunk_coefs(name: str, dim: str, sd: float) -> pt.TensorVariable:
    """Coefficients on standardized covariates with a fixed Normal(0, sd)
    prior (about 1 point per SD), so a useless covariate stays near zero. A
    learned shared scale was tried and made a funnel the sampler could not
    mix well."""
    return pm.Normal(name, 0.0, sd, dims=dim)


def build_talent_model(d: ModelData, likelihood: str = "sqrt_hurdle") -> pm.Model:
    coords = {
        "player": list(d.player_ids),
        "player_season": np.arange(d.n_ps),
        "obs": np.arange(d.n_obs),
        "prospect_cov": list(d.prospect_x_names),
        "season_cov": list(d.season_x_names),
    }
    pr = PRIORS[scale_of(likelihood)]
    with pm.Model(coords=coords) as model:
        # Rookie prior: draft capital + prospect covariates.
        alpha0 = pm.Normal("alpha0", *pr["alpha0"])
        alpha_pick = pm.Normal("alpha_pick", 0.0, pr["alpha_pick"])
        alpha_undrafted = pm.Normal("alpha_undrafted", 0.0, pr["alpha_undrafted"])
        sigma_rookie = pm.HalfNormal("sigma_rookie", pr["sigma_rookie"])
        rookie_mean = alpha0 + alpha_pick * d.log_pick_z + alpha_undrafted * d.undrafted
        if d.prospect_x.shape[1]:
            gamma = _shrunk_coefs("gamma", "prospect_cov", pr["covariate"])
            rookie_mean = rookie_mean + pt.dot(d.prospect_x, gamma)

        # Age curve, quadratic around 26.
        beta1 = pm.Normal("beta1", 0.0, pr["beta1"])
        beta2 = pm.Normal("beta2", 0.0, pr["beta2"])
        age_effect = beta1 * d.ps_age_c + beta2 * d.ps_age_c**2

        # First-season latent talent per player. Rookies are non-centered
        # (little data each); veterans are centered, because their long
        # careers pin their starting level and a non-centered form mixed
        # poorly on real data.
        vets = np.flatnonzero(d.pre_window > 0)
        rookies = np.flatnonzero(d.pre_window == 0)
        start = pt.zeros(d.n_players)
        if len(rookies):
            rookie_z = pm.Normal("rookie_z", 0.0, 1.0, shape=len(rookies))
            start = pt.set_subtensor(start[rookies], rookie_mean[rookies] + sigma_rookie * rookie_z)
        mu_vet = pm.Normal("mu_vet", *pr["mu_vet"])
        sigma_vet = pm.HalfNormal("sigma_vet", pr["sigma_vet"])
        if len(vets):
            vet_start = pm.Normal("vet_start", mu_vet, sigma_vet, shape=len(vets))
            start = pt.set_subtensor(start[vets], vet_start)

        # Random walk across a player's seasons from standardized steps.
        tau = pm.HalfNormal("tau", pr["tau"])
        later = np.flatnonzero(~d.ps_is_first)
        steps = start[d.ps_player] - age_effect  # first seasons: level = start - age effect
        if len(later):
            step_z = pm.Normal("step_z", 0.0, 1.0, shape=len(later))
            steps = pt.set_subtensor(steps[later], tau * np.sqrt(np.maximum(d.ps_gap[later], 1.0)) * step_z)
        csum = pt.cumsum(steps)
        level = csum - csum[d.ps_start] + steps[d.ps_start]

        latent = level + age_effect
        if d.season_x.shape[1]:
            eta = _shrunk_coefs("eta", "season_cov", pr["covariate"])
            latent = latent + pt.dot(d.season_x, eta)
        center = link(latent, likelihood)

        if likelihood == "sqrt_hurdle":
            sigma = pm.HalfNormal("sigma", 2.0)
            nu = pm.Deterministic("nu", 2.0 + pm.Gamma("nu_minus_2", 2.0, 0.1))
            h0 = pm.Normal("h0", 1.0, 1.5)
            h1 = pm.Normal("h1", 0.0, 1.5)
            loc = pm.Deterministic("loc", center, dims="player_season")
            params = {"sigma": sigma, "nu": nu, "h0": h0, "h1": h1, "zero_value": d.zero_week_mean}
            pm.Deterministic("talent", talent_from_center(likelihood, loc, params, pt), dims="player_season")
            pos = d.y > 0
            pm.Bernoulli("scored", p=nonzero_prob(loc[d.obs_ps], h0, h1), observed=pos.astype(np.int64), dims="obs")
            # Scoring weeks: y = Z^2 with Z ~ StudentT(nu, loc, sigma). The
            # density of y sums both roots +-sqrt(y); the Jacobian 1/(2 sqrt y)
            # is constant in the parameters and dropped.
            z = np.sqrt(d.y[pos])
            t = pm.StudentT.dist(nu=nu, mu=loc[d.obs_ps[pos]], sigma=sigma)
            pm.Potential("fpts", pt.sum(pt.logaddexp(pm.logp(t, z), pm.logp(t, -z))))
        else:
            talent = pm.Deterministic("talent", center, dims="player_season")
            sigma = pm.HalfNormal("sigma", 10.0)
            nu = pm.Gamma("nu", 2.0, 0.1)
            pm.StudentT("fpts", nu=nu, mu=talent[d.obs_ps], sigma=sigma, observed=d.y, dims="obs")
    return model
