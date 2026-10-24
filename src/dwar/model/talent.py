"""PyMC model of player-season talent (fantasy points per game played).

    fpts[i,s,w]    ~ StudentT(nu, talent[i,s], sigma)
    talent[i,s]    = level[i,s] + age_effect(age[i,s]) + season_x @ eta
    age_effect(a)  = beta1*(a-26) + beta2*(a-26)^2
    level[i,s]     = level[i,s-1] + tau*sqrt(gap)*z[i,s]
    talent[i,first] ~ Normal(prior_mean_i, prior_sd_i)   (before season covariates)

The prior is placed on the first season's talent, not its level, so the
prior means (rookie and veteran) are scoring rates at the age the player
entered and do not trade off against the age curve.

The rookie prior mean is draft capital plus prospect covariates (combine in
v1). Players whose careers began before the fitting window get a separate
veteran prior instead. Levels are non-centered (built from standard-normal
steps) to avoid random-walk funnels.
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from dwar.model.data import ModelData

COVARIATE_PRIOR_SD = 1.0

# Global parameters checked by the parameter-recovery test.
GLOBAL_PARAMS = (
    "alpha0", "alpha_pick", "alpha_undrafted", "sigma_rookie",
    "mu_vet", "sigma_vet", "tau", "beta1", "beta2", "sigma", "nu",
)


def _shrunk_coefs(name: str, dim: str) -> pt.TensorVariable:
    """Coefficients on standardized covariates with a Normal(0, 1) prior
    (points per SD), so a useless covariate stays near zero. A learned shared
    scale was tried and made a funnel the sampler could not mix well."""
    return pm.Normal(name, 0.0, COVARIATE_PRIOR_SD, dims=dim)


def build_talent_model(d: ModelData) -> pm.Model:
    coords = {
        "player": list(d.player_ids),
        "player_season": np.arange(d.n_ps),
        "obs": np.arange(d.n_obs),
        "prospect_cov": list(d.prospect_x_names),
        "season_cov": list(d.season_x_names),
    }
    with pm.Model(coords=coords) as model:
        # Rookie prior: draft capital + prospect covariates.
        alpha0 = pm.Normal("alpha0", 8.0, 5.0)
        alpha_pick = pm.Normal("alpha_pick", 0.0, 3.0)
        alpha_undrafted = pm.Normal("alpha_undrafted", 0.0, 3.0)
        sigma_rookie = pm.HalfNormal("sigma_rookie", 5.0)
        rookie_mean = alpha0 + alpha_pick * d.log_pick_z + alpha_undrafted * d.undrafted
        if d.prospect_x.shape[1]:
            gamma = _shrunk_coefs("gamma", "prospect_cov")
            rookie_mean = rookie_mean + pt.dot(d.prospect_x, gamma)

        # Age curve, quadratic around 26.
        beta1 = pm.Normal("beta1", 0.0, 1.0)
        beta2 = pm.Normal("beta2", 0.0, 0.2)
        age_effect = beta1 * d.ps_age_c + beta2 * d.ps_age_c**2

        # First-season talent per player. Rookies are non-centered (little data
        # each); veterans are centered, because their long careers pin their
        # starting level and a non-centered form mixed poorly on real data.
        vets = np.flatnonzero(d.pre_window > 0)
        rookies = np.flatnonzero(d.pre_window == 0)
        start = pt.zeros(d.n_players)
        if len(rookies):
            rookie_z = pm.Normal("rookie_z", 0.0, 1.0, shape=len(rookies))
            start = pt.set_subtensor(start[rookies], rookie_mean[rookies] + sigma_rookie * rookie_z)
        mu_vet = pm.Normal("mu_vet", 8.0, 5.0)
        sigma_vet = pm.HalfNormal("sigma_vet", 8.0)
        if len(vets):
            vet_start = pm.Normal("vet_start", mu_vet, sigma_vet, shape=len(vets))
            start = pt.set_subtensor(start[vets], vet_start)

        # Random walk across a player's seasons from standardized steps.
        tau = pm.HalfNormal("tau", 3.0)
        later = np.flatnonzero(~d.ps_is_first)
        steps = start[d.ps_player] - age_effect  # first seasons: level = start talent - age effect
        if len(later):
            step_z = pm.Normal("step_z", 0.0, 1.0, shape=len(later))
            steps = pt.set_subtensor(steps[later], tau * np.sqrt(np.maximum(d.ps_gap[later], 1.0)) * step_z)
        csum = pt.cumsum(steps)
        level = csum - csum[d.ps_start] + steps[d.ps_start]

        talent_expr = level + age_effect
        if d.season_x.shape[1]:
            eta = _shrunk_coefs("eta", "season_cov")
            talent_expr = talent_expr + pt.dot(d.season_x, eta)
        talent = pm.Deterministic("talent", talent_expr, dims="player_season")

        # Heavy-tailed weekly noise.
        sigma = pm.HalfNormal("sigma", 10.0)
        nu = pm.Gamma("nu", 2.0, 0.1)
        pm.StudentT("fpts", nu=nu, mu=talent[d.obs_ps], sigma=sigma, observed=d.y, dims="obs")
    return model
