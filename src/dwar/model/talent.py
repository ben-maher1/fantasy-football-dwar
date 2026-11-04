"""PyMC model of player-season talent: expected fantasy points per game played.

    latent[i,s]    = level[i,s] + age_effect(age[i,s]) + season_x @ eta
    age_effect(a)  = beta1*(a-26) + beta2*(a-26)^2         (quadratic, default)
                   = beta1*min(a-30, 0) + beta2*max(a-30, 0)   (hinge: two straight lines meeting at 30)
    level[i,s]     = level[i,s-1] + tau*exp(season_vol_x @ kappa)*sqrt(gap)*z[i,s]
    latent[i,first] ~ Normal(prior_mean_i, prior_sd_i)   (before season covariates)
    center[i,s]    = link(latent[i,s])
    fpts[i,s,w]    ~ weekly distribution around center[i,s]
    talent[i,s]    = expected weekly points under that distribution

`season_x` holds level factors (situation: QB quality, scheme, target
competition) and `season_vol_x` holds 0/1 range factors (changed teams, new
play-caller, new QB) that make that season's step larger: a changed
situation is a less predictable one. Both slots may be empty.

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
- sqrt_hurdle_log: the same weekly model, but loc = exp(latent), so the
  latent scale is log(loc). Every additive effect on it (the yearly
  random-walk step, the age curve, the priors) becomes proportional: a step
  of 0.1 moves loc about 10% and points per game about 20%, whatever the
  player's level. With the softplus link a step moves loc by a fixed amount,
  so high scorers (QBs above all) drift far more in points than real
  players do; step 5 validation found QB forecasts too wide for that reason.
- studentt: symmetric heavy tails on the points scale, as first specified;
  center = talent. Kept as the baseline it was compared against.

Hurdle-gamma, ExGaussian, and a Student-t on sqrt(points + 3) without a zero
part were also tried on real WRs (2006-2024 as of 2025-09-01); see PR #3 for
the comparison.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from dwar.model.data import ModelData

LIKELIHOODS = ("sqrt_hurdle", "sqrt_hurdle_log", "studentt")
HURDLE = ("sqrt_hurdle", "sqrt_hurdle_log")
AGE_KNOT = 30.0  # hinge age curve: where the slope may change
KAPPA_SD = 0.5  # log step-size change per range flag: 0.4 is a step about 50% larger
SQRT_PIVOT = 3.0  # sqrt-scale location (about 9 ppg) at which the zero-week log-odds are h0

# Global parameters checked by the parameter-recovery test.
LATENT_PARAMS = (
    "alpha0", "alpha_pick", "alpha_undrafted", "sigma_rookie",
    "mu_vet", "sigma_vet", "tau", "beta1", "beta2",
)
LIKELIHOOD_PARAMS = {
    "sqrt_hurdle": ("sigma", "nu", "h0", "h1"),
    "sqrt_hurdle_log": ("sigma", "nu", "h0", "h1"),
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
    # log(loc): loc 3 (about 9 ppg) is 1.1; 0.1 is about a 10% change in loc.
    "log": {
        "alpha0": (1.0, 0.5), "alpha_pick": 0.3, "alpha_undrafted": 0.3, "sigma_rookie": 0.5,
        "mu_vet": (1.0, 0.5), "sigma_vet": 0.7, "tau": 0.3, "beta1": 0.1, "beta2": 0.02, "covariate": 0.1,
    },
}


def _check(likelihood: str) -> None:
    if likelihood not in LIKELIHOODS:
        raise ValueError(f"unknown likelihood {likelihood!r}; choose from {LIKELIHOODS}")


def scale_of(likelihood: str) -> str:
    _check(likelihood)
    return {"sqrt_hurdle": "sqrt", "sqrt_hurdle_log": "log"}.get(likelihood, "points")


def for_position(likelihood: str | Mapping[str, str], position: str) -> str:
    """One likelihood for every position, or a position -> likelihood map."""
    return likelihood if isinstance(likelihood, str) else likelihood[position]


def is_hurdle(likelihood: str) -> bool:
    _check(likelihood)
    return likelihood in HURDLE


def softplus(x, lib=pt):
    return lib.log1p(lib.exp(-lib.abs(x))) + lib.maximum(x, 0)


def link(latent, likelihood: str, lib=pt):
    """Latent scale -> the weekly distribution's center. Softplus keeps the
    sqrt-scale location positive while staying close to identity for any
    useful player; exp makes latent effects proportional."""
    scale = scale_of(likelihood)
    if scale == "sqrt":
        return softplus(latent, lib)
    return lib.exp(latent) if scale == "log" else latent


def nonzero_prob(loc, h0, h1, lib=pt):
    """Hurdle likelihoods: probability a week scores above zero."""
    return 1.0 / (1.0 + lib.exp(-(h0 + h1 * (loc - SQRT_PIVOT))))


def talent_from_center(likelihood: str, center, params: dict, lib=np):
    """Expected weekly points given the distribution's center."""
    if is_hurdle(likelihood):
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
    if is_hurdle(likelihood):
        p = nonzero_prob(center, params["h0"], params["h1"], np)
        return np.where(rng.random(center.shape) < p, (center + t) ** 2, params.get("zero_value", 0.0))
    return center + t


def _shrunk_coefs(name: str, dim: str, sd: float) -> pt.TensorVariable:
    """Coefficients on standardized covariates with a fixed Normal(0, sd)
    prior (about 1 point per SD), so a useless covariate stays near zero. A
    learned shared scale was tried and made a funnel the sampler could not
    mix well."""
    return pm.Normal(name, 0.0, sd, dims=dim)


def age_curve_effect(age_c, beta1, beta2, curve: str = "quadratic"):
    """Age effect from age minus AGE_CENTER (26); works on numpy or tensors."""
    if curve == "hinge":
        a = age_c - (AGE_KNOT - 26.0)
        return beta1 * (a - abs(a)) / 2 + beta2 * (a + abs(a)) / 2
    return beta1 * age_c + beta2 * age_c**2


GROWTH_YEARS = (2, 3)  # career years that add a growth step to the rookie level, kept from then on
GROWTH_PRIOR = {"sqrt": (0.3, 0.15), "log": (0.1, 0.05), "points": (1.0, 0.5)}  # (step, draft slope) sds
EFFICIENCY_PRIOR = {"sqrt": 0.3, "log": 0.1}  # sd of the per-player efficiency scale


def growth_design(career_year: np.ndarray, log_pick_z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(rows, len(GROWTH_YEARS)) indicators that a season is in career year
    2 or later and 3 or later, and the same times the player's standardized
    log pick. The growth term is ind @ xi + (ind * pick) @ zeta: year 2 adds
    one step, year 3 a second, and both persist after, so draft capital can
    change how much a player is expected to grow. (A further step at year 4
    was tried; it was not identified apart from the random walk.)"""
    cy = np.asarray(career_year)
    ind = np.column_stack([cy >= g for g in GROWTH_YEARS]).astype(float)
    return ind, ind * np.asarray(log_pick_z)[:, None]


def has_opportunity(d: ModelData) -> bool:
    return d.x is not None and bool(np.isfinite(d.x).any())


def build_talent_model(
    d: ModelData, likelihood: str = "sqrt_hurdle", age_curve: str = "quadratic",
    round1: bool = False, growth: bool = False, opportunity: bool = False,
    opportunity_years: int = 0,
) -> pm.Model:
    """With `round1`, first-round picks get their own step in the rookie
    prior on top of log pick. With `growth`, career years 2 and 3 each add
    a lasting growth step whose size may depend on draft capital
    (`growth_design`). With `opportunity` (needs `d.x`),
    talent is opportunity talent plus a per-player efficiency: weekly
    opportunity points observe the first, weekly points the sum. With
    `opportunity_years` > 0, opportunity points are used only in a player's
    first that many career years (not for pre-window veterans), and only
    players with such weeks get an efficiency term."""
    use_opp = opportunity and is_hurdle(likelihood) and has_opportunity(d)
    if use_opp:
        seen = np.isfinite(d.x)
        if opportunity_years > 0:
            cy = np.zeros(d.n_ps) if d.ps_career_year is None else d.ps_career_year
            cy_obs = cy[d.obs_ps]
            seen &= (cy_obs >= 1) & (cy_obs <= opportunity_years)
        seen = np.flatnonzero(seen)
        has_eff = np.zeros(d.n_players, dtype=bool)
        has_eff[d.ps_player[d.obs_ps[seen]]] = True
        use_opp = len(seen) > 0
    coords = {
        "player": list(d.player_ids),
        "player_season": np.arange(d.n_ps),
        "obs": np.arange(d.n_obs),
        "prospect_cov": list(d.prospect_x_names),
        "season_cov": list(d.season_x_names),
        "season_vol_cov": list(d.season_vol_x_names),
        "growth_year": [str(g) for g in GROWTH_YEARS],
    }
    pr = PRIORS[scale_of(likelihood)]
    latent_scale = scale_of(likelihood)
    with pm.Model(coords=coords) as model:
        # Rookie prior: draft capital + prospect covariates.
        alpha0 = pm.Normal("alpha0", *pr["alpha0"])
        alpha_pick = pm.Normal("alpha_pick", 0.0, pr["alpha_pick"])
        alpha_undrafted = pm.Normal("alpha_undrafted", 0.0, pr["alpha_undrafted"])
        sigma_rookie = pm.HalfNormal("sigma_rookie", pr["sigma_rookie"])
        rookie_mean = alpha0 + alpha_pick * d.log_pick_z + alpha_undrafted * d.undrafted
        if round1 and d.round1 is not None and d.round1.any():
            # The first-round flag minus its straight-line fit on log pick:
            # the same model as the raw flag, but alpha_round1 no longer
            # trades off against alpha_pick (raw, the pair mixed slowly).
            X = np.column_stack([np.ones(d.n_players), d.log_pick_z, d.undrafted])
            r1 = d.round1 - X @ np.linalg.lstsq(X, d.round1, rcond=None)[0]
            alpha_round1 = pm.Normal("alpha_round1", 0.0, pr["alpha_pick"])
            rookie_mean = rookie_mean + alpha_round1 * r1
        if d.prospect_x.shape[1]:
            gamma = _shrunk_coefs("gamma", "prospect_cov", pr["covariate"])
            rookie_mean = rookie_mean + pt.dot(d.prospect_x, gamma)

        # Age curve: quadratic around 26, or a hinge at AGE_KNOT with slopes
        # age_slope_pre and age_slope_post (named apart so a saved fit says
        # which curve it used).
        if age_curve == "hinge":
            beta1 = pm.Normal("age_slope_pre", 0.0, pr["beta1"])
            beta2 = pm.Normal("age_slope_post", 0.0, pr["beta1"])
        else:
            beta1 = pm.Normal("beta1", 0.0, pr["beta1"])
            beta2 = pm.Normal("beta2", 0.0, pr["beta2"])
        age_effect = age_curve_effect(d.ps_age_c, beta1, beta2, age_curve)

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
            scale = tau * np.sqrt(np.maximum(d.ps_gap[later], 1.0))
            if d.season_vol_x.shape[1]:
                kappa = pm.Normal("kappa", 0.0, KAPPA_SD, dims="season_vol_cov")
                scale = scale * pt.exp(pt.dot(d.season_vol_x[later], kappa))
            steps = pt.set_subtensor(steps[later], scale * step_z)
        csum = pt.cumsum(steps)
        level = csum - csum[d.ps_start] + steps[d.ps_start]

        latent = level + age_effect
        if d.season_x.shape[1]:
            eta = _shrunk_coefs("eta", "season_cov", pr["covariate"])
            latent = latent + pt.dot(d.season_x, eta)
        if growth and d.ps_career_year is not None:
            ind, ind_pick = growth_design(d.ps_career_year, d.log_pick_z[d.ps_player])
            sd_shift, sd_slope = GROWTH_PRIOR[latent_scale]
            xi = pm.Normal("growth", 0.0, sd_shift, dims="growth_year")
            zeta = pm.Normal("growth_pick", 0.0, sd_slope, dims="growth_year")
            latent = latent + pt.dot(ind, xi) + pt.dot(ind_pick, zeta)
        latent_opp = latent
        if use_opp:
            # Efficiency: how far a player's scoring runs above his
            # opportunity, one value per career, shrunk toward zero.
            sigma_eff = pm.HalfNormal("sigma_eff", EFFICIENCY_PRIOR[latent_scale])
            eff_z = pm.Normal("eff_z", 0.0, 1.0, dims="player")
            latent = latent + (sigma_eff * eff_z * has_eff)[d.ps_player]
        center = link(latent, likelihood)

        if is_hurdle(likelihood):
            sigma = pm.HalfNormal("sigma", 2.0)
            nu = pm.Deterministic("nu", 2.0 + pm.Gamma("nu_minus_2", 2.0, 0.1))
            h0 = pm.Normal("h0", 1.0, 1.5)
            h1 = pm.HalfNormal("h1", 3.0)  # zero weeks get rarer as talent rises
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
            if use_opp:
                # Weekly opportunity points on the sqrt scale, centered on
                # the opportunity part of the same location (plus an offset,
                # since points and opportunity points differ on average).
                loc_opp = link(latent_opp, likelihood)
                delta_x = pm.Normal("delta_x", 0.0, 0.5)
                sigma_x = pm.HalfNormal("sigma_x", 1.0)
                nu_x = pm.Deterministic("nu_x", 2.0 + pm.Gamma("nu_x_minus_2", 2.0, 0.1))
                pm.Potential("xfpts", pt.sum(pm.logp(
                    pm.StudentT.dist(nu=nu_x, mu=loc_opp[d.obs_ps[seen]] + delta_x, sigma=sigma_x),
                    np.sqrt(np.maximum(d.x[seen], 0.0)),
                )))
        else:
            talent = pm.Deterministic("talent", center, dims="player_season")
            sigma = pm.HalfNormal("sigma", 10.0)
            nu = pm.Gamma("nu", 2.0, 0.1)
            pm.StudentT("fpts", nu=nu, mu=talent[d.obs_ps], sigma=sigma, observed=d.y, dims="obs")
    return model
