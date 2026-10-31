"""Sample the talent model, gate it on diagnostics, and write outputs.

A failed fit writes only `diagnostics.json` (marked failed) and raises
`FitFailed`; nothing else is saved.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import nutpie
import polars as pl
import xarray as xr
from xarray import DataTree

from dwar.config import ModelConfig
from dwar.features.player_tables import PlayerTables
from dwar.features.situation import Covariates
from dwar.model.data import ModelData, build_model_data
from dwar.model.diagnostics import Diagnostics, FitFailed, check_fit
from dwar.model.talent import LIKELIHOOD_PARAMS, build_talent_model, is_hurdle, simulate_weeks

log = logging.getLogger(__name__)

PERCENTILES = (5, 25, 50, 75, 95)
PPC_DRAWS = 200
BIG_WEEK = 25.0


@dataclass(frozen=True)
class FitResult:
    data: ModelData
    posterior: DataTree
    diagnostics: Diagnostics
    talent: pl.DataFrame


# Started at zero instead of a random point: with a log-scale latent, a
# random age-curve start (beta2 near +-2 at age 41) puts exp() past overflow.
UNJITTERED = ("beta1", "beta2", "age_slope_pre", "age_slope_post")


def sample(d: ModelData, cfg: ModelConfig) -> DataTree:
    model = build_talent_model(d, cfg.likelihood_for(d.position), cfg.age_curve.get(d.position, "quadratic"))
    jitter = {rv for rv in model.free_RVs if rv.name not in UNJITTERED}
    compiled = nutpie.compile_pymc_model(model, jitter_rvs=jitter)
    s = cfg.sampler
    return nutpie.sample(
        compiled, draws=s.draws, tune=s.tune, chains=s.chains, seed=cfg.seed, progress_bar=False,
        save_warmup=False,
    )


def _draws(posterior: DataTree, name: str) -> np.ndarray:
    """(chain*draw, ...) array of one posterior variable."""
    values = np.asarray(posterior["posterior"][name].values)
    return values.reshape(-1, *values.shape[2:])


def talent_summary(d: ModelData, posterior: DataTree) -> pl.DataFrame:
    talent = _draws(posterior, "talent")
    q = np.percentile(talent, PERCENTILES, axis=0)
    return d.ps_keys.with_columns(
        talent_mean=pl.Series(talent.mean(axis=0)),
        **{f"talent_p{p:02d}": pl.Series(q[k]) for k, p in enumerate(PERCENTILES)},
    )


def ppc_stats(y: np.ndarray) -> dict[str, np.ndarray]:
    """Summary statistics of weekly points; `y` may carry leading draw axes."""
    q10, q50, q90 = np.percentile(y, [10, 50, 90], axis=-1)
    return {
        "mean": y.mean(axis=-1),
        "sd": y.std(axis=-1),
        "share_zero_or_less": (y <= 0).mean(axis=-1),
        f"share_ge_{BIG_WEEK:g}": (y >= BIG_WEEK).mean(axis=-1),
        "p10": q10,
        "p50": q50,
        "p90": q90,
    }


def posterior_predictive_check(d: ModelData, posterior: DataTree, likelihood: str, seed: int) -> dict:
    """Observed weekly-points statistics vs the 90% range of the same
    statistics over simulated seasons. `fits` is true when observed falls
    inside that range."""
    rng = np.random.default_rng(seed)
    talent = _draws(posterior, "loc" if is_hurdle(likelihood) else "talent")
    params = {n: _draws(posterior, n) for n in LIKELIHOOD_PARAMS[likelihood]}
    picks = rng.choice(len(talent), size=min(PPC_DRAWS, len(talent)), replace=False)
    sims = np.stack([
        simulate_weeks(
            likelihood, talent[k][d.obs_ps],
            {"zero_value": d.zero_week_mean} | {n: v[k] for n, v in params.items()}, rng,
        )
        for k in picks
    ])
    observed, simulated = ppc_stats(d.y), ppc_stats(sims)
    out = {}
    for name, obs in observed.items():
        lo, hi = np.percentile(simulated[name], [5, 95])
        out[name] = {
            "observed": float(obs), "sim_p05": float(lo), "sim_p95": float(hi), "fits": bool(lo <= obs <= hi),
        }
    return out


def fit(d: ModelData, cfg: ModelConfig, out_dir: Path, meta: dict | None = None) -> FitResult:
    """Sample, check, and write outputs under `out_dir`. Raises FitFailed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    posterior = sample(d, cfg)
    runtime = time.perf_counter() - start

    diag = check_fit(posterior, cfg.diagnostics)
    diag.extra = {
        "runtime_s": round(runtime, 1),
        "seed": cfg.seed,
        "sampler": cfg.sampler.model_dump(),
        "position": d.position,
        "likelihood": cfg.likelihood_for(d.position),
        "n_obs": d.n_obs,
        "n_player_seasons": d.n_ps,
        "n_players": d.n_players,
        "prospect_covariates": list(d.prospect_x_names),
        "season_covariates": list(d.season_x_names),
        "versions": {p: version(p) for p in ("pymc", "nutpie", "arviz", "nflreadpy")},
        **(meta or {}),
    }
    if diag.passed:
        diag.extra["posterior_predictive"] = posterior_predictive_check(d, posterior, cfg.likelihood_for(d.position), cfg.seed)

    name = d.position.lower()
    (out_dir / f"{name}_diagnostics.json").write_text(json.dumps(diag.to_dict(), indent=2, default=str))
    if not diag.passed:
        log.error("%s fit failed: %s", d.position, "; ".join(diag.failures))
        raise FitFailed(diag)

    posterior.to_netcdf(out_dir / f"{name}.nc")
    talent = talent_summary(d, posterior)
    talent.write_parquet(out_dir / f"{name}_talent.parquet")
    return FitResult(data=d, posterior=posterior, diagnostics=diag, talent=talent)


def position_data(
    tables: PlayerTables, position: str, cfg: ModelConfig, cov: dict[str, Covariates] | None = None,
) -> ModelData:
    """One position's model data, with its situation factors if any."""
    c = (cov or {}).get(position)
    return build_model_data(
        tables, position, cfg.first_season, min_plays=cfg.min_plays.get(position, 0),
        season_covariates=c.level if c is not None and c.level.width > 2 else None,
        season_range_covariates=c.range if c is not None and c.range.width > 2 else None,
    )


def fit_positions(
    tables: PlayerTables, cfg: ModelConfig, out_dir: Path, meta: dict | None = None,
    cov: dict[str, Covariates] | None = None,
) -> dict[str, FitResult]:
    """Fit every configured position. All positions are attempted so each
    writes its diagnostics; if any fails, raises FitFailed for the first
    failure after the rest have run (the run is failed as a whole)."""
    results, failures = {}, []
    for position in cfg.positions:
        d = position_data(tables, position, cfg, cov)
        try:
            results[position] = fit(d, cfg, out_dir, meta)
        except FitFailed as e:
            failures.append(e)
    if failures:
        raise failures[0]
    return results


def load_fit(d: ModelData, out_dir: Path) -> FitResult | None:
    """A passing fit previously written by `fit` for `d.position`, or None.
    `d` must be built from the same tables the fit used."""
    name = d.position.lower()
    diag_path, nc_path = out_dir / f"{name}_diagnostics.json", out_dir / f"{name}.nc"
    if not (diag_path.exists() and nc_path.exists()):
        return None
    diag = Diagnostics(**json.loads(diag_path.read_text()))
    if not diag.passed:
        return None
    posterior = xr.open_datatree(nc_path)
    if posterior["posterior"].sizes.get("player_season") != d.n_ps:
        raise ValueError(f"saved {d.position} fit in {out_dir} does not match these tables")
    talent = pl.read_parquet(out_dir / f"{name}_talent.parquet")
    return FitResult(data=d, posterior=posterior, diagnostics=diag, talent=talent)
