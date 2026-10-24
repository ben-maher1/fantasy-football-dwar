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
from xarray import DataTree

from dwar.config import ModelConfig
from dwar.model.data import ModelData
from dwar.model.diagnostics import Diagnostics, FitFailed, check_fit
from dwar.model.talent import build_talent_model

log = logging.getLogger(__name__)

PERCENTILES = (5, 25, 50, 75, 95)
PPC_DRAWS = 200
BIG_WEEK = 25.0


@dataclass(frozen=True)
class FitResult:
    posterior: DataTree
    diagnostics: Diagnostics
    talent: pl.DataFrame


def sample(d: ModelData, cfg: ModelConfig) -> DataTree:
    compiled = nutpie.compile_pymc_model(build_talent_model(d))
    s = cfg.sampler
    return nutpie.sample(
        compiled, draws=s.draws, tune=s.tune, chains=s.chains, seed=cfg.seed, progress_bar=False
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


def posterior_predictive_check(d: ModelData, posterior: DataTree, seed: int) -> dict:
    """Simulated vs observed weekly points: mean, spread, share of big weeks."""
    rng = np.random.default_rng(seed)
    talent, sigma, nu = (_draws(posterior, n) for n in ("talent", "sigma", "nu"))
    picks = rng.choice(len(sigma), size=min(PPC_DRAWS, len(sigma)), replace=False)
    sims = np.stack([
        talent[k][d.obs_ps] + sigma[k] * rng.standard_t(nu[k], size=d.n_obs) for k in picks
    ])

    def stats(x: np.ndarray) -> np.ndarray:
        return np.stack([x.mean(axis=-1), x.std(axis=-1), (x >= BIG_WEEK).mean(axis=-1)], axis=-1)

    observed, simulated = stats(d.y), stats(sims)
    out = {}
    for k, name in enumerate(("mean", "sd", f"share_ge_{BIG_WEEK:g}")):
        lo, hi = np.percentile(simulated[:, k], [5, 95])
        out[name] = {"observed": float(observed[k]), "sim_p05": float(lo), "sim_p95": float(hi)}
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
        "n_obs": d.n_obs,
        "n_player_seasons": d.n_ps,
        "n_players": d.n_players,
        "prospect_covariates": list(d.prospect_x_names),
        "season_covariates": list(d.season_x_names),
        "versions": {p: version(p) for p in ("pymc", "nutpie", "arviz", "nflreadpy")},
        **(meta or {}),
    }
    if diag.passed:
        diag.extra["posterior_predictive"] = posterior_predictive_check(d, posterior, cfg.seed)

    name = d.position.lower()
    (out_dir / f"{name}_diagnostics.json").write_text(json.dumps(diag.to_dict(), indent=2, default=str))
    if not diag.passed:
        log.error("%s fit failed: %s", d.position, "; ".join(diag.failures))
        raise FitFailed(diag)

    posterior.to_netcdf(out_dir / f"{name}.nc")
    talent = talent_summary(d, posterior)
    talent.write_parquet(out_dir / f"{name}_talent.parquet")
    return FitResult(posterior=posterior, diagnostics=diag, talent=talent)
