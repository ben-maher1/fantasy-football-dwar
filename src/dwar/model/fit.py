"""Sample the talent model, gate it on diagnostics, and write outputs.

A failed fit writes only `diagnostics.json` (marked failed) and raises
`FitFailed`; nothing else is saved.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import nutpie
import polars as pl
import xarray as xr
from nutpie import zarr_store
from xarray import DataTree

from dwar.config import ModelConfig, data_dir
from dwar.features.player_tables import PlayerTables
from dwar.features.situation import Covariates
from dwar.model.data import ModelData, build_model_data
from dwar.model.diagnostics import Diagnostics, FitFailed, check_fit
from dwar.model.talent import (
    LIKELIHOOD_PARAMS,
    build_talent_model,
    end_talent,
    form_dev_last,
    is_hurdle,
    simulate_weeks,
    young_cum_last,
)

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
# age_shift is jittered: started at exactly 0, nutpie found no valid initial point.
UNJITTERED = ("beta1", "beta2", "beta3", "age_slope_pre", "age_slope_post", "age_rise", "age_fall", "age_fall2")


def sample(d: ModelData, cfg: ModelConfig) -> DataTree:
    p = d.position
    model = build_talent_model(
        d, cfg.likelihood_for(p), cfg.age_curve.get(p, "quadratic"),
        round1=p in cfg.rookie_round1, growth=p in cfg.growth,
        opportunity=p in cfg.opportunity, opportunity_years=cfg.opportunity_years,
        game_steps=p in cfg.game_steps, game_mode=cfg.game_mode, qb_drift=p in cfg.qb_drift,
        form_rho=cfg.form_rho, tau_early=p in cfg.tau_early, tau_y3=p in cfg.tau_y3, season_rho=p in cfg.season_rho, season_rho_mean=cfg.season_rho_mean,
        young_steps=p in cfg.young_steps,
    )
    jitter = {rv for rv in model.free_RVs if rv.name not in UNJITTERED}
    compiled = nutpie.compile_pymc_model(model, jitter_rvs=jitter)
    s = cfg.sampler
    # Draws go to a zarr store on disk while sampling: held in memory, the
    # game-by-game WR fit (a form term per game) ran out of memory.
    with tempfile.TemporaryDirectory(dir=data_dir()) as tmp:
        nutpie.sample(
            compiled, draws=cfg.draws_by_position.get(p, s.draws), tune=s.tune, chains=s.chains, seed=cfg.seed,
            progress_bar=False, save_warmup=False, zarr_store=zarr_store.LocalStore(tmp),
        )
        posterior = _compact(Path(tmp), d, cfg.likelihood_for(p))
    # The zarr store adds dict attributes (sampler settings) netCDF cannot save.
    for node in posterior.subtree:
        node.attrs = {k: v for k, v in node.attrs.items() if isinstance(v, (str, int, float, np.number))}
    return posterior


def add_end_talent(post: xr.Dataset, likelihood: str, zero_value: float) -> None:
    """With young steps, make `talent` each player-season's expected points
    after its last game (season level plus the steps so far), what
    replacement level and the value layer need; `loc` stays the season level."""
    p = {n: post[n].values[..., None] for n in ("sigma", "nu", "h0", "h1")} | {"zero_value": zero_value}
    post["talent"] = (post["talent"].dims, end_talent(likelihood, post["loc"].values, post["young_cum"].values, p))


def _compact(store: Path, d: ModelData, likelihood: str) -> DataTree:
    """The posterior and sample stats, loaded into memory, without warmup.
    The "form" mode's per-game shocks (`form_z`, one per game per draw: about
    3.4 GB for WR 2013-2026) are replaced by what forecasting needs, the form
    after each player-season's last game (`form_dev`), computed a chain at a
    time. Fit checks then cover the form term through its settings
    (rho_form, sigma_form, lam_form) rather than every per-game shock."""
    import zarr

    lazy = xr.open_datatree(store, engine="zarr", consolidated=False)
    post = lazy["posterior"].to_dataset()
    post = post.drop_vars([v for v in post.data_vars if str(v).endswith("__")])  # unconstrained copies
    if "form_z" in post:
        post = post.drop_vars("form_z").load()
        shocks = zarr.open_array(store / "posterior" / "form_z", mode="r")  # read a chain at a time
        dev = np.stack([
            form_dev_last(d, np.asarray(shocks[c]), post["rho_form"].values[c], post["sigma_form"].values[c],
                          post["lam_form"].values[c])
            for c in range(post.sizes["chain"])
        ])
        post["form_dev"] = (("chain", "draw", "form_player_season"), dev)
    if "young_z" in post:
        post = post.drop_vars("young_z").load()
        z = zarr.open_array(store / "posterior" / "young_z", mode="r")
        post["young_cum"] = (("chain", "draw", "form_player_season"), np.stack([
            young_cum_last(d, np.asarray(z[c]), post["sigma_young"].values[c]) for c in range(post.sizes["chain"])
        ]))
        add_end_talent(post, likelihood, d.zero_week_mean)
    stats = lazy["sample_stats"].to_dataset()
    # Per-draw stats only (divergences, energy, step size, ...): the store also
    # holds per-draw gradients and positions, several GB each here.
    stats = stats[[v for v in stats.data_vars if stats[v].dims == ("chain", "draw")]]
    return DataTree.from_dict({"posterior": post.load(), "sample_stats": stats.load()})


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
    opportunity: pl.DataFrame | None = None,
) -> ModelData:
    """One position's model data, with its situation factors if any, and its
    weekly opportunity points if the position uses them."""
    c = (cov or {}).get(position)
    extra = {"opportunity": opportunity} if opportunity is not None and position in cfg.opportunity else {}
    return build_model_data(
        tables, position, cfg.first_season_by_position.get(position, cfg.first_season), min_plays=cfg.min_plays.get(position, 0),
        season_covariates=c.level if c is not None and c.level.width > 2 else None,
        season_range_covariates=c.range if c is not None and c.range.width > 2 else None,
        injury_exits="drop" if position in cfg.injury_exits else "keep",
        **extra,
    )


def fit_positions(
    tables: PlayerTables, cfg: ModelConfig, out_dir: Path, meta: dict | None = None,
    cov: dict[str, Covariates] | None = None, opportunity: pl.DataFrame | None = None,
) -> dict[str, FitResult]:
    """Fit every configured position. All positions are attempted so each
    writes its diagnostics; if any fails, raises FitFailed for the first
    failure after the rest have run (the run is failed as a whole)."""
    results, failures = {}, []
    for position in cfg.positions:
        d = position_data(tables, position, cfg, cov, opportunity)
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
