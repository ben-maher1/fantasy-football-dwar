"""Parameter recovery (CLAUDE.md): fit simulated receivers whose true
parameters are known, through the real data -> model -> fit -> gate path."""

from __future__ import annotations

import numpy as np
import pytest

from dwar.config import load_model_config
from dwar.model.fit import fit, sample
from dwar.model.talent import scale_of

from .simulate import SEASON_EFFECT, USAGE_TRUTH, simulate


def core_config():
    """The config without the game-by-game terms the production WR model
    adds (tested on their own below): the simulated receivers have none."""
    return load_model_config().model_copy(update={
        "game_steps": (), "qb_drift": (), "draws_by_position": {}, "first_season_by_position": {},
    })


def interval(posterior, name: str, lo: float, hi: float) -> np.ndarray:
    values = np.asarray(posterior["posterior"][name].values)
    return np.percentile(values.reshape(-1, *values.shape[2:]), [lo, hi], axis=0)


@pytest.fixture(scope="module")
def recovered(tmp_path_factory):
    sim = simulate(n_players=300)
    result = fit(sim.data, core_config(), tmp_path_factory.mktemp("fit"))
    return sim, result


def test_fit_passes_diagnostics(recovered):
    _, result = recovered
    assert result.diagnostics.passed, result.diagnostics.failures


def test_global_parameters_recovered(recovered):
    """Every truth must sit inside its 99% interval. Of 12 independent 90%
    intervals about one misses by chance (two or fewer ~89% of the time), so
    allow two. Veteran prior parameters are checked against the realized
    veteran sample, which is all the data can reveal."""
    sim, result = recovered
    misses_90 = []
    for name, truth in sim.truth.items():
        lo90, hi90 = interval(result.posterior, name, 5, 95)
        lo99, hi99 = interval(result.posterior, name, 0.5, 99.5)
        assert lo99 <= truth <= hi99, f"{name}: {truth} outside 99% [{lo99:.3f}, {hi99:.3f}]"
        if not lo90 <= truth <= hi90:
            misses_90.append(name)
    assert len(misses_90) <= 2, misses_90


def test_prospect_coefficients_recovered(recovered):
    sim, result = recovered
    lo, hi = interval(result.posterior, "gamma", 0.5, 99.5)
    for j, name in enumerate(sim.data.prospect_x_names):
        truth = sim.gamma.get(name, 0.0)
        assert lo[j] <= truth <= hi[j], f"{name}: {truth} outside [{lo[j]:.2f}, {hi[j]:.2f}]"
    forty = sim.data.prospect_x_names.index("forty_z")
    assert hi[forty] < 0, "a real 40-time effect should be detected as negative"


def test_talent_interval_coverage(recovered):
    sim, result = recovered
    t = result.talent
    covered_90 = (t["talent_p05"].to_numpy() <= sim.talent) & (sim.talent <= t["talent_p95"].to_numpy())
    covered_50 = (t["talent_p25"].to_numpy() <= sim.talent) & (sim.talent <= t["talent_p75"].to_numpy())
    assert 0.80 <= covered_90.mean() <= 0.97
    assert 0.40 <= covered_50.mean() <= 0.60


def test_outputs_written(recovered, tmp_path_factory):
    _, result = recovered
    assert set(result.talent.columns) >= {
        "player_id", "season", "games_played", "talent_mean",
        "talent_p05", "talent_p25", "talent_p50", "talent_p75", "talent_p95",
    }
    ppc = result.diagnostics.extra["posterior_predictive"]
    assert ppc["mean"]["fits"] and ppc["share_zero_or_less"]["fits"]
    assert sum(s["fits"] for s in ppc.values()) >= len(ppc) - 1, ppc


def test_season_covariate_slot(tmp_path):
    cfg = core_config()
    effect = SEASON_EFFECT[scale_of(cfg.likelihood)]
    sim = simulate(n_players=120, season_effect=effect, seed=11)
    assert sim.data.season_x_names == ("boost",)
    result = fit(sim.data, cfg, tmp_path)
    lo, hi = interval(result.posterior, "eta", 0.5, 99.5)
    assert lo[0] <= effect <= hi[0] and lo[0] > 0


def test_range_covariate_slot(tmp_path):
    """A flag that makes a season's step exp(0.7), about twice, as large is
    recovered, and the yearly step size is not inflated to absorb it."""
    cfg = core_config()
    sim = simulate(n_players=300, range_effect=0.7, seed=13)
    assert sim.data.season_vol_x_names == ("moved",)
    result = fit(sim.data, cfg, tmp_path)
    assert result.diagnostics.passed, result.diagnostics.failures
    lo, hi = interval(result.posterior, "kappa", 0.5, 99.5)
    assert lo[0] <= 0.7 <= hi[0] and lo[0] > 0
    lo, hi = interval(result.posterior, "tau", 0.5, 99.5)
    assert lo <= sim.truth["tau"] <= hi


def test_same_seed_same_draws():
    sim = simulate(n_players=30, weeks_per_season=6, seed=3)
    cfg = core_config()
    cfg = cfg.model_copy(update={"sampler": cfg.sampler.model_copy(update={"tune": 100, "draws": 50})})
    a = sample(sim.data, cfg)["posterior"]["tau"].values
    b = sample(sim.data, cfg)["posterior"]["tau"].values
    np.testing.assert_array_equal(a, b)


@pytest.fixture(scope="module")
def recovered_usage(tmp_path_factory):
    sim = simulate(n_players=300, usage=True, seed=17)
    cfg = core_config().model_copy(update={"opportunity": ("WR",), "rookie_round1": ("WR",), "growth": ("WR",)})
    return sim, fit(sim.data, cfg, tmp_path_factory.mktemp("fit_usage"))


def test_usage_fit_recovers_growth_efficiency_and_opportunity(recovered_usage):
    """Opportunity points, per-player efficiency, the first-round step and
    career-year growth scaled by draft capital are all recovered."""
    sim, result = recovered_usage
    assert sim.data.x is not None and np.isfinite(sim.data.x).all()
    assert result.diagnostics.passed, result.diagnostics.failures
    for name, truth in sim.truth.items():
        lo, hi = interval(result.posterior, name, 0.5, 99.5)
        assert lo <= truth <= hi, f"{name}: {truth} outside 99% [{lo:.3f}, {hi:.3f}]"
    for name in ("growth", "growth_pick"):
        lo, hi = interval(result.posterior, name, 0.5, 99.5)
        for j, truth in enumerate(USAGE_TRUTH[name]):
            assert lo[j] <= truth <= hi[j], f"{name}[{j}]: {truth} outside [{lo[j]:.3f}, {hi[j]:.3f}]"
    lo, hi = interval(result.posterior, "growth", 0.5, 99.5)
    assert (lo > 0).all(), "real growth should be detected"


def test_usage_talent_interval_coverage(recovered_usage):
    sim, result = recovered_usage
    t = result.talent
    covered_90 = (t["talent_p05"].to_numpy() <= sim.talent) & (sim.talent <= t["talent_p95"].to_numpy())
    assert 0.80 <= covered_90.mean() <= 0.97


@pytest.fixture(scope="module")
def recovered_games(tmp_path_factory):
    sim = simulate(n_players=300, game_steps=True)
    # Lasting per-game steps ("walk", tested and not used in production)
    # mix slowly on sigma_game: 3000 draws.
    cfg = core_config().model_copy(update={"game_steps": ("WR",), "game_mode": "walk", "draws_by_position": {"WR": 3000}})
    return sim, fit(sim.data, cfg, tmp_path_factory.mktemp("fit_games"))


def test_game_steps_recovered(recovered_games):
    """Per-game steps and their early-career boost are recovered, and the
    season-level parameters still are."""
    sim, result = recovered_games
    assert result.diagnostics.passed, result.diagnostics.failures
    for name, truth in sim.truth.items():
        lo, hi = interval(result.posterior, name, 0.5, 99.5)
        assert lo <= truth <= hi, f"{name}: {truth} outside 99% [{lo:.3f}, {hi:.3f}]"


@pytest.fixture(scope="module")
def recovered_form(tmp_path_factory):
    from .simulate import FORM_RHO

    sim = simulate(n_players=300, form=True, seed=4)
    # Production WR draws (config draws_by_position); 2000 left beta1 at R-hat 1.011.
    cfg = core_config().model_copy(update={"game_steps": ("WR",), "game_mode": "form", "form_rho": FORM_RHO,
                                           "draws_by_position": {"WR": 3500}})
    return sim, fit(sim.data, cfg, tmp_path_factory.mktemp("fit_form"))


def test_form_recovered(recovered_form):
    """The production WR mode: in-season form's size (as a share of weekly
    noise) and early-career boost are recovered, with the decay rate held
    at its true value as config holds it, and season-level parameters still are."""
    sim, result = recovered_form
    assert result.diagnostics.passed, result.diagnostics.failures
    for name, truth in sim.truth.items():
        lo, hi = interval(result.posterior, name, 0.5, 99.5)
        assert lo <= truth <= hi, f"{name}: {truth} outside 99% [{lo:.3f}, {hi:.3f}]"
