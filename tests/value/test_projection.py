import numpy as np
import pytest

from dwar.value.projection import PositionDraws, inverse_softplus, project_talent
from dwar.model.talent import softplus


def draws(n: int, talent: list[float], beta1=0.0, beta2=0.0, tau=0.0) -> PositionDraws:
    t = np.tile(np.array(talent, float), (n, 1))
    return PositionDraws(
        likelihood="studentt", latent=t.copy(), talent=t,
        beta1=np.full(n, beta1), beta2=np.full(n, beta2), tau=np.full(n, tau),
        params={"sigma": np.full(n, 5.0), "nu": np.full(n, 5.0)}, zero_value=0.0,
    )


def test_flat_curve_and_no_steps_keep_talent_constant():
    rng = np.random.default_rng(0)
    out = project_talent(draws(10, [12.0, 7.0]), np.array([0, 1]), np.array([2025, 2024]),
                         np.array([26.0, 26.0]), np.arange(2024, 2030), rng)
    np.testing.assert_allclose(out[:, 0, :], 12.0)
    np.testing.assert_allclose(out[:, 1, :], 7.0)


def test_age_curve_moves_talent_and_never_looks_back():
    rng = np.random.default_rng(0)
    # 1 point a year of decline, last season 2025 at age 30.
    out = project_talent(draws(5, [15.0], beta1=-1.0), np.array([0]), np.array([2025]),
                         np.array([30.0]), np.arange(2024, 2029), rng)
    np.testing.assert_allclose(out[0, 0], [15.0, 15.0, 14.0, 13.0, 12.0])


def test_random_walk_variance_grows_with_seasons_ahead():
    rng = np.random.default_rng(1)
    out = project_talent(draws(50_000, [10.0], tau=2.0), np.array([0]), np.array([2025]),
                         np.array([26.0]), np.arange(2026, 2029), rng)
    np.testing.assert_allclose(out[:, 0, :].var(axis=0), [4.0, 8.0, 12.0], rtol=0.03)


def test_inverse_softplus_roundtrip():
    x = np.array([-3.0, 0.0, 0.5, 4.0, 20.0])
    np.testing.assert_allclose(inverse_softplus(softplus(x, np)), x, atol=1e-8)


def test_log_scale_draws_recover_latent_from_loc():
    from types import SimpleNamespace

    import xarray as xr

    loc = np.exp(np.array([[[-1.0, 0.5, 1.4]]]))
    one = lambda v: (("chain", "draw"), np.full((1, 1), v))
    post = xr.DataTree.from_dict({"posterior": xr.Dataset({
        "loc": (("chain", "draw", "player_season"), loc), "talent": (("chain", "draw", "player_season"), loc**2),
        "beta1": one(0.0), "beta2": one(0.0), "tau": one(0.1),
        "sigma": one(1.0), "nu": one(20.0), "h0": one(2.0), "h1": one(2.0),
    })})
    d = PositionDraws.from_posterior(SimpleNamespace(zero_week_mean=0.0), post, "sqrt_hurdle_log", np.array([0]))
    np.testing.assert_allclose(d.latent[0], [-1.0, 0.5, 1.4])


def test_heavy_tailed_steps_project_with_student_t():
    n = 20_000
    normal = draws(n, [10.0], tau=1.0)
    heavy = PositionDraws(**{**normal.__dict__, "nu_step": np.full(n, 3.0)})
    rng = np.random.default_rng(0)
    args = (np.array([0]), np.array([2020]), np.array([26.0]), np.array([2021]))
    a = project_talent(normal, *args, rng)[:, 0, 0] - 10.0
    b = project_talent(heavy, *args, rng)[:, 0, 0] - 10.0
    assert np.mean(np.abs(a) > 4) < 0.001 < np.mean(np.abs(b) > 4)
