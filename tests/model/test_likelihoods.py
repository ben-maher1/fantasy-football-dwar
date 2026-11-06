"""Every weekly distribution's mean must equal `talent_from_center`, so talent
is expected points per game whatever the likelihood."""

import numpy as np
import pytest

from dwar.model.talent import (
    LIKELIHOODS,
    link,
    nonzero_prob,
    scale_of,
    simulate_weeks,
    talent_from_center,
)

from .simulate import LIKELIHOOD_TRUTH

# Distribution centers spanning bench to elite receivers on each scale.
CENTERS = {"points": [2.0, 8.0, 18.0], "sqrt": [1.5, 3.0, 4.3], "log": [1.5, 3.0, 4.3]}


@pytest.mark.parametrize("likelihood", LIKELIHOODS)
@pytest.mark.parametrize("which", [0, 1, 2])
def test_simulated_mean_equals_talent(likelihood, which):
    rng = np.random.default_rng(0)
    center = CENTERS[scale_of(likelihood)][which]
    params = LIKELIHOOD_TRUTH[likelihood]
    y = simulate_weeks(likelihood, np.full(400_000, center), params, rng)
    talent = talent_from_center(likelihood, np.array([center]), params)[0]
    assert y.mean() == pytest.approx(talent, abs=0.05)


def test_zero_value_counts_toward_talent():
    rng = np.random.default_rng(2)
    params = LIKELIHOOD_TRUTH["sqrt_hurdle"] | {"zero_value": -0.6}
    center = np.full(400_000, 2.0)
    y = simulate_weeks("sqrt_hurdle", center, params, rng)
    assert (y <= 0).mean() > 0.2 and set(np.unique(y[y <= 0])) == {-0.6}
    assert y.mean() == pytest.approx(talent_from_center("sqrt_hurdle", center[:1], params)[0], abs=0.05)


def test_talent_is_increasing_in_center():
    talent = talent_from_center("sqrt_hurdle", np.linspace(0.5, 5.0, 50), LIKELIHOOD_TRUTH["sqrt_hurdle"])
    assert (np.diff(talent) > 0).all()


def test_hurdle_weeks_are_zero_or_positive_and_zero_rate_falls_with_talent():
    rng = np.random.default_rng(1)
    params = LIKELIHOOD_TRUTH["sqrt_hurdle"]
    lo_c, _, hi_c = CENTERS["sqrt"]
    low = simulate_weeks("sqrt_hurdle", np.full(100_000, lo_c), params, rng)
    high = simulate_weeks("sqrt_hurdle", np.full(100_000, hi_c), params, rng)
    assert (low >= 0).all()
    assert (low == 0).mean() > (high == 0).mean()
    expected_zero = 1 - nonzero_prob(lo_c, params["h0"], params["h1"], lib=np)
    assert (low == 0).mean() == pytest.approx(expected_zero, abs=0.01)


def test_softplus_link_is_positive_and_near_identity():
    x = np.array([-20.0, -2.0, 0.0, 5.0, 15.0])
    out = link(x, "sqrt_hurdle", lib=np)
    assert (out > 0).all()
    np.testing.assert_allclose(out[3:], x[3:], atol=0.01)
    np.testing.assert_array_equal(link(x, "studentt", lib=np), x)


def test_unknown_likelihood():
    with pytest.raises(ValueError, match="unknown likelihood"):
        simulate_weeks("poisson", np.ones(3), {}, np.random.default_rng())


def test_log_link_makes_a_latent_step_proportional():
    step = 0.1
    for level in (2.0, 4.0):
        x = np.log(level)
        ratio = link(np.array([x + step]), "sqrt_hurdle_log", lib=np) / link(np.array([x]), "sqrt_hurdle_log", lib=np)
        assert ratio[0] == pytest.approx(np.exp(step))


def test_form_dev_last_matches_the_recursion():
    from types import SimpleNamespace

    from dwar.model.talent import EARLY_GAMES, form_dev_last

    rng = np.random.default_rng(0)
    lens = [3, 1, 5]
    obs_ps = np.repeat(np.arange(3), lens)
    shuffle = rng.permutation(len(obs_ps))  # obs need not be in game order
    weeks = np.concatenate([np.arange(1, n + 1) for n in lens])
    cg = np.concatenate([np.arange(n) + 10 * p for p, n in enumerate(lens)]).astype(float)
    d = SimpleNamespace(obs_ps=obs_ps[shuffle], obs_week=weeks[shuffle], obs_career_games=cg[shuffle],
                        ps_player=np.array([0, 0, 1]), n_obs=len(obs_ps), n_ps=3)
    z = rng.standard_normal((2, len(obs_ps)))
    rho, sig, lam = np.array([0.6, 0.9]), np.array([0.3, 0.5]), np.array([0.2, -0.1])
    got = form_dev_last(d, z, rho, sig, lam)
    for k in range(2):
        for p in range(3):
            dev = 0.0
            for j in np.flatnonzero(obs_ps == p):
                o = np.flatnonzero(shuffle == j)[0]  # z is indexed by the shuffled obs
                e = sig[k] * np.sqrt(1 - rho[k] ** 2) * np.exp(lam[k] * np.exp(-cg[j] / EARLY_GAMES)) * z[k, o]
                dev = rho[k] * dev + e
            assert got[k, p] == pytest.approx(dev)


def test_young_cum_last_sums_steps_through_each_season_and_carries_over():
    from types import SimpleNamespace

    from dwar.model.talent import season_early, young_cum_last

    lens = [3, 1, 5]  # player 0: seasons 0 and 1; player 1: season 2
    obs_ps = np.repeat(np.arange(3), lens)
    weeks = np.concatenate([np.arange(1, n + 1) for n in lens])
    cg = np.array([0, 1, 2, 3, 0, 1, 2, 3, 4], float)
    d = SimpleNamespace(obs_ps=obs_ps, obs_week=weeks, obs_career_games=cg, ps_player=np.array([0, 0, 1]),
                        n_obs=len(obs_ps), n_ps=3)
    z = np.array([[1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0]])  # one per non-first game, in game order
    got = young_cum_last(d, z, np.array([0.5]))[0]
    w = 0.5 * season_early(cg)
    assert got[0] == pytest.approx(w[1] * 1 + w[2] * 2)
    assert got[1] == pytest.approx(got[0] + w[3] * 3)  # carried into the next season
    assert got[2] == pytest.approx(w[5:].sum())
