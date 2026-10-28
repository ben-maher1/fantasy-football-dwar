"""Every weekly distribution's mean must equal `talent_from_center`, so talent
is expected points per game whatever the likelihood."""

import numpy as np
import pytest

from dwar.model.talent import (
    LIKELIHOODS, link, nonzero_prob, scale_of, simulate_weeks, talent_from_center,
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


def test_heavy_steps_add_a_step_tail_parameter():
    from dwar.model.talent import build_talent_model

    from .simulate import simulate

    d = simulate(n_players=30, seed=1, likelihood="sqrt_hurdle_log").data
    names = lambda m: {v.name for v in m.free_RVs}
    assert "nu_step_minus_2" in names(build_talent_model(d, "sqrt_hurdle_log", heavy_steps=True))
    assert "nu_step_minus_2" not in names(build_talent_model(d, "sqrt_hurdle_log"))
