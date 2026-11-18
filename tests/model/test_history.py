"""History-fit hand-off (model/history.py): basis conversion is exact, the
main model takes the handed-off priors, and a small main fit given a history
fit's priors recovers the truth with tighter intervals than on its own."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from dwar.config import HistoryPrior
from dwar.model.data import UNDRAFTED_PICK
from dwar.model.fit import fit
from dwar.model.history import PickBasis, convert_pick_terms, handoff, history_config, pick_basis
from dwar.model.talent import build_talent_model

from .simulate import LATENT_TRUTH, simulate
from .test_recovery import core_config, interval


def _prior_at(basis: PickBasis, t: dict, L: np.ndarray, undrafted: bool = False):
    z = (L - basis.mu) / basis.sd
    if undrafted:
        return t["alpha0"] + t["alpha_pick"] * z + t["alpha_undrafted"], \
            np.log(t["sigma_rookie"]) + t["sd_pick"] * z + t["sd_undrafted"]
    q = (z**2 - basis.m2) / basis.s2
    return t["alpha0"] + t["alpha_pick"] * z + t["alpha_pick2"] * q, np.log(t["sigma_rookie"]) + t["sd_pick"] * z


def test_convert_pick_terms_keeps_prior_at_every_pick():
    rng = np.random.default_rng(3)
    h, m = PickBasis(4.1, 1.2, 1.0, 1.4), PickBasis(4.4, 0.95, 0.98, 1.3)
    t = {k: rng.normal(size=5) for k in ("alpha0", "alpha_pick", "alpha_pick2", "alpha_undrafted", "sd_pick", "sd_undrafted")}
    t["sigma_rookie"] = np.exp(rng.normal(size=5))
    c = convert_pick_terms(t, h, m)
    for L in np.log([1.0, 3.0, 12.0, 40.0, 120.0, 250.0]):
        for a, b in zip(_prior_at(h, t, L), _prior_at(m, c, L)):
            assert np.allclose(a, b)
    for a, b in zip(_prior_at(h, t, np.log(UNDRAFTED_PICK), True), _prior_at(m, c, np.log(UNDRAFTED_PICK), True)):
        assert np.allclose(a, b)


def test_pick_basis_matches_data():
    d = simulate(n_players=80, seed=5).data
    b = pick_basis(d)
    drafted = d.undrafted == 0
    assert np.allclose((np.log(d.draft_pick[drafted]) - b.mu) / b.sd, d.log_pick_z[drafted])


def test_informed_priors_used_and_unknown_names_refused():
    d = simulate(n_players=60, seed=11).data
    model = build_talent_model(d, informed={"alpha_pick": (-0.7, 0.01), "beta2": (-0.03, 0.002)})
    mu, sd = model["alpha_pick"].owner.inputs[-2:]
    assert np.isclose(mu.eval(), -0.7) and np.isclose(sd.eval(), 0.01)
    assert np.isclose(model["beta2"].owner.inputs[-1].eval(), 0.002)
    with pytest.raises(ValueError, match="sd_pick"):
        build_talent_model(d, informed={"sd_pick": (0.0, 0.1)})  # pick_sd is off


def test_history_config_strips_game_terms():
    cfg = core_config().model_copy(update={"game_steps": ("RB",), "usage": ("RB",), "young_steps": ("RB", "WR"),
                                            "situation_level": {"RB": ("competition",)}})
    h = history_config(cfg, "RB", HistoryPrior(first_season=1999, as_of=date(2013, 3, 1)))
    assert h.game_steps == () and h.usage == () and h.young_steps == ("WR",)
    assert "RB" not in h.situation_level and h.first_season_by_position["RB"] == 1999


@pytest.fixture(scope="module")
def handed_off(tmp_path_factory):
    """History: 300 simulated players; main: 60 others. Each simulation sets
    its truth in its own log-pick basis, so the history truth is compared
    with the hand-off after conversion to the main basis."""
    cfg = core_config().model_copy(update={"rookie_pick_quad": ("WR",), "pick_sd": ("WR",)})
    hist = simulate(n_players=300, seed=41, pick_quad=-0.3)
    main = simulate(n_players=60, seed=42, pick_quad=-0.3)
    h = fit(hist.data, cfg, tmp_path_factory.mktemp("hist"))
    priors, _ = handoff(h.posterior, hist.data, main.data, HistoryPrior(terms=("pick", "age")))
    alone = fit(main.data, cfg, tmp_path_factory.mktemp("alone"))
    informed = fit(main.data, cfg, tmp_path_factory.mktemp("informed"), informed=priors)
    tr = LATENT_TRUTH["sqrt"]
    truth = {k: np.array([tr[k]]) for k in ("alpha0", "alpha_pick", "alpha_undrafted", "sigma_rookie")}
    truth |= {"alpha_pick2": np.array([-0.3]), "sd_pick": np.array([0.0]), "sd_undrafted": np.array([0.0])}
    want = {k: float(v[0]) for k, v in convert_pick_terms(truth, pick_basis(hist.data), pick_basis(main.data)).items()}
    want |= {"beta1": tr["beta1"], "beta2": tr["beta2"]}
    return priors, alone, informed, want


def test_handoff_recovers_and_tightens(handed_off):
    priors, alone, informed, want = handed_off
    assert {"alpha_pick", "alpha_pick2", "alpha_undrafted", "sd_pick", "sd_undrafted", "beta1", "beta2"} <= set(priors)
    for n in ("alpha_pick", "alpha_pick2", "alpha_undrafted", "sd_pick", "beta1", "beta2"):
        m, sd = priors[n]
        assert abs(m - want[n]) < 3.3 * sd, (n, m, sd, want[n])  # history truth, in the main basis
    assert informed.diagnostics.passed, informed.diagnostics.failures
    for n in ("alpha_pick2", "beta2"):
        w_alone = np.diff(interval(alone.posterior, n, 5, 95))[0]
        w_inf = np.diff(interval(informed.posterior, n, 5, 95))[0]
        assert w_inf < w_alone, (n, w_inf, w_alone)
