import numpy as np
from dataclasses import replace

from dwar.model.talent import simulate_weeks
from dwar.value import league_odds as O
from dwar.value import links as LK
from tests.value.test_league_odds import _ctx, _setup

HURDLE = {"sigma": 1.0, "nu": 6.0, "h0": 4.0, "h1": 3.0, "zero_value": 0.0}


def test_game_draws_have_the_configured_correlations():
    L = LK.game_cholesky({"QB-WR1": 0.4, "RB1-RB2": -0.1}, {"QB-QB": 0.15})
    team = np.array([0, 0, 0, 0, 1])
    role = np.array([0, 3, 1, 2, 0])          # QB, WR1, RB1, RB2 on team 0; QB on team 1
    opp = np.tile(np.array([1, 0]), (20000, 1))
    z = LK.game_draws(team, role, np.zeros((2, 20000), bool), opp, L, np.random.default_rng(0))
    c = np.corrcoef(z)
    assert abs(c[0, 1] - 0.4) < 0.03 and abs(c[2, 3] + 0.1) < 0.03 and abs(c[0, 4] - 0.15) < 0.03
    assert abs(c[1, 2]) < 0.03


def test_bye_or_no_role_is_unlinked_and_qb2_takes_the_link_when_qb1_sits():
    L = LK.game_cholesky({"QB-WR1": 0.6}, {})
    team = np.array([0, 0, 0])
    role = np.array([0, 3, LK.QB2])
    W = 20000
    opp = np.tile(np.array([1, 0]), (W, 1))
    out = np.zeros((2, W), bool)
    out[0, W // 2:] = True                    # QB1 sits in the second half
    z = LK.game_draws(team, role, out, opp, L, np.random.default_rng(1))
    assert np.corrcoef(z[1, W // 2:], z[2, W // 2:])[0, 1] > 0.55
    assert abs(np.corrcoef(z[1, :W // 2], z[2, :W // 2])[0, 1]) < 0.03
    bye = LK.game_draws(team, role, out, np.full((W, 2), -1), L, np.random.default_rng(1))
    assert abs(np.corrcoef(bye[0], bye[1])[0, 1]) < 0.03


def test_roles_rank_by_expected_points_within_team():
    team = np.array([0, 0, 0, 1, 0, 0, -1])
    pos = np.array(["QB", "QB", "RB", "RB", "RB", "RB", "WR"], object)
    known = np.array([10.0, 20.0, 5.0, 9.0, 12.0, 1.0, 30.0])
    r = LK.roles(team, pos, known)
    assert r.tolist() == [LK.QB2, 0, 2, 1, 1, -1, -1]


def test_handoffs_move_teammates_only_in_weeks_the_starter_sits():
    team = np.array([0, 0, 0])
    role = np.array([0, 3, 2])                 # QB, WR1, RB2 (no RB1)
    avail = np.array([[True, False], [True, True], [True, True]])
    m = LK.handoff_mult(avail, team, role, 1, {"QB": {"WR1": -0.1}, "RB1": {"RB2": 0.2}})
    assert np.allclose(m, [[1, 1], [1, 0.9], [1, 1]])


def test_weekly_points_keep_the_weekly_distribution():
    c = np.full(200000, 3.0)
    rng = np.random.default_rng(4)
    a = LK.weekly_points(c, HURDLE, rng.standard_normal(c.shape))
    b = simulate_weeks("sqrt_hurdle", c, HURDLE, rng)
    assert abs(a.mean() - b.mean()) < 0.1
    assert np.allclose(np.quantile(a, [0.1, 0.5, 0.9]), np.quantile(b, [0.1, 0.5, 0.9]), atol=0.2)
    assert abs((a == 0).mean() - (b == 0).mean()) < 0.005


def test_league_sim_runs_with_links_on():
    ctx = _ctx([20.0, 40.0, 10.0])
    lg = _setup(ctx)
    L = LK.game_cholesky({"QB-WR1": 0.4}, {"QB-QB": 0.15})
    links = O.LinkInputs(np.array([0, 1, 1]), 2, {w: np.array([1, 0]) for w in range(1, 8)}, L,
                         {"QB": {"QB2": -0.13}})
    on = replace(ctx, links=links)
    rng = np.random.default_rng(2)
    paths = np.stack([O.simulate_path(on, lg, d % 4, 2, rng, ctx.talent.mean(0))[0] for d in range(30)])
    assert paths.shape == (30, 2, 2) and set(np.unique(paths)) <= {0, 1}
