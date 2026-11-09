import numpy as np
import polars as pl

from dwar.model.data import CARRY_SHARE_REF, usage_design


def _obs(cs, snap, season=2020):
    n = len(cs)
    return pl.DataFrame({"player_id": ["a"] * n, "season": [season] * n, "week": list(range(1, n + 1)),
                         "carry_share": cs, "offense_pct": snap})


def test_usage_uses_only_earlier_games():
    cs = [0.5, 0.4, 0.3, 0.2, 0.1, 0.6]
    snap = [0.8, 0.8, 0.8, 0.8, 0.2, 0.2]
    u = usage_design(_obs(cs, snap))
    assert u[0, 0] == 0.0  # first game: nothing known
    assert np.isclose(u[1, 0], 0.5 - CARRY_SHARE_REF)
    assert np.isclose(u[3, 0], np.mean(cs[:3]) - CARRY_SHARE_REF)
    assert np.isclose(u[5, 0], np.mean(cs[2:5]) - CARRY_SHARE_REF)
    assert (u[:4, 1] == 0).all()  # snap change needs 4 earlier games
    assert np.isclose(u[4, 1], 0.0)  # games 1-3 vs game 0: both 0.8
    assert np.isclose(u[5, 1], np.mean(snap[2:5]) - np.mean(snap[:2]))


def test_changing_a_game_leaves_earlier_usage_alone():
    cs = [0.5, 0.4, 0.3, 0.2, 0.1, 0.6]
    snap = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    a = usage_design(_obs(cs, snap))
    cs2, snap2 = cs.copy(), snap.copy()
    cs2[4], snap2[4] = 0.9, 0.9
    b = usage_design(_obs(cs2, snap2))
    assert np.array_equal(a[:5], b[:5])


def test_seasons_do_not_leak():
    o = pl.concat([_obs([0.5, 0.5], [0.8, 0.8]), _obs([0.1, 0.1], [0.3, 0.3], 2021)])
    u = usage_design(o)
    assert u[2, 0] == 0.0  # 2021's first game does not see 2020
