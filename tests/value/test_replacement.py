"""Replacement level on a toy 2-team league small enough to count by hand."""

import numpy as np
import pytest

from dwar.value.replacement import replacement_level, starters_used

PLAYERS = {
    "QB": [20, 18, 15, 10, 8],
    "RB": [15, 12, 9, 6],
    "WR": [14, 11, 10, 7],
    "TE": [10, 6, 4],
}
TALENT = np.array([[t for ts in PLAYERS.values() for t in ts]], float)
POSITIONS = np.array([p for p, ts in PLAYERS.items() for _ in ts])
SUPERFLEX = {"QB": 1, "RB": 1, "WR": 1, "TE": 1, "FLEX": 1, "SUPERFLEX": 1}


def test_superflex_allocation_by_hand():
    # Fixed: QB 20,18  RB 15,12  WR 14,11  TE 10,6.
    # FLEX x2 from RB 9,6 / WR 10,7 / TE 4: WR 10, RB 9.
    # SUPERFLEX x2 from QB 15,10,8 / RB 6 / WR 7 / TE 4: QB 15, QB 10.
    used = starters_used(TALENT, POSITIONS, SUPERFLEX, teams=2)
    assert {p: int(n[0]) for p, n in used.items()} == {"QB": 4, "RB": 3, "WR": 3, "TE": 2}
    repl = replacement_level(TALENT, POSITIONS, SUPERFLEX, teams=2)
    assert {p: float(v[0]) for p, v in repl.items()} == {"QB": 8, "RB": 6, "WR": 7, "TE": 4}


def test_one_qb_league_has_a_higher_qb_replacement():
    lineup = {k: v for k, v in SUPERFLEX.items() if k != "SUPERFLEX"}
    repl = replacement_level(TALENT, POSITIONS, lineup, teams=2)
    assert float(repl["QB"][0]) == 15


def test_vectorized_over_draws():
    talent = np.vstack([TALENT, TALENT + 1.0])
    repl = replacement_level(talent, POSITIONS, SUPERFLEX, teams=2)
    np.testing.assert_array_equal(repl["WR"], [7, 8])


def test_position_exhausted_gives_zero():
    repl = replacement_level(TALENT, POSITIONS, {"QB": 5}, teams=1)
    assert float(repl["QB"][0]) == 0.0


def test_unknown_slot_rejected():
    with pytest.raises(ValueError, match="unknown lineup slots"):
        starters_used(TALENT, POSITIONS, {"K": 1}, teams=2)


def test_waiver_level_is_first_unrostered():
    import numpy as np
    from dwar.value.replacement import waiver_level
    t = np.arange(30, 0, -1.0)[None, :]  # 30 .. 1
    pos = np.array(["WR"] * 30)
    w = waiver_level(t, pos, {"WR": 2}, teams=12)  # 24 rostered
    assert w["WR"][0] == 6.0


def test_start_share_starters_always_bench_sometimes():
    import numpy as np
    from dwar.value.replacement import share_at, start_share
    rng = np.random.default_rng(0)
    talent = {"WR": np.linspace(20, 1, 40), "RB": np.linspace(18, 1, 40)}
    grid = start_share(talent, {"WR": 1, "RB": 1}, teams=4, rostered={"WR": 3, "RB": 3},
                       p_play={"WR": 0.8, "RB": 0.8}, rng=rng, weeks=400)
    t, s = grid["WR"]
    assert s[-1] == 1.0                 # each team's best WR starts whenever he plays
    assert 0.0 < s[0] < 0.2             # third WR: only when both above him are out
    assert share_at(grid["WR"], np.array([25.0]))[0] == 1.0
    assert np.all(np.diff(share_at(grid["WR"], np.linspace(1, 20, 50))) >= 0)
