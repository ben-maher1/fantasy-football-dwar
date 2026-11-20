import numpy as np
import pytest

from dwar.model.talent import talent_from_center
from dwar.value.title import (
    LeagueInputs,
    SeasonFormat,
    availability,
    center_for,
    champion,
    deal,
    lineup_scores,
    rank_scores,
    round_robin,
    seeds,
    simulate_league,
    title_curve,
)

HURDLE = {"sigma": 0.9, "nu": 6.0, "h0": 1.5, "h1": 1.0, "zero_value": 0.0}


def test_format_weeks():
    assert SeasonFormat().playoff_weeks == 3
    assert SeasonFormat().weeks == 17
    assert SeasonFormat(playoff_teams=4, byes=0).playoff_weeks == 2


def test_round_robin_pairs_teams():
    s = round_robin(12, 14, np.random.default_rng(0))
    assert s.shape == (14, 12)
    for w in range(14):
        assert (s[w] != np.arange(12)).all()
        assert (s[w][s[w]] == np.arange(12)).all()   # symmetric
    # The first 11 weeks meet every opponent once.
    for t in range(12):
        assert sorted(s[:11, t]) == [o for o in range(12) if o != t]


def test_seeds_by_wins_then_points():
    sched = np.array([[1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]])
    scores = np.array([[100, 100, 100], [90, 120, 90], [80, 80, 130], [70, 70, 70]], float)
    # Team 0 goes 3-0, team 3 goes 0-3.
    order = seeds(scores, sched, 3)
    assert order[0] == 0 and order[-1] == 3


def test_best_team_always_wins_a_fixed_bracket():
    fmt = SeasonFormat()
    sched = round_robin(12, fmt.weeks, np.random.default_rng(1))
    scores = np.tile(np.arange(12, dtype=float)[:, None] * 10 + 50, (1, fmt.weeks))
    assert champion(scores, sched, fmt) == 11
    # The top seed loses its semifinal: the champion is the best team left.
    s2 = scores.copy()
    s2[11, fmt.regular_weeks + 1] = 0.0
    assert champion(s2, sched, fmt) == 10
    # A bye team cannot lose the first round.
    s3 = scores.copy()
    s3[11, fmt.regular_weeks] = 0.0
    assert champion(s3, sched, fmt) == 11
    # Batched variants match one by one.
    both = np.stack([scores, s2])
    assert list(champion(both, sched, fmt)) == [11, 10]


def test_lineup_scores_fills_fixed_then_flex():
    pos = np.array(["QB", "QB", "RB", "RB", "RB", "WR", "WR", "TE"], object)
    key = np.array([20, 15, 10, 9, 8, 7, 6, 5], float)
    pts = np.array([1, 100, 2, 3, 4, 5, 6, 7], float)
    lineup = {"QB": 1, "RB": 1, "WR": 1, "TE": 1, "FLEX": 2}
    # QB 0 (1), RB 2 (2), WR 5 (5), TE 7 (7), flex RB 3 (3) + RB 4 (4).
    assert lineup_scores(key, pts, pos, lineup) == 22
    key2 = key.copy()
    key2[0] = -np.inf     # QB 0 unavailable: QB 1 starts
    assert lineup_scores(key2, pts, pos, lineup) == 121


def test_availability_matches_share_and_run_length():
    rng = np.random.default_rng(2)
    share = np.array([0.9, 0.6, 0.2, 1.0, 0.0])
    a = availability(np.repeat(share, 4000), 17, 2.0, rng).reshape(5, 4000, 17)
    assert a.mean(axis=(1, 2)) == pytest.approx(share, abs=0.02)
    out = ~a[0]
    starts = (out[:, 1:] & ~out[:, :-1]).sum()
    ends = (~out[:, 1:] & out[:, :-1]).sum()
    assert out[:, 1:].sum() / max(starts, ends) == pytest.approx(2.0, rel=0.25)


def test_center_for_inverts_expected_points():
    talent = np.array([2.0, 6.0, 12.0, 20.0])
    c = center_for("sqrt_hurdle", talent, HURDLE)
    assert talent_from_center("sqrt_hurdle", c, HURDLE, np) == pytest.approx(talent, abs=1e-4)


def test_deal_counts_and_snake():
    rng = np.random.default_rng(3)
    pos = np.array(["RB"] * 30 + ["WR"] * 30, object)
    value = np.arange(60.0)
    team = deal(rank_scores(value), pos, {"RB": 2, "WR": 3}, 4, 0.0, rng)
    for k in range(4):
        assert ((team == k) & (pos == "RB")).sum() == 2
        assert ((team == k) & (pos == "WR")).sum() == 3
    # No noise: the best 8 RBs and 12 WRs are rostered.
    assert (team[22:30] >= 0).all() and (team[:22] < 0).all()


def _toy_league(star: float, star_share: float = 1.0) -> LeagueInputs:
    pos = np.array(["QB"] * 14 + ["RB"] * 30 + ["WR"] * 40 + ["TE"] * 14, object)
    talent = np.concatenate([np.linspace(14, 22, 14), np.linspace(4, 16, 30), np.linspace(4, 15, 40),
                             np.linspace(3, 10, 14)])
    talent[43] = star   # best RB
    share = np.full(len(pos), 0.85)
    share[43] = star_share
    return LeagueInputs(
        pos=pos, talent=talent, share=share, rank_z=rank_scores(talent + np.arange(len(pos)) * 1e-6),
        waiver={"QB": 12.0, "RB": 3.0, "WR": 3.0, "TE": 2.5},
        params={p: HURDLE for p in ("QB", "RB", "WR", "TE")},
        likelihood={p: "sqrt_hurdle" for p in ("QB", "RB", "WR", "TE")},
    )


LINEUP = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 3}
ROSTER = {"QB": 1, "RB": 4, "WR": 5, "TE": 1}


def test_league_star_adds_title_odds_and_benchwarmer_none():
    rng = np.random.default_rng(4)
    star_d, never_d, champs = [], [], []
    for _ in range(300):
        li = _toy_league(30.0)
        r = simulate_league(li, LINEUP, ROSTER, 6, SeasonFormat(playoff_teams=4, byes=0), 0.5, 2.0, rng)
        star_d.extend(r.delta[r.player == 43])
        never_d.extend(r.delta[r.started == 0])
        champs.append(r.champ)
        assert set(np.unique(r.delta)) <= {-1.0, 0.0, 1.0}
    assert np.mean(star_d) > 0.1          # a 30-PPG RB in a 6-team league
    assert np.all(np.array(never_d) == 0)  # never in a lineup: removing him changes nothing
    assert len(set(champs)) == 6


def test_league_player_who_never_plays_adds_nothing():
    rng = np.random.default_rng(5)
    for _ in range(50):
        r = simulate_league(_toy_league(30.0, star_share=0.0), LINEUP, ROSTER, 6,
                            SeasonFormat(playoff_teams=4, byes=0), 0.5, 2.0, rng)
        assert (r.delta[r.player == 43] == 0).all()


def test_title_curve_recovers_a_known_rising_curve():
    rng = np.random.default_rng(6)
    n = 40000
    t = rng.uniform(5, 20, n)
    s = rng.uniform(0, 1, n)
    p = 0.004 * (t - 5) * s          # true swing probability
    d = (rng.random(n) < p).astype(float)
    c = title_curve(t, s, d)
    assert c.at(np.array([15.0]), np.array([0.8]))[0] == pytest.approx(0.004 * 10 * 0.8, abs=0.006)
    assert (np.diff(c.value, axis=0) >= 0).all() and (np.diff(c.value, axis=1) >= 0).all()
