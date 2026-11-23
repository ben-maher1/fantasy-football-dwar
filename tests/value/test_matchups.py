import numpy as np
import polars as pl

from dwar.value import matchups as M


def _weeks(rows):
    return pl.DataFrame(rows, schema=["player_id", "position", "season", "week", "game_id", "team", "opponent_team", "fpts"],
                        orient="row")


def test_expectation_uses_only_earlier_games():
    rows = [("a", "QB", 2026, w, f"g{w}", "X", "Y", 20.0) for w in range(1, 6)]
    base = M.expectations(_weeks(rows), {"QB": 10.0})
    rows[-1] = rows[-1][:-1] + (60.0,)            # a big last game changes nothing before it
    later = M.expectations(_weeks(rows), {"QB": 10.0})
    assert base["expected"].to_list() == later["expected"].to_list()
    assert base["expected"][0] == 10.0           # first game: the position average
    assert 10.0 < base["expected"][4] < 20.0     # shrunk toward it


def test_defense_ratings_centered_and_ignore_later_seasons():
    rows = [("a", "RB", 2026, 1, "g1", "A", "SOFT", 20.0), ("b", "RB", 2026, 1, "g2", "B", "HARD", 5.0)]
    w = _weeks(rows).with_columns(expected=pl.lit(10.0))
    r = dict((o, x) for o, _, x in M.defense_ratings(w, 2026).iter_rows())
    assert r["SOFT"] > 0 > r["HARD"] and abs(r["SOFT"] + r["HARD"]) < 1e-12
    future = pl.concat([w, _weeks([("c", "RB", 2027, 1, "g3", "C", "SOFT", 0.0)]).with_columns(expected=pl.lit(10.0))])
    assert M.defense_ratings(future, 2026).sort("opponent_team").equals(M.defense_ratings(w, 2026).sort("opponent_team"))


def test_multipliers_bye_home_and_positions_left_out():
    sched = pl.DataFrame({"season": [2026] * 2, "game_type": ["REG"] * 2, "week": [12, 13],
                          "home_team": ["A", "C"], "away_team": ["B", "D"], "location": ["Home", "Home"]})
    games = M.team_weeks(sched, 2026, [12, 13])
    ratings = pl.DataFrame({"opponent_team": ["B"], "position": ["QB"], "rating": [0.2]})
    m = M.multipliers(["A", "A", "B", None, "ZZZ"], ["QB", "WR", "QB", "QB", "QB"], games, ratings,
                      {"QB": {"dvp": 0.5, "home": 0.1}}, [12, 13], byes=True)
    assert np.isclose(m[0, 0], 1 + 0.5 * 0.2 + 0.05) and m[0, 1] == 0.0    # A: home vs B, then bye
    assert m[1, 0] == 1.0 and m[1, 1] == 0.0                              # WR: no matchup effect, bye still out
    assert np.isclose(m[2, 0], 1 - 0.05)                                  # B away, A unrated
    assert (m[3] == 1).all() and (m[4] == 1).all()                        # unknown team / not on schedule
    assert (M.multipliers(["A"], ["WR"], games, ratings, {}, [12, 13], byes=False) == 1).all()
