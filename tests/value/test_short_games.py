import numpy as np
import polars as pl
import pytest

from dwar.value.short_games import PRIOR_GAMES, short_games


def weeks() -> pl.DataFrame:
    # A: a starter, 1 short game in 10. B: a backup, 3 short in 4. A WR row is ignored.
    rows = [("A", "QB", 2010, 40, 20.0)] * 9 + [("A", "QB", 2010, 5, 2.0)]
    rows += [("B", "QB", 2011, 3, 1.0)] * 3 + [("B", "QB", 2011, 35, 15.0)] + [("W", "WR", 2011, 0, 9.0)]
    return pl.DataFrame(rows, schema=["player_id", "position", "season", "plays", "fpts"], orient="row")


def test_rates_are_shrunk_toward_the_position_rate():
    s = short_games(weeks(), "QB", 15, 2006)
    base = 4 / 14
    assert s.base_rate == pytest.approx(base)
    assert s.rate["A"] == pytest.approx((1 + PRIOR_GAMES * base) / (10 + PRIOR_GAMES))
    assert s.rate["B"] == pytest.approx((3 + PRIOR_GAMES * base) / (4 + PRIOR_GAMES))
    assert s.rates(["B", "NEW"]).tolist() == pytest.approx([s.rate["B"], base])
    assert sorted(s.points.tolist()) == [1.0, 1.0, 1.0, 2.0]
    assert short_games(weeks(), "QB", 0, 2006) is None


def test_expected_and_simulated_points_mix_in_short_games():
    s = short_games(weeks(), "QB", 15, 2006)
    assert s.expected(np.array([20.0]), np.array([0.25]))[0] == pytest.approx(0.75 * 20 + 0.25 * 1.25)
    sims = s.simulate(np.full(200_000, 20.0), 0.25, np.random.default_rng(0))
    assert (sims < 20).mean() == pytest.approx(0.25, abs=0.01)
    assert set(np.unique(sims)) == {1.0, 2.0, 20.0}


def test_rates_by_preseason_role():
    """A was a QB1 in 2010 and B a backup in 2011. A benched starter gets the
    backup rate when he has no history as a backup."""
    roles = pl.DataFrame({"player_id": ["A", "B"], "season": [2010, 2011], "role": [1.0, 0.0]})
    s = short_games(weeks(), "QB", 15, 2006, roles)
    assert s.role_base == pytest.approx({1.0: 0.1, 0.0: 0.75})
    assert s.role_rate[("A", 1.0)] == pytest.approx((1 + PRIOR_GAMES * 0.1) / (10 + PRIOR_GAMES))
    got = s.rates(["A", "A", "NEW", "B"], [1.0, 0.0, 1.0, None])
    assert got.tolist() == pytest.approx([s.role_rate[("A", 1.0)], 0.75, 0.1, s.rate["B"]])
