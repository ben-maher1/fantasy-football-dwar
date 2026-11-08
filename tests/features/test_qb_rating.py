"""QB rating: a game's rating uses only the passer's earlier games."""

from __future__ import annotations

import polars as pl
import pytest

from dwar.features.qb_rating import SCALE, SHRINK_ATTEMPTS, team_qb_ratings


def stats(rows):
    return pl.DataFrame(rows, schema=["player_id", "season", "week", "team", "attempts", "passing_epa"], orient="row")


def test_rating_uses_only_earlier_games():
    s = stats([
        ("qb1", 2020, 1, "A", 30, 6.0),
        ("qb1", 2020, 2, "A", 20, -4.0),
        ("qb2", 2020, 2, "A", 5, 3.0),     # backup: fewer attempts, not the game's passer
        ("qb1", 2020, 3, "A", 40, 100.0),  # a huge later game must not leak into weeks 1-2
    ])
    r = {w: v for w, v in team_qb_ratings(s).sort("week").select("week", "qb_rating").iter_rows()}
    assert r[1] == 0.0                                  # no earlier games: the shrink target
    assert r[2] == pytest.approx(6.0 / (30 + SHRINK_ATTEMPTS) / SCALE)
    assert r[3] == pytest.approx(2.0 / (50 + SHRINK_ATTEMPTS) / SCALE)


def test_missing_columns_give_no_ratings():
    assert team_qb_ratings(pl.DataFrame({"player_id": ["a"], "season": [2020]})).height == 0


def test_qb_fade_recovers_persistence_and_fades():
    import numpy as np

    from dwar.features.qb_rating import qb_fade

    rng = np.random.default_rng(0)
    players, seasons = 400, 6
    q = np.empty((players, seasons))
    q[:, 0] = 0.3 + rng.standard_normal(players)
    for s in range(1, seasons):  # AR(1) toward 0.3 with persistence 0.4
        q[:, s] = 0.3 + 0.4 * (q[:, s - 1] - 0.3) + rng.standard_normal(players) * np.sqrt(1 - 0.16)
    f = qb_fade(np.repeat(np.arange(players), seasons), np.tile(np.arange(2010, 2010 + seasons), players), q.ravel())
    assert abs(f.keep[0] - 0.4) < 0.06
    assert abs(f.keep[1] - 0.16) < 0.06
    assert abs(f.mean - 0.3) < 0.1
    # The next offseason uses today's rating as is; later ones fade toward the mean.
    assert f.expected(2.0, 1) == 2.0
    assert f.expected(np.array([2.0, 2.0]), np.array([2, 3])) == pytest.approx(
        [f.mean + f.keep[0] * (2 - f.mean), f.mean + f.keep[1] * (2 - f.mean)])
