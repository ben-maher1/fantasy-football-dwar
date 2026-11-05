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
