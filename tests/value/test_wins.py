import numpy as np
import polars as pl
import pytest

from dwar.value.wins import team_score_sd, wins_per_game


def test_wins_per_game_known_values():
    assert wins_per_game(0.0, 25.0) == 0.0
    assert wins_per_game(-5.0, 25.0) == pytest.approx(-wins_per_game(5.0, 25.0))
    # PAR equal to sqrt(2) sigma lifts a 50% week to Phi(1) = 84.13%.
    assert wins_per_game(np.sqrt(2) * 25.0, 25.0) == pytest.approx(0.341345, abs=1e-6)


def test_wins_per_game_matches_simulated_matchups():
    rng = np.random.default_rng(0)
    sigma, par = 25.0, 6.0
    mine = rng.normal(100 + par, sigma, 1_000_000)
    theirs = rng.normal(100, sigma, 1_000_000)
    assert (mine > theirs).mean() - 0.5 == pytest.approx(wins_per_game(par, sigma), abs=0.002)


def test_team_score_sd_by_hand():
    weeks = pl.DataFrame({
        "player_id": ["A"] * 4 + ["B"] * 4,
        "season": [2024] * 8,
        "fpts": [8.0, 12.0, 10.0, 10.0, 1.0, 2.0, 3.0, 2.0],
    })
    seasons = pl.DataFrame({
        "player_id": ["A", "B"], "season": [2024, 2024], "position": ["WR", "WR"],
        "games_played": [4, 4], "fpts_per_game": [10.0, 2.0],
    })
    # One team starts one WR: A, whose weekly variance is 8/3.
    assert team_score_sd(weeks, seasons, [2024], {"WR": 1}, teams=1) == pytest.approx(np.sqrt(8 / 3))
    # Two teams each start one WR (A and B). Their weeks are pooled, so the
    # gap between the two starters counts toward the spread of team scores.
    pooled = np.var([8.0, 12.0, 10.0, 10.0, 1.0, 2.0, 3.0, 2.0], ddof=1)
    assert team_score_sd(weeks, seasons, [2024], {"WR": 1}, teams=2) == pytest.approx(np.sqrt(pooled))


def test_team_score_sd_needs_data():
    seasons = pl.DataFrame(schema={"player_id": pl.String, "season": pl.Int64, "position": pl.String,
                                   "games_played": pl.Int64, "fpts_per_game": pl.Float64})
    with pytest.raises(ValueError):
        team_score_sd(pl.DataFrame(), seasons, [2024], {"WR": 1}, teams=1)
