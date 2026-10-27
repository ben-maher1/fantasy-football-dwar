"""DWAR's season forecast end to end on simulated fits (no sampling)."""

from datetime import date

import numpy as np
import polars as pl

from dwar.features.asof import RawTables
from dwar.validate.backtest import backtest_as_of
from dwar.validate.forecast import forecast_season

from ..model.simulate import simulate
from ..value.test_dwar import fake_fit


def test_forecasts_for_known_players_and_flags_the_rest():
    sim = simulate(n_players=80, seed=9, likelihood="studentt")
    fit = fake_fit(sim, "WR", np.random.default_rng(0))
    last = int(sim.data.ps_keys["season"].max())
    known = sim.data.ps_keys.filter(pl.col("season") >= last - 1)["player_id"].unique().sort().head(5).to_list()
    assert len(known) == 5
    actual = pl.DataFrame({
        "player_id": [*known, "NEW"], "season": [last + 1] * 6, "position": ["WR"] * 6,
        "games_played": [16, 12, 8, 5, 17, 10], "fpts_per_game": [12.0, 8.0, 15.0, 4.0, 9.0, 7.0],
    })
    f = forecast_season({"WR": fit}, "studentt", actual, last + 1, n_draws=100, seed=1)
    assert f.filter(pl.col("player_id") == "NEW")["has_forecast"].to_list() == [False]
    known_rows = f.filter(pl.col("has_forecast"))
    assert known_rows.height == 5
    assert known_rows["pit"].is_between(0, 1).all() and known_rows["log_score"].is_finite().all()
    assert (known_rows["pred_sd"] > 0).all()
    assert known_rows["seasons_since_data"].is_in([1, 2]).all()


def test_backtest_as_of_is_the_day_before_kickoff():
    schedules = pl.DataFrame({
        "season": [2024, 2024, 2024], "game_type": ["REG", "REG", "WC"],
        "gameday": ["2024-09-08", "2024-09-05", "2025-01-11"],
    })
    raw = RawTables(schedules=schedules, **{k: pl.DataFrame() for k in (
        "player_stats", "rosters", "draft_picks", "combine", "snap_counts", "players")})
    assert backtest_as_of(raw, 2024) == date(2024, 9, 4)
