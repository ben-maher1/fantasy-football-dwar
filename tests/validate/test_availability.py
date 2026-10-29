"""Availability check end to end on simulated fits (no sampling)."""

import polars as pl

from dwar.validate.availability import availability_forecast, summarize_availability

from ..value.test_dwar import end_to_end  # noqa: F401  (fixture)


def test_availability_forecast_covers_last_seasons_players(end_to_end):  # noqa: F811
    fits, tables, sched, _, _ = end_to_end
    truth = tables.player_seasons
    season = int(truth["season"].max())
    rows = availability_forecast(fits, tables, sched, "studentt", truth, season, n_draws=50, seed=1)

    prev = truth.filter((pl.col("season") == season - 1) & (pl.col("games_played") > 0))
    assert sorted(rows["player_id"].to_list()) == sorted(prev["player_id"].to_list())
    now = set(truth.filter((pl.col("season") == season) & (pl.col("games_played") > 0))["player_id"])
    assert rows["survived"].to_list() == [p in now for p in rows["player_id"].to_list()]
    assert rows["p_survive"].is_between(0, 1).all()
    survivors = rows.filter(pl.col("survived"))
    assert survivors["games_pit"].is_between(0, 1).all()
    assert rows.filter(~pl.col("survived"))["games_pit"].is_null().all()

    s = summarize_availability(rows, ("WR", "RB"))
    assert set(s) == {"all", "WR", "RB"}
    assert s["all"]["survival"]["n"] == rows.height
    assert sum(b["n"] for b in s["all"]["survival"]["calibration"]) == rows.height
    assert 0 <= s["all"]["games"]["coverage_50"] <= s["all"]["games"]["coverage_90"] <= 1
