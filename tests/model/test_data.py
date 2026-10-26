from datetime import date

import numpy as np
import polars as pl
import pytest

from dwar.features.player_tables import PlayerTables
from dwar.model.data import UNDRAFTED_PICK, build_model_data, standardize_with_flags


def tables(weeks: list[tuple], ages: dict, static: list[dict]) -> PlayerTables:
    w = pl.DataFrame(weeks, schema=["player_id", "season", "position", "fpts"], orient="row")
    seasons = w.select("player_id", "season").unique().with_columns(
        age_sept1=pl.struct("player_id", "season").map_elements(
            lambda r: ages.get(r["player_id"], {}).get(r["season"]), return_dtype=pl.Float64
        )
    )
    return PlayerTables(
        as_of=date(2025, 1, 1),
        player_weeks=w,
        player_seasons=seasons,
        player_static=pl.DataFrame(static, schema_overrides={"draft_pick": pl.Int64, "forty": pl.Float64}),
    )


@pytest.fixture
def small() -> PlayerTables:
    weeks = [
        # A: 2010, 2011, then a gap year, 2013
        ("A", 2010, "WR", 10.0), ("A", 2010, "WR", 12.0), ("A", 2011, "WR", 8.0), ("A", 2013, "WR", 9.0),
        # B: undrafted, one season; also an RB game that must be ignored
        ("B", 2012, "WR", 3.0), ("B", 2012, "RB", 30.0),
        # C: veteran whose career began before the window; a 2004 game outside it
        ("C", 2004, "WR", 15.0), ("C", 2007, "WR", 14.0),
    ]
    ages = {"A": {2010: 22.0, 2011: 23.0, 2013: 25.0}, "B": {2012: 24.0}, "C": {2004: 26.0, 2007: 29.0}}
    static = [
        {"player_id": "A", "draft_pick": 10, "first_season": 2010, "forty": 4.40, "ht_in": 72, "wt": 200.0},
        {"player_id": "B", "draft_pick": None, "first_season": 2012, "forty": None, "ht_in": 70, "wt": 190.0},
        {"player_id": "C", "draft_pick": 100, "first_season": 2001, "forty": 4.60, "ht_in": 74, "wt": 210.0},
    ]
    return tables(weeks, ages, static)


def test_player_season_index(small):
    d = build_model_data(small, first_season=2006)
    keys = d.ps_keys.select("player_id", "season").rows()
    assert keys == [("A", 2010), ("A", 2011), ("A", 2013), ("B", 2012), ("C", 2007)]
    assert d.ps_player.tolist() == [0, 0, 0, 1, 2]
    assert d.ps_is_first.tolist() == [True, False, False, True, True]
    assert d.ps_gap.tolist() == [0, 1, 2, 0, 0]
    assert d.ps_start.tolist() == [0, 0, 0, 3, 4]
    assert d.ps_age_c.tolist() == [-4.0, -3.0, -1.0, -2.0, 3.0]


def test_observations_filtered_and_indexed(small):
    d = build_model_data(small, first_season=2006)
    # the RB game and the pre-window 2004 game are excluded
    assert d.y.tolist() == [10.0, 12.0, 8.0, 9.0, 3.0, 14.0]
    assert d.obs_ps.tolist() == [0, 0, 1, 2, 3, 4]
    assert d.ps_keys["games_played"].to_list() == [2, 1, 1, 1, 1]


def test_draft_features(small):
    d = build_model_data(small, first_season=2006)
    assert d.undrafted.tolist() == [0.0, 1.0, 0.0]
    assert d.pre_window.tolist() == [0.0, 0.0, 1.0]
    drafted = np.log([10, 100])
    mu, sd = drafted.mean(), drafted.std(ddof=1)
    expected = (np.log([10, UNDRAFTED_PICK, 100]) - mu) / sd
    np.testing.assert_allclose(d.log_pick_z, expected)


def test_combine_standardized_with_missing_flag(small):
    d = build_model_data(small, first_season=2006)
    assert d.prospect_x_names == ("forty_z", "ht_in_z", "wt_z", "forty_missing")
    forty_z = d.prospect_x[:, 0]
    assert forty_z[1] == 0.0 and d.prospect_x[1, 3] == 1.0  # B has no 40: zero plus a flag
    assert forty_z[0] < 0 < forty_z[2]


def test_standardize_with_flags_omits_constant_flags():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, None, 3.0]})
    out = standardize_with_flags(df, ("x", "y"))
    assert out.columns == ["x_z", "y_z", "y_missing"]
    np.testing.assert_allclose(out["x_z"].to_numpy(), [-1.0, 0.0, 1.0])


def test_standardize_with_flags_merges_identical_flags():
    df = pl.DataFrame({"h": [70.0, None, 72.0], "w": [200.0, None, 210.0], "f": [None, 4.5, 4.4]})
    out = standardize_with_flags(df, ("f", "h", "w"))
    assert out.columns == ["f_z", "h_z", "w_z", "f_missing", "h_w_missing"]
    assert out["h_w_missing"].to_list() == [0.0, 1.0, 0.0]


def test_season_covariates_slot(small):
    covs = pl.DataFrame({
        "player_id": ["C", "A", "A", "A", "B"], "season": [2007, 2010, 2011, 2013, 2012],
        "qb_epa": [0.5, 0.1, 0.2, 0.3, 0.4],
    })
    d = build_model_data(small, first_season=2006, season_covariates=covs)
    assert d.season_x_names == ("qb_epa",)
    assert d.season_x[:, 0].tolist() == [0.1, 0.2, 0.3, 0.4, 0.5]  # aligned to ps order


def test_season_covariates_must_be_complete(small):
    covs = pl.DataFrame({"player_id": ["A"], "season": [2010], "qb_epa": [0.1]})
    with pytest.raises(ValueError, match="every player-season"):
        build_model_data(small, first_season=2006, season_covariates=covs)


def test_prospect_covariates_slot(small):
    extra = pl.DataFrame({"player_id": ["B", "A", "C"], "breakout_age": [19.0, 20.0, 21.0]})
    d = build_model_data(small, first_season=2006, prospect_covariates=extra)
    assert d.prospect_x_names[-1] == "breakout_age"
    assert d.prospect_x[:, -1].tolist() == [20.0, 19.0, 21.0]


def test_zero_week_mean():
    weeks = [("A", 2010, "QB", -2.0), ("A", 2010, "QB", 0.0), ("A", 2010, "QB", 20.0), ("A", 2011, "QB", 15.0)]
    static = [{"player_id": "A", "draft_pick": 1, "first_season": 2010, "forty": None, "ht_in": 75, "wt": 220.0}]
    d = build_model_data(tables(weeks, {"A": {2010: 22.0, 2011: 23.0}}, static), "QB")
    assert d.zero_week_mean == -1.0


def test_empty_position_raises(small):
    with pytest.raises(ValueError, match="no TE games"):
        build_model_data(small, position="TE")
