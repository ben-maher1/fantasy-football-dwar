from datetime import date

import polars as pl
import pytest

from dwar.config import load_scoring
from dwar.features.asof import asof_snapshot
from dwar.features.player_tables import build_player_tables, write_player_tables

from .conftest import DRAFT_DATES

AFTER_2022 = date(2023, 6, 1)
MID_2022 = date(2022, 9, 20)  # after weeks 1-2 of 2022, before week 3


def tables(raw, as_of, scoring="ppr"):
    return build_player_tables(asof_snapshot(raw, as_of, DRAFT_DATES), load_scoring(scoring))


def row(df: pl.DataFrame, **match) -> dict:
    out = df.filter(*[pl.col(k) == v for k, v in match.items()])
    assert out.height == 1, out
    return out.row(0, named=True)


def test_points_per_game(raw):
    seasons = tables(raw, AFTER_2022).player_seasons
    vet = row(seasons, player_id="00-1", season=2022)
    # 5 rec, 60 yd per game in PPR = 11.0; three regular-season games
    assert vet["games_played"] == 3
    assert vet["fpts_total"] == pytest.approx(33.0)
    assert vet["fpts_per_game"] == pytest.approx(11.0)


def test_scoring_config_flows_through(raw):
    std = row(tables(raw, AFTER_2022, "standard").player_seasons, player_id="00-1", season=2022)
    assert std["fpts_per_game"] == pytest.approx(6.0)


def test_snap_only_week_counts_as_zero_point_game(raw):
    t = tables(raw, AFTER_2022)
    blocker = t.player_weeks.filter(pl.col("player_id") == "00-4").sort("week")
    # week 1 from stats, week 2 snaps only, week 3 had 0 offense snaps (not a game played)
    assert blocker["week"].to_list() == [1, 2]
    assert blocker["source"].to_list() == ["stats", "snaps_only"]
    assert blocker["fpts"].to_list() == pytest.approx([1.5, 0.0])
    season = row(t.player_seasons, player_id="00-4", season=2022)
    assert season["games_played"] == 2
    assert season["fpts_per_game"] == pytest.approx(0.75)


def test_season_complete_flag(raw):
    mid = tables(raw, MID_2022).player_seasons
    assert row(mid, player_id="00-1", season=2022)["season_complete"] is False
    assert row(mid, player_id="00-1", season=2022)["games_played"] == 2
    assert row(mid, player_id="00-1", season=2021)["season_complete"] is True


def test_ages(raw):
    t = tables(raw, AFTER_2022)
    # Vet born 1995-01-01 is 27.666 on Sept 1 2022
    assert row(t.player_seasons, player_id="00-1", season=2022)["age_sept1"] == pytest.approx(27.666)
    assert row(t.player_static, player_id="00-1")["age_at_as_of"] == pytest.approx(
        round((AFTER_2022 - date(1995, 1, 1)).days / 365.25, 3)
    )


def test_non_skill_positions_excluded(raw):
    t = tables(raw, AFTER_2022)
    for frame in t.frames().values():
        assert "00-6" not in frame["player_id"].to_list()
        assert set(frame["position"].unique()) <= {"QB", "RB", "WR", "TE"}


def test_static_rookie_and_draft_only_players(raw):
    static = tables(raw, AFTER_2022).player_static
    rookie = row(static, player_id="00-3")  # draft pick had no gsis_id; matched via pfr id
    assert (rookie["draft_year"], rookie["draft_pick"], rookie["ht_in"], rookie["forty"]) == (2022, 10, 71, 4.45)
    assert rookie["first_season"] == 2022 and rookie["career_games"] == 2
    future = row(static, player_id="00-5")  # drafted 2023, no games yet
    assert future["career_games"] is None
    assert (future["position"], future["team"], future["name"]) == ("WR", "CCC", "Future")


def test_static_team_is_last_game_team(raw):
    static = tables(raw, AFTER_2022).player_static
    assert row(static, player_id="00-1")["team"] == "AAA"  # not players.latest_team ("ZZZ")


def test_write_player_tables(raw, tmp_path):
    paths = write_player_tables(tables(raw, AFTER_2022), "ppr", root=tmp_path)
    assert sorted(p.name for p in paths) == ["player_seasons.parquet", "player_static.parquet", "player_weeks.parquet"]
    assert all(p.parent == tmp_path / "2023-06-01" / "ppr" for p in paths)
