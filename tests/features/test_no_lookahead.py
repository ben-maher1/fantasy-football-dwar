"""No-lookahead tests (CLAUDE.md rule 1).

Core idea: data dated after `as_of` must not change anything. These tests
delete it, corrupt it, probe the boundaries, and check every output row's
knowable date, on synthetic data where the truth is known.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from polars.testing import assert_frame_equal

from dwar.config import load_scoring
from dwar.features.asof import RawTables, asof_snapshot
from dwar.features.player_tables import build_player_tables

from .conftest import DRAFT_DATES, game_id, gameday, make_raw, stat_row

PPR = load_scoring("ppr")
AS_OF_DATES = [
    date(2021, 9, 20),   # mid-2021
    date(2022, 3, 10),   # offseason, before the 2022 combine is knowable
    date(2022, 5, 1),    # day after the 2022 draft
    date(2022, 9, 20),   # mid-2022
    date(2023, 1, 1),    # 2022 regular season over, playoffs pending
    date(2023, 6, 1),    # after the 2023 draft
]
BANNED_COLUMNS = {
    # draft-pick career outcomes
    "car_av", "w_av", "dr_av", "games", "to", "allpro", "probowls", "hof", "seasons_started",
    # players-table fields that describe today
    "status", "latest_team", "last_season", "years_of_experience",
    # combine draft results
    "draft_ovr",
    # nflverse's back-filled position, and its own fantasy totals
    "position_group", "fantasy_points", "fantasy_points_ppr",
}


def build(raw: RawTables, as_of: date) -> dict[str, pl.DataFrame]:
    snap = asof_snapshot(raw, as_of, DRAFT_DATES)
    tables = build_player_tables(snap, PPR)
    return {**{f"snapshot.{k}": v for k, v in snap.frames().items()}, **tables.frames()}


def assert_same(a: dict[str, pl.DataFrame], b: dict[str, pl.DataFrame]) -> None:
    assert a.keys() == b.keys()
    for name in a:
        try:
            assert_frame_equal(a[name], b[name], check_row_order=False)
        except AssertionError as e:
            raise AssertionError(f"{name}: {e}") from e


def truncate(raw: RawTables, as_of: date) -> RawTables:
    """Physically delete everything that happened on or after `as_of`, written
    independently of the gate: as if the data had been downloaded that day."""
    gamedays = raw.schedules.select("game_id", day=pl.col("gameday").str.to_date())
    before = gamedays.filter(pl.col("day") < as_of)["game_id"]
    started = (
        gamedays.join(raw.schedules.select("game_id", "season"), on="game_id")
        .group_by("season").agg(pl.col("day").min())
        .filter(pl.col("day") <= as_of)["season"]
    )
    drafts_done = [y for y in raw.draft_picks["season"].unique() if DRAFT_DATES.end_of(y) < as_of]
    return replace(
        raw,
        player_stats=raw.player_stats.filter(pl.col("game_id").is_in(before.implode())),
        snap_counts=raw.snap_counts.filter(pl.col("game_id").is_in(before.implode())),
        rosters=raw.rosters.filter(pl.col("season").is_in(started.implode())),
        draft_picks=raw.draft_picks.filter(pl.col("season").is_in(drafts_done)),
        combine=raw.combine.filter(pl.date(pl.col("season"), 3, 15) <= as_of),
    )


@pytest.mark.parametrize("as_of", AS_OF_DATES, ids=str)
def test_truncation_invariance(raw, as_of):
    assert_same(build(raw, as_of), build(truncate(raw, as_of), as_of))


@pytest.mark.parametrize("as_of", AS_OF_DATES, ids=str)
def test_poisoned_future_is_ignored(raw, as_of):
    """Absurd future data, changing nothing at or before `as_of`."""
    future_year = as_of.year + 1
    poisoned = replace(
        raw,
        schedules=pl.concat([raw.schedules, pl.DataFrame([{
            "game_id": game_id(future_year, 1), "season": future_year, "week": 1,
            "game_type": "REG", "gameday": gameday(future_year, 1).isoformat(),
        }])]),
        player_stats=pl.concat([raw.player_stats, pl.DataFrame([
            stat_row("00-1", "Vet", "WR", future_year, 1, receptions=999, receiving_yards=9999),
            stat_row("00-9", "Ghost", "WR", future_year, 1, receptions=50),
        ])], how="diagonal_relaxed"),
        rosters=pl.concat([raw.rosters, pl.DataFrame([
            {"season": future_year, "gsis_id": "00-1", "position": "QB", "team": "ZZZ"},
            {"season": future_year, "gsis_id": "00-9", "position": "WR", "team": "ZZZ"},
        ])]),
        draft_picks=pl.concat([raw.draft_picks, pl.DataFrame([
            {"season": future_year, "round": 1, "pick": 1, "team": "ZZZ", "gsis_id": "00-9",
             "pfr_player_id": "Ghos00", "position": "WR"},
        ])]),
        combine=pl.concat([raw.combine, pl.DataFrame([
            {"season": future_year, "pfr_id": "VetX00", "ht": "7-0", "wt": 400.0, "forty": 3.0,
             "bench": 99.0, "vertical": 99.0, "broad_jump": 999.0, "cone": 1.0, "shuttle": 1.0},
        ])]),
        players=pl.concat([raw.players, pl.DataFrame([
            {"gsis_id": "00-9", "display_name": "Ghost", "birth_date": "2003-01-01", "pfr_id": "Ghos00",
             "status": "ACT", "latest_team": "ZZZ", "last_season": 2026},
        ])]),
    )
    clean = build(raw, as_of)
    dirty = build(poisoned, as_of)
    # The snapshot's schedules legitimately differ only if the future season's
    # schedule was already published; it is not, for every date in AS_OF_DATES.
    assert_same(clean, dirty)
    assert "00-9" not in dirty["player_static"]["player_id"].to_list()


def player_game_ids(raw: RawTables, as_of: date, pid: str) -> list[str]:
    weeks = build(raw, as_of)["player_weeks"]
    return weeks.filter(pl.col("player_id") == pid)["game_id"].to_list()


def test_game_boundary(raw):
    day = gameday(2022, 2)
    assert game_id(2022, 2) not in player_game_ids(raw, day, "00-1")
    assert game_id(2022, 2) in player_game_ids(raw, day + timedelta(days=1), "00-1")


def test_draft_boundary(raw):
    end = DRAFT_DATES.end_of(2022)
    assert "00-3" not in build(raw, end)["player_static"]["player_id"].to_list()
    assert "00-3" in build(raw, end + timedelta(days=1))["player_static"]["player_id"].to_list()


def test_combine_boundary(raw):
    def forty(as_of: date):
        s = build(raw, as_of)["player_static"].filter(pl.col("player_id") == "00-3")
        return s["forty"].to_list()

    # Rookie is drafted 2022-04-30, so check combine visibility via the snapshot.
    def snap_forty(as_of: date):
        return asof_snapshot(raw, as_of, DRAFT_DATES).combine["forty"].to_list()

    assert snap_forty(date(2022, 3, 14)) == []
    assert snap_forty(date(2022, 3, 15)) == [4.45]
    assert forty(date(2022, 5, 1)) == [4.45]


def test_position_comes_from_rosters_not_backfilled_stats(raw):
    seasons = build(raw, date(2023, 6, 1))["player_seasons"].filter(pl.col("player_id") == "00-2")
    assert dict(zip(seasons["season"], seasons["position"])) == {2021: "WR", 2022: "RB"}


def test_unstarted_season_roster_is_invisible(raw):
    snap = asof_snapshot(raw, date(2022, 8, 1), DRAFT_DATES)
    assert snap.rosters["season"].max() == 2021


def test_playoffs_excluded(raw):
    weeks = build(raw, date(2023, 6, 1))["player_weeks"]
    assert weeks["week"].max() == 3


def test_banned_columns_absent(raw):
    for name, frame in build(raw, date(2023, 6, 1)).items():
        leaked = BANNED_COLUMNS & set(frame.columns)
        assert not leaked, f"{name} exposes {leaked}"


@settings(max_examples=60, deadline=None)
@given(as_of=st.dates(min_value=date(2021, 1, 1), max_value=date(2023, 12, 31)))
def test_every_row_knowable_by_as_of(as_of):
    raw = make_raw()
    for name, frame in build(raw, as_of).items():
        if frame.height and "knowable_on" in frame.columns:
            assert frame["knowable_on"].max() <= as_of, name
        if frame.height and "gameday" in frame.columns and not name.endswith("schedules"):
            assert frame["gameday"].max() < as_of, name
    # Every table must carry the column the check relies on.
    for name, frame in build(raw, as_of).items():
        assert "knowable_on" in frame.columns, name


@pytest.mark.network
def test_real_data_spot_check():
    """2023-09-01 on the real cache (pulls history if missing)."""
    from dwar.ingest.nflverse import pull

    for name in ("player_stats", "schedules", "rosters", "draft_picks", "combine", "snap_counts", "players"):
        pull(name)
    raw = RawTables.from_cache()
    as_of = date(2023, 9, 1)
    snap = asof_snapshot(raw, as_of)
    tables = build_player_tables(snap, PPR)

    assert snap.stats["gameday"].max() == date(2023, 1, 8)  # last 2022 regular-season Sunday
    static = tables.player_static
    names = set(static["name"].to_list())
    assert "Bijan Robinson" in names                      # 2023 draftee: visible, no games
    assert "Marvin Harrison Jr." not in names             # 2024 draftee: invisible
    bijan = static.filter(pl.col("name") == "Bijan Robinson")
    assert bijan["career_games"].item() is None
    patterson = tables.player_seasons.join(
        static.select("player_id", "name"), on="player_id"
    ).filter(pl.col("name") == "Cordarrelle Patterson")
    positions = dict(zip(patterson["season"], patterson["position"]))
    assert positions[2016] == "WR" and positions[2022] == "RB"
