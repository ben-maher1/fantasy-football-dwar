"""Situation factors on a tiny hand-worked world, plus a no-lookahead test.

Seasons 2020 and 2021, two regular-season weeks each, AAA hosting BBB.

- Q1: AAA's QB both seasons (drafted 2015, round 1), 0.2 EPA per dropback.
- Q2: BBB's QB in 2020 (undrafted), BBB's backup in 2021.
- Q3: rookie QB drafted by BBB in 2021 round 1, BBB's QB1 in 2021.
- W1: WR, AAA in 2020 (6 targets a game), BBB in 2021.
- W2: WR, AAA both seasons (4 targets a game in 2020).
- W3: rookie WR drafted by AAA in 2021 round 2.
- W4: WR, BBB in 2020 (5 targets a game), gone in 2021.

Play-callers: AAA "Cal A" both seasons (every neutral 2020 play +10 pass
rate over expected); BBB "Cal B" in 2020 (-10), "Cal C" in 2021.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from dwar.config import DraftDates, load_scoring
from dwar.features.asof import STAT_KEEP, RawTables, asof_snapshot
from dwar.features.player_tables import build_player_tables
from dwar.features.situation import SituationRaw, covariates, prepare, season_situation, situation_table

SEASONS = (2020, 2021)
WEEKS = (1, 2)
DRAFT_DATES = DraftDates(default_month_day="05-01", end_dates={})
AS_OF = date(2021, 9, 11)  # the day before 2021's first game
PLAYS_PER_TARGET_GAME = 12  # neutral plays per team-game, below

TEAM = {  # (player, season) -> team
    ("Q1", 2020): "AAA", ("Q1", 2021): "AAA",
    ("Q2", 2020): "BBB", ("Q2", 2021): "BBB",
    ("Q3", 2021): "BBB",
    ("W1", 2020): "AAA", ("W1", 2021): "BBB",
    ("W2", 2020): "AAA", ("W2", 2021): "AAA",
    ("W3", 2021): "AAA",
    ("W4", 2020): "BBB",
}
POSITION = {"Q1": "QB", "Q2": "QB", "Q3": "QB", "W1": "WR", "W2": "WR", "W3": "WR", "W4": "WR"}


def gameday(season: int, week: int) -> date:
    return date(season, 9, 13 if season == 2020 else 12) + timedelta(weeks=week - 1)


def game_id(season: int, week: int) -> str:
    return f"{season}_{week:02d}_BBB_AAA"


def schedules() -> pl.DataFrame:
    return pl.DataFrame([
        {"game_id": game_id(s, w), "season": s, "week": w, "game_type": "REG",
         "gameday": gameday(s, w).isoformat(), "home_team": "AAA", "away_team": "BBB"}
        for s in SEASONS for w in WEEKS
    ])


def player_stats() -> pl.DataFrame:
    rows = []
    for (pid, s), team in TEAM.items():
        if pid == "Q2" and s == 2021:
            continue  # the backup never plays
        for w in WEEKS:
            row = {c: 0 for c in STAT_KEEP} | {
                "player_id": pid, "player_display_name": pid, "season": s, "week": w,
                "game_id": game_id(s, w), "team": team, "opponent_team": "BBB" if team == "AAA" else "AAA",
            }
            row |= {"receptions": 3, "receiving_yards": 40} if POSITION[pid] == "WR" else {"passing_yards": 200}
            rows.append(row)
    return pl.DataFrame(rows)


def raw_tables() -> RawTables:
    empty_snaps = pl.DataFrame(schema={
        "game_id": pl.String, "season": pl.Int64, "week": pl.Int64, "game_type": pl.String,
        "pfr_player_id": pl.String, "team": pl.String, "opponent": pl.String, "offense_snaps": pl.Float64,
    })
    empty_combine = pl.DataFrame(schema={
        "season": pl.Int64, "pfr_id": pl.String, "ht": pl.String,
        **{c: pl.Float64 for c in ("wt", "forty", "bench", "vertical", "broad_jump", "cone", "shuttle")},
    })
    return RawTables(
        player_stats=player_stats(),
        schedules=schedules(),
        rosters=pl.DataFrame([
            {"season": s, "gsis_id": p, "position": POSITION[p], "team": t} for (p, s), t in TEAM.items()
        ]),
        draft_picks=pl.DataFrame([
            {"season": 2015, "round": 1, "pick": 5, "team": "AAA", "gsis_id": "Q1", "pfr_player_id": "Q1x", "position": "QB"},
            {"season": 2021, "round": 1, "pick": 2, "team": "BBB", "gsis_id": "Q3", "pfr_player_id": "Q3x", "position": "QB"},
            {"season": 2021, "round": 2, "pick": 40, "team": "AAA", "gsis_id": "W3", "pfr_player_id": "W3x", "position": "WR"},
        ]),
        combine=empty_combine,
        snap_counts=empty_snaps,
        players=pl.DataFrame([
            {"gsis_id": p, "display_name": p, "birth_date": "1996-01-01", "pfr_id": f"{p}x"} for p in POSITION
        ]),
    )


def pbp() -> pl.DataFrame:
    rows = []
    epa = {"Q1": 0.2, "Q2": -0.1}
    for s in SEASONS:
        for w in WEEKS:
            for team in ("AAA", "BBB"):
                qb = {"AAA": "Q1", "BBB": "Q2" if s == 2020 else "Q3"}[team]
                receivers = [p for (p, ss), t in TEAM.items() if ss == s and t == team and POSITION[p] == "WR"]
                targets = [r for r in receivers for _ in range({"W1": 6, "W2": 4, "W4": 5}.get(r, 0))]
                for k in range(PLAYS_PER_TARGET_GAME):
                    rec = targets[k] if k < len(targets) else None
                    rows.append({
                        "game_id": game_id(s, w), "season": s, "week": w, "season_type": "REG",
                        "game_date": gameday(s, w).isoformat(), "posteam": team, "defteam": "BBB" if team == "AAA" else "AAA",
                        "play_type": "pass", "qtr": 1, "down": 1, "score_differential": 0,
                        "pass": 1, "rush": 0, "qb_dropback": 1, "qb_epa": epa.get(qb, 0.0), "passer_player_id": qb,
                        "receiver_player_id": rec, "rusher_player_id": None,
                        "pass_oe": 10.0 if team == "AAA" else -10.0, "xpass": 0.5,
                    })
    return pl.DataFrame(rows, schema_overrides={"receiver_player_id": pl.String, "rusher_player_id": pl.String})


def situation_raw() -> SituationRaw:
    rosters, depth, callers = [], [], []
    for s in SEASONS:
        for w in WEEKS:
            for (p, ss), t in TEAM.items():
                if ss == s:
                    rosters.append({"season": s, "week": w, "game_type": "REG", "team": t, "status": "ACT", "gsis_id": p})
            for team, qb in {"AAA": "Q1", "BBB": "Q2" if s == 2020 else "Q3"}.items():
                depth.append({"season": s, "club_code": team, "week": w, "game_type": "REG", "depth_team": "1",
                              "formation": "Offense", "position": "QB", "gsis_id": qb})
                caller = {"AAA": "Cal A", "BBB": "Cal B" if s == 2020 else "Cal C"}[team]
                callers.append({"season": s, "week": w, "team": team, "game_id": game_id(s, w),
                                "off_play_caller": caller, "def_play_caller": "D", "head_coach": "H"})
    daily = pl.DataFrame(schema={"dt": pl.String, "team": pl.String, "gsis_id": pl.String,
                                 "pos_abb": pl.String, "pos_rank": pl.Int32, "season": pl.Int32})
    return SituationRaw(
        pbp=pbp(), depth_charts=pl.DataFrame(depth), depth_charts_daily=daily,
        rosters_weekly=pl.DataFrame(rosters), schedules=schedules(), playcallers=pl.DataFrame(callers),
    )


def build(raw: RawTables, sraw: SituationRaw, as_of: date = AS_OF) -> pl.DataFrame:
    snap = asof_snapshot(raw, as_of, DRAFT_DATES)
    tables = build_player_tables(snap, load_scoring("ppr"))
    keys = pl.DataFrame(
        [{"player_id": p, "season": s, "position": POSITION[p]} for p in POSITION for s in SEASONS],
        schema={"player_id": pl.String, "season": pl.Int32, "position": pl.String},
    )
    return situation_table(prepare(sraw), snap, tables, keys)


@pytest.fixture
def table() -> pl.DataFrame:
    return build(raw_tables(), situation_raw())


def row(table: pl.DataFrame, pid: str, season: int = 2021) -> dict:
    return table.filter((pl.col("player_id") == pid) & (pl.col("season") == season)).row(0, named=True)


def test_mover_gets_every_change_flag(table):
    w1 = row(table, "W1")
    assert (w1["team"], w1["qb1"], w1["caller"]) == ("BBB", "Q3", "Cal C")
    assert (w1["changed_team"], w1["new_caller"], w1["new_qb"]) == (1.0, 1.0, 1.0)
    assert w1["caller_pass_oe"] == 0.0  # a first-time play-caller gets the league average
    assert w1["vacated"] == pytest.approx(1.0)  # W4's targets all left BBB


def test_stayer_sees_vacated_targets_and_rookie(table):
    w2 = row(table, "W2")
    assert (w2["changed_team"], w2["new_caller"], w2["new_qb"]) == (0.0, 0.0, 0.0)
    assert w2["vacated"] == pytest.approx(12 / 20)  # W1's 2 x 6 of AAA's 2 x 10 targets
    assert w2["rookie_capital"] == 2.0               # W3, round 2
    assert w2["competition"] == 0.0                  # W1 left; Q1 and W3 had no 2020 targets
    # Q1: 0.2 on every dropback, prior for round 1 also 0.2.
    assert w2["qb_quality"] == pytest.approx(0.2)
    n = 2 * PLAYS_PER_TARGET_GAME  # AAA's neutral 2020 plays
    assert w2["caller_pass_oe"] == pytest.approx(10.0 * n / (n + 300))
    assert w2["team_pace"] == pytest.approx(PLAYS_PER_TARGET_GAME)


def test_rookie_counts_returning_competition_not_himself(table):
    w3 = row(table, "W3")
    league_targets_per_game = (10 + 5) / 2
    assert w3["competition"] == pytest.approx(4 / league_targets_per_game)  # W2
    assert w3["rookie_capital"] == 0.0
    assert w3["changed_team"] == 0.0  # no previous season


def test_qb_flags(table):
    q2, q3, q1 = row(table, "Q2"), row(table, "Q3"), row(table, "Q1")
    assert (q3["qb_starter"], q3["qb_threat"]) == (1.0, 0.0)  # the rookie is not a threat to himself
    assert (q2["qb_starter"], q2["qb_threat"], q2["new_caller"]) == (0.0, 1.0, 1.0)
    assert q1["qb_starter"] == 1.0
    assert q1["weapons"] == 0.0  # two-game seasons: no pass catcher reaches the 4-game minimum


def test_no_lookahead_for_the_upcoming_season():
    """Deleting or corrupting everything after the preseason date changes nothing."""
    raw, sraw = raw_tables(), situation_raw()
    full = build(raw, sraw)
    after = pl.col("gameday").str.to_date() >= AS_OF
    later_games = schedules().filter(after)["game_id"]
    cut_raw = replace(raw, player_stats=raw.player_stats.filter(~pl.col("game_id").is_in(later_games.implode())))
    cut = replace(
        sraw,
        pbp=sraw.pbp.filter(~pl.col("game_id").is_in(later_games.implode())),
        # Week 2 rosters and depth charts are knowable only the day before week 2.
        rosters_weekly=sraw.rosters_weekly.filter(~((pl.col("season") == 2021) & (pl.col("week") > 1))),
        depth_charts=sraw.depth_charts.filter(~((pl.col("season") == 2021) & (pl.col("week") > 1))),
        # The opening play-caller is announced preseason; later games are not known.
        playcallers=sraw.playcallers.filter(~((pl.col("season") == 2021) & (pl.col("week") > 1))),
    )
    assert_frame_equal(full, build(cut_raw, cut), check_row_order=False)

    corrupted = replace(
        sraw,
        rosters_weekly=sraw.rosters_weekly.with_columns(
            team=pl.when((pl.col("season") == 2021) & (pl.col("week") == 2)).then(pl.lit("ZZZ")).otherwise("team")
        ),
        playcallers=sraw.playcallers.with_columns(
            off_play_caller=pl.when((pl.col("season") == 2021) & (pl.col("week") == 2))
            .then(pl.lit("Someone Else")).otherwise("off_play_caller")
        ),
        pbp=sraw.pbp.with_columns(qb_epa=pl.when(pl.col("season") == 2021).then(9.9).otherwise("qb_epa")),
    )
    assert_frame_equal(full, build(raw, corrupted), check_row_order=False)


def test_standardized_on_reference_rows_only(table):
    reference = table.filter(pl.col("season") == 2021).select("player_id", "season")
    cov = covariates(table, "WR", reference)
    ref = table.filter((pl.col("season") == 2021) & (pl.col("position") == "WR"))
    mean, sd = cov.center["vacated"]
    assert mean == pytest.approx(ref["vacated"].mean())
    assert sd > 0
    lvl = cov.level.filter(pl.col("season") == 2021)
    assert lvl["vacated"].mean() == pytest.approx(0.0, abs=1e-12)
    assert set(cov.range.columns) == {"player_id", "season", "changed_team", "new_caller", "new_qb"}
    assert cov.level.null_count().sum_horizontal().item() == 0


def test_one_season_matches_the_table(table):
    raw, sraw = raw_tables(), situation_raw()
    snap = asof_snapshot(raw, AS_OF, DRAFT_DATES)
    tables = build_player_tables(snap, load_scoring("ppr"))
    keys = pl.DataFrame({"player_id": ["W1"], "position": ["WR"]})
    one = season_situation(prepare(sraw), snap, tables, keys, 2021)
    assert one.row(0, named=True)["changed_team"] == row(table, "W1")["changed_team"]
