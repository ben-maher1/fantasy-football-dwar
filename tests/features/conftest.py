"""A tiny synthetic nflverse world for point-in-time tests.

Seasons 2021 and 2022, three regular-season weeks each (Sept 12/19/26) plus
one playoff game the following January. Players:

- 00-1 Vet: WR, drafted 2018, plays every game in both seasons.
- 00-2 Switch: rostered WR in 2021 and RB in 2022. nflverse stats label him
  RB in both seasons, the back-filled "today" position the gate must ignore.
- 00-3 Rookie: drafted 2022 (draft ends 2022-04-30), combine 2022, plays 2022.
- 00-4 Blocker: TE with a 2022 week of offense snaps but no stat row.
- 00-5 Future: drafted 2023 (draft ends 2023-04-29), combine 2023, no games.
- 00-6 Kicker: K, must never appear in skill tables.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from dwar.config import DraftDates
from dwar.features.asof import STAT_KEEP, RawTables

SEASONS = (2021, 2022)
DRAFT_DATES = DraftDates(
    default_month_day="05-15",
    end_dates={2018: date(2018, 4, 28), 2022: date(2022, 4, 30), 2023: date(2023, 4, 29)},
)


def game_id(season: int, week: int) -> str:
    return f"{season}_{week:02d}_AAA_BBB"


def gameday(season: int, week: int) -> date:
    if week > 3:  # the playoff game
        return date(season + 1, 1, 20)
    return date(season, 9, 12) + timedelta(weeks=week - 1)


def schedules() -> pl.DataFrame:
    rows = []
    for s in SEASONS:
        for w in (1, 2, 3, 4):
            rows.append({
                "game_id": game_id(s, w), "season": s, "week": w,
                "game_type": "REG" if w <= 3 else "WC",
                "gameday": gameday(s, w).isoformat(),
            })
    return pl.DataFrame(rows)


def stat_row(pid: str, name: str, position: str, season: int, week: int, **stats) -> dict:
    row = {c: 0 for c in STAT_KEEP}
    row |= {
        "player_id": pid, "player_display_name": name, "position": position,
        "season": season, "week": week, "game_id": game_id(season, week),
        "team": "AAA", "opponent_team": "BBB", "season_type": "REG" if week <= 3 else "POST",
        "fantasy_points_ppr": 0.0,
    }
    return row | stats


def player_stats() -> pl.DataFrame:
    rows = []
    for s in SEASONS:
        for w in (1, 2, 3, 4):
            rows.append(stat_row("00-1", "Vet", "WR", s, w, receptions=5, receiving_yards=60))
            rows.append(stat_row("00-2", "Switch", "RB", s, w, carries=10, rushing_yards=40))
            rows.append(stat_row("00-6", "Kicker", "K", s, w))
        if s == 2022:
            for w in (1, 2):
                rows.append(stat_row("00-3", "Rookie", "RB", s, w, rushing_yards=80, rushing_tds=1))
            rows.append(stat_row("00-4", "Blocker", "TE", s, 1, receptions=1, receiving_yards=5))
    return pl.DataFrame(rows)


def snap_counts() -> pl.DataFrame:
    rows = [
        {"game_id": game_id(2022, w), "season": 2022, "week": w, "game_type": "REG",
         "pfr_player_id": "Bloc00", "team": "AAA", "opponent": "BBB", "offense_snaps": 30.0}
        for w in (1, 2)
    ]
    rows.append({"game_id": game_id(2022, 3), "season": 2022, "week": 3, "game_type": "REG",
                 "pfr_player_id": "Bloc00", "team": "AAA", "opponent": "BBB", "offense_snaps": 0.0})
    return pl.DataFrame(rows)


def rosters() -> pl.DataFrame:
    rows = []
    for s in SEASONS:
        rows += [
            {"season": s, "gsis_id": "00-1", "position": "WR", "team": "AAA"},
            {"season": s, "gsis_id": "00-2", "position": "WR" if s == 2021 else "RB", "team": "AAA"},
            {"season": s, "gsis_id": "00-4", "position": "TE", "team": "AAA"},
            {"season": s, "gsis_id": "00-6", "position": "K", "team": "AAA"},
        ]
    rows.append({"season": 2022, "gsis_id": "00-3", "position": "RB", "team": "AAA"})
    return pl.DataFrame(rows)


def draft_picks() -> pl.DataFrame:
    return pl.DataFrame([
        {"season": 2018, "round": 2, "pick": 40, "team": "AAA", "gsis_id": "00-1",
         "pfr_player_id": "VetX00", "position": "WR"},
        {"season": 2022, "round": 1, "pick": 10, "team": "AAA", "gsis_id": None,
         "pfr_player_id": "Rook00", "position": "RB"},
        {"season": 2023, "round": 1, "pick": 3, "team": "CCC", "gsis_id": "00-5",
         "pfr_player_id": "Futu00", "position": "WR"},
    ])


def combine() -> pl.DataFrame:
    return pl.DataFrame([
        {"season": 2022, "pfr_id": "Rook00", "ht": "5-11", "wt": 215.0, "forty": 4.45,
         "bench": None, "vertical": 36.0, "broad_jump": 120.0, "cone": None, "shuttle": None},
        {"season": 2023, "pfr_id": "Futu00", "ht": "6-1", "wt": 200.0, "forty": 4.38,
         "bench": None, "vertical": 40.0, "broad_jump": 130.0, "cone": None, "shuttle": None},
    ], schema_overrides={"bench": pl.Float64, "cone": pl.Float64, "shuttle": pl.Float64})


def players() -> pl.DataFrame:
    return pl.DataFrame([
        # status / latest_team / last_season describe today and must never pass the gate.
        {"gsis_id": "00-1", "display_name": "Vet", "birth_date": "1995-01-01", "pfr_id": "VetX00",
         "status": "ACT", "latest_team": "ZZZ", "last_season": 2026},
        {"gsis_id": "00-2", "display_name": "Switch", "birth_date": "1997-06-15", "pfr_id": "Swit00",
         "status": "ACT", "latest_team": "ZZZ", "last_season": 2026},
        {"gsis_id": "00-3", "display_name": "Rookie", "birth_date": "2000-09-01", "pfr_id": "Rook00",
         "status": "ACT", "latest_team": "ZZZ", "last_season": 2026},
        {"gsis_id": "00-4", "display_name": "Blocker", "birth_date": "1996-03-03", "pfr_id": "Bloc00",
         "status": "ACT", "latest_team": "ZZZ", "last_season": 2026},
        {"gsis_id": "00-5", "display_name": "Future", "birth_date": "2001-02-02", "pfr_id": "Futu00",
         "status": "ACT", "latest_team": "ZZZ", "last_season": 2026},
        {"gsis_id": "00-6", "display_name": "Kicker", "birth_date": "1990-01-01", "pfr_id": "Kick00",
         "status": "ACT", "latest_team": "ZZZ", "last_season": 2026},
    ])


def make_raw() -> RawTables:
    return RawTables(
        player_stats=player_stats(), schedules=schedules(), rosters=rosters(),
        draft_picks=draft_picks(), combine=combine(), snap_counts=snap_counts(), players=players(),
    )


@pytest.fixture
def raw() -> RawTables:
    return make_raw()
