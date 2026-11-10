from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from dwar.features.injuries import (
    GROUPS,
    InjuryRaw,
    body_group,
    current_status,
    injury_seasons,
    injury_weeks,
    team_weeks,
)
from dwar.value.availability import (
    INJURY_WEEKS_UNIT,
    availability_rows,
    injury_history_terms,
    injury_terms,
    status_effects,
    status_multiplier,
)


@pytest.mark.parametrize(("text", "group"), [
    ("Knee", "knee"), ("Right Knee", "knee"), ("Hamstring", "soft_tissue"), ("quadricep", "soft_tissue"),
    ("Concussion", "concussion"), ("Achilles", "achilles"), ("Ankle", "ankle_foot"), ("Toe", "ankle_foot"),
    ("Right Thumb", "hand_arm"), ("Collarbone", "hand_arm"), ("Ribs", "torso"), ("chest/rib/finger", "torso"),
    ("Fibula", "leg_bone"), ("Back", "hip_back"), ("Illness", "none"), ("Not injury related - resting player", "none"),
    ("Coach's Decision", "none"), (None, "other"), ("Spleen", "none"), ("Something odd", "other"),
])
def test_body_groups(text, group):
    assert body_group(text) == group


START = date(2024, 9, 8)
SEASON_WEEKS = 6


def schedules(seasons=(2023, 2024)) -> pl.DataFrame:
    rows = []
    for s in seasons:
        for w in range(1, SEASON_WEEKS + 1):
            rows.append({"season": s, "week": w, "game_type": "REG", "home_team": "AAA", "away_team": "OAK",
                         "gameday": (date(s, 9, 8) + timedelta(weeks=w - 1)).isoformat()})
    return pl.DataFrame(rows)


def roster(pid: str, season: int, statuses: list[str], codes: list[str] | None = None) -> list[dict]:
    codes = codes or ["A01"] * len(statuses)
    return [{"season": season, "week": w, "game_type": "REG", "team": "AAA", "gsis_id": pid, "position": "WR",
             "status": st, "status_description_abbr": c} for w, (st, c) in enumerate(zip(statuses, codes), start=1)]


def report(pid: str, season: int, week: int, part: str | None, status: str | None) -> dict:
    return {"season": float(season), "week": float(week), "game_type": "REG", "team": "AAA", "gsis_id": pid,
            "report_primary_injury": part, "practice_primary_injury": None, "report_status": status}


def world():
    """P: 2024 hamstring Out week 2, back week 3, knee Out week 4 then IR
    weeks 5-6. Q: suspended (R40) weeks 1-2, rested week 3 (Out, not
    injury). R: healthy, a bye-like missed week with no status (not an injury)."""
    rw = (roster("P", 2024, ["ACT", "ACT", "ACT", "ACT", "RES", "RES"], ["A01"] * 4 + ["R01", "R01"])
          + roster("Q", 2024, ["RES", "RES", "ACT", "ACT", "ACT", "ACT"], ["R40", "R40"] + ["A01"] * 4)
          + roster("R", 2024, ["ACT"] * 6))
    reps = [report("P", 2024, 2, "Hamstring", "Out"), report("P", 2024, 4, "Right Knee", "Out"),
            report("Q", 2024, 3, "Not injury related - resting player", "Out")]
    played = [("P", w) for w in (1, 3)] + [("Q", w) for w in (4, 5, 6)] + [("R", w) for w in (1, 2, 3, 5, 6)]
    played = pl.DataFrame({"player_id": [p for p, _ in played], "season": [2024] * len(played),
                           "week": [w for _, w in played]})
    raw = InjuryRaw(pl.DataFrame(reps), pl.DataFrame(rw))
    return raw, played


def test_weeks_missed_by_group():
    raw, played = world()
    w = injury_weeks(raw, schedules(), played, date(2025, 1, 1))
    s = injury_seasons(w).filter(pl.col("season") == 2024).sort("player_id")
    p = s.filter(pl.col("player_id") == "P").row(0, named=True)
    assert p["missed"] == 4 and p["missed_soft_tissue"] == 1 and p["missed_knee"] == 3  # IR weeks inherit knee
    assert s.filter(pl.col("player_id") == "Q")["missed"].item() == 0  # suspension and rest are not injuries
    assert s.filter(pl.col("player_id") == "R")["missed"].item() == 0  # a missed game without status
    assert set(GROUPS) >= {"knee", "soft_tissue"}


def test_no_lookahead_and_current_status():
    raw, played = world()
    as_of = date(2024, 9, 8) + timedelta(weeks=3)  # week 4's game day: weeks 1-4 knowable
    w = injury_weeks(raw, schedules(), played, as_of)
    assert w["week"].max() == 4
    later = injury_weeks(raw, schedules(), played, date(2025, 1, 1)).filter(pl.col("week") <= 4)
    assert w.sort("player_id", "week").equals(later.sort("player_id", "week").select(w.columns))
    st = current_status(w, 2024).sort("player_id")
    p = st.filter(pl.col("player_id") == "P").row(0, named=True)
    assert p["injured"] and not p["reserve"] and p["streak"] == 1
    full = current_status(injury_weeks(raw, schedules(), played, date(2025, 1, 1)), 2024)
    p = full.filter(pl.col("player_id") == "P").row(0, named=True)
    assert p["injured"] and p["reserve"] and p["streak"] == 3


def test_team_codes_follow_franchise():
    assert set(team_weeks(schedules())["team"].to_list()) == {"AAA", "LV"}


def test_injury_terms_scale_and_cap():
    groups = np.zeros(len(GROUPS))
    groups[GROUPS.index("knee")] = 8.0
    t = injury_terms(groups, 20.0, extra_last=4.0, groups=True)
    assert injury_terms(groups, 20.0).shape == (2,)  # totals only by default
    assert t[0] == pytest.approx(12.0 / INJURY_WEEKS_UNIT)
    assert t[1] == pytest.approx(17.0 / INJURY_WEEKS_UNIT)  # capped at a season
    assert t[2 + GROUPS.index("knee")] == pytest.approx(2.0)


def test_history_terms_use_only_earlier_seasons():
    raw, played = world()
    inj = injury_seasons(injury_weeks(raw, schedules(), played, date(2025, 1, 1)))
    keys = pl.DataFrame({"player_id": ["P", "P"], "t": [2024, 2025]})
    terms = injury_history_terms(keys, inj)
    assert terms.shape == (2, 2)
    assert terms[0].sum() == 0  # predicting 2024 sees nothing from 2024
    assert terms[1, 0] == pytest.approx(4 / INJURY_WEEKS_UNIT)


def test_availability_rows_with_injury_start_after_two_report_seasons():
    sched = pl.DataFrame([{"season": s, "gameday": date(s, 9, 10), "home_team": "A", "away_team": "B"}
                          for s in range(2008, 2014)])
    talent = pl.DataFrame({"player_id": ["P"] * 5, "season": list(range(2009, 2014)), "position": ["WR"] * 5,
                           "games_played": [1] * 5, "talent_mean": [10.0] * 5})
    seasons = talent.select("player_id", "season", age_sept1=pl.lit(25.0))
    inj = pl.DataFrame({"player_id": ["P"], "season": [2010], "team_games": [1], "missed": [2]}).with_columns(
        **{f"missed_{g}": pl.lit(2 if g == "knee" else 0) for g in GROUPS})
    games, survival = availability_rows(talent, seasons, sched, date(2014, 1, 1), inj, groups=True)
    assert games.height == 3  # 2011, 2012, 2013
    terms = np.array(games["injury"].to_list())
    # Predicting 2011 sees 2010's knee weeks as last season, 2012 as the season before.
    knee = 2 + GROUPS.index("knee")
    assert sorted(map(tuple, terms[:, [0, 1, knee]].round(3))) == [(0, 0, 0), (0, 0.5, 0), (0.5, 0, 0.5)]
    assert survival.height == 4  # survival has no injury terms: every complete pair from 2009


def test_status_effects_and_multiplier():
    raw, played = world()
    w = injury_weeks(raw, schedules(), played, date(2025, 1, 1))
    eff = status_effects(w, [2024])
    assert eff[("healthy", 0)] == 1.0
    assert all(0 < v <= 1 for v in eff.values())
    status = pl.DataFrame({"player_id": ["P", "R"], "injured": [True, False], "reserve": [True, False], "streak": [5, 0]})
    m = status_multiplier(status, np.array([3, 3, 3]), {("reserve_4plus", 0): 0.1}, ["P", "R", "Z"])
    assert m.tolist() == [0.1, 1.0, 1.0]
