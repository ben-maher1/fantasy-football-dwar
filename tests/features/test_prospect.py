from datetime import date

import polars as pl
import pytest

from dwar.features import prospect as P


def test_norm_name_and_college():
    assert P.norm_name("D.J. Moore Jr.") == "dj moore"
    assert P.norm_name("Will Fuller V") == "will fuller"
    assert P.norm_name("Ja'Marr Chase") == "jamarr chase"
    assert P.norm_college("Ohio St.") == P.norm_college("Ohio State")
    assert P.norm_college("Southern California") == "usc"
    assert P.norm_college("Boston Col.") == "boston college"
    assert P.college_similarity("Mississippi", "Ole Miss") == 1.0
    assert P.college_similarity("Michigan", "Michigan State") < P.COLLEGE_MATCH


def _receiving(rows):
    return pl.DataFrame(
        rows, schema=["season", "cfbd_id", "player", "team", "conference", "rec", "yds", "td"], orient="row"
    ).with_columns(pl.col("season").cast(pl.Int32), pl.col("rec", "yds", "td").cast(pl.Float64))


@pytest.fixture
def world():
    # Team "State": the prospect "1" plus a teammate "2" each season 2018-2021;
    # he enters the NFL in 2021, so his 2021 college season (impossible, but a
    # lookahead trap) must be ignored.
    rec = _receiving([
        (2018, "1", "Joe Prospect", "State", "X", 10, 100, 0),
        (2018, "2", "Other Guy", "State", "X", 40, 900, 9),
        (2019, "1", "Joe Prospect", "State", "X", 30, 600, 6),
        (2019, "2", "Other Guy", "State", "X", 30, 400, 4),
        (2020, "1", "Joe Prospect", "State", "X", 60, 1000, 10),
        (2020, "2", "Other Guy", "State", "X", 20, 1000, 10),
        (2021, "1", "Joe Prospect", "State", "X", 99, 9999, 99),
        (2020, "3", "Tank Smith", "Tech", "Y", 50, 800, 8),
        (2020, "4", "Joe Prospect", "Elsewhere", "Z", 5, 50, 0),
    ])
    team = pl.DataFrame({"season": [2018, 2019, 2020], "team": ["State"] * 3, "pass_att": [400.0, 400.0, 500.0]},
                        schema_overrides={"season": pl.Int32})
    sp = pl.DataFrame({"season": [2019, 2020], "team": ["State", "State"], "sp": [5.0, 12.0]},
                      schema_overrides={"season": pl.Int32})
    seasons = P.college_seasons(rec, team, sp)
    prospects = pl.DataFrame({
        "player_id": ["A", "B", "C"], "name": ["Joe Prospect", "Nathaniel Smith", "Nobody Here"],
        "birth_date": [date(1999, 3, 1), date(1999, 1, 1), date(1999, 1, 1)],
        "college": ["State", "Tech", "State"], "entry_year": [2021, 2021, 2021],
    }, schema_overrides={"entry_year": pl.Int32})
    recruits = pl.DataFrame({"class_year": [2017, 2021], "cfbd_id": ["1", "3"], "name": ["Joe Prospect", "Tank Smith"],
                             "position": ["WR", "WR"], "committed_to": ["State", "Tech"], "stars": [4, 3],
                             "rating": [0.95, 0.85]}, schema_overrides={"class_year": pl.Int32})
    return seasons, prospects, recruits


def test_match(world):
    seasons, prospects, _ = world
    m = dict(zip(*P.match(prospects, seasons).select("player_id", "how")))
    ids = dict(zip(*P.match(prospects, seasons).select("player_id", "cfbd_id")))
    assert m["A"] == "name+college" and ids["A"] == "1"   # the namesake at "Elsewhere" is not taken
    assert m["B"] == "lastname+college" and ids["B"] == "3"
    assert m["C"] == "none" and ids["C"] is None


def test_metrics_values_and_no_lookahead(world):
    seasons, prospects, recruits = world
    out = P.metrics(prospects, P.match(prospects, seasons), seasons, recruits, date(2021, 8, 1))
    a = out.filter(pl.col("player_id") == "A").to_dicts()[0]
    # Dominators: 2018 (100/1000 + 0/9)/2 = .05, 2019 (.6 + .6)/2 = .6, 2020 (.5 + .5)/2 = .5.
    assert a["dominator"] == pytest.approx(0.6)
    assert a["team_sp"] == 5.0                                    # SP+ of the best season, 2019
    assert a["breakout_age"] == pytest.approx((date(2019, 9, 1) - date(1999, 3, 1)).days / 365.25)
    assert a["yptpa"] == pytest.approx(2.0)                       # 1000 / 500 in 2020
    assert a["final_share"] == pytest.approx(60 / 80)            # 2020, not the 2021 lookahead row
    assert a["early_declare"] == 1.0                              # three college seasons
    assert a["recruit"] == 0.95
    b = out.filter(pl.col("player_id") == "B").to_dicts()[0]
    assert b["recruit"] is None                                   # his class year is after his NFL entry: unknown
    assert b["team_sp"] is None and b["yptpa"] is None            # no team tables for "Tech"
    assert "C" not in out["player_id"].to_list()


def test_recruit_name_fallback(world):
    seasons, prospects, recruits = world
    # No athlete id on the recruit row: matched by name, class and school instead.
    recruits = recruits.with_columns(cfbd_id=pl.lit(None, pl.Utf8))
    out = P.metrics(prospects, P.match(prospects, seasons), seasons, recruits, date(2021, 8, 1))
    assert out.filter(pl.col("player_id") == "A")["recruit"].item() == 0.95


def test_never_broke_out_and_old_drafts():
    rec = _receiving([(2006, "9", "Late Bloomer", "U", "X", 5, 50, 0), (2006, "8", "Star", "U", "X", 50, 950, 10)])
    seasons = P.college_seasons(rec, pl.DataFrame(schema={"season": pl.Int32, "team": pl.Utf8, "pass_att": pl.Float64}),
                                pl.DataFrame(schema={"season": pl.Int32, "team": pl.Utf8, "sp": pl.Float64}))
    prospects = pl.DataFrame({"player_id": ["L", "M"], "name": ["Late Bloomer", "Late Bloomer"],
                              "birth_date": [date(1985, 9, 1), date(1985, 9, 1)], "college": ["U", "U"],
                              "entry_year": [2007, 2008]}, schema_overrides={"entry_year": pl.Int32})
    recruits = pl.DataFrame(schema={"class_year": pl.Int32, "cfbd_id": pl.Utf8, "name": pl.Utf8, "committed_to": pl.Utf8,
                                    "rating": pl.Float64})
    out = P.metrics(prospects, P.match(prospects, seasons), seasons, recruits, date(2008, 8, 1))
    rows = {r["player_id"]: r for r in out.to_dicts()}
    # Drafted 2007: his pre-2004 seasons may be missing, so no breakout age or season count.
    assert rows["L"]["breakout_age"] is None and rows["L"]["early_declare"] is None
    # Drafted 2008 having never reached 20%: age in his last season (21) + 1.
    assert rows["M"]["breakout_age"] == pytest.approx(22.0, abs=0.01)


def test_age_reference_uses_only_players_entered_by_as_of(world):
    seasons, prospects, recruits = world
    m = P.match(prospects, seasons)
    early = P.metrics(prospects, m, seasons, recruits, date(2020, 8, 1))   # A has not entered yet
    late = P.metrics(prospects, m, seasons, recruits, date(2021, 8, 1))
    assert early.filter(pl.col("player_id") == "A")["yptpa_age"].item() is None
    assert late.filter(pl.col("player_id") == "A")["yptpa_age"].item() is not None


def test_combine_sources_prefers_box_scores():
    games = _receiving([(2015, "550945", "Will Fuller V", "Notre Dame", "Ind", 62, 1258, 14)]).with_columns(
        games=pl.lit(13, pl.Int32))
    season_rows = _receiving([(2015, "535947", "Chris Brown", "Notre Dame", "Ind", 48, 597, 4),
                              (2015, "77", "FCS Guy", "Small", "FCS", 70, 1200, 12)])
    out = P.combine_sources(games, season_rows)
    assert set(out["player"]) == {"Will Fuller V", "FCS Guy"}
