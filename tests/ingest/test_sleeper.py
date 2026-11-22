import json

import polars as pl

from dwar.ingest import sleeper as S

LEAGUE = {"season": "2026", "settings": {"draft_rounds": 2, "last_scored_leg": 2}}
ROSTERS = [
    {"roster_id": 1, "owner_id": "u1", "players": ["10", "11", "12"], "reserve": ["11"], "taxi": ["12"],
     "settings": {"wins": 2, "losses": 0, "fpts": 250, "fpts_decimal": 50, "division": 1}},
    {"roster_id": 2, "owner_id": "u2", "players": ["20"], "settings": {"wins": 0, "losses": 2, "fpts": 200}},
]
USERS = [{"user_id": "u1", "display_name": "ben"}, {"user_id": "u2", "display_name": "rival"}]
TRADED = [{"season": "2027", "round": 1, "roster_id": 2, "owner_id": 1, "previous_owner_id": 2},
          {"season": "2026", "round": 1, "roster_id": 1, "owner_id": 2, "previous_owner_id": 1}]
MATCHUPS = {1: [{"roster_id": 1, "matchup_id": 1, "points": 130.5}, {"roster_id": 2, "matchup_id": 1, "points": 99.0}],
            2: [{"roster_id": 1, "matchup_id": 1, "points": 120.0}, {"roster_id": 2, "matchup_id": 1, "points": 101.0}]}
PLAYERS = {
    "10": {"full_name": "Josh Allen", "position": "QB", "team": "BUF", "gsis_id": " 00-0034857"},
    "11": {"full_name": "D.J. Moore Jr.", "position": "WR", "team": "CHI", "gsis_id": None,
           "birth_date": "1997-04-14"},
    "12": {"full_name": "Rookie Guy", "position": "TE", "team": "MIA", "gsis_id": None},
    "20": {"full_name": "Kyle Monangai", "position": "RB", "team": "CHI", "gsis_id": None},
    "99": {"full_name": "Some Kicker", "position": "K"},
}
NFL_ROSTERS = pl.DataFrame({"gsis_id": ["00-0040001", "00-0034857"], "sleeper_id": ["20", None]})
NFL_PLAYERS = pl.DataFrame({
    "gsis_id": ["00-0034857", "00-0034000", "00-0040001"], "display_name": ["Josh Allen", "DJ Moore", "Kyle Monangai"],
    "position": ["QB", "WR", "RB"], "birth_date": ["1996-05-21", "1997-04-14", "2002-01-01"], "espn_id": [None, None, None],
})


def _league() -> S.League:
    return S.League(LEAGUE, ROSTERS, USERS, TRADED, MATCHUPS, PLAYERS)


def test_standings_and_weekly_results():
    st = S.standings(_league())
    assert st.row(0, named=True) == {"roster_id": 1, "owner": "ben", "division": 1, "wins": 2, "losses": 0,
                                     "ties": 0, "points_for": 250.5}
    wk = S.weekly_results(_league())
    assert wk.height == 4
    assert wk.filter((pl.col("week") == 1) & (pl.col("roster_id") == 2)).row(0) == (1, 2, 99.0, 1)


def test_rostered_slots():
    r = S.rostered(_league())
    slots = dict(zip(r["sleeper_id"], r["slot"], strict=True))
    assert slots == {"10": "active", "11": "reserve", "12": "taxi", "20": "active"}


def test_pick_owners_apply_future_trades_only():
    po = S.pick_owners(_league(), seasons=[2027, 2028])
    assert po.height == 2 * 2 * 2
    # Team 2's 2027 1st now belongs to team 1; the 2026 trade (a past draft) is ignored.
    row = po.filter((pl.col("season") == 2027) & (pl.col("round") == 1) & (pl.col("original_roster_id") == 2))
    assert row["owner_roster_id"].item() == 1
    assert (po.filter(pl.col("season") == 2028)["original_roster_id"]
            == po.filter(pl.col("season") == 2028)["owner_roster_id"]).all()


def test_norm_name():
    assert S.norm_name("D.J. Moore Jr.") == "djmoore"
    assert S.norm_name("Marvin Harrison Jr.") == S.norm_name("Marvin Harrison")


def test_crosswalk_methods_in_order():
    x = S.crosswalk(PLAYERS, NFL_ROSTERS, NFL_PLAYERS)
    got = {r["sleeper_id"]: (r["gsis_id"], r["method"]) for r in x.iter_rows(named=True)}
    assert got["10"] == ("00-0034857", "sleeper_gsis")          # Sleeper's own ID, whitespace stripped
    assert got["20"] == ("00-0040001", "nflverse_sleeper_id")
    assert got["11"] == ("00-0034000", "name_birth_date")        # suffix and dots ignored
    assert got["12"] == (None, None)                              # reported, not guessed
    assert "99" not in got                                        # kickers are not fantasy positions here


def test_load_reads_a_saved_pull(tmp_path):
    d = tmp_path / "sleeper" / "2026-11-22"
    (d / "L1").mkdir(parents=True)
    for name, obj in (("league", LEAGUE), ("rosters", ROSTERS), ("users", USERS), ("traded_picks", TRADED)):
        (d / "L1" / f"{name}.json").write_text(json.dumps(obj))
    (d / "L1" / "matchups_1.json").write_text(json.dumps(MATCHUPS[1]))
    (d / "players.json").write_text(json.dumps(PLAYERS))
    lg = S.load("L1", root=tmp_path)
    assert lg.season == 2026 and lg.pull_date.isoformat() == "2026-11-22"
    assert set(lg.matchups) == {1} and lg.players["10"]["team"] == "BUF"


def test_weekly_results_skip_unplayed_schedule_weeks():
    m = {**MATCHUPS, 3: [{"roster_id": 1, "matchup_id": 1, "points": 0.0},
                         {"roster_id": 2, "matchup_id": 1, "points": 0.0}]}
    lg = S.League(LEAGUE, ROSTERS, USERS, TRADED, m, PLAYERS)
    assert S.weekly_results(lg)["week"].max() == 2
