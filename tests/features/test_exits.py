"""Injury exits: low snap share, then the team's next game missed."""

from __future__ import annotations

import polars as pl

from dwar.features.exits import flag_injury_exits


def weeks(rows):
    return pl.DataFrame(rows, schema=["player_id", "season", "week", "team", "offense_pct"], orient="row")


def test_low_share_then_missed_next_game_is_an_exit():
    rows = [("a", 2024, w, "X", 0.8) for w in (1, 2, 3)] + [("a", 2024, 4, "X", 0.3), ("a", 2024, 7, "X", 0.8)]
    rows += [("b", 2024, w, "X", 0.5) for w in range(1, 8)]  # teammate: the team played weeks 1-7
    out = flag_injury_exits(weeks(rows)).filter(pl.col("player_id") == "a")
    assert out["injury_exit"].to_list() == [False, False, False, True, False]


def test_low_share_but_played_next_game_is_not_an_exit():
    rows = [("a", 2024, w, "X", 0.8) for w in (1, 2, 3)] + [("a", 2024, 4, "X", 0.3), ("a", 2024, 5, "X", 0.8)]
    assert not flag_injury_exits(weeks(rows))["injury_exit"].any()


def test_team_next_game_not_yet_played_is_not_an_exit():
    # Point in time: week 4 is the latest game in the tables, so a miss can't be known.
    rows = [("a", 2024, w, "X", 0.8) for w in (1, 2, 3)] + [("a", 2024, 4, "X", 0.3)]
    assert not flag_injury_exits(weeks(rows))["injury_exit"].any()


def test_needs_enough_history():
    rows = [("a", 2024, 1, "X", 0.8), ("a", 2024, 2, "X", 0.2), ("b", 2024, 3, "X", 0.5)]
    assert not flag_injury_exits(weeks(rows))["injury_exit"].any()
