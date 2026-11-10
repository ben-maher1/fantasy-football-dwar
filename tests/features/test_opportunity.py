from datetime import date

import numpy as np
import polars as pl
import pytest

from dwar.config import load_scoring
from dwar.features.opportunity import (
    MIN_BUCKET_PLAYS,
    opportunity_weeks,
    plays,
    value_table,
)


def play(season: int, week: int, **kw) -> dict:
    base = {
        "season": season, "week": week, "season_type": "REG", "game_date": f"{season}-10-{week:02d}",
        "two_point_attempt": 0.0, "qb_kneel": 0.0, "qb_spike": 0.0, "pass_attempt": 0.0, "rush_attempt": 0.0,
        "sack": 0.0, "qb_scramble": 0.0, "complete_pass": 0.0, "pass_touchdown": 0.0, "rush_touchdown": 0.0,
        "fumble_lost": 0.0, "interception": 0.0, "fumbled_1_player_id": None, "air_yards": None,
        "yardline_100": 50.0, "down": 1.0, "yards_gained": 0.0,
        "passer_player_id": None, "receiver_player_id": None, "rusher_player_id": None,
    }
    return base | kw


def target(season, week, receiver, *, air, yardline, complete, yards=0.0, td=0.0):
    return play(season, week, pass_attempt=1.0, passer_player_id="QB1", receiver_player_id=receiver,
                air_yards=air, yardline_100=yardline, complete_pass=float(complete), yards_gained=yards,
                pass_touchdown=td)


def world() -> pl.DataFrame:
    """2014: 40 deep red-zone targets (half caught for TDs) and 40 short
    midfield targets (all caught for 5 yards). 2015: player A gets two
    red-zone targets and catches neither; player B gets two short ones;
    the season ends in January."""
    rows = []
    for i in range(40):
        rows.append(target(2014, 1, "X", air=12.0, yardline=4.0, complete=i % 2 == 0, yards=4.0, td=float(i % 2 == 0)))
        rows.append(target(2014, 1, "Y", air=2.0, yardline=60.0, complete=True, yards=5.0))
    rows += [target(2015, 1, "A", air=12.0, yardline=4.0, complete=False) for _ in range(2)]
    rows += [target(2015, 1, "B", air=2.0, yardline=60.0, complete=True, yards=9.0) for _ in range(2)]
    rows.append(play(2015, 1, two_point_attempt=1.0, pass_attempt=1.0, receiver_player_id="A", complete_pass=1.0))
    rows.append(target(2015, 17, "Z", air=2.0, yardline=60.0, complete=True, yards=5.0) | {"game_date": "2016-01-03"})
    return pl.DataFrame(rows)


POSITIONS = pl.DataFrame({"player_id": ["A", "B"], "season": [2015, 2015], "position": ["WR", "TE"]})


def test_opportunity_values_the_role_not_the_result():
    out = opportunity_weeks(world(), date(2015, 12, 1), load_scoring("half_ppr"), POSITIONS).sort("player_id")
    a, b = out.row(0, named=True), out.row(1, named=True)
    # A's red-zone target: 0.5 catches, 2 yards, 0.5 TDs on average -> 0.5*0.5 + 0.2 + 3 = 3.45 each.
    assert a["xfpts"] == pytest.approx(2 * 3.45)
    # B's short target: 1 catch, 5 yards -> 0.5 + 0.5 = 1.0 each, whatever B actually gained.
    assert b["xfpts"] == pytest.approx(2 * 1.0)
    assert a["targets"] == 2  # the two-point try is not an opportunity


def test_te_premium_applies_to_expected_catches():
    te = load_scoring("half_ppr").model_copy(update={"rec_bonus_by_position": {"TE": 0.5}})
    out = opportunity_weeks(world(), date(2015, 12, 1), te, POSITIONS).filter(pl.col("player_id") == "B")
    assert out["xfpts"].item() == pytest.approx(2 * 1.5)


def test_no_lookahead():
    """The table uses only seasons complete before the as-of date, and only
    plays before it are valued: later plays change nothing."""
    pbp = world()
    early = opportunity_weeks(pbp, date(2015, 12, 1), load_scoring("half_ppr"), POSITIONS)
    later = pl.concat([pbp, pl.DataFrame([target(2015, 9, "A", air=12.0, yardline=4.0, complete=True, yards=4.0, td=1.0)
                                          for _ in range(50)]).with_columns(game_date=pl.lit("2016-01-03"))])
    again = opportunity_weeks(later, date(2015, 12, 1), load_scoring("half_ppr"), POSITIONS)
    assert early.sort("player_id").equals(again.sort("player_id"))
    # Before 2014 is complete there is no table, so nothing is valued.
    assert opportunity_weeks(pbp, date(2014, 9, 30), load_scoring("half_ppr"), POSITIONS).is_empty()


def test_small_buckets_use_their_kind_average():
    p = plays(world())
    table, by_kind = value_table(p, [2014])
    assert (table["n"] >= MIN_BUCKET_PLAYS).all()
    small = plays(world().filter(pl.col("season") == 2015).with_columns(season=pl.lit(2014)))
    t2, k2 = value_table(small, [2014])
    targets = t2.filter(pl.col("kind") == "target")
    assert np.allclose(targets["receptions"].to_numpy(), k2.filter(pl.col("kind") == "target")["receptions"].item())
