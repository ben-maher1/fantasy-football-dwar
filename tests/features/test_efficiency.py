import polars as pl
import pytest

from dwar.features.efficiency import DEFAULT_YPT, SCALE, SHRINK_TARGETS, efficiency_ratings, with_efficiency_as_qb


def _weeks() -> pl.DataFrame:
    # Season 1: league 100 yds / 10 targets. Season 2: player A 2 games.
    return pl.DataFrame({
        "player_id": ["A", "B", "A", "A"],
        "season": [1, 1, 2, 2],
        "week": [1, 1, 1, 2],
        "position": ["TE"] * 4,
        "targets": [5, 5, 10, 10],
        "receiving_yards": [40, 60, 50, 150],
        "qb_rating": [0.3, 0.3, 0.3, 0.3],
    })


def test_first_season_uses_default_and_later_uses_earlier_seasons() -> None:
    e = efficiency_ratings(_weeks(), "TE").sort("player_id", "season", "week")
    a1 = e.filter((pl.col("player_id") == "A") & (pl.col("season") == 1))["eff_rating"].item()
    assert a1 == pytest.approx(((40 + SHRINK_TARGETS * DEFAULT_YPT) / (5 + SHRINK_TARGETS) - DEFAULT_YPT) / SCALE)
    a2 = e.filter((pl.col("player_id") == "A") & (pl.col("season") == 2))["eff_rating"].to_list()
    prior = 10.0  # season 1 only
    assert a2[0] == pytest.approx(((50 + SHRINK_TARGETS * prior) / (10 + SHRINK_TARGETS) - prior) / SCALE)
    assert a2[1] == pytest.approx(((200 + SHRINK_TARGETS * prior) / (20 + SHRINK_TARGETS) - prior) / SCALE)


def test_no_lookahead_later_games_do_not_change_earlier_ratings() -> None:
    w = _weeks()
    full = efficiency_ratings(w, "TE")
    cut = efficiency_ratings(w.filter(~((pl.col("season") == 2) & (pl.col("week") == 2))), "TE")
    key = ["player_id", "season", "week"]
    j = cut.join(full, on=key, suffix="_full")
    assert (j["eff_rating"] - j["eff_rating_full"]).abs().max() < 1e-12


def test_replaces_only_that_position() -> None:
    w = pl.concat([_weeks(), _weeks().with_columns(position=pl.lit("WR"), player_id=pl.lit("C"))])
    out = with_efficiency_as_qb(w, "TE")
    assert out.filter(pl.col("position") == "WR")["qb_rating"].to_list() == [0.3] * 4
    assert out.filter(pl.col("position") == "TE")["qb_rating"].to_list() != [0.3] * 4
