import numpy as np
import polars as pl
import pytest

from dwar.validate.baselines import baseline_forecasts, baseline_points


def history() -> pl.DataFrame:
    # Year-over-year change is exactly 0.5 * (26 - age) for everyone, so the
    # fitted age curve is exact: +0.5 at age 25.
    rows = [
        # P: drafted round 2, ages 23-25 in 2021-2023.
        ("P", 2021, 10, 6.0, 23.0), ("P", 2022, 16, 7.5, 24.0), ("P", 2023, 17, 8.5, 25.0),
        # R: drafted round 1, ages 26-28 in 2019-2021.
        ("R", 2019, 16, 10.0, 26.0), ("R", 2020, 16, 10.0, 27.0), ("R", 2021, 16, 9.5, 28.0),
        ("R", 2022, 16, 8.5, 29.0),
    ]
    return pl.DataFrame(rows, schema=["player_id", "season", "games_played", "fpts_per_game", "age_sept1"], orient="row").with_columns(
        position=pl.lit("WR")
    )


STATIC = pl.DataFrame({"player_id": ["P", "R", "S"], "draft_round": [2, 1, 1]})
TARGETS = pl.DataFrame({"player_id": ["P", "S"], "position": ["WR", "WR"]})


def test_baselines_by_hand():
    f = baseline_points(history(), STATIC, TARGETS, 2024).sort("player_id")
    p = f.row(0, named=True)
    assert p["last"] == 8.5
    assert p["weighted3"] == pytest.approx((3 * 17 * 8.5 + 2 * 16 * 7.5 + 1 * 10 * 6.0) / (51 + 32 + 10))
    assert p["age_adjusted"] == pytest.approx(9.0)
    assert not p["is_rookie"]
    # S has never played: the round-1 rookie average, which is R's first season.
    s = f.row(1, named=True)
    assert s["is_rookie"] and s["last"] == s["weighted3"] == s["age_adjusted"] == 10.0


def test_season_being_forecast_is_never_used():
    leaked = pl.concat([history(), pl.DataFrame({
        "player_id": ["P"], "season": [2024], "games_played": [17], "fpts_per_game": [30.0],
        "age_sept1": [26.0], "position": ["WR"],
    })], how="diagonal_relaxed")
    assert baseline_points(leaked, STATIC, TARGETS, 2024).equals(baseline_points(history(), STATIC, TARGETS, 2024))


def test_log_scores_are_finite_with_spread_from_past_misses():
    h = history()
    actual = pl.DataFrame({"player_id": ["P"], "position": ["WR"], "games_played": [12], "fpts_per_game": [9.5]})
    f = baseline_forecasts(h, STATIC, actual, 2024)
    for m in ("last", "weighted3", "age_adjusted"):
        assert np.isfinite(f[f"{m}_log_score"][0]) and f[f"{m}_sd"][0] > 0
