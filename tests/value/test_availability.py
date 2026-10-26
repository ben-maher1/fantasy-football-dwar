from datetime import date

import numpy as np
import polars as pl
import pytest

from dwar.value.availability import (
    AvailabilityDraws, _design, _sigmoid, availability_rows, fit_logistic, season_games,
)


def test_logistic_recovers_coefficients_and_dispersion():
    rng = np.random.default_rng(0)
    n = 40_000
    age, talent = rng.uniform(21, 34, n), rng.uniform(2, 20, n)
    x = _design(age, talent)
    truth = np.array([0.5, -0.05, -0.01, 0.12])
    p = _sigmoid(x @ truth)
    fit = fit_logistic(x, rng.binomial(17, p), np.full(n, 17))
    assert (np.abs(fit.coef - truth) < 4 * np.sqrt(np.diag(fit.cov))).all(), (fit.coef, truth)
    assert fit.dispersion == pytest.approx(1.0, abs=0.05)
    # Beta-binomial games (injury seasons) are overdispersed.
    shares = rng.beta(p * 4, (1 - p) * 4)
    assert fit_logistic(x, rng.binomial(17, shares), np.full(n, 17)).dispersion > 2


def test_games_share_keeps_the_mean_and_adds_spread():
    rng = np.random.default_rng(1)
    coefs = np.tile([1.0, 0.0, 0.0, 0.0], (2000, 1))
    draws = AvailabilityDraws({"WR": coefs}, {"WR": 4.0}, {"WR": coefs}, 17)
    share = draws.games_share("WR", np.full((2000, 50), 26.0), np.full((2000, 50), 10.0), rng)
    assert share.mean() == pytest.approx(_sigmoid(1.0), abs=0.01)
    assert share.std() > 0.1


def schedules(seasons_done: list[int], in_progress: int) -> pl.DataFrame:
    rows = []
    for s in [*seasons_done, in_progress]:
        for w in range(1, 4):
            rows.append({"season": s, "gameday": date(s, 9, 7 * w), "home_team": "A", "away_team": "B"})
    return pl.DataFrame(rows)


def test_rows_use_only_completed_seasons():
    sched = schedules([2021, 2022], in_progress=2023)
    as_of = date(2023, 9, 10)  # 2023 under way
    talent = pl.DataFrame({
        "player_id": ["P"] * 3 + ["Q"] * 2,
        "season": [2021, 2022, 2023, 2021, 2023],
        "position": ["WR"] * 5,
        "games_played": [3, 2, 1, 3, 1],
        "talent_mean": [10.0, 11.0, 12.0, 6.0, 5.0],
    })
    seasons = talent.select("player_id", "season", age_sept1=pl.lit(25.0))
    games, survival = availability_rows(talent, seasons, sched, as_of)
    # Games: 2021 and 2022 rows only; 3 games per team per season.
    assert sorted(games["successes"].to_list()) == [2, 3, 3]
    assert set(games["trials"].to_list()) == {3}
    # Survival needs season s + 1 complete: only 2021 -> 2022. P played in
    # 2022, Q did not (Q's 2023 return is not yet a known outcome for 2022).
    assert sorted(survival["successes"].to_list()) == [0, 1]
    assert season_games(sched).filter(pl.col("season") == 2023)["games"].item() == 3


def test_later_seasons_change_nothing():
    """No lookahead: seasons ending after `as_of` add no training rows."""
    as_of = date(2023, 1, 15)
    talent = pl.DataFrame({
        "player_id": ["P", "P", "Q"], "season": [2021, 2022, 2021], "position": ["WR"] * 3,
        "games_played": [3, 3, 2], "talent_mean": [10.0, 9.0, 6.0],
    })
    seasons = talent.select("player_id", "season", age_sept1=pl.lit(25.0))
    base = availability_rows(talent, seasons, schedules([2021], in_progress=2022), as_of)
    future = pl.concat([talent, pl.DataFrame({
        "player_id": ["Q"], "season": [2023], "position": ["WR"], "games_played": [3], "talent_mean": [30.0],
    })])
    later = availability_rows(
        future, future.select("player_id", "season", age_sept1=pl.lit(25.0)),
        schedules([2021, 2022], in_progress=2023), as_of,
    )
    for a, b in zip(base, later):
        assert a.sort(a.columns).equals(b.sort(b.columns))
