from datetime import date

import numpy as np
import polars as pl
import pytest

from dwar.value.availability import (
    AvailabilityDraws,
    _design,
    _sigmoid,
    availability_rows,
    fit_games,
    fit_logistic,
    ppg_survival_rows,
    season_games,
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


def test_ppg_survival_rows_use_points_and_only_completed_seasons():
    sched = schedules([2020, 2021, 2022], in_progress=2023)
    seasons = pl.DataFrame({
        "player_id": ["P", "P", "P", "Q", "Q"], "season": [2020, 2021, 2023, 2020, 2022],
        "position": ["RB"] * 5, "games_played": [3, 2, 1, 3, 1],
        "fpts_per_game": [15.0, 12.0, 9.0, 6.0, 5.0], "age_sept1": [30.0, 31.0, 33.0, 27.0, 29.0],
    })
    rows = ppg_survival_rows(seasons, sched, date(2023, 9, 10), first=2020)
    # 2020 -> 2021 and 2021 -> 2022 only (2023 is under way): P played 2021,
    # not 2022; Q skipped 2021, played 2022 is a 2021 row Q does not have.
    assert sorted(rows.select("age", "talent", "successes").rows()) == [(27.0, 6.0, 0), (30.0, 15.0, 1), (31.0, 12.0, 0)]
    assert ppg_survival_rows(seasons, sched, date(2023, 9, 10), first=2021).height == 1


def test_lagged_rows_pair_games_with_last_seasons_talent():
    sched = schedules([2021, 2022], in_progress=2023)
    talent = pl.DataFrame({
        "player_id": ["P", "P", "Q"], "season": [2021, 2022, 2022], "position": ["WR"] * 3,
        "games_played": [3, 2, 3], "talent_mean": [10.0, 11.0, 6.0],
    })
    seasons = talent.select("player_id", "season", age_sept1=pl.lit(25.0))
    games, survival = availability_rows(talent, seasons, sched, date(2023, 9, 10), lagged=True)
    # Only P's 2022 has a season before it; it carries 2021's talent.
    assert games.select("talent", "successes").rows() == [(10.0, 2)]
    plain = availability_rows(talent, seasons, sched, date(2023, 9, 10))[1]
    assert survival.sort(survival.columns).equals(plain.sort(plain.columns))  # survival unchanged


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


def test_talent_hinge_flattens_games_above_the_knot():
    rng = np.random.default_rng(2)
    n = 30_000
    age, talent = rng.uniform(22, 30, n), rng.uniform(0, 20, n)
    # True share rises with talent up to 10, then stays flat.
    p = _sigmoid(-1.0 + 0.3 * np.minimum(talent, 10.0))
    games = rng.binomial(17, p)
    rows = pl.DataFrame({"position": ["RB"] * n, "age": age, "talent": talent, "successes": games, "trials": [17] * n})
    straight = fit_games(rows, None, False, False)
    hinged = fit_games(rows, 10.0, False, False)
    elite = np.array([[26.0]]), np.array([[19.0]])
    draws = lambda f, knot: AvailabilityDraws({"RB": f.coef[None]}, {"RB": 1.0}, {"RB": certain_coefs()}, 17,
                                              games_knot={"RB": knot} if knot else {})
    truth = _sigmoid(-1.0 + 3.0)
    assert abs(draws(hinged, 10.0).games_mean("RB", *elite)[0, 0] - truth) < 0.01
    assert draws(straight, None).games_mean("RB", *elite)[0, 0] > truth + 0.02  # a straight line over-credits


def certain_coefs() -> np.ndarray:
    return np.array([[50.0, 0.0, 0.0, 0.0]])


def test_draft_terms_only_in_early_career_and_undrafted_as_late_pick():
    from dwar.value.availability import PICK_CENTER, UNDRAFTED_PICK, draft_terms

    d = draft_terms(np.array([1.0, 4.0, 5.0, 2.0]), np.array([64.0, 233.0, 10.0, np.nan]))
    assert np.allclose(d[:, 0], [1, 1, 0, 1])
    assert np.isclose(d[0, 1], 0.0)
    assert np.isclose(d[1, 1], np.log(233.0 / PICK_CENTER))
    assert d[2, 1] == 0.0
    assert np.isclose(d[3, 1], np.log(UNDRAFTED_PICK / PICK_CENTER))


def test_survival_prob_needs_draft_terms_when_the_model_has_them():
    coefs = np.tile([2.0, 0.0, 0.0, 0.0, 0.0, -1.0], (10, 1))
    draws = AvailabilityDraws({}, {}, {"RB": coefs}, 17, draft_survival=("RB",))
    age, tal = np.full((10, 2), 24.0), np.full((10, 2), 10.0)
    with pytest.raises(ValueError):
        draws.survival_prob("RB", age, tal)
    late = draws.survival_prob("RB", age, tal, np.array([[1.0, np.log(4.0)], [1.0, 0.0]]))
    assert (late[:, 0] < late[:, 1]).all()


def test_draft_columns_count_career_years_from_draft_or_first_season():
    from dwar.value.availability import _draft_cols

    static = pl.DataFrame({"player_id": ["D", "U"], "draft_year": [2020, None], "draft_pick": [128, None],
                           "first_season": [2020, 2021]})
    rows = pl.DataFrame({"player_id": ["D", "D", "U"], "season": [2021, 2024, 2021]})
    d = _draft_cols(rows, static, pl.col("season"))
    assert np.allclose(d[:, 0], [1, 0, 1])           # D year 2, D year 5, U year 1
    assert np.isclose(d[0, 1], np.log(2.0))
