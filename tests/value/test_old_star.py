import numpy as np
import polars as pl

from dwar.value.old_star import keep_rates, latest_full_season, pull_back


def seasons_table() -> pl.DataFrame:
    rows = [
        # A: star at 30 in 2020, then 10 and 5 PPG.
        ("A", 2020, 30.0, 20.0, 16), ("A", 2021, 31.0, 10.0, 16), ("A", 2022, 32.0, 5.0, 10),
        # B: star at 29 in 2021, then 12; his 2023 is not complete yet.
        ("B", 2021, 29.0, 16.0, 16), ("B", 2022, 30.0, 12.0, 12), ("B", 2023, 31.0, 30.0, 16),
        # C: young star, ignored.
        ("C", 2020, 24.0, 20.0, 16), ("C", 2021, 25.0, 20.0, 16),
    ]
    return pl.DataFrame(rows, schema=["player_id", "season", "age_sept1", "fpts_per_game", "games_played"],
                        orient="row").with_columns(position=pl.lit("RB"))


def test_keep_rates_use_old_stars_and_completed_seasons_only():
    keep = keep_rates(seasons_table(), "RB", 29.0, 12.0, 8, complete=(2020, 2021, 2022))
    # h=1: A 20->10, A 2021 (10 PPG) is not a star, B 16->12 (B 2022 -> 2023 not complete): 22/36.
    assert np.isclose(keep[0], 22.0 / 36.0)
    assert np.isclose(keep[1], 5.0 / 20.0)      # h=2: A only
    assert np.isclose(keep[2], keep[1])         # no h=3 history: carries h=2
    # No lookahead: B's 2023 changes nothing until 2023 is complete.
    assert np.allclose(keep, keep_rates(seasons_table().filter(pl.col("season") < 2023), "RB", 29.0, 12.0, 8,
                                        complete=(2020, 2021, 2022)))


def test_pull_back_moves_future_seasons_of_old_stars_only():
    ps = seasons_table()
    last = latest_full_season(ps.filter(pl.col("season") <= 2022), ["A", "B", "C"], 8)
    assert last["season"].to_list() == [2022, 2022, 2021]
    talent = np.full((4, 3, 3), 10.0)            # draws, players, seasons 2022-2024
    seasons = np.array([2022, 2023, 2024])
    keep = np.array([0.5, 0.25, 0.25])
    out = pull_back(talent, seasons, current=2022, last=last, keep=keep, min_age=29.0, min_ppg=12.0, weight=0.75)
    assert np.allclose(out[:, :, 0], 10.0)       # current season untouched
    assert np.allclose(out[:, 0, :], 10.0)       # A's latest full season is 5 PPG: not a star
    assert np.allclose(out[:, 2, :], 10.0)       # C is young
    # B: 12 PPG at 30 in 2022 -> 2023 target 6.0, 2024 target 3.0.
    assert np.allclose(out[:, 1, 1], 0.25 * 10 + 0.75 * 6.0)
    assert np.allclose(out[:, 1, 2], 0.25 * 10 + 0.75 * 3.0)
