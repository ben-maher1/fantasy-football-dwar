import numpy as np
import polars as pl

from dwar.config import LeagueFormat, RosterSettings
from dwar.value.dwar import play_rate_by_rank
from dwar.value.replacement import _rate, marginal_points


def _fmt() -> LeagueFormat:
    return LeagueFormat(name="t", scoring="ppr", teams=2, lineup={"QB": 1, "WR": 1},
                        roster=RosterSettings(bench=2, by_position={"QB": 2, "WR": 2}, weekly=("QB",)))


def _seasons() -> pl.DataFrame:
    # Season 1: four QBs and four WRs ranked by PPG. Season 2: the top two
    # play every game, the next two play half or are gone.
    rows = []
    for p in ("QB", "WR"):
        for i in range(4):
            rows.append({"player_id": f"{p}{i}", "season": 1, "position": p, "games_played": 16,
                         "fpts_per_game": 20.0 - i})
        for i, g in enumerate([17, 17, 9, 0]):
            if g:
                rows.append({"player_id": f"{p}{i}", "season": 2, "position": p, "games_played": g,
                             "fpts_per_game": 10.0})
    return pl.DataFrame(rows)


def test_play_rate_by_rank_uses_next_season_games_and_full_games_for_weekly():
    full = pl.DataFrame({"player_id": ["QB0", "QB1", "QB2"], "season": [2, 2, 2], "full": [17, 17, 1]})
    r = play_rate_by_rank(_seasons(), (2,), _fmt(), 4, full, {2: 17})
    # Raw rates (18 weeks): WR [17, 17, 9, 0]/18, smoothed over a window of 3 ranks, non-increasing.
    raw = np.array([17, 17, 9, 0]) / 18
    pad = np.pad(raw, 1, mode="edge")
    want = np.minimum.accumulate(np.convolve(pad, np.ones(3) / 3, mode="valid"))
    assert np.allclose(r["WR"], want)
    # QB counts full games: the third QB's single start, not 9 games.
    assert r["QB"][2] < r["WR"][2]
    assert (np.diff(r["QB"]) <= 1e-12).all()


def test_rate_scalar_or_by_rank():
    assert _rate({"WR": 0.8}, "WR", np.array([0, 5])) == 0.8
    v = _rate({"WR": np.array([0.9, 0.5])}, "WR", np.array([0, 1, 7]))
    assert list(v) == [0.9, 0.5, 0.5]


def test_backups_who_rarely_play_make_starters_worth_more():
    talent = {"QB": np.array([20.0, 19.0, 15.0, 14.0])}
    lineup, rostered, waiver = {"QB": 1}, {"QB": 2}, {"QB": 5.0}
    same = marginal_points(talent, lineup, 2, rostered, {"QB": 0.8}, waiver, np.random.default_rng(0), weeks=3000)
    ranked = marginal_points(talent, lineup, 2, rostered, {"QB": np.array([0.8, 0.8, 0.3, 0.3])}, waiver,
                             np.random.default_rng(0), weeks=3000)
    # Talent ascending: the two starters are the last two entries.
    assert (ranked["QB"][1][-2:] > same["QB"][1][-2:] + 1.0).all()


def test_qb_starter_rates_flat_then_zero():
    from dwar.value.dwar import qb_starter_rates

    rate = np.linspace(0.8, 0.4, 36)
    r = qb_starter_rates(rate, 12, 32)
    assert np.allclose(r[:32], rate[:12].mean())
    # Ranks past the starters take the last rate: spares never play.
    assert (_rate({"QB": r}, "QB", np.array([31, 32, 35])) == np.array([r[0], 0.0, 0.0])).all()
    assert np.isclose(qb_starter_rates(0.72, 12, 32)[0], 0.72)
