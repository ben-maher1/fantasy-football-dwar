from datetime import date
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import xarray as xr

from dwar.config import LeagueFormat, Picks
from dwar.model.fit import FitResult
from dwar.value.availability import AvailabilityDraws, Logistic
from dwar.value.dwar import (
    RookieStart,
    SeasonClock,
    ValueInputs,
    dwar_tables,
    season_values,
)
from dwar.value.picks import (
    RookieSeasonModel,
    _group_labels,
    _pick_scale,
    couple_to_talent,
    fit_rookie_seasons,
    prior_shape_correction,
    rookie_season_rows,
    rookie_start,
    sample_rookies,
)
from dwar.value.wins import wins_per_game

from .test_dwar import certain, end_to_end, schedule  # noqa: F401  (fixture)
from .test_projection import draws as const_draws


def static_rows(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema_overrides={"draft_pick": pl.Int64, "draft_year": pl.Int64,
                                                "draft_round": pl.Int64, "birth_date": pl.Date})


def test_sample_rookies_uses_draft_day_position_and_age():
    static = static_rows([
        {"player_id": "A", "name": "A", "position": "TE", "draft_position": "WR", "draft_year": 2020,
         "draft_pick": 10, "draft_round": 1, "birth_date": date(1998, 3, 1)},
        {"player_id": "B", "name": "B", "position": "RB", "draft_position": "RB", "draft_year": 2012,
         "draft_pick": 40, "draft_round": 2, "birth_date": date(1990, 1, 1)},   # before first class
        {"player_id": "C", "name": "C", "position": "WR", "draft_position": "WR", "draft_year": None,
         "draft_pick": None, "draft_round": None, "birth_date": date(1998, 1, 1)},  # undrafted
        {"player_id": "D", "name": "D", "position": "QB", "draft_position": "QB", "draft_year": 2021,
         "draft_pick": 70, "draft_round": 3, "birth_date": None},
    ])
    out = sample_rookies(static, 2013, ("QB", "RB", "WR", "TE"))
    assert out["player_id"].to_list() == ["A", "D"]
    a = out.row(0, named=True)
    assert a["position"] == "WR" and a["round1"] and a["age"] == pytest.approx(22.5, abs=0.01)
    assert out.row(1, named=True)["age"] == pytest.approx(22.0)   # no birth date: default


def test_rookie_season_rows_are_point_in_time():
    """A rookie season that has not ended is not a training row, and the
    next season counts only once it has ended."""
    static = static_rows([
        {"player_id": p, "name": p, "position": "WR", "draft_position": "WR", "draft_year": y,
         "draft_pick": 50, "draft_round": 2, "birth_date": date(y - 22, 1, 1)}
        for p, y in [("A", 2023), ("B", 2024), ("C", 2025)]
    ])
    seasons = pl.DataFrame({"player_id": ["B", "C"], "season": [2025, 2025], "games_played": [6, 3]})
    weeks = pl.DataFrame(schema={"player_id": pl.String, "season": pl.Int64, "position": pl.String, "plays": pl.Int64})
    sched = schedule({s: [date(s, 9, 10), date(s, 12, 28)] for s in (2023, 2024, 2025)})
    rows = rookie_season_rows(static, seasons, weeks, sched, date(2025, 11, 1), 2013, {})
    by = {r["player_id"]: r for r in rows.iter_rows(named=True)}
    assert set(by) == {"A", "B"}                 # 2025 had not ended on Nov 1
    assert by["A"]["games"] == 0 and by["A"]["next_played"] is False
    assert by["B"]["games"] == 0 and by["B"]["next_played"] is None   # 2025 not over: unknown


def test_rookie_season_models_learn_that_early_picks_play_more():
    rng = np.random.default_rng(0)
    n = 4000
    pick = rng.integers(1, 257, n)
    p_play = 1 / (1 + np.exp(-(1.5 - 1.2 * np.log(pick / 64))))
    played = rng.random(n) < p_play
    games = np.where(played, rng.binomial(17, 0.6, n).clip(1), 0)
    rows = pl.DataFrame({
        "player_id": [str(i) for i in range(n)], "position": "WR", "draft_year": 2015, "draft_pick": pick,
        "round1": pick <= 32, "games": games, "full": games, "team_games": 17,
        "next_played": np.where(played, None, rng.random(n) < 0.3).tolist(),
        "talent": np.where(played, rng.normal(6, 2, n), np.nan),
    })
    m = fit_rookie_seasons(rows, ())["WR"]
    s = rookie_start(m, np.array([5.0, 200.0]), np.array([True, False]), 2027, np.full((4000, 2), 6.0), 17, rng)
    played_share = (s.share > 0).mean(axis=0)
    assert played_share[0] > played_share[1] + 0.3
    assert (s.share[s.share > 0] >= 1 / 17).all() and m.full is None
    assert np.isclose(s.late[:, 1].mean(), 0.3, atol=0.05)   # late picks: most non-players


def test_pick_scale_recovers_the_fit_standardization():
    picks = np.array([1, 10, 50, 100, 200, 250], float)
    z = (np.log(picks) - 3.9) / 1.1
    d = SimpleNamespace(player_ids=tuple("ABCDEF"), undrafted=np.zeros(6), log_pick_z=z, n_players=6,
                        round1=(picks <= 32).astype(float))
    static = pl.DataFrame({"player_id": list("ABCDEF"), "draft_pick": picks.astype(int)})
    mu, sd, coef = _pick_scale(SimpleNamespace(data=d), static)
    assert mu == pytest.approx(3.9) and sd == pytest.approx(1.1) and coef.shape == (3,)


def test_group_labels_split_a_round_in_thirds():
    assert [(n, list(r)) for n, r in _group_labels(12)] == [
        ("Early", [1, 2, 3, 4]), ("Mid", [5, 6, 7, 8]), ("Late", [9, 10, 11, 12])]
    assert [len(r) for _, r in _group_labels(10)] == [3, 4, 3]


def test_rookie_counts_nothing_before_his_rookie_season():
    """By hand: a rookie entering in 2028 who plays every game is worth only
    2028 (discounted twice); one who never plays and never comes back, 0."""
    n = 10
    clock = SeasonClock(current=2026, complete=(2025,), games=17, remaining={}, remaining_default=10)
    inputs = ValueInputs(
        clock=clock, draws={"WR": const_draws(n, [15.0, 15.0])}, replacement={"WR": np.full(n, 10.0)},
        sigma=25.0, availability=AvailabilityDraws({"WR": certain(n)}, {"WR": 1.0}, {"WR": certain(n)}, 17),
    )
    fmt = LeagueFormat(name="t", scoring="ppr", teams=2, lineup={"WR": 1}, discount=0.5)
    seasons = np.arange(2025, 2029)
    talent = np.full((n, 2, len(seasons)), 15.0)
    ages = np.full_like(talent, 22.0)
    start = RookieStart(np.array([2028, 2028]), np.column_stack([np.ones(n), np.zeros(n)]), None, np.zeros((n, 2)))
    total, tr = season_values("WR", inputs, fmt, np.random.default_rng(0), talent, ages, seasons,
                              np.array([2028, 2028]), ["r1", "r2"], np.zeros(2), None, None,
                              rookie=start, trace=True)
    w = wins_per_game(5.0, 25.0)
    np.testing.assert_allclose(total[:, 0], 0.25 * 17 * w, rtol=1e-5)
    np.testing.assert_allclose(total[:, 1], 0.0)
    assert tr.games.shape == (n, 2, 3) and (tr.alive[:, :, :2] == 0).all()


def test_pick_values_end_to_end(end_to_end):  # noqa: F811
    fits, tables, sched, fmt, cfg = end_to_end
    # The fake fits need the rookie prior's posterior.
    fits = {p: _with_prior(r) for p, r in fits.items()}
    cfg = cfg.model_copy(update={"picks": Picks(first_class=2008, history_first=2008, years=2)})
    fmt = fmt.model_copy(update={"rookie_rounds": 2})
    table, summary, pv = dwar_tables(fits, tables, sched, fmt, cfg, "studentt", seed=1)
    plain, _, none = dwar_tables(fits, tables, sched, fmt, cfg, "studentt", seed=1, picks=False)
    assert table.equals(plain) and none is None   # picks never move player values
    p = pv.picks
    slots = p.filter(pl.col("kind") == "slot")
    assert slots.height == 2 * 2 * fmt.teams and set(p["pick_year"]) == set(summary["rookie_draft_picks"]["years"])
    for _, g in slots.group_by("pick_year"):
        v = g.sort("round", "slot")["dwar_mean"].to_numpy()
        assert (np.diff(v) <= 1e-9).all()          # each later pick is worth no more
    first = slots.filter(pl.col("round") == 1)
    assert (first.filter(pl.col("pick_year") == min(p["pick_year"]))["dwar_mean"].mean()
            >= first.filter(pl.col("pick_year") == max(p["pick_year"]))["dwar_mean"].mean())  # later draft, fewer seasons
    q = slots.select("dwar_p05", "dwar_p25", "dwar_p50", "dwar_p75", "dwar_p95").to_numpy()
    assert (np.diff(q, axis=1) >= 0).all() and slots["hit_rate"].is_between(0, 1).all()
    assert set(p["kind"]) == {"slot", "early", "mid", "late", "round"}


def _with_prior(r: FitResult) -> FitResult:
    post = r.posterior["posterior"].to_dataset()
    shape = (post.sizes["chain"], post.sizes["draw"])
    for name, v in {"alpha0": 9.0, "alpha_pick": -2.0, "sigma_rookie": 2.0}.items():
        post[name] = (("chain", "draw"), np.full(shape, v))
    return FitResult(r.data, xr.DataTree.from_dict({"posterior": post}), r.diagnostics, r.talent)


def test_rookie_season_model_shape():
    lg = Logistic(np.zeros(4), np.eye(4) * 1e-6)
    lt = Logistic(np.array([0, 0, 0, 0, 1.0]), np.eye(5) * 1e-6)
    m = RookieSeasonModel(lg, lt, None, lg, 10)
    s = rookie_start(m, np.array([1.0, 2.0, 3.0]), np.array([1, 1, 1]), 2030, np.tile([-3.0, 0.0, 3.0], (7, 1)), 17,
                     np.random.default_rng(1))
    assert s.share.shape == (7, 3) and s.ratio is None and (s.season == 2030).all()
    played = s.share > 0
    mean = [s.share[:, k][played[:, k]].mean() for k in range(3)]
    assert mean[0] < mean[1] < mean[2]   # more talent, more games


def test_couple_to_talent_keeps_each_players_games_and_pairs_them_with_talent():
    rng = np.random.default_rng(3)
    talent = rng.normal(5, 2, (4000, 3))
    share = rng.beta(2, 2, (4000, 3))
    out = couple_to_talent(share, talent, 0.5, rng)
    np.testing.assert_allclose(np.sort(out, axis=0), np.sort(share, axis=0))   # same games, re-paired
    r = [np.corrcoef(talent[:, k], out[:, k])[0, 1] for k in range(3)]
    assert all(0.4 < c < 0.55 for c in r)
    assert abs(np.corrcoef(talent[:, 0], couple_to_talent(share, talent, 0.0, rng)[:, 0])[0, 1]) < 0.05


def test_prior_shape_correction_recovers_a_bend_in_log_pick():
    """Rookies whose fitted deviations from the prior line follow a known
    quadratic in standardized log pick: the correction returns it."""
    rng = np.random.default_rng(0)
    n, draws = 300, 50
    picks = rng.integers(1, 257, n).astype(float)
    mu, sd = 4.0, 1.2
    z = (np.log(picks) - mu) / sd
    true = np.array([0.05, 0.1, -0.15])
    dev = true @ np.stack([np.ones(n), z, z**2]) + rng.normal(0, 0.05, (draws, n))
    ids = tuple(str(i) for i in range(n))
    post = xr.Dataset({"sigma_rookie": (("chain", "draw"), np.full((1, draws), 2.0)),
                       "rookie_z": (("chain", "draw", "rookie"), (dev / 2.0)[None])})
    data = SimpleNamespace(player_ids=ids, pre_window=np.zeros(n), undrafted=np.zeros(n), log_pick_z=z)
    r = SimpleNamespace(data=data, posterior={"posterior": post})
    static = pl.DataFrame({"player_id": list(ids), "draft_pick": picks.astype(int)})
    c = prior_shape_correction(r, np.arange(draws), static, mu, sd)
    assert c.shape == (draws, 3)
    np.testing.assert_allclose(c.mean(axis=0), true, atol=0.02)
