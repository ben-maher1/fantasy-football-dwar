from dataclasses import replace
from datetime import date
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import xarray as xr

from dwar.config import LeagueFormat, ValueConfig
from dwar.features.player_tables import PlayerTables
from dwar.model.fit import FitResult
from dwar.value.availability import AvailabilityDraws
from dwar.value.dwar import OUTPUT_COLUMNS, SeasonClock, ValueInputs, dwar_table, position_dwar, season_clock
from dwar.value.wins import wins_per_game

from ..model.simulate import simulate
from .test_projection import draws as const_draws


def schedule(season_days: dict[int, list[date]]) -> pl.DataFrame:
    return pl.DataFrame([
        {"season": s, "gameday": d, "home_team": "AAA", "away_team": "BBB"}
        for s, days in season_days.items() for d in days
    ])


def test_season_clock_mid_season_counts_remaining_team_games():
    sched = schedule({2025: [date(2025, 9, 7), date(2025, 12, 28)], 2026: [date(2026, 9, 6), date(2026, 10, 4), date(2026, 11, 1)]})
    clock = season_clock(sched, date(2026, 10, 1))
    assert (clock.current, clock.complete, clock.games) == (2026, (2025,), 2)
    assert clock.remaining == {"AAA": 2, "BBB": 2}


def test_season_clock_offseason_before_schedule_is_a_full_season():
    sched = schedule({2025: [date(2025, 9, 7), date(2025, 12, 28)]})
    clock = season_clock(sched, date(2026, 3, 1))
    assert (clock.current, clock.remaining, clock.remaining_default) == (2026, {}, 2)


def certain(n: int) -> np.ndarray:
    return np.tile([50.0, 0.0, 0.0, 0.0], (n, 1))  # logit 50: probability 1


def test_one_player_by_hand():
    n = 20
    clock = SeasonClock(current=2026, complete=(2025,), games=17, remaining={"AAA": 10}, remaining_default=10)
    inputs = ValueInputs(
        clock=clock,
        draws={"WR": const_draws(n, [15.0])},
        replacement={"WR": np.full(n, 10.0)},
        sigma=25.0,
        availability=AvailabilityDraws({"WR": certain(n)}, {"WR": 1.0}, {"WR": certain(n)}, 17),
    )
    r = SimpleNamespace(data=SimpleNamespace(
        ps_keys=pl.DataFrame({"player_id": ["P"], "season": [2026], "games_played": [7]}),
        ps_age_c=np.array([0.0]),
    ))
    tables = SimpleNamespace(player_static=pl.DataFrame(
        {"player_id": ["P"], "name": ["Pat"], "team": ["AAA"], "age_at_as_of": [26.1]}
    ))
    fmt = LeagueFormat(name="t", scoring="ppr", teams=2, lineup={"WR": 1}, discount=0.5)
    cfg = ValueConfig(draws=n, horizon_seasons=2, replacement_seasons=1, pool_min_games=1)
    players, d = position_dwar("WR", r, inputs, tables, fmt, cfg, np.random.default_rng(0))
    w = wins_per_game(5.0, 25.0)
    np.testing.assert_allclose(d[:, 0], 10 * w + 0.5 * 17 * w + 0.25 * 17 * w, rtol=1e-5)
    assert players["name"].to_list() == ["Pat"]


def test_below_replacement_is_worth_zero_not_negative():
    n = 10
    clock = SeasonClock(current=2026, complete=(2025,), games=17, remaining={}, remaining_default=17)
    inputs = ValueInputs(clock, {"WR": const_draws(n, [8.0])}, {"WR": np.full(n, 10.0)}, 25.0,
                         AvailabilityDraws({"WR": certain(n)}, {"WR": 1.0}, {"WR": certain(n)}, 17))
    r = SimpleNamespace(data=SimpleNamespace(
        ps_keys=pl.DataFrame({"player_id": ["P"], "season": [2025], "games_played": [17]}), ps_age_c=np.array([0.0]),
    ))
    tables = SimpleNamespace(player_static=pl.DataFrame({"player_id": ["P"], "name": ["P"], "team": ["X"], "age_at_as_of": [26.0]}))
    fmt = LeagueFormat(name="t", scoring="ppr", teams=2, lineup={"WR": 1})
    cfg = ValueConfig(draws=n, horizon_seasons=3, replacement_seasons=1, pool_min_games=1)
    _, d = position_dwar("WR", r, inputs, tables, fmt, cfg, np.random.default_rng(0))
    np.testing.assert_array_equal(d, 0.0)


def fake_fit(sim, position: str, rng: np.random.Generator, n_draws: int = 60) -> FitResult:
    """A FitResult whose posterior draws scatter around the simulated truth,
    so the value layer can run end to end without sampling."""
    d = sim.data
    talent = sim.talent[None, None, :] + rng.normal(0, 1.0, (2, n_draws, d.n_ps))
    scalar = lambda v: (("chain", "draw"), np.full((2, n_draws), v))
    posterior = xr.Dataset({
        "talent": (("chain", "draw", "player_season"), talent),
        "beta1": scalar(0.1), "beta2": scalar(-0.05), "tau": scalar(1.5),
        "sigma": scalar(5.0), "nu": scalar(5.0),
    })
    summary = d.ps_keys.with_columns(talent_mean=pl.Series(talent.mean(axis=(0, 1))))
    return FitResult(data=d, posterior=xr.DataTree.from_dict({"posterior": posterior}), diagnostics=None, talent=summary)


def league(sims: dict[str, object]) -> tuple[PlayerTables, pl.DataFrame]:
    weeks = pl.concat([s.tables.player_weeks.with_columns(position=pl.lit(p)) for p, s in sims.items()])
    ages = pl.concat([s.tables.player_seasons for s in sims.values()])
    seasons = (
        weeks.group_by("player_id", "season", "position")
        .agg(games_played=pl.len(), fpts_per_game=pl.col("fpts").mean())
        .join(ages, on=["player_id", "season"])
    )
    static = pl.concat([
        s.tables.player_static.with_columns(name="player_id", team=pl.lit("AAA"), age_at_as_of=pl.lit(27.0))
        for s in sims.values()
    ])
    last = int(seasons["season"].max())
    days = {s: [date(s, 9, 10), date(s, 12, 20)] for s in range(2006, last)} | {last: [date(last, 9, 10), date(last, 12, 20)]}
    as_of = date(last, 10, 1)
    tables = PlayerTables(as_of=as_of, player_weeks=weeks, player_seasons=seasons, player_static=static)
    return tables, schedule(days)


def relabel(sim, prefix: str):
    rename = lambda df: df.with_columns(pl.col("player_id").str.replace("SIM", prefix))
    tables = replace(sim.tables, player_weeks=rename(sim.tables.player_weeks),
                     player_seasons=rename(sim.tables.player_seasons), player_static=rename(sim.tables.player_static))
    data = replace(sim.data, ps_keys=rename(sim.data.ps_keys), player_ids=tuple(p.replace("SIM", prefix) for p in sim.data.player_ids))
    return replace(sim, tables=tables, data=data)


@pytest.fixture(scope="module")
def end_to_end():
    rng = np.random.default_rng(5)
    sims = {"WR": simulate(n_players=120, seed=7, likelihood="studentt"),
            "RB": relabel(simulate(n_players=80, seed=8, likelihood="studentt"), "RBX")}
    fits = {p: fake_fit(s, p, rng) for p, s in sims.items()}
    tables, sched = league(sims)
    fmt = LeagueFormat(name="t", scoring="ppr", teams=4, lineup={"RB": 1, "WR": 2, "FLEX": 1})
    cfg = ValueConfig(draws=50, horizon_seasons=5, replacement_seasons=3, pool_min_games=4)
    return fits, tables, sched, fmt, cfg


def test_end_to_end_table(end_to_end):
    fits, tables, sched, fmt, cfg = end_to_end
    table, summary = dwar_table(fits, tables, sched, fmt, cfg, "studentt", seed=1)
    assert table.columns == list(OUTPUT_COLUMNS)
    assert table.height > 0 and set(table["position"]) <= {"WR", "RB"}
    assert table["player_id"].is_unique().all()
    q = table.select("dwar_p05", "dwar_p25", "dwar_p50", "dwar_p75", "dwar_p95").to_numpy()
    assert (np.diff(q, axis=1) >= 0).all() and (q >= 0).all()
    assert summary["rookie_draft_picks"] == "unavailable in v1"
    assert summary["seasons_valued"][0] == summary["current_season"] and len(summary["seasons_valued"]) == 6
    again, _ = dwar_table(fits, tables, sched, fmt, cfg, "studentt", seed=1)
    assert table.equals(again)  # fixed seed, same table
