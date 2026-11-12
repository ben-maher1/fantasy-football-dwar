"""The step-through filter: calibrated when the settings are the truth, and
point in time (a forecast never changes when later games change)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from dwar.features.player_tables import PlayerTables
from dwar.model.data import build_model_data
from dwar.validate.stepthrough import HORIZON, Settings, step_through

from ..model.simulate import (
    FIRST_SEASON,
    GAME_TRUTH,
    LATENT_TRUTH,
    LIKELIHOOD_TRUTH,
    TRUE_GAMMA,
    simulate,
)

S = 20  # settings "draws": the truth repeated


def true_settings(d, game_steps: bool) -> Settings:
    t, lk = LATENT_TRUTH["sqrt"], LIKELIHOOD_TRUTH["sqrt_hurdle"]
    one = np.ones(S)
    gamma = np.array([[TRUE_GAMMA["sqrt"].get(n, 0.0) for n in d.prospect_x_names]] * S)
    g = GAME_TRUTH if game_steps else {"sigma_game": 0.0, "lam_game": 0.0}
    return Settings(
        likelihood="sqrt_hurdle", **{k: v * one for k, v in t.items()}, **{k: v * one for k, v in lk.items()},
        zero_value=d.zero_week_mean, gamma=gamma, alpha_round1=0 * one, round1_fit=np.zeros(3),
        growth=np.zeros((S, 2)), growth_pick=np.zeros((S, 2)),
        sigma_game=g["sigma_game"] * one, lam_game=g["lam_game"] * one, prospect_names=d.prospect_x_names,
    )


@pytest.fixture(scope="module")
def sim():
    return simulate(n_players=120, game_steps=True, seed=3)


def test_calibrated_with_true_settings(sim):
    """With the generating settings, the 90% and 50% ranges cover about 90%
    and 50% of outcomes, for the rest of a season and the next."""
    out = step_through(true_settings(sim.data, True), sim.data, (2014, 2015, 2016), 2025, seed=1, m=30, r=300)
    assert set(out["horizon"].unique()) <= set(range(-1, HORIZON + 1))
    for h in (0, 1):
        pit = out.filter(pl.col("horizon") == h)["pit"].to_numpy()
        assert len(pit) > 200
        assert 0.84 <= (np.abs(pit - 0.5) <= 0.45).mean() <= 0.96, h
        assert 0.42 <= (np.abs(pit - 0.5) <= 0.25).mean() <= 0.58, h


def test_no_lookahead(sim):
    """Changing a player's last season changes no forecast made before it."""
    tables = sim.tables
    last = tables.player_weeks.group_by("player_id").agg(pl.col("season").max().alias("last"))
    changed = tables.player_weeks.join(last, on="player_id").with_columns(
        fpts=pl.when(pl.col("season") == pl.col("last")).then(pl.col("fpts") * 3 + 5).otherwise("fpts")
    ).drop("last")
    t2 = PlayerTables(as_of=tables.as_of, player_weeks=changed, player_seasons=tables.player_seasons,
                      player_static=tables.player_static)
    d2 = build_model_data(t2, first_season=FIRST_SEASON, unplayed_rookies=False)
    s = true_settings(sim.data, True)
    a = step_through(s, sim.data, (2013, 2014), 2025, seed=5, m=10, r=50)
    b = step_through(s, d2, (2013, 2014), 2025, seed=5, m=10, r=50)
    keys = ["player_id", "season", "games_before", "horizon"]
    lasts = last.rename({"last": "last_season"})
    j = a.join(b, on=keys, suffix="_b").join(lasts, on="player_id")
    before = j.filter(pl.col("season") < pl.col("last_season"))
    assert before.height > 100
    assert np.allclose(before["pred_mean"].to_numpy(), before["pred_mean_b"].to_numpy())


def test_early_noise_flattens_early_games_only(sim):
    from dataclasses import replace

    from dwar.validate.stepthrough import _loglik

    s = replace(true_settings(sim.data, False), lam_noise=np.full(S, 1.0))
    base = true_settings(sim.data, False)
    draw = np.zeros(2, dtype=int)
    latent = np.array([2.0, 4.0])   # a particle near the week's points and one far off
    def spread(st, cg):
        ll = _loglik(st, latent, 4.0, draw, cg)
        return ll[0] - ll[1]
    # A first game separates near from far particles less than a veteran game does.
    assert spread(s, 0.0) < spread(base, 0.0)
    assert spread(s, 200.0) == pytest.approx(spread(base, 200.0), rel=1e-6)
