import numpy as np
import polars as pl
import pytest

from dwar.validate.forecast import kde_log_density, pit
from dwar.validate.score import compare, coverage, summarize


def rows(n: int, dwar_noise: float, base_noise: float, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    actual = rng.uniform(2, 20, n)
    cols = {
        "player_id": [f"p{i}" for i in range(n)], "position": ["WR"] * n, "is_rookie": [False] * n,
        "has_forecast": [True] * n, "games_played": rng.integers(4, 18, n), "fpts_per_game": actual,
        "talent_mean": actual + rng.normal(0, dwar_noise, n),
        "log_score": np.full(n, -1.0), "pit": rng.uniform(0, 1, n),
    }
    for m in ("last", "weighted3", "age_adjusted"):
        cols[m] = actual + rng.normal(0, base_noise, n)
        cols[f"{m}_log_score"] = np.full(n, -2.0)
    return pl.DataFrame(cols)


def test_better_forecast_beats_baselines():
    out = compare(rows(2000, dwar_noise=1.0, base_noise=3.0))
    assert out["beats_all"]
    for b in out["baselines"].values():
        assert b["mae_gain"] > 0 and b["mae_gain_90"][0] > 0 and b["log_score_gain"] == pytest.approx(1.0, abs=0.01)  # floor shaves a little


def test_perfect_forecast_has_zero_mae():
    assert compare(rows(100, dwar_noise=0.0, base_noise=1.0))["dwar"]["mae"] == 0.0


def test_worse_forecast_fails_and_summary_gates():
    r = rows(500, dwar_noise=3.0, base_noise=1.0)
    assert not compare(r)["beats_all"]
    assert summarize(r, ("WR",))["passed"] is False


def test_coverage_of_uniform_pit_matches_levels():
    cov = coverage(rows(20_000, 1.0, 1.0))
    for level, share in cov.items():
        assert share == pytest.approx(float(level), abs=0.02)


def test_too_narrow_forecasts_undercover():
    rng = np.random.default_rng(1)
    pits = [pit(rng.normal(0, 1, 2000), x) for x in rng.normal(0, 2, 2000)]  # truth twice as wide
    cov = coverage(pl.DataFrame({"pit": pits}))
    assert cov["0.5"] < 0.35 and cov["0.9"] < 0.75


def test_kde_log_density_near_normal():
    rng = np.random.default_rng(2)
    assert kde_log_density(rng.normal(0, 1, 5000), 0.0) == pytest.approx(-0.5 * np.log(2 * np.pi), abs=0.05)


def test_floor_caps_a_freak_miss():
    from dwar.validate.score import FLOOR_RANGE, FLOOR_WEIGHT, floored

    out = floored(np.array([-150.0, -2.5]))
    assert out[0] == pytest.approx(np.log(FLOOR_WEIGHT / FLOOR_RANGE))  # about -8.3, not -150
    assert out[1] == pytest.approx(-2.5, abs=0.02)                      # ordinary scores barely move
