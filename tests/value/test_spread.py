import numpy as np
import polars as pl
import pytest

from dwar.value.spread import spread_factor, widen


def history(seasons: list[int], miss: float) -> pl.DataFrame:
    n = len(seasons)
    return pl.DataFrame({"season": seasons, "fpts_per_game": [10.0 + miss * (-1) ** i for i in range(n)],
                         "pred_mean": [10.0] * n, "pred_sd": [2.0] * n})


def test_factor_is_rms_standardized_miss_floored_at_one():
    assert spread_factor(history([2012, 2012, 2013, 2013], 3.0)) == pytest.approx(1.5)
    assert spread_factor(history([2012, 2012, 2013, 2013], 1.0)) == 1.0  # never narrows
    assert spread_factor(history([2012, 2012], 3.0)) == 1.0  # one season is too little history


def test_widen_keeps_the_mean_and_scales_the_spread():
    x = np.random.default_rng(0).normal(5.0, 2.0, (4000, 3))
    w = widen(x, 1.5, axis=0)
    np.testing.assert_allclose(w.mean(axis=0), x.mean(axis=0))
    np.testing.assert_allclose(w.std(axis=0), 1.5 * x.std(axis=0))
    assert widen(x, 1.0) is x
