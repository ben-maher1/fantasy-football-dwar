import numpy as np
import polars as pl

from dwar.value.scoring_scale import scale_factors


def test_factor_is_own_ratio_shrunk_toward_position():
    rows = []
    # WR a: 16 games, 10 base -> 13 new (ratio 1.3). WR b: 16 games, 10 -> 11 (1.1). Position ratio 1.2.
    for i, new in (("a", 13.0), ("b", 11.0)):
        rows += [{"player_id": i, "position": "WR", "season": 2025, "base": 10.0, "new": new} for _ in range(16)]
    # Old season outside the 3-year window is ignored.
    rows += [{"player_id": "a", "position": "WR", "season": 2020, "base": 10.0, "new": 30.0}]
    w = pl.DataFrame(rows)
    f = scale_factors(w, "base", "new", ["a", "b", "c"], ["WR", "WR", "WR"], 2026, prior_games=8.0)
    # a: (208 + 8*1.2*10) / (160 + 80) = 304/240
    assert np.isclose(f[0], 304 / 240) and np.isclose(f[1], (176 + 96) / 240)
    assert np.isclose(f[2], 1.2)                 # no games: the position ratio
    assert f[1] < 1.2 < f[0]
