"""Spread factors: widen a position's forecast ranges by how much its past
forecasts understated their misses.

Step 5 found QB season forecasts too narrow (a 90% range caught about 79%
of outcomes): the talent model does not see what moves QBs year to year
(team, coaching, line changes). Until situation factors exist, a forecast
draw x is stretched about the forecast mean m as m + k * (x - m), where k is
the root mean square of past standardized misses (actual - mean) / sd,
floored at 1 so ranges are only ever widened.

In the backtest, k for season s comes only from forecasts of seasons before
s, so it is point-in-time. Weekly runs read k from config/value.yaml, set from
the latest backtest.
"""

from __future__ import annotations

import numpy as np
import polars as pl

MIN_SEASONS = 2  # seasons of past forecasts needed before widening


def spread_factor(history: pl.DataFrame) -> float:
    """k from past forecasts (fpts_per_game, pred_mean, pred_sd, season),
    measured before any widening; 1 with too little history."""
    h = history.filter(pl.col("pred_sd") > 0)
    if h["season"].n_unique() < MIN_SEASONS:
        return 1.0
    z = (h["fpts_per_game"] - h["pred_mean"]) / h["pred_sd"]
    return max(1.0, float(np.sqrt((z**2).mean())))


def widen(draws: np.ndarray, k: float, axis: int = 0) -> np.ndarray:
    """Stretch draws about their mean along `axis` by factor `k`."""
    if k == 1.0:
        return draws
    m = draws.mean(axis=axis, keepdims=True)
    return m + k * (draws - m)
