"""Pull-back for productive young players drafted late or not at all (RBs,
2026-11-10).

ben: low-drafted RBs lose work more easily even when they play well. In the
step-through tests (2014-25), RBs drafted in round 3 or later (or undrafted)
coming off an 8+ PPG season in career years 1-3 scored about what the model
forecast the next season, but 15% less two seasons on and 20% less three on
(ratios by period: 0.83-0.88 and 0.77-0.83). Neither draft terms in games
and survival (kept, value/availability.py) nor draft-scaled growth in the
talent model (tied, not kept) closed that, so the value layer scales those
seasons directly: talent in season s *= factors[h - 1], h = s minus the
player's latest full season (capped at the last factor). Measured on the
leave-one-period-out test: MAE 3.19 -> 3.09 PPG on 276 cases, better in
three of four periods (reports/late_pick/README.md). Like the QB spread
factor, the factors are learned from backtest misses and fixed in config.

Off since 2026-11-10 (ben): it cut productive late-round stars too hard, and
smoother fades were no clearer win (reports/late_pick/README.md).
"""

from __future__ import annotations

import numpy as np
import polars as pl


def qualifying(
    last: pl.DataFrame, static: pl.DataFrame, min_round: int, max_career_year: int, min_ppg: float,
) -> np.ndarray:
    """Players (rows of `last`, from old_star.latest_full_season) whose latest
    full season was in career years 1-`max_career_year` with `min_ppg`+ PPG,
    drafted in round `min_round` or later, or undrafted."""
    j = last.join(static.select("player_id", "draft_round", "draft_year", "first_season"),
                  on="player_id", how="left", maintain_order="left")
    cy = j.select(pl.col("season") - pl.coalesce("draft_year", "first_season") + 1).to_series()
    late = j["draft_round"].is_null() | (j["draft_round"] >= min_round)
    ok = late & cy.is_between(1, max_career_year) & (j["ppg"] >= min_ppg) & j["season"].is_not_null()
    return ok.fill_null(False).to_numpy()


def pull_back(
    talent: np.ndarray, seasons: np.ndarray, current: int, last_season: np.ndarray, hit: np.ndarray,
    factors: tuple[float, ...],
) -> np.ndarray:
    """talent (draws, players, seasons) with seasons after `current` scaled
    for players in `hit`."""
    out = talent.copy()
    f = np.asarray(factors, float)
    for j, s in enumerate(seasons):
        if s <= current:
            continue
        idx = np.flatnonzero(hit)
        if not len(idx):
            continue
        h = np.clip((s - last_season[idx]).astype(int), 1, len(f))
        out[:, idx, j] = talent[:, idx, j] * f[h - 1][None, :]
    return out
