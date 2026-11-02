"""Score DWAR against the baselines and check calibration.

All methods are scored on the same player-seasons: those with a DWAR
forecast and every baseline (excluded rows are counted). MAE is weighted
by games played; the log score is a plain mean per player-season. A
difference is reported with a bootstrap 90% interval over player-seasons.

Every method's predictive density gets the same floor before the log is
taken: 1% of probability spread evenly over 0-40 PPG (chosen by ben,
2026-10-30). Without it, a handful of freak breakouts, scored far in a
forecast's tail, outweighed hundreds of ordinary player-seasons.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from dwar.validate.baselines import METHODS

FLOOR_WEIGHT, FLOOR_RANGE = 0.01, 40.0
COVERAGE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
BOOTSTRAP = 1000


def scored_rows(results: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, int]]:
    """Rows every method can be scored on, and counts of rows left out."""
    has_all = pl.col("has_forecast") & pl.all_horizontal([pl.col(m).is_not_null() for m in METHODS])
    excluded = {
        "no_dwar_forecast": int((~results["has_forecast"]).sum()),
        "no_baseline": int(results.filter(pl.col("has_forecast") & ~has_all).height),
    }
    return results.filter(has_all), excluded


def floored(log_score: np.ndarray) -> np.ndarray:
    """Log density after mixing in the floor (see module docstring)."""
    return np.log((1 - FLOOR_WEIGHT) * np.exp(log_score) + FLOOR_WEIGHT / FLOOR_RANGE)


def _mae(actual: np.ndarray, pred: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * np.abs(actual - pred)) / np.sum(w))


def compare(rows: pl.DataFrame, seed: int = 0) -> dict:
    """DWAR vs each baseline on `rows`: MAE and log score, the differences
    (positive = DWAR better) with bootstrap 90% intervals, and `beats`."""
    rng = np.random.default_rng(seed)
    actual, w = rows["fpts_per_game"].to_numpy(), rows["games_played"].to_numpy().astype(float)
    dwar_pred, dwar_ls = rows["talent_mean"].to_numpy(), floored(rows["log_score"].to_numpy())
    boot = rng.integers(0, len(rows), size=(BOOTSTRAP, len(rows)))
    out = {
        "n": len(rows),
        "dwar": {"mae": _mae(actual, dwar_pred, w), "log_score": float(dwar_ls.mean())},
        "baselines": {},
    }
    for m in METHODS:
        pred, ls = rows[m].to_numpy(), floored(rows[f"{m}_log_score"].to_numpy())
        mae_gain = np.array([_mae(actual[i], pred[i], w[i]) - _mae(actual[i], dwar_pred[i], w[i]) for i in boot])
        ls_gain = (dwar_ls[boot] - ls[boot]).mean(axis=1)
        mae_diff = _mae(actual, pred, w) - out["dwar"]["mae"]
        ls_diff = float(dwar_ls.mean() - ls.mean())
        out["baselines"][m] = {
            "mae": _mae(actual, pred, w),
            "log_score": float(ls.mean()),
            "mae_gain": mae_diff, "mae_gain_90": [float(x) for x in np.percentile(mae_gain, [5, 95])],
            "log_score_gain": ls_diff, "log_score_gain_90": [float(x) for x in np.percentile(ls_gain, [5, 95])],
            "beats": bool(mae_diff > 0 and ls_diff > 0),
        }
    out["beats_all"] = all(b["beats"] for b in out["baselines"].values())
    return out


def coverage(rows: pl.DataFrame) -> dict[str, float]:
    """Share of outcomes inside DWAR's central intervals at each level."""
    centered = np.abs(rows["pit"].to_numpy() - 0.5)
    return {f"{level:.1f}": float((centered <= level / 2).mean()) for level in COVERAGE_LEVELS}


def summarize(results: pl.DataFrame, positions: tuple[str, ...], seed: int = 0) -> dict:
    """Comparison and coverage overall, per position, and for rookies.
    `passed` requires DWAR to beat every baseline overall and at every
    position (rookies are reported, not gated)."""
    rows, excluded = scored_rows(results)
    groups = {"all": rows} | {p: rows.filter(pl.col("position") == p) for p in positions}
    groups["rookies"] = rows.filter(pl.col("is_rookie"))
    out = {"excluded": excluded, "groups": {}}
    for name, g in groups.items():
        if g.is_empty():
            continue
        out["groups"][name] = compare(g, seed) | {"coverage": coverage(g)}
    gated = ["all", *positions]
    out["passed"] = all(out["groups"].get(n, {}).get("beats_all", False) for n in gated)
    return out
