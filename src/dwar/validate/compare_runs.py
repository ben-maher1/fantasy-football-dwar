"""Paired comparison of two backtest runs on the same player-seasons.

Used to decide which situation factors stay: a factor group is kept when
removing it makes development MAE or log score worse at a position it
applies to, and neither better.

    uv run python -m dwar.validate.compare_runs A step5
"""

from __future__ import annotations

import json
import sys

import numpy as np
import polars as pl

from dwar.config import data_dir
from dwar.validate.score import BOOTSTRAP, scored_rows

POSITIONS = ("QB", "RB", "WR", "TE")
KEYS = ["player_id", "season", "position"]


def run_forecasts(name: str, seasons: tuple[int, ...] | None = None) -> pl.DataFrame:
    paths = sorted((data_dir() / "experiments" / name / "forecasts").glob("*.parquet"))
    frames = [pl.read_parquet(p) for p in paths if seasons is None or int(p.stem) in seasons]
    if not frames:
        raise FileNotFoundError(f"no forecasts for run {name}")
    rows, _ = scored_rows(pl.concat(frames, how="diagonal_relaxed"))
    return rows


def _mae(actual: np.ndarray, pred: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * np.abs(actual - pred)) / np.sum(w))


def paired(a: pl.DataFrame, b: pl.DataFrame, seed: int = 0) -> dict:
    """Run `a` minus run `b` (positive = `a` better), with bootstrap 90%
    intervals over player-seasons, plus each run's calibration."""
    cols = ["fpts_per_game", "games_played", "talent_mean", "log_score", "pit"]
    rows = a.select(*KEYS, *cols).join(b.select(*KEYS, *cols[2:]), on=KEYS, suffix="_b", how="inner")
    out = {}
    for name in ("all", *POSITIONS):
        g = rows if name == "all" else rows.filter(pl.col("position") == name)
        if g.is_empty():
            continue
        y, w = g["fpts_per_game"].to_numpy(), g["games_played"].to_numpy().astype(float)
        pa, pb = g["talent_mean"].to_numpy(), g["talent_mean_b"].to_numpy()
        la, lb = g["log_score"].to_numpy(), g["log_score_b"].to_numpy()
        boot = np.random.default_rng(seed).integers(0, len(g), size=(BOOTSTRAP, len(g)))
        mae_gain = np.array([_mae(y[i], pb[i], w[i]) - _mae(y[i], pa[i], w[i]) for i in boot])
        ls_gain = (la[boot] - lb[boot]).mean(axis=1)

        def calib(pit: np.ndarray) -> dict:
            return {"cover90": float((np.abs(pit - 0.5) <= 0.45).mean()),
                    "cover50": float((np.abs(pit - 0.5) <= 0.25).mean()),
                    "bottom10": float((pit < 0.1).mean())}

        out[name] = {
            "n": len(g),
            "mae": [_mae(y, pa, w), _mae(y, pb, w)],
            "mae_gain": _mae(y, pb, w) - _mae(y, pa, w),
            "mae_gain_90": [float(x) for x in np.percentile(mae_gain, [5, 95])],
            "log_score": [float(la.mean()), float(lb.mean())],
            "log_score_gain": float(la.mean() - lb.mean()),
            "log_score_gain_90": [float(x) for x in np.percentile(ls_gain, [5, 95])],
            "calibration": [calib(g["pit"].to_numpy()), calib(g["pit_b"].to_numpy())],
        }
    return out


def table(result: dict, a: str, b: str) -> str:
    lines = [f"{a} vs {b} (positive = {a} better)",
             f"{'group':6} {'n':>5} {'MAE gain':>22} {'log score gain':>24} {'90% cover':>12} {'bottom 10%':>12}"]
    for g, r in result.items():
        c = r["calibration"]
        lines.append(
            f"{g:6} {r['n']:5d} {r['mae_gain']:+.3f} ({r['mae_gain_90'][0]:+.3f},{r['mae_gain_90'][1]:+.3f}) "
            f"{r['log_score_gain']:+.3f} ({r['log_score_gain_90'][0]:+.3f},{r['log_score_gain_90'][1]:+.3f}) "
            f"{c[0]['cover90']:.2f}/{c[1]['cover90']:.2f} {c[0]['bottom10']:.2f}/{c[1]['bottom10']:.2f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    a, b = sys.argv[1], sys.argv[2]
    seasons = tuple(int(s) for s in sys.argv[3].split(",")) if len(sys.argv) > 3 else None
    result = paired(run_forecasts(a, seasons), run_forecasts(b, seasons))
    print(table(result, a, b))
    out = data_dir() / "experiments" / f"compare_{a}_vs_{b}.json"
    out.write_text(json.dumps(result, indent=2))
