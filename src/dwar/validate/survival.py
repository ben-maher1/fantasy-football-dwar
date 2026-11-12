"""Out-of-sample check of the career survival model (does a player who
played in season t - 1 play in season t), with a focus on older players.

The model DWAR uses trains survival on fitted talent, which exists only
from each position's fit window (RB and WR from 2013). Very few old players
fall in that window, and a 30-year-old star RB came out with a 62% chance
of playing at 34 against 22% in history (ben, 2026-11-10). Variants, all
trained on last season's points per game (stand-in for talent, as in
validate/injury.py) from seasons before t only:

- recent: the production window per position (RB/WR 2013 on, QB/TE 2006 on);
- long: every season from 1999;
- long_h30 / long_h28: long, plus a steeper age decline past 30 (or 28):
  an extra term max(age - knot, 0).

Scored by log loss on seasons 2013-2022 (dev) and 2023-2025 (holdout), for
all players, fantasy-relevant ones (8+ PPG last season), and relevant ones
aged 29 or more. Calibration: mean predicted vs actual by age group among
relevant players.

Run: `uv run python -m dwar.validate.survival --scoring half_ppr`.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from dwar.config import REPO_ROOT, load_scoring
from dwar.features.asof import RawTables, asof_snapshot
from dwar.features.player_tables import build_player_tables
from dwar.value.availability import _design, _sigmoid, fit_logistic

log = logging.getLogger(__name__)

POSITIONS = ("QB", "RB", "WR", "TE")
DEV = tuple(range(2013, 2023))
HOLDOUT = (2023, 2024, 2025)
FIRST = 1999
RECENT_FIRST = {"QB": 2006, "TE": 2006, "RB": 2013, "WR": 2013}
RELEVANT_PPG = 8.0
OLD_AGE = 29.0
AGE_GROUPS = ((0, 27), (27, 29), (29, 31), (31, 33), (33, 99))
VARIANTS = {"recent": (True, None), "long": (False, None), "long_h30": (False, 30.0), "long_h28": (False, 28.0)}


def design(age: np.ndarray, talent: np.ndarray, knot: float | None) -> np.ndarray:
    x = _design(age, talent)
    if knot is None:
        return x
    return np.concatenate([x, np.maximum(np.asarray(age, float) - knot, 0.0)[:, None]], axis=1)


def survival_rows(seasons_tbl: pl.DataFrame) -> pl.DataFrame:
    """One row per player-season played (s), with whether he played in s + 1."""
    played = seasons_tbl.filter(pl.col("games_played") > 0)
    nxt = played.select("player_id", pl.col("season") - 1, next_played=pl.lit(1))
    return (
        played.filter(pl.col("age_sept1").is_not_null())
        .join(nxt, on=["player_id", "season"], how="left")
        .with_columns(pl.col("next_played").fill_null(0))
        .select("player_id", "season", "position", age="age_sept1", ppg="fpts_per_game", y="next_played")
    )


def forecast(rows: pl.DataFrame, t: int) -> pl.DataFrame:
    """Season t's survival forecasts (rows with season == t - 1) per variant,
    each trained on rows with season + 1 < t."""
    out = []
    for p in POSITIONS:
        rp = rows.filter(pl.col("position") == p)
        test = rp.filter(pl.col("season") == t - 1)
        if test.is_empty():
            continue
        preds = {}
        for name, (recent, knot) in VARIANTS.items():
            first = RECENT_FIRST[p] if recent else FIRST
            tr = rp.filter((pl.col("season") + 1 < t) & (pl.col("season") >= first))
            y = tr["y"].to_numpy().astype(float)
            fit = fit_logistic(design(tr["age"].to_numpy(), tr["ppg"].to_numpy(), knot), y, np.ones_like(y))
            preds[name] = _sigmoid(design(test["age"].to_numpy(), test["ppg"].to_numpy(), knot) @ fit.coef)
        out.append(test.with_columns(t=pl.lit(t), **{f"p_{k}": pl.Series(v) for k, v in preds.items()}))
    return pl.concat(out)


def score(df: pl.DataFrame) -> dict:
    y = df["y"].to_numpy()
    res: dict = {"n": df.height, "actual": round(float(y.mean()), 4) if df.height else None}
    for name in VARIANTS:
        p = np.clip(df[f"p_{name}"].to_numpy(), 1e-6, 1 - 1e-6)
        res[name] = {"logloss": round(float(-np.mean(np.where(y == 1, np.log(p), np.log(1 - p)))), 5),
                     "mean_pred": round(float(p.mean()), 4)}
    return res


def summarize(f: pl.DataFrame) -> dict:
    rel = f.filter(pl.col("ppg") >= RELEVANT_PPG)
    out = {"all": score(f), "relevant": score(rel), "relevant_29plus": score(rel.filter(pl.col("age") >= OLD_AGE))}
    for p in POSITIONS:
        rp = rel.filter(pl.col("position") == p)
        out[p] = {"relevant": score(rp), "relevant_29plus": score(rp.filter(pl.col("age") >= OLD_AGE)),
                  "by_age": {f"{lo}-{hi}": score(rp.filter((pl.col("age") >= lo) & (pl.col("age") < hi)))
                             for lo, hi in AGE_GROUPS}}
    return out


def run(scoring: str, out_dir: Path) -> dict:
    snap = asof_snapshot(RawTables.from_cache(), date.today())
    tables = build_player_tables(snap, load_scoring(scoring))
    rows = survival_rows(tables.player_seasons.filter(pl.col("position").is_in(POSITIONS)))
    f = pl.concat([forecast(rows, t) for t in (*DEV, *HOLDOUT)])
    summary = {"scoring": scoring, "variants": list(VARIANTS),
               "dev": summarize(f.filter(pl.col("t").is_in(DEV))),
               "holdout": summarize(f.filter(pl.col("t").is_in(HOLDOUT)))}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    f.write_parquet(out_dir / "forecasts.parquet")
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--scoring", default="half_ppr")
    ap.add_argument("--out", default=str(REPO_ROOT / "reports" / "survival"))
    a = ap.parse_args()
    run(a.scoring, Path(a.out))


if __name__ == "__main__":
    main()
