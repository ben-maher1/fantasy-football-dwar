"""Out-of-sample check: does draft capital help the games-played and
survival models for players early in their careers? (ben, 2026-11-10:
low-drafted RBs lose work more easily even when they play well.)

For each season t, models are trained on seasons before t only (from 1999),
with last season's PPG standing in for talent (as in validate/survival.py),
then forecast season t for players who played in t - 1:

- survival: does he play in t (log loss);
- games: games played in t among those who played (binomial log loss per
  game, MAE in games).

Variants: base (age, age^2, PPG; games adds a talent hinge at the 75th
percentile, as the value layer does) and draft (+ two terms for career
years 1-4: a flag, and the flag times log draft pick centered at pick 64,
undrafted counted as pick 260). Scored on 2013-2022 (dev) and 2023-2025
(holdout), for all, fantasy-relevant (8+ PPG), and relevant players in
career years 1-4.

Run: `uv run python -m dwar.validate.draft_avail --scoring half_ppr`.
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
from dwar.value.availability import (
    HINGE_QUANTILE,
    _design,
    _sigmoid,
    fit_logistic,
    season_games,
)

log = logging.getLogger(__name__)

POSITIONS = ("QB", "RB", "WR", "TE")
DEV = tuple(range(2013, 2023))
HOLDOUT = (2023, 2024, 2025)
FIRST = 1999
RELEVANT_PPG = 8.0
EARLY_YEARS = 4
PICK_CENTER = 64.0
UNDRAFTED_PICK = 260.0


def draft_terms(career_year: np.ndarray, pick: np.ndarray) -> np.ndarray:
    """(n, 2): early flag (career years 1-EARLY_YEARS) and flag x log(pick / 64)."""
    early = ((career_year >= 1) & (career_year <= EARLY_YEARS)).astype(float)
    p = np.where(np.isnan(pick), UNDRAFTED_PICK, pick)
    return np.column_stack([early, early * np.log(p / PICK_CENTER)])


def rows(player_seasons: pl.DataFrame, static: pl.DataFrame, schedules: pl.DataFrame) -> pl.DataFrame:
    """One row per player-season played (s), with next season's games."""
    games = season_games(schedules).select("season", "games")
    played = player_seasons.filter((pl.col("games_played") > 0) & pl.col("age_sept1").is_not_null())
    first = played.group_by("player_id").agg(first=pl.col("season").min())
    nxt = played.select("player_id", pl.col("season") - 1, g_next="games_played")
    return (
        played.join(first, on="player_id")
        .join(static.select("player_id", "draft_year", "draft_pick"), on="player_id", how="left")
        .join(nxt, on=["player_id", "season"], how="left")
        .join(games.select(pl.col("season") - 1, n_next="games"), on="season", how="left")
        .with_columns(
            # Career year of the season being forecast (s + 1).
            cy=(pl.col("season") + 1 - pl.coalesce("draft_year", "first") + 1),
            g_next=pl.col("g_next").fill_null(0),
        )
        .select("player_id", "season", "position", "cy", "g_next", "n_next",
                age="age_sept1", ppg="fpts_per_game", pick="draft_pick")
    )


def _x(r: pl.DataFrame, draft: bool, knot: float | None) -> np.ndarray:
    x = _design(r["age"].to_numpy() + 1.0, r["ppg"].to_numpy(), None, knot)
    if not draft:
        return x
    return np.concatenate([x, draft_terms(r["cy"].to_numpy().astype(float),
                                          r["pick"].cast(pl.Float64).to_numpy())], axis=1)


def forecast(r_all: pl.DataFrame, t: int) -> pl.DataFrame:
    out = []
    for p in POSITIONS:
        rp = r_all.filter(pl.col("position") == p)
        train = rp.filter((pl.col("season") + 1 < t) & (pl.col("season") >= FIRST))
        test = rp.filter(pl.col("season") == t - 1)
        if test.is_empty():
            continue
        gtr = train.filter(pl.col("g_next") > 0)
        knot = float(gtr["ppg"].quantile(HINGE_QUANTILE))
        cols = {}
        for name, draft in (("base", False), ("draft", True)):
            ys = (train["g_next"] > 0).to_numpy().astype(float)
            fs = fit_logistic(_x(train, draft, None), ys, np.ones_like(ys))
            cols[f"ps_{name}"] = _sigmoid(_x(test, draft, None) @ fs.coef)
            fg = fit_logistic(_x(gtr, draft, knot), np.minimum(gtr["g_next"].to_numpy(), gtr["n_next"].to_numpy()),
                              gtr["n_next"].to_numpy())
            cols[f"pg_{name}"] = _sigmoid(_x(test, draft, knot) @ fg.coef)
        out.append(test.with_columns(t=pl.lit(t), **{k: pl.Series(v) for k, v in cols.items()}))
    return pl.concat(out)


def score(f: pl.DataFrame) -> dict:
    res: dict = {"n": f.height}
    y = (f["g_next"] > 0).to_numpy()
    pl_ = f.filter(pl.col("g_next") > 0)
    g, n = np.minimum(pl_["g_next"].to_numpy(), pl_["n_next"].to_numpy()), pl_["n_next"].to_numpy()
    for v in ("base", "draft"):
        ps = np.clip(f[f"ps_{v}"].to_numpy(), 1e-6, 1 - 1e-6)
        pg = np.clip(pl_[f"pg_{v}"].to_numpy(), 1e-6, 1 - 1e-6)
        res[v] = {
            "surv_logloss": round(float(-np.mean(np.where(y, np.log(ps), np.log(1 - ps)))), 5) if f.height else None,
            "surv_pred": round(float(ps.mean()), 4) if f.height else None,
            "games_logloss": round(float(-np.sum(g * np.log(pg) + (n - g) * np.log(1 - pg)) / max(n.sum(), 1)), 5),
            "games_mae": round(float(np.mean(np.abs(n * pg - g))), 4) if len(g) else None,
            "games_pred": round(float(np.mean(n * pg)), 3) if len(g) else None,
        }
    res["surv_actual"] = round(float(y.mean()), 4) if f.height else None
    res["games_actual"] = round(float(g.mean()), 3) if len(g) else None
    return res


def summarize(f: pl.DataFrame) -> dict:
    rel = f.filter(pl.col("ppg") >= RELEVANT_PPG)
    early = rel.filter(pl.col("cy").is_between(1, EARLY_YEARS))
    out = {"all": score(f), "relevant": score(rel), "relevant_early": score(early)}
    for p in POSITIONS:
        out[p] = {"relevant": score(rel.filter(pl.col("position") == p)),
                  "relevant_early": score(early.filter(pl.col("position") == p))}
    return out


def run(scoring: str, out_dir: Path) -> dict:
    snap = asof_snapshot(RawTables.from_cache(), date.today())
    tables = build_player_tables(snap, load_scoring(scoring))
    r = rows(tables.player_seasons.filter(pl.col("position").is_in(POSITIONS)), tables.player_static, snap.schedules)
    f = pl.concat([forecast(r, t) for t in (*DEV, *HOLDOUT)])
    summary = {"scoring": scoring, "dev": summarize(f.filter(pl.col("t").is_in(DEV))),
               "holdout": summarize(f.filter(pl.col("t").is_in(HOLDOUT)))}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--scoring", default="half_ppr")
    ap.add_argument("--out", default=str(REPO_ROOT / "reports" / "draft_avail"))
    a = ap.parse_args()
    run(a.scoring, Path(a.out))


if __name__ == "__main__":
    main()
