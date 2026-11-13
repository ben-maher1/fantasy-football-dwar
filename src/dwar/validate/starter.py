"""Out-of-sample check: does last season's full-game share help the QB
games-played and survival models? (ben, 2026-11-13: Justin Fields, a
backup with a good scoring rate as a starter, should be "nerfed incredibly
hard" for being a backup.)

A full game is one with at least `min_plays` plays (config/model.yaml; QB
15). Rows as in validate/inactive.py: for each season t, models trained on
seasons before t, forecasting t for players who played in t - 1. Talent
stands in as the value layer has it: points per full game for the games
model (fitted QB talent leaves short weeks out), season points per game
for survival (as `ppg_survival_rows`).

Variants: base (the value layer's current design, healthy non-playing
share and draft terms included) and starter (+ `starter_terms`: the share
of team games played as full games in t - 1, and the share times talent).
Scored on games played (as the value layer uses it) and on full games
(full-game share model, same terms), 2013-2022 (dev) and 2023-2025
(holdout), for all QBs and for backups (under 15% full games) who score
12+ per full game.

Run: `uv run python -m dwar.validate.starter --scoring half_ppr`.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from dwar.config import REPO_ROOT, load_model_config, load_scoring, load_value_config
from dwar.features.asof import RawTables, asof_snapshot
from dwar.features.injuries import InjuryRaw, injury_seasons, injury_weeks
from dwar.features.player_tables import build_player_tables
from dwar.validate.draft_avail import rows
from dwar.validate.inactive import GAMES_FIRST, with_terms
from dwar.validate.inactive import _x as base_x
from dwar.value.availability import (
    HINGE_QUANTILE,
    _sigmoid,
    fit_logistic,
    inactive_seasons,
    starter_seasons,
    starter_terms,
)

log = logging.getLogger(__name__)

POSITION = "QB"
DEV = tuple(range(2013, 2023))
HOLDOUT = (2023, 2024, 2025)
FIRST = 1999
VARIANTS = ("base", "starter")
BACKUP_SHARE = 0.15
BACKUP_PPG = 12.0


def qb_rows(as_of: date, scoring: str, min_plays: int) -> pl.DataFrame:
    """`with_terms` rows for QBs, plus last season's full-game share
    (`fshare`), points per full game (`ppg_full`, season PPG if none) and
    next season's full games (`full_next`)."""
    snap = asof_snapshot(RawTables.from_cache(), as_of)
    tables = build_player_tables(snap, load_scoring(scoring))
    inj = injury_seasons(injury_weeks(InjuryRaw.from_cache(), snap.schedules, tables.player_weeks, as_of))
    ps = tables.player_seasons.filter(pl.col("position") == POSITION)
    r = with_terms(rows(ps, tables.player_static, snap.schedules), inactive_seasons(ps, inj, snap.schedules, as_of), inj)
    pw = tables.player_weeks.filter(pl.col("position") == POSITION)
    full = pw.group_by("player_id", pl.col("season").cast(pl.Int64)).agg(
        full=(pl.col("plays") >= min_plays).sum(),
        ppg_full=pl.col("fpts").filter(pl.col("plays") >= min_plays).mean(),
    )
    share = starter_seasons(tables.player_weeks, snap.schedules, as_of, {POSITION: min_plays})
    return (
        r.join(share.rename({"starter": "fshare"}), on=["player_id", "season"], how="left")
        .join(full, on=["player_id", "season"], how="left")
        .join(full.select("player_id", season=pl.col("season") - 1, full_next="full"), on=["player_id", "season"], how="left")
        .with_columns(fshare=pl.col("fshare").fill_null(0.0), full_next=pl.col("full_next").fill_null(0),
                      ppg_full=pl.coalesce("ppg_full", "ppg"))
    )


def _x(r: pl.DataFrame, draft: bool, variant: str, games: bool, knot: float | None) -> np.ndarray:
    t = r.with_columns(ppg="ppg_full") if games else r
    x = base_x(t, draft, True, games, knot)
    if variant == "starter":
        x = np.concatenate([x, starter_terms(t["fshare"].to_numpy(), t["ppg"].to_numpy())], axis=1)
    return x


def forecast(r: pl.DataFrame, t: int, draft_survival: bool, draft_games: bool) -> pl.DataFrame:
    train = r.filter((pl.col("season") + 1 < t) & (pl.col("season") >= FIRST))
    test = r.filter(pl.col("season") == t - 1)
    gtr = train.filter((pl.col("g_next") > 0) & (pl.col("season") + 1 >= GAMES_FIRST))
    ftr = train.filter(pl.col("season") + 1 >= GAMES_FIRST)
    knot = float(gtr["ppg_full"].quantile(HINGE_QUANTILE))
    cols = {}
    for v in VARIANTS:
        ys = (train["g_next"] > 0).to_numpy().astype(float)
        fs = fit_logistic(_x(train, draft_survival, v, False, None), ys, np.ones_like(ys))
        cols[f"ps_{v}"] = _sigmoid(_x(test, draft_survival, v, False, None) @ fs.coef)
        n = gtr["n_next"].to_numpy()
        fg = fit_logistic(_x(gtr, draft_games, v, True, knot), np.minimum(gtr["g_next"].to_numpy(), n), n)
        cols[f"pg_{v}"] = _sigmoid(_x(test, draft_games, v, True, knot) @ fg.coef)
        n = ftr["n_next"].to_numpy()
        ff = fit_logistic(_x(ftr, draft_games, v, True, knot), np.minimum(ftr["full_next"].to_numpy(), n), n)
        cols[f"pf_{v}"] = _sigmoid(_x(test, draft_games, v, True, knot) @ ff.coef)
    return test.with_columns(t=pl.lit(t), **{k: pl.Series(v) for k, v in cols.items()})


def _binom(g: np.ndarray, n: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ll = float(-np.sum(g * np.log(p) + (n - g) * np.log(1 - p)) / max(n.sum(), 1))
    return round(ll, 5), round(float(np.mean(np.abs(n * p - g))), 4)


def score(f: pl.DataFrame) -> dict:
    res: dict = {"n": f.height}
    if f.is_empty():
        return res
    y = (f["g_next"] > 0).to_numpy()
    pl_ = f.filter(pl.col("g_next") > 0)
    g, n = np.minimum(pl_["g_next"].to_numpy(), pl_["n_next"].to_numpy()), pl_["n_next"].to_numpy()
    fg, fn = np.minimum(f["full_next"].to_numpy(), f["n_next"].to_numpy()), f["n_next"].to_numpy()
    for v in VARIANTS:
        ps = np.clip(f[f"ps_{v}"].to_numpy(), 1e-6, 1 - 1e-6)
        gl, gm = _binom(g, n, pl_[f"pg_{v}"].to_numpy())
        fl, fm = _binom(fg, fn, f[f"pf_{v}"].to_numpy())
        res[v] = {
            "surv_logloss": round(float(-np.mean(np.where(y, np.log(ps), np.log(1 - ps)))), 5),
            "games_logloss": gl, "games_mae": gm,
            "games_pred": round(float(np.mean(n * pl_[f"pg_{v}"].to_numpy())), 3) if len(g) else None,
            "full_logloss": fl, "full_mae": fm, "full_pred": round(float(np.mean(fn * f[f"pf_{v}"].to_numpy())), 3),
        }
    res["surv_actual"] = round(float(y.mean()), 4)
    res["games_actual"] = round(float(g.mean()), 3) if len(g) else None
    res["full_actual"] = round(float(fg.mean()), 3)
    return res


def summarize(f: pl.DataFrame) -> dict:
    backup = f.filter((pl.col("fshare") < BACKUP_SHARE) & (pl.col("ppg_full") >= BACKUP_PPG))
    return {"all": score(f), "backup_12ppg": score(backup), "starter_50pct": score(f.filter(pl.col("fshare") >= 0.5))}


def run(scoring: str, out_dir: Path) -> dict:
    as_of = date.today()  # noqa: DTZ011
    min_plays = load_model_config().min_plays.get(POSITION, 15)
    r = qb_rows(as_of, scoring, min_plays)
    cfg = load_value_config()
    f = pl.concat([forecast(r, t, POSITION in cfg.draft_survival, POSITION in cfg.draft_games) for t in (*DEV, *HOLDOUT)])
    summary = {"scoring": scoring, "min_plays": min_plays, "dev": summarize(f.filter(pl.col("t").is_in(DEV))),
               "holdout": summarize(f.filter(pl.col("t").is_in(HOLDOUT)))}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--scoring", default="half_ppr")
    ap.add_argument("--out", default=str(REPO_ROOT / "reports" / "starter"))
    a = ap.parse_args()
    print(json.dumps(run(a.scoring, Path(a.out)), indent=1))


if __name__ == "__main__":
    main()
