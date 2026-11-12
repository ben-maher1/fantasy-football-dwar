"""Out-of-sample check: do healthy non-playing weeks help the games-played
and survival models? (ben, 2026-11-12: a backup who sits healthy most
weeks, like an undrafted WR on and off the practice squad, should not get
credit for the randomness of a thin record; his missed games are not
injuries.)

For each season t, models are trained on seasons before t only, with last
season's PPG standing in for talent (as in validate/draft_avail.py), then
forecast season t for players who played in t - 1:

- survival: does he play in t (log loss); trained from 1999, with seasons
  before injury reports (2009) flagged as unknown;
- games: games played in t among those who played (binomial log loss per
  game, MAE in games); trained from 2011, with injury totals for t - 1 and
  t - 2 as the value layer has them.

Variants: base (the value layer's design: age, age^2, PPG, a talent hinge
for games, injury totals for games, draft terms where config/value.yaml has
them) and inactive (+ `inactive_terms`: last season's healthy non-playing
share). Scored on 2013-2022 (dev) and 2023-2025 (holdout), for all,
fantasy-relevant (8+ PPG) and low (under 8 PPG) players.

Run: `uv run python -m dwar.validate.inactive --scoring half_ppr`.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from dwar.config import REPO_ROOT, load_scoring, load_value_config
from dwar.features.asof import RawTables, asof_snapshot
from dwar.features.injuries import FIRST_SEASON as INJURY_FIRST_SEASON
from dwar.features.injuries import InjuryRaw, injury_seasons, injury_weeks
from dwar.features.player_tables import build_player_tables
from dwar.validate.draft_avail import draft_terms, rows
from dwar.value.availability import (
    HINGE_QUANTILE,
    _design,
    _sigmoid,
    fit_logistic,
    inactive_seasons,
    inactive_terms,
    injury_history_terms,
)

log = logging.getLogger(__name__)

POSITIONS = ("QB", "RB", "WR", "TE")
DEV = tuple(range(2013, 2023))
HOLDOUT = (2023, 2024, 2025)
FIRST = 1999
GAMES_FIRST = INJURY_FIRST_SEASON + 2   # games rows need two seasons of reports
RELEVANT_PPG = 8.0
VARIANTS = ("base", "inactive")


def with_terms(r: pl.DataFrame, inactive: pl.DataFrame, injury: pl.DataFrame) -> pl.DataFrame:
    """`rows` plus last season's inactive share (`ina`, `ina_known`) and the
    injury totals for the season forecast (`inj`)."""
    j = r.with_columns(pl.col("season").cast(pl.Int64)).join(inactive, on=["player_id", "season"], how="left")
    inj = injury_history_terms(j.select("player_id", t=pl.col("season") + 1), injury)
    return j.with_columns(
        ina=pl.col("inactive").fill_null(0.0),
        ina_known=pl.col("season") >= INJURY_FIRST_SEASON,
        inj=pl.Series(inj.tolist(), dtype=pl.List(pl.Float64)),
    )


def _x(r: pl.DataFrame, draft: bool, inactive: bool, games: bool, knot: float | None) -> np.ndarray:
    inj = np.array(r["inj"].to_list()) if games else None
    x = _design(r["age"].to_numpy() + 1.0, r["ppg"].to_numpy(), inj, knot)
    cols = [x]
    if draft:
        cols.append(draft_terms(r["cy"].to_numpy().astype(float), r["pick"].cast(pl.Float64).to_numpy()))
    if inactive:
        cols.append(inactive_terms(r["ina"].to_numpy(), r["ina_known"].to_numpy()))
    return np.concatenate(cols, axis=1)


def forecast(r_all: pl.DataFrame, t: int, draft_survival: tuple[str, ...], draft_games: tuple[str, ...]) -> pl.DataFrame:
    out = []
    for p in POSITIONS:
        rp = r_all.filter(pl.col("position") == p)
        train = rp.filter((pl.col("season") + 1 < t) & (pl.col("season") >= FIRST))
        test = rp.filter(pl.col("season") == t - 1)
        if test.is_empty():
            continue
        gtr = train.filter((pl.col("g_next") > 0) & (pl.col("season") + 1 >= GAMES_FIRST))
        knot = float(gtr["ppg"].quantile(HINGE_QUANTILE))
        ds, dg = p in draft_survival, p in draft_games
        cols = {}
        for v in VARIANTS:
            ina = v == "inactive"
            ys = (train["g_next"] > 0).to_numpy().astype(float)
            fs = fit_logistic(_x(train, ds, ina, False, None), ys, np.ones_like(ys))
            cols[f"ps_{v}"] = _sigmoid(_x(test, ds, ina, False, None) @ fs.coef)
            n = gtr["n_next"].to_numpy()
            fg = fit_logistic(_x(gtr, dg, ina, True, knot), np.minimum(gtr["g_next"].to_numpy(), n), n)
            cols[f"pg_{v}"] = _sigmoid(_x(test, dg, ina, True, knot) @ fg.coef)
        out.append(test.with_columns(t=pl.lit(t), **{k: pl.Series(v) for k, v in cols.items()}))
    return pl.concat(out)


def score(f: pl.DataFrame) -> dict:
    res: dict = {"n": f.height}
    if f.is_empty():
        return res
    y = (f["g_next"] > 0).to_numpy()
    pl_ = f.filter(pl.col("g_next") > 0)
    g, n = np.minimum(pl_["g_next"].to_numpy(), pl_["n_next"].to_numpy()), pl_["n_next"].to_numpy()
    for v in VARIANTS:
        ps = np.clip(f[f"ps_{v}"].to_numpy(), 1e-6, 1 - 1e-6)
        pg = np.clip(pl_[f"pg_{v}"].to_numpy(), 1e-6, 1 - 1e-6)
        res[v] = {
            "surv_logloss": round(float(-np.mean(np.where(y, np.log(ps), np.log(1 - ps)))), 5),
            "surv_pred": round(float(ps.mean()), 4),
            "games_logloss": round(float(-np.sum(g * np.log(pg) + (n - g) * np.log(1 - pg)) / max(n.sum(), 1)), 5),
            "games_mae": round(float(np.mean(np.abs(n * pg - g))), 4) if len(g) else None,
            "games_pred": round(float(np.mean(n * pg)), 3) if len(g) else None,
        }
    res["surv_actual"] = round(float(y.mean()), 4)
    res["games_actual"] = round(float(g.mean()), 3) if len(g) else None
    return res


def by_share(f: pl.DataFrame) -> list[dict]:
    """Survival: actual vs each variant's forecast by inactive-share bucket."""
    b = f.with_columns(bucket=pl.when(pl.col("ina") < 0.1).then(pl.lit("<10%"))
                       .when(pl.col("ina") < 0.3).then(pl.lit("10-30%"))
                       .when(pl.col("ina") < 0.6).then(pl.lit("30-60%")).otherwise(pl.lit("60%+")))
    return (
        b.group_by("bucket").agg(n=pl.len(), actual=(pl.col("g_next") > 0).mean(),
                                 **{v: pl.col(f"ps_{v}").mean() for v in VARIANTS})
        .sort("bucket").with_columns(pl.col(pl.Float64).round(3)).to_dicts()
    )


def summarize(f: pl.DataFrame) -> dict:
    rel = f.filter(pl.col("ppg") >= RELEVANT_PPG)
    low = f.filter(pl.col("ppg") < RELEVANT_PPG)
    out = {"all": score(f), "relevant": score(rel), "low": score(low), "survival_by_share": by_share(f)}
    for p in POSITIONS:
        fp = f.filter(pl.col("position") == p)
        out[p] = {"all": score(fp), "relevant": score(fp.filter(pl.col("ppg") >= RELEVANT_PPG))}
    return out


def run(scoring: str, out_dir: Path) -> dict:
    as_of = date.today()  # noqa: DTZ011
    snap = asof_snapshot(RawTables.from_cache(), as_of)
    tables = build_player_tables(snap, load_scoring(scoring))
    weeks = injury_weeks(InjuryRaw.from_cache(), snap.schedules, tables.player_weeks, as_of)
    inj = injury_seasons(weeks)
    ps = tables.player_seasons.filter(pl.col("position").is_in(POSITIONS))
    ina = inactive_seasons(ps, inj, snap.schedules, as_of)
    r = with_terms(rows(ps, tables.player_static, snap.schedules), ina, inj)
    cfg = load_value_config()
    f = pl.concat([forecast(r, t, cfg.draft_survival, cfg.draft_games) for t in (*DEV, *HOLDOUT)])
    summary = {"scoring": scoring, "dev": summarize(f.filter(pl.col("t").is_in(DEV))),
               "holdout": summarize(f.filter(pl.col("t").is_in(HOLDOUT)))}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--scoring", default="half_ppr")
    ap.add_argument("--out", default=str(REPO_ROOT / "reports" / "inactive"))
    a = ap.parse_args()
    run(a.scoring, Path(a.out))


if __name__ == "__main__":
    main()
