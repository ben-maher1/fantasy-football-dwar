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
Scored on survival, games played and full games (the value layer's
full-games model, same terms, among players who played), 2013-2022 (dev) and 2023-2025
(holdout), for all QBs and for backups (under 15% full games) who score
12+ per full game.

Also (`rest_of_season`): rest-of-season full games after week 4, 9 and 13,
from last season's share or this season's so far (trained 2010-19, scored
2020-25), which decides what the value layer feeds the games model for the
rest of the current season.

And (`horizons`): full games 2 and 3 seasons out from the one-step
full-games model, with the share input chained (last projected share),
frozen (last observed), capped (chained, at most the last observed) or
capped at the last observed share of games he was not hurt for.

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
    INJURY_WEEKS_UNIT,
    _sigmoid,
    fit_logistic,
    inactive_seasons,
    season_games,
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
    knot = float(gtr["ppg_full"].quantile(HINGE_QUANTILE))
    cols = {}
    for v in VARIANTS:
        ys = (train["g_next"] > 0).to_numpy().astype(float)
        fs = fit_logistic(_x(train, draft_survival, v, False, None), ys, np.ones_like(ys))
        cols[f"ps_{v}"] = _sigmoid(_x(test, draft_survival, v, False, None) @ fs.coef)
        n = gtr["n_next"].to_numpy()
        fg = fit_logistic(_x(gtr, draft_games, v, True, knot), np.minimum(gtr["g_next"].to_numpy(), n), n)
        cols[f"pg_{v}"] = _sigmoid(_x(test, draft_games, v, True, knot) @ fg.coef)
        ff = fit_logistic(_x(gtr, draft_games, v, True, knot), np.minimum(gtr["full_next"].to_numpy(), n), n)
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
    fg = np.minimum(pl_["full_next"].to_numpy(), n)
    for v in VARIANTS:
        ps = np.clip(f[f"ps_{v}"].to_numpy(), 1e-6, 1 - 1e-6)
        gl, gm = _binom(g, n, pl_[f"pg_{v}"].to_numpy())
        fl, fm = _binom(fg, n, pl_[f"pf_{v}"].to_numpy())
        res[v] = {
            "surv_logloss": round(float(-np.mean(np.where(y, np.log(ps), np.log(1 - ps)))), 5),
            "games_logloss": gl, "games_mae": gm,
            "games_pred": round(float(np.mean(n * pl_[f"pg_{v}"].to_numpy())), 3) if len(g) else None,
            "full_logloss": fl, "full_mae": fm, "full_pred": round(float(np.mean(n * pl_[f"pf_{v}"].to_numpy())), 3) if len(g) else None,
        }
    res["surv_actual"] = round(float(y.mean()), 4)
    res["games_actual"] = round(float(g.mean()), 3) if len(g) else None
    res["full_actual"] = round(float(fg.mean()), 3) if len(g) else None
    return res


def summarize(f: pl.DataFrame) -> dict:
    backup = f.filter((pl.col("fshare") < BACKUP_SHARE) & (pl.col("ppg_full") >= BACKUP_PPG))
    return {"all": score(f), "backup_12ppg": score(backup), "starter_50pct": score(f.filter(pl.col("fshare") >= 0.5))}


def rest_of_season(player_weeks: pl.DataFrame, schedules: pl.DataFrame, min_plays: int) -> list[dict]:
    pw = player_weeks.filter((pl.col("position") == POSITION) & pl.col("season").is_between(2009, 2025)).select(
        "player_id", pl.col("season").cast(pl.Int64), "week", "fpts", "plays", full=(pl.col("plays") >= min_plays).cast(pl.Int64))
    sg = season_games(schedules).select(pl.col("season").cast(pl.Int64), "games")
    last = (pw.filter(pl.col("week") <= 18).group_by("player_id", "season")
            .agg(fl=pl.col("full").sum(), pf=pl.col("fpts").filter(pl.col("plays") >= min_plays).mean())
            .join(sg, on="season").select("player_id", season=pl.col("season") + 1, last=pl.col("fl") / pl.col("games"),
                                          ppg_last=pl.col("pf")))
    out = []
    for cut in (4, 9, 13):
        so = pw.filter(pl.col("week") <= cut).group_by("player_id", "season").agg(so=pl.col("full").sum())
        rest = pw.filter((pl.col("week") > cut) & (pl.col("week") <= 18)).group_by("player_id", "season").agg(rest=pl.col("full").sum())
        d = (so.join(last, on=["player_id", "season"], how="left").join(rest, on=["player_id", "season"], how="left")
             .join(sg, on="season").filter(pl.col("season") >= 2010)
             .with_columns(last=pl.col("last").fill_null(0.0), rest=pl.col("rest").fill_null(0), n=pl.col("games") - cut,
                           sof=pl.col("so") / cut, ppg_last=pl.col("ppg_last").fill_null(10.0).fill_nan(10.0))
             .with_columns(rest=pl.min_horizontal("rest", "n")))
        tr, te = d.filter(pl.col("season") <= 2019), d.filter(pl.col("season") >= 2020)
        for name, cols in (("last", ["last"]), ("so_far", ["sof"]), ("both", ["last", "sof"])):
            def x(r: pl.DataFrame, cols: list[str] = cols) -> np.ndarray:
                return np.column_stack([np.ones(r.height), *[r[c].to_numpy() for c in cols], r["ppg_last"].to_numpy() / 10])
            f = fit_logistic(x(tr), tr["rest"].to_numpy(), tr["n"].to_numpy())
            p = _sigmoid(x(te) @ f.coef)
            ll, mae = _binom(te["rest"].to_numpy(), te["n"].to_numpy(), p)
            bk = te.with_columns(p=pl.Series(p)).filter((pl.col("so") == 0) & (pl.col("last") >= 0.4))
            out.append({"after_week": cut, "input": name, "logloss": ll, "mae": mae, "backups_n": bk.height,
                        "backups_actual": round(float(bk["rest"].mean()), 2),
                        "backups_pred": round(float((bk["n"] * bk["p"]).mean()), 2)})
    return out


def horizons(r: pl.DataFrame) -> dict:
    fut = r.select("player_id", "season", "full_next")
    r = (r.join(fut.select("player_id", season=pl.col("season") - 1, full2="full_next"), on=["player_id", "season"], how="left")
         .join(fut.select("player_id", season=pl.col("season") - 2, full3="full_next"), on=["player_id", "season"], how="left")
         .with_columns(pl.col("full2", "full3").fill_null(0)))
    modes = ("chain", "frozen", "capped", "capped_healthy")
    out = []
    for t in range(2013, 2024):
        tr = r.filter((pl.col("season") + 1 < t) & (pl.col("season") + 1 >= GAMES_FIRST))
        te = r.filter(pl.col("season") == t - 1)
        knot = float(tr["ppg_full"].quantile(HINGE_QUANTILE))
        n = tr["n_next"].to_numpy()
        f = fit_logistic(_x(tr, True, "starter", True, knot), np.minimum(tr["full_next"].to_numpy(), n), n)
        p1 = _sigmoid(_x(te, True, "starter", True, knot) @ f.coef)
        sh = te["fshare"].to_numpy()
        # Share of the games he was not hurt for (injury weeks from the injury terms).
        hurt = np.array(te["inj"].to_list())[:, 0] * INJURY_WEEKS_UNIT
        g_season = te["n_next"].to_numpy().astype(float)  # team games (next season stands in)
        healthy = np.minimum(sh * g_season / np.maximum(g_season - hurt, 1.0), 1.0)
        res = {"p1": p1}
        for m in modes:
            x, prev = te, p1
            for k in (2, 3):
                share_in = {"chain": prev, "frozen": sh, "capped": np.minimum(prev, sh),
                            "capped_healthy": np.minimum(prev, healthy)}[m]
                x = x.with_columns(age=pl.col("age") + 1, cy=pl.col("cy") + 1, fshare=pl.Series(share_in))
                prev = res[f"{m}{k}"] = _sigmoid(_x(x, True, "starter", True, knot) @ f.coef)
        out.append(te.with_columns(**{k: pl.Series(v) for k, v in res.items()}))
    f = pl.concat(out)
    n = float(r["n_next"].max())
    groups = {"all": f, "backup_14ppg": f.filter((pl.col("fshare") < BACKUP_SHARE) & (pl.col("ppg_full") >= 14)),
              "mid_15_50pct": f.filter(pl.col("fshare").is_between(BACKUP_SHARE, 0.5)),
              "starter_50pct": f.filter(pl.col("fshare") >= 0.5)}
    summary = {}
    for g, d in groups.items():
        row: dict = {"n": d.height}
        for k in (2, 3):
            a = d[f"full{k}"].to_numpy()
            row[f"actual_{k}"] = round(float(a.mean()), 2)
            for m in modes:
                p = d[f"{m}{k}"].to_numpy() * n
                row[f"{m}_{k}"] = {"pred": round(float(p.mean()), 2), "mae": round(float(np.mean(np.abs(p - a))), 2)}
        summary[g] = row
    return summary


def run(scoring: str, out_dir: Path) -> dict:
    as_of = date.today()  # noqa: DTZ011
    min_plays = load_model_config().min_plays.get(POSITION, 15)
    r = qb_rows(as_of, scoring, min_plays)
    cfg = load_value_config()
    f = pl.concat([forecast(r, t, POSITION in cfg.draft_survival, POSITION in cfg.draft_games) for t in (*DEV, *HOLDOUT)])
    snap = asof_snapshot(RawTables.from_cache(), as_of)
    tables = build_player_tables(snap, load_scoring(scoring))
    summary = {"scoring": scoring, "min_plays": min_plays, "dev": summarize(f.filter(pl.col("t").is_in(DEV))),
               "holdout": summarize(f.filter(pl.col("t").is_in(HOLDOUT))),
               "rest_of_season": rest_of_season(tables.player_weeks, snap.schedules, min_plays),
               "horizons": horizons(r)}
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
