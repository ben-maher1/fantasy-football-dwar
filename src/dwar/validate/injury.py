"""Out-of-sample check of the availability refinements: talent hinge,
injury history, and current injury status.

For each season t, models are trained on seasons before t only, then
forecast season t for every player who played in t - 1:

- games: for those who played in t, games played (beta-binomial log score,
  MAE), under each games-model variant;
- survival: does the player play in t (log loss), with and without injury terms;
- current status: from weeks 4, 8 and 12 of t, games played in the rest of
  t, with and without the current-status factor.

Games variants (all fit on the same rows): base (the step 5 model), hinge
(talent stops adding games above the 75th percentile), injury (injury
totals), hinge_injury (both: the model DWAR uses), hinge_injury_groups
(plus body-part terms). To keep this fast, last season's points per game
stands in for talent. Scores are also reported for fantasy-relevant players
(at least `RELEVANT_PPG` points per game last season).

Run: `uv run python -m dwar.validate.injury --scoring half_ppr`.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import betabinom

from dwar.config import REPO_ROOT, load_scoring
from dwar.features.asof import RawTables, asof_snapshot
from dwar.features.injuries import FIRST_SEASON, InjuryRaw, injury_seasons, injury_weeks
from dwar.features.player_tables import build_player_tables
from dwar.value.availability import (
    HINGE_QUANTILE, INJURY_GROUP_TERMS, INJURY_TOTALS, Logistic, _design, _sigmoid, availability_rows, fit_games,
    fit_logistic, injury_history_terms, season_games, status_effects, status_rows,
)

log = logging.getLogger(__name__)

POSITIONS = ("QB", "RB", "WR", "TE")
DEV = tuple(range(2013, 2023))
HOLDOUT = (2023, 2024, 2025)
CHECKPOINTS = (4, 8, 12)
RELEVANT_PPG = 8.0
# name -> (talent hinge, injury totals, body-part groups, trained on last season's talent)
GAMES_VARIANTS = {
    "base": (False, False, False, False), "hinge": (True, False, False, False),
    "injury": (False, True, False, False), "hinge_injury": (True, True, False, False),
    "hinge_injury_groups": (True, True, True, False), "hinge_injury_lagged": (True, True, False, True),
}
SURVIVAL_VARIANTS = ("base", "injury")
COEF_NAMES = ("const", "age", "age2", "talent", "talent_above_knot", *INJURY_TOTALS, *INJURY_GROUP_TERMS)


def _kappa(fit: Logistic, n: int) -> float:
    phi = fit.dispersion
    if phi <= 1.0 or n <= 1:
        return 1e6
    rho = min((phi - 1.0) / (n - 1.0), 0.99)
    return 1.0 / rho - 1.0


def _sums(prefix: str, rel: np.ndarray, values: dict[str, np.ndarray]) -> dict[str, float]:
    """Sums over all rows and over relevant rows (keys suffixed `_rel`)."""
    out: dict[str, float] = {}
    for sfx, m in (("", np.ones_like(rel, dtype=bool)), ("_rel", rel)):
        out |= {f"{k}{sfx}": float(v[m].sum()) for k, v in values.items()}
        out[f"n_{prefix}{sfx}"] = float(m.sum())
    return out


def _known(seasons_tbl: pl.DataFrame, inj: pl.DataFrame, t: int):
    known = seasons_tbl.filter(pl.col("season") < t)
    talent = known.select("player_id", "season", "position", "games_played", talent_mean="fpts_per_game")
    return known, talent, inj.filter(pl.col("season") < t)


def season_forecast(seasons_tbl: pl.DataFrame, inj: pl.DataFrame, schedules: pl.DataFrame, t: int) -> dict:
    """Per position: summed scores of season t's forecasts, per variant."""
    start = schedules.filter(pl.col("season") == t)["gameday"].min()
    known, talent, inj_known = _known(seasons_tbl, inj, t)
    games_rows, _ = availability_rows(talent, known, schedules, start - timedelta(days=1), inj_known, groups=True)
    lag_rows, _ = availability_rows(talent, known, schedules, start - timedelta(days=1), inj_known, groups=True,
                                    lagged=True)
    n_t = int(season_games(schedules).filter(pl.col("season") == t)["games"].item())

    prev = known.filter((pl.col("season") == t - 1) & (pl.col("games_played") > 0) & pl.col("age_sept1").is_not_null())
    prev = prev.join(seasons_tbl.filter(pl.col("season") == t).select("player_id", games_t="games_played"),
                     on="player_id", how="left")
    terms = injury_history_terms(prev.select("player_id", t=pl.lit(t)), inj_known, groups=True)
    n_tot = len(INJURY_TOTALS)

    # Survival training rows (s -> s + 1, s + 1 < t) with the same injury terms.
    nxt = known.filter(pl.col("games_played") > 0).select("player_id", pl.col("season") - 1, next_played=pl.lit(1))
    surv = (
        known.filter((pl.col("games_played") > 0) & (pl.col("season") + 1 < t) & (pl.col("season") + 1 >= FIRST_SEASON + 2)
                     & pl.col("age_sept1").is_not_null())
        .join(nxt, on=["player_id", "season"], how="left")
    )
    surv_terms = injury_history_terms(surv.select("player_id", t=pl.col("season") + 1), inj_known)

    out: dict[str, dict] = {}
    for p in POSITIONS:
        g = games_rows.filter(pl.col("position") == p)
        mask = (prev["position"] == p).to_numpy()
        if g.is_empty() or not mask.any():
            continue
        pr = prev.filter(pl.col("position") == p)
        age, tal = pr["age_sept1"].to_numpy(), pr["fpts_per_game"].to_numpy()
        y = pr["games_t"].fill_null(0).to_numpy()
        played, rel = y > 0, tal >= RELEVANT_PPG
        y_g = np.minimum(y[played], n_t)
        g_lag = lag_rows.filter(pl.col("position") == p)
        knots = {False: float(g["talent"].quantile(HINGE_QUANTILE)), True: float(g_lag["talent"].quantile(HINGE_QUANTILE))}
        res: dict = {"coef": {}, "knot": knots[False], "knot_lagged": knots[True]}
        for name, (hinge, use_inj, groups, lagged) in GAMES_VARIANTS.items():
            src = g_lag if lagged else g
            knot = knots[lagged]
            cols = slice(None) if groups else slice(0, n_tot)
            rows = src.with_columns(injury=pl.Series(np.array(src["injury"].to_list())[:, cols].tolist(),
                                                     dtype=pl.List(pl.Float64)))
            fit = fit_games(rows, knot if hinge else None, use_inj, groups)
            x = _design(age + 1.0, tal, terms[mask][:, cols] if use_inj else None, knot if hinge else None)[played]
            p_g = np.clip(_sigmoid(x @ fit.coef), 1e-6, 1 - 1e-6)
            k = _kappa(fit, n_t)
            res |= _sums(f"games_{name}", rel[played], {
                f"games_ls_{name}": betabinom.logpmf(y_g, n_t, p_g * k, (1 - p_g) * k),
                f"games_ae_{name}": np.abs(n_t * p_g - y_g),
                f"games_err_{name}": n_t * p_g - y_g,
            })
            res["coef"][name] = fit.coef.round(4).tolist()

        sm = (surv["position"] == p).to_numpy()
        s = surv.filter(pl.col("position") == p)
        if s.height:
            ys = s["next_played"].fill_null(0).to_numpy().astype(float)
            ll = {}
            for name in SURVIVAL_VARIANTS:
                inj_tr = surv_terms[sm] if name == "injury" else None
                fit = fit_logistic(_design(s["age_sept1"].to_numpy(), s["fpts_per_game"].to_numpy(), inj_tr), ys, np.ones_like(ys))
                ps = np.clip(_sigmoid(_design(age, tal, terms[mask][:, :n_tot] if name == "injury" else None) @ fit.coef),
                             1e-6, 1 - 1e-6)
                ll[f"surv_ll_{name}"] = -np.where(played, np.log(ps), np.log(1 - ps))
            res |= _sums("survival", rel, ll)
        out[p] = res
    return out


def status_forecast(
    weeks: pl.DataFrame, seasons_tbl: pl.DataFrame, inj: pl.DataFrame, schedules: pl.DataFrame, t: int,
) -> dict:
    """Rest-of-season games from weeks 4/8/12 of t, with and without the
    current-status factor (both on the hinge + injury games model trained
    on last season's talent)."""
    effects = status_effects(weeks, list(range(FIRST_SEASON, t)))
    rows = status_rows(weeks, [t]).filter(pl.col("week").is_in(CHECKPOINTS))
    start = schedules.filter(pl.col("season") == t)["gameday"].min()
    known, talent, inj_known = _known(seasons_tbl, inj, t)
    games_rows, _ = availability_rows(talent, known, schedules, start - timedelta(days=1), inj_known, lagged=True)
    prev = known.filter(pl.col("season") == t - 1).select("player_id", "position", "age_sept1", "fpts_per_game")
    rows = rows.join(prev, on="player_id", how="inner").filter(pl.col("age_sept1").is_not_null())
    terms = injury_history_terms(rows.select("player_id", t=pl.lit(t)), inj_known)
    out: dict = {"base": {"n": 0, "ae": 0.0, "ls": 0.0}, "status": {"n": 0, "ae": 0.0, "ls": 0.0}}
    for p in POSITIONS:
        g = games_rows.filter(pl.col("position") == p)
        mask = (rows["position"] == p).to_numpy()
        if g.is_empty() or not mask.any():
            continue
        knot = float(g["talent"].quantile(HINGE_QUANTILE))
        fit = fit_games(g, knot, True, False)
        r = rows.filter(pl.col("position") == p)
        share = _sigmoid(_design(r["age_sept1"].to_numpy() + 1.0, r["fpts_per_game"].to_numpy(), terms[mask], knot) @ fit.coef)
        mult = np.array([effects.get((b, int(rb)), 1.0) for b, rb in zip(r["bucket"].to_list(), r["rem_bucket"].to_list())])
        n, y = r["remaining"].to_numpy(), r["played_after"].to_numpy()
        k = np.array([_kappa(fit, int(m)) for m in n])
        for name, sh in (("base", share), ("status", share * mult)):
            sh = np.clip(sh, 1e-6, 1 - 1e-6)
            out[name]["n"] += int(len(y))
            out[name]["ae"] += float(np.sum(np.abs(n * sh - y)))
            out[name]["ls"] += float(np.sum(betabinom.logpmf(y, n, sh * k, (1 - sh) * k)))
    out["n_injured"] = rows.filter(pl.col("bucket") != "healthy").height
    return out


def _total(per_season: dict, seasons: tuple[int, ...], positions: tuple[str, ...], rel: bool) -> dict:
    acc: dict[str, float] = {}
    for t in seasons:
        for p in positions:
            for k, v in per_season[t].get(p, {}).items():
                if not isinstance(v, float) or k.endswith("_rel") != rel or k.startswith("knot"):
                    continue
                key = k.removesuffix("_rel")
                acc[key] = acc.get(key, 0.0) + v
    games = {}
    for v in GAMES_VARIANTS:
        n = max(acc.get(f"n_games_{v}", 0.0), 1.0)
        games[v] = {"logscore": acc.get(f"games_ls_{v}", 0.0) / n, "mae": acc.get(f"games_ae_{v}", 0.0) / n,
                    "bias": acc.get(f"games_err_{v}", 0.0) / n, "n": int(n)}
    n = max(acc.get("n_survival", 0.0), 1.0)
    survival = {v: {"logloss": acc.get(f"surv_ll_{v}", 0.0) / n, "n": int(n)} for v in SURVIVAL_VARIANTS}
    return {"games": games, "survival": survival}


def run(scoring: str, out_dir: Path) -> dict:
    as_of = date.today()
    raw = RawTables.from_cache()
    snap = asof_snapshot(raw, as_of)
    tables = build_player_tables(snap, load_scoring(scoring))
    weeks = injury_weeks(InjuryRaw.from_cache(), snap.schedules, tables.player_weeks, as_of)
    inj = injury_seasons(weeks)
    seasons_tbl = tables.player_seasons.filter(pl.col("position").is_in(POSITIONS))

    per_season, status = {}, {}
    for t in (*DEV, *HOLDOUT):
        log.info("season %d", t)
        per_season[t] = season_forecast(seasons_tbl, inj, snap.schedules, t)
        status[t] = status_forecast(weeks, seasons_tbl, inj, snap.schedules, t)

    summary: dict = {"scoring": scoring, "relevant_ppg": RELEVANT_PPG, "dev_seasons": list(DEV),
                     "holdout_seasons": list(HOLDOUT)}
    for label, seasons in (("dev", DEV), ("holdout", HOLDOUT)):
        summary[label] = {}
        for group, rel in (("all", False), ("relevant", True)):
            summary[label][group] = _total(per_season, seasons, POSITIONS, rel)
            for p in POSITIONS:
                summary[label][f"{p}_{group}"] = _total(per_season, seasons, (p,), rel)
        st = {k: {"n": 0, "ae": 0.0, "ls": 0.0} for k in ("base", "status")}
        for t in seasons:
            for k in st:
                for m in ("n", "ae", "ls"):
                    st[k][m] += status[t][k][m]
        summary[label]["status"] = {
            k: {"mae": v["ae"] / max(v["n"], 1), "logscore": v["ls"] / max(v["n"], 1), "n": v["n"]} for k, v in st.items()
        } | {"n_injured": sum(status[t]["n_injured"] for t in seasons)}
    last = max(per_season)
    summary["weights_latest"] = {
        p: {"knot": round(per_season[last][p]["knot"], 2)}
        | {v: dict(zip(COEF_NAMES, per_season[last][p]["coef"][v])) for v in ("hinge_injury", "hinge_injury_groups")}
        for p in POSITIONS if p in per_season[last]
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--scoring", default="half_ppr")
    ap.add_argument("--out", default=str(REPO_ROOT / "reports" / "injury"))
    a = ap.parse_args()
    run(a.scoring, Path(a.out))


if __name__ == "__main__":
    main()
