"""Game-by-game career test (plan "game-by-game career model and role
context", approved by ben 2026-11-05).

Each variant is fit once on WR seasons 2013-2019, then stepped through
2020-2022 one game at a time (`validate/stepthrough.py`), scored against
what happened through 2025. Variants are compared on the same forecasts.

    uv run python -m dwar.validate.career run base      # the current WR model
    uv run python -m dwar.validate.career run games     # + per-game talent steps
    uv run python -m dwar.validate.career compare games base
    uv run python -m dwar.validate.career run base holdout   # final 2023-2025 check
    uv run python -m dwar.validate.career compare form4 base holdout
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import xarray as xr

from dwar.config import data_dir, load_model_config, load_scoring
from dwar.features.asof import RawTables, asof_snapshot
from dwar.features.player_tables import build_player_tables
from dwar.model.data import build_model_data
from dwar.model.diagnostics import check_fit
from dwar.model.fit import position_data, sample
from dwar.validate.score import BOOTSTRAP
from dwar.validate.stepthrough import DISCOUNT, HORIZON, Settings, step_through

log = logging.getLogger(__name__)

POSITION = "WR"
FIRST_SEASON = 2013
FIT_AS_OF = date(2020, 8, 1)      # before the 2020 season: the fit sees 2013-2019
TEST_SEASONS = (2020, 2021, 2022)
# The final check, run once on the kept variant and base: fit on 2013-2022,
# step through 2023-2025. Outcomes stop at 2025, so there are no discounted
# totals here (they need three later seasons); per-season scores combine
# with the discount instead.
PERIODS = {
    "dev": (FIT_AS_OF, TEST_SEASONS),
    "holdout": (date(2023, 8, 1), (2023, 2024, 2025)),
}
OUTCOMES_AS_OF = date(2026, 3, 1)  # after the 2025 season
LAST_SEASON = 2025
N_SETTINGS = 100                   # posterior draws of the settings used by the filter
# Particles per draw. 40 was too few: with no rejuvenation the particles of a
# season's level collapse onto a few values and the filter under-reacts to
# new games (early_m400). Results before 2026-11-05 evening used 40 and are
# archived in experiments/career/m40.
PARTICLES = 400

VARIANTS: dict[str, dict] = {
    # Variants start from config/model.yaml, which now has the game-by-game
    # terms on, so earlier variants turn off what they lacked.
    "base": {"game_steps": (), "qb_drift": ()},  # growth and first-round step on
    "games": {"game_steps": (POSITION,)},        # + talent steps after every game
    # games was a bit worse: its lasting per-game steps made later seasons
    # too uncertain. form keeps lasting change at the offseason and makes
    # in-season movement a decaying form term.
    "form": {"game_steps": (POSITION,), "game_mode": "form"},
    # form's fit just missed the checks on sigma_form (R-hat 1.011, ESS 283): longer tuning.
    "form2": {"game_steps": (POSITION,), "game_mode": "form", "sampler": {"tune": 2500}},
    # form2 still mixed slowly on rho_form (ESS 274): form3 scales the weekly
    # shock by sqrt(1 - rho^2), so sigma_form is the long-run form SD.
    "form3": {"game_steps": (POSITION,), "game_mode": "form"},
    # form3 forecast the same as form but still had ESS 274 on rho_form with
    # 8000 draws: rho mixes slowly but steadily, so take more draws.
    "form4": {"game_steps": (POSITION,), "game_mode": "form", "sampler": {"draws": 3500}},
    # form4 forecast the same and was just over on rho_form's R-hat (1.011).
    "form5": {"game_steps": (POSITION,), "game_mode": "form", "sampler": {"draws": 5000}},
    # form5's 2023 holdout fit diverged: the form term's rho^-j scaling
    # overflowed at small rho. form6 = form with the stable decay-matrix
    # version (same model) at the usual draws.
    "form6": {"game_steps": (POSITION,), "game_mode": "form", "qb_drift": ()},
    # + offseason drift with the QB rating (ben, 2026-11-05: WRs with a
    # bottom-quarter QB were over-forecast ~0.9 PPG the next season).
    "qb": {"game_steps": (POSITION,), "game_mode": "form", "qb_drift": (POSITION,)},
    # qb forecast better than form6 but missed on rho_form (R-hat 1.018): more draws.
    "qb2": {"game_steps": (POSITION,), "game_mode": "form", "qb_drift": (POSITION,), "sampler": {"draws": 3500}},
    # qb2's 2023 holdout fit had one chain stuck at a low-rho mode: qb3 =
    # qb2 with rho_form's prior moved off that mode (Beta(8, 2)).
    "qb3": {"game_steps": (POSITION,), "game_mode": "form", "qb_drift": (POSITION,), "sampler": {"draws": 3500}},
    # The 2026 WR fit (2013-2026) had a chain in a second mode where fast
    # "form" replaced weekly noise; qb4 = qb3 with form's spread a fraction
    # of the weekly noise's (model/talent.py form_ratio).
    "qb4": {"game_steps": (POSITION,), "game_mode": "form", "qb_drift": (POSITION,), "sampler": {"draws": 3500},
            "reuse_fit": {"dev": "qbfade"}},
    # ben, 2026-11-05 (Jalen Coker): qb4 carries today's QB into every
    # projected season. A WR's QB rating keeps only ~0.26 of its distance
    # from average a season later (QBs and teams change): qbfade fades it
    # (features/qb_rating.QBFade). Same model as qb4 (holdout: reuses qb3's fit).
    "qbfade": {"game_steps": (POSITION,), "game_mode": "form", "qb_drift": (POSITION,), "sampler": {"draws": 3500},
               "fade_qb": True, "reuse_fit": {"holdout": "qb3"}},
    # qbfade tied qb4 (dev discounted MAE +0.003, log -0.001; holdout +0.001/0.000)
    # and is kept (config/value.yaml qb_fade): it matches how little a QB rating carries.
    # ... and a cubic WR age curve: the quadratic peaks at ~24. Tied too (dev
    # MAE -0.003, log -0.002); peak still ~25, gentler after 30: not kept.
    # ben, 2026-11-05: young WRs running hot beat their forecasts (rest of
    # season and next) by 1-2 PPG. tau_early lets talent move more between
    # seasons early in a career; exits leaves injury-exit games out of the fit.
    "early": {"game_steps": (POSITION,), "game_mode": "form", "qb_drift": (POSITION,), "sampler": {"draws": 3500},
              "fade_qb": True, "tau_early": (POSITION,)},
    "exits": {"game_steps": (POSITION,), "game_mode": "form", "qb_drift": (POSITION,), "sampler": {"draws": 3500},
              "fade_qb": True, "injury_exits": (POSITION,)},
    # Results (dev, vs qbfade): early tied (lam_tau -0.13 +- 0.26: young WRs'
    # talent does not move more between seasons); exits worse at every
    # horizon (discounted log -0.013). Neither kept.
    # early's step-through with 10x the particles: does the filter itself under-react?
    "early_m400": {"game_steps": (POSITION,), "game_mode": "form", "qb_drift": (POSITION,), "sampler": {"draws": 3500},
                   "fade_qb": True, "tau_early": (POSITION,), "particles": 400, "reuse_fit": {"dev": "early"}},
    # ben, 2026-11-06: with the fixed test, hot young WRs still beat their
    # forecasts (2023-25: +1.3 rest of season, +1.1 next). young adds lasting
    # in-season steps for young players only (games' steps for everyone made
    # later seasons too uncertain).
    "young": {"game_steps": (POSITION,), "game_mode": "form", "qb_drift": (POSITION,), "sampler": {"draws": 3500},
              "fade_qb": True, "young_steps": (POSITION,)},
    "cubic": {"game_steps": (POSITION,), "game_mode": "form", "qb_drift": (POSITION,), "sampler": {"draws": 3500},
              "fade_qb": True, "age_curve": {POSITION: "cubic"}},
}


def root(name: str, period: str = "dev") -> Path:
    base = data_dir() / "experiments" / "career"
    return base / name if period == "dev" else base / period / name


def _tables(as_of: date):
    return build_player_tables(asof_snapshot(RawTables.from_cache(), as_of), load_scoring("half_ppr"))


def run(name: str, period: str = "dev", seed: int = 0) -> Path:
    fit_as_of, test_seasons = PERIODS[period]
    base = load_model_config()
    v = dict(VARIANTS[name])
    fade = v.pop("fade_qb", False)
    reuse = v.pop("reuse_fit", {}).get(period)
    particles = v.pop("particles", PARTICLES)
    if "sampler" in v:
        v["sampler"] = base.sampler.model_copy(update=v["sampler"])
    cfg = base.model_copy(update={"first_season": FIRST_SEASON, "positions": (POSITION,)} | v)
    out = root(name, period)
    out.mkdir(parents=True, exist_ok=True)
    fit_tables = _tables(fit_as_of)
    d = position_data(fit_tables, POSITION, cfg)
    nc = out / "fit.nc"
    if reuse and not nc.exists():
        # Same model as `reuse`: only the step-through differs.
        nc.symlink_to(root(reuse, period) / "fit.nc")
    if nc.exists():
        posterior = xr.open_datatree(nc)
    else:
        start = time.perf_counter()
        posterior = sample(d, cfg)
        runtime = time.perf_counter() - start
        diag = check_fit(posterior, cfg.diagnostics)
        (out / "diagnostics.json").write_text(json.dumps(diag.to_dict() | {"runtime_s": round(runtime, 1)}, indent=2, default=str))
        posterior.to_netcdf(nc)
        log.info("%s fit: passed=%s runtime %.0fs %s", name, diag.passed, runtime, diag.failures)
    total = posterior["posterior"].sizes["chain"] * posterior["posterior"].sizes["draw"]
    draws = np.random.default_rng(seed).choice(total, N_SETTINGS, replace=False)
    settings = Settings.from_posterior(d, posterior, cfg.likelihood_for(POSITION), np.sort(draws), fade_qb=fade)
    exits = POSITION in cfg.injury_exits
    if exits:
        # Injury exits are left out of the fit; forecasts add them back at
        # the rate (and with the points) seen in the fit's own seasons.
        w = fit_tables.player_weeks.filter((pl.col("position") == POSITION) & (pl.col("season") >= cfg.first_season_by_position.get(POSITION, cfg.first_season)))
        hurt = w.filter(pl.col("injury_exit"))
        settings = replace(settings, exit_rate=hurt.height / w.height, exit_points=hurt["fpts"].to_numpy())
        log.info("injury exits: %.1f%% of games, %.2f points on average", 100 * settings.exit_rate, hurt["fpts"].mean())
    full = build_model_data(
        _tables(OUTCOMES_AS_OF), POSITION, FIRST_SEASON, min_plays=cfg.min_plays.get(POSITION, 0),
        unplayed_rookies=False, reference_players=d.player_ids, injury_exits="flag" if exits else "keep",
    )
    if full.prospect_x_names != settings.prospect_names:
        raise ValueError(f"prospect columns differ: {full.prospect_x_names} vs {settings.prospect_names}")
    start = time.perf_counter()
    rows = step_through(settings, full, test_seasons, LAST_SEASON, seed=cfg.seed, m=particles)
    log.info("%s step-through: %d rows in %.0fs", name, rows.height, time.perf_counter() - start)
    path = out / "forecasts.parquet"
    rows.write_parquet(path)
    return path


def _points(d, seasons: tuple[int, ...], last_season: int) -> pl.DataFrame:
    """Forecast points like `step_through`'s with plain features: draft slot,
    career games and PPG so far, PPG over the last 8 games, and each
    stretch's actual PPG (horizon -1 = discounted total points)."""
    from dwar.validate.stepthrough import players

    rows = []
    for p in players(d):
        hist: list[float] = []
        by = dict(zip(p.seasons, range(len(p.seasons))))
        for j, season in enumerate(p.seasons):
            g = p.games[j]
            for k in range(len(g) + 1):
                if season in seasons and k < len(g):
                    later = [g[k:]] + [p.games[by[season + h]] if season + h in by else None for h in range(1, HORIZON + 1)]
                    base = {"player_id": p.player_id, "season": season, "games_before": k,
                            "pick_z": p.pick_z, "undrafted": p.undrafted, "career_games": float(p.career_games[j][k]),
                            "ppg_sofar": float(np.mean(hist)) if hist else 0.0, "n_sofar": len(hist),
                            "ppg_recent": float(np.mean(hist[-8:])) if hist else 0.0}
                    for h, a in enumerate(later):
                        if a is not None and season + h <= last_season:
                            rows.append(base | {"horizon": h, "games": len(a), "actual": float(a.mean())})
                    if season + HORIZON <= last_season:
                        tot = sum(DISCOUNT**h * float(a.sum()) for h, a in enumerate(later) if a is not None)
                        rows.append(base | {"horizon": -1, "games": int(sum(len(a) for a in later if a is not None)),
                                            "actual": tot})
                if k < len(g):
                    hist.append(float(g[k]))
    return pl.DataFrame(rows)


def history_baseline() -> Path:
    """A plain regression baseline, fit on forecast points from 2013-2019
    (outcomes through 2019) and applied to the test points: per stretch and
    career stage, actual on [1, pick z, undrafted, PPG so far, last-8 PPG,
    log(1 + games so far)], with a normal predictive (residual SD by stretch
    and stage; for totals, also scaled by games)."""
    from scipy.stats import norm

    fit_tables = _tables(FIT_AS_OF)
    cfg = load_model_config().model_copy(update={"first_season": FIRST_SEASON})
    d_train = position_data(fit_tables, POSITION, cfg)
    full = build_model_data(_tables(OUTCOMES_AS_OF), POSITION, FIRST_SEASON, unplayed_rookies=False,
                            reference_players=d_train.player_ids)
    train = _points(full, tuple(range(FIRST_SEASON, 2020)), 2019)
    test = _points(full, TEST_SEASONS, LAST_SEASON)

    def design(df: pl.DataFrame) -> np.ndarray:
        return np.column_stack([np.ones(df.height), df["pick_z"], df["undrafted"], df["ppg_sofar"],
                                df["ppg_recent"], np.log1p(df["n_sofar"].to_numpy())])

    out = []
    for (h, st), te in test.with_columns(stage=_stage(pl.col("career_games"))).group_by("horizon", "stage"):
        tr = train.with_columns(stage=_stage(pl.col("career_games"))).filter((pl.col("horizon") == h) & (pl.col("stage") == st))
        scale_tr = tr["games"].to_numpy() if h == -1 else np.ones(tr.height)
        scale_te = te["games"].to_numpy() if h == -1 else np.ones(te.height)
        y = tr["actual"].to_numpy() / np.maximum(scale_tr, 1)
        b = np.linalg.lstsq(design(tr), y, rcond=None)[0]
        resid_sd = float(np.std(y - design(tr) @ b))
        # Residual SD per game falls with games for PPG stretches; use the
        # train residuals' fit on 1/sqrt(games) for a per-row SD.
        if h >= 0:
            gg = tr["games"].to_numpy().astype(float)
            r2 = (y - design(tr) @ b) ** 2
            c = np.linalg.lstsq(np.column_stack([np.ones_like(gg), 1 / gg]), r2, rcond=None)[0]
            sd = np.sqrt(np.maximum(c[0] + c[1] / te["games"].to_numpy(), 0.25))
        else:
            sd = np.full(te.height, resid_sd)
        mean = (design(te) @ b) * np.maximum(scale_te, 1)
        sd = sd * np.maximum(scale_te, 1)
        a = te["actual"].to_numpy()
        out.append(te.select("player_id", "season", "games_before", "horizon", "games", "actual", "career_games",
                             "pick_z", "undrafted").with_columns(
            pred_mean=pl.Series(mean), log_score=pl.Series(norm.logpdf(a, mean, sd)), pit=pl.Series(norm.cdf(a, mean, sd)),
        ))
    path = root("history")
    path.mkdir(parents=True, exist_ok=True)
    pl.concat(out).write_parquet(path / "forecasts.parquet")
    return path / "forecasts.parquet"


KEYS = ["player_id", "season", "games_before", "horizon"]
FLOOR_WEIGHT, FLOOR_PPG = 0.01, 40.0


def _floored(log_score: np.ndarray, games: np.ndarray, total: bool) -> np.ndarray:
    """1% floor spread over 0-40 PPG (over 0-40 points per game times games for totals)."""
    width = FLOOR_PPG * (games if total else 1.0)
    return np.log((1 - FLOOR_WEIGHT) * np.exp(log_score) + FLOOR_WEIGHT / width)


def _stage(career_games: pl.Expr) -> pl.Expr:
    return (pl.when(career_games == 0).then(pl.lit("0"))
            .when(career_games <= 3).then(pl.lit("1-3"))
            .when(career_games <= 8).then(pl.lit("4-8"))
            .when(career_games <= 17).then(pl.lit("9-17"))
            .otherwise(pl.lit("18+")))


def compare(a: str, b: str, period: str = "dev", seed: int = 0) -> dict:
    """Variant `a` minus `b` (positive = `a` better) on the same forecasts,
    per stretch and career stage, with bootstrap 90% intervals over players.
    `discounted` combines the per-season scores with DWAR's discount."""
    fa, fb = (pl.read_parquet(root(n, period) / "forecasts.parquet") for n in (a, b))
    j = fa.join(fb.select(*KEYS, "pred_mean", "log_score", "pit"), on=KEYS, suffix="_b").with_columns(
        stage=_stage(pl.col("career_games"))
    )
    total = (j["horizon"] == -1).to_numpy()
    g = j["games"].to_numpy().astype(float)
    j = j.with_columns(
        ls_a=pl.Series(np.where(total, _floored(j["log_score"].to_numpy(), g, True), _floored(j["log_score"].to_numpy(), g, False))),
        ls_b=pl.Series(np.where(total, _floored(j["log_score_b"].to_numpy(), g, True), _floored(j["log_score_b"].to_numpy(), g, False))),
        err_a=(pl.col("actual") - pl.col("pred_mean")).abs(),
        err_b=(pl.col("actual") - pl.col("pred_mean_b")).abs(),
    )
    players = j["player_id"].unique().sort().to_list()
    rng = np.random.default_rng(seed)
    boot = rng.integers(0, len(players), size=(BOOTSTRAP // 5, len(players)))

    def summary(g: pl.DataFrame) -> dict:
        # Per-player sums so the bootstrap resamples players (their forecasts are correlated).
        per = g.group_by("player_id").agg(
            n=pl.len(), w=pl.col("games").sum(),
            ea=(pl.col("err_a") * pl.col("games")).sum(), eb=(pl.col("err_b") * pl.col("games")).sum(),
            la=pl.col("ls_a").sum(), lb=pl.col("ls_b").sum(),
        )
        per = pl.DataFrame({"player_id": players}).join(per, on="player_id", how="left").fill_null(0)
        n, w, ea, eb, la, lb = (per[c].to_numpy().astype(float) for c in ("n", "w", "ea", "eb", "la", "lb"))
        mae = lambda e, i=slice(None): float(e[i].sum() / max(w[i].sum(), 1))
        ls = lambda l, i=slice(None): float(l[i].sum() / max(n[i].sum(), 1))
        mae_gain = np.array([mae(eb, i) - mae(ea, i) for i in boot])
        ls_gain = np.array([ls(la, i) - ls(lb, i) for i in boot])
        pit_a, pit_b = g["pit"].to_numpy(), g["pit_b"].to_numpy()
        return {
            "n": g.height, "mae": [mae(ea), mae(eb)], "mae_gain": mae(eb) - mae(ea),
            "mae_gain_90": np.percentile(mae_gain, [5, 95]).round(4).tolist(),
            "log_score": [ls(la), ls(lb)], "log_score_gain": ls(la) - ls(lb),
            "log_score_gain_90": np.percentile(ls_gain, [5, 95]).round(4).tolist(),
            "cover90": [float((np.abs(pit_a - 0.5) <= 0.45).mean()), float((np.abs(pit_b - 0.5) <= 0.45).mean())],
            "cover50": [float((np.abs(pit_a - 0.5) <= 0.25).mean()), float((np.abs(pit_b - 0.5) <= 0.25).mean())],
        }

    out = {"total": summary(j.filter(pl.col("horizon") == -1))} if total.any() else {}
    for h in range(HORIZON + 1):
        out[f"season+{h}"] = summary(j.filter(pl.col("horizon") == h))
    seasons = j.filter(pl.col("horizon") >= 0)
    for st in ("0", "1-3", "4-8", "9-17", "18+"):
        out[f"stage {st}"] = summary(seasons.filter(pl.col("stage") == st))
        if total.any():
            out[f"total, stage {st}"] = summary(j.filter((pl.col("horizon") == -1) & (pl.col("stage") == st)))
    wts = np.array([DISCOUNT**h for h in range(HORIZON + 1)])
    out["discounted"] = {
        k: float(sum(wt * out[f"season+{h}"][k] for h, wt in enumerate(wts)) / wts.sum())
        for k in ("mae_gain", "log_score_gain")
    }
    return out


def table(result: dict, a: str, b: str) -> str:
    lines = [f"{a} vs {b} (positive = {a} better)",
             f"{'group':18} {'n':>6} {'MAE gain':>24} {'log score gain':>24} {'90% cover':>11} {'50% cover':>11}"]
    for k, r in result.items():
        if k == "discounted":
            continue
        lines.append(
            f"{k:18} {r['n']:6d} {r['mae_gain']:+.3f} ({r['mae_gain_90'][0]:+.3f},{r['mae_gain_90'][1]:+.3f}) "
            f"{r['log_score_gain']:+.3f} ({r['log_score_gain_90'][0]:+.3f},{r['log_score_gain_90'][1]:+.3f}) "
            f"  {r['cover90'][0]:.2f}/{r['cover90'][1]:.2f}   {r['cover50'][0]:.2f}/{r['cover50'][1]:.2f}"
        )
    dsc = result["discounted"]
    lines.append(f"discounted per-season: MAE gain {dsc['mae_gain']:+.3f}, log score gain {dsc['log_score_gain']:+.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cmd = sys.argv[1]
    period = sys.argv[4] if cmd == "compare" and len(sys.argv) > 4 else (sys.argv[3] if cmd == "run" and len(sys.argv) > 3 else "dev")
    if cmd == "run":
        print(history_baseline() if sys.argv[2] == "history" else run(sys.argv[2], period))
    elif cmd == "compare":
        res = compare(sys.argv[2], sys.argv[3], period)
        print(table(res, sys.argv[2], sys.argv[3]))
        (root("", period) / f"compare_{sys.argv[2]}_vs_{sys.argv[3]}.json").write_text(json.dumps(res, indent=2))
