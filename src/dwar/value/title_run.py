"""Title value prototype run (plan approved by ben, 2026-11-20): one future
season on one league format, reusing a finished DWAR run's fits.

    uv run python -m dwar.value.title_run --as-of 2026-11-20 \\
        --format 1qb_3flex_half_ppr --leagues 2000

Writes `outputs/{as_of}/{format}/title/`: `title_{season}.parquet` (one row
per player: DWAR, single-season DWAR, title odds added with 50% and 90%
ranges, raw per-player swings, contender and rebuilder values) and
`title_{season}.json` (settings, timings and the checks in the plan).
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from dwar.config import load_format, load_model_config, load_scoring, load_value_config, uses_situation
from dwar.features.asof import RawTables, asof_snapshot
from dwar.features.injuries import InjuryRaw, injury_weeks
from dwar.features.opportunity import opportunity_table
from dwar.features.player_tables import build_player_tables
from dwar.features.situation import SituationRaw, model_covariates, prepare
from dwar.model.fit import load_fit, position_data
from dwar.model.talent import for_position
from dwar.pipeline import output_dir
from dwar.value.dwar import position_values, season_clock, value_inputs
from dwar.value.title import (
    LeagueInputs,
    SeasonFormat,
    rank_scores,
    simulate_league,
    title_curve,
)

log = logging.getLogger(__name__)
PERCENTILES = (5, 25, 50, 75, 95)
# Mean length of a missed-week stretch inside a season, 2013-2025 QB/RB/WR/TE
# with 4+ games (2.06 weeks; stretches running to season's end not counted).
OUT_RUN_WEEKS = 2.1


@dataclass
class Pool:
    """Every valued player, one future season, all draws."""

    player_id: list[str]
    name: list[str]
    position: np.ndarray        # (players,)
    team: list[str]
    age: np.ndarray
    dwar_mean: np.ndarray       # (players,) five-season DWAR, the dealing key
    dwar_season: np.ndarray     # (draws, players) wins added this season (undiscounted)
    talent: np.ndarray          # (draws, players)
    share: np.ndarray           # (draws, players) full-game share of the season, 0 if out of the league
    waiver: dict[str, float]
    params: dict[str, dict[str, np.ndarray]]   # position -> name -> (draws,)
    likelihood: dict[str, str]
    sigma: float                # DWAR's weekly team-score SD
    season: int                 # the season valued


def build_pool(as_of: date, format_name: str, season_ahead: int = 1) -> Pool:
    """Rebuild a finished run's value inputs from its saved fits and collect
    one season's talent and games (`season_ahead` seasons after the current
    one) with the same random draws as `dwar.parquet`."""
    fmt = load_format(format_name)
    model_cfg, cfg = load_model_config(), load_value_config()
    out = output_dir(as_of, format_name)
    raw = RawTables.from_cache()
    snap = asof_snapshot(raw, as_of)
    tables = build_player_tables(snap, load_scoring(fmt.scoring))
    cov = {}
    if uses_situation(model_cfg):
        current = season_clock(snap.schedules, as_of).current
        cov = model_covariates(
            prepare(SituationRaw.from_cache()), snap, tables, model_cfg.first_season,
            model_cfg.situation_level, model_cfg.situation_range, (current, current + 1),
        )
    opp = opportunity_table(tables, load_scoring(fmt.scoring)) if model_cfg.opportunity else None
    fits = {}
    for p in model_cfg.positions:
        r = load_fit(position_data(tables, p, model_cfg, cov, opp), out / "fits")
        if r is None:
            raise FileNotFoundError(f"no passing {p} fit in {out / 'fits'}: run the DWAR pipeline first")
        fits[p] = r
    weeks = injury_weeks(InjuryRaw.from_cache(), snap.schedules, tables.player_weeks, as_of) if cfg.injury_history else None
    # Same order of random draws as value/dwar.py dwar_tables.
    rng = np.random.default_rng(model_cfg.seed)
    inputs = value_inputs(fits, tables, snap.schedules, fmt, cfg, model_cfg.likelihoods(), rng,
                          model_cfg.min_plays, cov, weeks)
    rows, parts = [], []
    for p, r in fits.items():
        latest, total, tr = position_values(p, r, inputs, tables, fmt, cfg, rng, trace=True)
        if latest.is_empty():
            continue
        j = season_ahead   # trace seasons start at the current one
        disc = fmt.discount ** season_ahead
        full = tr.full[:, :, j] if p in cfg.full_games else 1.0
        share = np.clip(tr.games[:, :, j] * full / inputs.availability.season_games, 0, 1) * tr.alive[:, :, j]
        rows.append(latest.select("player_id", "name", "team", age="age_at_as_of", last_season="season")
                    .with_columns(position=pl.lit(p), dwar_mean=pl.Series(total.mean(0)),
                                  col=pl.int_range(pl.len()) + sum(x.shape[1] for x, *_ in parts)))
        parts.append((tr.talent[:, :, j], share, tr.wins[:, :, j] / disc))
    info = pl.concat(rows)
    # A player valued at two positions (a switch) keeps the latest, as dwar.parquet does.
    info = (info.sort(["last_season", "dwar_mean"], descending=True)
            .unique("player_id", keep="first", maintain_order=True).sort("col"))
    keep = info["col"].to_numpy()
    talent, share, wins = (np.concatenate([x[k] for x in parts], axis=1)[:, keep] for k in range(3))
    pdraws = inputs.draws
    return Pool(
        player_id=info["player_id"].to_list(), name=info["name"].to_list(),
        position=np.array(info["position"].to_list(), object), team=info["team"].to_list(),
        age=info["age"].to_numpy(), dwar_mean=info["dwar_mean"].to_numpy(), dwar_season=wins,
        talent=talent, share=share,
        waiver={p: float(np.mean(v)) for p, v in inputs.replacement.items()},
        params={p: {**{n: np.asarray(v) for n, v in d.params.items()}} | {"zero_value": d.zero_value}
                for p, d in pdraws.items()},
        likelihood={p: for_position(model_cfg.likelihoods(), p) for p in pdraws},
        sigma=float(inputs.sigma),
        season=inputs.clock.current + season_ahead,
    )


# --- parallel league runs ----------------------------------------------------

_POOL: Pool | None = None
_SETTINGS: dict = {}
_KNOWN: np.ndarray | None = None


def _league_inputs(pool: Pool, d: int, spread: dict[str, float] | None = None) -> LeagueInputs:
    """One draw's league inputs. `spread` multiplies a position's weekly
    sqrt-scale spread (sigma); centers are re-solved so expected points stay
    the same."""
    spread = spread or {}
    params = {p: {n: (float(v[d]) if np.ndim(v) else float(v)) for n, v in ps.items()} for p, ps in pool.params.items()}
    for p, k in spread.items():
        params[p]["sigma"] *= k
    return LeagueInputs(
        pos=pool.position, talent=pool.talent[d], share=pool.share[d], rank_z=rank_scores(pool.dwar_mean),
        waiver=pool.waiver,
        params=params,
        likelihood=pool.likelihood,
        # Managers set lineups by what they can know (posterior mean talent), not this draw's truth.
        known=_KNOWN if _KNOWN is not None else pool.talent.mean(0),
    )


def _run_chunk(args: tuple[int, int, int]) -> dict:
    start, n, seed = args
    s = _SETTINGS
    rng = np.random.default_rng([seed, start])
    D = _POOL.talent.shape[0]
    cols = {k: [] for k in ("league", "draw", "player", "team", "delta", "started", "lost", "strength_rank")}
    champ_rank, team_sd = [], []
    for li in range(start, start + n):
        d = li % D
        r = simulate_league(_league_inputs(_POOL, d, s["spread"]), s["lineup"], s["by_position"], s["teams"],
                            s["fmt"], s["noise"], OUT_RUN_WEEKS, rng)
        srank = np.argsort(np.argsort(-r.expected))   # preseason roster strength
        cols["league"].append(np.full(len(r.player), li))
        cols["draw"].append(np.full(len(r.player), d))
        cols["player"].append(r.player)
        cols["team"].append(r.team)
        cols["delta"].append(r.delta)
        cols["started"].append(r.started)
        cols["lost"].append(r.lost)
        cols["strength_rank"].append(srank[r.team])
        champ_rank.append(int(srank[r.champ]))
        team_sd.append(r.week_sd)
    return {k: np.concatenate(v) for k, v in cols.items()} | {"champ_rank": np.array(champ_rank),
                                                             "team_sd": np.array(team_sd)}


def run_leagues(pool: Pool, n_leagues: int, fmt_name: str, season: SeasonFormat, noise: float,
                seed: int, workers: int = 4, chunk: int = 50,
                spread: dict[str, float] | None = None) -> dict[str, np.ndarray]:
    global _POOL, _SETTINGS, _KNOWN
    lf = load_format(fmt_name)
    _POOL = pool
    _KNOWN = pool.talent.mean(0)
    _SETTINGS = dict(lineup=dict(lf.lineup), by_position=dict(lf.roster.by_position), teams=lf.teams,
                     fmt=season, noise=noise, spread=spread or {})
    jobs = [(s, min(chunk, n_leagues - s), seed) for s in range(0, n_leagues, chunk)]
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as p:
        parts = p.map(_run_chunk, jobs)
    return {k: np.concatenate([x[k] for x in parts]) for k in parts[0]}


# --- summary -----------------------------------------------------------------


def summarize(pool: Pool, res: dict[str, np.ndarray], teams: int) -> tuple[pl.DataFrame, dict]:
    """Per-player table: curve values (all draws), raw means, and the
    contender / rebuilder split (top and bottom third of rosters by
    simulated weekly score)."""
    D, n = pool.talent.shape
    T = pool.talent[res["draw"], res["player"]]
    S = pool.share[res["draw"], res["player"]]
    third = teams // 3
    groups = {"all": np.ones(len(T), bool), "contender": res["strength_rank"] < third,
              "rebuilder": res["strength_rank"] >= teams - third}
    out = {}
    curves = {}
    for g, m in groups.items():
        vals = np.zeros((D, n))
        for p in np.unique(pool.position):
            sel = m & (pool.position[res["player"]] == p)
            c = title_curve(T[sel], S[sel], res["delta"][sel])
            curves[(g, p)] = c
            cols = np.flatnonzero(pool.position == p)
            vals[:, cols] = c.at(pool.talent[:, cols], pool.share[:, cols])
        out[g] = vals * 100.0   # percentage points of title odds
    raw_sum = np.bincount(res["player"], weights=res["delta"], minlength=n)
    raw_n = np.bincount(res["player"], minlength=n)
    raw_mean = np.where(raw_n > 0, raw_sum / np.maximum(raw_n, 1), np.nan) * 100
    raw_var = np.bincount(res["player"], weights=res["delta"] ** 2, minlength=n) / np.maximum(raw_n, 1) * 1e4 - raw_mean**2
    raw_se = np.sqrt(np.maximum(raw_var, 0) / np.maximum(raw_n, 1))
    q = np.percentile(out["all"], PERCENTILES, axis=0)
    table = pl.DataFrame({
        "player_id": pool.player_id, "name": pool.name, "position": list(pool.position), "team": pool.team,
        "age": pool.age, "dwar_mean": pool.dwar_mean, "dwar_season": pool.dwar_season.mean(0),
        "title_mean": out["all"].mean(0), **{f"title_p{pc:02d}": q[k] for k, pc in enumerate(PERCENTILES)},
        "title_contender": out["contender"].mean(0), "title_rebuilder": out["rebuilder"].mean(0),
        "title_raw": raw_mean, "title_raw_se": raw_se, "leagues_rostered": raw_n,
        "talent_mean": pool.talent.mean(0), "share_mean": pool.share.mean(0),
    }).sort("title_mean", descending=True)
    return table, {"curves": {f"{g}/{p}": {"talent": c.talent.round(3).tolist(), "share": c.share.round(3).tolist(),
                                           "value_pct": (c.value * 100).round(3).tolist()}
                              for (g, p), c in curves.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", required=True, type=date.fromisoformat)
    ap.add_argument("--format", required=True)
    ap.add_argument("--leagues", type=int, default=2000)
    ap.add_argument("--season-ahead", type=int, default=1)
    ap.add_argument("--noise", type=float, default=0.5, help="dealing noise on DWAR rank normal scores")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--tag", default="")
    ap.add_argument("--spread", default="", help="weekly spread factors, e.g. RB=1.35,WR=1.2,TE=1.25")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    t0 = time.perf_counter()
    pool = build_pool(a.as_of, a.format, a.season_ahead)
    t1 = time.perf_counter()
    log.info("pool built: %d players in %.0fs", len(pool.player_id), t1 - t0)
    season = SeasonFormat()
    spread = {k: float(v) for k, v in (x.split("=") for x in a.spread.split(",") if x)}
    res = run_leagues(pool, a.leagues, a.format, season, a.noise, a.seed, a.workers, spread=spread)
    t2 = time.perf_counter()
    log.info("%d leagues in %.0fs", a.leagues, t2 - t1)
    lf = load_format(a.format)
    table, extra = summarize(pool, res, lf.teams)
    season_year = pool.season
    out = output_dir(a.as_of, a.format) / "title"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"title_{season_year}{a.tag}"
    table.write_parquet(out / f"{stem}.parquet")
    champ = np.bincount(res["champ_rank"], minlength=lf.teams) / len(res["champ_rank"])
    summary = {
        "as_of": a.as_of.isoformat(), "format": a.format, "season": season_year, "leagues": a.leagues,
        "noise": a.noise, "seed": a.seed, "spread": spread, "out_run_weeks": OUT_RUN_WEEKS,
        "season_format": {"regular_weeks": season.regular_weeks, "playoff_teams": season.playoff_teams,
                          "byes": season.byes},
        "waiver": pool.waiver,
        "runtime_s": {"pool": round(t1 - t0, 1), "leagues": round(t2 - t1, 1)},
        "check_team_week_sd": {"simulated": round(float(res["team_sd"].mean()), 2), "dwar_sigma": round(pool.sigma, 2)},
        "title_share_by_strength_rank": champ.round(4).tolist(),
        **extra,
    }
    (out / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    log.info("wrote %s", out / f"{stem}.parquet")


if __name__ == "__main__":
    main()
