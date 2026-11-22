"""Title odds for a real Sleeper league, season by season (ben, 2026-11-22;
Claude Doc "Plan: team-specific values from Sleeper").

Every team's real roster plays out the rest of this season against the
other teams on the real schedule, with wins and points already banked
counting toward seeding; the playoffs follow the league's own settings.
Then each later season (to the board's horizon) is played the same way:
each player's talent, games and survival for that season come from the
same projection draws that make his DWAR, every rookie pick a team owns
becomes a rookie of that year's class (its slot set by the team's finish
the season before, worst team first), and nothing else changes (no trades
or waiver pickups: a player who leaves is replaced by the waiver level).

Lineups are set by what managers can know (posterior mean talent), each
player plays his own projected share of weeks, taxi players cannot start
this season, and there is no waiver QB (an empty QB slot scores 0).

CLI: python -m dwar.value.league_odds --as-of 2026-11-20 --format 1qb_3flex_half_ppr --league <id>
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import time
from dataclasses import dataclass, replace
from datetime import date

import numpy as np
import polars as pl

from dwar.config import (
    load_format,
    load_model_config,
    load_scoring,
    load_value_config,
    uses_situation,
)
from dwar.features.asof import RawTables, asof_snapshot
from dwar.features.injuries import InjuryRaw, injury_weeks
from dwar.features.opportunity import opportunity_table
from dwar.features.player_tables import build_player_tables
from dwar.features.situation import SituationRaw, model_covariates, prepare
from dwar.ingest import sleeper as S
from dwar.model.fit import load_fit, position_data
from dwar.model.talent import for_position
from dwar.pipeline import output_dir
from dwar.value.dwar import position_values, season_clock, value_inputs
from dwar.value.title import (
    LeagueInputs,
    SeasonFormat,
    playoffs,
    round_robin,
    seeds,
    team_scores,
)

log = logging.getLogger(__name__)
OUT_RUN_WEEKS = 2.1
SPREAD = {"RB": 1.35, "WR": 1.2, "TE": 1.25}    # weekly spread calibration (reports/title_value)
NO_QB_WAIVER = {"QB": 0, "RB": 5, "WR": 5, "TE": 4}


@dataclass
class Context:
    """Everything a simulated league path needs, all draws."""

    season0: int                 # this season
    player_id: list[str]
    position: np.ndarray         # (P,)
    talent: np.ndarray           # (D, P, J) expected points per game, season j = season0 + j
    share: np.ndarray            # (D, P, J) share of fantasy weeks played (this season: of weeks left)
    rookies: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]  # year -> rank, pos, talent, share
    waiver: dict[str, float]
    params: dict[str, dict[str, np.ndarray]]
    likelihood: dict[str, str]


def build_context(as_of: date, format_name: str) -> Context:
    """Rebuild the board's inputs from saved fits (same draws as dwar.parquet)
    and keep every player's per-season talent and games, plus each future
    rookie class's sample rookies."""
    from dwar.value.picks import pick_values

    fmt = load_format(format_name)
    model_cfg, cfg = load_model_config(), load_value_config()
    out = output_dir(as_of, format_name)
    snap = asof_snapshot(RawTables.from_cache(), as_of)
    tables = build_player_tables(snap, load_scoring(fmt.scoring))
    cov = {}
    if uses_situation(model_cfg):
        current = season_clock(snap.schedules, as_of).current
        cov = model_covariates(prepare(SituationRaw.from_cache()), snap, tables, model_cfg.first_season,
                               model_cfg.situation_level, model_cfg.situation_range, (current, current + 1))
    opp = opportunity_table(tables, load_scoring(fmt.scoring)) if model_cfg.opportunity else None
    fits = {p: load_fit(position_data(tables, p, model_cfg, cov, opp), out / "fits") for p in model_cfg.positions}
    if any(r is None for r in fits.values()):
        raise FileNotFoundError(f"missing passing fits in {out / 'fits'}: run the DWAR pipeline first")
    weeks = injury_weeks(InjuryRaw.from_cache(), snap.schedules, tables.player_weeks, as_of) if cfg.injury_history else None
    rng = np.random.default_rng(model_cfg.seed)
    inputs = value_inputs(fits, tables, snap.schedules, fmt, cfg, model_cfg.likelihoods(), rng,
                          model_cfg.min_plays, cov, weeks)
    G = inputs.availability.season_games
    ids, pos, tal, sh = [], [], [], []
    for p, r in fits.items():
        latest, _, tr = position_values(p, r, inputs, tables, fmt, cfg, rng, trace=True)
        if latest.is_empty():
            continue
        full = tr.full if p in cfg.full_games else 1.0
        left = np.array([inputs.clock.remaining.get(t, inputs.clock.remaining_default) for t in latest["team"]], float)
        denom = np.full(tr.games.shape, float(G))
        denom[:, :, 0] = np.maximum(left, 1.0)[None, :]
        ids += latest["player_id"].to_list()
        pos += [p] * latest.height
        tal.append(tr.talent)
        sh.append(np.clip(tr.games * full / denom, 0, 1) * tr.alive)
    D = min(t.shape[0] for t in tal)
    talent = np.concatenate([t[:D] for t in tal], 1)
    share = np.concatenate([s[:D] for s in sh], 1)
    # A player fitted at two positions keeps the first (latest-season fits come first per position order).
    _, first = np.unique(np.array(ids), return_index=True)
    keep = np.sort(first)
    traces: dict = {}
    pick_values(fits, tables, snap.schedules, inputs, fmt, cfg, dict(model_cfg.min_plays or {}),
                np.random.default_rng([model_cfg.seed, 7]), traces=traces)
    rookies = {}
    for year, (rank, rpos, tr) in traces.items():
        full = np.where(rpos[None, :, None] == "QB", tr.full, 1.0)
        rookies[year] = (rank, rpos, tr.talent[:D], (np.clip(tr.games * full / G, 0, 1) * tr.alive)[:D])
    return Context(
        season0=inputs.clock.current, player_id=[ids[i] for i in keep], position=np.array(pos, object)[keep],
        talent=talent[:, keep], share=share[:, keep], rookies=rookies,
        waiver={p: float(np.mean(v)) for p, v in inputs.replacement.items()},
        params={p: {**{n: np.asarray(v) for n, v in d.params.items()}} | {"zero_value": d.zero_value}
                for p, d in inputs.draws.items()},
        likelihood={p: for_position(model_cfg.likelihoods(), p) for p in inputs.draws},
    )


@dataclass
class LeagueSetup:
    """The real league: rosters, banked results, schedule, picks."""

    teams: int
    owners: list[str]                 # by team index (roster_id - 1)
    roster: np.ndarray                # (P,) team index of each context player, -1 if not rostered
    taxi: np.ndarray                  # (P,) bool: on a taxi squad (cannot start this season)
    wins: np.ndarray                  # (teams,) banked (ties count half)
    points: np.ndarray                # (teams,) banked points for
    schedule_left: np.ndarray         # (weeks left, teams) opponent index
    fmt: SeasonFormat                 # regular weeks, playoff teams, byes
    picks: list[tuple[int, int, int, int]]   # (year, round, original team, owner team)
    rounds: int
    lineup: dict[str, int]


def league_setup(lg: S.League, ctx: Context, xwalk: pl.DataFrame, lineup: dict[str, int]) -> LeagueSetup:
    st = S.standings(lg)
    teams = st.height
    tid = {int(r): i for i, r in enumerate(st["roster_id"].to_list())}
    ros = S.rostered(lg).join(xwalk.select("sleeper_id", "gsis_id"), on="sleeper_id", how="left")
    where = {g: i for i, g in enumerate(ctx.player_id)}
    roster = np.full(len(ctx.player_id), -1)
    taxi = np.zeros(len(ctx.player_id), bool)
    for r in ros.iter_rows(named=True):
        i = where.get(r["gsis_id"])
        if i is not None:
            roster[i] = tid[r["roster_id"]]
            taxi[i] = r["slot"] == "taxi"
    s = lg.league["settings"]
    reg = int(s.get("playoff_week_start") or 15) - 1
    last = int(s.get("last_scored_leg") or 0)
    sched = np.zeros((reg - last, teams), int)
    for w in range(last + 1, reg + 1):
        games: dict[int, list[int]] = {}
        for g in lg.matchups.get(w, []):
            games.setdefault(int(g["matchup_id"]), []).append(tid[int(g["roster_id"])])
        if len(games) * 2 != teams or any(len(v) != 2 for v in games.values()):
            raise ValueError(f"week {w}: no full Sleeper schedule saved (pull again)")
        for a, b in games.values():
            sched[w - last - 1, a], sched[w - last - 1, b] = b, a
    n_playoff = int(s.get("playoff_teams") or 6)
    fmt = SeasonFormat(regular_weeks=reg, playoff_teams=n_playoff, byes=2 if n_playoff == 6 else 0)
    po = S.pick_owners(lg, seasons=[y for y in ctx.rookies if y > lg.season][:3])
    picks = [(y, k, tid[o], tid[n]) for y, k, o, n in po.iter_rows()]
    # Picks past Sleeper's three tradable years: every team keeps its own.
    for y in sorted(ctx.rookies):
        if y not in {p[0] for p in picks}:
            picks += [(y, k, t, t) for k in range(1, int(s.get("draft_rounds") or 5) + 1) for t in range(teams)]
    return LeagueSetup(teams, st["owner"].to_list(), roster, taxi,
                       (st["wins"] + 0.5 * st["ties"]).to_numpy().astype(float), st["points_for"].to_numpy(),
                       sched, fmt, picks, int(s.get("draft_rounds") or 5), lineup)


def _inputs(ctx: Context, d: int, j: int, extra: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
            known: np.ndarray) -> LeagueInputs:
    params = {p: {n: (float(v[d % len(v)]) if np.ndim(v) else float(v)) for n, v in ps.items()}
              for p, ps in ctx.params.items()}
    for p, k in SPREAD.items():
        if p in params:
            params[p]["sigma"] *= k
    pos, tal, sh = ctx.position, ctx.talent[d, :, j], ctx.share[d, :, j]
    kn = known[:, j]
    if extra is not None:
        pos, tal, sh = np.concatenate([pos, extra[0]]), np.concatenate([tal, extra[1]]), np.concatenate([sh, extra[2]])
        kn = np.concatenate([kn, extra[1]])
    return LeagueInputs(pos=pos, talent=tal, share=sh, rank_z=np.zeros(len(pos)), waiver=ctx.waiver,
                        params=params, likelihood=ctx.likelihood, known=kn)


def simulate_path(ctx: Context, lg: LeagueSetup, d: int, seasons: int, rng: np.random.Generator,
                  known: np.ndarray) -> np.ndarray:
    """One league path from this season on: (seasons, teams) final places
    (0 = champion)."""
    T = lg.teams
    places = np.zeros((seasons, T), int)
    # This season: the weeks left, then the playoffs.
    left = lg.schedule_left.shape[0]
    pw = lg.fmt.playoff_weeks
    li = _inputs(ctx, d, 0, None, known)
    li = replace(li, share=np.where(lg.taxi, 0.0, li.share))
    sc = team_scores(li, lg.roster, T, lg.lineup, left + pw, OUT_RUN_WEEKS, rng, NO_QB_WAIVER)
    reg = sc[:, :left]
    opp = np.take_along_axis(reg, lg.schedule_left.T, 0)
    wins = lg.wins + (reg > opp).sum(1) + 0.5 * (reg == opp).sum(1)
    pf = lg.points + reg.sum(1)
    order = np.argsort(-(wins * 1e6 + pf), kind="stable")
    fin = playoffs(order, sc[:, left:], lg.fmt.playoff_teams, lg.fmt.byes)
    places[0, fin] = np.arange(T)
    # Later seasons: rookies join from the picks each team owns.
    added_pos, added_tal, added_sh, added_team = [], [], [], []
    for j in range(1, seasons):
        year = ctx.season0 + j
        if year in ctx.rookies:
            rank, rpos, rtal, rsh = ctx.rookies[year]
            slot_of = {t: int(np.flatnonzero(fin[::-1] == t)[0]) + 1 for t in range(T)}   # worst finish picks first
            for (y, k, orig, owner) in lg.picks:
                if y != year:
                    continue
                cand = np.flatnonzero(rank == (k - 1) * T + slot_of[orig])
                if len(cand):
                    c = rng.choice(cand)
                    added_pos.append(rpos[c])
                    added_tal.append(rtal[d % rtal.shape[0], c])
                    added_sh.append(rsh[d % rsh.shape[0], c])
                    added_team.append(owner)
        extra = None
        if added_pos:
            extra = (np.array(added_pos, object), np.array([t[j] for t in added_tal]), np.array([s[j] for s in added_sh]))
        li = _inputs(ctx, d, j, extra, known)
        team = np.concatenate([lg.roster, np.array(added_team, int)]) if added_pos else lg.roster
        fmt = lg.fmt
        sc = team_scores(li, team, T, lg.lineup, fmt.weeks, OUT_RUN_WEEKS, rng, NO_QB_WAIVER)
        sched = round_robin(T, fmt.weeks, rng)
        order = seeds(sc, sched, fmt.regular_weeks)
        fin = playoffs(order, sc[:, fmt.regular_weeks:], fmt.playoff_teams, fmt.byes)
        places[j, fin] = np.arange(T)
    return places


_CTX: Context | None = None
_LG: LeagueSetup | None = None
_KNOWN: np.ndarray | None = None


def _chunk(args: tuple[int, int, int, int]) -> np.ndarray:
    start, n, seasons, seed = args
    rng = np.random.default_rng([seed, start])
    D = _CTX.talent.shape[0]
    return np.stack([simulate_path(_CTX, _LG, i % D, seasons, rng, _KNOWN) for i in range(start, start + n)])


def run(ctx: Context, lg: LeagueSetup, n: int, seasons: int, seed: int = 2026, workers: int = 4,
        chunk: int = 100) -> np.ndarray:
    """(n, seasons, teams) final places."""
    global _CTX, _LG, _KNOWN
    _CTX, _LG, _KNOWN = ctx, lg, ctx.talent.mean(0)
    jobs = [(s, min(chunk, n - s), seasons, seed) for s in range(0, n, chunk)]
    with mp.get_context("fork").Pool(workers) as p:
        parts = p.map(_chunk, jobs)
    return np.concatenate(parts)


def summarize(places: np.ndarray, lg: LeagueSetup, season0: int) -> pl.DataFrame:
    n, J, T = places.shape
    rows = []
    for j in range(J):
        for t in range(T):
            p = places[:, j, t]
            title = float((p == 0).mean())
            rows.append({"season": season0 + j, "team": t, "owner": lg.owners[t], "title": title,
                         "title_se": float(np.sqrt(title * (1 - title) / n)),
                         "final": float((p <= 1).mean()), "playoffs": float((p < lg.fmt.playoff_teams).mean()),
                         "mean_place": float(p.mean() + 1)})
    return pl.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", required=True, type=date.fromisoformat)
    ap.add_argument("--format", required=True)
    ap.add_argument("--league", required=True)
    ap.add_argument("--leagues", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--pull", action="store_true", help="fetch a fresh Sleeper snapshot first")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    t0 = time.perf_counter()
    if a.pull:
        S.pull(a.league)
    sl = S.load(a.league)
    ctx = build_context(a.as_of, a.format)
    t1 = time.perf_counter()
    log.info("context: %d players, %d draws in %.0fs", len(ctx.player_id), ctx.talent.shape[0], t1 - t0)
    from dwar.config import data_dir
    from dwar.ingest.nflverse import load_raw
    xw = S.crosswalk(sl.players, load_raw("rosters").select("gsis_id", pl.col("sleeper_id").cast(pl.Utf8)),
                     load_raw("players"), S.rostered(sl)["sleeper_id"].to_list())
    lg = league_setup(sl, ctx, xw, dict(load_format(a.format).lineup))
    seasons = ctx.talent.shape[2]
    places = run(ctx, lg, a.leagues, seasons, a.seed, a.workers)
    t2 = time.perf_counter()
    log.info("%d league paths x %d seasons in %.0fs", a.leagues, seasons, t2 - t1)
    out = output_dir(a.as_of, a.format) / f"league_{a.league}"
    out.mkdir(parents=True, exist_ok=True)
    summarize(places, lg, ctx.season0).write_parquet(out / "odds.parquet")
    np.save(out / "places.npy", places.astype(np.int8))
    (out / "run.json").write_text(json.dumps({
        "as_of": a.as_of.isoformat(), "format": a.format, "league": a.league,
        "sleeper_pull": sl.pull_date.isoformat() if sl.pull_date else None, "leagues": a.leagues, "seed": a.seed,
        "rostered_matched": int((lg.roster >= 0).sum()), "rostered_total": S.rostered(sl).height,
        "weeks_left": int(lg.schedule_left.shape[0]), "runtime_s": {"context": round(t1 - t0), "sim": round(t2 - t1)},
        "data_dir": str(data_dir()),
    }, indent=2))
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
