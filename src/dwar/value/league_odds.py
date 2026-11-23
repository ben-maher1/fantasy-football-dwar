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
from dwar.model.talent import for_position, simulate_weeks
from dwar.pipeline import output_dir
from dwar.value.dwar import position_values, season_clock, value_inputs
from dwar.value import matchups as M
from dwar.value.title import (
    LeagueInputs,
    SeasonFormat,
    availability,
    bracket,
    center_for,
    lineup_scores,
    playoffs,
    round_robin,
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
    # NFL week -> (P,) factor on expected points this season (value/matchups.py): matchup, 0 on a bye.
    # None (or a missing week) = every week alike.
    week_mult: dict[int, np.ndarray] | None = None


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
    week_mult = None
    if cfg.matchups is not None:
        week_mult = matchup_weeks(tables.player_weeks, snap.schedules, inputs.clock.current,
                                  [ids[i] for i in keep], list(np.array(pos, object)[keep]), cfg.matchups)
    return Context(
        week_mult=week_mult,
        season0=inputs.clock.current, player_id=[ids[i] for i in keep], position=np.array(pos, object)[keep],
        talent=talent[:, keep], share=share[:, keep], rookies=rookies,
        waiver={p: float(np.mean(v)) for p, v in inputs.replacement.items()},
        params={p: {**{n: np.asarray(v) for n, v in d.params.items()}} | {"zero_value": d.zero_value}
                for p, d in inputs.draws.items()},
        likelihood={p: for_position(model_cfg.likelihoods(), p) for p in inputs.draws},
    )


def matchup_weeks(player_weeks: pl.DataFrame, schedules: pl.DataFrame, season: int, ids: list[str],
                  positions: list[str], settings) -> dict[int, np.ndarray]:
    """NFL week -> (players,) matchup/bye factor for every regular-season
    week of `season`, from the games in `player_weeks` (as-of filtered)."""
    pw = player_weeks.filter(pl.col("season") >= season - 3).select(
        "player_id", "position", "season", "week", "game_id", "team", "opponent_team", "fpts")
    hist = player_weeks.filter(pl.col("season").is_between(season - 5, season - 1))
    pos_mean = dict(hist.group_by("position").agg(pl.col("fpts").mean()).iter_rows())
    rated = M.expectations(pw, pos_mean)
    ratings = M.defense_ratings(rated, season)
    now = pw.filter(pl.col("season") == season).sort("week")
    team = dict(now.group_by("player_id").agg(pl.col("team").last()).iter_rows())
    weeks = sorted(schedules.filter(pl.col("season") == season)["week"].unique().to_list())
    games = M.team_weeks(schedules, season, weeks)
    m = M.multipliers([team.get(i) for i in ids], positions, games, ratings, dict(settings.effects), weeks, settings.byes)
    return {int(w): m[:, k] for k, w in enumerate(weeks)}


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
    first_week: int = 1               # NFL week of the first week left


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
                       sched, fmt, picks, int(s.get("draft_rounds") or 5), lineup, last + 1)


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


@dataclass(frozen=True)
class Asset:
    """A with-and-without question: `team`'s title odds with and without
    `ref` ("remove" one of his own, or "add" one from another team)."""

    kind: str      # "player" or "pick"
    ref: int       # context player index, or index into LeagueSetup.picks
    team: int      # whose odds
    mode: str      # "remove", "add" or "trade"
    parts: tuple = ()   # for a trade: ((kind, ref, "remove" | "add"), ...) all for `team`


def assets_for(lg: LeagueSetup, me: int) -> list[Asset]:
    """Every rostered player and owned pick, removed from its owner; and
    every one owned by another team, added to `me`."""
    out = [Asset("player", int(i), int(lg.roster[i]), "remove") for i in np.flatnonzero(lg.roster >= 0)]
    out += [Asset("pick", k, owner, "remove") for k, (_, _, _, owner) in enumerate(lg.picks)]
    out += [Asset("player", int(i), me, "add") for i in np.flatnonzero((lg.roster >= 0) & (lg.roster != me))]
    out += [Asset("pick", k, me, "add") for k, (_, _, _, owner) in enumerate(lg.picks) if owner != me]
    return out


def drop_for(ctx: Context, lg: LeagueSetup, me: int) -> int:
    """The player `me` would cut to make room for an added one: the
    lowest projected points over the horizon (posterior mean)."""
    mine = np.flatnonzero(lg.roster == me)
    proj = (ctx.talent.mean(0) * ctx.share.mean(0)).sum(1)
    return int(mine[np.argmin(proj[mine])])


def _season(li: LeagueInputs, team: np.ndarray, T: int, lineup: dict[str, int], weeks: int,
            rng: np.random.Generator, cfs: dict[int, list[tuple]], drop: dict[int, int],
            mult: np.ndarray | None = None) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Weekly lineup points (T, weeks) for fixed rosters, and for each team
    k in `cfs` its points (C_k, weeks) under each counterfactual: ("remove",
    i) takes pool player i off, ("add", i) adds him (cutting `drop[k]`).
    Every version sees the same weeks, so luck cancels. `mult` (pool,
    weeks) scales each player's expected points each week (matchups;
    managers see it when they set lineups) and 0 is a week out (bye)."""
    rost = np.flatnonzero(team >= 0)
    loc = np.full(len(team), -1)
    loc[rost] = np.arange(len(rost))
    pos_r = li.pos[rost]
    avail = availability(li.share[rost], weeks, OUT_RUN_WEEKS, rng)
    pts = np.zeros((len(rost), weeks))
    if mult is None:
        for p in lineup:
            sel = pos_r == p
            if sel.any():
                c = center_for(li.likelihood[p], li.talent[rost][sel], li.params[p])
                pts[sel] = simulate_weeks(li.likelihood[p], np.repeat(c[:, None], weeks, 1), li.params[p], rng)
        key_r = np.where(avail, li.known[rost][:, None], -np.inf)
    else:
        m = mult[rost]
        avail &= m > 0
        for p in lineup:
            sel = pos_r == p
            if sel.any():
                tal = li.talent[rost][sel][:, None] * np.where(m[sel] > 0, m[sel], 1.0)
                c = center_for(li.likelihood[p], tal.ravel(), li.params[p]).reshape(tal.shape)
                pts[sel] = simulate_weeks(li.likelihood[p], c, li.params[p], rng)
        key_r = np.where(avail, li.known[rost][:, None] * m, -np.inf)
    wpos = np.array([p for p, n in NO_QB_WAIVER.items() for _ in range(n) if p in li.params], object)
    wc = {p: center_for(li.likelihood[p], np.array([li.waiver[p]]), li.params[p])[0] for p in set(wpos)}
    wkey = np.array([[li.waiver[p]] * weeks for p in wpos], float).reshape(len(wpos), weeks)
    extra_pos = np.array([p for p in ("QB", "RB", "WR", "TE") for _ in range(4)], object)   # room for 4 adds each
    base = np.zeros((T, weeks))
    out: dict[int, np.ndarray] = {}
    team_r = team[rost]
    for k in range(T):
        mine = np.flatnonzero(team_r == k)
        wpts = np.stack([simulate_weeks(li.likelihood[p], np.full(weeks, wc[p]), li.params[p], rng) for p in wpos]) \
            if len(wpos) else np.zeros((0, weeks))
        key = np.concatenate([key_r[mine], wkey], 0)            # (S, W)
        pk = np.concatenate([pts[mine], wpts], 0)
        spos = np.concatenate([pos_r[mine], wpos])
        base[k] = lineup_scores(key.T, pk.T, spos, lineup)
        todo = cfs.get(k)
        if not todo:
            continue
        slot = {int(r): j for j, r in enumerate(mine)}          # rost index -> slot
        C, S_ = len(todo), len(spos)
        E = len(extra_pos)
        K = np.broadcast_to(np.concatenate([key, np.full((E, weeks), -np.inf)], 0), (C, S_ + E, weeks)).copy()
        Pt = np.broadcast_to(np.concatenate([pk, np.zeros((E, weeks))], 0), (C, S_ + E, weeks)).copy()
        for c, (mode, i) in enumerate(todo):
            moves = list(i) if mode == "trade" else [(mode, i)]
            used: dict[str, int] = {}
            for m, j in moves:
                r = loc[j] if j >= 0 else -1
                if r < 0:
                    continue
                if m == "remove":
                    if r in slot:
                        K[c, slot[r]] = -np.inf
                else:
                    p = pos_r[r]
                    e = S_ + int(np.flatnonzero(extra_pos == p)[used.get(p, 0)])
                    used[p] = used.get(p, 0) + 1
                    K[c, e], Pt[c, e] = key_r[r], pts[r]
                    dr = loc[drop[k]] if (k in drop and mode != "trade") else -1
                    if dr in slot:
                        K[c, slot[dr]] = -np.inf
        out[k] = lineup_scores(K.transpose(0, 2, 1), Pt.transpose(0, 2, 1), np.concatenate([spos, extra_pos]), lineup)
    return base, out


def _champs(scores: np.ndarray, sched: np.ndarray, wins0: np.ndarray, pf0: np.ndarray, fmt: SeasonFormat) -> np.ndarray:
    """Champion (...,) from (..., T, W) scores: `sched` (regular weeks, T)
    opponents, banked wins and points added before seeding."""
    W = sched.shape[0]
    reg = scores[..., :W]
    opp = np.take_along_axis(reg, np.broadcast_to(sched.T, reg.shape), axis=-2)
    wins = wins0 + (reg > opp).sum(-1) + 0.5 * (reg == opp).sum(-1)
    pf = pf0 + reg.sum(-1)
    order = np.argsort(-(wins * 1e6 + pf), axis=-1, kind="stable")[..., : fmt.playoff_teams]
    return bracket(scores, order, W, fmt.byes)


def simulate_path(ctx: Context, lg: LeagueSetup, d: int, seasons: int, rng: np.random.Generator,
                  known: np.ndarray, assets: list[Asset] | None = None, drop: dict[int, int] | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
    """One league path from this season on: (seasons, teams) final places
    (0 = champion), and (seasons, assets) title changes (-1, 0, 1) for each
    asset's team with versus without it."""
    assets, drop = assets or [], drop or {}
    T = lg.teams
    places = np.zeros((seasons, T), int)
    delta = np.zeros((seasons, len(assets)), np.int8)
    P = len(ctx.player_id)
    pick_idx = np.full(len(lg.picks), -1)          # pool index of each pick's rookie, once drafted
    added_pos, added_tal, added_sh, added_team = [], [], [], []
    fin = None
    for j in range(seasons):
        year = ctx.season0 + j
        if j and year in ctx.rookies:
            rank, rpos, rtal, rsh = ctx.rookies[year]
            slot_of = {t: int(np.flatnonzero(fin[::-1] == t)[0]) + 1 for t in range(T)}   # worst finish picks first
            for q, (y, k, orig, owner) in enumerate(lg.picks):
                if y != year:
                    continue
                cand = np.flatnonzero(rank == (k - 1) * T + slot_of[orig])
                if len(cand):
                    c = rng.choice(cand)
                    pick_idx[q] = P + len(added_pos)
                    added_pos.append(rpos[c])
                    added_tal.append(rtal[d % rtal.shape[0], c])
                    added_sh.append(rsh[d % rsh.shape[0], c])
                    added_team.append(owner)
        extra = None
        if added_pos:
            extra = (np.array(added_pos, object), np.array([t[j] for t in added_tal]), np.array([x[j] for x in added_sh]))
        li = _inputs(ctx, d, j, extra, known)
        team = np.concatenate([lg.roster, np.array(added_team, int)]) if added_pos else lg.roster.copy()
        if j == 0:
            li = replace(li, share=np.where(lg.taxi, 0.0, li.share))
            reg_w, sched, w0, p0 = lg.schedule_left.shape[0], lg.schedule_left, lg.wins, lg.points
        else:
            reg_w = lg.fmt.regular_weeks
            sched, w0, p0 = round_robin(T, reg_w, rng), np.zeros(T), np.zeros(T)
        weeks = reg_w + lg.fmt.playoff_weeks
        cfs: dict[int, list[tuple]] = {}
        where: dict[int, list[int]] = {}
        for a, x in enumerate(assets):
            if x.mode == "trade":
                moves = tuple((m, r if kd == "player" else int(pick_idx[r])) for kd, r, m in x.parts)
                moves = tuple(mv for mv in moves if mv[1] >= 0)
                if not moves:
                    continue
                cfs.setdefault(x.team, []).append(("trade", moves))
                where.setdefault(x.team, []).append(a)
                continue
            i = x.ref if x.kind == "player" else int(pick_idx[x.ref])
            if i < 0:
                continue
            cfs.setdefault(x.team, []).append((x.mode, i))
            where.setdefault(x.team, []).append(a)
        mult = None
        if j == 0 and ctx.week_mult:
            ones = np.ones(len(team))
            mult = np.stack([np.concatenate([ctx.week_mult.get(lg.first_week + w, ones[:P]), ones[P:]])
                             for w in range(weeks)], 1)
        base, cf = _season(li, team, T, lg.lineup, weeks, rng, cfs, drop, mult)
        reg = base[:, :reg_w]
        opp = np.take_along_axis(reg, sched.T, 0)
        wins = w0 + (reg > opp).sum(1) + 0.5 * (reg == opp).sum(1)
        pf = p0 + reg.sum(1)
        order = np.argsort(-(wins * 1e6 + pf), kind="stable")
        fin = playoffs(order, base[:, reg_w:], lg.fmt.playoff_teams, lg.fmt.byes)
        places[j, fin] = np.arange(T)
        for k, sc_k in cf.items():
            S_ = np.broadcast_to(base, (len(sc_k), T, weeks)).copy()
            S_[:, k] = sc_k
            won = (_champs(S_, sched, w0, p0, lg.fmt) == k).astype(np.int8) - np.int8(fin[0] == k)
            delta[j, where[k]] = np.where(where_mode(assets, where[k]) == "remove", -won, won)
    return places, delta


def where_mode(assets: list[Asset], idx: list[int]) -> np.ndarray:
    return np.array([assets[a].mode for a in idx], object)


_CTX: Context | None = None
_LG: LeagueSetup | None = None
_KNOWN: np.ndarray | None = None
_ASSETS: list[Asset] = []
_DROP: dict[int, int] = {}


def _chunk(args: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    start, n, seasons, seed = args
    rng = np.random.default_rng([seed, start])
    D = _CTX.talent.shape[0]
    res = [simulate_path(_CTX, _LG, i % D, seasons, rng, _KNOWN, _ASSETS, _DROP) for i in range(start, start + n)]
    return np.stack([r[0] for r in res]).astype(np.int8), np.stack([r[1] for r in res])


def run(ctx: Context, lg: LeagueSetup, n: int, seasons: int, seed: int = 2026, workers: int = 4,
        chunk: int = 50, assets: list[Asset] | None = None, drop: dict[int, int] | None = None
        ) -> tuple[np.ndarray, np.ndarray]:
    """(n, seasons, teams) final places and (n, seasons, assets) title changes."""
    global _CTX, _LG, _KNOWN, _ASSETS, _DROP
    _CTX, _LG, _KNOWN, _ASSETS, _DROP = ctx, lg, ctx.talent.mean(0), assets or [], drop or {}
    jobs = [(s, min(chunk, n - s), seasons, seed) for s in range(0, n, chunk)]
    with mp.get_context("fork").Pool(workers) as p:
        parts = p.map(_chunk, jobs)
    return np.concatenate([a for a, _ in parts]), np.concatenate([b for _, b in parts])


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


DISCOUNT = 0.80


def asset_values(delta: np.ndarray, assets: list[Asset], ctx: Context, lg: LeagueSetup,
                 discount: float = DISCOUNT) -> pl.DataFrame:
    """Per asset: title-odds points by season and the discounted total
    (this season weight 1), with its standard error over league paths."""
    n, J, _ = delta.shape
    w = discount ** np.arange(J)
    tot = np.tensordot(delta.astype(np.float32), w.astype(np.float32), axes=([1], [0]))   # (n, A)
    mean_j = delta.mean(0) * 100
    rows = []
    for a, x in enumerate(assets):
        if x.kind == "trade":
            ident, pos, owner = f"trade {x.ref}", "TRADE", x.team
        elif x.kind == "player":
            ident, pos, owner = ctx.player_id[x.ref], str(ctx.position[x.ref]), int(lg.roster[x.ref])
        else:
            y, k, orig, owner = lg.picks[x.ref]
            ident, pos = f"{y} round {k} ({lg.owners[orig]})", "PICK"
        rows.append({"kind": x.kind, "id": ident, "position": pos, "owner": lg.owners[owner],
                     "for_team": lg.owners[x.team], "mode": x.mode,
                     "value": float(tot[:, a].mean() * 100), "se": float(tot[:, a].std() * 100 / np.sqrt(n)),
                     **{f"v{ctx.season0 + j}": float(mean_j[j, a]) for j in range(J)}})
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
    ap.add_argument("--me", default=None, help="Sleeper display name: also value other teams' assets to this team")
    ap.add_argument("--no-values", action="store_true", help="title odds only")
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
    assets, drop = [], {}
    if not a.no_values:
        me = lg.owners.index(a.me) if a.me else -1
        assets = assets_for(lg, me) if me >= 0 else [x for x in assets_for(lg, 0) if x.mode == "remove"]
        drop = {me: drop_for(ctx, lg, me)} if me >= 0 else {}
    places, delta = run(ctx, lg, a.leagues, seasons, a.seed, a.workers, assets=assets, drop=drop)
    t2 = time.perf_counter()
    log.info("%d league paths x %d seasons in %.0fs", a.leagues, seasons, t2 - t1)
    out = output_dir(a.as_of, a.format) / f"league_{a.league}"
    out.mkdir(parents=True, exist_ok=True)
    summarize(places, lg, ctx.season0).write_parquet(out / "odds.parquet")
    np.save(out / "places.npy", places.astype(np.int8))
    if assets:
        asset_values(delta, assets, ctx, lg).write_parquet(out / "values.parquet")
    (out / "run.json").write_text(json.dumps({
        "as_of": a.as_of.isoformat(), "format": a.format, "league": a.league,
        "sleeper_pull": sl.pull_date.isoformat() if sl.pull_date else None, "leagues": a.leagues, "seed": a.seed,
        "me": a.me, "assets": len(assets), "drop": ctx.player_id[drop[me]] if drop else None,
        "rostered_matched": int((lg.roster >= 0).sum()), "rostered_total": S.rostered(sl).height,
        "weeks_left": int(lg.schedule_left.shape[0]), "runtime_s": {"context": round(t1 - t0), "sim": round(t2 - t1)},
        "data_dir": str(data_dir()),
    }, indent=2))
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
