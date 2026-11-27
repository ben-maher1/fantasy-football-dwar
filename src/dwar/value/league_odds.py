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

from dwar.features.scoring import needs_long_plays
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
from dwar.value import links as LK
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
TRADABLE = 3     # rookie drafts Sleeper lets teams trade; later ones are simulated (own picks) but not valued as assets
DISCOUNT = 0.85   # default only: the CLI passes the format's discount (ben, 2026-11-24: 0.85)


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
    links: "LinkInputs | None" = None


@dataclass
class LinkInputs:
    """Weekly links within NFL games (value/links.py)."""

    team: np.ndarray                 # (P,) NFL team index of each context player, -1 unknown
    n_teams: int
    opp_weeks: dict[int, np.ndarray]  # NFL week this season -> (n_teams,) opponent index, -1 bye
    chol: np.ndarray                 # (14, 14) Cholesky factor of one game's role draws
    handoffs: dict[str, dict[str, float]]


def build_context(as_of: date, format_name: str) -> Context:
    """Rebuild the board's inputs from saved fits (same draws as dwar.parquet)
    and keep every player's per-season talent and games, plus each future
    rookie class's sample rookies."""
    from dwar.value.picks import pick_values

    fmt = load_format(format_name)
    model_cfg, cfg = load_model_config(), load_value_config()
    out = output_dir(as_of, format_name)
    snap = asof_snapshot(RawTables.from_cache(long_plays=needs_long_plays(load_scoring(fmt.scoring))), as_of)
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
    ids, pos, tal, sh, nfl = [], [], [], [], []
    for p, r in fits.items():
        latest, _, tr = position_values(p, r, inputs, tables, fmt, cfg, rng, trace=True)
        if latest.is_empty():
            continue
        full = tr.full if p in cfg.full_games else 1.0
        left = np.array([inputs.clock.remaining.get(t, inputs.clock.remaining_default) for t in latest["team"]], float)
        denom = np.full(tr.games.shape, float(G))
        denom[:, :, 0] = np.maximum(left, 1.0)[None, :]
        ids += latest["player_id"].to_list()
        nfl += latest["team"].to_list()
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
    # Every rookie draft in the horizon gets a class, not just the ones the board values: teams keep
    # drafting with their own picks after the tradable years (ben, 2026-11-24).
    pcfg = cfg.model_copy(update={"picks": cfg.picks.model_copy(update={"years": cfg.horizon_seasons})}) \
        if cfg.picks is not None else cfg
    pick_values(fits, tables, snap.schedules, inputs, fmt, pcfg, dict(model_cfg.min_plays or {}),
                np.random.default_rng([model_cfg.seed, 7]), traces=traces)
    rookies = {}
    for year, (rank, rpos, tr) in traces.items():
        full = np.where(rpos[None, :, None] == "QB", tr.full, 1.0)
        rookies[year] = (rank, rpos, tr.talent[:D], (np.clip(tr.games * full / G, 0, 1) * tr.alive)[:D])
    week_mult = None
    if cfg.matchups is not None:
        week_mult = matchup_weeks(tables.player_weeks, snap.schedules, inputs.clock.current,
                                  [ids[i] for i in keep], list(np.array(pos, object)[keep]), cfg.matchups)
    links = None
    if cfg.links is not None:
        links = link_inputs(snap.schedules, inputs.clock.current, [nfl[i] for i in keep], cfg.links)
    return Context(
        week_mult=week_mult, links=links,
        season0=inputs.clock.current, player_id=[ids[i] for i in keep], position=np.array(pos, object)[keep],
        talent=talent[:, keep], share=share[:, keep], rookies=rookies,
        waiver={p: float(np.mean(v)) for p, v in inputs.replacement.items()},
        params={p: {**{n: np.asarray(v) for n, v in d.params.items()}} | {"zero_value": d.zero_value}
                for p, d in inputs.draws.items()},
        likelihood={p: for_position(model_cfg.likelihoods(), p) for p in inputs.draws},
    )


def link_inputs(schedules: pl.DataFrame, season: int, teams: list[str | None], settings) -> LinkInputs:
    """NFL team indices, this season's weekly opponents and the game draw
    factor for the league sim's links."""
    weeks = sorted(schedules.filter(pl.col("season") == season)["week"].unique().to_list())
    games = M.team_weeks(schedules, season, weeks)
    names = sorted(set(games["team"].to_list()))
    ix = {t: k for k, t in enumerate(names)}
    opp = {}
    for w in weeks:
        o = np.full(len(names), -1)
        for t, op in games.filter(pl.col("week") == w).select("team", "opponent").iter_rows():
            o[ix[t]] = ix[op]
        opp[int(w)] = o
    team = np.array([ix.get(t, -1) if t is not None else -1 for t in teams], int)
    return LinkInputs(team, len(names), opp, LK.game_cholesky(dict(settings.mate), dict(settings.opp)),
                      {k: dict(v) for k, v in settings.handoffs.items()})


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
    division: np.ndarray | None = None  # (teams,) Sleeper division; None or one division = no division seeding
    # Free agents simulated alongside the rosters so they can be added (redraft pickups); never in a lineup
    # unless added. Context player indices.
    free: np.ndarray | None = None
    waiver_slots: dict[str, int] | None = None   # waiver refills per position; None = NO_QB_WAIVER


def seed_order(wins: np.ndarray, pf: np.ndarray, division: np.ndarray | None) -> np.ndarray:
    """Teams by playoff seed, (..., T) from (..., T) wins and points for.
    Sleeper's default with divisions (ben, 2026-11-26): each division winner
    (best record, points for breaking ties) takes a top seed, so the byes go
    to them; everyone else follows by record, then points for."""
    key = wins * 1e6 + pf
    if division is not None and len(np.unique(division)) > 1:
        win = np.zeros(key.shape, bool)
        for d in np.unique(division):
            k = np.where(division == d, key, -np.inf)
            np.put_along_axis(win, k.argmax(-1)[..., None], True, -1)
        key = key + win * 1e12
    return np.argsort(-key, axis=-1, kind="stable")


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
                       sched, fmt, picks, int(s.get("draft_rounds") or 5), lineup, last + 1,
                       st["division"].to_numpy())


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
    held = {min(p[0] for p in lg.picks) + k for k in range(TRADABLE)} if lg.picks else set()
    out += [Asset("pick", k, owner, "remove") for k, (y, _, _, owner) in enumerate(lg.picks) if y in held]
    out += [Asset("player", int(i), me, "add") for i in np.flatnonzero((lg.roster >= 0) & (lg.roster != me))]
    out += [Asset("pick", k, me, "add") for k, (y, _, _, owner) in enumerate(lg.picks) if owner != me and y in held]
    return out


def drop_for(ctx: Context, lg: LeagueSetup, me: int) -> int:
    """The player `me` would cut to make room for an added one: the
    lowest projected points over the horizon (posterior mean)."""
    mine = np.flatnonzero(lg.roster == me)
    proj = (ctx.talent.mean(0) * ctx.share.mean(0)).sum(1)
    return int(mine[np.argmin(proj[mine])])


def redraft_context(ctx: Context) -> Context:
    """Redraft mode (ben, 2026-11-26): only the rest of this season counts,
    so keep season 0 and no rookie classes."""
    return replace(ctx, talent=ctx.talent[:, :, :1].copy(), share=ctx.share[:, :, :1].copy(), rookies={})


def scale_points(ctx: Context, factor: np.ndarray) -> Context:
    """Every player's expected points times his own `factor` (P,), e.g. a
    league that also scores first downs, from fits under base scoring
    (value/scoring_scale.py). The weekly spread on the model's square-root
    scale is kept."""
    return replace(ctx, talent=ctx.talent * np.asarray(factor, float)[None, :, None])


def rest_of_season(ctx: Context) -> np.ndarray:
    """(P,) expected points left this season: mean talent x mean share."""
    return (ctx.talent[:, :, 0] * ctx.share[:, :, 0]).mean(0)


def free_agent_waiver(ctx: Context, lg: LeagueSetup, rank: int = 3) -> dict[str, float]:
    """Waiver level from the league's real free agents: per position, the
    expected points per week left (rate x share of weeks played) of the
    `rank`-th best unrostered player (the best one or two tend to be
    claimed first). A refill plays every week, so a backup's rate when he
    starts would overstate it (a 16-PPG backup QB who plays 3 weeks in 10)."""
    exp = rest_of_season(ctx)
    out = dict(ctx.waiver)
    for p in out:
        idx = np.flatnonzero((lg.roster < 0) & (ctx.position == p))
        if len(idx):
            out[p] = float(exp[idx[np.argsort(-exp[idx], kind="stable")][min(rank, len(idx)) - 1]])
    return out


def free_agent_assets(ctx: Context, lg: LeagueSetup, me: int, n: int) -> tuple[list[Asset], np.ndarray]:
    """The `n` unrostered players with the most points left this season, as
    pickups for `me` ("add", cutting `me`'s drop player), and their indices."""
    exp = rest_of_season(ctx)
    idx = np.flatnonzero(lg.roster < 0)
    top = idx[np.argsort(-exp[idx], kind="stable")][:n]
    return [Asset("player", int(i), me, "add") for i in top], top


def redraft_setup(ctx: Context, lg: LeagueSetup, me: int, free_agents: int, rank: int = 3
                  ) -> tuple[Context, LeagueSetup, list[Asset]]:
    """Redraft context and league: one season, no picks, waiver level from
    the real free agents (a QB too: redraft managers stream one), and the
    top free agents simulated so `me` can add them."""
    ctx = replace(ctx, waiver=free_agent_waiver(ctx, lg, rank))
    fa, top = free_agent_assets(ctx, lg, me, free_agents) if me >= 0 else ([], np.zeros(0, int))
    slots = {p: max(n, 1) for p, n in NO_QB_WAIVER.items()}
    return ctx, replace(lg, picks=[], free=top, waiver_slots=slots), fa


def horizon_points(ctx: Context, discount: float = DISCOUNT) -> np.ndarray:
    """(P,) expected points over the horizon, future seasons discounted."""
    exp = (ctx.talent * ctx.share).mean(0)                     # (P, seasons)
    return exp @ (discount ** np.arange(exp.shape[1]))


def waiver_pool(ctx: Context, lg: LeagueSetup, n: int, also: np.ndarray, discount: float = DISCOUNT) -> np.ndarray:
    """Dynasty free agents worth simulating: the `n` unrostered players with
    the most discounted points over the horizon, plus any in `also`
    (unrostered rookies, whose value is mostly later)."""
    idx = np.flatnonzero(lg.roster < 0)
    top = idx[np.argsort(-horizon_points(ctx, discount)[idx], kind="stable")][:n]
    extra = np.asarray(also, int)
    return np.unique(np.concatenate([top, extra[lg.roster[extra] < 0]]))


def lowest_cut(values: dict[int, float], candidates: np.ndarray) -> int:
    """The candidate (a player whose cut frees a bench spot) worth the
    least to his team; ties go to the first listed."""
    if not len(candidates):
        raise ValueError("no player can be cut")
    return int(min(candidates, key=lambda i: values.get(int(i), 0.0)))


def waiver_assets(fa: np.ndarray, me: int, stash: np.ndarray | None = None) -> list[Asset]:
    """Each free agent added to `me` (cutting `me`'s drop player), and each
    in `stash` put on an open taxi spot instead (no cut)."""
    out = [Asset("player", int(i), me, "add") for i in fa]
    return out + [Asset("player", int(i), me, "stash") for i in (stash if stash is not None else [])]


def _season(li: LeagueInputs, team: np.ndarray, T: int, lineup: dict[str, int], weeks: int,
            rng: np.random.Generator, cfs: dict[int, list[tuple]], drop: dict[int, int],
            mult: np.ndarray | None = None, link: tuple | None = None,
            waiver_slots: dict[str, int] | None = None) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Weekly lineup points (T, weeks) for fixed rosters, and for each team
    k in `cfs` its points (C_k, weeks) under each counterfactual: ("remove",
    i) takes pool player i off, ("add", i) adds him (cutting `drop[k]`).
    Every version sees the same weeks, so luck cancels. `mult` (pool,
    weeks) scales each player's expected points each week (matchups;
    managers see it when they set lineups) and 0 is a week out (bye).
    `link` (pool NFL team, role, (weeks, teams) NFL opponents, Cholesky
    factor, handoffs, number of NFL teams) links weekly draws within NFL
    games and moves teammates' expected points when a QB1 or RB1 sits."""
    rost = np.flatnonzero(team >= 0)      # rostered, plus free agents on team T (never scored unless added)
    loc = np.full(len(team), -1)
    loc[rost] = np.arange(len(rost))
    pos_r = li.pos[rost]
    z = None
    if link is not None:
        lt, lr, lo, L, hand, nT = link
        av = availability(li.share, weeks, OUT_RUN_WEEKS, rng)
        if mult is not None:
            av &= mult > 0
        hm = LK.handoff_mult(av, lt, lr, nT, hand)
        q1 = LK.holders(lt, lr, nT)[:, 0]
        qb_out = np.zeros((nT, weeks), bool)
        qb_out[q1 >= 0] = ~av[q1[q1 >= 0]]
        z = LK.game_draws(lt, lr, qb_out, lo, L, rng)[rost]
        avail = av[rost]
        mult = hm if mult is None else mult * hm
    else:
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
                pts[sel] = (simulate_weeks(li.likelihood[p], c, li.params[p], rng) if z is None
                            else LK.weekly_points(c, li.params[p], z[sel]))
        key_r = np.where(avail, li.known[rost][:, None] * m, -np.inf)
    wpos = np.array([p for p, n in (waiver_slots or NO_QB_WAIVER).items() for _ in range(n) if p in li.params], object)
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


def _champs(scores: np.ndarray, sched: np.ndarray, wins0: np.ndarray, pf0: np.ndarray, fmt: SeasonFormat,
            division: np.ndarray | None = None) -> np.ndarray:
    """Champion (...,) from (..., T, W) scores: `sched` (regular weeks, T)
    opponents, banked wins and points added before seeding."""
    W = sched.shape[0]
    reg = scores[..., :W]
    opp = np.take_along_axis(reg, np.broadcast_to(sched.T, reg.shape), axis=-2)
    wins = wins0 + (reg > opp).sum(-1) + 0.5 * (reg == opp).sum(-1)
    pf = pf0 + reg.sum(-1)
    order = seed_order(wins, pf, division)[..., : fmt.playoff_teams]
    return bracket(scores, order, W, fmt.byes)


def simulate_path(ctx: Context, lg: LeagueSetup, d: int, seasons: int, rng: np.random.Generator,
                  known: np.ndarray, assets: list[Asset] | None = None, drop: dict[int, int] | None = None
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One league path from this season on: (seasons, teams) final places
    (0 = champion), (seasons, assets) title changes (-1, 0, 1) for each
    asset's team with versus without it, and (seasons, teams) playoff seeds
    (0 = top seed)."""
    assets, drop = assets or [], drop or {}
    T = lg.teams
    places = np.zeros((seasons, T), int)
    seeds = np.zeros((seasons, T), int)
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
        if lg.free is not None and len(lg.free):
            team[lg.free] = T
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
            if x.mode == "stash":
                # taxi stash: no cut, and he cannot start this season (counts 0)
                if j > 0:
                    cfs.setdefault(x.team, []).append(("trade", (("add", i),)))
                    where.setdefault(x.team, []).append(a)
                continue
            cfs.setdefault(x.team, []).append((x.mode, i))
            where.setdefault(x.team, []).append(a)
        mult = None
        if j == 0 and ctx.week_mult:
            ones = np.ones(len(team))
            mult = np.stack([np.concatenate([ctx.week_mult.get(lg.first_week + w, ones[:P]), ones[P:]])
                             for w in range(weeks)], 1)
        link = None
        if ctx.links is not None:
            L_ = ctx.links
            lt = np.concatenate([L_.team, np.full(len(team) - P, -1)])
            lr = LK.roles(lt, li.pos, li.known if li.known is not None else li.talent)
            if j == 0:
                nobye = np.full(L_.n_teams, -1)
                lo = np.stack([L_.opp_weeks.get(lg.first_week + w, nobye) for w in range(weeks)])
            else:
                lo = LK.random_opponents(L_.n_teams, weeks, rng)
            link = (lt, lr, lo, L_.chol, L_.handoffs, L_.n_teams)
        base, cf = _season(li, team, T, lg.lineup, weeks, rng, cfs, drop, mult, link, lg.waiver_slots)
        reg = base[:, :reg_w]
        opp = np.take_along_axis(reg, sched.T, 0)
        wins = w0 + (reg > opp).sum(1) + 0.5 * (reg == opp).sum(1)
        pf = p0 + reg.sum(1)
        order = seed_order(wins, pf, lg.division)
        seeds[j, order] = np.arange(T)
        fin = playoffs(order, base[:, reg_w:], lg.fmt.playoff_teams, lg.fmt.byes)
        places[j, fin] = np.arange(T)
        for k, sc_k in cf.items():
            S_ = np.broadcast_to(base, (len(sc_k), T, weeks)).copy()
            S_[:, k] = sc_k
            won = (_champs(S_, sched, w0, p0, lg.fmt, lg.division) == k).astype(np.int8) - np.int8(fin[0] == k)
            delta[j, where[k]] = np.where(where_mode(assets, where[k]) == "remove", -won, won)
    return places, delta, seeds


def where_mode(assets: list[Asset], idx: list[int]) -> np.ndarray:
    return np.array([assets[a].mode for a in idx], object)


_CTX: Context | None = None
_LG: LeagueSetup | None = None
_KNOWN: np.ndarray | None = None
_ASSETS: list[Asset] = []
_DROP: dict[int, int] = {}


def _chunk(args: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start, n, seasons, seed = args
    rng = np.random.default_rng([seed, start])
    D = _CTX.talent.shape[0]
    res = [simulate_path(_CTX, _LG, i % D, seasons, rng, _KNOWN, _ASSETS, _DROP) for i in range(start, start + n)]
    return (np.stack([r[0] for r in res]).astype(np.int8), np.stack([r[1] for r in res]),
            np.stack([r[2] for r in res]).astype(np.int8))


def run(ctx: Context, lg: LeagueSetup, n: int, seasons: int, seed: int = 2026, workers: int = 4,
        chunk: int = 50, assets: list[Asset] | None = None, drop: dict[int, int] | None = None,
        with_seeds: bool = False) -> tuple[np.ndarray, ...]:
    """(n, seasons, teams) final places and (n, seasons, assets) title changes,
    plus (n, seasons, teams) playoff seeds when `with_seeds`."""
    global _CTX, _LG, _KNOWN, _ASSETS, _DROP
    _CTX, _LG, _KNOWN, _ASSETS, _DROP = ctx, lg, ctx.talent.mean(0), assets or [], drop or {}
    jobs = [(s, min(chunk, n - s), seasons, seed) for s in range(0, n, chunk)]
    with mp.get_context("fork").Pool(workers) as p:
        parts = p.map(_chunk, jobs)
    out = tuple(np.concatenate([q[i] for q in parts]) for i in range(3))
    return out if with_seeds else out[:2]


def summarize(places: np.ndarray, lg: LeagueSetup, season0: int, seeds: np.ndarray | None = None) -> pl.DataFrame:
    n, J, T = places.shape
    rows = []
    for j in range(J):
        for t in range(T):
            p = places[:, j, t]
            title = float((p == 0).mean())
            rows.append({"season": season0 + j, "team": t, "owner": lg.owners[t], "title": title,
                         "title_se": float(np.sqrt(title * (1 - title) / n)),
                         "final": float((p <= 1).mean()), "playoffs": float((p < lg.fmt.playoff_teams).mean()),
                         "mean_place": float(p.mean() + 1),
                         **({"bye": float((seeds[:, j, t] < lg.fmt.byes).mean())} if seeds is not None else {})})
    return pl.DataFrame(rows)




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
        rows.append({"kind": x.kind, "id": ident, "position": pos, "owner": lg.owners[owner] if owner >= 0 else "FA",
                     "for_team": lg.owners[x.team], "mode": x.mode,
                     "value": float(tot[:, a].mean() * 100), "se": float(tot[:, a].std() * 100 / np.sqrt(n)),
                     **{f"v{ctx.season0 + j}": float(mean_j[j, a]) for j in range(J)}})
    return pl.DataFrame(rows)


def waivers_main(a: argparse.Namespace, sl: S.League, ctx: Context, lg: LeagueSetup, xw: pl.DataFrame,
                 me: int, t0: float, t1: float) -> None:
    """Dynasty waiver pickups (ben, 2026-11-27). Pass 1 values `me`'s active
    players (with and without) to find the cut; pass 2 adds each free agent
    while cutting him, or stashes a rookie on an open taxi spot."""
    from dwar.config import data_dir
    from dwar.ingest.nflverse import load_raw
    if me < 0:
        raise SystemExit("--waivers needs --me")
    fmt = load_format(a.format)
    st = S.standings(sl)
    rid = int(st["roster_id"][me])
    where = {g: i for i, g in enumerate(ctx.player_id)}
    ros = (S.rostered(sl).filter(pl.col("roster_id") == rid)
           .join(xw.select("sleeper_id", "gsis_id"), on="sleeper_id", how="left"))
    active = ros.filter(pl.col("slot") == "active")
    idx = [where.get(g) for g in active["gsis_id"].to_list()]
    cands = np.array([i for i in idx if i is not None], int)
    unmatched = [n for n, i in zip(active["name"].to_list(), idx) if i is None]
    settings = sl.league["settings"]
    capacity = sum(1 for x in sl.league["roster_positions"] if x not in ("IR", "TAXI"))
    open_active = capacity - active.height
    taxi_open = int(settings.get("taxi_slots") or 0) - ros.filter(pl.col("slot") == "taxi").height
    season = sl.season
    pls = load_raw("players").select("gsis_id", "rookie_season", "draft_year")
    rookie_ids = set(pls.filter(pl.coalesce("rookie_season", "draft_year") == season)["gsis_id"].to_list())
    rookies = np.array([i for g, i in where.items() if g in rookie_ids], int)
    pool = waiver_pool(ctx, lg, a.free_agents, rookies, fmt.discount)
    lg = replace(lg, free=pool)
    seasons = ctx.talent.shape[2]
    out = output_dir(a.as_of, a.format) / f"league_{a.league}_waivers"
    out.mkdir(parents=True, exist_ok=True)
    cut, why = None, "open active spot"
    if open_active <= 0 and unmatched:
        why = f"not on the board: {unmatched[0]}"
    elif open_active <= 0:
        mine = [Asset("player", int(i), me, "remove") for i in cands]
        _, d1 = run(ctx, lg, a.leagues, seasons, a.seed, a.workers, assets=mine)
        cv = asset_values(d1, mine, ctx, lg, fmt.discount)
        cv.write_parquet(out / "cut_values.parquet")
        cut = lowest_cut(dict(zip(cands.tolist(), cv["value"].to_list())), cands)
        why = "lowest title value among active players"
        log.info("cut: %s (%.2f)", ctx.player_id[cut], float(cv["value"].min()))
    t2 = time.perf_counter()
    stash = np.intersect1d(pool, rookies) if taxi_open > 0 else np.zeros(0, int)
    assets = waiver_assets(pool, me, stash)
    _, d2 = run(ctx, lg, a.leagues, seasons, a.seed, a.workers, assets=assets,
                drop={me: cut} if cut is not None else {})
    t3 = time.perf_counter()
    asset_values(d2, assets, ctx, lg, fmt.discount).write_parquet(out / "values.parquet")
    (out / "run.json").write_text(json.dumps({
        "as_of": a.as_of.isoformat(), "format": a.format, "league": a.league,
        "sleeper_pull": sl.pull_date.isoformat() if sl.pull_date else None, "leagues": a.leagues, "seed": a.seed,
        "me": a.me, "free_agents": int(len(pool)), "stash": int(len(stash)), "rookies_free": int(np.isin(rookies, pool).sum()),
        "cut": ctx.player_id[cut] if cut is not None else None, "cut_reason": why,
        "active": active.height, "capacity": capacity, "taxi_open": taxi_open, "unmatched_active": unmatched,
        "discount": fmt.discount, "runtime_s": {"context": round(t1 - t0), "cut": round(t2 - t1), "adds": round(t3 - t2)},
        "data_dir": str(data_dir()),
    }, indent=2))
    log.info("wrote %s", out)


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
    ap.add_argument("--redraft", action="store_true",
                    help="only this season counts: no picks, free-agent waiver level and pickups (ben, 2026-11-26)")
    ap.add_argument("--free-agents", type=int, default=60,
                    help="redraft or --waivers: free agents valued as pickups for --me")
    ap.add_argument("--waivers", action="store_true",
                    help="dynasty waiver pickups for --me: each free agent added, cutting --me's active "
                         "player worth least to his title odds; rookies to an open taxi spot (ben, 2026-11-27)")
    ap.add_argument("--fits-format", default=None,
                    help="format whose saved fits to use (default --format); its scoring is the base for --scale-scoring")
    ap.add_argument("--scale-scoring", default=None,
                    help="league scoring to reach by scaling each player's points from the fits' scoring "
                         "(value/scoring_scale.py), e.g. first downs")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    t0 = time.perf_counter()
    if a.pull:
        S.pull(a.league)
    sl = S.load(a.league)
    fits_format = a.fits_format or a.format
    ctx = build_context(a.as_of, fits_format)
    if a.scale_scoring:
        from dwar.value.scoring_scale import factors_for
        ctx = scale_points(ctx, factors_for(a.as_of, load_format(fits_format).scoring, a.scale_scoring,
                                            ctx.player_id, list(ctx.position), ctx.season0))
    if a.redraft:
        ctx = redraft_context(ctx)
    t1 = time.perf_counter()
    log.info("context: %d players, %d draws in %.0fs", len(ctx.player_id), ctx.talent.shape[0], t1 - t0)
    from dwar.config import data_dir
    from dwar.ingest.nflverse import load_raw
    xw = S.crosswalk(sl.players, load_raw("rosters").select("gsis_id", pl.col("sleeper_id").cast(pl.Utf8)),
                     load_raw("players"), S.rostered(sl)["sleeper_id"].to_list())
    lg = league_setup(sl, ctx, xw, dict(load_format(a.format).lineup))
    me = lg.owners.index(a.me) if a.me else -1
    if a.waivers:
        return waivers_main(a, sl, ctx, lg, xw, me, t0, t1)
    fa: list[Asset] = []
    if a.redraft:
        ctx, lg, fa = redraft_setup(ctx, lg, me, a.free_agents)
        log.info("redraft waiver level: %s", {p: round(v, 1) for p, v in ctx.waiver.items()})
    seasons = ctx.talent.shape[2]
    assets, drop = [], {}
    if not a.no_values:
        assets = assets_for(lg, me) if me >= 0 else [x for x in assets_for(lg, 0) if x.mode == "remove"]
        assets += fa
        drop = {me: drop_for(ctx, lg, me)} if me >= 0 else {}
    places, delta, seeds = run(ctx, lg, a.leagues, seasons, a.seed, a.workers, assets=assets, drop=drop, with_seeds=True)
    t2 = time.perf_counter()
    log.info("%d league paths x %d seasons in %.0fs", a.leagues, seasons, t2 - t1)
    out = output_dir(a.as_of, a.format) / (f"league_{a.league}" + ("_redraft" if a.redraft else ""))
    out.mkdir(parents=True, exist_ok=True)
    summarize(places, lg, ctx.season0, seeds).write_parquet(out / "odds.parquet")
    np.save(out / "places.npy", places.astype(np.int8))
    if assets:
        asset_values(delta, assets, ctx, lg, load_format(a.format).discount).write_parquet(out / "values.parquet")
    (out / "run.json").write_text(json.dumps({
        "as_of": a.as_of.isoformat(), "format": a.format, "league": a.league,
        "sleeper_pull": sl.pull_date.isoformat() if sl.pull_date else None, "leagues": a.leagues, "seed": a.seed,
        "me": a.me, "assets": len(assets), "redraft": a.redraft, "fits_format": fits_format,
        "scale_scoring": a.scale_scoring, "waiver": ctx.waiver, "drop": ctx.player_id[drop[me]] if drop else None,
        "rostered_matched": int((lg.roster >= 0).sum()), "rostered_total": S.rostered(sl).height,
        "weeks_left": int(lg.schedule_left.shape[0]), "runtime_s": {"context": round(t1 - t0), "sim": round(t2 - t1)},
        "data_dir": str(data_dir()),
    }, indent=2))
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
