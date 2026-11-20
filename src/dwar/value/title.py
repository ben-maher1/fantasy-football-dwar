"""Title value: the extra championship chance a player gives a typical team.

Plan approved by ben, 2026-11-20 (Claude Doc "Plan: championship-based
player value"). DWAR averages a player's lineup points into wins; title
value plays whole fantasy seasons out week by week, standings and playoffs
included, and asks how often his team wins the title with him versus with
his roster spot left to the waiver wire.

Everything a player brings comes from the DWAR inputs for one future
season (value/dwar.py `SeasonTrace`): his talent (expected points per game
played), his expected share of games, and whether he is still in the
league, one posterior draw per simulated league. Weekly points are drawn
from each position's fitted weekly distribution (model/talent.py
`simulate_weeks`); missed weeks come in stretches (a two-state chain whose
mean absence is `out_run_weeks`).

Leagues are dealt from the DWAR board: per position, the top `teams x
by_position` players by DWAR rank plus noise, snake order, so leagues hold
real contenders and real rebuilders. Lineups start the best available
players by expected points (managers do not know the week's outcome), with
waiver players always available at the waiver level.

The same simulated weeks are reused with and without each rostered player
(common random numbers), so only his team's scores change. Per-player
swings are binary and noisy, so values are smoothed into a curve of title
odds added against talent and games share, by position (`title_curve`).

All functions are pure numpy; `simulate_leagues` is the driver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri

from dwar.model.talent import simulate_weeks, talent_from_center

FLEX = ("RB", "WR", "TE")


@dataclass(frozen=True)
class SeasonFormat:
    """Fantasy season shape (assumed until ben gives his league's)."""

    regular_weeks: int = 14
    playoff_teams: int = 6
    byes: int = 2

    @property
    def playoff_weeks(self) -> int:
        # 6 teams with 2 byes, or 4 teams with none: 3 or 2 rounds.
        n, r = self.playoff_teams, 0
        while n > 1:
            n = (n + 1) // 2 if r else (n - self.byes) // 2 + self.byes
            r += 1
        return r

    @property
    def weeks(self) -> int:
        return self.regular_weeks + self.playoff_weeks


# --- schedule, standings, playoffs -----------------------------------------


def round_robin(teams: int, weeks: int, rng: np.random.Generator) -> np.ndarray:
    """(weeks, teams) opponent of each team each week: a shuffled circle
    round robin, repeated as needed. `teams` must be even."""
    if teams % 2:
        raise ValueError("round_robin needs an even team count")
    order = rng.permutation(teams)
    rounds = []
    ring = list(range(1, teams))
    for _ in range(teams - 1):
        opp = np.empty(teams, int)
        pairs = [(0, ring[-1])] + [(ring[i], ring[-2 - i]) for i in range((teams - 2) // 2)]
        for a, b in pairs:
            opp[order[a]], opp[order[b]] = order[b], order[a]
        rounds.append(opp)
        ring = ring[-1:] + ring[:-1]
    rounds = [rounds[i] for i in rng.permutation(len(rounds))]
    return np.stack([rounds[w % len(rounds)] for w in range(weeks)])


def seeds(scores: np.ndarray, schedule: np.ndarray, regular_weeks: int) -> np.ndarray:
    """Teams ordered best first, by wins then points for. `scores` is
    (..., teams, weeks); returns (..., teams) team indices."""
    reg = scores[..., :regular_weeks]
    opp = np.take_along_axis(reg, np.broadcast_to(schedule[:regular_weeks].T, reg.shape), axis=-2)
    wins = (reg > opp).sum(-1) + 0.5 * (reg == opp).sum(-1)
    pf = reg.sum(-1)
    key = wins * 1e6 + pf     # points for never reach 1e6 in a season
    return np.argsort(-key, axis=-1, kind="stable")


def champion(scores: np.ndarray, schedule: np.ndarray, fmt: SeasonFormat) -> np.ndarray:
    """Title winner (...,) from season scores (..., teams, weeks). Byes go
    to the top seeds; each round reseeds so the best seed left plays the
    worst; an exact tie goes to the better seed."""
    order = seeds(scores, schedule, fmt.regular_weeks)[..., : fmt.playoff_teams]   # (..., P) by seed
    week = fmt.regular_weeks
    alive = order
    first = True
    while alive.shape[-1] > 1:
        n_bye = fmt.byes if first else 0
        bye, play = alive[..., :n_bye], alive[..., n_bye:]
        k = play.shape[-1] // 2
        hi, lo = play[..., :k], play[..., ::-1][..., :k]
        s = scores[..., week]
        s_hi = np.take_along_axis(s, hi, -1)
        s_lo = np.take_along_axis(s, lo, -1)
        winners = np.where(s_hi >= s_lo, hi, lo)
        # Keep seed order: byes, then winners by their original seed.
        merged = np.concatenate([bye, winners], -1)
        mrank = (merged[..., :, None] == order[..., None, :]).argmax(-1)   # seed position of each
        alive = np.take_along_axis(merged, np.argsort(mrank, -1), -1)
        week += 1
        first = False
    return alive[..., 0]


# --- players: availability, weekly points, lineups -------------------------


def availability(share: np.ndarray, weeks: int, run_weeks: float, rng: np.random.Generator) -> np.ndarray:
    """(players, weeks) True when a player plays. Each player plays a share
    `share` of weeks on average, missing them in stretches averaging
    `run_weeks` (a two-state chain started from its long-run state). Very low
    shares lengthen the stretches so the average still holds."""
    s = np.clip(np.asarray(share, float), 1e-6, 1.0)
    back = np.full_like(s, 1.0 / run_weeks)                  # P(out -> in)
    leave = back * (1.0 - s) / s                             # P(in -> out), keeps the share
    over = leave > 1.0
    back = np.where(over, s / (1.0 - s + 1e-12), back)
    leave = np.where(over, 1.0, leave)
    on = rng.random(s.shape) < s
    out = np.empty((len(s), weeks), bool)
    for w in range(weeks):
        out[:, w] = on
        u = rng.random(s.shape)
        on = np.where(on, u >= leave, u < back)
    return out


def center_for(likelihood: str, talent: np.ndarray, params: dict, iters: int = 40) -> np.ndarray:
    """The weekly distribution's center whose expected points are `talent`
    (inverse of `talent_from_center`, by bisection; scalar params)."""
    t = np.asarray(talent, float)
    lo, hi = np.full_like(t, -5.0), np.full_like(t, 12.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        up = talent_from_center(likelihood, mid, params, np) < t
        lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
    return 0.5 * (lo + hi)


def lineup_scores(key: np.ndarray, pts: np.ndarray, pos: np.ndarray, lineup: dict[str, int]) -> np.ndarray:
    """Points of the best lineup chosen by `key` (expected points, -inf when
    unavailable), scored by `pts`. `key`, `pts`: (..., slots); `pos`: (slots,)
    position labels. FLEX takes RB/WR/TE left after the fixed slots."""
    key = np.array(key, float, copy=True)
    total = np.zeros(key.shape[:-1])
    for p, n in lineup.items():
        if p == "FLEX" or n == 0:
            continue
        kp = np.where(pos == p, key, -np.inf)
        idx = np.argsort(-kp, axis=-1)[..., :n]
        total += np.where(np.take_along_axis(kp, idx, -1) > -np.inf, np.take_along_axis(pts, idx, -1), 0.0).sum(-1)
        np.put_along_axis(key, idx, -np.inf, -1)
    n = lineup.get("FLEX", 0)
    if n:
        kf = np.where(np.isin(pos, FLEX), key, -np.inf)
        idx = np.argsort(-kf, axis=-1)[..., :n]
        total += np.where(np.take_along_axis(kf, idx, -1) > -np.inf, np.take_along_axis(pts, idx, -1), 0.0).sum(-1)
    return total


def deal(rank_z: np.ndarray, pos: np.ndarray, by_position: dict[str, int], teams: int,
         noise: float, rng: np.random.Generator) -> np.ndarray:
    """Team (0..teams-1) of each player, -1 if not rostered. Per position,
    players are ordered by `rank_z + noise x N(0, 1)` and the top
    `teams x by_position[p]` are dealt in snake order from a random team
    order."""
    team = np.full(len(pos), -1)
    for p, n in by_position.items():
        idx = np.flatnonzero(pos == p)
        if len(idx) == 0 or n == 0:
            continue
        key = rank_z[idx] + noise * rng.standard_normal(len(idx))
        chosen = idx[np.argsort(-key)][: teams * n]
        perm = rng.permutation(teams)
        r = np.arange(len(chosen))
        snake = np.where((r // teams) % 2 == 0, r % teams, teams - 1 - r % teams)
        team[chosen] = perm[snake]
    return team


def rank_scores(value: np.ndarray) -> np.ndarray:
    """Normal scores of `value`'s ranks (best = largest), for dealing."""
    n = len(value)
    r = np.argsort(np.argsort(-np.asarray(value)))
    return -ndtri((r + 0.5) / n)


# --- one league ------------------------------------------------------------


@dataclass(frozen=True)
class LeagueInputs:
    """One simulated league season, from one posterior draw."""

    pos: np.ndarray        # (players,) position labels of the pool
    talent: np.ndarray     # (players,) expected points per game played this season
    share: np.ndarray      # (players,) expected share of the season's games played
    rank_z: np.ndarray     # (players,) dealing key (DWAR rank normal score)
    waiver: dict[str, float]          # waiver-level talent by position
    params: dict[str, dict]           # weekly-distribution params by position (scalars, this draw)
    likelihood: dict[str, str]


@dataclass(frozen=True)
class LeagueResult:
    player: np.ndarray     # (rostered,) pool index
    team: np.ndarray       # (rostered,) team
    delta: np.ndarray      # (rostered,) title(with) - title(without): -1, 0 or 1
    started: np.ndarray    # (rostered,) weeks in the lineup
    champ: int             # champion team
    strength: np.ndarray   # (teams,) mean weekly score
    week_sd: float         # week-to-week SD of a team's score, averaged over teams
    expected: np.ndarray   # (teams,) preseason strength: best lineup's expected points x share


WAIVER_COPIES = {"QB": 1, "RB": 5, "WR": 5, "TE": 4}


def simulate_league(li: LeagueInputs, lineup: dict[str, int], by_position: dict[str, int], teams: int,
                    fmt: SeasonFormat, noise: float, run_weeks: float, rng: np.random.Generator) -> LeagueResult:
    """Deal, play one season, and measure each rostered player's title
    swing: the title is decided again with his weeks taken away (his spot
    falls to the next best available, waiver players included)."""
    weeks = fmt.weeks
    team = deal(li.rank_z, li.pos, by_position, teams, noise, rng)
    rostered = np.flatnonzero(team >= 0)
    # Waiver copies per team: always available at the waiver level.
    wpos = np.array([p for p, n in WAIVER_COPIES.items() if p in by_position for _ in range(n)])
    n_r = max(int((team == k).sum()) for k in range(teams))
    slots = n_r + len(wpos)
    key = np.full((teams, slots, weeks), -np.inf)
    pts = np.zeros((teams, slots, weeks))
    spos = np.empty((teams, slots), object)
    owner = np.full((teams, slots), -1)
    avail_all = availability(li.share[rostered], weeks, run_weeks, rng)
    all_pts = np.zeros((len(rostered), weeks))
    for p in by_position:
        sel = li.pos[rostered] == p
        if sel.any():
            c = center_for(li.likelihood[p], li.talent[rostered][sel], li.params[p])
            all_pts[sel] = simulate_weeks(li.likelihood[p], np.repeat(c[:, None], weeks, 1), li.params[p], rng)
    wc = {p: center_for(li.likelihood[p], np.array([li.waiver[p]]), li.params[p])[0] for p in WAIVER_COPIES if p in by_position}
    for k in range(teams):
        mine = np.flatnonzero(team[rostered] == k)
        n = len(mine)
        owner[k, :n] = rostered[mine]
        spos[k, :n] = li.pos[rostered[mine]]
        key[k, :n] = np.where(avail_all[mine], li.talent[rostered[mine]][:, None], -np.inf)
        pts[k, :n] = all_pts[mine]
        spos[k, n_r:] = wpos
        for j, p in enumerate(wpos):
            key[k, n_r + j] = li.waiver[p]
            pts[k, n_r + j] = simulate_weeks(li.likelihood[p], np.full(weeks, wc[p]), li.params[p], rng)
        spos[k, n:n_r] = "NONE"
    # Variants per team: base, then each of its roster slots removed.
    V = n_r + 1
    kv = np.repeat(key[:, None], V, 1)                       # (teams, V, slots, weeks)
    for v in range(1, V):
        kv[:, v, v - 1] = -np.inf
    kv = np.moveaxis(kv, -1, -2)                             # (teams, V, weeks, slots)
    pv = np.broadcast_to(np.moveaxis(pts, -1, -2)[:, None], kv.shape)
    scores = np.stack([lineup_scores(kv[k], pv[k], spos[k], lineup) for k in range(teams)])  # (teams, V, weeks)
    base = scores[:, 0]                                      # (teams, weeks)
    # Weeks each slot started in the base lineup: points lost when removed > 0.
    # Preseason strength: the best lineup by expected points per team game.
    exp_pts = np.zeros((teams, slots))
    for k in range(teams):
        mine = owner[k] >= 0
        exp_pts[k, mine] = li.talent[owner[k, mine]] * li.share[owner[k, mine]]
        exp_pts[k, n_r:] = [li.waiver[p] for p in wpos]
    exp_key = np.where((owner >= 0) | (np.arange(slots) >= n_r)[None], exp_pts, -np.inf)
    expected = np.array([lineup_scores(exp_key[k], exp_pts[k], spos[k], lineup) for k in range(teams)])
    sched = round_robin(teams, weeks, rng)
    champ = int(champion(base, sched, fmt))
    # Counterfactual seasons: team k's row replaced by its variant v.
    cf = np.repeat(base[None, None], teams, 0).repeat(V - 1, 1)  # (teams, V-1, teams, weeks)
    for k in range(teams):
        cf[k, :, k] = scores[k, 1:]
    ch = champion(cf, sched, fmt)                            # (teams, V-1)
    won = (champ == np.arange(teams))[:, None].astype(int)
    lost = (ch == np.arange(teams)[:, None]).astype(int)
    delta = won - lost                                       # (teams, V-1)
    started = (np.abs(scores[:, 1:] - scores[:, :1]) > 1e-9).sum(-1)   # weeks he was in the lineup
    m = owner >= 0
    m[:, n_r:] = False
    sl = m[:, :n_r]
    return LeagueResult(
        player=owner[:, :n_r][sl], team=np.repeat(np.arange(teams)[:, None], n_r, 1)[sl],
        delta=delta[sl].astype(float), started=started[sl], champ=champ, strength=base.mean(-1),
        week_sd=float(base.std(-1, ddof=1).mean()), expected=expected,
    )


# --- smoothing --------------------------------------------------------------


@dataclass(frozen=True)
class TitleCurve:
    """Title odds added (probability) by talent x games share, one position."""

    talent: np.ndarray     # (nt,) grid
    share: np.ndarray      # (ns,) grid
    value: np.ndarray      # (nt, ns)

    def at(self, talent: np.ndarray, share: np.ndarray) -> np.ndarray:
        """Bilinear read-off, clamped to the grid."""
        t = np.clip(talent, self.talent[0], self.talent[-1])
        s = np.clip(share, self.share[0], self.share[-1])
        i = np.clip(np.searchsorted(self.talent, t) - 1, 0, len(self.talent) - 2)
        j = np.clip(np.searchsorted(self.share, s) - 1, 0, len(self.share) - 2)
        ft = (t - self.talent[i]) / (self.talent[i + 1] - self.talent[i])
        fs = (s - self.share[j]) / (self.share[j + 1] - self.share[j])
        v = self.value
        return ((1 - ft) * (1 - fs) * v[i, j] + ft * (1 - fs) * v[i + 1, j]
                + (1 - ft) * fs * v[i, j + 1] + ft * fs * v[i + 1, j + 1])


def title_curve(talent: np.ndarray, share: np.ndarray, delta: np.ndarray, n_talent: int = 40,
                n_share: int = 11, bw_talent: float | None = None, bw_share: float = 0.12) -> TitleCurve:
    """Kernel-smoothed mean title swing by talent and games share (Gaussian
    weights), then made non-decreasing in each direction: a better or
    healthier player never adds fewer title odds."""
    talent, share, delta = (np.asarray(a, float) for a in (talent, share, delta))
    tg = np.quantile(talent, np.linspace(0.005, 0.995, n_talent))
    tg = np.unique(tg)
    sg = np.linspace(0.0, 1.0, n_share)
    bw_t = bw_talent if bw_talent is not None else 0.6 * talent.std() * len(talent) ** (-0.2) * 4
    wt = np.exp(-0.5 * ((talent[None] - tg[:, None]) / bw_t) ** 2)      # (nt, n)
    ws = np.exp(-0.5 * ((share[None] - sg[:, None]) / bw_share) ** 2)   # (ns, n)
    num = wt @ (ws * delta[None]).T
    den = wt @ ws.T
    v = num / np.maximum(den, 1e-12)
    v = np.maximum.accumulate(v, axis=0)
    v = np.maximum.accumulate(v, axis=1)
    return TitleCurve(tg, sg, v)
