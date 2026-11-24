"""Teammate and opponent links in the league title-odds sim (ben, 2026-11-24;
Claude Doc "Plan: stacks and teammate competition in title odds").

Each player's weekly points keep their own distribution (so DWAR is
unchanged), but within an NFL game the draws are linked: a QB's big week
tends to be his receivers' big week too, backfield mates take points from
each other, and opposing QBs rise and fall together in shootouts. When a
team's QB or RB1 sits, his teammates' expected points move.

Measured on 1999-2025 regular-season weeks (/mnt/project-files/dwar-wip/
stacks): each player-week's normal score under his own pre-game predictive
distribution, correlated by role within a team (QB-WR1 0.39, QB-TE1 0.28,
RB1-RB2 -0.09, ...) and across the two teams of a game (QB-QB 0.15). ben
chose to put QB-pass catcher and same-position teammate links in at the
measured size; opponent links and the QB-out / RB1-out effects passed a
1999-2019 vs 2020-25 check.

  role    a player's place on his NFL team by expected points (managers'
          view): QB (top QB), RB1-2, WR1-3, TE1; others are unlinked
  draw    per NFL game and week, 14 linked normals (7 roles x 2 teams), each
          turned into points through that player's own weekly distribution
  handoff when the QB1 is out, QB2 takes his link slot and he and his
          teammates' expected points scale by (1 + effect); RB1 out lifts RB2
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr, stdtrit

from dwar.model.talent import nonzero_prob

ROLES = ("QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE1")
SLOTS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}
QB2 = len(ROLES)          # handoff-only role: the backup QB


def _key(a: str, b: str) -> str:
    return f"{a}-{b}"


def game_cholesky(mate: dict[str, float], opp: dict[str, float]) -> np.ndarray:
    """Cholesky factor of the (14, 14) correlation of one NFL game's role
    draws: the first 7 are one team's roles, the next 7 the other's. `mate`
    and `opp` map "A-B" (either order) to a correlation; missing pairs are 0."""
    R = len(ROLES)
    M = np.eye(2 * R)
    for i, a in enumerate(ROLES):
        for j, b in enumerate(ROLES):
            if i != j:
                r = mate.get(_key(a, b), mate.get(_key(b, a), 0.0))
                M[i, j] = M[R + i, R + j] = r
            r = opp.get(_key(a, b), opp.get(_key(b, a), 0.0))
            M[i, R + j] = M[R + j, i] = r
    w, v = np.linalg.eigh(M)
    if w.min() < 1e-6:                        # nearest valid correlation (eigenvalue floor)
        M = (v * np.maximum(w, 1e-6)) @ v.T
        d = np.sqrt(np.diag(M))
        M = M / d[:, None] / d[None, :]
    return np.linalg.cholesky(M)


def roles(team: np.ndarray, pos: np.ndarray, known: np.ndarray) -> np.ndarray:
    """(P,) role index into ROLES (QB2 for the backup QB), -1 unlinked.
    `team`: NFL team index, -1 unknown. Ranked by `known` within team and
    position."""
    out = np.full(len(team), -1)
    for p, n in SLOTS.items():
        sel = np.flatnonzero((pos == p) & (team >= 0))
        if not len(sel):
            continue
        order = sel[np.lexsort((-known[sel], team[sel]))]
        t = team[order]
        rank = np.arange(len(order)) - np.searchsorted(t, t)     # 0-based rank within team
        first = ROLES.index("QB" if p == "QB" else f"{p}1")
        ok = rank < n
        out[order[ok]] = first + rank[ok]
        if p == "QB":
            out[order[rank == 1]] = QB2
    return out


def holders(team: np.ndarray, role: np.ndarray, n_teams: int) -> np.ndarray:
    """(n_teams, roles incl. QB2) pool index of each role's player, -1 none."""
    h = np.full((n_teams, len(ROLES) + 1), -1)
    sel = np.flatnonzero(role >= 0)
    h[team[sel], role[sel]] = sel
    return h


def handoff_mult(avail: np.ndarray, team: np.ndarray, role: np.ndarray, n_teams: int,
                 effects: dict[str, dict[str, float]]) -> np.ndarray:
    """(P, weeks) factor on expected points from teammates being out.
    `effects`: out role -> {teammate role (or "QB2"): effect}."""
    mult = np.ones(avail.shape)
    h = holders(team, role, n_teams)
    names = list(ROLES) + ["QB2"]
    for out_role, eff in effects.items():
        who = h[:, names.index(out_role)]
        has = who >= 0
        out = np.zeros((n_teams, avail.shape[1]), bool)
        out[has] = ~avail[who[has]]
        for r, e in eff.items():
            mate = h[:, names.index(r)]
            ok = (mate >= 0) & has
            mult[mate[ok]] *= np.where(out[ok], 1.0 + e, 1.0)
    return mult


def game_draws(team: np.ndarray, role: np.ndarray, qb_out: np.ndarray, opp: np.ndarray, L: np.ndarray,
               rng: np.random.Generator) -> np.ndarray:
    """(P, weeks) standard normals, linked within each NFL game by role.
    `opp`: (weeks, n_teams) opponent team index (-1 = no game: unlinked).
    `qb_out`: (n_teams, weeks) True when the team's QB1 is out, so QB2
    takes the QB link."""
    P, W = len(team), opp.shape[0]
    z = rng.standard_normal((P, W))
    R = len(ROLES)
    n_teams = opp.shape[1]
    linked = np.flatnonzero((role >= 0) & (team >= 0))
    for w in range(W):
        o = opp[w]
        a = np.flatnonzero((o > np.arange(n_teams)) & (o >= 0))   # each game once, by its lower team
        if not len(a):
            continue
        g = np.full(n_teams, -1)
        side = np.zeros(n_teams, int)
        g[a], g[o[a]] = np.arange(len(a)), np.arange(len(a))
        side[o[a]] = 1
        Zg = rng.standard_normal((len(a), 2 * R)) @ L.T
        t = team[linked]
        rl = role[linked]
        qb2_on = (rl == QB2) & qb_out[t, w]
        slot = np.where(qb2_on, 0, rl)
        ok = (g[t] >= 0) & (slot < R)
        idx = linked[ok]
        z[idx, w] = Zg[g[t[ok]], side[t[ok]] * R + slot[ok]]
    return z


def weekly_points(center: np.ndarray, params: dict, z: np.ndarray) -> np.ndarray:
    """Weekly points for standard normal `z` under the sqrt-hurdle weekly
    distribution (the same one `simulate_weeks` draws from): zero below the
    hurdle, otherwise (center + sigma * t)^2 at that quantile."""
    u = ndtr(z)
    p = nonzero_prob(center, params["h0"], params["h1"], np)
    v = np.clip((u - (1.0 - p)) / np.maximum(p, 1e-12), 1e-9, 1 - 1e-9)
    t = params["sigma"] * stdtrit(params["nu"], v)
    return np.where(u < 1.0 - p, params.get("zero_value", 0.0), (center + t) ** 2)


def random_opponents(n_teams: int, weeks: int, rng: np.random.Generator) -> np.ndarray:
    """(weeks, n_teams) a random pairing of NFL teams each week (later
    seasons, whose schedules are not known)."""
    out = np.empty((weeks, n_teams), int)
    for w in range(weeks):
        perm = rng.permutation(n_teams)
        a, b = perm[0::2], perm[1::2]
        out[w, a], out[w, b] = b, a
    return out
