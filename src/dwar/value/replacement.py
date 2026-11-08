"""Replacement level from league lineup settings.

Starters are allocated the way managers fill lineups: each fixed slot takes
the best players at its position, then each flex slot type takes the best
players left among its eligible positions. Replacement at a position is
the best player left after every starting slot in the league is filled
(the (N+1)th player, N = starters used at that position).

Everything is vectorized over posterior draws: `talent` is (draws, players).
"""

from __future__ import annotations

import numpy as np

# Which positions each flex-type lineup slot accepts. Fixed slots are named
# by their position. Flex slots are filled in this order, narrowest first.
FLEX_ELIGIBLE: dict[str, tuple[str, ...]] = {
    "FLEX": ("RB", "WR", "TE"),
    "SUPERFLEX": ("QB", "RB", "WR", "TE"),
}


def starters_used(
    talent: np.ndarray, positions: np.ndarray, lineup: dict[str, int], teams: int
) -> dict[str, np.ndarray]:
    """Number of starters at each position, per draw, after optimal
    allocation. `positions` labels the columns of `talent`."""
    talent = np.atleast_2d(talent)
    pos_names = tuple(dict.fromkeys(positions.tolist()))
    unknown = set(lineup) - set(pos_names) - set(FLEX_ELIGIBLE)
    if unknown:
        raise ValueError(f"unknown lineup slots {sorted(unknown)}; flex slots: {sorted(FLEX_ELIGIBLE)}")
    # Each position's talent sorted best first, padded with -inf.
    width = max((positions == p).sum() for p in pos_names)
    ranked = {}
    for p in pos_names:
        t = -np.sort(-talent[:, positions == p], axis=1)
        ranked[p] = np.pad(t, ((0, 0), (0, width - t.shape[1])), constant_values=-np.inf)
    n_draws = talent.shape[0]
    used = {p: np.minimum(teams * lineup.get(p, 0), (positions == p).sum()) * np.ones(n_draws, int) for p in pos_names}
    rows = np.arange(n_draws)
    for slot in sorted((s for s in lineup if s in FLEX_ELIGIBLE), key=lambda s: len(FLEX_ELIGIBLE[s])):
        eligible = [p for p in FLEX_ELIGIBLE[slot] if p in ranked]
        for _ in range(teams * lineup[slot]):
            # The best remaining player among eligible positions takes the slot.
            heads = np.stack([ranked[p][rows, np.minimum(used[p], width - 1)] for p in eligible])
            heads[np.array([used[p] >= width for p in eligible])] = -np.inf
            best = heads.argmax(axis=0)
            filled = np.isfinite(heads.max(axis=0))
            for k, p in enumerate(eligible):
                used[p] = used[p] + ((best == k) & filled)
    return used


def replacement_level(
    talent: np.ndarray, positions: np.ndarray, lineup: dict[str, int], teams: int
) -> dict[str, np.ndarray]:
    """Talent of the best unused player at each position, per draw. A
    position with no player left has replacement level 0."""
    talent = np.atleast_2d(talent)
    used = starters_used(talent, positions, lineup, teams)
    out = {}
    for p, n in used.items():
        t = -np.sort(-talent[:, positions == p], axis=1)
        idx = np.minimum(n, t.shape[1] - 1)
        level = t[np.arange(t.shape[0]), idx]
        out[p] = np.where(n < t.shape[1], level, 0.0)
    return out


# --- Waiver-wire baseline with lineup weeks (deep-roster leagues) ---------

def waiver_level(
    talent: np.ndarray, positions: np.ndarray, rostered: dict[str, int], teams: int
) -> dict[str, np.ndarray]:
    """Talent of the best player left on waivers at each position, per draw:
    the (teams * rostered[p] + 1)th best. 0 when the pool runs out first."""
    talent = np.atleast_2d(talent)
    out = {}
    for p in dict.fromkeys(positions.tolist()):
        t = -np.sort(-talent[:, positions == p], axis=1)
        n = teams * rostered.get(p, 0)
        out[p] = t[:, n] if n < t.shape[1] else np.zeros(t.shape[0])
    return out


def _best_lineup(avail: dict[str, np.ndarray], talent: dict[str, np.ndarray], lineup: dict[str, int]) -> dict[str, np.ndarray]:
    """One team's starters for one week: boolean masks per position over its
    players (sorted best first), fixed slots then flex slots narrowest first."""
    start = {p: np.zeros(len(t), bool) for p, t in talent.items()}
    for p, n in lineup.items():
        if p in FLEX_ELIGIBLE or p not in talent:
            continue
        idx = np.flatnonzero(avail[p])[:n]
        start[p][idx] = True
    for slot in sorted((s for s in lineup if s in FLEX_ELIGIBLE), key=lambda s: len(FLEX_ELIGIBLE[s])):
        for _ in range(lineup[slot]):
            best, best_p, best_i = -np.inf, None, -1
            for p in FLEX_ELIGIBLE[slot]:
                if p not in talent:
                    continue
                free = np.flatnonzero(avail[p] & ~start[p])
                if len(free) and talent[p][free[0]] > best:
                    best, best_p, best_i = talent[p][free[0]], p, free[0]
            if best_p is not None:
                start[best_p][best_i] = True
    return start


def start_share(
    talent: dict[str, np.ndarray], lineup: dict[str, int], teams: int, rostered: dict[str, int],
    p_play: dict[str, float], rng: np.random.Generator, weeks: int = 2000,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """How often a rostered player is in his team's lineup in a week he plays.

    `talent[p]` holds league-wide talent (PPG) at a position. The best
    teams * rostered[p] are dealt to teams in a snake within the position,
    each plays a given week with probability p_play[p] (byes and injuries),
    and each team starts its best available lineup. Returns, per position,
    (talent ascending, share) to interpolate any player's share from his
    talent."""
    roster: list[dict[str, np.ndarray]] = [{} for _ in range(teams)]
    pools = {}
    for p, n in rostered.items():
        if p not in talent or n == 0:
            continue
        t = np.sort(np.asarray(talent[p]))[::-1][: teams * n]
        pools[p] = t
        team_of = np.array([(r % teams) if (r // teams) % 2 == 0 else teams - 1 - (r % teams) for r in range(len(t))])
        for k in range(teams):
            roster[k][p] = np.flatnonzero(team_of == k)  # league ranks, best first
    starts = {p: np.zeros(len(t)) for p, t in pools.items()}
    plays = {p: np.zeros(len(t)) for p, t in pools.items()}
    for _ in range(weeks):
        for k in range(teams):
            ranks = roster[k]
            tal = {p: pools[p][r] for p, r in ranks.items()}
            avail = {p: rng.random(len(r)) < p_play.get(p, 1.0) for p, r in ranks.items()}
            st = _best_lineup(avail, tal, lineup)
            for p, r in ranks.items():
                plays[p][r] += avail[p]
                starts[p][r] += st[p]
    out = {}
    for p, t in pools.items():
        share = starts[p] / np.maximum(plays[p], 1)
        order = np.argsort(t)
        out[p] = (t[order], share[order])
    return out


def share_at(grid: tuple[np.ndarray, np.ndarray], talent: np.ndarray) -> np.ndarray:
    """A player's lineup share from his talent (see `start_share`): smoothed
    by a running mean over neighbouring ranks, 1 above the best rostered
    player's talent, the lowest rostered player's share below it."""
    t, s = grid
    k = max(1, len(s) // 24)
    smooth = np.convolve(np.pad(s, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
    smooth = np.maximum.accumulate(smooth)  # more talent never starts less
    return np.interp(talent, t, smooth, left=smooth[0], right=1.0)
