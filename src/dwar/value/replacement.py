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
