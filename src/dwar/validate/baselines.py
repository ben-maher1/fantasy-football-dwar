"""The three CLAUDE.md baselines, each built only from seasons before the
one forecast:

- last: points per game in the player's most recent season.
- weighted3: points per game over the last three seasons, weighted 3/2/1
  from most recent, times games played.
- age_adjusted: last, plus the position's average year-over-year change at
  that age (a games-weighted quadratic in age fit on past seasons).

Rookies (no earlier NFL season) get, for all three, the average rookie
points per game at their position and draft round in past seasons.

For the log score each baseline is a normal distribution whose variance,
a + b / games, is fit on the same baseline's misses in the five seasons
before.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import norm

METHODS = ("last", "weighted3", "age_adjusted")
UNDRAFTED_ROUND = 8
SPREAD_SEASONS = 5
MIN_GAMES = 4
AGE_CENTER = 26.0


def _age_delta_coefs(history: pl.DataFrame) -> dict[str, np.ndarray]:
    """Per position, games-weighted quadratic fit of next-season minus this-
    season points per game on age, from consecutive seasons with MIN_GAMES+."""
    ps = history.filter(pl.col("games_played") >= MIN_GAMES)
    nxt = ps.select("player_id", "position", pl.col("season") - 1, ppg_next="fpts_per_game", g_next="games_played")
    pairs = ps.join(nxt, on=["player_id", "season", "position"]).drop_nulls("age_sept1")
    out = {}
    for (position,), p in pairs.group_by("position"):
        a = p["age_sept1"].to_numpy() - AGE_CENTER
        x = np.stack([np.ones_like(a), a, a**2], axis=1)
        y = (p["ppg_next"] - p["fpts_per_game"]).to_numpy()
        w = np.minimum(p["games_played"].to_numpy(), p["g_next"].to_numpy()).astype(float)
        out[position] = np.linalg.lstsq(x * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
    return out


def _rookie_means(history: pl.DataFrame, static: pl.DataFrame) -> pl.DataFrame:
    first = history.sort("season").group_by("player_id", maintain_order=True).first()
    return (
        first.filter(pl.col("games_played") >= MIN_GAMES)
        .join(static.select("player_id", "draft_round"), on="player_id", how="left")
        .with_columns(draft_round=pl.col("draft_round").fill_null(UNDRAFTED_ROUND))
        .group_by("position", "draft_round")
        .agg(rookie_ppg=(pl.col("fpts_per_game") * pl.col("games_played")).sum() / pl.col("games_played").sum())
    )


def baseline_points(
    history: pl.DataFrame, static: pl.DataFrame, targets: pl.DataFrame, season: int
) -> pl.DataFrame:
    """Point forecasts for `targets` (player_id, position) in `season` from
    `history` (player_seasons rows; only seasons before `season` are used)."""
    history = history.filter(pl.col("season") < season)
    played = history.filter(pl.col("games_played") > 0)
    last = played.sort("season").group_by("player_id", maintain_order=True).last().select(
        "player_id", last="fpts_per_game", last_age="age_sept1", last_season="season"
    )
    recent = played.filter(pl.col("season") >= season - 3).with_columns(
        w=(4 - (season - pl.col("season"))) * pl.col("games_played")
    )
    weighted = recent.group_by("player_id").agg(
        weighted3=(pl.col("w") * pl.col("fpts_per_game")).sum() / pl.col("w").sum()
    )
    coefs = _age_delta_coefs(history)
    rookies = _rookie_means(history, static)
    out = (
        targets.select("player_id", "position")
        .join(last, on="player_id", how="left")
        .join(weighted, on="player_id", how="left")
        .join(static.select("player_id", "draft_round"), on="player_id", how="left")
        .with_columns(draft_round=pl.col("draft_round").fill_null(UNDRAFTED_ROUND))
        .join(rookies, on=["position", "draft_round"], how="left")
    )
    a = out["last_age"].fill_null(AGE_CENTER).to_numpy() - AGE_CENTER
    delta = np.array([
        c[0] + c[1] * ai + c[2] * ai**2 if (c := coefs.get(p)) is not None else 0.0
        for p, ai in zip(out["position"].to_list(), a)
    ])
    rookie = pl.col("last").is_null()
    return out.with_columns(age_delta=pl.Series(delta)).select(
        "player_id", "position",
        is_rookie=rookie,
        last=pl.when(rookie).then("rookie_ppg").otherwise("last"),
        weighted3=pl.when(rookie).then("rookie_ppg").otherwise(pl.coalesce("weighted3", "last")),
        age_adjusted=pl.when(rookie).then("rookie_ppg").otherwise(pl.col("last") + pl.col("age_delta")),
    )


def _spread_coefs(history: pl.DataFrame, static: pl.DataFrame, season: int) -> dict[tuple[str, str], tuple[float, float]]:
    """(method, position) -> (a, b) with miss variance a + b / games, from
    each method's misses in the SPREAD_SEASONS seasons before `season`."""
    misses = []
    for t in range(season - SPREAD_SEASONS, season):
        actual = history.filter((pl.col("season") == t) & (pl.col("games_played") >= MIN_GAMES))
        if actual.is_empty():
            continue
        f = baseline_points(history, static, actual.select("player_id", "position"), t)
        misses.append(f.join(actual.select("player_id", "position", "games_played", "fpts_per_game"), on=["player_id", "position"]))
    out = {}
    if not misses:
        return out
    m = pl.concat(misses)
    for method in METHODS:
        for (position,), p in m.drop_nulls(method).group_by("position"):
            sq = ((p["fpts_per_game"] - p[method]) ** 2).to_numpy()
            x = np.stack([np.ones(len(sq)), 1.0 / p["games_played"].to_numpy()], axis=1)
            a, b = np.linalg.lstsq(x, sq, rcond=None)[0]
            out[(method, position)] = (max(a, 0.25), max(b, 0.0))
    return out


def baseline_forecasts(
    history: pl.DataFrame, static: pl.DataFrame, actual: pl.DataFrame, season: int
) -> pl.DataFrame:
    """Baseline points and log scores for each player-season in `actual`
    (player_id, position, games_played, fpts_per_game)."""
    points = baseline_points(history, static, actual.select("player_id", "position"), season)
    spread = _spread_coefs(history, static, season)
    f = actual.join(points, on=["player_id", "position"])
    cols = []
    for method in METHODS:
        ab = np.array([spread.get((method, p), (np.nan, np.nan)) for p in f["position"].to_list()])
        sd = np.sqrt(ab[:, 0] + ab[:, 1] / f["games_played"].to_numpy())
        ls = norm.logpdf(f["fpts_per_game"].to_numpy(), f[method].to_numpy(), sd)
        cols += [pl.Series(f"{method}_sd", sd), pl.Series(f"{method}_log_score", ls)]
    return f.with_columns(cols)
