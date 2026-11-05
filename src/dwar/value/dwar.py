"""From fitted talent to DWAR: the discounted sum of future wins above
replacement, with its full posterior distribution.

Per posterior draw and player:

    DWAR = sum_k discount^k * alive_k * games_k * wins_per_game(max(talent_k - repl, 0))

over k = 0 (the rest of the current season) through `horizon` full seasons
after it. A season below replacement is worth zero: a manager benches or
drops that player. `alive_k` follows the survival model from the player's
latest season; `games_k` is the share of games played (availability model)
times the season's games, or the team's remaining games in the current
season.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import polars as pl

from dwar.config import LeagueFormat, ValueConfig
from dwar.features.player_tables import PlayerTables
from dwar.features.situation import Covariates
from dwar.model.fit import FitResult
from dwar.model.talent import for_position
from dwar.features.injuries import GROUPS as INJURY_GROUPS
from dwar.value.availability import (
    AvailabilityDraws, fit_availability, injury_terms, season_games, status_effects, status_multiplier,
)
from dwar.value.projection import PositionDraws, future_situation, project_talent
from dwar.value.replacement import replacement_level
from dwar.value.short_games import ROLE_FACTOR, ShortGames, short_games
from dwar.value.spread import widen
from dwar.value.wins import team_score_sd, wins_per_game

PERCENTILES = (5, 25, 50, 75, 95)
OUTPUT_COLUMNS = (
    "player_id", "name", "position", "age", "team",
    "dwar_mean", *(f"dwar_p{p:02d}" for p in PERCENTILES),
)


@dataclass(frozen=True)
class SeasonClock:
    """Where `as_of` falls in the NFL calendar."""

    current: int                      # first season whose regular season has not ended
    complete: tuple[int, ...]         # seasons whose regular season has ended
    games: int                        # games per team in a full season
    remaining: dict[str, int]         # team -> regular-season games left in `current`
    remaining_default: int            # for a player whose team is unknown


def season_clock(schedules: pl.DataFrame, as_of: date) -> SeasonClock:
    by_season = season_games(schedules).sort("season")
    complete = by_season.filter(pl.col("last_gameday") < as_of)
    if complete.is_empty():
        raise ValueError(f"no completed season before {as_of}")
    games = int(complete["games"][-1])
    upcoming = by_season.filter(pl.col("last_gameday") >= as_of)
    if upcoming.is_empty():  # next schedule not published yet: a full season remains
        return SeasonClock(int(complete["season"][-1]) + 1, tuple(complete["season"]), games, {}, games)
    current = int(upcoming["season"][0])
    left = schedules.filter((pl.col("season") == current) & (pl.col("gameday") >= as_of))
    remaining = (
        pl.concat([left.select(team="home_team"), left.select(team="away_team")])
        .group_by("team").agg(n=pl.len())
    )
    remaining_map = dict(zip(remaining["team"].to_list(), remaining["n"].to_list()))
    default = int(np.median(list(remaining_map.values()))) if remaining_map else 0
    return SeasonClock(current, tuple(complete["season"]), games, remaining_map, default)


def draw_indices(n_total: int, n: int) -> np.ndarray:
    """Evenly spaced posterior draws (deterministic, so reruns match)."""
    return np.unique(np.linspace(0, n_total - 1, min(n, n_total)).round().astype(int))


@dataclass(frozen=True)
class InjuryInputs:
    """Injury history as of the run (features/injuries.py)."""

    seasons: pl.DataFrame                    # injury_seasons: weeks missed per player-season, by group
    status: pl.DataFrame                     # current_status for the current season
    effects: dict[tuple[str, int], float]    # status_effects: factor on remaining games

    def actual(self, player_ids: list[str], season: int) -> tuple[np.ndarray, np.ndarray]:
        """Weeks missed in `season` so far by group (players, groups) and in total."""
        cols = [f"missed_{g}" for g in INJURY_GROUPS]
        rows = pl.DataFrame({"player_id": player_ids}).join(
            self.seasons.filter(pl.col("season") == season).select("player_id", "missed", *cols),
            on="player_id", how="left",
        )
        groups = rows.select([pl.col(c).fill_null(0) for c in cols]).to_numpy().astype(float)
        return groups, rows["missed"].fill_null(0).to_numpy().astype(float)


@dataclass(frozen=True)
class ValueInputs:
    clock: SeasonClock
    draws: dict[str, PositionDraws]
    replacement: dict[str, np.ndarray]  # position -> (draws,)
    sigma: float                        # weekly team-score SD
    availability: AvailabilityDraws
    short: dict[str, ShortGames] = field(default_factory=dict)  # positions with short appearances
    situation: dict[str, Covariates] = field(default_factory=dict)  # positions with situation factors
    injury: InjuryInputs | None = None


def replacement_from_fits(
    fits: dict[str, FitResult], draws: dict[str, PositionDraws], fmt: LeagueFormat,
    seasons: tuple[int, ...], min_games: int,
) -> dict[str, np.ndarray]:
    """Replacement level per position and draw, averaged over `seasons`."""
    levels: dict[str, list[np.ndarray]] = {p: [] for p in fits}
    for season in seasons:
        cols, labels = [], []
        for p, r in fits.items():
            keys = r.data.ps_keys
            idx = np.flatnonzero((keys["season"] == season).to_numpy() & (keys["games_played"] >= min_games).to_numpy())
            cols.append(draws[p].talent[:, idx])
            labels += [p] * len(idx)
        if not labels:
            continue
        repl = replacement_level(np.concatenate(cols, axis=1), np.array(labels), dict(fmt.lineup), fmt.teams)
        for p, v in repl.items():
            levels[p].append(v)
    return {p: np.mean(v, axis=0) for p, v in levels.items() if v}


def value_inputs(
    fits: dict[str, FitResult], tables: PlayerTables, schedules: pl.DataFrame,
    fmt: LeagueFormat, cfg: ValueConfig, likelihood: str | Mapping[str, str], rng: np.random.Generator,
    min_plays: Mapping[str, int] | None = None, situation: dict[str, Covariates] | None = None,
    injury_weeks: pl.DataFrame | None = None,
) -> ValueInputs:
    """`injury_weeks` (features.injuries.injury_weeks, as of the run) turns on
    injury history in the availability models and the current-status cut."""
    clock = season_clock(schedules, tables.as_of)
    first = next(iter(fits.values()))
    n_total = first.posterior["posterior"].sizes["chain"] * first.posterior["posterior"].sizes["draw"]
    idx = draw_indices(n_total, cfg.draws)
    draws = {
        p: PositionDraws.from_posterior(
            r.data, r.posterior, for_position(likelihood, p), idx, fade_qb=p in cfg.qb_fade,
        )
        for p, r in fits.items()
    }
    recent = clock.complete[-cfg.replacement_seasons:]
    replacement = replacement_from_fits(fits, draws, fmt, recent, cfg.pool_min_games)
    sigma = team_score_sd(
        tables.player_weeks, tables.player_seasons, list(recent), dict(fmt.lineup), fmt.teams, cfg.pool_min_games
    )
    talent = pl.concat([r.talent.with_columns(position=pl.lit(p)) for p, r in fits.items()])
    injury = None
    if injury_weeks is not None:
        from dwar.features.injuries import current_status, injury_seasons

        injury = InjuryInputs(
            injury_seasons(injury_weeks), current_status(injury_weeks, clock.current),
            status_effects(injury_weeks, list(clock.complete)),
        )
    availability = fit_availability(
        talent, tables.player_seasons, schedules, tables.as_of, injury.seasons if injury else None,
        cfg.games_hinge_quantile, lagged=cfg.games_lagged_talent,
    ).draw(len(idx), rng)
    situation = situation or {}

    def roles(p: str) -> pl.DataFrame | None:
        cov = situation.get(p)
        return cov.roles(ROLE_FACTOR[p]) if cov is not None and p in cfg.short_by_role and p in ROLE_FACTOR else None

    short = {
        p: s for p, r in fits.items()
        if (s := short_games(
            tables.player_weeks, p, (min_plays or {}).get(p, 0), int(r.data.ps_keys["season"].min()), roles(p),
        ))
    }
    return ValueInputs(clock, draws, replacement, sigma, availability, short, situation, injury)


def _situation(
    position: str, r: FitResult, inputs: ValueInputs, latest: pl.DataFrame, seasons: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Projection shift and step scale from each player's known situation
    for the season after his latest fitted one."""
    cov = inputs.situation.get(position)
    if cov is None:
        return None, None
    nxt = latest.select("player_id", season=(pl.col("season") + 1).cast(pl.Int32))

    def aligned(df: pl.DataFrame, names: tuple[str, ...]) -> np.ndarray | None:
        if not names:
            return None
        rows = nxt.join(df.with_columns(pl.col("season").cast(pl.Int32)), on=["player_id", "season"], how="left")
        return rows.select([pl.col(n).fill_null(0.0) for n in names]).to_numpy().astype(float)

    return future_situation(
        inputs.draws[position], r.data, latest["ps_idx"].to_numpy(), latest["season"].to_numpy(), seasons,
        aligned(cov.level, r.data.season_x_names), aligned(cov.range, r.data.season_vol_x_names),
    )


def _next_roles(position: str, inputs: ValueInputs, latest: pl.DataFrame) -> list[float | None] | None:
    """Each player's preseason role for the season after his latest one,
    held for every season valued."""
    cov = inputs.situation.get(position)
    short = inputs.short.get(position)
    if cov is None or short is None or not short.role_base or position not in ROLE_FACTOR:
        return None
    roles = cov.roles(ROLE_FACTOR[position])
    nxt = latest.select("player_id", season=(pl.col("season") + 1).cast(pl.Int32))
    return nxt.join(roles.with_columns(pl.col("season").cast(pl.Int32)), on=["player_id", "season"], how="left")["role"].to_list()


class _InjuryPath:
    """Injury weeks along one projection: actual weeks for past and current
    seasons, projected ones (from games missed) for later seasons, turned into
    each season's `injury_terms`."""

    def __init__(
        self, inputs: ValueInputs, position: str, player_ids: list[str], remaining: np.ndarray,
        shape: tuple[int, int],
    ) -> None:
        self.on = inputs.injury is not None and inputs.availability.injury
        self.current = inputs.clock.current
        self.remaining = remaining
        self.shape = shape
        self.share = inputs.availability.injury_share.get(position, 0.0)
        self.season_games = inputs.availability.season_games
        self.multiplier = np.ones(len(player_ids))
        self.groups: dict[int, np.ndarray] = {}   # season -> (players, groups) actual weeks
        self.total: dict[int, np.ndarray] = {}    # season -> (draws, players) weeks, actual + projected
        if not self.on:
            return
        injury = inputs.injury
        assert injury is not None
        for season in (self.current - 2, self.current - 1, self.current):
            g, t = injury.actual(player_ids, season)
            self.groups[season] = g
            self.total[season] = np.broadcast_to(t, shape).astype(float)
        self.multiplier = status_multiplier(injury.status, remaining, injury.effects, player_ids)

    def terms(self, season: int) -> np.ndarray | None:
        if not self.on:
            return None
        groups = self.groups.get(season - 1, np.zeros((self.shape[1], len(INJURY_GROUPS))))
        last_total = self.total.get(season - 1, np.zeros(self.shape))
        extra = last_total - groups.sum(axis=1)[None, :]   # projected weeks, body part unknown
        prior = self.total.get(season - 2, np.zeros(self.shape))
        return injury_terms(np.broadcast_to(groups, (*self.shape, groups.shape[1])), prior, extra)

    def record(self, season: int, share: np.ndarray) -> None:
        """Add a projected season's injury weeks: games missed to injury
        (`share` of the season's games played)."""
        if not self.on:
            return
        if season == self.current:
            healthy_missed = self.remaining * (1.0 - share) * self.share
            status_missed = self.remaining * share * (1.0 - self.multiplier)
            self.total[season] = self.total[season] + healthy_missed + status_missed
        else:
            self.total[season] = self.season_games * (1.0 - share) * self.share


def position_dwar(
    position: str, r: FitResult, inputs: ValueInputs, tables: PlayerTables,
    fmt: LeagueFormat, cfg: ValueConfig, rng: np.random.Generator,
) -> tuple[pl.DataFrame, np.ndarray]:
    """DWAR draws (draws, players) for one position's active players."""
    clock, pd_ = inputs.clock, inputs.draws[position]
    latest = (
        r.data.ps_keys.with_row_index("ps_idx")
        .with_columns(age_c=pl.Series(r.data.ps_age_c))
        .sort("season").group_by("player_id", maintain_order=True).last()
        .filter(pl.col("season") >= clock.current - 1)
        .join(tables.player_static.select("player_id", "name", "team", "age_at_as_of"), on="player_id", how="left")
        .sort("player_id")
    )
    if latest.is_empty():
        return latest, np.zeros((len(pd_.beta1), 0))
    last_season = latest["season"].to_numpy()
    last_age = latest["age_c"].to_numpy() + 26.0
    seasons = np.arange(clock.current - 1, clock.current + cfg.horizon_seasons + 1)
    shift, vol = _situation(position, r, inputs, latest, seasons)
    talent = project_talent(pd_, latest["ps_idx"].to_numpy(), last_season, last_age, seasons, rng, shift, vol)
    remaining = np.array([clock.remaining.get(t, clock.remaining_default) for t in latest["team"].to_list()])
    if pd_.form_dev is not None:
        # The rest of the current season also carries recent form, fading game by game.
        cur = seasons == clock.current
        in_cur = last_season == clock.current
        if cur.any() and in_cur.any():
            ps_idx = latest["ps_idx"].to_numpy()
            form = pd_.form_rest(ps_idx, remaining)
            with_form = pd_.expected_points(pd_.latent[:, ps_idx] + form)
            j = int(np.flatnonzero(cur)[0])
            talent[:, :, j] = np.where(in_cur[None], with_form, talent[:, :, j])
    talent = widen(talent, cfg.spread_factor.get(position, 1.0), axis=0)
    ages = last_age[None, :, None] + (seasons[None, None, :] - last_season[None, :, None])
    ages = np.broadcast_to(ages, talent.shape)

    av = inputs.availability
    repl = inputs.replacement[position][:, None]
    short = inputs.short.get(position)
    short_rate = short.rates(latest["player_id"].to_list(), _next_roles(position, inputs, latest))[None, :] if short else 0.0
    short_wins = wins_per_game(np.maximum(short.mean_points - repl, 0.0), inputs.sigma) if short else 0.0
    alive = np.ones(talent.shape[:2])
    total = np.zeros(talent.shape[:2])
    inj = _InjuryPath(inputs, position, latest["player_id"].to_list(), remaining, talent.shape[:2])
    for j, season in enumerate(seasons[1:], start=1):
        terms = inj.terms(season)
        # Survive from the previous season into this one, unless the player
        # has already played in it (or later).
        p_survive = av.survival_prob(position, ages[:, :, j - 1], talent[:, :, j - 1])
        survives = rng.random(alive.shape) < p_survive
        alive = np.where(season <= last_season, alive, alive * survives)
        share = av.games_share(position, ages[:, :, j], talent[:, :, j], rng, terms)
        if season == clock.current:
            games = share * inj.multiplier * remaining
        else:
            games = share * av.season_games
        inj.record(season, share)
        full_wins = wins_per_game(np.maximum(talent[:, :, j] - repl, 0.0), inputs.sigma)
        wins = games * ((1.0 - short_rate) * full_wins + short_rate * short_wins)
        total += fmt.discount ** (season - clock.current) * alive * wins
    return latest, total


def dwar_table(
    fits: dict[str, FitResult], tables: PlayerTables, schedules: pl.DataFrame,
    fmt: LeagueFormat, cfg: ValueConfig, likelihood: str | Mapping[str, str], seed: int,
    min_plays: Mapping[str, int] | None = None, situation: dict[str, Covariates] | None = None,
    injury_weeks: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, dict]:
    """The DWAR output table and a summary of the inputs behind it."""
    rng = np.random.default_rng(seed)
    inputs = value_inputs(fits, tables, schedules, fmt, cfg, likelihood, rng, min_plays, situation, injury_weeks)
    frames = []
    for p, r in fits.items():
        players, d = position_dwar(p, r, inputs, tables, fmt, cfg, rng)
        if players.is_empty():
            continue
        q = np.percentile(d, PERCENTILES, axis=0)
        frames.append(players.select(
            "player_id", "name", pl.lit(p).alias("position"), age="age_at_as_of", team="team",
            last_season="season",
        ).with_columns(
            dwar_mean=pl.Series(d.mean(axis=0)),
            **{f"dwar_p{pc:02d}": pl.Series(q[k]) for k, pc in enumerate(PERCENTILES)},
        ))
    table = (
        pl.concat(frames)
        # A player fitted at two positions (a switch) keeps the latest one.
        .sort(["last_season", "dwar_mean"], descending=True)
        .unique("player_id", keep="first", maintain_order=True)
        .select(OUTPUT_COLUMNS)
        .sort("dwar_mean", descending=True)
    )
    summary = {
        "current_season": inputs.clock.current,
        "seasons_valued": [inputs.clock.current + k for k in range(cfg.horizon_seasons + 1)],
        "remaining_games_default": inputs.clock.remaining_default,
        "team_score_sd": round(inputs.sigma, 3),
        "points_per_win": round(float(1.0 / wins_per_game(1.0, inputs.sigma)), 2),
        "replacement_level": {
            p: {"mean": round(float(v.mean()), 3), "p05": round(float(np.percentile(v, 5)), 3),
                "p95": round(float(np.percentile(v, 95)), 3)}
            for p, v in inputs.replacement.items()
        },
        "draws": len(next(iter(inputs.draws.values())).beta1),
        "rookie_draft_picks": "unavailable in v1",
        "injury_history": inputs.injury is not None,
        "games_talent_knot": {p: round(k, 2) for p, k in inputs.availability.games_knot.items()},
    }
    return table, summary
