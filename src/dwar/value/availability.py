"""Games played and career survival, estimated from history.

Both are binomial logistic regressions, one per position, on age (centered
at 26, with a square term) and talent (expected points per game):

- games: games played out of the season's games, among players who played
  at least once. Overdispersion (injury seasons cluster) is kept: a
  projected season's share of games is drawn from a Beta whose spread
  matches the observed Pearson dispersion.
- survival: whether a player who played in season s plays in season s + 1.

Training rows come only from seasons whose regular season ended before
`as_of`, so a fit never sees an outcome that was not yet known.
Coefficient uncertainty is carried by drawing from the normal
approximation to each fit.

Two refinements of the games model, each kept because it beat the plain
model out of sample (validate/injury.py):

- talent hinge: playing time stops rising with talent above a knot (the
  position's 75th percentile of talent among training rows). A straight
  line in talent credited elite players with nearly every game, which
  history does not show (top RBs average about 13 of 17).
- lagged talent: games in season s are paired with talent from s - 1, the
  way a projection uses them (see `availability_rows`).
- injury history: weeks missed to injury in seasons t - 1 and t - 2 for
  the season t being predicted. Body-part terms were tested and did not
  help, so only totals are used by default (`injury_groups` turns the body
  parts back on, as deviations from the total with a tighter prior). With
  injury terms, games rows start in the first season with two seasons of
  reports. Survival uses neither: both made its forecasts worse.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from dwar.features.injuries import FIRST_SEASON as INJURY_FIRST_SEASON
from dwar.features.injuries import GROUPS

AGE_CENTER = 26.0
BASE_TERMS = 4
HINGE_QUANTILE = 0.75       # games model: talent knot, as a quantile of training talent
INJURY_WEEKS_UNIT = 4.0     # injury terms are in units of 4 weeks missed
INJURY_WEEKS_CAP = 17.0     # a season counts at most this many weeks
BASE_RIDGE = 0.01
GROUP_RIDGE = 16.0          # prior sd 0.25 on a group's deviation per 4 weeks
INJURY_TOTALS = ("injury_last", "injury_prior")
INJURY_GROUP_TERMS = tuple(f"injury_last_{g}" for g in GROUPS)
# Draft terms (validate/draft_avail.py, 2026-11-10): for career years 1-4, a
# flag and the flag times log(pick / 64), undrafted counted as pick 260.
EARLY_YEARS = 4
PICK_CENTER = 64.0
UNDRAFTED_PICK = 260.0


def draft_terms(career_year: np.ndarray, pick: np.ndarray) -> np.ndarray:
    """(..., 2) draft terms for the season a player is in his `career_year`."""
    cy = np.asarray(career_year, float)
    early = ((cy >= 1) & (cy <= EARLY_YEARS)).astype(float)
    p = np.where(np.isnan(np.asarray(pick, float)), UNDRAFTED_PICK, pick)
    return np.stack([early, early * np.log(p / PICK_CENTER)], axis=-1)


def _draft_cols(df: pl.DataFrame, static: pl.DataFrame, season_expr: pl.Expr) -> np.ndarray:
    """Draft terms for rows of `df` (player_id, plus the columns `season_expr`
    reads) in the season `season_expr` gives. Undrafted players' career year
    counts from their first season played."""
    s = static.select("player_id", "draft_year", "draft_pick", "first_season")
    j = df.select("player_id", target=season_expr).join(s, on="player_id", how="left", maintain_order="left")
    cy = j.select((pl.col("target") - pl.coalesce("draft_year", "first_season") + 1).cast(pl.Float64).fill_null(99.0))
    return draft_terms(cy.to_series().to_numpy(), j["draft_pick"].cast(pl.Float64).to_numpy())


def _design(
    age: np.ndarray, talent: np.ndarray, injury: np.ndarray | None = None, knot: float | None = None,
) -> np.ndarray:
    """Columns: 1, age - 26, its square, talent, [talent above `knot`], [injury terms]."""
    a = np.asarray(age, float) - AGE_CENTER
    t = np.asarray(talent, float)
    cols = [np.ones_like(a), a, a**2, t]
    if knot is not None:
        cols.append(np.maximum(t - knot, 0.0))
    x = np.stack(cols, axis=-1)
    if injury is None:
        return x
    return np.concatenate([x, np.broadcast_to(injury, (*x.shape[:-1], injury.shape[-1]))], axis=-1)


def injury_terms(
    last_groups: np.ndarray, prior_total: np.ndarray, extra_last: np.ndarray | float = 0.0, groups: bool = False,
) -> np.ndarray:
    """Injury columns for predicting season t: `last_groups` (..., len(GROUPS))
    are weeks missed in t - 1 by group, `prior_total` (...) weeks in t - 2,
    and `extra_last` (...) projected weeks in t - 1 with no known body part.
    Returns the two totals, then (with `groups`) the t - 1 weeks by group."""
    last_groups = np.minimum(np.asarray(last_groups, float), INJURY_WEEKS_CAP)
    total = np.minimum(last_groups.sum(axis=-1) + extra_last, INJURY_WEEKS_CAP)
    prior = np.minimum(np.asarray(prior_total, float), INJURY_WEEKS_CAP)
    cols = [total[..., None], np.broadcast_to(prior, total.shape)[..., None]]
    if groups:
        cols.append(np.broadcast_to(last_groups, (*total.shape, last_groups.shape[-1])))
    return np.concatenate(cols, axis=-1) / INJURY_WEEKS_UNIT


def ridge_vector(n_terms: int, n_groups: int = 0) -> np.ndarray:
    """Prior precision per design column: weak everywhere except the last
    `n_groups` columns (body-part deviations)."""
    r = np.full(n_terms, BASE_RIDGE)
    if n_groups:
        r[-n_groups:] = GROUP_RIDGE
    return r


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


@dataclass(frozen=True)
class Logistic:
    coef: np.ndarray        # (k,)
    cov: np.ndarray         # (k, k)
    dispersion: float = 1.0  # Pearson chi2 / df; 1 = binomial

    def draw_coefs(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.multivariate_normal(self.coef, self.cov, size=n)


def fit_logistic(
    x: np.ndarray, successes: np.ndarray, trials: np.ndarray, ridge: float | np.ndarray = BASE_RIDGE
) -> Logistic:
    """Binomial logistic regression by Newton's method, with a weak Normal(0, 1/sqrt(ridge))
    prior on each coefficient so a position where everyone plays every game (no
    variation) still gives a finite fit."""
    successes, trials = np.asarray(successes, float), np.asarray(trials, float)
    ridge = np.broadcast_to(np.asarray(ridge, float), (x.shape[1],))
    prior = np.diag(ridge)
    beta = np.zeros(x.shape[1])
    for _ in range(200):
        p = _sigmoid(x @ beta)
        w = trials * p * (1 - p)
        hess = x.T @ (w[:, None] * x) + prior
        step = np.linalg.solve(hess, x.T @ (successes - trials * p) - ridge * beta)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    else:
        raise RuntimeError("logistic regression did not converge")
    p = _sigmoid(x @ beta)
    w = trials * p * (1 - p)
    cov = np.linalg.inv(x.T @ (w[:, None] * x) + prior)
    resid = (successes - trials * p) / np.sqrt(np.maximum(w, 1e-12))
    dispersion = float(np.sum(resid**2) / max(len(successes) - x.shape[1], 1))
    return Logistic(coef=beta, cov=cov * max(dispersion, 1.0), dispersion=dispersion)


@dataclass(frozen=True)
class AvailabilityModel:
    games: dict[str, Logistic]
    survival: dict[str, Logistic]
    season_games: int  # games per team in the latest complete season
    injury: bool = False          # games model has injury terms
    injury_groups: bool = False   # ... including body-part groups
    games_knot: dict[str, float] = field(default_factory=dict)  # talent hinge per position
    # Share of a player's missed team games that were missed to injury, by
    # position: turns a projected season's missed games into injury weeks.
    injury_share: dict[str, float] = field(default_factory=dict)
    draft_survival: tuple[str, ...] = ()  # positions whose survival model has draft terms
    draft_games: tuple[str, ...] = ()     # ... and games model

    def draw(self, n_draws: int, rng: np.random.Generator) -> AvailabilityDraws:
        return AvailabilityDraws(
            # Sorted, so the random stream does not depend on dict order.
            games={p: self.games[p].draw_coefs(n_draws, rng) for p in sorted(self.games)},
            games_dispersion={p: m.dispersion for p, m in self.games.items()},
            survival={p: self.survival[p].draw_coefs(n_draws, rng) for p in sorted(self.survival)},
            season_games=self.season_games,
            injury=self.injury,
            injury_groups=self.injury_groups,
            games_knot=dict(self.games_knot),
            injury_share=dict(self.injury_share),
            draft_survival=self.draft_survival,
            draft_games=self.draft_games,
        )


@dataclass(frozen=True)
class AvailabilityDraws:
    """Coefficient draws, one row per posterior draw."""

    games: dict[str, np.ndarray]
    games_dispersion: dict[str, float]
    survival: dict[str, np.ndarray]
    season_games: int
    injury: bool = False
    injury_groups: bool = False
    games_knot: dict[str, float] = field(default_factory=dict)
    injury_share: dict[str, float] = field(default_factory=dict)
    draft_survival: tuple[str, ...] = ()
    draft_games: tuple[str, ...] = ()

    def survival_prob(
        self, position: str, age: np.ndarray, talent: np.ndarray, draft: np.ndarray | None = None,
    ) -> np.ndarray:
        """P(plays next season); `age` and `talent` are (draws, players);
        `draft` the next season's `draft_terms` (players, 2), used if the
        position's survival model has them."""
        x = _design(age, talent, self._draft(position, "survival", draft, age.shape))
        return _sigmoid(np.einsum("dpk,dk->dp", x, self.survival[position]))

    def _draft(self, position: str, model: str, draft: np.ndarray | None, shape: tuple) -> np.ndarray | None:
        on = position in (self.draft_survival if model == "survival" else self.draft_games)
        if not on:
            return None
        if draft is None:
            raise ValueError(f"{position} {model} model needs draft terms")
        return np.broadcast_to(draft, (*shape, draft.shape[-1]))

    def games_mean(
        self, position: str, age: np.ndarray, talent: np.ndarray, injury: np.ndarray | None = None,
        draft: np.ndarray | None = None,
    ) -> np.ndarray:
        """Expected share of games played; `injury` is the season's
        `injury_terms` (used only if the model has injury terms)."""
        x = _design(age, talent, injury if self.injury else None, self.games_knot.get(position))
        d = self._draft(position, "games", draft, age.shape)
        if d is not None:
            x = np.concatenate([x, d], axis=-1)
        return np.clip(_sigmoid(np.einsum("dpk,dk->dp", x, self.games[position])), 1e-6, 1 - 1e-6)

    def games_share(
        self, position: str, age: np.ndarray, talent: np.ndarray, rng: np.random.Generator,
        injury: np.ndarray | None = None, draft: np.ndarray | None = None,
    ) -> np.ndarray:
        """Share of a season's games played, drawn with season-to-season spread."""
        p = self.games_mean(position, age, talent, injury, draft)
        n, phi = self.season_games, self.games_dispersion[position]
        if phi <= 1.0 or n <= 1:
            return p
        rho = min((phi - 1.0) / (n - 1.0), 0.99)
        kappa = 1.0 / rho - 1.0
        return rng.beta(p * kappa, (1 - p) * kappa)


def season_games(schedules: pl.DataFrame) -> pl.DataFrame:
    """Regular-season games per team and last game date, by season."""
    per_team = pl.concat([
        schedules.select("season", "gameday", team="home_team"),
        schedules.select("season", "gameday", team="away_team"),
    ])
    return per_team.group_by("season").agg(
        games=pl.len().truediv(pl.col("team").n_unique()).round().cast(pl.Int64),
        last_gameday=pl.col("gameday").max(),
    )


def injury_history_terms(keys: pl.DataFrame, injury: pl.DataFrame, groups: bool = False) -> np.ndarray:
    """`injury_terms` for each row of `keys` (player_id, target season `t`):
    weeks missed in t - 1 and in t - 2, from the
    `features.injuries.injury_seasons` table (absent seasons are zero)."""
    cols = [f"missed_{g}" for g in GROUPS]
    k = keys.select("player_id", t=pl.col("t").cast(pl.Int32)).with_row_index("_row")
    inj = injury.with_columns(pl.col("season").cast(pl.Int32))
    last = k.join(inj.select("player_id", "season", *cols), left_on=["player_id", pl.col("t") - 1],
                  right_on=["player_id", "season"], how="left").sort("_row")
    prior = k.join(inj.select("player_id", "season", "missed"), left_on=["player_id", pl.col("t") - 2],
                   right_on=["player_id", "season"], how="left").sort("_row")
    return injury_terms(
        last.select([pl.col(c).fill_null(0) for c in cols]).to_numpy().astype(float),
        prior["missed"].fill_null(0).to_numpy().astype(float), groups=groups,
    )


def availability_rows(
    talent: pl.DataFrame, player_seasons: pl.DataFrame, schedules: pl.DataFrame, as_of,
    injury: pl.DataFrame | None = None, groups: bool = False, lagged: bool = False,
    static: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Training rows for the games and survival models.

    `talent` has player_id, season, position, talent_mean (one fitted talent
    per player-season). Returns (games_rows, survival_rows), each with
    position, age, talent, successes, trials. With `injury` (the
    injury_seasons table) games rows start once two seasons of reports
    exist and carry an `injury` column of injury terms. With `lagged`, games
    rows pair each season's games with the player's talent the season before
    (players with no season before are left out)."""
    seasons = season_games(schedules).filter(pl.col("last_gameday") < as_of)
    complete = seasons["season"]
    base = talent.join(player_seasons.select("player_id", "season", "age_sept1"), on=["player_id", "season"]).filter(
        pl.col("age_sept1").is_not_null()
    )
    games = base.join(seasons, on="season").filter(pl.col("games_played") > 0)
    if lagged:
        # Talent from the season before, as a projection knows it. Same-season
        # talent is estimated from the very games being predicted, which
        # overstated games for good players by about one a season.
        prior = talent.filter(pl.col("games_played") > 0).select(
            "player_id", season=pl.col("season") + 1, prior_talent="talent_mean"
        )
        games = games.join(prior, on=["player_id", "season"], how="inner").with_columns(talent_mean="prior_talent")
    games = games.select(
        "player_id", "season", "position", age="age_sept1", talent="talent_mean",
        successes=pl.min_horizontal("games_played", "games"), trials="games",
    )
    if static is not None:
        games = games.with_columns(draft=pl.Series(_draft_cols(games, static, pl.col("season")).tolist(),
                                                   dtype=pl.List(pl.Float64)))
    played = talent.filter(pl.col("games_played") > 0).select("player_id", "season").unique()
    survival = (
        base.filter((pl.col("games_played") > 0) & (pl.col("season") + 1).is_in(complete.implode()))
        .join(played.with_columns(pl.col("season") - 1, next_played=pl.lit(1)), on=["player_id", "season"], how="left")
        .select(
            "position", age="age_sept1", talent="talent_mean",
            successes=pl.col("next_played").fill_null(0), trials=pl.lit(1),
        )
    )
    if injury is None:
        return games.drop("player_id", "season"), survival
    games = games.filter(pl.col("season") >= INJURY_FIRST_SEASON + 2)
    terms = injury_history_terms(games.select("player_id", t="season"), injury, groups)
    games = games.drop("player_id", "season").with_columns(injury=pl.Series(terms.tolist(), dtype=pl.List(pl.Float64)))
    return games, survival


def ppg_survival_rows(
    player_seasons: pl.DataFrame, schedules: pl.DataFrame, as_of, first: int, static: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Survival rows from points per game rather than fitted talent, from
    season `first` on: fitted talent starts in each position's fit window
    (2013 for RB/WR), which holds very few old players, and the model then
    kept a 30-year-old star RB 62% likely to play at 34 (history: about 22%).
    Only seasons whose next season has ended count, so nothing after
    `as_of` is seen."""
    complete = season_games(schedules).filter(pl.col("last_gameday") < as_of)["season"]
    played = player_seasons.filter(pl.col("games_played") > 0)
    nxt = played.select("player_id", pl.col("season") - 1, next_played=pl.lit(1))
    rows = (
        played.filter(pl.col("age_sept1").is_not_null() & (pl.col("season") >= first)
                      & (pl.col("season") + 1).is_in(complete.implode()))
        .join(nxt, on=["player_id", "season"], how="left")
    )
    out = rows.select("position", age="age_sept1", talent="fpts_per_game",
                      successes=pl.col("next_played").fill_null(0), trials=pl.lit(1))
    if static is not None:
        # Draft terms for the season survived into (s + 1).
        out = out.with_columns(draft=pl.Series(_draft_cols(rows, static, pl.col("season") + 1).tolist(),
                                               dtype=pl.List(pl.Float64)))
    return out


def injury_share(player_seasons: pl.DataFrame, injury: pl.DataFrame, complete: list[int]) -> dict[str, float]:
    """Per position: weeks missed to injury / team games not played, over
    complete seasons with reports, among players who played."""
    rows = (
        player_seasons.filter(pl.col("season").is_in(complete) & (pl.col("season") >= INJURY_FIRST_SEASON)
                              & (pl.col("games_played") > 0))
        .select("player_id", "season", "position", "games_played")
        .join(injury.select("player_id", "season", "team_games", "missed"), on=["player_id", "season"], how="inner")
        .group_by("position")
        .agg(missed=pl.col("missed").sum(), absent=(pl.col("team_games") - pl.col("games_played")).clip(0).sum())
    )
    return {p: float(m / a) if a > 0 else 0.0 for p, m, a in rows.select("position", "missed", "absent").iter_rows()}


def fit_games(rows: pl.DataFrame, knot: float | None, injury: bool, groups: bool, draft: bool = False) -> Logistic:
    inj = np.array(rows["injury"].to_list()) if injury else None
    x = _design(rows["age"].to_numpy(), rows["talent"].to_numpy(), inj, knot)
    ridge = ridge_vector(x.shape[1], len(GROUPS) if injury and groups else 0)
    if draft:
        d = np.array(rows["draft"].to_list())
        x, ridge = np.concatenate([x, d], axis=1), np.concatenate([ridge, np.full(d.shape[1], BASE_RIDGE)])
    return fit_logistic(x, rows["successes"].to_numpy(), rows["trials"].to_numpy(), ridge)


def fit_availability(
    talent: pl.DataFrame, player_seasons: pl.DataFrame, schedules: pl.DataFrame, as_of,
    injury: pl.DataFrame | None = None, hinge_quantile: float | None = None, injury_groups: bool = False,
    lagged: bool = False, survival_first: int | None = None, static: pl.DataFrame | None = None,
    draft_survival: tuple[str, ...] = (), draft_games: tuple[str, ...] = (),
) -> AvailabilityModel:
    """Games and survival models per position. `injury` (injury_seasons) adds
    injury terms to the games model; `hinge_quantile` its talent hinge.
    `survival_first` trains survival on points per game from that season on
    instead of fitted talent (`ppg_survival_rows`). `draft_survival` and
    `draft_games` add `draft_terms` (from `static`, the player_static table)
    to those positions' models."""
    if (draft_survival or draft_games) and static is None:
        raise ValueError("draft terms need the player_static table")
    games_rows, survival_rows = availability_rows(
        talent, player_seasons, schedules, as_of, injury, injury_groups, lagged,
        static if draft_games else None,
    )
    if survival_first is not None:
        positions = talent["position"].unique().to_list()
        survival_rows = ppg_survival_rows(player_seasons.filter(pl.col("position").is_in(positions)),
                                          schedules, as_of, survival_first, static if draft_survival else None)
    elif draft_survival:
        raise ValueError("draft terms in survival need survival_first")
    games, survival, knots = {}, {}, {}
    for (position,), r in games_rows.sort("position").group_by("position", maintain_order=True):
        if hinge_quantile is not None:
            knots[position] = float(r["talent"].quantile(hinge_quantile))
        games[position] = fit_games(r, knots.get(position), injury is not None, injury_groups,
                                    draft=position in draft_games)
    for (position,), r in survival_rows.sort("position").group_by("position", maintain_order=True):
        d = np.array(r["draft"].to_list()) if position in draft_survival else None
        survival[position] = fit_logistic(
            _design(r["age"].to_numpy(), r["talent"].to_numpy(), d), r["successes"].to_numpy(), r["trials"].to_numpy(),
        )
    complete = season_games(schedules).filter(pl.col("last_gameday") < as_of).sort("season")
    share = injury_share(player_seasons, injury, complete["season"].to_list()) if injury is not None else {}
    return AvailabilityModel(
        games=games, survival=survival, season_games=int(complete["games"][-1]), injury=injury is not None,
        injury_groups=injury_groups, games_knot=knots, injury_share=share,
        draft_survival=tuple(draft_survival), draft_games=tuple(draft_games),
    )


STATUS_BUCKETS = ("healthy", "out_1", "out_2plus", "reserve_1to3", "reserve_4plus")
REMAINING_BUCKETS = (4, 9)     # remaining games 1-4, 5-9, 10+
STATUS_PRIOR_GAMES = 20.0      # pseudo-games pulling a thin cell toward healthy


def status_bucket(injured: pl.Expr, reserve: pl.Expr, streak: pl.Expr) -> pl.Expr:
    return (
        pl.when(~injured).then(pl.lit("healthy"))
        .when(reserve & (streak >= 4)).then(pl.lit("reserve_4plus"))
        .when(reserve).then(pl.lit("reserve_1to3"))
        .when(streak >= 2).then(pl.lit("out_2plus"))
        .otherwise(pl.lit("out_1"))
    )


def remaining_bucket(remaining: pl.Expr) -> pl.Expr:
    lo, hi = REMAINING_BUCKETS
    return pl.when(remaining <= lo).then(0).when(remaining <= hi).then(1).otherwise(2)


def status_rows(weeks: pl.DataFrame, seasons: list[int]) -> pl.DataFrame:
    """One row per player-week of `seasons` (complete seasons only) for players
    who had played that season: status after the week, team games left, and
    games the player played in them."""
    w = weeks.filter(pl.col("season").is_in(seasons)).sort("player_id", "season", "week")
    by = ["player_id", "season"]
    run = (pl.col("out") != pl.col("out").shift(1)).fill_null(True).cum_sum().over(by)
    return (
        w.with_columns(
            run=run,
            played_so_far=pl.col("played").cast(pl.Int64).cum_sum().over(by),
            remaining=(pl.len() - pl.int_range(1, pl.len() + 1)).over(by),
            played_after=(pl.col("played").cast(pl.Int64).sum() - pl.col("played").cast(pl.Int64).cum_sum()).over(by),
        )
        .with_columns(streak=pl.int_range(1, pl.len() + 1).over([*by, "run"]))
        .filter((pl.col("played_so_far") > 0) & (pl.col("remaining") > 0))
        .select(
            "player_id", "season", "week", "remaining", "played_after",
            bucket=status_bucket(pl.col("out"), pl.col("reserve") & pl.col("out"), pl.col("streak")),
            rem_bucket=remaining_bucket(pl.col("remaining")),
        )
    )


def status_effects(weeks: pl.DataFrame, seasons: list[int]) -> dict[tuple[str, int], float]:
    """Share of remaining games played, relative to healthy players, by
    (status bucket, remaining-games bucket); learned from `seasons`."""
    rows = status_rows(weeks, seasons)
    cells = rows.group_by("bucket", "rem_bucket").agg(
        played=pl.col("played_after").sum(), games=pl.col("remaining").sum()
    )
    healthy = {r: p / g for b, r, p, g in cells.iter_rows() if b == "healthy" and g > 0}
    out: dict[tuple[str, int], float] = {}
    for b, r, p, g in cells.iter_rows():
        h = healthy.get(r)
        if not h:
            continue
        share = (p + STATUS_PRIOR_GAMES * h) / (g + STATUS_PRIOR_GAMES)
        out[(b, r)] = 1.0 if b == "healthy" else min(share / h, 1.0)
    return out


def status_multiplier(
    status: pl.DataFrame, remaining: np.ndarray, effects: dict[tuple[str, int], float], player_ids: list[str],
) -> np.ndarray:
    """Per player (in `player_ids` order): the factor on this season's
    remaining games from current injury status (1 when healthy or unknown)."""
    lo, hi = REMAINING_BUCKETS
    rb = np.where(remaining <= lo, 0, np.where(remaining <= hi, 1, 2))
    s = pl.DataFrame({"player_id": player_ids}).join(
        status.select("player_id", bucket=status_bucket(pl.col("injured"), pl.col("reserve"), pl.col("streak"))),
        on="player_id", how="left",
    )
    buckets = s["bucket"].fill_null("healthy").to_list()
    return np.array([effects.get((b, int(r)), 1.0) for b, r in zip(buckets, rb)])
