"""Rookie draft pick values (plan "Plan: value rookie draft picks",
approved by ben 2026-11-14).

A pick is worth the full range of futures of the rookie it lands on draft
day. Every QB, RB, WR and TE drafted in the NFL since `picks.first_class`
is a sample of draft-day credentials: NFL draft pick, position and age. Each
one is valued as a brand-new rookie entering the league in the pick's year,
with nothing known but those credentials:

- talent in the rookie season is drawn from the fitted rookie prior (draft
  capital, the first-round step), with the position's average rookie
  situation, then projected like any player (age curve, random walk,
  growth, QB drift for WRs, the QB spread factor);
- the rookie season's games come from rookie-season models fitted here on
  history (chance of playing at all, share of games, QB full games, and for
  a rookie who never played, the chance he plays the next year), because
  the board's games and survival models learn from players who already
  have a season behind them;
- later seasons run through the same survival, games and lineup value as
  the board (`dwar.season_values`), inside the same window: a 2028 pick's
  rookie is valued for 2028 to the window's last season.

Each class is put in draft-day order by those values (Superflex gets its
own order), and pick k of a draft is the k-th rookie of a class. A pick's
value pools the k-th rookie's draws over every class, so it never depends
on what the one or two real players taken at that slot happened to do
(ben, 2026-11-14), and a later pick always takes a rookie with weaker
credentials.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import polars as pl

from dwar.config import LeagueFormat, ValueConfig
from dwar.features.player_tables import PlayerTables
from dwar.model.fit import FitResult
from dwar.model.data import pick_curve
from dwar.value.availability import (
    Logistic,
    _sigmoid,
    fit_logistic,
    full_game_counts,
    season_games,
)
from dwar.value.dwar import (
    PERCENTILES,
    RookieStart,
    SeasonTrace,
    ValueInputs,
    draw_indices,
    season_values,
)
from dwar.value.projection import PositionDraws, project_talent
from dwar.value.spread import widen

PICK_CENTER = 64.0          # log(pick / 64) in the rookie-season models
DEFAULT_ROOKIE_AGE = 22.0   # age at September 1 for a rookie with no birth date
HIT_GAMES = 8.0             # a "hit" season: this many games at starter level or better
BUST_GAMES = 4.0            # a bust never has a season of this many games above waiver level


def sample_rookies(static: pl.DataFrame, first_class: int, positions: tuple[str, ...]) -> pl.DataFrame:
    """Drafted QB/RB/WR/TE from `first_class` on, with draft-day credentials:
    player_id, name, position (at the draft), draft_year, draft_pick,
    round1, age (at September 1 of the draft year). `static` is already
    as-of gated, so only drafts that had happened by then are here."""
    pos = pl.coalesce("draft_position", "position") if "draft_position" in static.columns else pl.col("position")
    age = ((pl.date(pl.col("draft_year"), 9, 1) - pl.col("birth_date")).dt.total_days() / 365.25
           if "birth_date" in static.columns else pl.lit(None, pl.Float64))
    out = (
        static.filter(pl.col("draft_pick").is_not_null() & (pl.col("draft_year") >= first_class))
        .with_columns(position=pos)
        .filter(pl.col("position").is_in(list(positions)))
        .select(
            "player_id", "name", "position", pl.col("draft_year").cast(pl.Int64),
            pl.col("draft_pick").cast(pl.Int64), round1=(pl.col("draft_round") == 1).fill_null(False),
            age=age,
        )
    )
    # A missing birth date takes the median age of the same draft round.
    rnd = ((pl.col("draft_pick") - 1) // 32).clip(0, 6)
    return (
        out.with_columns(pl.col("age").fill_null(pl.col("age").median().over(rnd)))
        .with_columns(pl.col("age").fill_null(DEFAULT_ROOKIE_AGE))
        .sort("draft_year", "draft_pick")
    )


def _design(pick: np.ndarray, round1: np.ndarray) -> np.ndarray:
    lp = np.log(np.asarray(pick, float) / PICK_CENTER)
    return np.column_stack([np.ones_like(lp), lp, lp**2, np.asarray(round1, float)])


@dataclass(frozen=True)
class RookieSeasonModel:
    """One position's rookie-season models (logistic on log pick, its square,
    round 1). Rookie-season talent was tried as a games term (fit on fitted
    rookie-season talent): applied to prior draws, whose spread is wider
    than fitted talent's, it cut rookies' games about 3 a season below
    history, so it is left out."""

    play: Logistic             # plays at least one game as a rookie
    games: Logistic            # share of the season's games, among those who played
    full: Logistic | None      # full games per game played (positions with a full-games threshold)
    late: Logistic             # plays the next season, among those who did not play as rookies
    n: int                     # rookies learned from


def rookie_season_rows(
    static: pl.DataFrame, player_seasons: pl.DataFrame, player_weeks: pl.DataFrame, schedules: pl.DataFrame,
    as_of, first: int, min_plays: dict[str, int], talent: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """One row per drafted player whose rookie season has ended by `as_of`:
    position, draft_pick, round1, games (played), full, team_games, next_played
    (null when the next season has not ended), and talent: the fitted talent
    of the rookie season (`talent`: player_id, season, talent_mean; null when
    he did not play or the season is outside the fits)."""
    sg = season_games(schedules).filter(pl.col("last_gameday") < as_of).select(
        pl.col("season").cast(pl.Int64), team_games="games")
    done = sg["season"].to_list()
    ps = player_seasons.select("player_id", pl.col("season").cast(pl.Int64), "games_played")
    full = full_game_counts(player_weeks, min_plays)
    rookies = sample_rookies(static, first, ("QB", "RB", "WR", "TE")).filter(pl.col("draft_year").is_in(done))
    return (
        rookies.join(sg, left_on="draft_year", right_on="season", how="inner")
        .join(ps, left_on=["player_id", "draft_year"], right_on=["player_id", "season"], how="left")
        .join(full, left_on=["player_id", "draft_year"], right_on=["player_id", "season"], how="left")
        .join(ps.select("player_id", season=pl.col("season") - 1, next_games="games_played"),
              left_on=["player_id", "draft_year"], right_on=["player_id", "season"], how="left")
        .with_columns(
            games=pl.min_horizontal(pl.col("games_played").fill_null(0), "team_games"),
            full=pl.col("full").fill_null(0),
            next_played=pl.when((pl.col("draft_year") + 1).is_in(done))
            .then(pl.col("next_games").fill_null(0) > 0).otherwise(None),
        )
        .join(talent.select("player_id", pl.col("season").cast(pl.Int64), talent="talent_mean")
              if talent is not None else pl.DataFrame(schema={"player_id": pl.String, "season": pl.Int64,
                                                              "talent": pl.Float64}),
              left_on=["player_id", "draft_year"], right_on=["player_id", "season"], how="left")
        .select("player_id", "position", "draft_year", "draft_pick", "round1", "games", "full", "team_games",
                "next_played", "talent")
    )


def fit_rookie_seasons(rows: pl.DataFrame, full_positions: tuple[str, ...]) -> dict[str, RookieSeasonModel]:
    out = {}
    for (position,), r in rows.sort("position").group_by("position", maintain_order=True):
        x = _design(r["draft_pick"].to_numpy(), r["round1"].to_numpy())
        g = r["games"].to_numpy().astype(float)
        played = g > 0
        play = fit_logistic(x, played.astype(float), np.ones(len(g)))
        games = fit_logistic(x[played], g[played], r["team_games"].to_numpy()[played].astype(float))
        full = None
        if position in full_positions:
            full = fit_logistic(x[played], r["full"].to_numpy()[played].astype(float), g[played])
        nxt = r["next_played"].to_numpy()
        sat = ~played & np.array([v is not None for v in nxt])
        late = fit_logistic(x[sat], np.array([float(v) for v in nxt[sat]]), np.ones(int(sat.sum())))
        out[position] = RookieSeasonModel(play, games, full, late, r.height)
    return out


def rookie_start(
    model: RookieSeasonModel, pick: np.ndarray, round1: np.ndarray, season: int, talent: np.ndarray,
    season_games_n: int, rng: np.random.Generator, couple: float = 0.0,
) -> RookieStart:
    """Rookie-season games draws for players drafted at `pick`, entering in
    `season`, with rookie-season talent draws `talent` (draws, players).

    With `couple` > 0, each player's games draws are re-paired with his
    talent draws through a Gaussian copula with that correlation: the
    better draws of a rookie play more often, while his games distribution
    (set by pick alone) is unchanged (`couple_to_talent`)."""
    n_draws = talent.shape[0]
    x = _design(pick, round1)
    xt = np.concatenate([np.broadcast_to(x, (n_draws, *x.shape)), talent[:, :, None]], axis=-1)

    def p(m: Logistic) -> np.ndarray:
        c = m.draw_coefs(n_draws, rng)
        if c.shape[1] == x.shape[1]:
            return _sigmoid(c @ x.T)                        # (draws, players)
        return _sigmoid(np.einsum("dpk,dk->dp", xt, c))

    plays = rng.random((n_draws, len(pick))) < p(model.play)
    mean = np.clip(p(model.games), 1e-6, 1 - 1e-6)
    phi, n = model.games.dispersion, season_games_n
    if phi > 1.0 and n > 1:
        rho = min((phi - 1.0) / (n - 1.0), 0.99)
        kappa = 1.0 / rho - 1.0
        share = rng.beta(mean * kappa, (1 - mean) * kappa)
    else:
        share = mean
    share = np.where(plays, np.maximum(share, 1.0 / n), 0.0)   # a rookie who plays plays at least once
    if couple > 0:
        share = couple_to_talent(share, talent, couple, rng)
    ratio = p(model.full) if model.full is not None else None
    return RookieStart(np.full(len(pick), season), share, ratio, p(model.late))


def couple_to_talent(values: np.ndarray, talent: np.ndarray, rho: float, rng: np.random.Generator) -> np.ndarray:
    """`values` (draws, players) re-ordered within each player so that their
    ranks follow a Gaussian copula with correlation `rho` to `talent`'s
    ranks. Each column keeps exactly its own values."""
    t = talent - talent.mean(axis=0)
    t = t / np.maximum(t.std(axis=0), 1e-12)
    u = rho * t + np.sqrt(1 - rho**2) * rng.standard_normal(t.shape)
    out = np.empty_like(values)
    cols = np.arange(values.shape[1])[None, :]
    out[np.argsort(u, axis=0), cols] = np.sort(values, axis=0)
    return out


def _pick_scale(r: FitResult, static: pl.DataFrame) -> tuple[float, float, np.ndarray | None]:
    """(mu, sd) of the fit's log pick standardization, and the first-round
    flag's straight-line fit on [1, log_pick_z, undrafted] (model/talent.py)."""
    d = r.data
    picks = pl.DataFrame({"player_id": list(d.player_ids)}).join(
        static.select("player_id", "draft_pick"), on="player_id", how="left", maintain_order="left"
    )["draft_pick"].cast(pl.Float64).to_numpy()
    ok = np.isfinite(picks) & (d.undrafted == 0)
    slope, intercept = np.polyfit(np.log(picks[ok]), d.log_pick_z[ok], 1)
    sd = 1.0 / slope
    mu = -intercept * sd
    coef = None
    if d.round1 is not None and d.round1.any():
        X = np.column_stack([np.ones(d.n_players), d.log_pick_z, d.undrafted])
        coef = np.linalg.lstsq(X, d.round1, rcond=None)[0]
    return float(mu), float(sd), coef


def prior_shape_correction(
    r: FitResult, idx: np.ndarray, static: pl.DataFrame, mu: float, sd: float,
) -> np.ndarray:
    """(draws, 3) coefficients on [1, z, z^2] (z = standardized log pick)
    of each drafted rookie's posterior deviation from the rookie prior's
    line, by least squares in each draw.

    The prior is a straight line in log pick (plus a first-round step for
    some positions). On the 2026-11-14 fits the fit's own drafted rookies
    sit below it at the top of the draft and above it from picks ~17 to 100
    (picks study, resid.py), so a straight line overrates top-8 picks and
    underrates the rest of round 1 through round 3. Deviations of rookies
    with little data are shrunk toward zero, so this correction understates
    the bend rather than overstating it."""
    d, post = r.data, r.posterior["posterior"]
    rk = np.flatnonzero(d.pre_window == 0)
    picks = pl.DataFrame({"player_id": [d.player_ids[i] for i in rk]}).join(
        static.select("player_id", "draft_pick"), on="player_id", how="left", maintain_order="left"
    )["draft_pick"].cast(pl.Float64).to_numpy()
    ok = np.isfinite(picks) & (d.undrafted[rk] == 0)
    if ok.sum() < 30:
        return np.zeros((len(idx), 3))

    def take(name: str) -> np.ndarray:
        v = np.asarray(post[name].values)
        return v.reshape(-1, *v.shape[2:])[idx]

    lz = d.log_pick_z[rk]
    rsd = take("sigma_rookie")[:, None]
    if "sd_pick" in post:
        rsd = rsd * np.exp(take("sd_pick")[:, None] * lz[None, :] + take("sd_undrafted")[:, None] * d.undrafted[rk][None, :])
    dev = (rsd * take("rookie_z"))[:, ok]
    z = (np.log(picks[ok]) - mu) / sd
    X = np.column_stack([np.ones_like(z), z, z**2])
    return dev @ np.linalg.pinv(X).T


def rookie_latent(
    r: FitResult, idx: np.ndarray, static: pl.DataFrame, pick: np.ndarray, round1: np.ndarray,
    rng: np.random.Generator, shape: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Rookie-season latent talent draws (draws, players) from the fitted
    rookie prior, plus the position's average rookie situation; and each
    player's standardized log pick.

    With `shape`, the prior's mean also gets a quadratic in log pick fitted,
    draw by draw, to the fit's own drafted rookies' deviations from the
    prior line (`prior_shape_correction`)."""
    post = r.posterior["posterior"]

    def take(name: str) -> np.ndarray:
        v = np.asarray(post[name].values)
        return v.reshape(-1, *v.shape[2:])[idx]

    mu, sd, coef = _pick_scale(r, static)
    z = (np.log(np.asarray(pick, float)) - mu) / sd
    d = r.data
    curve = "curve_k" in post
    if curve:
        # Curve-in-raw-pick prior (model/talent.py curve_k): no first-round
        # step, and no log-pick shape correction on top of it.
        cz = pick_curve(np.asarray(pick, float), float(np.asarray(post["curve_k"].values).ravel()[0]), d.draft_pick_ref)
        mean = take("alpha0")[:, None] + take("alpha_pick")[:, None] * cz[None, :]
    else:
        mean = take("alpha0")[:, None] + take("alpha_pick")[:, None] * z[None, :]
    if "alpha_round1" in post and coef is not None and not curve:
        r1 = np.asarray(round1, float) - np.column_stack([np.ones_like(z), z, np.zeros_like(z)]) @ coef
        mean = mean + take("alpha_round1")[:, None] * r1[None, :]
    if shape and not curve:
        mean = mean + prior_shape_correction(r, idx, static, mu, sd) @ np.stack([np.ones_like(z), z, z**2])
    rookie_sd = take("sigma_rookie")[:, None]
    if "sd_pick" in post:
        rookie_sd = rookie_sd * np.exp(take("sd_pick")[:, None] * z[None, :])
    if "sd_curve" in post and curve:
        rookie_sd = rookie_sd * np.exp(take("sd_curve")[:, None] * cz[None, :])
    latent = mean + rookie_sd * rng.standard_normal(mean.shape)
    if "eta" in post and d.season_x.shape[1]:
        first = np.flatnonzero(d.ps_is_first & (d.pre_window[d.ps_player] == 0))
        if len(first):
            latent = latent + (take("eta") @ d.season_x[first].mean(axis=0))[:, None]
    return latent, z


def _with_rookies(pd_: PositionDraws, r: FitResult, latent: np.ndarray, z: np.ndarray) -> tuple[PositionDraws, np.ndarray]:
    """`pd_` with the rookies appended as fitted rookie seasons; their indices."""
    d, n, k = r.data, pd_.latent.shape[1], latent.shape[1]
    first = np.flatnonzero(d.ps_is_first & (d.pre_window[d.ps_player] == 0))

    def cat(a: np.ndarray | None, v: float | np.ndarray, axis: int = -1) -> np.ndarray | None:
        if a is None:
            return None
        if axis == 0:
            fill = np.broadcast_to(np.asarray(v, a.dtype), (k, *a.shape[1:]))
            return np.concatenate([a, fill], axis=0)
        fill = np.broadcast_to(np.asarray(v, float), (*a.shape[:-1], k))
        return np.concatenate([a, fill], axis=-1)

    qb_end = float(np.nanmean(d.ps_qb_end[first])) if getattr(d, "ps_qb_end", None) is not None and len(first) else 0.0
    out = replace(
        pd_,
        latent=np.concatenate([pd_.latent, latent], axis=1),
        talent=np.concatenate([pd_.talent, pd_.expected_points(latent)], axis=1),
        career_year=cat(pd_.career_year, 1, axis=0),
        ps_pick_z=None if pd_.ps_pick_z is None else np.concatenate([pd_.ps_pick_z, z]),
        ps_career_games_end=cat(pd_.ps_career_games_end, 0.0, axis=0),
        ps_qb_end=cat(pd_.ps_qb_end, qb_end, axis=0),
        form_dev=cat(pd_.form_dev, 0.0),
        ps_usage_next=cat(pd_.ps_usage_next, 0.0, axis=0),
    )
    return out, np.arange(n, n + k)


def value_rookies(
    position: str, r: FitResult, inputs: ValueInputs, fmt: LeagueFormat, cfg: ValueConfig,
    rookies: pl.DataFrame, year: int, model: RookieSeasonModel, idx: np.ndarray, static: pl.DataFrame,
    rng: np.random.Generator,
) -> tuple[np.ndarray, SeasonTrace]:
    """DWAR draws (draws, rookies) for `rookies` (one position) entering the league in `year`."""
    clock, pd_ = inputs.clock, inputs.draws[position]
    pick = rookies["draft_pick"].to_numpy().astype(float)
    round1 = rookies["round1"].to_numpy()
    latent, z = rookie_latent(r, idx, static, pick, round1, rng, cfg.picks is not None and cfg.picks.prior_shape)
    aug, ps_idx = _with_rookies(pd_, r, latent, z)
    seasons = np.arange(clock.current - 1, clock.current + cfg.horizon_seasons + 1)
    last_season = np.full(len(pick), year)
    last_age = rookies["age"].to_numpy().astype(float)
    talent = project_talent(aug, ps_idx, last_season, last_age, seasons, rng)
    talent = widen(talent, cfg.spread_factor.get(position, 1.0), axis=0)
    if cfg.picks is not None and position in cfg.picks.talent_scale:
        talent = talent * cfg.picks.talent_scale[position]   # calibrated to real careers (picks study, check2.py)
    ages = np.broadcast_to(last_age[None, :, None] + (seasons[None, None, :] - year), talent.shape)
    start = rookie_start(model, pick, round1, year, talent[:, :, int(np.flatnonzero(seasons == year)[0])],
                         inputs.availability.season_games, rng,
                         cfg.picks.games_talent_corr.get(position, 0.0) if cfg.picks is not None else 0.0)
    use_draft = position in inputs.availability.draft_survival or position in inputs.availability.draft_games
    return season_values(
        position, inputs, fmt, rng, talent, ages, seasons, last_season,
        [f"rookie:{p}" for p in rookies["player_id"].to_list()], np.zeros(len(pick)),
        np.full(len(pick), float(year)) if use_draft else None, pick if use_draft else None,
        rookie=start, trace=True,
    )


@dataclass(frozen=True)
class PickValues:
    picks: pl.DataFrame     # one row per pick (exact slots and early/mid/late groups) per year
    rookies: pl.DataFrame   # one row per sample rookie per year: value summary, draft-day rank, hit/bust
    models: dict[str, RookieSeasonModel]
    levels: dict[str, dict[str, float]]   # PPG bars: "hit" (last weekly starter), "bust" (waiver level)


def _group_labels(teams: int) -> list[tuple[str, range]]:
    third = teams / 3
    cuts = [0, round(third), round(2 * third), teams]
    return [(name, range(cuts[i] + 1, cuts[i + 1] + 1)) for i, name in enumerate(("Early", "Mid", "Late"))]


def _ordinal(n: int) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(n, f"{n}th")


def pick_values(
    fits: dict[str, FitResult], tables: PlayerTables, schedules: pl.DataFrame, inputs: ValueInputs,
    fmt: LeagueFormat, cfg: ValueConfig, min_plays: dict[str, int], rng: np.random.Generator,
) -> PickValues:
    """Values of the next `cfg.picks.years` rookie drafts' picks, 1.01 to
    `fmt.rookie_rounds`.`fmt.teams`, plus early/mid/late groups per round."""
    pc = cfg.picks
    assert pc is not None
    static = tables.player_static
    positions = tuple(fits)
    rookies = sample_rookies(static, pc.first_class, positions)
    fitted = pl.concat([r.talent.select("player_id", "season", "talent_mean") for r in fits.values()])
    rows = rookie_season_rows(static, tables.player_seasons, tables.player_weeks, schedules, tables.as_of,
                              pc.history_first, min_plays, fitted)
    models = fit_rookie_seasons(rows, tuple(p for p in cfg.full_games if p in min_plays))
    first = next(iter(fits.values()))
    n_total = first.posterior["posterior"].sizes["chain"] * first.posterior["posterior"].sizes["draw"]
    idx = draw_indices(n_total, cfg.draws)
    starter = {p: float(np.mean(v)) for p, v in _starter_levels(fits, inputs, fmt, cfg).items()}
    waiver = {p: float(np.mean(v)) for p, v in inputs.replacement.items()}
    sd_week = weekly_sd(tables.player_weeks, pc.first_class)
    n_picks = fmt.rookie_rounds * fmt.teams
    pick_rows, rookie_rows = [], []
    for year in range(inputs.clock.current + 1, inputs.clock.current + 1 + pc.years):
        draws_by, meta = [], []
        for p in positions:
            sub = rookies.filter(pl.col("position") == p)
            if sub.is_empty() or p not in models:
                continue
            total, tr = value_rookies(p, fits[p], inputs, fmt, cfg, sub, year, models[p], idx, static, rng)
            # Seasons as they would be scored: season PPG = talent plus the
            # noise of that many weeks, so hits and busts mean what they
            # would in a box score (and can be checked against real careers).
            on = tr.alive > 0
            g = np.where(on, tr.games, 0.0)
            ppg = tr.talent + sd_week[p] * rng.standard_normal(g.shape) / np.sqrt(np.maximum(g, 1.0))
            hit = (on & (g >= HIT_GAMES) & (ppg >= starter[p])).any(axis=-1)
            ok = (on & (g >= BUST_GAMES) & (ppg > waiver[p])).any(axis=-1)
            pts = (g * ppg).sum(axis=-1)
            pq = np.percentile(pts, (5, 25, 75, 95), axis=0)
            draws_by.append(total)
            meta.append(sub.with_columns(
                hit=pl.Series(hit.mean(axis=0)), bust=pl.Series(1.0 - ok.mean(axis=0)),
                points_mean=pl.Series(pts.mean(axis=0)),
                **{f"points_p{q:02d}": pl.Series(v) for q, v in zip((5, 25, 75, 95), pq)},
            ))
        d = np.concatenate(draws_by, axis=1)
        m = pl.concat(meta).with_columns(
            dwar_mean=pl.Series(d.mean(axis=0)), **{
                f"dwar_p{q:02d}": pl.Series(v) for q, v in zip(PERCENTILES, np.percentile(d, PERCENTILES, axis=0))
            }, col=pl.Series(np.arange(d.shape[1])), pick_year=pl.lit(year),
        ).with_columns(
            rank=pl.col("dwar_mean").rank("ordinal", descending=True).over("draft_year").cast(pl.Int64)
        )
        rookie_rows.append(m)
        slot_cols = {k: m.filter(pl.col("rank") == k)["col"].to_numpy() for k in range(1, n_picks + 1)}
        slot_meta = {k: m.filter(pl.col("rank") == k) for k in range(1, n_picks + 1)}

        def summary(ks: list[int], slot_cols=slot_cols, d=d, slot_meta=slot_meta, **label) -> dict:
            cols = np.concatenate([slot_cols[k] for k in ks])
            v = d[:, cols].ravel()
            sm = pl.concat([slot_meta[k] for k in ks])
            mix = {f"share_{p.lower()}": float((sm["position"] == p).mean()) for p in positions}
            return {**label, "dwar_mean": float(v.mean()),
                    **{f"dwar_p{q:02d}": float(x) for q, x in zip(PERCENTILES, np.percentile(v, PERCENTILES))},
                    "hit_rate": float(sm["hit"].mean()), "bust_rate": float(sm["bust"].mean()),
                    "classes": sm["draft_year"].n_unique(), **mix}

        for k in range(1, n_picks + 1):
            rd, slot = (k - 1) // fmt.teams + 1, (k - 1) % fmt.teams + 1
            pick_rows.append(summary([k], pick_year=year, round=rd, slot=slot, kind="slot",
                                     label=f"{year} {rd}.{slot:02d}"))
        for rd in range(1, fmt.rookie_rounds + 1):
            base = (rd - 1) * fmt.teams
            for name, slots in _group_labels(fmt.teams):
                pick_rows.append(summary([base + s for s in slots], pick_year=year, round=rd, slot=None,
                                         kind=name.lower(), label=f"{year} {name} {_ordinal(rd)}"))
            pick_rows.append(summary([base + s for s in range(1, fmt.teams + 1)], pick_year=year, round=rd,
                                     slot=None, kind="round", label=f"{year} {_ordinal(rd)}"))
    picks = pl.DataFrame(pick_rows).with_columns(
        player_id=pl.concat_str(pl.lit("PICK-"), pl.col("label").str.replace_all(" ", "-")),
        position=pl.lit("PICK"),
    )
    return PickValues(picks, pl.concat(rookie_rows).drop("col"), models, {"hit": starter, "bust": waiver})


def weekly_sd(player_weeks: pl.DataFrame, first: int) -> dict[str, float]:
    """Within player-season SD of weekly points, pooled, by position (seasons from `first`)."""
    v = (
        player_weeks.filter(pl.col("season") >= first)
        .group_by("position", "player_id", "season")
        .agg(var=pl.col("fpts").var(), n=pl.len())
        .filter(pl.col("n") >= 4)
        .group_by("position")
        .agg(sd=((pl.col("var") * (pl.col("n") - 1)).sum() / (pl.col("n") - 1).sum()).sqrt())
    )
    return dict(zip(v["position"].to_list(), v["sd"].to_list()))


def _starter_levels(
    fits: dict[str, FitResult], inputs: ValueInputs, fmt: LeagueFormat, cfg: ValueConfig,
) -> dict[str, np.ndarray]:
    """Talent of the last weekly starter at each position (the board's
    starter replacement level): the bar for a "hit" season."""
    from dwar.value.dwar import replacement_from_fits

    recent = inputs.clock.complete[-cfg.replacement_seasons:]
    return replacement_from_fits(fits, inputs.draws, fmt, recent, cfg.pool_min_games)
