"""Step a fitted talent model through later seasons one game at a time.

The model's settings (prior, age curve, growth, random-walk and game-step
sizes, weekly noise) come from a fit on earlier seasons and are held fixed.
Each player's talent is then tracked game by game with a particle filter:
before every game in the test seasons, the filter's current belief is
projected forward and scored against what the player went on to score over
the rest of that season and each of the next `HORIZON` seasons, plus their
discounted total (the same stretch and discount DWAR values).

Point in time: settings use only the fit's seasons, standardization of
draft pick and combine uses the fit's players (`build_model_data`
`reference_players`), and a forecast made before a game uses only the games
before it. Games played in each scored stretch are taken as known, so the
check is of talent, not availability (missed games and departures count as
no games).

Given the settings, players are independent, so each is filtered alone.
Talent's random walk (offseason steps plus per-game steps) only ever enters
the weekly distribution as a sum, so the filter tracks one state per
particle: latent talent before the season's age and growth effects.
Particles are grouped by posterior draw and resampled within their draw, so
the forecast mixes evenly over the fit's uncertainty in the settings.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.stats import t as student_t

from dwar.features.qb_rating import LOW_KNOT as QB_LOW_KNOT
from dwar.features.qb_rating import QBFade, qb_fade
from dwar.model.data import ModelData
from dwar.model.talent import (
    AGE_SHIFT_MAX,
    EARLY_GAMES,
    GROWTH_YEARS,
    age_curve_effect,
    link,
    nonzero_prob,
    season_early,
    simulate_weeks,
)
from dwar.validate.forecast import kde_log_density, pit

HORIZON = 5  # full seasons scored after the current one (DWAR horizon_seasons; was 3 before 2026-11-07)
DISCOUNT = 0.80


@dataclass(frozen=True)
class Settings:
    """Posterior draws of the global parameters, (S,) each unless noted."""

    likelihood: str
    alpha0: np.ndarray
    alpha_pick: np.ndarray
    alpha_undrafted: np.ndarray
    sigma_rookie: np.ndarray
    mu_vet: np.ndarray
    sigma_vet: np.ndarray
    tau: np.ndarray
    beta1: np.ndarray
    beta2: np.ndarray
    sigma: np.ndarray
    nu: np.ndarray
    h0: np.ndarray
    h1: np.ndarray
    zero_value: float
    gamma: np.ndarray            # (S, m) prospect coefficients; m may be 0
    alpha_round1: np.ndarray     # zeros without a first-round step
    round1_fit: np.ndarray       # (3,) the fit's regression of the round-1 flag on [1, log pick z, undrafted]
    growth: np.ndarray           # (S, len(GROWTH_YEARS)); zeros without growth
    growth_pick: np.ndarray
    sigma_game: np.ndarray       # zeros without game steps
    lam_game: np.ndarray
    prospect_names: tuple[str, ...]
    rho_form: np.ndarray | None = None   # "form" mode (model/talent.py): decaying in-season form
    sigma_form: np.ndarray | None = None
    lam_form: np.ndarray | None = None
    qb_drift: np.ndarray | None = None      # offseason drift with the QB rating; None without it
    qb_drift_low: np.ndarray | None = None
    fade: QBFade | None = None              # later seasons' QB fades toward average; None = carried forward
    beta3: np.ndarray | None = None         # cubic age term; None = quadratic
    lam_tau: np.ndarray | None = None       # early-career boost to the offseason step (tau_early); None = off
    rho_season: np.ndarray | None = None    # pull toward level_mean between seasons (season_rho); None = random walk
    level_mean: np.ndarray | None = None
    rho_bump: np.ndarray | None = None      # fading season bump (season_bump): AR(1) rate and stationary SD
    sigma_bump: np.ndarray | None = None
    sigma_fluke: np.ndarray | None = None   # rookie season's own non-lasting boost/dip (rookie_fluke); None = off
    bump_first: bool = True                 # False: no bump in a player's first season
    lam_y3: np.ndarray | None = None        # first-round picks' boost to the step into career year 3 (tau_y3)
    sigma_young: np.ndarray | None = None   # young players' lasting per-game step (young_steps); None = off
    exit_rate: float = 0.0                  # chance a game is an injury exit (games left out of the fit)
    exit_points: np.ndarray | None = None   # points in past injury-exit games, drawn for simulated exits
    age_curve: str = "quadratic"            # "plateau": beta1-3 are age_rise/fall/fall2; "talent": peak shifts
    age_shift: np.ndarray | None = None     # talent curve: years of peak shift per SD of level
    age_ref: np.ndarray | None = None       # talent curve: mean and SD of fit players' levels
    age_ref_sd: np.ndarray | None = None

    def step_sd(self, draw: np.ndarray, career_games: float, career_year: int = 0, round1: float = 0.0) -> np.ndarray:
        """Offseason step SD into a season starting after `career_games` games
        (career year `career_year` of the new season; 0 = unknown)."""
        out = self.tau[draw]
        if self.lam_tau is not None:
            out = out * np.exp(self.lam_tau[draw] * season_early(career_games))
        if self.lam_y3 is not None and career_year == 3 and round1 > 0:
            out = out * np.exp(self.lam_y3[draw])
        return out

    def revert(self, x: np.ndarray, draw: np.ndarray, years: float) -> np.ndarray:
        """Level `years` seasons on before its new step: pulled toward
        level_mean by rho_season^years (unchanged for a random walk)."""
        if self.rho_season is None:
            return x
        m, r = self.level_mean[draw], self.rho_season[draw] ** max(years, 0.0)
        return m + r * (x - m)

    def bump_start(self, draw: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """A first season's bump (zero without season_bump)."""
        if self.rho_bump is None or not self.bump_first:
            return np.zeros(len(draw))
        return self.sigma_bump[draw] * rng.standard_normal(len(draw))

    def bump_next(self, b: np.ndarray, draw: np.ndarray, years: float, rng: np.random.Generator,
                  from_first: bool = False) -> np.ndarray:
        """The bump `years` seasons on: faded by rho_bump^years plus a fresh
        part (a fresh stationary draw out of a bump-free first season)."""
        if self.rho_bump is None:
            return np.zeros_like(b)  # no bump model; a rookie fluke ends with the season
        if from_first and not self.bump_first:
            return self.sigma_bump[draw] * rng.standard_normal(len(b))
        r = self.rho_bump[draw] ** max(years, 1.0)
        return r * b + self.sigma_bump[draw] * np.sqrt(1.0 - r**2) * rng.standard_normal(len(b))

    def drift(self, q: float, draw: np.ndarray, offseason: int = 1) -> np.ndarray | float:
        """Mean offseason move given the QB rating `q` at the end of the season
        before; `offseason` counts projected offseasons (later ones fade q
        toward average when `fade` is set)."""
        if self.qb_drift is None:
            return 0.0
        if self.fade is not None:
            q = self.fade.expected(q, offseason)
        return self.qb_drift[draw] * q + self.qb_drift_low[draw] * min(q - QB_LOW_KNOT, 0.0)

    @property
    def form(self) -> bool:
        return self.rho_form is not None

    @property
    def n(self) -> int:
        return len(self.tau)

    @classmethod
    def from_posterior(cls, d: ModelData, posterior, likelihood: str, draws: np.ndarray,
                       fade_qb: bool = False, bump_first: bool = True) -> Settings:
        post = posterior["posterior"]

        def take(name: str, default: float | None = None, width: int | None = None) -> np.ndarray:
            if name not in post:
                if default is None:
                    raise KeyError(name)
                return np.full((len(draws),) if width is None else (len(draws), width), default)
            v = np.asarray(post[name].values)
            return v.reshape(-1, *v.shape[2:])[draws]

        if d.season_x.shape[1] or d.season_vol_x.shape[1]:
            raise NotImplementedError("stepping through situation factors is not built yet")
        X = np.column_stack([np.ones(d.n_players), d.log_pick_z, d.undrafted])
        r1 = d.round1 if d.round1 is not None else np.zeros(d.n_players)
        g = len(GROWTH_YEARS)
        plateau, talent = "age_rise" in post, "age_shift" in post
        return cls(
            likelihood=likelihood,
            alpha0=take("alpha0"), alpha_pick=take("alpha_pick"), alpha_undrafted=take("alpha_undrafted"),
            sigma_rookie=take("sigma_rookie"), mu_vet=take("mu_vet"), sigma_vet=take("sigma_vet"),
            tau=take("tau"), beta1=take("age_rise" if plateau else "beta1"), beta2=take("age_fall" if plateau else "beta2"),
            sigma=take("sigma"), nu=take("nu"), h0=take("h0"), h1=take("h1"), zero_value=d.zero_week_mean,
            gamma=take("gamma", 0.0, d.prospect_x.shape[1]) if d.prospect_x.shape[1] else np.zeros((len(draws), 0)),
            alpha_round1=take("alpha_round1", 0.0),
            round1_fit=np.linalg.lstsq(X, r1, rcond=None)[0],
            growth=take("growth", 0.0, g), growth_pick=take("growth_pick", 0.0, g),
            sigma_game=take("sigma_game", 0.0), lam_game=take("lam_game", 0.0),
            prospect_names=d.prospect_x_names,
            rho_form=take("rho_form") if "rho_form" in post else None,
            sigma_form=take("sigma_form") if "rho_form" in post else None,
            lam_form=take("lam_form") if "rho_form" in post else None,
            qb_drift=take("qb_drift") if "qb_drift" in post else None,
            qb_drift_low=take("qb_drift_low") if "qb_drift" in post else None,
            fade=qb_fade(d.ps_player, d.ps_keys["season"].to_numpy(), d.ps_qb_end)
            if fade_qb and d.ps_qb_end is not None else None,
            beta3=take("beta3") if "beta3" in post else (take("age_fall2") if plateau else None),
            age_curve="plateau" if plateau else ("talent" if talent else "quadratic"),
            age_shift=take("age_shift") if talent else None,
            age_ref=take("age_ref") if talent else None,
            age_ref_sd=take("age_ref_sd") if talent else None,
            lam_tau=take("lam_tau") if "lam_tau" in post else None,
            lam_y3=take("lam_y3") if "lam_y3" in post else None,
            rho_season=take("rho_season") if "rho_season" in post else None,
            level_mean=take("level_mean") if "rho_season" in post else None,
            sigma_young=take("sigma_young") if "sigma_young" in post else None,
            rho_bump=take("rho_bump") if "rho_bump" in post else None,
            sigma_bump=take("sigma_bump") if "rho_bump" in post else None,
            sigma_fluke=take("sigma_fluke") if "sigma_fluke" in post else None,
            bump_first=bump_first,
        )


def _age_effect(s: Settings, age_c: float, draw: np.ndarray | None = None,
                shift: np.ndarray | None = None) -> np.ndarray:
    """Age effect per posterior draw, or per particle when `draw` is given
    (and, for the talent-shifted curve, `shift` in years per particle)."""
    b1, b2 = s.beta1, s.beta2
    b3 = s.beta3
    if draw is not None:
        b1, b2 = b1[draw], b2[draw]
        b3 = None if b3 is None else b3[draw]
    if s.age_curve == "plateau":
        return age_curve_effect(age_c, b1, b2, "plateau", b3)
    a = age_c - shift if shift is not None else age_c
    out = b1 * a + b2 * a**2
    return out if b3 is None else out + b3 * a**3


def _talent_shift(s: Settings, start: np.ndarray, age_c0: float, draw: np.ndarray) -> np.ndarray | None:
    """Talent curve: each particle's peak shift from its first-season level."""
    if s.age_curve != "talent":
        return None
    adj = start - (s.beta1[draw] * age_c0 + s.beta2[draw] * age_c0**2)
    z = (adj - s.age_ref[draw]) / s.age_ref_sd[draw]
    return np.clip(s.age_shift[draw] * z, -AGE_SHIFT_MAX, AGE_SHIFT_MAX)


def _growth(s: Settings, cy: int, pick_z: float) -> np.ndarray:
    ind = np.array([float(cy >= y) for y in GROWTH_YEARS]) if cy > 0 else np.zeros(len(GROWTH_YEARS))
    return s.growth @ ind + s.growth_pick @ (ind * pick_z)


def _loglik(s: Settings, latent: np.ndarray, y: float, draw: np.ndarray) -> np.ndarray:
    """Log density of one week's points for each particle (constant terms dropped)."""
    loc = link(latent, s.likelihood, lib=np)
    p = nonzero_prob(loc, s.h0[draw], s.h1[draw], np)
    if y <= 0:
        return np.log1p(-p)
    z = np.sqrt(y)
    sig, nu = s.sigma[draw], s.nu[draw]
    dens = student_t.pdf(z, nu, loc, sig) + student_t.pdf(-z, nu, loc, sig)
    return np.log(p) + np.log(np.maximum(dens, 1e-300))


@dataclass
class Player:
    """One player's in-window career: seasons with games, and each season's weekly points in order."""

    player_id: str
    pre_window: bool
    pick_z: float
    undrafted: float
    round1: float
    prospect: np.ndarray
    seasons: list[int]
    age_c: list[float]
    career_year: list[int]
    gaps: list[float]
    games: list[np.ndarray]          # weekly points per season, in week order
    career_games: list[np.ndarray]   # career games before each of those weeks
    qb: list[np.ndarray]             # each week's QB rating (0 when unknown)
    exits: list[np.ndarray] | None = None  # each week: an injury exit (scored, but not used to update)

    def qb_before(self, j: int, k: int) -> float:
        """The QB rating in the last game before game k of season j (0 before any)."""
        if k > 0:
            return float(self.qb[j][k - 1])
        return float(self.qb[j - 1][-1]) if j > 0 and len(self.qb[j - 1]) else 0.0


def players(d: ModelData) -> list[Player]:
    order = np.lexsort((d.obs_week, d.obs_ps))
    obs_ps = d.obs_ps[order]
    out = []
    seasons = d.ps_keys["season"].to_numpy()
    for p in range(d.n_players):
        ps = np.flatnonzero(d.ps_player == p)
        out.append(Player(
            player_id=d.player_ids[p], pre_window=bool(d.pre_window[p] > 0), pick_z=float(d.log_pick_z[p]),
            undrafted=float(d.undrafted[p]), round1=float(d.round1[p]) if d.round1 is not None else 0.0,
            prospect=d.prospect_x[p], seasons=[int(seasons[i]) for i in ps], age_c=[float(d.ps_age_c[i]) for i in ps],
            career_year=[int(d.ps_career_year[i]) if d.ps_career_year is not None else 0 for i in ps],
            gaps=[float(d.ps_gap[i]) for i in ps],
            games=[d.y[order][obs_ps == i] for i in ps],
            career_games=[d.obs_career_games[order][obs_ps == i] for i in ps],
            qb=[(d.obs_qb[order] if d.obs_qb is not None else np.zeros(d.n_obs))[obs_ps == i] for i in ps],
            exits=None if d.obs_exit is None else [d.obs_exit[order][obs_ps == i] for i in ps],
        ))
    return out


class Filter:
    """Particles for one player: `m` per posterior draw."""

    def __init__(self, s: Settings, pl_: Player, m: int, rng: np.random.Generator):
        self.s, self.p, self.m, self.rng = s, pl_, m, rng
        self.draw = np.repeat(np.arange(s.n), m)
        if pl_.pre_window:
            mean, sd = s.mu_vet, s.sigma_vet
        else:
            x = np.array([1.0, pl_.pick_z, pl_.undrafted])
            r1 = pl_.round1 - x @ s.round1_fit
            mean = (s.alpha0 + s.alpha_pick * pl_.pick_z + s.alpha_undrafted * pl_.undrafted
                    + s.alpha_round1 * r1 + s.gamma @ pl_.prospect)
            sd = s.sigma_rookie
        start = mean[self.draw] + sd[self.draw] * rng.standard_normal(len(self.draw))
        # x: latent talent before the season's age and growth effects; the
        # first season's latent is `start` (the model's convention).
        self.shift = _talent_shift(s, start, pl_.age_c[0], self.draw)
        self.x = start - _age_effect(s, pl_.age_c[0], self.draw, self.shift)
        self.dev = np.zeros(len(self.draw))  # in-season form ("form" mode); resets each season
        self.bump = s.bump_start(self.draw, rng)  # this season's passing bump (season_bump)
        self.in_first = True                      # still in the player's first season
        self.fluke = (s.sigma_fluke[self.draw] * rng.standard_normal(len(self.draw))
                      if s.sigma_fluke is not None and not pl_.pre_window else np.zeros(len(self.draw)))
        self.logw = np.zeros(len(self.draw))

    def offseason(self, gap: float, q: float = 0.0, career_games: float = 0.0, career_year: int = 0) -> None:
        """Into a new season; `q` is the QB rating at the end of the season
        before, `career_games` the games played before the new one."""
        self.x = (self.s.revert(self.x, self.draw, max(gap, 1.0)) + self.s.drift(q, self.draw)
                  + self.s.step_sd(self.draw, career_games, career_year, self.p.round1) * np.sqrt(max(gap, 1.0)) * self.rng.standard_normal(len(self.x)))
        self.dev = np.zeros(len(self.x))
        self.bump = self.s.bump_next(self.bump, self.draw, gap, self.rng, self.in_first)
        self.in_first = False
        self.fluke = np.zeros(len(self.x))

    def before_game(self, first_in_window: bool, career_games: float) -> None:
        self.x, self.dev = _game_move(self.s, self.x, self.dev, self.draw, first_in_window, career_games, self.rng)

    def season_shift(self, j: int) -> np.ndarray:
        p = self.p
        return _age_effect(self.s, p.age_c[j], self.draw, self.shift) + _growth(self.s, p.career_year[j], p.pick_z)[self.draw]

    def update(self, j: int, y: float) -> None:
        self.logw = self.logw + _loglik(self.s, self.x + self.bump + self.fluke + self.dev + self.season_shift(j), y, self.draw)
        w = self.logw.reshape(self.s.n, self.m)
        w = np.exp(w - w.max(axis=1, keepdims=True))
        w = w / w.sum(axis=1, keepdims=True)
        ess = 1.0 / (w**2).sum(axis=1)
        if ess.min() < self.m / 2:
            u = (self.rng.random((self.s.n, 1)) + np.arange(self.m)) / self.m  # systematic, within each draw
            cum = np.cumsum(w, axis=1)
            idx = (cum[:, None, :] < u[:, :, None]).sum(axis=2)
            idx = np.minimum(idx, self.m - 1) + np.arange(self.s.n)[:, None] * self.m
            self.x, self.dev, self.bump = self.x[idx.ravel()], self.dev[idx.ravel()], self.bump[idx.ravel()]
            self.fluke = self.fluke[idx.ravel()]
            if self.shift is not None:
                self.shift = self.shift[idx.ravel()]
            self.logw = np.zeros(len(self.x))
        else:
            self.logw = np.log(w.ravel())

    def sample(self, r: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
        """`r` particles drawn by weight within draw groups, cycling over draws."""
        w = np.exp(self.logw.reshape(self.s.n, self.m))
        w = w / w.sum(axis=1, keepdims=True)
        groups = np.arange(r) % self.s.n
        cum = np.cumsum(w, axis=1)[groups]
        pick = np.minimum((cum < self.rng.random(r)[:, None]).sum(axis=1), self.m - 1)
        idx = groups * self.m + pick
        shift = None if self.shift is None else self.shift[idx]
        # The rookie fluke rides on the bump slot: both are this season's
        # non-lasting part (only one is on in any fit).
        return self.x[idx].copy(), self.dev[idx].copy(), self.draw[idx], shift, self.bump[idx] + self.fluke[idx]


def _game_move(s: Settings, x: np.ndarray, dev: np.ndarray, draw: np.ndarray, first_in_window: bool,
               career_games: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Talent's move into a game: a lasting step ("walk"; none for a player's
    first game in the window), or a form update dev = rho * dev + e ("form")."""
    early = np.exp(-career_games / EARLY_GAMES)
    if s.form:
        e = s.sigma_form[draw] * np.sqrt(1 - s.rho_form[draw] ** 2) * np.exp(s.lam_form[draw] * early) * rng.standard_normal(len(x))
        if s.sigma_young is not None and not first_in_window:
            x = x + s.sigma_young[draw] * season_early(career_games) * rng.standard_normal(len(x))
        return x, s.rho_form[draw] * dev + e
    if first_in_window:
        return x, dev
    return x + s.sigma_game[draw] * np.exp(s.lam_game[draw] * early) * rng.standard_normal(len(x)), dev


def _simulate_ahead(f: Filter, j: int, done: int, horizon_games: list[int], rng: np.random.Generator,
                    r: int) -> list[np.ndarray | None]:
    """Simulated points per scored stretch from the filter's state: the rest of
    season j (after `done` games), then each later season with games. Returns,
    per stretch, (r, games) weekly points or None when the stretch has no games."""
    s, p = f.s, f.p
    x, dev, draw, ashift, bump = f.sample(r)
    params = {"sigma": s.sigma[draw][:, None], "nu": s.nu[draw][:, None], "h0": s.h0[draw][:, None],
              "h1": s.h1[draw][:, None], "zero_value": s.zero_value}
    out: list[np.ndarray | None] = []
    cg = float(p.career_games[j][done]) if done < len(p.career_games[j]) else float(p.career_games[j][-1] + 1)
    prev_season = p.seasons[j]
    in_first = f.in_first
    # Later seasons' QBs are unknown: the current one is carried forward,
    # or faded toward average (Settings.fade).
    q = p.qb_before(j, done)
    for h, n_games in enumerate(horizon_games):
        season = p.seasons[j] + h
        if h > 0 and n_games:
            # One drift per step between seasons played, as in the fit.
            # The QB behind this drift is season (j + h - 1)'s: h - 1 seasons on.
            cy_new = p.career_year[j] + h if p.career_year[j] > 0 else 0
            x = s.revert(x, draw, season - prev_season) + s.drift(q, draw, h) + s.step_sd(draw, cg, cy_new, p.round1) * np.sqrt(season - prev_season) * rng.standard_normal(r)
            dev = np.zeros(r)
            bump = s.bump_next(bump, draw, season - prev_season, rng, in_first)
            in_first = False
        if not n_games:
            out.append(None)
            continue
        prev_season = season
        age_c = p.age_c[j] + h
        cy = p.career_year[j] + h if p.career_year[j] > 0 else 0
        shift = _age_effect(s, age_c, draw, ashift) + _growth(s, cy, p.pick_z)[draw]
        lat = np.empty((r, n_games))
        for k in range(n_games):
            x, dev = _game_move(s, x, dev, draw, h == 0 and k == 0 and done == 0 and j == 0, cg, rng)
            lat[:, k] = x + bump + dev + shift
            cg += 1
        center = link(lat, s.likelihood, lib=np)
        sim = simulate_weeks(s.likelihood, center, params, rng)
        if s.exit_rate > 0 and s.exit_points is not None:
            hurt = rng.random(sim.shape) < s.exit_rate
            sim = np.where(hurt, rng.choice(s.exit_points, sim.shape), sim)
        out.append(sim)
    return out


def step_through(
    s: Settings, d: ModelData, test_seasons: tuple[int, ...], last_season: int, seed: int,
    m: int = 40, r: int = 400, horizon: int = HORIZON, discount: float = DISCOUNT,
) -> pl.DataFrame:
    """One row per (player, test season, games already played that season,
    stretch): `horizon` 0 is the rest of that season, 1..horizon later
    seasons, and -1 the discounted total. Stretches past `last_season` (the
    last season with outcomes in `d`) are not scored."""
    rows = []
    for p in players(d):
        if not any(se in test_seasons for se in p.seasons):
            continue
        rng = np.random.default_rng((seed, zlib.crc32(p.player_id.encode())))  # per player: runs don't interact
        f = Filter(s, p, m, rng)
        by_season = dict(zip(p.seasons, range(len(p.seasons))))
        started = False  # the first game used takes no step (the fit's convention)
        for j, season in enumerate(p.seasons):
            if j > 0:
                f.offseason(p.gaps[j], p.qb_before(j, 0), float(p.career_games[j][0]) if len(p.career_games[j]) else 0.0,
                            p.career_year[j])
            g = p.games[j]
            for k in range(len(g) + 1):
                if season in test_seasons and k < len(g):
                    later = [len(g) - k] + [
                        len(p.games[by_season[season + h]]) if season + h in by_season else 0
                        for h in range(1, horizon + 1)
                    ]
                    scored = [season + h <= last_season for h in range(horizon + 1)]
                    sims = _simulate_ahead(f, j, k, later, rng, r)
                    actual = [g[k:]] + [p.games[by_season[season + h]] if season + h in by_season else None
                                        for h in range(1, horizon + 1)]
                    base = {"player_id": p.player_id, "season": season, "games_before": k,
                            "career_games": float(p.career_games[j][k]), "career_year": p.career_year[j],
                            "pick_z": p.pick_z, "undrafted": p.undrafted}
                    for h in range(horizon + 1):
                        if sims[h] is None or not scored[h]:
                            continue
                        ppg, a = sims[h].mean(axis=1), float(actual[h].mean())
                        rows.append(base | {"horizon": h, "games": len(actual[h]), "actual": a,
                                            "pred_mean": float(ppg.mean()), "log_score": kde_log_density(ppg, a),
                                            "pit": pit(ppg, a)})
                    if all(scored):
                        tot = sum(discount**h * sims[h].sum(axis=1) for h in range(horizon + 1) if sims[h] is not None)
                        a_tot = sum(discount**h * float(actual[h].sum()) for h in range(horizon + 1)
                                    if actual[h] is not None)
                        rows.append(base | {"horizon": -1, "games": int(sum(later)), "actual": a_tot,
                                            "pred_mean": float(np.mean(tot)), "log_score": kde_log_density(tot, a_tot),
                                            "pit": pit(tot, a_tot)})
                if k < len(g) and not (p.exits is not None and p.exits[j][k]):
                    f.before_game(not started, float(p.career_games[j][k]))
                    f.update(j, float(g[k]))
                    started = True
    return pl.DataFrame(rows)
