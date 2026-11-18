"""Rookie-prior and age-curve terms learned from a longer history and handed
to a position's main fit as priors (plan "RB rookie prior learned from longer
draft history", ben 2026-11-18).

The main RB fit is game-by-game with usage terms that need snap counts, so it
starts in 2013. That window holds only five top-12 RB picks, all hits, which
pins the top of the draft prior too high and too tight. The history fit is a
season-level model (same likelihood, age curve and rookie prior; no game
steps, usage or situation factors) on seasons `first_season` to just before
`as_of` (1999-2012 by default), so it does not overlap the main fit's
seasons and the hand-off is a plain sequential update: history posterior ->
main prior -> main posterior.

Handed off, as Normal(mean, sd) of the posterior draws:
- "pick": the rookie prior's shape by draft slot (alpha_pick, alpha_pick2,
  alpha_undrafted, sd_pick, sd_undrafted). The level and spread of the
  average rookie (alpha0, sigma_rookie) stay the main fit's own, so era
  differences in RB scoring do not leak in as a level change (the plan's
  scale check; `handoff` reports both fits' values side by side).
- "age": the age curve (beta1, beta2), centred at 26 in both fits.

Log pick is standardized over each fit's own drafted players, so the pick
terms are re-expressed draw by draw in the main fit's basis (`convert_pick_terms`).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from xarray import DataTree

from dwar.config import HistoryPrior, ModelConfig
from dwar.model.data import UNDRAFTED_PICK, ModelData

log = logging.getLogger(__name__)

PICK_TERMS = ("alpha_pick", "alpha_pick2", "alpha_undrafted", "sd_pick", "sd_undrafted")
AGE_TERMS = ("beta1", "beta2")


@dataclass(frozen=True)
class PickBasis:
    """How a fit standardized log pick: z = (log pick - mu) / sd over its
    drafted reference, and the quadratic feature (z^2 - m2) / s2 over its
    drafted players (data.pick_quad_feature)."""
    mu: float
    sd: float
    m2: float
    s2: float


def pick_basis(d: ModelData) -> PickBasis:
    drafted = d.undrafted == 0
    if d.draft_pick is None or drafted.sum() < 3:
        raise ValueError("pick basis needs drafted players with picks")
    L = np.log(d.draft_pick[drafted])
    b, a = np.polyfit(L, d.log_pick_z[drafted], 1)  # z = a + b L
    z2 = d.log_pick_z[drafted] ** 2
    return PickBasis(mu=float(-a / b), sd=float(1.0 / b), m2=float(z2.mean()), s2=float(z2.std(ddof=1)))


def convert_pick_terms(t: dict[str, np.ndarray], h: PickBasis, m: PickBasis) -> dict[str, np.ndarray]:
    """Rookie-prior terms from basis `h` to basis `m`, giving the same prior
    mean and spread at every log pick (and for undrafted players). `t` holds
    alpha0, alpha_pick, alpha_pick2 (0 if absent), alpha_undrafted,
    sigma_rookie, sd_pick, sd_undrafted (0 if absent), as draw arrays."""
    a0, a1, a2, au = t["alpha0"], t["alpha_pick"], t.get("alpha_pick2", 0.0 * t["alpha0"]), t["alpha_undrafted"]
    ls, s, su = np.log(t["sigma_rookie"]), t.get("sd_pick", 0.0 * a0), t.get("sd_undrafted", 0.0 * a0)
    Lu = np.log(UNDRAFTED_PICK)

    def mean_h(L):
        z = (L - h.mu) / h.sd
        return a0 + a1 * z + a2 * (z**2 - h.m2) / h.s2

    # Mean as c0 + c1 L + c2 L^2; then the same polynomial in basis m.
    c2 = a2 / (h.s2 * h.sd**2)
    c1 = a1 / h.sd - 2 * c2 * h.mu
    a2m = c2 * m.s2 * m.sd**2
    a1m = m.sd * (c1 + 2 * c2 * m.mu)
    a0m = mean_h(m.mu) + a2m * m.m2 / m.s2  # at L = m.mu: z = 0, quad = -m2/s2
    und_h = a0 + a1 * (Lu - h.mu) / h.sd + au  # undrafted: line term at the undrafted pick, no quad term
    aum = und_h - a0m - a1m * (Lu - m.mu) / m.sd
    sm = s * m.sd / h.sd
    lsm = ls + s * (m.mu - h.mu) / h.sd
    return {
        "alpha0": a0m, "alpha_pick": a1m, "alpha_pick2": a2m, "alpha_undrafted": aum,
        "sigma_rookie": np.exp(lsm), "sd_pick": sm, "sd_undrafted": su,  # sd_undrafted is basis-free
    }


def history_config(cfg: ModelConfig, position: str, hp: HistoryPrior) -> ModelConfig:
    """The season-level model for `position`'s history fit: the main fit's
    likelihood, age curve and rookie prior, without game steps, usage,
    situation factors or prospect metrics, from `hp.first_season`."""
    drop = lambda names: tuple(p for p in names if p != position)  # noqa: E731
    return cfg.model_copy(update={
        "first_season_by_position": {**cfg.first_season_by_position, position: hp.first_season},
        "game_steps": drop(cfg.game_steps), "young_steps": drop(cfg.young_steps), "usage": drop(cfg.usage),
        "qb_drift": drop(cfg.qb_drift), "eff_drift": drop(cfg.eff_drift), "fill_in": drop(cfg.fill_in),
        "early_noise": drop(cfg.early_noise), "injury_exits": drop(cfg.injury_exits),
        "situation_level": {k: v for k, v in cfg.situation_level.items() if k != position},
        "situation_range": {k: v for k, v in cfg.situation_range.items() if k != position},
        "prospect": {k: v for k, v in cfg.prospect.items() if k != position},
        "history_prior": {},
    })


def _draws(post: DataTree, name: str) -> np.ndarray | None:
    p = post["posterior"]
    return np.asarray(p[name].values).reshape(-1) if name in p else None


def rookie_terms(post: DataTree) -> dict[str, np.ndarray]:
    names = ("alpha0", "alpha_pick", "alpha_pick2", "alpha_undrafted", "sigma_rookie", "sd_pick", "sd_undrafted")
    return {n: v for n in names if (v := _draws(post, n)) is not None}


def handoff(hist_post: DataTree, hist_d: ModelData, main_d: ModelData, hp: HistoryPrior) -> tuple[dict, dict]:
    """(priors, report): Normal(mean, sd) priors for the main fit, by
    parameter name, and a report of the history fit's terms in the main
    fit's basis (including alpha0 and sigma_rookie, which are not handed off)."""
    priors, report = {}, {"terms": list(hp.terms), "widen": hp.widen, "first_season": hp.first_season,
                          "as_of": hp.as_of.isoformat()}
    if "pick" in hp.terms:
        conv = convert_pick_terms(rookie_terms(hist_post), pick_basis(hist_d), pick_basis(main_d))
        report["pick_basis"] = {"history": pick_basis(hist_d).__dict__, "main": pick_basis(main_d).__dict__}
        for n, v in conv.items():
            report[n] = {"mean": float(np.mean(v)), "sd": float(np.std(v))}
            if n in PICK_TERMS and _draws(hist_post, n) is not None:  # only terms the history model has
                priors[n] = (float(np.mean(v)), float(np.std(v)) * hp.widen)
    if "age" in hp.terms:
        for n in AGE_TERMS:
            v = _draws(hist_post, n)
            if v is None:
                raise ValueError(f"history fit has no {n} (age terms need the quadratic age curve)")
            priors[n] = (float(np.mean(v)), float(np.std(v)) * hp.widen)
            report[n] = {"mean": float(np.mean(v)), "sd": float(np.std(v))}
    report["priors"] = {k: list(v) for k, v in priors.items()}
    return priors, report


def history_dir(root: Path, position: str, hp: HistoryPrior) -> Path:
    return root / f"{position.lower()}_{hp.first_season}_{hp.as_of.isoformat()}"


def write_report(report: dict, out_dir: Path, position: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{position.lower()}_history_prior.json").write_text(json.dumps(report, indent=2))


def history_priors(
    position: str, cfg: ModelConfig, main_d: ModelData, hist_tables, root: Path, out_dir: Path | None = None,
) -> dict[str, tuple[float, float]]:
    """Fit (or load) `position`'s history fit from `hist_tables` (player
    tables as of `cfg.history_prior[position].as_of`) under `root`, and
    return the priors it hands to the main fit. Raises FitFailed if the
    history fit fails its checks (nothing is handed off from a bad fit)."""
    from dwar.model.data import build_model_data
    from dwar.model.fit import fit, load_fit

    hp = cfg.history_prior[position]
    hcfg = history_config(cfg, position, hp)
    hd = build_model_data(hist_tables, position, hp.first_season, min_plays=cfg.min_plays.get(position, 0),
                          censored_start=True)
    where = history_dir(root, position, hp)
    r = load_fit(hd, where)
    if r is None:
        log.info("%s history fit %d to %s", position, hp.first_season, hp.as_of)
        r = fit(hd, hcfg, where, {"as_of": hp.as_of.isoformat(), "history_fit": True})
    priors, report = handoff(r.posterior, hd, main_d, hp)
    if out_dir is not None:
        write_report(report, out_dir, position)
    return priors


def add_scale_check(out_dir: Path, position: str, main_post: DataTree) -> None:
    """The plan's scale check: the main fit's own average-rookie level and
    spread beside the history fit's (same basis), in the hand-off report."""
    path = out_dir / f"{position.lower()}_history_prior.json"
    if not path.exists():
        return
    report = json.loads(path.read_text())
    report["main"] = {n: {"mean": float(v.mean()), "sd": float(v.std())}
                      for n, v in rookie_terms(main_post).items()}
    for n in AGE_TERMS:
        if (v := _draws(main_post, n)) is not None:
            report["main"][n] = {"mean": float(v.mean()), "sd": float(v.std())}
    path.write_text(json.dumps(report, indent=2))
