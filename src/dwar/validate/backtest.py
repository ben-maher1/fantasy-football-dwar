"""Out-of-sample backtest: fit as of the day before each season's first game,
then forecast that season.

Fits are saved per season under `{root}/fits/{season}/` and reused, so an
interrupted backtest resumes where it stopped. A fit that fails the
diagnostics gate is retried once with twice the tuning and draws; both
attempts are recorded, never dropped.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from functools import cache
from pathlib import Path

import polars as pl

from dwar.config import ModelConfig, load_scoring, uses_situation
from dwar.features.asof import REGULAR_SEASON, RawTables, asof_snapshot
from dwar.features.player_tables import PlayerTables, build_player_tables
from dwar.features.situation import (
    Covariates,
    SituationRaw,
    Sources,
    model_covariates,
    prepare,
)
from dwar.model.diagnostics import FitFailed
from dwar.model.fit import FitResult, fit, load_fit, position_data

log = logging.getLogger(__name__)

SCORING = "ppr"


def season_start(raw: RawTables, season: int) -> date:
    """Date of the season's first regular-season game."""
    games = raw.schedules.filter((pl.col("season") == season) & (pl.col("game_type") == REGULAR_SEASON))
    if games.is_empty():
        raise ValueError(f"no {season} regular-season schedule")
    return date.fromisoformat(games["gameday"].min())


def backtest_as_of(raw: RawTables, season: int) -> date:
    return season_start(raw, season) - timedelta(days=1)


def season_tables(raw: RawTables, season: int) -> PlayerTables:
    """Everything knowable the day before `season` kicks off."""
    return build_player_tables(asof_snapshot(raw, backtest_as_of(raw, season)), load_scoring(SCORING))


@cache
def situation_sources() -> Sources:
    return prepare(SituationRaw.from_cache())


def season_covariates(
    raw: RawTables, tables: PlayerTables, season: int, cfg: ModelConfig, src: Sources | None = None,
) -> dict[str, Covariates]:
    """Each position's situation factors for its fit rows and for `season`."""
    if not uses_situation(cfg):
        return {}
    return model_covariates(
        src or situation_sources(), asof_snapshot(raw, tables.as_of), tables, cfg.first_season,
        cfg.situation_level, cfg.situation_range, (season,),
    )


def _retry_config(cfg: ModelConfig) -> ModelConfig:
    s = cfg.sampler
    return cfg.model_copy(update={"sampler": s.model_copy(update={"tune": 2 * s.tune, "draws": 2 * s.draws})})


def fit_season(
    raw: RawTables, season: int, cfg: ModelConfig, root: Path,
) -> tuple[PlayerTables, dict[str, FitResult], dict[str, Covariates]]:
    """Fit (or load) every position as of the start of `season`. Raises
    FitFailed if a position fails both attempts. Also returns each
    position's situation factors (empty without any)."""
    tables = season_tables(raw, season)
    cov = season_covariates(raw, tables, season, cfg)
    out = root / "fits" / str(season)
    record_path = out / "fit_log.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    meta = {"as_of": tables.as_of.isoformat(), "scoring": SCORING, "backtest_season": season}
    fits = {}
    for position in cfg.positions:
        d = position_data(tables, position, cfg, cov)
        cached = load_fit(d, out)
        if cached is not None:
            fits[position] = cached
            continue
        attempts, failure = [], None
        for attempt_cfg in (cfg, _retry_config(cfg)):
            try:
                fits[position] = fit(d, attempt_cfg, out, meta)
                attempts.append({"sampler": attempt_cfg.sampler.model_dump(), "passed": True})
                break
            except FitFailed as e:
                failure = e
                attempts.append({"sampler": attempt_cfg.sampler.model_dump(), "passed": False, "failures": e.diagnostics.failures})
                log.warning("%s %s fit failed (%s)", season, position, "; ".join(e.diagnostics.failures))
        record[position] = attempts
        out.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps(record, indent=2))
        if position not in fits:
            raise failure
    return tables, fits, cov
