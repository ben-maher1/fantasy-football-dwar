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
from pathlib import Path

import polars as pl

from dwar.config import ModelConfig, load_scoring
from dwar.features.asof import REGULAR_SEASON, RawTables, asof_snapshot
from dwar.features.player_tables import PlayerTables, build_player_tables
from dwar.model.data import build_model_data
from dwar.model.diagnostics import FitFailed
from dwar.model.fit import FitResult, fit, load_fit

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


def _retry_config(cfg: ModelConfig) -> ModelConfig:
    s = cfg.sampler
    return cfg.model_copy(update={"sampler": s.model_copy(update={"tune": 2 * s.tune, "draws": 2 * s.draws})})


def fit_season(raw: RawTables, season: int, cfg: ModelConfig, root: Path) -> tuple[PlayerTables, dict[str, FitResult]]:
    """Fit (or load) every position as of the start of `season`. Raises
    FitFailed if a position fails both attempts."""
    tables = season_tables(raw, season)
    out = root / "fits" / str(season)
    record_path = out / "fit_log.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    meta = {"as_of": tables.as_of.isoformat(), "scoring": SCORING, "backtest_season": season}
    fits = {}
    for position in cfg.positions:
        d = build_model_data(tables, position, cfg.first_season, min_plays=cfg.min_plays.get(position, 0))
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
    return tables, fits
