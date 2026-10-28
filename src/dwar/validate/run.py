"""Step 5 validation: backtest DWAR's season forecasts against the baselines.

`validate` fits (or reloads) each season as of the day before its first
game, forecasts it, scores DWAR and the baselines on the same player-
seasons, and writes `results.parquet`, `summary.json`, and calibration
plots under `root`. Development seasons and held-out seasons are scored
separately; held-out seasons are scored only when asked, once nothing will
be tuned any more.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import polars as pl

from dwar.config import load_model_config, load_scoring
from dwar.features.asof import RawTables, asof_snapshot
from dwar.features.player_tables import build_player_tables
from dwar.validate.availability import availability_forecast, summarize_availability
from dwar.validate.backtest import SCORING, fit_season
from dwar.validate.baselines import baseline_forecasts
from dwar.validate.forecast import forecast_season
from dwar.validate.report import calibration_plot, pit_histograms
from dwar.validate.score import scored_rows, summarize

log = logging.getLogger(__name__)

DEV_SEASONS = tuple(range(2012, 2023))
HOLDOUT_SEASONS = (2023, 2024, 2025)
FORECAST_DRAWS = 1000


def season_results(raw: RawTables, season: int, truth: pl.DataFrame, root: Path) -> pl.DataFrame:
    """DWAR and baseline forecasts for one season, cached on disk."""
    path = root / "forecasts" / f"{season}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    cfg = load_model_config()
    tables, fits = fit_season(raw, season, cfg, root)
    actual = truth.filter(pl.col("season") == season).select(
        "player_id", "season", "position", "games_played", "fpts_per_game"
    )
    dwar = forecast_season(fits, cfg.likelihoods(), actual, season, FORECAST_DRAWS, cfg.seed, tables, cfg.min_plays)
    base = baseline_forecasts(tables.player_seasons, tables.player_static, dwar.select(actual.columns), season)
    out = dwar.join(base.drop("season", "games_played", "fpts_per_game"), on=["player_id", "position"], how="left")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(path)
    return out


def season_availability(raw: RawTables, season: int, truth: pl.DataFrame, root: Path) -> pl.DataFrame:
    """Survival and games forecasts for one season, cached on disk."""
    path = root / "availability" / f"{season}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    cfg = load_model_config()
    tables, fits = fit_season(raw, season, cfg, root)
    schedules = asof_snapshot(raw, tables.as_of).schedules
    out = availability_forecast(fits, tables, schedules, cfg.likelihoods(), truth, season, FORECAST_DRAWS, cfg.seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(path)
    return out


def validate(
    root: Path, truth_as_of: date, seasons: tuple[int, ...] = DEV_SEASONS, label: str = "dev",
    raw: RawTables | None = None,
) -> dict:
    """Backtest `seasons`, write results under `root/label`, return the summary."""
    raw = raw or RawTables.from_cache()
    truth = build_player_tables(asof_snapshot(raw, truth_as_of), load_scoring(SCORING)).player_seasons
    results = pl.concat([season_results(raw, s, truth, root) for s in seasons], how="diagonal_relaxed")
    positions = load_model_config().positions
    summary = {"seasons": list(seasons), "truth_as_of": truth_as_of.isoformat()} | summarize(results, positions)

    availability = pl.concat([season_availability(raw, s, truth, root) for s in seasons])
    summary["availability"] = summarize_availability(availability, positions)

    out = root / label
    out.mkdir(parents=True, exist_ok=True)
    results.write_parquet(out / "results.parquet")
    availability.write_parquet(out / "availability.parquet")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    rows, _ = scored_rows(results)
    names = ["all", *positions, "rookies"]
    calibration_plot(
        {g: summary["groups"][g]["coverage"] for g in names if g in summary["groups"]},
        out / "calibration.png", f"Interval coverage, {seasons[0]}-{seasons[-1]}",
    )
    groups = {"all": rows} | {p: rows.filter(pl.col("position") == p) for p in positions}
    groups["rookies"] = rows.filter(pl.col("is_rookie"))
    pit_histograms(rows, groups, out / "pit.png", f"Forecast PIT, {seasons[0]}-{seasons[-1]}")
    return summary
