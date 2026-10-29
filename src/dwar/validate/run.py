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

from dwar.config import load_model_config, load_scoring, load_value_config
from dwar.features.asof import RawTables, asof_snapshot
from dwar.features.player_tables import build_player_tables
from dwar.validate.availability import availability_forecast, summarize_availability
from dwar.validate.backtest import SCORING, fit_season
from dwar.validate.baselines import baseline_forecasts
from dwar.validate.forecast import Situation, forecast_season
from dwar.validate.report import calibration_plot, pit_histograms
from dwar.validate.score import scored_rows, summarize
from dwar.value.spread import spread_factor

log = logging.getLogger(__name__)

DEV_SEASONS = tuple(range(2012, 2023))
HOLDOUT_SEASONS = (2023, 2024, 2025)
FORECAST_DRAWS = 1000


def _past_forecasts(root: Path, season: int, position: str) -> pl.DataFrame:
    """Earlier backtest seasons' forecasts at `position`, before widening:
    what a spread factor for `season` may learn from."""
    paths = sorted(p for p in (root / "forecasts").glob("*.parquet") if int(p.stem) < season)
    cols = ["season", "fpts_per_game", "pred_mean", "pred_sd"]
    if not paths:
        return pl.DataFrame(schema={c: pl.Float64 for c in cols})
    rows = pl.concat([pl.read_parquet(p) for p in paths], how="diagonal_relaxed").filter(
        (pl.col("position") == position) & pl.col("has_forecast")
    )
    if "pred_sd_raw" in rows.columns:
        rows = rows.with_columns(pred_sd=pl.coalesce("pred_sd_raw", "pred_sd"))
    return rows.select(cols)


def season_results(raw: RawTables, season: int, truth: pl.DataFrame, root: Path) -> pl.DataFrame:
    """DWAR and baseline forecasts for one season, cached on disk."""
    path = root / "forecasts" / f"{season}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    cfg = load_model_config()
    tables, fits, cov = fit_season(raw, season, cfg, root)
    situation = {
        p: Situation(
            c.level.filter(pl.col("season") == season).drop("season"),
            c.range.filter(pl.col("season") == season).drop("season"),
        )
        for p, c in cov.items()
    }
    spread = {p: spread_factor(_past_forecasts(root, season, p)) for p in load_value_config().calibrate_spread}
    actual = truth.filter(pl.col("season") == season).select(
        "player_id", "season", "position", "games_played", "fpts_per_game"
    )
    dwar = forecast_season(
        fits, cfg.likelihoods(), actual, season, FORECAST_DRAWS, cfg.seed, tables, cfg.min_plays, spread, situation,
    )
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
    tables, fits, _ = fit_season(raw, season, cfg, root)
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
    # For config/value.yaml: the factor learned from every season scored here.
    summary["spread_factor"] = {
        p: round(spread_factor(_past_forecasts(root, max(seasons) + 1, p)), 3)
        for p in load_value_config().calibrate_spread
    }
    if "spread" in results.columns:
        summary["spread_by_season"] = {
            p: {str(k): v for k, v in results.filter(pl.col("position") == p)
                .group_by("season").agg(pl.col("spread").max()).sort("season").rows()}
            for p in load_value_config().calibrate_spread
        }

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
