"""Pull nflverse datasets through nflreadpy into dated parquet snapshots.

Our cache in ``data/raw/nflverse/`` is the single source of truth; nflreadpy's
own cache is turned off. Reads go through `load_raw`, never the network.

Freshness: a season's file is *final* once it was pulled on or after
March 1 of the following year (after the Super Bowl and stat corrections).
Final files are never re-downloaded. Non-final files (the season in progress,
or one pulled before it finished) and non-seasonal tables are refreshed when
older than `max_age`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from typing import Any

import nflreadpy
import polars as pl
from nflreadpy.config import update_config

from dwar.config import data_dir
from dwar.ingest.cache import (
    SchemaChangedError,
    latest_file,
    latest_schema,
    schema_of,
    write_files,
)

log = logging.getLogger(__name__)

SOURCE = "nflverse"
DEFAULT_MAX_AGE = timedelta(hours=24)

# Career outcomes in load_draft_picks (AV, games, Pro Bowls, last season, career
# stats) are future information for any fit before a player's career ends, so
# only draft-day facts are kept.
DRAFT_PICK_COLUMNS = (
    "season", "round", "pick", "team", "gsis_id", "pfr_player_id", "cfb_player_id",
    "pfr_player_name", "position", "category", "side", "college", "age",
)
# load_combine carries draft results, which are unknown between the combine
# and the draft. Draft capital comes from draft_picks instead.
COMBINE_DROP = ("draft_team", "draft_round", "draft_ovr")
# Play-by-play is large (about 370 columns); keep what the situation factors use.
PBP_COLUMNS = (
    "game_id", "season", "week", "season_type", "game_date", "posteam", "defteam",
    "play_type", "qtr", "down", "score_differential", "pass", "rush", "qb_dropback",
    "qb_epa", "passer_player_id", "receiver_player_id", "rusher_player_id", "pass_oe", "xpass",
)


@dataclass(frozen=True)
class DatasetSpec:
    loader: str
    seasonal: bool = True
    first_season: int | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    keep: tuple[str, ...] | None = None
    drop: tuple[str, ...] = ()
    last_season: int | None = None
    adds_season: bool = False  # the loader returns no season column; fetch one season at a time and add it


DATASETS: dict[str, DatasetSpec] = {
    "player_stats": DatasetSpec("load_player_stats", first_season=1999, kwargs={"summary_level": "week"}),
    "schedules": DatasetSpec("load_schedules", first_season=1999),
    "rosters": DatasetSpec("load_rosters", first_season=1999),
    "draft_picks": DatasetSpec("load_draft_picks", first_season=1990, keep=DRAFT_PICK_COLUMNS),
    "combine": DatasetSpec("load_combine", first_season=2000, drop=COMBINE_DROP),
    "snap_counts": DatasetSpec("load_snap_counts", first_season=2013),
    "pbp": DatasetSpec("load_pbp", first_season=1999, keep=PBP_COLUMNS),
    # Depth charts changed format in 2025: weekly files before, timestamped
    # daily snapshots (column `dt`, no season) from 2025 on.
    "depth_charts": DatasetSpec("load_depth_charts", first_season=2001, last_season=2024),
    "depth_charts_daily": DatasetSpec("load_depth_charts", first_season=2025, adds_season=True),
    "rosters_weekly": DatasetSpec("load_rosters_weekly", first_season=2002),
    "players": DatasetSpec("load_players", seasonal=False),
    "ff_playerids": DatasetSpec("load_ff_playerids", seasonal=False),
}


def season_file(season: int) -> str:
    return f"season={season}.parquet"


ALL_FILE = "all.parquet"


def is_final(season: int, pulled_at: datetime) -> bool:
    return pulled_at.date() >= date(season + 1, 3, 1)


def _spec(dataset: str) -> DatasetSpec:
    if dataset not in DATASETS:
        raise KeyError(f"Unknown dataset '{dataset}'. Known: {sorted(DATASETS)}")
    return DATASETS[dataset]


def _fetch(spec: DatasetSpec, seasons: list[int] | None) -> pl.DataFrame:
    update_config(cache_mode="off")
    loader = getattr(nflreadpy, spec.loader)
    if spec.adds_season:
        df = pl.concat(
            [loader([s], **spec.kwargs).with_columns(season=pl.lit(s, pl.Int32)) for s in seasons or []],
            how="diagonal_relaxed",
        )
    else:
        df = loader(seasons, **spec.kwargs) if spec.seasonal else loader(**spec.kwargs)
    if spec.keep is not None:
        missing = sorted(set(spec.keep) - set(df.columns))
        if missing:
            raise SchemaChangedError(f"{spec.loader} no longer returns {missing}")
        df = df.select(spec.keep)
    return df.drop(spec.drop, strict=False)


def _check_schema(dataset: str, df: pl.DataFrame, root: Path, allow_schema_change: bool) -> None:
    previous = latest_schema(root, SOURCE, dataset)
    current = schema_of(df)
    if previous is None or previous == current or allow_schema_change:
        return
    added = sorted(set(current) - set(previous))
    removed = sorted(set(previous) - set(current))
    retyped = sorted(c for c in set(current) & set(previous) if current[c] != previous[c])
    raise SchemaChangedError(
        f"{dataset} schema changed: added={added} removed={removed} retyped={retyped}. "
        "Review, then re-run with allow_schema_change=True."
    )


def stale_seasons(
    dataset: str,
    seasons: list[int],
    now: datetime,
    max_age: timedelta = DEFAULT_MAX_AGE,
    root: Path | None = None,
) -> list[int]:
    root = root or data_dir() / "raw"
    stale = []
    for season in seasons:
        cached = latest_file(root, SOURCE, dataset, season_file(season))
        if cached is None:
            stale.append(season)
        elif not is_final(season, cached.pulled_at) and now - cached.pulled_at > max_age:
            stale.append(season)
    return stale


def pull(
    dataset: str,
    seasons: list[int] | None = None,
    *,
    now: datetime | None = None,
    max_age: timedelta = DEFAULT_MAX_AGE,
    root: Path | None = None,
    allow_schema_change: bool = False,
) -> list[Path]:
    """Download whatever is missing or stale for `dataset`. Returns paths written."""
    spec = _spec(dataset)
    root = root or data_dir() / "raw"
    now = now or datetime.now(UTC)
    package_version = version("nflreadpy")

    if not spec.seasonal:
        cached = latest_file(root, SOURCE, dataset, ALL_FILE)
        if cached is not None and now - cached.pulled_at <= max_age:
            return []
        df = _fetch(spec, None)
        _check_schema(dataset, df, root, allow_schema_change)
        return write_files(root, SOURCE, dataset, {ALL_FILE: df}, now, package_version)

    if seasons is None:
        seasons = list(range(spec.first_season, (spec.last_season or nflreadpy.get_current_season()) + 1))
    seasons = [s for s in seasons if s >= (spec.first_season or s) and s <= (spec.last_season or s)]
    todo = stale_seasons(dataset, seasons, now, max_age, root)
    if not todo:
        return []
    df = _fetch(spec, todo)
    _check_schema(dataset, df, root, allow_schema_change)
    frames = {}
    for season in todo:
        part = df.filter(pl.col("season") == season)
        if part.is_empty():
            log.warning("%s: no rows for season %s; nothing cached", dataset, season)
            continue
        frames[season_file(season)] = part
    return write_files(root, SOURCE, dataset, frames, now, package_version)


def pull_all(seasons: list[int] | None = None, **kwargs: Any) -> dict[str, list[Path]]:
    return {name: pull(name, seasons, **kwargs) for name in DATASETS}


def load_raw(
    dataset: str,
    seasons: list[int] | None = None,
    *,
    pull_date: date | None = None,
    root: Path | None = None,
) -> pl.DataFrame:
    """Read a dataset from the cache: the newest copy of each season, or the
    newest on or before `pull_date` to reproduce an earlier run. Never downloads."""
    spec = _spec(dataset)
    root = root or data_dir() / "raw"

    if not spec.seasonal:
        cached = latest_file(root, SOURCE, dataset, ALL_FILE, pull_date)
        if cached is None:
            raise FileNotFoundError(f"No cached {dataset}; run pull('{dataset}') first")
        return pl.read_parquet(cached.path)

    if seasons is None:
        seasons = _cached_seasons(root, dataset)
    paths = []
    for season in seasons:
        cached = latest_file(root, SOURCE, dataset, season_file(season), pull_date)
        if cached is None:
            raise FileNotFoundError(f"No cached {dataset} for {season}; run pull first")
        paths.append(cached.path)
    if not paths:
        raise FileNotFoundError(f"No cached {dataset}; run pull('{dataset}') first")
    return pl.concat([pl.read_parquet(p) for p in paths], how="diagonal_relaxed")


def _cached_seasons(root: Path, dataset: str) -> list[int]:
    base = root / SOURCE
    if not base.exists():
        return []
    names = {p.stem for p in base.glob(f"*/{dataset}/season=*.parquet")}
    return sorted(int(n.removeprefix("season=")) for n in names)
